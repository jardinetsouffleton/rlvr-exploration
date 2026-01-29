
import os
import sys

# --- Environment Setup matching run_safe.sh ---
# Ensure HF_HOME is set to the shared directory if not already set
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/mnt/shared/boileo/hf_cache"

# Also set HF_DATASETS_CACHE to avoid disk space issues
os.environ["HF_DATASETS_CACHE"] = os.environ.get("HF_DATASETS_CACHE", "/mnt/shared/boileo/hf_cache/datasets")

# Set PyTorch memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Import config early to ensure env vars are propagated if config does validation
from config import MODEL_ID, OUTPUT_DIR, TSP_DATA_DIR, LR, GROUP_SIZE, GENERATION_BATCH_SIZE, STEPS, MAX_NEW_TOKENS, BETA, REWARD_BASELINE, PROMPT_MODE, PROBLEM_TYPE, SAT_VARS, SAT_CLAUSES, SAT_CURRICULUM, SAT_MIN_VARS, SAT_MAX_VARS, SAT_STEP_INTERVAL, SAT_STEP_SIZE, CLAUSE_RATIO, SAT_BENCHMARK_SIZES, SAT_BENCHMARK_INSTANCES, TSP_DUAL_MIN_CITIES, TSP_DUAL_MAX_CITIES, TSP_DUAL_FORMAT_REWARD, TSP_DUAL_OVERESTIMATE_MULT, TSP_DUAL_OVERESTIMATE_PENALTY

import torch
import logging
import csv
import json
import numpy as np
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback

# Import utils from existing files
from tsp_utils import generate_tsp_instance, render_tsp_instance, parse_model_output, calculate_tour_length, solve_tsp_nearest_neighbor, solve_tsp_optimal, get_tsp_prompt
from sat_utils import generate_sat_instance, get_sat_prompt, parse_sat_output, check_sat_solution
from tsp_dual_utils import get_tsp_dual_prompt, parse_dual_bound, calculate_dual_reward
from sat_instance_generator import init_instance_pool, get_instance_pool, shutdown_instance_pool

# Global step counter for curriculum (updated via callback)
_current_step = 0

def get_curriculum_size(step: int) -> int:
    """
    Calculate current problem size based on training step.
    Deterministically increases size at fixed intervals.
    """
    if not SAT_CURRICULUM:
        return SAT_VARS
    
    # How many size increases have occurred
    increases = step // SAT_STEP_INTERVAL
    n_vars = SAT_MIN_VARS + (increases * SAT_STEP_SIZE)
    
    # Clamp to max
    return min(n_vars, SAT_MAX_VARS)


class CurriculumCallback(TrainerCallback):
    """Callback to update the global step counter for curriculum progression."""
    
    def on_step_begin(self, args, state, control, **kwargs):
        global _current_step
        _current_step = state.global_step
        
        # Log curriculum progression at intervals
        if state.global_step % SAT_STEP_INTERVAL == 0:
            n_vars = get_curriculum_size(state.global_step)
            logging.info(f"Curriculum step {state.global_step}: n_vars={n_vars}")

# Helper to determine model type (copied from train_rlvr.py)
def get_model_context(model_id):
    is_vision = "VL" in model_id or "Vision" in model_id
    if is_vision:
        config_class = "AutoModelForVision2Seq" # Not used directly in wrapper but good for logic
        processor_class = AutoProcessor
    else:
        config_class = AutoModelForCausalLM
        processor_class = AutoTokenizer
    return is_vision, config_class, processor_class

# --- 1. Dataset Wrapper & Registry ---

class ProblemRegistry:
    _registry = {}
    _counter = 0
    
    @classmethod
    def register(cls, data):
        cls._counter += 1
        cls._registry[cls._counter] = data
        return cls._counter
    
    @classmethod
    def get(cls, problem_id):
        return cls._registry.get(problem_id)
        
    @classmethod
    def clear(cls):
        # Optional: cleanup old entries if memory grows too large
        if len(cls._registry) > 10000:
            cls._registry.clear()
            cls._counter = 0

def create_problem_samples(problem_type, num_samples, tokenizer):
    """Generator function for creating dataset samples."""
    for _ in range(num_samples):
        if problem_type == "tsp":
            n_cities = np.random.randint(5, 8)
            coords = generate_tsp_instance(n_cities=n_cities)
            
            content_payload = get_tsp_prompt(n_cities, coords=coords, mode=PROMPT_MODE)
            
            pid = ProblemRegistry.register({
                "coords": coords,
                "n_cities": n_cities
            })
            
            # Build messages and apply chat template (return text, not tokens)
            messages = [{"role": "user", "content": content_payload}]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"User: {content_payload}\nAssistant:"
            
            yield {"prompt": prompt_text, "problem_id": pid}

        elif problem_type == "sat":
            # Get current difficulty from curriculum
            n_vars = get_curriculum_size(_current_step)
            n_clauses = int(n_vars * CLAUSE_RATIO)
            
            # Try to get pre-verified solvable instance from pool
            pool = get_instance_pool()
            if pool is not None:
                instance = pool.get_instance(n_vars, timeout=5.0)
                if instance:
                    clauses = instance['clauses']
                    n_vars = instance['n_vars']
                    n_clauses = instance['n_clauses']
                else:
                    # Fallback: generate directly (may not be solvable)
                    clauses = generate_sat_instance(n_vars=n_vars, n_clauses=n_clauses)
            else:
                # No pool initialized, generate directly
                clauses = generate_sat_instance(n_vars=n_vars, n_clauses=n_clauses)
            
            text_prompt = get_sat_prompt(clauses, n_vars, mode=PROMPT_MODE)
            
            pid = ProblemRegistry.register({
                "clauses": clauses,
                "n_vars": n_vars
            })
            
            messages = [{"role": "user", "content": text_prompt}]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"User: {text_prompt}\nAssistant:"

            yield {"prompt": prompt_text, "problem_id": pid}

        elif problem_type == "tsp_dual":
            # TSP Dual Bound Problem: predict a lower bound on optimal tour cost
            n_cities = np.random.randint(TSP_DUAL_MIN_CITIES, TSP_DUAL_MAX_CITIES + 1)
            coords = generate_tsp_instance(n_cities=n_cities)
            
            # Compute optimal tour cost (ground truth for reward)
            _, optimal_cost = solve_tsp_optimal(coords)
            
            content_payload = get_tsp_dual_prompt(n_cities, coords=coords, mode=PROMPT_MODE)
            
            pid = ProblemRegistry.register({
                "coords": coords,
                "n_cities": n_cities,
                "optimal_cost": optimal_cost  # Store for reward calculation
            })
            
            messages = [{"role": "user", "content": content_payload}]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"User: {content_payload}\nAssistant:"
            
            yield {"prompt": prompt_text, "problem_id": pid}

# --- 2. Reward Functions ---

def tsp_reward_func(completions, **kwargs):
    rewards = []
    
    # problem_id typically comes as a list or tensor. If tensor, move to cpu/numpy
    pids = kwargs.get("problem_id", [])
    
    for i, tour_text in enumerate(completions):
        pid = pids[i]
        if hasattr(pid, "item"): pid = pid.item()
        
        data = ProblemRegistry.get(pid)
        c_coords = data["coords"]
        c_n_cities = data["n_cities"]
        
        parsed_tour = parse_model_output(tour_text, c_n_cities)
        
        # Copied logic from train_rlvr.py
        if parsed_tour is None:
            rewards.append(-10.0)
            continue

        unique_cities = set(parsed_tour)
        has_all_cities = (len(unique_cities) == c_n_cities)

        if not has_all_cities:
            rewards.append(-5.0)
            continue
            
        if len(parsed_tour) == c_n_cities + 1:
            if parsed_tour[0] != parsed_tour[-1]:
                 rewards.append(-5.0)
                 continue
        elif len(parsed_tour) == c_n_cities:
             parsed_tour = list(parsed_tour) + [parsed_tour[0]]
        else:
             rewards.append(-5.0)
             continue
        
        model_len = calculate_tour_length(parsed_tour, c_coords)
        
        # Calculate baseline
        if REWARD_BASELINE == "optimal":
             _, baseline_len = solve_tsp_optimal(c_coords)
        else:
             _, baseline_len = solve_tsp_nearest_neighbor(c_coords)
        
        gap = (baseline_len - model_len) / baseline_len
        gap = max(gap, -3.0)
        rewards.append(1.0 + gap)
        
    return rewards

def sat_reward_func(completions, **kwargs):
    rewards = []
    pids = kwargs.get("problem_id", [])
    
    for i, sol_text in enumerate(completions):
        pid = pids[i]
        if hasattr(pid, "item"): pid = pid.item()
        
        data = ProblemRegistry.get(pid)
        c_clauses = data["clauses"]
        c_n_vars = data["n_vars"]
        
        assignment = parse_sat_output(sol_text, c_n_vars)
        
        # Logic from train_rlvr.py
        if assignment is None:
            rewards.append(-1.0)
            continue
            
        unique_vars = {abs(l) for l in assignment}
        if len(unique_vars) != c_n_vars:
            rewards.append(-1.0)
            continue
            
        vals = {abs(l): (l > 0) for l in assignment}
        unsatisfied_count = 0
        for clause in c_clauses:
            clause_sat = False
            for lit in clause:
                var = abs(lit)
                val = vals.get(var, False)
                if (lit > 0 and val) or (lit < 0 and not val):
                    clause_sat = True
                    break
            if not clause_sat:
                unsatisfied_count += 1
                
        penalty = -1.0 * unsatisfied_count
        penalty = max(penalty, -1.0)
        rewards.append(1.0 + penalty)
    
    return rewards

def tsp_dual_reward_func(completions, **kwargs):
    """
    Reward function for TSP dual bound prediction.
    
    Combines format reward + asymmetric gap penalty:
    - Format reward: +1.0 for valid number in <answer> tags, -1.0 otherwise
    - Gap penalty: -|gap| for underestimation, -2*|gap| for overestimation
    """
    rewards = []
    pids = kwargs.get("problem_id", [])
    
    for i, output_text in enumerate(completions):
        pid = pids[i]
        if hasattr(pid, "item"): pid = pid.item()
        
        data = ProblemRegistry.get(pid)
        optimal_cost = data["optimal_cost"]
        
        # Parse the predicted bound
        predicted = parse_dual_bound(output_text)
        
        if predicted is None:
            # Invalid format - no number found
            rewards.append(-TSP_DUAL_FORMAT_REWARD)
            continue
        
        if predicted <= 0:
            # Invalid bound value (must be positive)
            rewards.append(-TSP_DUAL_FORMAT_REWARD)
            continue
        
        # Valid format - add format reward
        format_reward = TSP_DUAL_FORMAT_REWARD
        
        # Calculate gap reward using the asymmetric penalty
        gap_reward = calculate_dual_reward(
            predicted, 
            optimal_cost, 
            overestimate_mult=TSP_DUAL_OVERESTIMATE_MULT,
            overestimate_penalty=TSP_DUAL_OVERESTIMATE_PENALTY
        )
        
        # Total reward: format bonus + gap penalty (gap_reward is negative)
        total_reward = format_reward + gap_reward
        rewards.append(total_reward)
    
    return rewards

# --- 3. Main Training ---

def main():
    # Setup
    is_vision, _, processor_class = get_model_context(MODEL_ID)
    
    # Load processor/tokenizer
    processor = processor_class.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token

    # Initialize SAT instance pool for background generation
    if PROBLEM_TYPE == "sat" and SAT_CURRICULUM:
        logging.info(f"Initializing SAT instance pool (sizes {SAT_MIN_VARS}-{SAT_MAX_VARS}, ratio={CLAUSE_RATIO})...")
        init_instance_pool(
            min_vars=SAT_MIN_VARS,
            max_vars=SAT_MAX_VARS,
            clause_ratio=CLAUSE_RATIO,
            buffer_size=50,
            num_workers=2
        )
        # Give pool time to generate initial instances
        import time
        time.sleep(3)
        logging.info("Instance pool initialized.")

    # Dataset
    # Load samples from disk (user-provided structure: TSP_DATA_DIR/{n_cities}/*.json)
    logging.info(f"Loading training samples from {TSP_DATA_DIR}...")
    
    samples = []
    
    # Iterate over the requested range of cities
    cities_range = range(TSP_DUAL_MIN_CITIES, TSP_DUAL_MAX_CITIES + 1)
    
    total_loaded = 0
    
    for n_cities in cities_range:
        city_dir = os.path.join(TSP_DATA_DIR, str(n_cities))
        if not os.path.exists(city_dir):
            continue
            
        # List all JSON files
        files = [f for f in os.listdir(city_dir) if f.endswith('.json')]
        
        for f in files:
            path = os.path.join(city_dir, f)
            try:
                with open(path, 'r') as fp:
                    data = json.load(fp)
                
                # Ensure data is valid
                if "coords" not in data or "optimal_cost" not in data:
                    continue
                    
                # Convert coords to numpy
                data["coords"] = np.array(data["coords"])
                
                # Generate prompt (it's not saved in the instance file usually)
                prompt_content = get_tsp_dual_prompt(data["n_cities"], coords=data["coords"], mode=PROMPT_MODE)
                
                # Apply chat template
                messages = [{"role": "user", "content": prompt_content}]
                if hasattr(processor, "apply_chat_template"):
                    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    prompt_text = f"User: {prompt_content}\nAssistant:"
                
                # Register
                pid = ProblemRegistry.register(data)
                samples.append({"prompt": prompt_text, "problem_id": pid})
                total_loaded += 1
                
            except Exception as e:
                logging.warning(f"Failed to load {path}: {e}")

    if total_loaded > 0:
        logging.info(f"Loaded {total_loaded} samples from disk.")
    else:
        logging.warning("No samples found in disk cache! Falling back to generation.")
        # Fallback to generation
        NUM_SAMPLES = 1000 
        logging.info(f"Generating {NUM_SAMPLES} training samples...")
        samples = list(create_problem_samples(PROBLEM_TYPE, NUM_SAMPLES, processor))

    dataset = Dataset.from_list(samples)
    logging.info(f"Dataset created with {len(dataset)} samples.")
    
    # Reward selection
    if PROBLEM_TYPE == "tsp":
        reward_funcs = [tsp_reward_func]
    elif PROBLEM_TYPE == "tsp_dual":
        reward_funcs = [tsp_dual_reward_func]
    else:
        reward_funcs = [sat_reward_func]

    # Training Args
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=f"grpo_{PROBLEM_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        learning_rate=LR,
        per_device_train_batch_size=1, # Micro-batch size per device. TRL handles accumulation.
        gradient_accumulation_steps=4, # Adjust based on memory
        num_generations=GROUP_SIZE,
        generation_batch_size=GROUP_SIZE, # Must be divisible by num_generations
        max_prompt_length=512,
        max_completion_length=MAX_NEW_TOKENS,
        num_train_epochs=1, # We are using max_steps
        max_steps=STEPS, # From config.py
        save_steps=50,
        logging_steps=1,
        bf16=True,
        use_vllm=False, # Disable vLLM for now to avoid complexity, can enable later
        gradient_checkpointing=True, # Critical for memory
        beta=BETA,
        
    )

    # Setup callbacks
    callbacks = []
    if PROBLEM_TYPE == "sat" and SAT_CURRICULUM:
        callbacks.append(CurriculumCallback())

    # Trainer
    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset, # Providing iterable dataset
        processing_class=processor, # TRL will load if not provided, or we can provide
        callbacks=callbacks,
    )
    
    logging.info("Starting GRPO Training...")
    trainer.train()
    
    # Save
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_path)
    logging.info(f"Model saved to {final_path}")
    
    # Save Experiment Configuration
    config_data = {
        "model_id": MODEL_ID,
        "problem_type": PROBLEM_TYPE,
        "group_size": GROUP_SIZE,
        "learning_rate": LR,
        "reward_baseline": REWARD_BASELINE,
        "beta": BETA,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_mode": PROMPT_MODE
    }
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config_data, f, indent=4)
    
    # Benchmarking: Compare base model vs trained model
    from benchmark import evaluate_model
    
    _, ModelClass, _ = get_model_context(MODEL_ID)
    
    csv_file_path = os.path.join(OUTPUT_DIR, 'benchmark_results.csv')
    
    if PROBLEM_TYPE == "tsp":
        # Initialize CSV file with TSP headers
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Stage', 'N_Cities', 'Validity_Rate', 'Avg_Optimality_Gap', 'Model_Wins_vs_NN', 'Avg_Tokens'])

        def log_results(stage, size, metrics):
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    stage, 
                    size, 
                    f"{metrics['validity_rate']:.2f}", 
                    f"{metrics['avg_gap']:.2f}", 
                    metrics['model_wins'], 
                    f"{metrics['avg_tokens']:.2f}"
                ])
        
        eval_sizes = [10, 20, 30, 40, 50]
    
    elif PROBLEM_TYPE == "tsp_dual":
        # Initialize CSV file with TSP Dual headers
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Stage', 'N_Cities', 'Validity_Rate', 'Avg_Gap', 'Overestimate_Rate', 'Avg_Nodes', 'Avg_Tokens'])

        def log_results(stage, size, metrics):
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    stage, 
                    size, 
                    f"{metrics['validity_rate']:.2f}", 
                    f"{metrics['avg_gap']:.2f}", 
                    f"{metrics.get('overestimate_rate', 0.0):.2f}",
                    f"{metrics.get('avg_nodes', 0.0):.1f}",
                    f"{metrics['avg_tokens']:.2f}"
                ])
        
        eval_sizes = [5, 7, 10, 15, 20]
        
    else:  # SAT
        # Initialize CSV file with SAT headers
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Stage', 'N_Vars', 'Validity_Rate', 'Solved_Rate', 'Avg_Tokens'])

        def log_results(stage, size, metrics):
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    stage, 
                    size, 
                    f"{metrics['validity_rate']:.2f}", 
                    metrics['model_wins'],  # For SAT, this is solved count
                    f"{metrics['avg_tokens']:.2f}"
                ])
        
        eval_sizes = SAT_BENCHMARK_SIZES
    
    # Clean up trainer model from memory
    del trainer
    torch.cuda.empty_cache()
    
    # Shutdown instance pool before benchmarking
    if PROBLEM_TYPE == "sat":
        shutdown_instance_pool()
    
    logging.info("\n--- Baseline Benchmarks ---")
    
    # Reload Base Model
    logging.info("Reloading Base Model for Benchmarking...")
    base_model = ModelClass.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()
    
    for size in eval_sizes:
        logging.info(f"Evaluating base model on size {size}...")
        metrics = evaluate_model(
            base_model, processor, 
            n_instances=SAT_BENCHMARK_INSTANCES if PROBLEM_TYPE == "sat" else 20, 
            n_cities=size,  # For SAT, this is n_vars
            device=base_model.device, 
            seed=42
        )
        log_results("Baseline", size, metrics)
        
    del base_model
    torch.cuda.empty_cache()
    
    logging.info("\n--- Post-Training Benchmarks ---")
    
    # Reload Trained Model
    logging.info("Reloading Trained Model for Benchmarking...")
    trained_model = ModelClass.from_pretrained(
        final_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    trained_model.eval()
    
    for size in eval_sizes:
        logging.info(f"Evaluating trained model on size {size}...")
        metrics = evaluate_model(
            trained_model, processor, 
            n_instances=SAT_BENCHMARK_INSTANCES if PROBLEM_TYPE == "sat" else 20, 
            n_cities=size,  # For SAT, this is n_vars
            device=trained_model.device, 
            seed=42
        )
        log_results("Post-Training", size, metrics)
            
    logging.info(f"\nBenchmark results saved to {csv_file_path}")
    
    # --- Final MIP Integration Benchmark for tsp_dual ---
    if PROBLEM_TYPE == "tsp_dual":
        logging.info("\n" + "=" * 60)
        logging.info("FINAL MIP INTEGRATION BENCHMARK")
        logging.info("=" * 60)
        
        from final_benchmark import run_final_benchmark
        
        # Run benchmark at multiple sizes within training range
        benchmark_sizes = [TSP_DUAL_MIN_CITIES, TSP_DUAL_MAX_CITIES]
        # Add midpoint if range is large enough
        if TSP_DUAL_MAX_CITIES - TSP_DUAL_MIN_CITIES >= 4:
            mid = (TSP_DUAL_MIN_CITIES + TSP_DUAL_MAX_CITIES) // 2
            benchmark_sizes = [TSP_DUAL_MIN_CITIES, mid, TSP_DUAL_MAX_CITIES]
        
        for n_cities in benchmark_sizes:
            logging.info(f"\n--- Benchmarking on {n_cities} cities ---")
            benchmark_output = os.path.join(OUTPUT_DIR, f"final_mip_benchmark_{n_cities}cities")
            
            try:
                summary = run_final_benchmark(
                    base_model_path=MODEL_ID,
                    trained_model_path=final_path,
                    n_instances=50,
                    n_cities=n_cities,
                    seed=42,
                    time_limit=60.0,
                    output_dir=benchmark_output
                )
                
                logging.info(f"Results saved to {benchmark_output}")
                
            except Exception as e:
                logging.error(f"Benchmark failed for {n_cities} cities: {e}")
    
    # Cleanup
    del trained_model
    torch.cuda.empty_cache()
    logging.info("\nTraining and benchmarking complete!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
