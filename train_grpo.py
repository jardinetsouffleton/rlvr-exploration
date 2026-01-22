
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
from config import MODEL_ID, OUTPUT_DIR, LR, GROUP_SIZE, GENERATION_BATCH_SIZE, STEPS, MAX_NEW_TOKENS, BETA, REWARD_BASELINE, PROMPT_MODE, PROBLEM_TYPE, SAT_VARS, SAT_CLAUSES, SAT_CURRICULUM, SAT_START_VARS, SAT_END_VARS

import torch
import logging
import csv
import json
import numpy as np
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from trl import GRPOTrainer, GRPOConfig

# Import utils from existing files
from tsp_utils import generate_tsp_instance, render_tsp_instance, parse_model_output, calculate_tour_length, solve_tsp_nearest_neighbor, solve_tsp_optimal, get_tsp_prompt
from sat_utils import generate_sat_instance, get_sat_prompt, parse_sat_output, check_sat_solution

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
            if SAT_CURRICULUM:
                n_vars = np.random.randint(SAT_START_VARS, SAT_END_VARS + 1)
                ratio = SAT_CLAUSES / SAT_VARS
                n_clauses = int(n_vars * ratio)
            else:
                n_vars = np.random.randint(SAT_VARS-2, SAT_VARS+3)
                n_clauses = SAT_CLAUSES
            
            clauses = generate_sat_instance(n_vars=n_vars, n_clauses=n_clauses, seed=None)
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
            rewards.append(-20.0)
            continue
            
        unique_vars = {abs(l) for l in assignment}
        if len(unique_vars) != c_n_vars:
            rewards.append(-10.0)
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
        penalty = max(penalty, -10.0)
        rewards.append(1.0 + penalty)
    
    return rewards

# --- 3. Main Training ---

def main():
    # Setup
    is_vision, _, processor_class = get_model_context(MODEL_ID)
    
    # Load processor/tokenizer
    processor = processor_class.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token

    # Dataset
    # Generate samples in-memory to avoid disk cache issues
    NUM_SAMPLES = 1000 # Pre-generate this many samples (reduced for testing)
    logging.info(f"Generating {NUM_SAMPLES} training samples...")
    samples = list(create_problem_samples(PROBLEM_TYPE, NUM_SAMPLES, processor))
    dataset = Dataset.from_list(samples)
    logging.info(f"Dataset created with {len(dataset)} samples.")
    # Reward selection
    if PROBLEM_TYPE == "tsp":
        reward_funcs = [tsp_reward_func]
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

    # Trainer
    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset, # Providing iterable dataset
        processing_class=processor, # TRL will load if not provided, or we can provide
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
    # Only run for TSP since benchmark.py is TSP-specific
    if PROBLEM_TYPE == "tsp":
        from benchmark import evaluate_model
        
        _, ModelClass, _ = get_model_context(MODEL_ID)
        
        csv_file_path = os.path.join(OUTPUT_DIR, 'benchmark_results.csv')
        
        # Initialize CSV file with headers
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
        
        # Clean up trainer model from memory
        del trainer
        torch.cuda.empty_cache()
        
        logging.info("\n--- Baseline Benchmarks ---")
        eval_sizes = [10, 20, 30, 40, 50]
        
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
            metrics = evaluate_model(base_model, processor, n_instances=20, n_cities=size, device=base_model.device, seed=42)
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
            metrics = evaluate_model(trained_model, processor, n_instances=20, n_cities=size, device=trained_model.device, seed=42)
            log_results("Post-Training", size, metrics)
                
        logging.info(f"\nBenchmark results saved to {csv_file_path}")
    else:
        logging.info("Benchmarking skipped (only available for TSP problem type)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
