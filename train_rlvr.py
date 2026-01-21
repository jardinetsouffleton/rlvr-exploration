import os
# import psutil
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tsp_utils import generate_tsp_instance, render_tsp_instance, parse_model_output, calculate_tour_length, solve_tsp_nearest_neighbor, solve_tsp_optimal, get_tsp_prompt
from sat_utils import generate_sat_instance, get_sat_prompt, parse_sat_output, check_sat_solution, solve_sat_backtracking
import numpy as np
from tqdm import tqdm
import copy
import logging
import csv
import gc

# Configuration
from qwen_vl_utils import process_vision_info
from benchmark import evaluate_model
from config import MODEL_ID, OUTPUT_DIR, LR, GROUP_SIZE, GENERATION_BATCH_SIZE, TARGET_TOTAL_BATCH_SIZE, N_CITIES, STEPS, MAX_NEW_TOKENS, BETA, REWARD_BASELINE, PROMPT_MODE, PROBLEM_TYPE, SAT_VARS, SAT_CLAUSES, SAT_VARS_PER_CLAUSE, SAT_CURRICULUM, SAT_START_VARS, SAT_END_VARS
import json


# Setup Logging
logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def get_tsp_reward(tour, coords):
    """
    Robust Reward Function for TSP GRPO.
    """
    if tour is None:
        return -10.0 
    
    n_cities = len(coords)
    unique_cities = set(tour)
    has_all_cities = (len(unique_cities) == n_cities)
    
    if not has_all_cities:
        return -5.0 
    
    if len(tour) == n_cities + 1:
        if tour[0] != tour[-1]:
            return -5.0 
    elif len(tour) == n_cities:
        tour = list(tour) + [tour[0]]
    else:
        return -5.0

    model_len = calculate_tour_length(tour, coords)
    
    if globals().get("REWARD_BASELINE") == "optimal":
        _, baseline_len = solve_tsp_optimal(coords)
    else:
        _, baseline_len = solve_tsp_nearest_neighbor(coords)
        
    gap = (baseline_len - model_len) / baseline_len
    gap = max(gap, -3.0)
    
    return 1.0 + gap

def get_sat_reward(assignment, clauses, n_vars):
    """
    Reward function for SAT.
    """
    # 1. Format Failure
    if assignment is None:
        return -20 # Less severe than TSP because partial parsing is harder? or same?
        
    # 2. Validity Check
    # We expect a list of N_VARS signed integers unique by absolute value
    unique_vars = {abs(l) for l in assignment}
    
    if len(unique_vars) != n_vars:
        # Partial assignment or duplicates
        return -10.0
        
    # 3. Satisfaction
    # Calculate unsatisfied clauses
    vals = {abs(l): (l > 0) for l in assignment}
    unsatisfied_count = 0
    for clause in clauses:
        clause_sat = False
        for lit in clause:
            var = abs(lit)
            val = vals.get(var, False) 
            if (lit > 0 and val) or (lit < 0 and not val):
                clause_sat = True
                break
        if not clause_sat:
            unsatisfied_count += 1
            
    # Penalty per unsatisfied clause
    # We want to encourage getting it right.
    # -0.1 per clause. 30 clauses -> -3.0 max penalty.
    # Base valid reward = 1.0
    # Range: [-2.0, 1.0]
    
    penalty = -1 * unsatisfied_count
    # Clamp to ensure it doesn't go below constraint failure (theoretical check)
    penalty = max(penalty, -10.0)
    
    return 1.0 + penalty

def get_sat_stats(assignment, clauses, n_vars):
    """
    Returns (is_solved, unsatisfied_count)
    """
    if assignment is None:
        return False, len(clauses) # Worst case
        
    unique_vars = {abs(l) for l in assignment}
    if len(unique_vars) != n_vars:
         # Treat constraint failure as all clauses violated? 
         # Or just don't count it for violation stats? 
         # Let's say all violated for simplicity or N/A
         return False, len(clauses)

    vals = {abs(l): (l > 0) for l in assignment}
    unsatisfied_count = 0
    for clause in clauses:
        clause_sat = False
        for lit in clause:
            var = abs(lit)
            val = vals.get(var, False) 
            if (lit > 0 and val) or (lit < 0 and not val):
                clause_sat = True
                break
        if not clause_sat:
            unsatisfied_count += 1
            
    is_solved = (unsatisfied_count == 0)
    return is_solved, unsatisfied_count

def get_reward(model_output_parsed, problem_instance):
    if PROBLEM_TYPE == "tsp":
        return get_tsp_reward(model_output_parsed, problem_instance)
    elif PROBLEM_TYPE == "sat":
        clauses, n_vars = problem_instance
        return get_sat_reward(model_output_parsed, clauses, n_vars)
    else:
         raise ValueError(f"Unknown problem type {PROBLEM_TYPE}")

# Helper to determine model type and context
def get_model_context(model_id):
    is_vision = "VL" in model_id or "Vision" in model_id
    
    if is_vision:
        logging.info(f"Detected Vision-Language Model: {model_id}")
        config_class = AutoModelForVision2Seq
        processor_class = AutoProcessor
    else:
        logging.info(f"Detected Text-Only Model: {model_id}")
        config_class = AutoModelForCausalLM
        processor_class = AutoTokenizer
        
    return is_vision, config_class, processor_class

def train_model(model, processor, steps=STEPS, save_path=None):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "rlvr_model_final")

    # Determine context based on model type
    is_vision = isinstance(model, AutoModelForVision2Seq) or (hasattr(model, "config") and "VL" in model.config.architectures[0])
    # Fallback check if class check fails (e.g. wrapped) - usually instance check is enough if we loaded correctly.
    # But since we pass the model in, let's trust the logic that loaded it or re-check.
    # Actually, simpler: check the processor.
    is_vision = hasattr(processor, "image_processor") and processor.image_processor is not None


    # Load reference model (reloaded)
    logging.info(f"Loading Reference Model for KL divergence...")
    
    _, ModelClass, _ = get_model_context(MODEL_ID)
    
    ref_model = ModelClass.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=LR)
    
    # Gradient Accumulation Setup
    # Effective Batch per step = 1 (Problem) * GROUP_SIZE
    effective_parallel_size = GROUP_SIZE
    grad_accum_steps = int(np.ceil(TARGET_TOTAL_BATCH_SIZE / effective_parallel_size))
    
    logging.info(f"Target Batch: {TARGET_TOTAL_BATCH_SIZE} | Per-Step: {effective_parallel_size} | Accum Steps: {grad_accum_steps}")
    logging.info(f"Effective Batch Size (Accumulated): {effective_parallel_size * grad_accum_steps}")

    # Learning Rate Scheduler
    num_update_steps = steps // grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, # Default to 0 warmup
        num_training_steps=num_update_steps
    )

    logging.info(f"Starting Training Loop (RLVR/GRPO) for {steps} steps ({num_update_steps} update steps)...")
    
    # Reset random seed for training to ensure diversity from fixed-seed eval
    np.random.seed(None)
    
    # Initialize Metrics CSV
    metrics_file = os.path.join(OUTPUT_DIR, "training_metrics.csv")
    with open(metrics_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "reward_mean", "loss", "valid_tours", "avg_tokens", "memory_mb"])
    
    model.train()
    
    for step in range(steps):
        # 1. Data Generation (Experience Collection)
        batch_rewards = []
        
        # Problem Generation
        if PROBLEM_TYPE == "tsp":
            # TSP Generation
            n_cities = np.random.randint(5, 8)
            problem_instance = generate_tsp_instance(n_cities=n_cities)
            coords = problem_instance
            
            if is_vision:
                 image = render_tsp_instance(coords, title="Find the shortest path")
                 content_payload = [
                        {"type": "image", "image": image},
                        {"type": "text", "text": get_tsp_prompt(n_cities, coords=coords, mode=PROMPT_MODE)}
                    ]
            else:
                 content_payload = get_tsp_prompt(n_cities, coords=coords, mode=PROMPT_MODE)
                 
        elif PROBLEM_TYPE == "sat":
            # SAT Generation
            if SAT_CURRICULUM:
                # Curriculum Learning: Scale max_vars linearly
                progress = step / steps
                current_max = SAT_START_VARS + (SAT_END_VARS - SAT_START_VARS) * progress
                # Sample n_vars from [start, current_max]
                n_vars = np.random.randint(SAT_START_VARS, int(current_max) + 1)
                
                # Scale clauses to maintain difficulty (ratio from config)
                # Base ratio: SAT_CLAUSES / SAT_VARS
                ratio = SAT_CLAUSES / SAT_VARS
                n_clauses = int(n_vars * ratio)
            else:
                n_vars = np.random.randint(SAT_VARS-2, SAT_VARS+3) # Variation
                n_clauses = SAT_CLAUSES
            clauses = generate_sat_instance(n_vars=n_vars, n_clauses=n_clauses, seed=None)
            problem_instance = (clauses, n_vars)
            
            text_prompt = get_sat_prompt(clauses, n_vars, mode=PROMPT_MODE)
            
            if is_vision:
                 # Create placeholder image for SAT
                 from PIL import Image, ImageDraw
                 img = Image.new('RGB', (200, 100), color = (255, 255, 255))
                 d = ImageDraw.Draw(img)
                 d.text((10,10), "SAT Problem", fill=(0,0,0))
                 
                 content_payload = [
                        {"type": "image", "image": img},
                        {"type": "text", "text": text_prompt}
                    ]
            else:
                 content_payload = text_prompt
        
        # Build Messages (Single Instance)
        messages = [{"role": "user", "content": content_payload}]
        
        # Prepare inputs template
        if is_vision:
             text_input_template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
             image_inputs_template, video_inputs_template = process_vision_info(messages)
        else:
             text_input_template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
             image_inputs_template = None
             video_inputs_template = None

        # Micro-Batch Generation
        all_generated_ids = []
        all_completions_text = []

        num_generations = int(np.ceil(GROUP_SIZE / GENERATION_BATCH_SIZE))
        
        for gen_i in range(num_generations):
            curr_batch_size = min(GENERATION_BATCH_SIZE, GROUP_SIZE - gen_i * GENERATION_BATCH_SIZE)
            if curr_batch_size <= 0: break
            
            # Prepare inputs for this chunk
            # Prepare inputs for this chunk
            texts = [text_input_template] * curr_batch_size
            
            call_kwargs = {
                "text": texts,
                "padding": True,
                "return_tensors": "pt"
            }
            if is_vision:
                 if image_inputs_template:
                     call_kwargs["images"] = image_inputs_template * curr_batch_size
                 if video_inputs_template:
                     call_kwargs["videos"] = video_inputs_template * curr_batch_size
                 
            # Qwen Processor call
            # Note: We move inputs to device immediately
            inputs = processor(**call_kwargs).to(model.device)
            
            # Rollout
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=1.5,
                    top_k=20,
                    top_p=0.95
                )
            
            # Decode and Store
            input_len = inputs.input_ids.shape[1]
            comp_ids = gen_ids[:, input_len:]
            chunk_texts = processor.batch_decode(comp_ids, skip_special_tokens=True)
            
            all_completions_text.extend(chunk_texts)
            all_generated_ids.append(gen_ids.cpu())
            
            # Cleanup
            del inputs, gen_ids, comp_ids
            torch.cuda.empty_cache()
            
        # Combine generated_ids
        # Pad sequences to max length across all chunks
        all_ids_flat = [seq for batch in all_generated_ids for seq in batch]
        
        if hasattr(processor, "tokenizer"):
             pad_token_id = processor.tokenizer.pad_token_id
        else:
             pad_token_id = processor.pad_token_id
             
        # Pad stack
        full_generated_ids = torch.nn.utils.rnn.pad_sequence(
            all_ids_flat, batch_first=True, padding_value=pad_token_id
        ) # (GROUP_SIZE, MaxSeqLen)

        # 3. Compute Rewards
        rewards = []
        valid_count = 0
        sat_solved_count = 0
        total_unsat = 0
        
        for text in all_completions_text:
            if PROBLEM_TYPE == "tsp":
                parsed = parse_model_output(text, n_cities) 
                r = get_reward(parsed, problem_instance)
                rewards.append(r)
                if r > -0.9: valid_count += 1
                
            elif PROBLEM_TYPE == "sat":
                 parsed = parse_sat_output(text, n_vars) 
                 r = get_reward(parsed, problem_instance)
                 rewards.append(r)
                 
                 is_solved, n_unsat = get_sat_stats(parsed, clauses, n_vars)
                 if is_solved: sat_solved_count += 1
                 total_unsat += n_unsat
                 if is_solved: valid_count += 1
            
        rewards_tensor = torch.tensor(rewards, device=model.device, dtype=torch.float32)
        
        # 4. GRPO Advantage
        if GROUP_SIZE > 1:
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std() + 1e-8
            advantages = (rewards_tensor - mean_r) / std_r
        else:
            advantages = torch.zeros_like(rewards_tensor)
            mean_r = rewards_tensor.mean() # for logging
        
        # 5. Micro-Batched Training (Forward/Backward)
        
        total_loss_accum = 0.0
        total_tokens_accum = 0 # for logging
        
        # We need to process full_generated_ids in chunks again
        # Re-using num_generations logic or just simple stepping
        
        for chunk_start in range(0, GROUP_SIZE, GENERATION_BATCH_SIZE):
            chunk_end = min(GROUP_SIZE, chunk_start + GENERATION_BATCH_SIZE)
            curr_chunk_size = chunk_end - chunk_start
            
            chunk_ids = full_generated_ids[chunk_start:chunk_end].to(model.device)
            chunk_adv = advantages[chunk_start:chunk_end].to(model.device)
            
            # Re-calculate input len? 
            # Caution: padding might have changed effective input_len if left-padding vs right-padding?
            # Qwen uses left padding? Usually CausalLM does right padding for training, left for generation.
            # model.generate returns left-padded? 
            # Wait, `pad_sequence` right-pads by default.
            # If the original inputs were left-padded, and we generated, the generation is appended.
            # The input part should be identical across batch except for padding.
            # But here `full_generated_ids` is (GROUP_SIZE, L).
            # We treat it as standard input.
            # We assume `input_len` (from first chunk) is roughly correct for masking if we just want to mask prompt.
            # BUT, since we padded `full_generated_ids`, the prompt position is preserved relative to start 0?
            # Yes, `pad_sequence` keeps start aligned.
            # IMPORTANT: We need `input_len` to mask out the prompt from loss.
            # Since all prompts are identical text, `input_len` is constant.
            # Actually, `processor(padding=True)` might have introduced padding in the prompt if we had multiple inputs?
            # Here all inputs are identical, so `input_len` is constant and there is NO padding in the prompt part.
            # So `input_len` from the first generation chunk is safe to use.
            # Variable `input_len` was defined in the generation loop. It holds the last chunk's input len.
            # Which is fine since they are identical.
            
            # Forward Policy
            policy_outputs = model(input_ids=chunk_ids, attention_mask=(chunk_ids != pad_token_id).long())
            
            # Forward Reference
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=chunk_ids, attention_mask=(chunk_ids != pad_token_id).long())
                
            # LogProbs
            # shift logits inside
            policy_logits = policy_outputs.logits[:, input_len-1 : -1, :]
            ref_logits = ref_outputs.logits[:, input_len-1 : -1, :]
            
            chunk_gen_tokens = chunk_ids[:, input_len:]
            # Note: chunk_ids was padded. chunk_gen_tokens includes the padding at the end.
            
            policy_logprobs = torch.log_softmax(policy_logits, dim=-1)
            token_logprobs = torch.gather(policy_logprobs, 2, chunk_gen_tokens.unsqueeze(-1)).squeeze(-1)
            
            ref_logprobs = torch.log_softmax(ref_logits, dim=-1)
            ref_token_logprobs = torch.gather(ref_logprobs, 2, chunk_gen_tokens.unsqueeze(-1)).squeeze(-1)
             
            pad_mask = (chunk_gen_tokens != pad_token_id).float()
            
            # KL
            kl_div = token_logprobs - ref_token_logprobs
            
            # Loss
            advantage_expanded = chunk_adv.unsqueeze(1).expand_as(token_logprobs)
            loss_per_token = - (advantage_expanded * token_logprobs) + BETA * kl_div
            
            # Mean over chunk tokens
            chunk_loss = (loss_per_token * pad_mask).sum() / pad_mask.sum()
            
            # Weighted average for Total Loss?
            # GRPO: Expectation over Group. 
            # Sum(Loss_i) / GroupSize.
            # Here chunk_loss is Mean(Loss_i for i in Chunk).
            # So Contribution = chunk_loss * (ChunkSize / GroupSize).
            
            weight = curr_chunk_size / GROUP_SIZE
            weighted_loss = chunk_loss * weight
            
            # Normalize by Accum Steps
            final_loss = weighted_loss / grad_accum_steps
            
            final_loss.backward()
            
            total_loss_accum += weighted_loss.item()
            total_tokens_accum += pad_mask.sum().item()
            
            del policy_outputs, ref_outputs, policy_logits, ref_logits
            del policy_logprobs, ref_logprobs, token_logprobs, ref_token_logprobs
            del chunk_loss, final_loss, pad_mask
            torch.cuda.empty_cache()

        
        is_update_step = (step + 1) % grad_accum_steps == 0
        if is_update_step:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        mean_r_val = mean_r.item()
        loss_val = total_loss_accum # This is already averaged over group
        
        # Metrics & Logging
        with torch.no_grad():
             avg_tokens = total_tokens_accum / GROUP_SIZE
             
             mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
             torch.cuda.reset_peak_memory_stats()
             
             update_tag = "[UPDATE]" if is_update_step else ""
             
             if PROBLEM_TYPE == "tsp":
                  logging.info(f"Step {step} {update_tag}: Reward Mean={mean_r_val:.4f}, Loss={loss_val:.4f}, ValidTours={valid_count}/{GROUP_SIZE}, AvgTokens={avg_tokens:.1f}, Mem={mem_mb:.0f}MB")
             else:
                   avg_unsat = total_unsat / GROUP_SIZE
                   current_lr = scheduler.get_last_lr()[0]
                   logging.info(f"Step {step} {update_tag}: Reward Mean={mean_r_val:.4f}, Loss={loss_val:.4f}, Solved={sat_solved_count}/{GROUP_SIZE}, AvgViolations={avg_unsat:.2f}, AvgTokens={avg_tokens:.1f}, Mem={mem_mb:.0f}MB, LR={current_lr:.2e}")
             
             # Save to CSV
             with open(metrics_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if PROBLEM_TYPE == "tsp":
                     writer.writerow([step, mean_r_val, loss_val, valid_count, avg_tokens, mem_mb])
                else:
                     writer.writerow([step, mean_r_val, loss_val, sat_solved_count, avg_tokens, mem_mb])
             
             if step % 10 == 0:
                 logging.info(f"Sample Output: {all_completions_text[0]}")
                 
        # Explicit Memory Cleanup
        del all_generated_ids, full_generated_ids, all_completions_text, batch_rewards, rewards_tensor, advantages, input_len
        
        torch.cuda.empty_cache()
        gc.collect()

    
    logging.info("Training Complete. Saving model...")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    logging.info(f"Model saved to {save_path}")

    return model
    
if __name__ == "__main__":
    print(f"Loading model: {MODEL_ID}...")
    
    is_vision, ModelClass, ProcessorClass = get_model_context(MODEL_ID)
    
    processor = ProcessorClass.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = ModelClass.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    # Enable Gradient Checkpointing to save memory
    # model.gradient_checkpointing_enable() # Disabled for 1.5B to avoid requires_grad issues and because it fits in memory

    
    # Run internal loop
    from benchmark import evaluate_model
    
    # Save Experiment Configuration
    config_data = {
        "model_id": MODEL_ID,
        "steps": STEPS,
        "group_size": GROUP_SIZE,
        "learning_rate": LR,
        "reward_baseline": REWARD_BASELINE,
        "beta": BETA,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_mode": PROMPT_MODE
    }
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config_data, f, indent=4)
        
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

    train_model(model, processor, steps=STEPS) 
    
    # Clean up trained model from memory to make room for benchmarking
    del model
    torch.cuda.empty_cache()
    
    print("\n--- Baseline Benchmarks ---")
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
        print(f"\nEvaluating on size {size}...")
        metrics = evaluate_model(base_model, processor, n_instances=20, n_cities=size, device=base_model.device, seed=42)
        log_results("Baseline", size, metrics)
        
    del base_model
    torch.cuda.empty_cache()
    
    print("\n--- Post-Training Benchmarks ---")
    
    # Reload Trained Model
    logging.info("Reloading Trained Model for Benchmarking...")
    save_path = os.path.join(OUTPUT_DIR, "rlvr_model_final")
    trained_model = ModelClass.from_pretrained(
        save_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    trained_model.eval()
    
    for size in eval_sizes:
        print(f"\nEvaluating on size {size}...")
        metrics = evaluate_model(trained_model, processor, n_instances=20, n_cities=size, device=trained_model.device, seed=42)
        log_results("Post-Training", size, metrics)
            
    print(f"\nResults saved to {csv_file_path}")
