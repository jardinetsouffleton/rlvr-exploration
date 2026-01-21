import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, AutoTokenizer
from tsp_utils import generate_tsp_instance, render_tsp_instance, parse_model_output, calculate_tour_length, solve_tsp_nearest_neighbor, solve_tsp_optimal, get_tsp_prompt
from sat_utils import generate_sat_instance, get_sat_prompt, parse_sat_output, check_sat_solution, solve_sat_backtracking
from config import PROBLEM_TYPE, SAT_VARS, SAT_CLAUSES, SAT_VARS_PER_CLAUSE
from qwen_vl_utils import process_vision_info
import numpy as np
from tqdm import tqdm

def evaluate_model(model, processor, n_instances=50, n_cities=10, device="cuda", seed=42):
    """
    Evaluates the model on a fixed seed of instances.
    Returns dict of metrics.
    """
    if PROBLEM_TYPE == "tsp":
        print(f"Running TSP benchmark on {n_instances} instances (N={n_cities})...")
    else:
        print(f"Running SAT benchmark on {n_instances} instances (Vars={SAT_VARS}, Clauses={SAT_CLAUSES})...")

    model.eval()
    
    valid_count = 0
    total_gap = 0.0
    nn_wins = 0 
    model_wins = 0 
    total_tokens = 0
    sat_solved_count = 0
    avg_unsat_clauses = 0
    
    # Use fixed seed for reproducibility across runs
    np.random.seed(seed)
    seeds = np.random.randint(0, 100000, n_instances)
    
    for i in tqdm(range(n_instances)):
        problem_instance = None
        coords = None
        clauses = None
        image = None
        
        if PROBLEM_TYPE == "tsp":
            # Generate Instance
            coords = generate_tsp_instance(n_cities=n_cities, seed=seeds[i])
            image = render_tsp_instance(coords, title="Shortest Path?")
            # Get Ground Truth (Optimal)
            _, opt_len = solve_tsp_optimal(coords)
            # Get NN baseline (optional comparison)
            _, nn_len = solve_tsp_nearest_neighbor(coords)
        elif PROBLEM_TYPE == "sat":
             # Use config SAT_VARS if n_cities param is just recycled, or respect n_cities as n_vars?
             # Let's use n_cities as n_vars for flexibility if passed.
             current_n_vars = n_cities if n_cities > 0 else SAT_VARS
             clauses = generate_sat_instance(n_vars=current_n_vars, n_clauses=SAT_CLAUSES, seed=int(seeds[i]))
             # No image for now
             # Create dummy image if needed for logic below
             pass
        
        # Query Model
        from config import PROMPT_MODE # Import here to pick up latest config
        
        content_payload = []
        if PROBLEM_TYPE == "tsp":
             if hasattr(processor, "image_processor") and processor.image_processor is not None:
                  content_payload = [
                        {"type": "image", "image": image},
                        {"type": "text", "text": get_tsp_prompt(n_cities, coords=coords, mode=PROMPT_MODE)}
                    ]
             else:
                  content_payload = get_tsp_prompt(n_cities, coords=coords, mode=PROMPT_MODE)
        elif PROBLEM_TYPE == "sat":
             current_n_vars = n_cities if n_cities > 0 else SAT_VARS
             prompt = get_sat_prompt(clauses, current_n_vars, mode=PROMPT_MODE)
             if hasattr(processor, "image_processor") and processor.image_processor is not None:
                  # Dummy image for structure consistency if needed
                  from PIL import Image
                  img = Image.new('RGB', (100, 100), color=(255,255,255))
                  content_payload = [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
             else:
                  content_payload = prompt

        if hasattr(processor, "image_processor") and processor.image_processor is not None:
             # Vision Model Logic
            messages = [
                {
                    "role": "user",
                    "content": content_payload
                }
            ]
            
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
        else:
            # Text Only Logic
            messages = [{"role": "user", "content": content_payload}]
            
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = processor(
                text=[text_input],
                padding=True,
                return_tensors="pt"
            ).to(device)

        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False, # Deterministic greedy decoding for benchmark
                temperature=0.0
            )
            
        input_len = inputs.input_ids.shape[1]
        output_ids = generated_ids[:, input_len:]
        num_tokens = output_ids.shape[1]
        output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Parse
        total_tokens += num_tokens
        # Parse
        total_tokens += num_tokens
        
        if PROBLEM_TYPE == "tsp":
            tour = parse_model_output(output_text, n_cities)
            
            # Validation Logic for Metrics
            is_valid = False
            if tour is not None:
                 if len(tour) == n_cities + 1:
                     if len(set(tour)) == n_cities:
                         if tour[0] == tour[-1]:
                             is_valid = True
                             
            if is_valid:
                valid_count += 1
                model_len = calculate_tour_length(tour, coords)
                
                # Gap: (Model - Optimal) / Optimal * 100
                gap = (model_len - opt_len) / opt_len
                total_gap += gap
                
                if model_len < nn_len:
                    model_wins += 1 # Beat NN (since beating opt is impossible unless float error)
                elif model_len > nn_len:
                    nn_wins += 1
            else:
                pass
                
        elif PROBLEM_TYPE == "sat":
             current_n_vars = n_cities if n_cities > 0 else SAT_VARS
             assignment = parse_sat_output(output_text, current_n_vars)
             
             if assignment is not None:
                 # Check strict structure validity (unique vars etc)
                 # parse_sat_output already handles uniqueness and range.
                 # Check if size is correct?
                 if len(assignment) == current_n_vars:
                     valid_count += 1
                     is_sat, fraction = check_sat_solution(clauses, assignment)
                     if is_sat:
                         sat_solved_count += 1
                     avg_unsat_clauses += (1.0 - fraction)
                 
    valid_rate = (valid_count / n_instances) * 100
    
    if PROBLEM_TYPE == "tsp":
        avg_gap = (total_gap / valid_count * 100) if valid_count > 0 else float('inf')
        
        print(f"Benchmark Results (N={n_cities}):")
        print(f"  Validity Rate: {valid_rate:.1f}%")
        print(f"  Avg Optimality Gap (vs Optimal): {avg_gap:+.2f}%")
        print(f"  Model beat Nearest Neighbor: {model_wins} times")
        
        return {
            "validity_rate": valid_rate,
            "avg_gap": avg_gap,
            "model_wins": model_wins,
            "avg_tokens": total_tokens / n_instances
        }
    else:
        # SAT Metrics
        solved_rate = (sat_solved_count / n_instances) * 100
        print(f"Benchmark Results (Vars={n_cities}):")
        print(f"  Validity Rate: {valid_rate:.1f}%")
        print(f"  Solved Rate: {solved_rate:.1f}%")
        
        return {
            "validity_rate": valid_rate,
            "avg_gap": 0.0, # Not applicable
            "model_wins": sat_solved_count,
            "avg_tokens": total_tokens / n_instances
        }

if __name__ == "__main__":
    # Standalone run
    from config import MODEL_ID
    # MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct" # manual override
    print(f"Loading {MODEL_ID} for baseline benchmark...")
    
    try:
        if "VL" in MODEL_ID or "Vision" in MODEL_ID:
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        else:
            processor = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            
        evaluate_model(model, processor)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
