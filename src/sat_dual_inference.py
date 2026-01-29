"""
SAT Dual Bound Inference Module (Partial Assignment)

Provides functionality to load a trained model and use it to predict
Safe Partial Assignments for SAT instances.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.sat_dual_utils import get_sat_partial_prompt, parse_partial_assignment
from src.sat_utils import generate_sat_instance, solve_sat_backtracking

# Config (could be imported)
PROMPT_MODE = "cot" 

def load_model(model_path: str, device: str = "auto"):
    """
    Load a trained model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def predict_partial_assignment(
    model, 
    tokenizer, 
    clauses: list,
    n_vars: int,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    max_retries: int = 3
) -> list[int]:
    """
    Use the model to predict a partial assignment.
    
    Returns:
        List of integers (literals), or empty list if failure.
    """
    prompt = get_sat_partial_prompt(clauses, n_vars, mode=PROMPT_MODE)
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text_input = f"User: {prompt}\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(text_input, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Retry loop
    for attempt in range(max_retries + 1):
        current_temp = temperature
        do_sample = (temperature > 0)
        
        if attempt > 0:
            if current_temp < 0.1:
                current_temp = 0.7 
                do_sample = True
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=current_temp if do_sample else None,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode output
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Parse
        predicted = parse_partial_assignment(output_text, n_vars)
        
        # Note: parse_partial_assignment returns [] if empty, valid.
        # It's hard to distinguish "formatting error" from "empty prediction".
        # We assume if it returns a list (even empty), it's a parsed result.
        # But for robustness, if we get nothing and we are retrying, maybe we can assume it failed?
        # Actually [] is valid. Let's return it.
        return predicted
            
    return []


def batch_predict(
    model,
    tokenizer,
    instances: list[tuple[list, int]], # list of (clauses, n_vars)
    max_new_tokens: int = 512,
    batch_size: int = 4
) -> list[list[int]]:
    """
    Predict partial assignments for multiple instances.
    """
    results = []
    
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i + batch_size]
        for clauses, n_vars in batch:
            assignment = predict_partial_assignment(model, tokenizer, clauses, n_vars, max_new_tokens)
            results.append(assignment)
    
    return results


# Standalone test
if __name__ == "__main__":
    from src.config import MODEL_ID
    
    print(f"Loading model: {MODEL_ID}")
    try:
        model, tokenizer = load_model(MODEL_ID)
        
        # Generate test instance
        n_vars = 10
        clauses = generate_sat_instance(n_vars=n_vars, n_clauses=20, seed=42)
        
        print(f"SAT Instance: {n_vars} vars, {len(clauses)} clauses")
        
        # Predict
        predicted = predict_partial_assignment(model, tokenizer, clauses, n_vars)
        print(f"Predicted Assignment: {predicted}")
        
        # Check validity
        solution = solve_sat_backtracking(clauses, n_vars, fixed_assignment=predicted)
        if solution:
            print(f"VALID! Extensible to: {solution}")
            coverage = len(predicted) / n_vars
            print(f"Coverage: {coverage:.2%}")
        else:
            print("INVALID! Contradicts logic.")
            
    except Exception as e:
        print(f"Test failed: {e}")
        # If model load fails (e.g. no GPU), we just skip
