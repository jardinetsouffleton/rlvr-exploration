"""
TSP Dual Bound Inference Module

Provides functionality to load a trained model and use it to predict
dual (lower) bounds for TSP instances.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.tsp_dual_utils import get_tsp_dual_prompt, parse_dual_bound
from src.config import PROMPT_MODE


def load_model(model_path: str, device: str = "auto"):
    """
    Load a trained model and tokenizer for dual bound prediction.
    
    Args:
        model_path: Path to the trained model directory
        device: Device to load model on ("auto", "cuda", "cpu")
        
    Returns:
        tuple: (model, tokenizer)
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


def predict_dual_bound(
    model, 
    tokenizer, 
    coords: np.ndarray,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    max_retries: int = 3
) -> float | None:
    """
    Use the model to predict a dual (lower) bound for a TSP instance.
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        coords: 2D numpy array of city coordinates (n_cities x 2)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        max_retries: Number of retries if parsing fails (uses sampling for retries)
        
    Returns:
        Predicted lower bound as float, or None if parsing fails
    """
    n_cities = len(coords)
    prompt = get_tsp_dual_prompt(n_cities, coords, mode=PROMPT_MODE)
    
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
        # On retries, force sampling if strictly greedy was requested
        current_temp = temperature
        do_sample = (temperature > 0)
        
        if attempt > 0:
            # If failed once, try sampling to get different result
            if current_temp < 0.1:
                current_temp = 0.7 
                do_sample = True
            print(f"  [Retry {attempt}/{max_retries}] Parsing failed, retrying with temp={current_temp}...")
        
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
        
        # Parse the predicted bound
        predicted = parse_dual_bound(output_text)
        
        if predicted is not None:
            return predicted
            
    return None


def batch_predict_bounds(
    model,
    tokenizer,
    instances: list[np.ndarray],
    max_new_tokens: int = 512,
    batch_size: int = 4
) -> list[float | None]:
    """
    Predict dual bounds for multiple TSP instances.
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        instances: List of coordinate arrays
        max_new_tokens: Maximum tokens per generation
        batch_size: Number of instances to process at once
        
    Returns:
        List of predicted bounds (None for failed parses)
    """
    results = []
    
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i + batch_size]
        
        # For simplicity, process one at a time (batched inference would need padding)
        for coords in batch:
            bound = predict_dual_bound(model, tokenizer, coords, max_new_tokens)
            results.append(bound)
    
    return results


# Standalone test
if __name__ == "__main__":
    from src.tsp_utils import generate_tsp_instance, solve_tsp_optimal
    from src.config import MODEL_ID
    
    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = load_model(MODEL_ID)
    
    # Generate test instance
    coords = generate_tsp_instance(n_cities=5, seed=42)
    _, optimal = solve_tsp_optimal(coords)
    
    print(f"Instance: 5 cities, optimal tour = {optimal:.4f}")
    
    # Predict bound
    predicted = predict_dual_bound(model, tokenizer, coords)
    
    if predicted is not None:
        gap = (predicted - optimal) / optimal * 100
        print(f"Predicted bound: {predicted:.4f}")
        print(f"Gap from optimal: {gap:+.2f}%")
        print(f"Valid bound (≤ optimal): {predicted <= optimal}")
    else:
        print("Failed to parse prediction")
