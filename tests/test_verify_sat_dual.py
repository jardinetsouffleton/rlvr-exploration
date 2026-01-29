"""
Verification script for SAT Dual infrastructure.
"""
import sys
import numpy as np
from src.sat_utils import generate_sat_instance, solve_sat_backtracking
from src.sat_dual_utils import get_sat_partial_prompt, parse_partial_assignment, calculate_partial_assignment_reward

def test_prompt_generation():
    print("Testing Prompt Generation...")
    clauses = generate_sat_instance(n_vars=5, n_clauses=10, seed=42)
    prompt = get_sat_partial_prompt(clauses, n_vars=5, mode="cot")
    assert "Find a partial boolean assignment" in prompt
    assert "Example:" in prompt
    print("PASS: Prompt generation looks correct.")

def test_parsing():
    print("Testing Parsing...")
    
    # Case 1: Valid COT
    output_1 = """
    <thought>
    x1 needs to be true.
    </thought>
    <answer>
    1 -2 3
    </answer>
    """
    parsed_1 = parse_partial_assignment(output_1, n_vars=5)
    assert set(parsed_1) == {1, -2, 3}
    
    # Case 2: Out of bounds
    output_2 = "<answer>1 6 -2</answer>" # 6 is > 5
    parsed_2 = parse_partial_assignment(output_2, n_vars=5)
    assert set(parsed_2) == {1, -2} # 6 should be filtered
    
    # Case 3: Empty
    output_3 = "<answer></answer>"
    parsed_3 = parse_partial_assignment(output_3, n_vars=5)
    assert parsed_3 == []
    
    print("PASS: Parsing logic works.")

def test_reward_logic():
    print("Testing Reward Logic (Solver Integration)...")
    
    # Generate a solvable instance
    # Fixed seed ensures we know the solution if we want, but we'll use solver.
    n_vars = 5
    clauses = generate_sat_instance(n_vars=n_vars, n_clauses=5, seed=42)
    
    # Get a Full Solution first to know what IS valid
    full_sol = solve_sat_backtracking(clauses, n_vars)
    print(f"Full solution found: {full_sol}")
    assert full_sol is not None, "Instance should be solvable for this test"
    
    # Test 1: Valid Partial (subset of full)
    partial_valid = full_sol[:2] # First 2 vars
    reward_valid = calculate_partial_assignment_reward(partial_valid, clauses, n_vars)
    print(f"Ref Partial: {partial_valid}, Reward: {reward_valid}")
    assert reward_valid > 0
    assert abs(reward_valid - (2/5)) < 1e-6
    
    # Test 2: Invalid Partial (contradicts full)
    # Note: It MIGHT be valid for ANOTHER solution, but likely not if we flip a constrained var.
    # Let's try to flip a variable.
    partial_invalid = [-1 * x for x in partial_valid] 
    # Check if this flip is actually invalid (it might be valid if multiple solutions exist)
    # We force check
    is_actually_valid = solve_sat_backtracking(clauses, n_vars, fixed_assignment=partial_invalid)
    
    reward_invalid = calculate_partial_assignment_reward(partial_invalid, clauses, n_vars)
    print(f"Inv Partial: {partial_invalid}, Is Extensible: {is_actually_valid is not None}, Reward: {reward_invalid}")
    
    if is_actually_valid is None:
        assert reward_invalid < 0 # Should be penalized
    else:
        print("WARN: Flipped assignment happened to be valid (multiple solutions).")
        assert reward_invalid > 0

    print("PASS: Reward logic works.")

if __name__ == "__main__":
    try:
        test_prompt_generation()
        test_parsing()
        test_reward_logic()
        print("\nALL TESTS PASSED!")
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
