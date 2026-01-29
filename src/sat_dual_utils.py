"""
SAT Dual Bound Utilities (Partial Assignment)

This module provides utilities for the SAT "Dual" task, defined as predicting
a Safe Partial Assignment. A partial assignment is "safe" if it is a subset
of at least one valid full satisfying assignment. This guides the search by
fixing these variables.
"""

import re
import numpy as np
from src.sat_utils import solve_sat_backtracking

def get_sat_partial_prompt(clauses: list, n_vars: int, mode: str = "cot") -> str:
    """
    Generates a prompt asking the LLM to predict a SAFE PARTIAL assignment.
    
    Args:
        clauses: List of clauses (lists of signed integers)
        n_vars: Number of variables
        mode: "cot" or "direct"
        
    Returns:
        Formatted prompt string
    """
    problem_str = f"Find a partial boolean assignment for {n_vars} variables that is GUARANTEED to be part of a valid solution (satisfying assignment) for the following clauses:\n"
    
    for i, clause in enumerate(clauses):
        clause_str = "(" + " OR ".join([f"{'NOT ' if l < 0 else ''}x{abs(l)}" for l in clause]) + ")"
        problem_str += f"{i+1}. {clause_str}\n"
        
    base = problem_str + f"""
Variables are x1 to x{n_vars}. 
Your task is to identify a subset of variables that you are confident about.
Return the assignment as a list of non-zero integers (positive=True, negative=False).
Example: '1 -3' means x1=True, x3=False.
Use minimal assumptions. It is better to assign fewer variables correctly than to guess incorrectly.
"""
    
    if mode == "cot":
        return base + """
First, output your reasoning inside <thought> tags. Explain which variables are constrained.
Then, output your partial assignment inside <answer> tags.
Example:
<thought>
Clause 1 forces x1 to be True because...
</thought>
<answer>
1
</answer>"""
    elif mode == "direct":
        return base + "Output your partial assignment inside <answer> tags."
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")


def parse_partial_assignment(output_text: str, n_vars: int) -> list[int] | None:
    """
    Parses partial assignment from model output.
    Returns a list of unique literals (integers).
    """
    # Try to extract content inside <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    if answer_match:
        content_to_parse = answer_match.group(1)
    else:
        # Fallback: look for last non-empty line or just parse text?
        # Let's trust the logic from src.sat_utils but applied here
        content_to_parse = output_text

    # Extract all numbers
    numbers = [int(s) for s in re.findall(r'-?\b\d+\b', content_to_parse)]
    
    # Filter valid range [-n_vars, n_vars] excluding 0
    valid_nums = []
    seen_vars = set()
    
    for n in numbers:
        abs_n = abs(n)
        if 1 <= abs_n <= n_vars:
            if abs_n not in seen_vars:
                valid_nums.append(n)
                seen_vars.add(abs_n)
            else:
                # Contradiction in output (e.g. "1 -1") or duplicate -> fail parse?
                # Or just ignore subsequent? Let's ignore subsequent for robustness.
                pass
                
    # Note: Empty assignment is valid (0 coverage) but parsing usually implies we found SOMETHING.
    # If no numbers found, return empty list [] rather than None? 
    # Let's return [] if valid but empty, None if parse error.
    # Actually, distinguishing "found nothing" from "parse error" is hard. 
    # Let's assume if there are no numbers, it predicted nothing (empty assignment).
    return valid_nums


def calculate_partial_assignment_reward(
    predicted_assignment: list[int], 
    clauses: list, 
    n_vars: int,
    invalid_penalty: float = 5.0
) -> float:
    """
    Calculates the reward for a partial assignment.
    
    Reward Logic:
    1. Check if the partial assignment is 'Safe' (Extensible to a full solution).
       We use the solver to verify this.
    2. If NOT Safe (Invalid): Return -invalid_penalty.
    3. If Safe: Return coverage fraction (num_assigned / n_vars).
       Range: [0.0, 1.0].
       
    Args:
        predicted_assignment: List of literals
        clauses: SAT clauses
        n_vars: Number of variables
        invalid_penalty: Penalty for unsafe assignments (positive float)
        
    Returns:
        Reward (negative for invalid, positive for valid)
    """
    
    # 0. Empty check
    if not predicted_assignment:
        return 0.0 # Safe but useless
        
    # 1. Verification: Is it extensible?
    # We rely on an exact solver (backtracking) to verify.
    # Note: For very large instances, this might be slow, but for training size (small) it's fine.
    solution = solve_sat_backtracking(clauses, n_vars, fixed_assignment=predicted_assignment)
    
    if solution is None:
        # Not extensible -> UNSAFE -> Heavy Penalty
        return -invalid_penalty
    else:
        # Extensible -> SAFE -> Reward based on coverage
        coverage = len(predicted_assignment) / n_vars
        return coverage
