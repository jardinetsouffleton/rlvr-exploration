import random
import re
import numpy as np

def generate_sat_instance(n_vars: int = 10, n_clauses: int = 30, vars_per_clause: int = 3, seed: int = None):
    """
    Generates a random 3-SAT instance.
    Returns a list of clauses. Each clause is a list of integers.
    Positive integer k means variable k. Negative integer -k means NOT variable k.
    Variables are 1-indexed (1 to n_vars).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    clauses = []
    for _ in range(n_clauses):
        # Pick 3 distinct variables
        vars_in_clause = random.sample(range(1, n_vars + 1), vars_per_clause)
        # Randomly negate
        clause = []
        for v in vars_in_clause:
            if random.random() < 0.5:
                clause.append(-v)
            else:
                clause.append(v)
        clauses.append(clause)
    return clauses

def get_sat_prompt(clauses: list, n_vars: int, mode: str = "cot") -> str:
    """
    Generates the SAT prompt.
    """
    problem_str = f"Find a boolean assignment for {n_vars} variables that satisfies the following clauses:\n"
    
    for i, clause in enumerate(clauses):
        clause_str = "(" + " OR ".join([f"{'NOT ' if l < 0 else ''}x{abs(l)}" for l in clause]) + ")"
        problem_str += f"{i+1}. {clause_str}\n"
        
    base = problem_str + f"\nVariables are x1 to x{n_vars}. Return the assignment as a list of non-zero integers from -{n_vars} to {n_vars} (excluding 0). Positive means True, Negative means False. Example: 1 -2 3 means x1=T, x2=F, x3=T."
    
    if mode == "cot":
        return base + (
            " Solve this problem by using explicit reasoning. Do not generate computer code. Your output should follow this format:\n\n"
            "<thought>\n"
            "To satisfy (x1 OR x2) AND (NOT x1 OR x3), I can set x1=True to satisfy the first clause. "
            "For the second clause (NOT x1 OR x3), since x1 is True, NOT x1 is False, so I must set x3=True. "
            "x2 is free, so I will pick x2=False.\n"
            "</thought>\n"
            "<answer>\n"
            "1 -2 3\n"
            "</answer>"
        )
    elif mode == "direct":
        return base + " Output the assignment inside <answer> tags."
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

def parse_sat_output(output_text: str, n_vars: int) -> list[int] | None:
    """
    Parses SAT assignment from model output.
    """
    # Try to extract content inside <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    if answer_match:
        content_to_parse = answer_match.group(1)
    else:
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
    
    if not valid_nums:
        return None
        
    return valid_nums

def check_sat_solution(clauses: list, assignment: list[int]) -> tuple[bool, float]:
    """
    Checks if assignment satisfies clauses.
    Returns (is_satisfied, fraction_satisfied).
    Assignment: list of integers (positive=True, negative=False).
    """
    # Map variable to bool
    vals = {}
    for l in assignment:
        vals[abs(l)] = (l > 0)
        
    satisfied_count = 0
    for clause in clauses:
        clause_sat = False
        for lit in clause:
            var = abs(lit)
            # Default to False if not in assignment? Or should we penalize?
            # Let's assume standard SAT: if var not assigned, it can be anything? 
            # Usually we require full assignment.
            val = vals.get(var, False) 
            if lit > 0 and val:
                clause_sat = True
                break
            if lit < 0 and not val:
                clause_sat = True
                break
        
        if clause_sat:
            satisfied_count += 1
            
    fraction = satisfied_count / len(clauses)
    return (satisfied_count == len(clauses)), fraction

def solve_sat_backtracking(clauses: list, n_vars: int):
    """
    Simple backtracking solver.
    Returns assignment list or None.
    """
    
    def is_valid(assignment):
        # Check current assignment against clauses
        # Only check clauses where all vars are assigned? 
        # Actually standard DPLL style checking
        vals = {abs(l): (l > 0) for l in assignment}
        
        for clause in clauses:
            # Check if clause is already structurally satisfied or violated
            is_sat = False
            is_undet = False
            for lit in clause:
                var = abs(lit)
                if var in vals:
                    val = vals[var]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        is_sat = True
                        break
                else:
                    is_undet = True
            
            if not is_sat and not is_undet:
                return False # Violated clause
        return True

    def backtrack(assignment):
        if len(assignment) == n_vars:
            return assignment
            
        # Pick next var
        current_vars = {abs(l) for l in assignment}
        next_var = 1
        while next_var in current_vars:
            next_var += 1
            
        # Try True
        if is_valid(assignment + [next_var]):
            res = backtrack(assignment + [next_var])
            if res: return res
            
        # Try False
        if is_valid(assignment + [-next_var]):
            res = backtrack(assignment + [-next_var])
            if res: return res
            
        return None

    return backtrack([])
