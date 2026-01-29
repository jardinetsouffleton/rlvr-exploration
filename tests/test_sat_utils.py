import pytest
from src.sat_utils import (
    generate_sat_instance,
    check_sat_solution,
    solve_sat_backtracking,
    parse_sat_output
)

def test_generate_sat_instance():
    """Test SAT instance generation."""
    clauses = generate_sat_instance(n_vars=5, n_clauses=10, seed=42)
    assert len(clauses) == 10
    for c in clauses:
        assert len(c) == 3 # 3-SAT default
        for lit in c:
            assert  1 <= abs(lit) <= 5

def test_check_sat_solution():
    """Test solution checker."""
    # (x1 or x2) AND (-x1 or x2)
    # Solution: x2=True, x1 can be anything
    clauses = [[1, 2], [-1, 2]]
    
    # Valid assignment
    assignment = [1, 2] # x1=T, x2=T
    is_sat, fraction = check_sat_solution(clauses, assignment)
    assert is_sat is True
    assert fraction == 1.0
    
    # Invalid assignment
    assignment = [1, -2] # x1=T, x2=F
    # Clause 1: (T or F) = T
    # Clause 2: (F or F) = F
    is_sat, fraction = check_sat_solution(clauses, assignment)
    assert is_sat is False
    assert fraction == 0.5

def test_solve_sat_backtracking():
    """Test simple backtracking solver."""
    clauses = generate_sat_instance(n_vars=5, n_clauses=10, seed=42)
    # This specific instance might be solvable or not, but solver shouldn't crash
    sol = solve_sat_backtracking(clauses, n_vars=5)
    
    if sol is not None:
        is_sat, _ = check_sat_solution(clauses, sol)
        assert is_sat is True

def test_parse_sat_output():
    """Test parsing logic."""
    # JSON format
    output = "Solution: [1, -2, 3, -4, 5]"
    assignment = parse_sat_output(output, n_vars=5)
    assert assignment == [1, -2, 3, -4, 5]
    
    # <answer> tags
    output = "<answer>1 -2 3 -4 5</answer>"
    assignment = parse_sat_output(output, n_vars=5)
    assert assignment == [1, -2, 3, -4, 5]
