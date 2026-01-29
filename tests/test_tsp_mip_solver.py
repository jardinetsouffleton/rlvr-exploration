import pytest
import numpy as np
from src.tsp_mip_solver import solve_tsp, solve_with_llm_bound
from src.tsp_utils import solve_tsp_optimal, generate_tsp_instance

def test_mip_constraints_impossible_bound():
    """Test Case A: The 'Impossible' Bound (Overestimation)."""
    # 1. Setup
    np.random.seed(42)
    coords = generate_tsp_instance(n_cities=8, seed=42)
    _, true_opt = solve_tsp_optimal(coords)
    
    # 2. Inject impossible lower bound
    fake_bound = true_opt * 1.5
    
    # 3. Solve with fallback disabled to verify constraint holds
    _, cost_A, stats_A = solve_with_llm_bound(
        coords, 
        predicted_bound=fake_bound, 
        time_limit=5.0,
        gap_tolerance=0.0,
        fallback_on_infeasible=False
    )
    
    # 4. Assertions
    # If the solver respected the bound, the cost must be >= fake_bound * 0.999
    # OR it should be infeasible (cost=inf).
    # It must NOT be the true optimal.
    assert (cost_A >= fake_bound * 0.999) or (stats_A['status'] == 'OptimizationStatus.INFEASIBLE')
    assert abs(cost_A - true_opt) > 0.001, "Solver ignored the bound and found optimal!"

def test_mip_constraints_valid_bound():
    """Test Case B: The 'Tight' Bound (Valid Underestimation)."""
    # 1. Setup
    np.random.seed(42)
    coords = generate_tsp_instance(n_cities=8, seed=42)
    _, true_opt = solve_tsp_optimal(coords)
    
    # 2. Inject valid lower bound
    tight_bound = true_opt * 0.95
    
    _, cost_B, stats_B = solve_with_llm_bound(
        coords, 
        predicted_bound=tight_bound, 
        time_limit=5.0
    )
    
    # 3. Assertions
    # Should find optimal solution
    assert abs(cost_B - true_opt) < 0.001, "Solver failed to find optimal with valid bound"
