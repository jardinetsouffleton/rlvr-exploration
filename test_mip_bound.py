
import numpy as np
import time
from tsp_mip_solver import solve_tsp, solve_with_llm_bound
from tsp_utils import solve_tsp_optimal, generate_tsp_instance

def test_mip_constraints():
    print("=== Testing MIP Solver Bound Integration ===\n")
    
    # 1. Setup a simple random instance
    np.random.seed(42)
    # Using small N ensures fast solve
    coords = generate_tsp_instance(n_cities=8, seed=42)
    
    # 2. Get Ground Truth
    print("Solving for ground truth...")
    _, true_opt = solve_tsp_optimal(coords)
    print(f"True Optimal Cost: {true_opt:.4f}")
    
    # 3. Test Case A: The "Impossible" Bound (Overestimation)
    # If the solver respects the bound, this MUST fail (Infeasible).
    # If the solver ignores the bound, it will just find the optimal solution.
    fake_bound = true_opt * 1.5
    print(f"\n[Test A] Injecting impossible lower bound: {fake_bound:.4f} (True: {true_opt:.4f})")
    print("Expectation: Solver should return status = INFEASIBLE or fail to find solution")
    
    _, cost_A, stats_A = solve_with_llm_bound(
        coords, 
        predicted_bound=fake_bound, 
        time_limit=5.0,
        gap_tolerance=0.0,
        fallback_on_infeasible=False
    )
    
    print(f"Result A: Status={stats_A['status']}, Cost={cost_A}")
    
    # In minimization, if we enforce cost >= X (where X > optimal), 
    # the solver should return a solution with cost >= X (a suboptimal tour)
    # OR return Infeasible if no such tour exists (unlikely for TSP).
    # The critical check is that it did NOT return true_opt.
    if cost_A >= fake_bound * 0.999:
        print(f"✅ PASS: Solver respected the bound. Found cost {cost_A:.4f} >= {fake_bound:.4f} (Excluded optimal {true_opt:.4f})")
    elif abs(cost_A - true_opt) < 0.001:
        print("❌ FAIL: Solver IGNORED the bound and found the optimal solution anyway!")
    else:
        print(f"❓ WARNING: Unexpected outcome: {cost_A}")

    # 4. Test Case B: The "Tight" Bound (Valid Underestimation)
    # A bound slightly below optimal should still allow finding the optimal.
    tight_bound = true_opt * 0.95
    print(f"\n[Test B] Injecting loose valid lower bound: {tight_bound:.4f}")
    
    _, cost_B, stats_B = solve_with_llm_bound(
        coords, 
        predicted_bound=tight_bound, 
        time_limit=5.0
    )
    
    print(f"Result B: Status={stats_B['status']}, Cost={cost_B}")
    
    if abs(cost_B - true_opt) < 0.001:
        print("✅ PASS: Solver found optimal solution with valid bound.")
    else:
        print("❌ FAIL: Solver failed to find optimal solution with valid bound.")

if __name__ == "__main__":
    test_mip_constraints()
