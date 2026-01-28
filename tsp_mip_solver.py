"""
TSP MIP Solver with LLM-Predicted Dual Bounds

Implements a MIP formulation for TSP using python-mip, with the ability
to integrate LLM-predicted lower bounds to accelerate the search.
"""

import numpy as np
import re
import io
import sys
from mip import Model, xsum, BINARY, INTEGER, minimize, OptimizationStatus
import time


def _parse_nodes_from_cbc_output(output: str) -> int:
    """Parse number of nodes explored from CBC solver output."""
    # Remove line breaks that CBC adds for terminal wrapping
    # CBC wraps at ~80 chars, so join lines that don't start with typical CBC prefixes
    clean_output = re.sub(r'\n(?![A-Z])', '', output)
    
    # Look for the "Search completed" line which has the final node count
    # Pattern: "Search completed - ... took X iterations and Y nodes"
    match = re.search(r'Search completed.*?and\s+(\d+)\s+nodes', clean_output)
    if match:
        return int(match.group(1))
    
    # Fallback: look for any "and X nodes" pattern (take the last one)
    matches = re.findall(r'and\s+(\d+)\s+nodes', clean_output)
    if matches:
        return int(matches[-1])  # Last match is typically the final count
    
    # Alternative pattern: "X nodes (Y.YY seconds)"
    match = re.search(r'(\d+)\s+nodes\s*\(', clean_output)
    if match:
        return int(match.group(1))
    
    return 0


def _solve_with_node_tracking(model, time_limit: float) -> tuple[OptimizationStatus, int, str]:
    """
    Solve model and capture node count from CBC output.
    
    Uses file descriptor level redirection to capture CBC's C-level output.
    
    Returns:
        tuple: (status, nodes_explored, raw_output)
    """
    import os
    import tempfile
    
    # Enable verbose output temporarily
    old_verbose = model.verbose
    model.verbose = 1
    
    # Capture stdout at the file descriptor level (for C-level output)
    # Save old stdout fd
    stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    
    # Create temp file for capture
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp:
        tmp_path = tmp.name
    
    try:
        # Redirect stdout fd to temp file
        with open(tmp_path, 'w') as tmp_file:
            os.dup2(tmp_file.fileno(), stdout_fd)
            
            # Solve
            status = model.optimize(max_seconds=time_limit)
            
            # Flush to ensure all output is written
            sys.stdout.flush()
        
        # Restore stdout
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)
        
        # Read captured output
        with open(tmp_path, 'r') as f:
            output = f.read()
            
    finally:
        model.verbose = old_verbose
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    nodes = _parse_nodes_from_cbc_output(output)
    return status, nodes, output


def build_tsp_model(coords: np.ndarray, name: str = "TSP") -> tuple[Model, list, list]:
    """
    Build a MIP model for TSP using Miller-Tucker-Zemlin (MTZ) formulation.
    
    Args:
        coords: 2D numpy array of city coordinates (n x 2)
        name: Model name
        
    Returns:
        tuple: (model, x_vars, distance_matrix)
    """
    n = len(coords)
    
    # Compute distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    # Create model
    model = Model(name)
    model.verbose = 0  # Suppress solver output
    
    # Decision variables: x[i][j] = 1 if edge (i,j) is in tour
    x = [[model.add_var(var_type=BINARY, name=f"x_{i}_{j}") 
          for j in range(n)] for i in range(n)]
    
    # MTZ variables for subtour elimination: u[i] = position of city i in tour
    u = [model.add_var(var_type=INTEGER, lb=1, ub=n, name=f"u_{i}") 
         for i in range(n)]
    
    # Objective: minimize total distance
    model.objective = minimize(
        xsum(dist[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)
    )
    
    # Constraints
    # 1. Each city must be left exactly once
    for i in range(n):
        model += xsum(x[i][j] for j in range(n) if j != i) == 1
    
    # 2. Each city must be entered exactly once
    for j in range(n):
        model += xsum(x[i][j] for i in range(n) if i != j) == 1
    
    # 3. MTZ subtour elimination (for cities 1 to n-1)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model += u[i] - u[j] + n * x[i][j] <= n - 1
    
    return model, x, dist


def solve_tsp(
    coords: np.ndarray,
    time_limit: float = 60.0,
    gap_tolerance: float = 0.01
) -> tuple[list[int], float, dict]:
    """
    Solve TSP using MIP formulation.
    
    Args:
        coords: City coordinates
        time_limit: Maximum solve time in seconds
        gap_tolerance: Acceptable optimality gap (0.01 = 1%)
        
    Returns:
        tuple: (tour, tour_cost, stats)
    """
    n = len(coords)
    model, x, dist = build_tsp_model(coords)
    
    # Set optimality gap tolerance
    model.max_mip_gap = gap_tolerance
    
    start_time = time.time()
    status, nodes_explored, _ = _solve_with_node_tracking(model, time_limit)
    solve_time = time.time() - start_time
    
    tour = []
    tour_cost = float('inf')
    
    if status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
        # Extract tour from solution
        tour_cost = model.objective_value
        
        # Build tour by following edges
        tour = [0]
        current = 0
        while len(tour) < n:
            for j in range(n):
                if j != current and x[current][j].x > 0.5:
                    tour.append(j)
                    current = j
                    break
        tour.append(0)  # Return to start
    
    stats = {
        "status": str(status),
        "solve_time": solve_time,
        "nodes_explored": nodes_explored,
        "gap": model.gap if hasattr(model, 'gap') and model.gap is not None else 0.0
    }
    
    return tour, tour_cost, stats


def solve_with_llm_bound(
    coords: np.ndarray,
    predicted_bound: float,
    time_limit: float = 60.0,
    gap_tolerance: float = 0.01,
    use_bound_as_constraint: bool = True
) -> tuple[list[int], float, dict]:
    """
    Solve TSP with an LLM-predicted lower bound to accelerate search.
    
    The predicted bound can be used in two ways:
    1. Add as objective constraint (prunes search space)
    2. Use for early termination (if solution is within gap of bound)
    
    Args:
        coords: City coordinates
        predicted_bound: LLM-predicted lower bound on tour cost
        time_limit: Maximum solve time in seconds
        gap_tolerance: Acceptable optimality gap
        use_bound_as_constraint: If True, add objective >= bound constraint
        
    Returns:
        tuple: (tour, tour_cost, stats)
    """
    n = len(coords)
    model, x, dist = build_tsp_model(coords, name="TSP_LLM_Bound")
    
    # Store original objective expression for constraint
    obj_expr = xsum(dist[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)
    
    if use_bound_as_constraint and predicted_bound > 0:
        # Add lower bound constraint on objective
        # This tells the solver: "don't explore solutions cheaper than this"
        # If the bound is valid (underestimate), this helps prune
        # If overestimate, this may cut off the optimal solution!
        model += obj_expr >= predicted_bound * 0.95  # Small buffer for safety
    
    # Set optimality gap tolerance
    model.max_mip_gap = gap_tolerance
    
    start_time = time.time()
    status, nodes_explored, _ = _solve_with_node_tracking(model, time_limit)
    solve_time = time.time() - start_time
    
    tour = []
    tour_cost = float('inf')
    
    if status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
        tour_cost = model.objective_value
        
        # Extract tour
        tour = [0]
        current = 0
        while len(tour) < n:
            for j in range(n):
                if j != current and x[current][j].x > 0.5:
                    tour.append(j)
                    current = j
                    break
        tour.append(0)
    elif status == OptimizationStatus.INFEASIBLE:
        # Bound was too tight (overestimate), fallback to unconstrained solve
        return solve_tsp(coords, time_limit, gap_tolerance)
    
    stats = {
        "status": str(status),
        "solve_time": solve_time,
        "nodes_explored": nodes_explored,
        "predicted_bound": predicted_bound,
        "used_bound_constraint": use_bound_as_constraint,
        "gap": model.gap if hasattr(model, 'gap') and model.gap is not None else 0.0
    }
    
    return tour, tour_cost, stats


def compare_with_baseline(
    coords: np.ndarray,
    predicted_bound: float | None = None,
    time_limit: float = 60.0
) -> dict:
    """
    Compare solving with and without LLM-predicted bound.
    
    Args:
        coords: City coordinates
        predicted_bound: Optional LLM bound (if None, only runs baseline)
        time_limit: Maximum solve time per run
        
    Returns:
        dict with comparison metrics including nodes explored
    """
    # Baseline: solve without bound
    tour_base, cost_base, stats_base = solve_tsp(coords, time_limit)
    
    results = {
        "n_cities": len(coords),
        "optimal_cost": cost_base,
        "baseline_time": stats_base["solve_time"],
        "baseline_nodes": stats_base["nodes_explored"],
        "baseline_status": stats_base["status"]
    }
    
    if predicted_bound is not None:
        # With LLM bound
        tour_llm, cost_llm, stats_llm = solve_with_llm_bound(
            coords, predicted_bound, time_limit
        )
        
        baseline_nodes = stats_base["nodes_explored"]
        llm_nodes = stats_llm["nodes_explored"]
        
        results.update({
            "predicted_bound": predicted_bound,
            "bound_gap": (cost_base - predicted_bound) / cost_base * 100 if cost_base else 0,
            "llm_time": stats_llm["solve_time"],
            "llm_nodes": llm_nodes,
            "llm_status": stats_llm["status"],
            "speedup": stats_base["solve_time"] / stats_llm["solve_time"] if stats_llm["solve_time"] > 0 else 1.0,
            "node_reduction": (baseline_nodes - llm_nodes) / baseline_nodes * 100 if baseline_nodes > 0 else 0.0,
            "solutions_match": abs(cost_base - cost_llm) < 0.001 if cost_llm != float('inf') else False
        })
    
    return results


# Standalone test
if __name__ == "__main__":
    from tsp_utils import generate_tsp_instance, solve_tsp_optimal
    from tsp_dual_utils import calculate_mst_bound
    
    print("=== MIP Solver Test ===\n")
    
    # Generate test instance
    np.random.seed(42)
    coords = generate_tsp_instance(n_cities=8, seed=42)
    
    # Get reference solution
    _, ortools_cost = solve_tsp_optimal(coords)
    print(f"OR-Tools optimal: {ortools_cost:.4f}")
    
    # Get MST bound (simple heuristic)
    mst_bound = calculate_mst_bound(coords)
    print(f"MST bound: {mst_bound:.4f}")
    
    # Solve with MIP
    tour, cost, stats = solve_tsp(coords)
    print(f"MIP solution: {cost:.4f}")
    print(f"Solve time: {stats['solve_time']:.3f}s")
    print(f"Nodes explored: {stats['nodes_explored']}")
    print(f"Status: {stats['status']}")
    
    # Compare with MST bound
    print("\n=== With MST Bound ===")
    results = compare_with_baseline(coords, predicted_bound=mst_bound)
    print(f"Baseline time: {results['baseline_time']:.3f}s, nodes: {results['baseline_nodes']}")
    print(f"With bound time: {results['llm_time']:.3f}s, nodes: {results['llm_nodes']}")
    print(f"Speedup: {results['speedup']:.2f}x")
    print(f"Node reduction: {results['node_reduction']:.1f}%")
    print(f"Solutions match: {results['solutions_match']}")

