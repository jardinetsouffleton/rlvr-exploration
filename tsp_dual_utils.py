"""
TSP Dual Bound Utilities

This module provides utilities for the TSP dual-bound prediction paradigm,
where the LLM predicts a lower bound on the optimal tour cost instead of
directly solving for a tour.
"""

import re
import numpy as np
from tsp_utils import generate_tsp_instance, solve_tsp_optimal


def get_tsp_dual_prompt(n_cities: int, coords: np.ndarray, mode: str = "cot") -> str:
    """
    Generates a prompt asking the LLM to predict a lower bound on the optimal TSP tour cost.
    
    Args:
        n_cities: Number of cities in the TSP instance
        coords: 2D numpy array of city coordinates (shape: n_cities x 2)
        mode: Prompt mode - "cot" for chain-of-thought, "direct" for direct answer
        
    Returns:
        Formatted prompt string
    """
    # Calculate and format distance matrix
    dist_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    
    # Format distance matrix as a clean list of lists for better parsing
    matrix_str = "Distance Matrix:\n[\n"
    for i in range(n_cities):
        # Format each row: [0.00, 3.45, ...]
        row_values = [f"{x:.2f}" for x in dist_matrix[i]]
        matrix_str += "  [" + ", ".join(row_values) + "],\n"
    matrix_str += "]"
    
    base = f"""You are given a Traveling Salesperson Problem (TSP) with {n_cities} cities.
The goal is to find the shortest tour that visits all cities exactly once and returns to the starting city.

{matrix_str}

Your task is to predict a LOWER BOUND on the optimal tour length. 
A lower bound is a value that is guaranteed to be less than or equal to the actual optimal tour length.

Think about:
- The minimum spanning tree (MST) provides a lower bound
- The sum of the two smallest edges from each city, divided by 2, provides a bound
- Consider distances between cities carefully

Provide your predicted lower bound as a single decimal number."""

    if mode == "cot":
        return base + """

First, analyze the problem and reason about the lower bound inside <thought> tags.
Then, output your predicted lower bound (a single number) inside <answer> tags.
Example: <answer>4.25</answer>"""
    elif mode == "direct":
        return base + """

Output your predicted lower bound as a single decimal number inside <answer> tags.
Example: <answer>4.25</answer>"""
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")


def parse_dual_bound(output_text: str) -> float | None:
    """
    Extracts a numeric lower bound prediction from model output.
    
    Args:
        output_text: The model's response text
        
    Returns:
        The predicted bound as a float, or None if parsing fails
    """
    # Try to extract from <answer> tags first
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
    else:
        # Fallback: try to find last line with a number
        content = output_text
    
    # Extract all floating point numbers
    numbers = re.findall(r'[-+]?\d*\.?\d+', content)
    
    if not numbers:
        return None
    
    try:
        # Take the last number found (most likely to be the final answer)
        return float(numbers[-1])
    except (ValueError, IndexError):
        return None


def calculate_dual_reward(predicted: float, optimal: float, 
                          overestimate_mult: float = 2.0) -> float:
    """
    Calculates the reward for a dual bound prediction.
    
    The reward penalizes the gap between predicted and optimal, with
    asymmetric penalties:
    - Underestimation (predicted < optimal): -gap
    - Overestimation (predicted > optimal): -2*gap (or configurable multiplier)
    
    This encourages tight bounds while penalizing invalid (too high) bounds more.
    
    Args:
        predicted: The predicted lower bound
        optimal: The actual optimal tour length
        overestimate_mult: Penalty multiplier for overestimation (default: 2.0)
        
    Returns:
        The reward value (negative, representing penalty)
    """
    if optimal <= 0:
        return -10.0  # Invalid optimal value
    
    gap = abs((predicted - optimal) / optimal)
    
    if predicted > optimal:
        # Overestimation - invalid bound, penalize more heavily
        return -overestimate_mult * gap
    else:
        # Underestimation - valid bound, linear penalty for looseness
        return -gap


def calculate_mst_bound(coords: np.ndarray) -> float:
    """
    Calculates a simple MST-based lower bound for TSP.
    The MST cost is a valid lower bound since any tour is a spanning tree plus one edge.
    
    This can be used as a baseline or for prompt context.
    
    Args:
        coords: 2D numpy array of city coordinates
        
    Returns:
        MST cost as a lower bound
    """
    n = len(coords)
    if n <= 1:
        return 0.0
    
    # Simple Prim's algorithm
    dist_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    
    in_tree = [False] * n
    min_edge = [float('inf')] * n
    min_edge[0] = 0.0
    mst_cost = 0.0
    
    for _ in range(n):
        # Find minimum edge to add
        u = -1
        for v in range(n):
            if not in_tree[v] and (u == -1 or min_edge[v] < min_edge[u]):
                u = v
        
        in_tree[u] = True
        mst_cost += min_edge[u]
        
        # Update minimum edges
        for v in range(n):
            if not in_tree[v] and dist_matrix[u][v] < min_edge[v]:
                min_edge[v] = dist_matrix[u][v]
    
    return mst_cost


def calculate_1tree_bound(coords: np.ndarray) -> float:
    """
    Calculates a 1-tree lower bound for TSP.
    This is generally tighter than the MST bound.
    
    A 1-tree is an MST on cities 1..n-1, plus the two shortest edges from city 0.
    
    Args:
        coords: 2D numpy array of city coordinates
        
    Returns:
        1-tree cost as a lower bound
    """
    n = len(coords)
    if n <= 2:
        return calculate_mst_bound(coords)
    
    dist_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    
    # MST on cities 1..n-1
    remaining_coords = coords[1:]
    mst_cost = calculate_mst_bound(remaining_coords)
    
    # Two shortest edges from city 0
    edges_from_0 = sorted(dist_matrix[0][1:])
    two_shortest = edges_from_0[0] + edges_from_0[1] if len(edges_from_0) >= 2 else sum(edges_from_0)
    
    return mst_cost + two_shortest
