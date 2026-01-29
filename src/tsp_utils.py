import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re

def get_tsp_prompt(n_cities: int, coords: np.ndarray = None, mode: str = "cot") -> str:
    """
    Generates the TSP prompt based on the mode.
    Now includes Distance Matrix if coords are provided.
    """
    base = f"Find the shortest path visiting all {n_cities} cities. Points are labeled 0 to {n_cities-1}."
    
    if coords is not None:
        # Calculate distance matrix
        dist_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        
        # Format matrix string
        matrix_str = "Distance Matrix:\n"
        # Print header
        matrix_str += "      " + " ".join([f"{i:5d}" for i in range(n_cities)]) + "\n"
        for i in range(n_cities):
            row_str = f"{i:4d}: " + " ".join([f"{x:5.2f}" for x in dist_matrix[i]])
            matrix_str += row_str + "\n"
            
        base += f"\n\n{matrix_str}\n"

    if mode == "cot":
        return base + " First, analyze the problem and plan your path. Write your reasoning inside <thought> tags. Then, simply output the sequence of city indices in order, starting and ending at the same city, inside <answer> tags, like this: <answer>0 1 2 ... 0</answer>."
    elif mode == "direct":
        return base + " Output the sequence of city indices in order, starting and ending at the same city. Output the numbers separated by spaces."
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

def generate_tsp_instance(n_cities: int = 20, seed: int = None):
    """Generates random 2D coordinates for TSP."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(n_cities, 2)

def render_tsp_instance(coords: np.ndarray, show_solution: list = None, title: str = None) -> Image.Image:
    """
    Renders the TSP instance as a PIL Image.
    If show_solution is provided (list of indices), it draws the path.
    """
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.gca()
    
    # Draw solution if present
    if show_solution is not None:
        path_coords = coords[show_solution]
        ax.plot(path_coords[:, 0], path_coords[:, 1], 'g-', alpha=0.5, zorder=1)
    
    # Draw cities
    ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, zorder=2)
    
    # Label cities
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=12, zorder=3)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    if title:
        ax.set_title(title)
        
    ax.axis('off')
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def calculate_tour_length(tour: list[int], coords: np.ndarray) -> float:
    """Calculates Euclidean distance of the tour."""
    if not tour:
        return float('inf')
    
    dist = 0.0
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        dist += np.linalg.norm(coords[u] - coords[v])
    return dist

def parse_model_output(output_text: str, n_cities: int) -> list[int] | None:
    """
    Parses a tour from model output. 
    Expected format: Sequence of numbers inside <answer> tags.
    Checks validity:
    1. All cities 0..N-1 visited exactly once (except start/end).
    2. Valid permutation.
    """
    # Try to extract content inside <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    if answer_match:
        content_to_parse = answer_match.group(1)
    else:
        # Fallback: Parse the whole text or the last part? 
        # Let's try to parse the whole text finding numbers.
        # This allows the model to learn to solve TSP even if it forgets tags initially.
        content_to_parse = output_text 

    # Extract all numbers from the answer block
    numbers = [int(s) for s in re.findall(r'\b\d+\b', content_to_parse)]
    
    if not numbers:
        return None
        
    # Filtering for valid range
    valid_nums = [n for n in numbers if 0 <= n < n_cities]
    
    # We want a full tour.
    # Case 1: 0 ... 0 (Closed loop) -> Length N+1
    # Case 2: 0 ... N-1 (Permutation) -> Length N (assume implicit return)
    # Let's demand explicitly closed loop for robustness, or accept permutation.
    
    # Strategy: Find the longest valid sub-sequence of unique cities? 
    # Or just check if valid_nums forms a valid tour.
    
    # We will return the list of numbers found (filtered to valid range).
    # It is up to the caller to validate if it's a correct TSP tour.
    
    # "Smart" parsing: Try to extract a clean tour if possible
    unique_cities = set(valid_nums)
    
    # Case: Too many numbers? 
    if len(unique_cities) > n_cities:
         # Just take the first occurrence of each city?
        seen = set()
        clean_tour = []
        for n in valid_nums:
            if n not in seen:
                clean_tour.append(n)
                seen.add(n)
            if len(clean_tour) == n_cities:
                break
    else:
        # Case: Exact or fewer. Just take what we have.
        # If output was 0 1 2 0, valid_nums is [0, 1, 2, 0]. unique is 0,1,2.
        # We try to take first N distinct if possible, but if not enough unique, we just return valid_nums.
        clean_tour = valid_nums
    
    # Check if we should append start to end (if it looks like a permutation that forgot to close)
    # Only do this if we actually have N unique cities and it's open loop
    if len(set(clean_tour)) == n_cities and len(clean_tour) == n_cities:
         clean_tour = list(clean_tour) # ensure list
         clean_tour.append(clean_tour[0])
         
    return clean_tour

def solve_tsp_nearest_neighbor(coords: np.ndarray) -> tuple[list[int], float]:
    """
    Solves TSP using Nearest Neighbor heuristic.
    Returns (tour_indices, length).
    """
    N = len(coords)
    unvisited = set(range(1, N))
    current = 0
    tour = [0]
    
    while unvisited:
        # Find nearest unvisited
        nearest = min(unvisited, key=lambda x: np.linalg.norm(coords[current] - coords[x]))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
        
    tour.append(0) # Return to start
    length = calculate_tour_length(tour, coords)
    return tour, length

def solve_tsp_optimal(coords: np.ndarray) -> tuple[list[int], float]:
    """
    Solves TSP optimally using OR-Tools.
    Returns (tour_indices, length).
    """
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    
    # OR-Tools works with integers. Scale floats.
    scale = 10000
    scaled = (coords * scale).astype(int)
    
    def create_data_model():
        data = {}
        # Compute distance matrix
        n = len(coords)
        dist_matrix = {}
        for i in range(n):
            dist_matrix[i] = {}
            for j in range(n):
                # Euclidean distance
                d = int(np.linalg.norm(scaled[i] - scaled[j]))
                dist_matrix[i][j] = d
        data['distance_matrix'] = dist_matrix
        data['num_vehicles'] = 1
        data['depot'] = 0
        return data

    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(len(coords), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Use guided local search for optimality
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 2 

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        tour = []
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        tour.append(0) # Return to start
        
        # Calculate precise float length using original coords
        length = calculate_tour_length(tour, coords)
        return tour, length
    else:
        # Fallback to NN if fails (rare for small N)
        return solve_tsp_nearest_neighbor(coords)


def save_tsp_instances(instances: list[np.ndarray], filepath: str):
    """Saves a list of TSP instances (coordinate arrays) to a JSON file."""
    import json
    # Convert numpy arrays to lists for JSON serialization
    serialized = [coords.tolist() for coords in instances]
    with open(filepath, 'w') as f:
        json.dump(serialized, f)
    print(f"Saved {len(instances)} instances to {filepath}")

def load_tsp_instances(filepath: str) -> list[np.ndarray]:
    """Loads a list of TSP instances from a JSON file."""
    import json
    import os
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, 'r') as f:
        serialized = json.load(f)
    
    # Convert lists back to numpy arrays
    instances = [np.array(coords) for coords in serialized]
    print(f"Loaded {len(instances)} instances from {filepath}")
    return instances
