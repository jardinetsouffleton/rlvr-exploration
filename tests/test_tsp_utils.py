import pytest
import numpy as np
from src.tsp_utils import (
    generate_tsp_instance, 
    calculate_tour_length, 
    solve_tsp_nearest_neighbor, 
    parse_model_output
)

def test_generate_tsp_instance():
    """Test instance generation."""
    coords = generate_tsp_instance(n_cities=10, seed=42)
    assert coords.shape == (10, 2)
    assert np.all(coords >= 0) and np.all(coords <= 1)
    
    # Check reproducibility
    coords2 = generate_tsp_instance(n_cities=10, seed=42)
    assert np.allclose(coords, coords2)

def test_calculate_tour_length():
    """Test tour length calculation."""
    # Unit square: (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
    # Length = 1 + 1 + 1 + 1 = 4
    coords = np.array([[0,0], [1,0], [1,1], [0,1]])
    tour = [0, 1, 2, 3, 0]
    length = calculate_tour_length(tour, coords)
    assert np.isclose(length, 4.0)

def test_parse_model_output():
    """Test parsing of model output."""
    output_text = "Here is the tour: <tour>[0, 1, 2, 3, 0]</tour>"
    tour = parse_model_output(output_text, n_cities=4)
    assert tour == [0, 1, 2, 3, 0]
    
    # Test valid fallback (just list)
    output_text = "[0, 1, 2, 0]"
    tour = parse_model_output(output_text, n_cities=3)
    assert tour == [0, 1, 2, 0]
    
    # Test invalid
    output_text = "No tour here"
    tour = parse_model_output(output_text, n_cities=5)
    assert tour is None

def test_solve_tsp_nearest_neighbor():
    """Test NN solver baseline."""
    coords = generate_tsp_instance(n_cities=10, seed=42)
    tour, length = solve_tsp_nearest_neighbor(coords)
    
    assert len(tour) == 11 # 10 cities + return
    assert tour[0] == tour[-1]
    assert length > 0
    assert len(set(tour)) == 10
