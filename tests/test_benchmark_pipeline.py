import pytest
import shutil
import os
import numpy as np
from unittest.mock import MagicMock, patch
from src.final_benchmark import run_final_benchmark

def test_run_final_benchmark_mocked(mock_model, mock_tokenizer, tmpdir):
    """Test the full benchmark pipeline with mocked components."""
    
    # Create temp output directory
    output_dir = str(tmpdir.mkdir("benchmark_results"))
    
    # Mock specific calls inside run_final_benchmark
    with patch("src.final_benchmark.load_model", return_value=(mock_model, mock_tokenizer)), \
         patch("src.final_benchmark.solve_tsp_optimal", return_value=([0,1,0], 10.0)), \
         patch("src.final_benchmark.solve_tsp", return_value=([0,1,0], 10.0, {"solve_time": 0.1, "nodes_explored": 5, "status": "OPTIMAL"})), \
         patch("src.final_benchmark.solve_with_llm_bound", return_value=([0,1,0], 10.0, {"solve_time": 0.05, "nodes_explored": 3, "status": "OPTIMAL"})), \
         patch("src.final_benchmark.predict_dual_bound", return_value=9.5), \
         patch("src.final_benchmark.generate_tsp_instance", return_value=np.array([[0,0], [1,1]])):

        summary = run_final_benchmark(
            base_model_path="dummy",
            trained_model_path="dummy",
            n_instances=2,
            n_cities=5,
            seed=42,
            time_limit=1.0, # Fast
            output_dir=output_dir
        )
        
        # Verify summary structure
        assert "n_instances" in summary
        assert summary["n_instances"] == 2
        assert "vanilla" in summary
        assert "base_model" in summary
        assert "trained_model" in summary
        
        # Verify metrics calculation
        # Base model: speedup should be approx 2x (0.1 / 0.05)
        assert summary["base_model"]["valid_predictions"] == 2
        assert summary["base_model"]["avg_speedup_vs_vanilla"] > 1.0
        
        # Verify files created
        assert os.path.exists(os.path.join(output_dir, "summary.json"))
        assert os.path.exists(os.path.join(output_dir, "detailed_results.csv"))
        assert os.path.exists(os.path.join(output_dir, "benchmark_report.txt"))

def test_fallback_logic_mocked(mock_model, mock_tokenizer, tmpdir):
    """Test that invalid bounds trigger fallback or appropriate handling."""
    output_dir = str(tmpdir.mkdir("benchmark_fallback"))
    
    # Mock prediction to be INVALID (overestimate) -> Bound = 15.0 vs Optimal = 10.0
    with patch("src.final_benchmark.load_model", return_value=(mock_model, mock_tokenizer)), \
         patch("src.final_benchmark.solve_tsp_optimal", return_value=([0,1,0], 10.0)), \
         patch("src.final_benchmark.solve_tsp", return_value=([0,1,0], 10.0, {"solve_time": 0.1, "nodes_explored": 5, "status": "OPTIMAL"})), \
         patch("src.final_benchmark.solve_with_llm_bound", return_value=([0,1,0], 10.0, {"solve_time": 0.2, "nodes_explored": 10, "status": "OPTIMAL"})), \
         patch("src.final_benchmark.predict_dual_bound", return_value=15.0), \
         patch("src.final_benchmark.generate_tsp_instance", return_value=np.array([[0,0], [1,1]])):

        summary = run_final_benchmark(
            base_model_path="dummy",
            trained_model_path="dummy",
            n_instances=1,
            n_cities=5,
            seed=42,
            time_limit=1.0,
            output_dir=output_dir
        )
        
        # Bound 15.0 > Optimal 10.0 -> Invalid
        # Logic in benchmark: 
        #   base_valid = base_bound <= optimal_cost
        # So valid_predictions should be 0
        assert summary["base_model"]["valid_predictions"] == 0
        assert summary["base_model"]["invalid_predictions"] == 1
