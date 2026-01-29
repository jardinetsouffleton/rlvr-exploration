#!/usr/bin/env python3
"""
End-to-End TSP Dual-Bound Training and Benchmarking Pipeline

This script:
1. Configures the training for TSP dual-bound prediction
2. Trains the model using GRPO
3. Runs final 3-way MIP benchmark (vanilla vs base vs trained)

Usage:
    python run_tsp_dual_pipeline.py [--steps 100] [--n_cities 10] [--benchmark_instances 50]
"""

import argparse
import os
import sys
import json
from datetime import datetime


def update_config(
    n_cities_min: int = 5,
    n_cities_max: int = 10,
    steps: int = 100,
    output_dir: str = None,
    data_dir: str = None
):
    """Update config.py with TSP dual parameters."""
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./tsp_dual_output_{timestamp}"
    
    # Read current config
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Update relevant settings
    import re
    
    # Set PROBLEM_TYPE to tsp_dual
    config_content = re.sub(
        r'PROBLEM_TYPE\s*=\s*["\'][^"\']*["\']',
        'PROBLEM_TYPE = "tsp_dual"',
        config_content
    )
    
    # Update steps
    config_content = re.sub(
        r'STEPS\s*=\s*\d+',
        f'STEPS = {steps}',
        config_content
    )
    
    # Update city range
    config_content = re.sub(
        r'TSP_DUAL_MIN_CITIES\s*=\s*\d+',
        f'TSP_DUAL_MIN_CITIES = {n_cities_min}',
        config_content
    )
    config_content = re.sub(
        r'TSP_DUAL_MAX_CITIES\s*=\s*\d+',
        f'TSP_DUAL_MAX_CITIES = {n_cities_max}',
        config_content
    )
    
    config_content = re.sub(
        r'OUTPUT_DIR\s*=\s*["\'][^"\']*["\']',
        f'OUTPUT_DIR = "{output_dir}"',
        config_content
    )

    # Update data dir if provided
    if data_dir:
        # Check if TSP_DATA_DIR exists in config, if not append it
        if 'TSP_DATA_DIR =' in config_content:
             config_content = re.sub(
                r'TSP_DATA_DIR\s*=\s*.*',
                f'TSP_DATA_DIR = "{data_dir}"',
                config_content
            )
        else:
             config_content += f'\nTSP_DATA_DIR = "{data_dir}"\n'
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end TSP dual-bound training and benchmarking"
    )
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps (default: 100)")
    parser.add_argument("--n_cities_min", type=int, default=5,
                        help="Minimum cities during training (default: 5)")
    parser.add_argument("--n_cities_max", type=int, default=10,
                        help="Maximum cities during training (default: 10)")
    parser.add_argument("--benchmark_instances", type=int, default=50,
                        help="Number of benchmark instances (default: 50)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only run benchmark (requires --trained_model)")
    parser.add_argument("--trained_model", type=str, default=None,
                        help="Path to pre-trained model (for --skip_training)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Start directory for persistent data (default: uses config default)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TSP DUAL-BOUND TRAINING AND BENCHMARKING PIPELINE")
    print("=" * 60)
    print()
    
    # Step 1: Configure
    print("[1/3] Configuring training parameters...")
    output_dir = update_config(
        steps=args.steps,
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )
    print(f"  - Problem: TSP Dual Bound Prediction")
    print(f"  - Cities: {args.n_cities_min}-{args.n_cities_max}")
    print(f"  - Training steps: {args.steps}")
    print(f"  - Output: {output_dir}")
    print()
    
    if args.skip_training:
        if not args.trained_model:
            print("ERROR: --skip_training requires --trained_model")
            sys.exit(1)
        trained_model_path = args.trained_model
        print("[2/3] Skipping training (using provided model)")
    else:
        # Step 2: Train
        print("[2/3] Starting training...")
        print("-" * 40)
        
        # Import and run training
        from src.train_grpo import main as train_main
        train_main()
        
        trained_model_path = os.path.join(output_dir, "final")
        print("-" * 40)
        print(f"  Training complete. Model saved to: {trained_model_path}")
    
    print()
    
    # Step 3: Final benchmark (if not already run by train_grpo)
    # The train_grpo already runs the benchmark, but let's ensure it's done
    benchmark_dir = os.path.join(output_dir, "final_mip_benchmark")
    
    if os.path.exists(os.path.join(benchmark_dir, "summary.json")):
        print("[3/3] Final benchmark already completed during training")
    else:
        print(f"[3/3] Running final 3-way MIP benchmark ({args.benchmark_instances} instances)...")
        print("-" * 40)
        
        from src.final_benchmark import run_final_benchmark
        from src.config import MODEL_ID, TSP_DATA_DIR
        
        # Use middle of training range for benchmark
        benchmark_cities = (args.n_cities_min + args.n_cities_max) // 2
        
        summary = run_final_benchmark(
            base_model_path=MODEL_ID,
            trained_model_path=trained_model_path,
            n_instances=args.benchmark_instances,
            n_cities=benchmark_cities,
            seed=42,
            time_limit=60.0,
            output_dir=benchmark_dir,
            data_dir=TSP_DATA_DIR
        )
        
        print("-" * 40)
    
    # Print final summary
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print()
    print("Output files:")
    print(f"  📁 {output_dir}/")
    print(f"     ├── final/                          # Trained model")
    print(f"     ├── benchmark_results.csv           # Training metrics")
    print(f"     ├── tsp_dual_training_log.csv       # Detailed reward log")
    print(f"     ├── final_mip_benchmark_Xcities/    # Benchmark per size")
    print(f"     │   ├── detailed_results.csv        # Per-instance (50)")
    print(f"     │   ├── summary.json                # Aggregated metrics")
    print(f"     │   └── benchmark_report.txt        # Human-readable")
    print()
    
    # Find and display available benchmark results
    import glob
    benchmark_dirs = sorted(glob.glob(os.path.join(output_dir, "final_mip_benchmark_*")))
    
    if benchmark_dirs:
        print("Benchmark Results:")
        for bd in benchmark_dirs:
            summary_path = os.path.join(bd, "summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        s = json.load(f)
                    
                    dirname = os.path.basename(bd)
                    print(f"\n  {dirname}:")
                    print(f"    Vanilla:  {s['vanilla']['avg_time']:.3f}s, {s['vanilla']['avg_nodes']:.0f} nodes")
                    print(f"    Base:     {s['base_model']['valid_rate']:.1f}% valid, {s['base_model']['avg_speedup_vs_vanilla']:.2f}x speedup")
                    print(f"    Trained:  {s['trained_model']['valid_rate']:.1f}% valid, {s['trained_model']['avg_speedup_vs_vanilla']:.2f}x speedup")
                except Exception as e:
                    print(f"    Error reading stats for {bd}")

        
        print()
        print("View detailed reports:")
        for bd in benchmark_dirs:
            print(f"  cat {os.path.join(bd, 'benchmark_report.txt')}")


if __name__ == "__main__":
    main()
