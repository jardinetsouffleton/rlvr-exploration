/mnt/home/rlvr-exploration/final_benchmark.py"""
Final Benchmark: Compare MIP solving with different dual-bound oracles.

Compares three approaches:
1. Vanilla MIP (no oracle)
2. Base model as dual-bound oracle
3. Trained model as dual-bound oracle

Outputs a CSV with detailed results and a summary report.
"""

import argparse
import numpy as np
import json
import csv
import os
from datetime import datetime
from tqdm import tqdm

from tsp_utils import generate_tsp_instance, solve_tsp_optimal, save_tsp_instances, load_tsp_instances
from tsp_dual_utils import calculate_mst_bound
from tsp_mip_solver import solve_tsp, solve_with_llm_bound
from tsp_dual_inference import load_model, predict_dual_bound


def run_final_benchmark(
    base_model_path: str,
    trained_model_path: str,
    n_instances: int = 50,
    n_cities: int = 10,
    seed: int = 42,
    time_limit: float = 60.0,
    time_limit: float = 60.0,
    output_dir: str = "final_benchmark",
    data_dir: str = None
):
    """
    Run comprehensive benchmark comparing vanilla, base model, and trained model.
    
    Args:
        base_model_path: Path to base (untrained) model
        trained_model_path: Path to trained model
        n_instances: Number of test instances
        n_cities: Cities per instance
        seed: Random seed
        time_limit: MIP solve time limit per instance
        output_dir: Directory to save results
        data_dir: Directory to save/load persistent instances (default: None, uses output_dir)
        
    Returns:
        dict: Summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Final Benchmark ===")
    print(f"Instances: {n_instances}, Cities: {n_cities}")
    print(f"Time limit: {time_limit}s per instance")
    print()
    
    # Load models
    print("Loading base model...")
    base_model, base_tokenizer = load_model(base_model_path)
    
    print("Loading trained model...")
    trained_model, trained_tokenizer = load_model(trained_model_path)
    
    # Generate fixed test instances
    # Check for existing instances
    instances = None
    if data_dir:
        # Try directory structure specific to n_cities
        city_dir = os.path.join(data_dir, str(n_cities))
        if os.path.exists(city_dir):
            files = sorted([f for f in os.listdir(city_dir) if f.endswith('.json')])
            if files:
                print(f"Loading instances from {city_dir} ({len(files)} files found)...")
                instances = []
                for f in files:
                    try:
                        with open(os.path.join(city_dir, f), 'r') as fp:
                            d = json.load(fp)
                            if "coords" in d:
                                instances.append(np.array(d["coords"]))
                    except:
                        pass
                
                # Limit to n_instances
                if len(instances) > n_instances:
                     instances = instances[:n_instances]
                
        # If not found in directory structure, check for legacy single file
        if not instances:
             filename = f"benchmark_instances_n{n_cities}_i{n_instances}.json"
             instances_path = os.path.join(data_dir, filename)
             instances = load_tsp_instances(instances_path)
             
    else:
        # Backward compatibility / local run
        instances_path = os.path.join(output_dir, "benchmark_instances.json")
        instances = load_tsp_instances(instances_path)
    
    if instances is None:
        print(f"Generating {n_instances} new instances...")
        np.random.seed(seed)
        seeds = np.random.randint(0, 100000, n_instances)
        instances = []
        for i in range(n_instances):
             coords = generate_tsp_instance(n_cities, seed=int(seeds[i]))
             instances.append(coords)
        
        # Save for future use
        save_tsp_instances(instances, instances_path)
    else:
        print(f"Using {len(instances)} pre-generated instances from {instances_path}")
        # Validation
        if len(instances) != n_instances:
            print(f"WARNING: Loaded {len(instances)} instances, but requested {n_instances}. Using loaded instances.")
            # We could slice or fail, but let's just use what we have
    
    # Update loop to use instances list
    
    # Results storage
    results = []
    
    # CSV file for detailed results
    csv_path = os.path.join(output_dir, "detailed_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instance", "N_Cities", "Optimal_Cost",
            "Vanilla_Time", "Vanilla_Nodes", "Vanilla_Status",
            "Base_Bound", "Base_Valid", "Base_Time", "Base_Nodes", "Base_Status",
            "Trained_Bound", "Trained_Valid", "Trained_Time", "Trained_Nodes", "Trained_Status"
        ])
    
    print(f"\nRunning benchmark on {n_instances} instances...")
    
    for i, coords in tqdm(enumerate(instances), total=len(instances)):
        # coords is already the numpy array
        
        # Get optimal cost for reference
        _, optimal_cost = solve_tsp_optimal(coords)
        
        # 1. Vanilla MIP (no oracle)
        _, vanilla_cost, vanilla_stats = solve_tsp(coords, time_limit=time_limit)
        
        # 2. Base model prediction
        base_bound = predict_dual_bound(base_model, base_tokenizer, coords)
        if base_bound is not None and base_bound > 0:
            base_valid = base_bound <= optimal_cost
            _, base_cost, base_stats = solve_with_llm_bound(coords, base_bound, time_limit)
        else:
            base_valid = None
            base_cost = vanilla_cost
            base_stats = {"solve_time": 0, "nodes_explored": 0, "status": "FAILED"}
        
        # 3. Trained model prediction
        trained_bound = predict_dual_bound(trained_model, trained_tokenizer, coords)
        if trained_bound is not None and trained_bound > 0:
            trained_valid = trained_bound <= optimal_cost
            _, trained_cost, trained_stats = solve_with_llm_bound(coords, trained_bound, time_limit)
        else:
            trained_valid = None
            trained_cost = vanilla_cost
            trained_stats = {"solve_time": 0, "nodes_explored": 0, "status": "FAILED"}
        
        # Store result
        result = {
            "instance": i,
            "n_cities": n_cities,
            "optimal_cost": optimal_cost,
            "vanilla_time": vanilla_stats["solve_time"],
            "vanilla_nodes": vanilla_stats["nodes_explored"],
            "vanilla_status": vanilla_stats["status"],
            "base_bound": base_bound,
            "base_valid": base_valid,
            "base_time": base_stats["solve_time"],
            "base_nodes": base_stats["nodes_explored"],
            "base_status": base_stats["status"],
            "trained_bound": trained_bound,
            "trained_valid": trained_valid,
            "trained_time": trained_stats["solve_time"],
            "trained_nodes": trained_stats["nodes_explored"],
            "trained_status": trained_stats["status"],
        }
        results.append(result)
        
        # Append to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                i, n_cities, f"{optimal_cost:.4f}",
                f"{vanilla_stats['solve_time']:.3f}", vanilla_stats['nodes_explored'], vanilla_stats['status'],
                f"{base_bound:.4f}" if base_bound else "FAILED", base_valid, 
                f"{base_stats['solve_time']:.3f}", base_stats['nodes_explored'], base_stats['status'],
                f"{trained_bound:.4f}" if trained_bound else "FAILED", trained_valid,
                f"{trained_stats['solve_time']:.3f}", trained_stats['nodes_explored'], trained_stats['status'],
            ])
    
    # Calculate summary statistics
    summary = calculate_summary(results)
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Generate report
    report_path = os.path.join(output_dir, "benchmark_report.txt")
    generate_report(summary, report_path)
    
    print(f"\n=== Results saved to {output_dir} ===")
    print(f"  - detailed_results.csv")
    print(f"  - summary.json")
    print(f"  - benchmark_report.txt")
    
    # Print summary
    print_summary(summary)
    
    return summary


def calculate_summary(results: list[dict]) -> dict:
    """Calculate summary statistics from results."""
    n = len(results)
    
    # Vanilla stats
    vanilla_times = [r["vanilla_time"] for r in results]
    vanilla_nodes = [r["vanilla_nodes"] for r in results]
    
    # Base model stats (only valid predictions)
    base_valid = [r for r in results if r["base_valid"] is True]
    base_invalid = [r for r in results if r["base_valid"] is False]
    base_failed = [r for r in results if r["base_valid"] is None]
    
    # Trained model stats (only valid predictions)
    trained_valid = [r for r in results if r["trained_valid"] is True]
    trained_invalid = [r for r in results if r["trained_valid"] is False]
    trained_failed = [r for r in results if r["trained_valid"] is None]
    
    # Calculate speedups and node reductions (vs vanilla, only for valid bounds)
    def calc_improvements(subset, all_results):
        if not subset:
            return {"speedup": 0, "node_reduction": 0}
        
        speedups = []
        node_reductions = []
        for r in subset:
            # Find corresponding vanilla result
            vanilla_time = r["vanilla_time"]
            vanilla_nodes = r["vanilla_nodes"]
            
            if vanilla_time > 0:
                speedups.append(vanilla_time / r["base_time"] if "base" in str(subset[0].get("base_valid", "")) else vanilla_time / r["trained_time"])
            if vanilla_nodes > 0:
                node_reductions.append((vanilla_nodes - (r["base_nodes"] if "base" in str(subset[0].get("base_valid", "")) else r["trained_nodes"])) / vanilla_nodes * 100)
        
        return {
            "avg_speedup": np.mean(speedups) if speedups else 0,
            "avg_node_reduction": np.mean(node_reductions) if node_reductions else 0
        }
    
    summary = {
        "n_instances": n,
        "vanilla": {
            "avg_time": np.mean(vanilla_times),
            "std_time": np.std(vanilla_times),
            "avg_nodes": np.mean(vanilla_nodes),
            "std_nodes": np.std(vanilla_nodes),
        },
        "base_model": {
            "valid_predictions": len(base_valid),
            "invalid_predictions": len(base_invalid),
            "failed_predictions": len(base_failed),
            "valid_rate": len(base_valid) / n * 100 if n > 0 else 0,
            "avg_time": np.mean([r["base_time"] for r in base_valid]) if base_valid else 0,
            "avg_nodes": np.mean([r["base_nodes"] for r in base_valid]) if base_valid else 0,
            "avg_speedup_vs_vanilla": np.mean([results[i]["vanilla_time"] / r["base_time"] for i, r in enumerate(results) if r["base_valid"] is True and r["base_time"] > 0]) if base_valid else 0,
            "avg_node_reduction_vs_vanilla": np.mean([(results[i]["vanilla_nodes"] - r["base_nodes"]) / results[i]["vanilla_nodes"] * 100 for i, r in enumerate(results) if r["base_valid"] is True and results[i]["vanilla_nodes"] > 0]) if base_valid else 0,
        },
        "trained_model": {
            "valid_predictions": len(trained_valid),
            "invalid_predictions": len(trained_invalid),
            "failed_predictions": len(trained_failed),
            "valid_rate": len(trained_valid) / n * 100 if n > 0 else 0,
            "avg_time": np.mean([r["trained_time"] for r in trained_valid]) if trained_valid else 0,
            "avg_nodes": np.mean([r["trained_nodes"] for r in trained_valid]) if trained_valid else 0,
            "avg_speedup_vs_vanilla": np.mean([results[i]["vanilla_time"] / r["trained_time"] for i, r in enumerate(results) if r["trained_valid"] is True and r["trained_time"] > 0]) if trained_valid else 0,
            "avg_node_reduction_vs_vanilla": np.mean([(results[i]["vanilla_nodes"] - r["trained_nodes"]) / results[i]["vanilla_nodes"] * 100 for i, r in enumerate(results) if r["trained_valid"] is True and results[i]["vanilla_nodes"] > 0]) if trained_valid else 0,
        }
    }
    
    return summary


def generate_report(summary: dict, output_path: str):
    """Generate human-readable benchmark report."""
    lines = [
        "=" * 60,
        "FINAL BENCHMARK REPORT",
        "=" * 60,
        "",
        f"Total instances: {summary['n_instances']}",
        "",
        "-" * 60,
        "1. VANILLA MIP (No Oracle)",
        "-" * 60,
        f"  Average solve time: {summary['vanilla']['avg_time']:.3f}s (±{summary['vanilla']['std_time']:.3f})",
        f"  Average nodes explored: {summary['vanilla']['avg_nodes']:.1f} (±{summary['vanilla']['std_nodes']:.1f})",
        "",
        "-" * 60,
        "2. BASE MODEL (Untrained) as Oracle",
        "-" * 60,
        f"  Valid bound rate: {summary['base_model']['valid_rate']:.1f}%",
        f"    - Valid predictions: {summary['base_model']['valid_predictions']}",
        f"    - Invalid (overestimate): {summary['base_model']['invalid_predictions']}",
        f"    - Failed to parse: {summary['base_model']['failed_predictions']}",
        f"  Average solve time: {summary['base_model']['avg_time']:.3f}s",
        f"  Average nodes explored: {summary['base_model']['avg_nodes']:.1f}",
        f"  Speedup vs vanilla: {summary['base_model']['avg_speedup_vs_vanilla']:.2f}x",
        f"  Node reduction vs vanilla: {summary['base_model']['avg_node_reduction_vs_vanilla']:.1f}%",
        "",
        "-" * 60,
        "3. TRAINED MODEL as Oracle",
        "-" * 60,
        f"  Valid bound rate: {summary['trained_model']['valid_rate']:.1f}%",
        f"    - Valid predictions: {summary['trained_model']['valid_predictions']}",
        f"    - Invalid (overestimate): {summary['trained_model']['invalid_predictions']}",
        f"    - Failed to parse: {summary['trained_model']['failed_predictions']}",
        f"  Average solve time: {summary['trained_model']['avg_time']:.3f}s",
        f"  Average nodes explored: {summary['trained_model']['avg_nodes']:.1f}",
        f"  Speedup vs vanilla: {summary['trained_model']['avg_speedup_vs_vanilla']:.2f}x",
        f"  Node reduction vs vanilla: {summary['trained_model']['avg_node_reduction_vs_vanilla']:.1f}%",
        "",
        "=" * 60,
        "IMPROVEMENT FROM TRAINING",
        "=" * 60,
        f"  Valid bound rate: {summary['base_model']['valid_rate']:.1f}% → {summary['trained_model']['valid_rate']:.1f}%",
        f"  Speedup improvement: {summary['base_model']['avg_speedup_vs_vanilla']:.2f}x → {summary['trained_model']['avg_speedup_vs_vanilla']:.2f}x",
        f"  Node reduction: {summary['base_model']['avg_node_reduction_vs_vanilla']:.1f}% → {summary['trained_model']['avg_node_reduction_vs_vanilla']:.1f}%",
        "",
    ]
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def print_summary(summary: dict):
    """Print summary to console."""
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"\n{'Method':<25} {'Time':>10} {'Nodes':>10} {'Valid%':>10}")
    print("-" * 55)
    print(f"{'Vanilla MIP':<25} {summary['vanilla']['avg_time']:>10.3f}s {summary['vanilla']['avg_nodes']:>10.0f} {'N/A':>10}")
    print(f"{'Base Model Oracle':<25} {summary['base_model']['avg_time']:>10.3f}s {summary['base_model']['avg_nodes']:>10.0f} {summary['base_model']['valid_rate']:>9.1f}%")
    print(f"{'Trained Model Oracle':<25} {summary['trained_model']['avg_time']:>10.3f}s {summary['trained_model']['avg_nodes']:>10.0f} {summary['trained_model']['valid_rate']:>9.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Final benchmark comparing MIP solving approaches")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base (untrained) model")
    parser.add_argument("--trained_model", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--n_instances", type=int, default=50,
                        help="Number of test instances")
    parser.add_argument("--n_cities", type=int, default=10,
                        help="Number of cities per instance")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--time_limit", type=float, default=60.0,
                        help="MIP solve time limit per instance")
    parser.add_argument("--output_dir", type=str, default="final_benchmark",
                        help="Output directory for results")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory for persistent instances")
    
    args = parser.parse_args()
    
    run_final_benchmark(
        base_model_path=args.base_model,
        trained_model_path=args.trained_model,
        n_instances=args.n_instances,
        n_cities=args.n_cities,
        seed=args.seed,
        time_limit=args.time_limit,
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )


if __name__ == "__main__":
    main()
