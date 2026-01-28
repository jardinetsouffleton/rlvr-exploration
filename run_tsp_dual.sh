#!/bin/bash
#
# TSP Dual-Bound Training and Benchmarking Pipeline
#
# Trains an LLM to predict dual bounds for TSP and benchmarks the
# improvement in MIP solving with the trained model as an oracle.
#
# Usage:
#   ./run_tsp_dual.sh                  # Default settings
#   ./run_tsp_dual.sh --steps 200      # Custom training steps
#   ./run_tsp_dual.sh --help           # Show all options
#

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default parameters
STEPS=500
N_CITIES_MIN=10
N_CITIES_MAX=20
BENCHMARK_INSTANCES=50
DATA_DIR=".cache/instances/tsp_dual"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --n_cities_min|--min)
            N_CITIES_MIN="$2"
            shift 2
            ;;
        --n_cities_max|--max)
            N_CITIES_MAX="$2"
            shift 2
            ;;
        --benchmark_instances|--instances)
            BENCHMARK_INSTANCES="$2"
            shift 2
            ;;
        --output_dir|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_dir|--data)
            DATA_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "TSP Dual-Bound Training Pipeline"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --steps N              Training steps (default: 100)"
            echo "  --min N                Minimum cities (default: 5)"
            echo "  --max N                Maximum cities (default: 10)"
            echo "  --instances N          Benchmark instances (default: 50)"
            echo "  --output_dir DIR       Output directory"
            echo "  --help                 Show this help"
            echo ""
            echo "Example:"
            echo "  $0 --steps 200 --min 8 --max 15 --output_dir ./my_run"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set HuggingFace cache directories (matching run_safe.sh and train_grpo.py)
HF_CACHE_DIR="/mnt/shared/boileo/hf_cache"
export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Ensure cache directory exists
mkdir -p "${HF_HOME}"

echo "============================================================"
echo "TSP DUAL-BOUND TRAINING AND BENCHMARKING PIPELINE"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Training steps:      $STEPS"
echo "  City range:          $N_CITIES_MIN - $N_CITIES_MAX"
echo "  Benchmark instances: $BENCHMARK_INSTANCES"
echo ""

# Run the pipeline
CMD="python run_tsp_dual_pipeline.py \
    --steps $STEPS \
    --n_cities_min $N_CITIES_MIN \
    --n_cities_max $N_CITIES_MAX \
    --benchmark_instances $BENCHMARK_INSTANCES"

if [ ! -z "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

if [ ! -z "$DATA_DIR" ]; then
    CMD="$CMD --data_dir $DATA_DIR"
fi

$CMD

echo ""
echo "Done!"
