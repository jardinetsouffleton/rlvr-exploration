import os

# 1. Environment Configuration (Must be before other imports that use these)
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(WORKSPACE_DIR, ".cache")

# Use HF_HOME from environment if set, otherwise default to local cache
if "HF_HOME" in os.environ:
    HF_HOME = os.environ["HF_HOME"]
else:
    HF_HOME = os.path.join(CACHE_DIR, "huggingface")

from datetime import datetime

# Shared output directory
BASE_OUTPUT_DIR = "./tsp_dual_output_20260127_191133"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, timestamp)

# Set Environment Variables
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets") # Also move datasets to safe location

# Ensure directories exist
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared Data Directory (for persisting generated instances)
TSP_DATA_DIR = ".cache/instances/tsp_dual"
os.makedirs(TSP_DATA_DIR, exist_ok=True)

# 2. Model Configuration
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 

# 3. Training Configuration
LR = 5e-6
GROUP_SIZE = 12
GENERATION_BATCH_SIZE = 12
TARGET_TOTAL_BATCH_SIZE = 64
STEPS = 500
MAX_NEW_TOKENS = 10000
BETA = 0.04

# 4. Problem Configuration
PROBLEM_TYPE = "tsp_dual" # Options: "tsp", "graph_coloring", "sat"

# TSP Config
N_CITIES = 10
REWARD_BASELINE = "nn" # Options: "nn", "optimal"
PROMPT_MODE = "cot" # Options: "cot", "direct"

# TSP Dual Bound Config (for PROBLEM_TYPE = "tsp_dual")
TSP_DUAL_MIN_CITIES = 5       # Min cities for training instances
TSP_DUAL_MAX_CITIES = 10      # Max cities for training instances
TSP_DUAL_FORMAT_REWARD = 1.0  # Reward for correctly formatted output
TSP_DUAL_OVERESTIMATE_MULT = 20.0  # Penalty multiplier for overestimation
TSP_DUAL_OVERESTIMATE_PENALTY = 5.0 # Flat penalty for overestimation (worse than 0.0)


# SAT Config
SAT_VARS = 5  # Default/starting size (used when curriculum disabled)
SAT_CLAUSES = 21  # Default clauses (will be overridden by ratio when curriculum enabled)
SAT_VARS_PER_CLAUSE = 3

# SAT Deterministic Curriculum
SAT_CURRICULUM = False
SAT_MIN_VARS = 5        # Starting problem size
SAT_MAX_VARS = 7     # Maximum problem size
SAT_STEP_INTERVAL = 70  # Increase size every N training steps
SAT_STEP_SIZE = 1       # Variables to add per step
CLAUSE_RATIO = 2.5      # Clauses-per-variable ratio (4.2 is 3-SAT phase transition)

# Benchmark sizes
SAT_BENCHMARK_SIZES = [3, 4, 5, 6, 7, 10, 15, 20]
SAT_BENCHMARK_INSTANCES = 20

