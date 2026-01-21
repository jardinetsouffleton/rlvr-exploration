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
BASE_OUTPUT_DIR = "/mnt/shared/boileo/rlvr_tests"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, timestamp)

# Set Environment Variables
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets") # Also move datasets to safe location

# Ensure directories exist
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Model Configuration
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 

# 3. Training Configuration
LR = 1e-5
GROUP_SIZE = 12
GENERATION_BATCH_SIZE = 4
TARGET_TOTAL_BATCH_SIZE = 64
STEPS = 1500
MAX_NEW_TOKENS = 4096
BETA = 0.04

# 4. Problem Configuration
PROBLEM_TYPE = "sat" # Options: "tsp", "graph_coloring", "sat"

# TSP Config
N_CITIES = 10
REWARD_BASELINE = "nn" # Options: "nn", "optimal"
PROMPT_MODE = "cot" # Options: "cot", "direct"

# Graph Coloring Config
GRAPH_NODES = 10
GRAPH_EDGE_PROB = 0.5
MIN_COLORS = 3

# SAT Config
SAT_VARS = 5
SAT_CLAUSES = 10
SAT_VARS_PER_CLAUSE = 3

# SAT Curriculum
SAT_CURRICULUM = True
SAT_START_VARS = 5
SAT_END_VARS = 10
