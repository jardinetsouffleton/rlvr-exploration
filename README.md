# RLVR for Combinatorial Optimization (TSP & SAT)

This project implements a Reinforcement Learning with Verifiable Rewards (RLVR) loop for solving combinatorial optimization problems—specifically **Traveling Salesperson Problem (TSP)** and **Boolean Satisfiability (SAT)**—using the **GRPO (Group Relative Policy Optimization)** algorithm. It supports both vision-language models (e.g., Qwen2.5-VL) and text-only models (e.g., Qwen2.5-Instruct).

## 🚀 Key Features
- **Multi-Problem Support**: Train on TSP (visual/text) or SAT (text).
- **GRPO Training**: Uses Group Relative Policy Optimization for efficient reinforcement learning.
- **Micro-Batching**: Supports large `GROUP_SIZE` (e.g., 16+) on limited VRAM by breaking generation into smaller micro-batches (`GENERATION_BATCH_SIZE`).
- **Verifiable Rewards**: 
  - **TSP**: Rewards valid tours based on optimality gap vs. Neural/Optimal baselines.
  - **SAT**: Rewards valid variable assignments based on the number of satisfied clauses.
- **Curriculum Learning**: Linearly scales problem difficulty (e.g., number of variables in SAT) over training steps.

## 🛠️ Configuration
The project is fully configurable via **`config.py`**. You should edit this file to change experiments.

### Key Configuration Variables

| Category | Variable | Description |
| :--- | :--- | :--- |
| **Problem** | `PROBLEM_TYPE` | `"sat"` or `"tsp"`. Determines the problem domain. |
| **Model** | `MODEL_ID` | HF Model ID (e.g. `"Qwen/Qwen2.5-1.5B-Instruct"`). |
| **Training** | `GROUP_SIZE` | Total number of outputs generated per prompt (GRPO requires >1). |
| | `GENERATION_BATCH_SIZE` | Max batch size for generation to fit in VRAM. If `GROUP_SIZE` > this, it micro-batches. |
| | `STEPS` | Total training steps. |
| | `PROMPT_MODE` | `"cot"` (Chain of Thought) or `"direct"` (Output answer only). |
| **SAT Config** | `SAT_CURRICULUM` | `True` to enable difficulty scaling. |
| | `SAT_START_VARS` | Number of variables at step 0. |
| | `SAT_END_VARS` | Number of variables at final step. |
| **TSP Config** | `N_CITIES` | Number of cities for TSP instances. |

## 🏃 Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configure Experiment
Open `config.py` and set your desired parameters.
*Example: To train for SAT with curriculum:*
```python
PROBLEM_TYPE = "sat"
SAT_CURRICULUM = True
SAT_START_VARS = 5
SAT_END_VARS = 15
GROUP_SIZE = 16
GENERATION_BATCH_SIZE = 4 # Fits in VRAM by running 4 consecutive batches of 4
```

### 3. Run Training (Recommended)
The central entry point is the **`run_safe.sh`** script. It handles virtual environment setup, dependency installation, and environment variable configuration (e.g., HF cache) before launching the training.

```bash
./run_safe.sh
```

This script will:
1. Create/Activate a virtual environment in `.venv`.
2. Install dependencies from `requirements.txt`.
3. Set `HF_HOME` to a shared cache directory to avoid quota issues.
4. Execute `train_rlvr.py` with the settings from `config.py`.

*Note: You can still run `python train_rlvr.py` manually if you have your environment set up.*


## 🧠 Components
- **`config.py`**: Central configuration file.
- **`train_rlvr.py`**: Core GRPO training loop with micro-batching and gradient accumulation.
- **`sat_utils.py`**: Generators and verifiers for SAT problems.
- **`tsp_utils.py`**: Generators, renderers, and verifiers for TSP.
- **`benchmark.py`**: Evaluation logic against baselines.
