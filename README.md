# RLVR for Combinatorial Optimization (TSP & SAT)

This project implements a Reinforcement Learning with Verifiable Rewards (RLVR) loop for solving combinatorial optimization problems—specifically **Traveling Salesperson Problem (TSP)** and **Boolean Satisfiability (SAT)**—using the **GRPO (Group Relative Policy Optimization)** algorithm via the [TRL library](https://huggingface.co/docs/trl).

## 🚀 Key Features
- **Multi-Problem Support**: Train on TSP (text) or SAT (text).
- **TRL GRPOTrainer**: Uses HuggingFace TRL's `GRPOTrainer` for efficient reinforcement learning.
- **Verifiable Rewards**: 
  - **TSP**: Rewards valid tours based on optimality gap vs. Nearest Neighbor/Optimal baselines.
  - **TSP Dual**: Rewards valid lower bounds on optimal cost (MIP acceleration).
  - **SAT**: Rewards valid variable assignments based on the number of satisfied clauses.
- **Curriculum Learning**: Linearly scales problem difficulty (e.g., number of variables in SAT) over training steps.
- **WandB Integration**: Training metrics logged to Weights & Biases.
- **Benchmarking**: Automatic comparison of base vs. trained model on TSP instances.

## 🛠️ Configuration
The project is fully configurable via **`config.py`**. Edit this file to change experiments.

### Key Configuration Variables

| Category | Variable | Description |
| :--- | :--- | :--- |
| **Problem** | `PROBLEM_TYPE` | `"sat"` or `"tsp"`. Determines the problem domain. |
| **Model** | `MODEL_ID` | HF Model ID (e.g. `"Qwen/Qwen2.5-1.5B-Instruct"`). |
| **Training** | `GROUP_SIZE` | Number of completions generated per prompt (GRPO group size). |
| | `STEPS` | Total training steps. |
| | `LR` | Learning rate. |
| | `BETA` | KL divergence penalty coefficient. |
| | `PROMPT_MODE` | `"cot"` (Chain of Thought) or `"direct"` (Output answer only). |
| **SAT Config** | `SAT_CURRICULUM` | `True` to enable difficulty scaling. |
| | `SAT_START_VARS` | Number of variables at step 0. |
| | `SAT_END_VARS` | Number of variables at final step. |
| **TSP Config** | `N_CITIES` | Number of cities for TSP instances. |
| | `REWARD_BASELINE` | `"nn"` (Nearest Neighbor) or `"optimal"`. |

## 🏃 Usage

### 1. Installation
```bash
pip install -r requirements.txt
pip install trl  # TRL library for GRPOTrainer
```

### 2. Configure Experiment
Open `config.py` and set your desired parameters.

*Example: To train for SAT with curriculum:*
```python
PROBLEM_TYPE = "sat"
SAT_CURRICULUM = True
SAT_START_VARS = 5
SAT_END_VARS = 15
GROUP_SIZE = 12
STEPS = 1500
```

### 3. Run Training (Recommended)
The central entry point is the **`run_safe.sh`** script. It handles virtual environment setup, dependency installation, and environment variable configuration before launching training.

```bash
./run_safe.sh
```

This script will:
1. Create/Activate a virtual environment in `.venv`.
2. Install dependencies from `requirements.txt`.
3. Set `HF_HOME` to a shared cache directory.
4. Execute `train_grpo.py` with the settings from `config.py`.

*Alternatively, run manually:*
```bash
source .venv/bin/activate
python train_grpo.py
```

### 4. Running TSP Dual Pipeline
For the complete TSP Dual training and MIP benchmarking loop:
```bash
python run_tsp_dual_pipeline.py --steps 500 --n_cities_min 5 --n_cities_max 10
```
This script handles configuration, training, and the final 3-way benchmark (Vanilla vs Base vs Trained).

## 🧠 Components
| File | Description |
| :--- | :--- |
| `config.py` | Central configuration file. |
| `train_grpo.py` | Main training script using TRL's `GRPOTrainer`. |
| `sat_utils.py` | Generators and verifiers for SAT problems. |
| `tsp_utils.py` | Generators, renderers, and verifiers for TSP. |
| `tsp_dual_utils.py` | Generators and verifiers for TSP Dual Bound problems. |
| `tsp_mip_solver.py` | MIP solver integration for TSP Dual benchmarks. |
| `benchmark.py` | Evaluation logic against baselines. |
| `final_benchmark.py` | Final verification benchmark for TSP Dual. |
| `run_tsp_dual_pipeline.py` | End-to-end pipeline for TSP Dual training and benchmarking. |

## 📊 Outputs
After training, outputs are saved to `OUTPUT_DIR` (configured in `config.py`):
- `final_model/` - Trained model checkpoint
- `config.json` - Experiment configuration
- `benchmark_results.csv` - Comparison of base vs. trained model (TSP only)
