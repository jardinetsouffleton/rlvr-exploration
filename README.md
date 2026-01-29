# Combinatorial Optimization with GRPO

This repository contains code for training Large Language Models to solve combinatorial optimization problems (like TSP and SAT) using Group Relative Policy Optimization (GRPO).

## Structure
- `src/`: Source code modules.
- `tests/`: Unit and integration tests.
- `train_grpo.py`: Main training script.
- `tsp_mip_solver.py`: MIP solver integration with LLM bounds.

## Installation
```bash
pip install -r requirements.txt
```

## Running Code
The code is now structured as a package. You should run scripts from the root directory.

**Training:**
```bash
python -m src.train_grpo
```

**Benchmarking:**
```bash
python -m src.run_tsp_dual_pipeline
```

## Testing
We use `pytest` for testing. The tests cover utilities, MIP solver integration, and inference logic (with mocked LLMs).

To run all tests:
```bash
python -m pytest tests/
```

To run a specific test file:
```bash
python -m pytest tests/test_tsp_min_solver.py
```
