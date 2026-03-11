# HW4: Metro Violations LLM Agent

Multi-step ToolEnv environment for an LLM agent that analyzes metro violation data using SQL tools. Trained with GRPO (Group Relative Policy Optimization).

## Quick Start

### 1. Generate Data (locally)

```bash
cd d:\omut\hw4

# Smoke test (5 episodes per difficulty)
python scripts/generate_data.py --smoke

# Full generation (~2350 train + 500 eval)
python scripts/generate_data.py
```

### 2. Run Baseline (GPU recommended)

```bash
# Run on one eval bucket
python agent/run_agent.py --model baseline --data data/eval_d1.jsonl --verbose

# Run on all buckets
for i in 1 2 3 4 5; do
    python agent/run_agent.py --model baseline --data data/eval_d${i}.jsonl --output-dir logs
done
```

### 3. Train GRPO (GPU required)

```bash
pip install -r requirements_gpu.txt

# Via script
python training/grpo_train.py --epochs 2 --batch-size 2 --num-generations 4

# Or use the notebook
jupyter notebook training/train_notebook.ipynb
```

### 4. Evaluate & Compare

```bash
# Run GRPO model on eval
for i in 1 2 3 4 5; do
    python agent/run_agent.py --model grpo --model-path checkpoints/grpo/final --data data/eval_d${i}.jsonl
done

# Compare
python scripts/compare_results.py --baseline-dir logs --grpo-dir logs
```

## Project Structure

```
base/           # Data, ToolEnv, TrajectoryVerifier (abstract)
env/            # DB generator, tools, MetroViolationsEnv, episode generator
verifier/       # MetroTrajectoryVerifier
agent/          # BaselineAgent + run_agent script
training/       # GRPO trainer, reward function, notebook
scripts/        # Data generation, eval, comparison
data/           # Generated datasets
logs/           # Training logs, metrics, trajectories
```

## Environment Details

- **10 difficulty levels**: d1 (simple count) → d10 (full case workflow)
- **7 tools**: get_schema, get_table_sample, run_sql, lookup_station, lookup_violation, get_case, open_case
- **Policy rules**: confirmation before mutations, no hallucinated entities
- **Reward**: outcome (+1/-1) + shaping (step/tool/violation penalties)
