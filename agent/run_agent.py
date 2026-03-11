"""
Run agent on eval datasets and save trajectories + metrics.

Usage:
    python agent/run_agent.py --model baseline --data data/eval_d1.jsonl --limit 5 --verbose
    python agent/run_agent.py --model grpo --model-path ./checkpoints/grpo --data data/eval_d1.jsonl
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.data import Data
from env.metro_env import MetroViolationsEnv
from verifier.trajectory_verifier import MetroTrajectoryVerifier


def load_episodes(data_path: str, limit: int = None) -> list:
    """Load episodes from JSONL file."""
    episodes = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(Data.from_json(line))
                if limit and len(episodes) >= limit:
                    break
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Run agent on eval data")
    parser.add_argument("--model", choices=["baseline", "grpo"], default="baseline")
    parser.add_argument("--model-path", default=None, help="Path to fine-tuned model")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", required=True, help="Path to eval JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Max episodes to run")
    parser.add_argument("--output-dir", default="logs", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load episodes
    episodes = load_episodes(args.data, args.limit)
    print(f"Loaded {len(episodes)} episodes from {args.data}")

    # Create agent
    model_path = args.model_path or args.model_name
    if args.model == "grpo" and args.model_path:
        model_path = args.model_path

    from agent.baseline_agent import BaselineAgent
    agent = BaselineAgent(
        model_name=model_path,
        device=args.device,
    )

    # Create env and verifier
    env = MetroViolationsEnv()
    verifier = MetroTrajectoryVerifier()

    # Run episodes
    data_basename = os.path.splitext(os.path.basename(args.data))[0]
    trajectories_path = os.path.join(
        args.output_dir, f"trajectories_{args.model}_{data_basename}.jsonl"
    )
    metrics_path = os.path.join(
        args.output_dir, f"metrics_{args.model}_{data_basename}.json"
    )

    all_metrics = []
    total_success = 0
    total_reward = 0.0

    with open(trajectories_path, "w", encoding="utf-8") as traj_f:
        for i, episode in enumerate(episodes):
            print(f"\n[{i+1}/{len(episodes)}] {episode.question_id} (d={episode.difficulty}, {episode.task_type})")
            if args.verbose:
                print(f"  Q: {episode.question_text[:100]}...")

            # Run agent
            result = agent.run_episode(env, episode, verbose=args.verbose)

            # Verify trajectory
            metrics = verifier.verify_trajectory(env, episode, result["actions"])

            # Log
            print(f"  success={metrics['success']}, reward={metrics['total_reward']:.3f}, "
                  f"steps={metrics['steps']}, tools={metrics['tool_calls']}, "
                  f"violations={metrics['policy_violations']}")

            total_success += int(metrics["success"])
            total_reward += metrics["total_reward"]
            all_metrics.append({
                "question_id": episode.question_id,
                "difficulty": episode.difficulty,
                "task_type": episode.task_type,
                **metrics,
            })

            # Save trajectory
            traj_f.write(json.dumps({
                **result,
                "metrics": {k: v for k, v in metrics.items() if k != "info_trace"},
            }, ensure_ascii=False) + "\n")

    # Save aggregate metrics
    n = len(episodes)
    summary = {
        "model": args.model,
        "data_file": args.data,
        "total_episodes": n,
        "success_rate": total_success / n if n else 0,
        "mean_reward": total_reward / n if n else 0,
        "mean_steps": sum(m["steps"] for m in all_metrics) / n if n else 0,
        "mean_tool_calls": sum(m["tool_calls"] for m in all_metrics) / n if n else 0,
        "mean_policy_violations": sum(m["policy_violations"] for m in all_metrics) / n if n else 0,
        "per_episode": all_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Summary ===")
    print(f"  Success rate: {summary['success_rate']:.2%}")
    print(f"  Mean reward:  {summary['mean_reward']:.3f}")
    print(f"  Mean steps:   {summary['mean_steps']:.1f}")
    print(f"  Mean tools:   {summary['mean_tool_calls']:.1f}")
    print(f"  Mean violations: {summary['mean_policy_violations']:.2f}")
    print(f"\n  Trajectories: {trajectories_path}")
    print(f"  Metrics:      {metrics_path}")


if __name__ == "__main__":
    main()
