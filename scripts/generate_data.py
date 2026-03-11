"""
Generate train and eval datasets for the metro violations environment.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --smoke   # generate 5 episodes for quick test
"""

import os
import sys
import argparse
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.metro_env import MetroViolationsEnv


def main():
    parser = argparse.ArgumentParser(description="Generate metro violations datasets")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 5 episodes per difficulty")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    env = MetroViolationsEnv()

    if args.smoke:
        print("=== SMOKE TEST ===")
        for d in range(1, 11):
            episodes = env.generate(
                num_of_questions=5,
                max_attempts=50,
                difficulty=d,
                seed=args.seed + d,
            )
            print(f"  d{d}: generated {len(episodes)} episodes")
            for ep in episodes[:2]:
                print(f"    - [{ep.task_type}] {ep.question_text[:80]}...")
                print(f"      expected: {ep.expected_result}")
        print("Smoke test passed!")
        return

    # ── Train dataset ──────────────────────────────────────────────
    # d1-d3: 300 each, d4-d6: 250 each, d7-d8: 200 each, d9-d10: 150 each
    train_counts = {
        1: 300, 2: 300, 3: 300,
        4: 250, 5: 250, 6: 250,
        7: 200, 8: 200,
        9: 150, 10: 150,
    }

    print("=== Generating TRAIN dataset ===")
    train_path = os.path.join(args.output_dir, "train.jsonl")
    total_train = 0
    with open(train_path, "w", encoding="utf-8") as f:
        for d in range(1, 11):
            count = train_counts[d]
            print(f"  d{d}: generating {count} episodes...", end=" ", flush=True)
            episodes = env.generate(
                num_of_questions=count,
                max_attempts=200,
                difficulty=d,
                seed=args.seed * 100 + d,
            )
            for ep in episodes:
                f.write(ep.to_json() + "\n")
            total_train += len(episodes)
            print(f"got {len(episodes)}")

    print(f"  Total train: {total_train} episodes -> {train_path}")

    # ── Eval datasets (5 buckets) ──────────────────────────────────
    # bucket 1: d1-d2, bucket 2: d3-d4, ..., bucket 5: d9-d10
    buckets = {
        1: [1, 2],
        2: [3, 4],
        3: [5, 6],
        4: [7, 8],
        5: [9, 10],
    }

    print("\n=== Generating EVAL datasets ===")
    for bucket_id, difficulties in buckets.items():
        eval_path = os.path.join(args.output_dir, f"eval_d{bucket_id}.jsonl")
        per_diff = 100 // len(difficulties)  # 50 per difficulty in each bucket

        total_eval = 0
        with open(eval_path, "w", encoding="utf-8") as f:
            for d in difficulties:
                print(f"  eval_d{bucket_id} / d{d}: generating {per_diff} episodes...", end=" ", flush=True)
                episodes = env.generate(
                    num_of_questions=per_diff,
                    max_attempts=200,
                    difficulty=d,
                    seed=args.seed * 1000 + bucket_id * 100 + d,
                )
                for ep in episodes:
                    f.write(ep.to_json() + "\n")
                total_eval += len(episodes)
                print(f"got {len(episodes)}")

        print(f"  Bucket {bucket_id}: {total_eval} episodes -> {eval_path}")

    print("\nDone! All datasets generated.")


if __name__ == "__main__":
    main()
