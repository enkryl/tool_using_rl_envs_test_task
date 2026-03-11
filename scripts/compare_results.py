"""
Compare baseline vs GRPO agent results and generate analysis plots.

Usage:
    python scripts/compare_results.py \
        --baseline logs/metrics_baseline_eval_d1.json logs/metrics_baseline_eval_d2.json ... \
        --grpo logs/metrics_grpo_eval_d1.json logs/metrics_grpo_eval_d2.json ...
    
    # Or with glob:
    python scripts/compare_results.py --baseline-dir logs --grpo-dir logs
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_metrics(file_paths: list) -> dict:
    """Load and combine metrics from multiple JSON files, keyed by bucket."""
    combined = {}
    for path in sorted(file_paths):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Extract bucket name from filename (e.g., "eval_d1" from "metrics_baseline_eval_d1.json")
        basename = os.path.basename(path)
        parts = basename.replace("metrics_", "").replace(".json", "").split("_")
        # Find the eval_dX part
        bucket = None
        for i, p in enumerate(parts):
            if p.startswith("eval"):
                bucket = "_".join(parts[i:])
                break
        if bucket is None:
            bucket = basename
        combined[bucket] = data
    return combined


def print_comparison_table(baseline: dict, grpo: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Bucket':<12} {'Metric':<20} {'Baseline':>12} {'GRPO':>12} {'Delta':>12}")
    print("=" * 90)

    metrics_keys = ["success_rate", "mean_reward", "mean_steps", "mean_tool_calls", "mean_policy_violations"]
    labels = ["Success Rate", "Mean Reward", "Mean Steps", "Mean Tools", "Mean Violations"]

    all_buckets = sorted(set(list(baseline.keys()) + list(grpo.keys())))

    for bucket in all_buckets:
        b_data = baseline.get(bucket, {})
        g_data = grpo.get(bucket, {})

        for key, label in zip(metrics_keys, labels):
            b_val = b_data.get(key, 0)
            g_val = g_data.get(key, 0)
            delta = g_val - b_val

            if key == "success_rate":
                print(f"{bucket:<12} {label:<20} {b_val:>11.1%} {g_val:>11.1%} {delta:>+11.1%}")
            else:
                print(f"{bucket:<12} {label:<20} {b_val:>12.3f} {g_val:>12.3f} {delta:>+12.3f}")
        print("-" * 90)


def generate_plots(baseline: dict, grpo: dict, output_dir: str):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots. Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)
    buckets = sorted(set(list(baseline.keys()) + list(grpo.keys())))
    x = range(len(buckets))
    bucket_labels = [b.replace("eval_", "").upper() for b in buckets]

    metrics_to_plot = [
        ("success_rate", "Success Rate", True),
        ("mean_reward", "Mean Reward", False),
        ("mean_steps", "Mean Steps", False),
        ("mean_tool_calls", "Mean Tool Calls", False),
        ("mean_policy_violations", "Mean Policy Violations", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Baseline vs GRPO Comparison", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for idx, (key, title, is_pct) in enumerate(metrics_to_plot):
        ax = axes[idx]
        b_vals = [baseline.get(b, {}).get(key, 0) for b in buckets]
        g_vals = [grpo.get(b, {}).get(key, 0) for b in buckets]

        width = 0.35
        bars1 = ax.bar([i - width/2 for i in x], b_vals, width, label="Baseline", color="#5B8DEF", alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], g_vals, width, label="GRPO", color="#FF6B6B", alpha=0.8)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(bucket_labels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        if is_pct:
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Remove extra subplot
    axes[-1].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs GRPO results")
    parser.add_argument("--baseline", nargs="+", help="Baseline metric JSON files")
    parser.add_argument("--grpo", nargs="+", help="GRPO metric JSON files")
    parser.add_argument("--baseline-dir", default=None, help="Directory with baseline metrics")
    parser.add_argument("--grpo-dir", default=None, help="Directory with GRPO metrics")
    parser.add_argument("--output-dir", default="logs/plots", help="Output directory for plots")
    args = parser.parse_args()

    # Collect file paths
    baseline_files = args.baseline or []
    grpo_files = args.grpo or []

    if args.baseline_dir:
        baseline_files += [
            os.path.join(args.baseline_dir, f)
            for f in sorted(os.listdir(args.baseline_dir))
            if f.startswith("metrics_baseline") and f.endswith(".json")
        ]
    if args.grpo_dir:
        grpo_files += [
            os.path.join(args.grpo_dir, f)
            for f in sorted(os.listdir(args.grpo_dir))
            if f.startswith("metrics_grpo") and f.endswith(".json")
        ]

    if not baseline_files and not grpo_files:
        print("Error: Provide --baseline/--grpo files or --baseline-dir/--grpo-dir")
        sys.exit(1)

    baseline = load_metrics(baseline_files) if baseline_files else {}
    grpo = load_metrics(grpo_files) if grpo_files else {}

    if baseline and grpo:
        print_comparison_table(baseline, grpo)
        generate_plots(baseline, grpo, args.output_dir)
    elif baseline:
        print("\nBaseline results only:")
        for bucket, data in sorted(baseline.items()):
            print(f"  {bucket}: success={data.get('success_rate', 0):.1%}, "
                  f"reward={data.get('mean_reward', 0):.3f}")
    elif grpo:
        print("\nGRPO results only:")
        for bucket, data in sorted(grpo.items()):
            print(f"  {bucket}: success={data.get('success_rate', 0):.1%}, "
                  f"reward={data.get('mean_reward', 0):.3f}")


if __name__ == "__main__":
    main()
