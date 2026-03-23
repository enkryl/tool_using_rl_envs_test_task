import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

# Find the most recent training log
log_files = glob.glob('logs/training/train_vllm_*.jsonl')
if not log_files:
    print("No training logs found in logs/training/")
    exit()

latest_log = max(log_files, key=os.path.getmtime)
print(f"Reading logs from: {latest_log}")

# Read and parse logs
train_logs = []
with open(latest_log, 'r') as f:
    for line in f:
        try:
            train_logs.append(json.loads(line.strip()))
        except:
            continue

df = pd.DataFrame(train_logs)

if not df.empty:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GRPO vLLM Training Curves', fontsize=16, fontweight='bold')
    
    # Define metrics to plot
    plots = [
        ('loss', 'Loss', 'tab:red'),
        ('mean_reward', 'Mean Reward', 'tab:blue'),
        ('success_rate', 'Success Rate', 'tab:green'),
        ('time_gen', 'Generation Time (s)', 'tab:orange'),
    ]
    
    # We use rollout_batch as the X-axis
    df['step'] = df['rollout_batch']
    w = max(1, len(df) // 10)
    
    for ax, (key, title, color) in zip(axes.flatten(), plots):
        if key in df.columns:
            sm = df[key].rolling(window=w, min_periods=1).mean()
            ax.plot(df['step'], df[key], color=color, alpha=0.3, label='raw')
            ax.plot(df['step'], sm, color=color, linewidth=2, label=f'smooth (w={w})')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Rollout Batch')
            ax.grid(alpha=0.3)
            ax.legend()
            
            # Format success rate as percentage if needed
            if key == 'success_rate':
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
                
    plt.tight_layout()
    plot_path = 'logs/training/vllm_training_curves.png'
    os.makedirs('logs/training', exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.show()
else:
    print('No training logs found in the latest file.')
