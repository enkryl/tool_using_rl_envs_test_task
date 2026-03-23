"""
GRPO training script for the metro violations agent.

Implements online multi-step rollouts:
- For each episode, generates G rollouts through env interaction
- Computes trajectory-level rewards
- Uses GRPO (Group Relative Policy Optimization) to update the model

Usage (on GPU server):
    python training/grpo_train.py --config training/config.yaml
    
Or see train_notebook.ipynb for interactive usage.
"""

import os
import sys
import json
import time
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.data import Data
from env.metro_env import MetroViolationsEnv
from verifier.trajectory_verifier import MetroTrajectoryVerifier
from training.reward_fn import serialize_trajectory, parse_trajectory


# ─── Config ────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "checkpoints/grpo"
    
    # Training
    num_epochs: int = 3
    batch_size: int = 4           # episodes per batch
    num_generations: int = 4      # rollouts per episode (G)
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    
    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.7
    max_steps_per_episode: int = 15
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Logging
    log_dir: str = "logs/training"
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 200
    
    # Data
    train_data_path: str = "data/train.jsonl"
    eval_data_paths: list = field(default_factory=lambda: [
        "data/eval_d1.jsonl",
        "data/eval_d2.jsonl", 
        "data/eval_d3.jsonl",
        "data/eval_d4.jsonl",
        "data/eval_d5.jsonl",
    ])
    eval_limit: int = 20       # episodes per eval bucket during training
    
    # Curriculum learning: epoch_number -> max_difficulty
    # Episodes with difficulty > max_difficulty are skipped in that epoch.
    # If None, all data is used every epoch (no curriculum).
    curriculum_schedule: dict = field(default_factory=lambda: {
        1: 4,    # epoch 1: only d1–d4 (learn format + basic tool use)
        2: 7,    # epoch 2: add d5–d7 (multi-step, ambiguous)
        3: 10,   # epoch 3: all difficulties (full complexity)
    })
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "metro-violations-grpo"
    wandb_run_name: str = None


SYSTEM_PROMPT = """You are a metro violations analyst. You analyze violation data using available tools.

Action format (MUST follow exactly):
- To use a tool: TOOL_CALL {"name": "<tool_name>", "args": {<arguments>}}
- To send a free-text message: just type your message
- To submit your final answer: FINAL_ANSWER <your answer>

IMPORTANT RULES:
1. Before any state-changing tool (like open_case), you MUST first propose the action in free-text and wait for confirmation.
2. Only use entity IDs (person_id, station_id, violation_code) that appear in the task or in tool outputs.
3. Start by understanding the database schema if needed.
4. Be concise: minimize unnecessary tool calls.
5. When you have the answer, submit it with FINAL_ANSWER prefix."""


class GRPOTrainer:
    """
    GRPO trainer with online multi-step rollouts.
    
    For each episode:
    1. Generate G rollouts by interacting with the env
    2. Compute trajectory-level rewards
    3. Normalize advantages within the group
    4. Update policy with GRPO loss
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.env = MetroViolationsEnv()
        self.verifier = MetroTrajectoryVerifier()
        
        # Setup logging
        os.makedirs(config.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            config.log_dir,
            f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("GRPOTrainer")
        
    def setup_model(self):
        """Load model and tokenizer, optionally with LoRA."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.logger.info(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Enable gradient checkpointing to save VRAM
        self.model.gradient_checkpointing_enable()
        
        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
    
    def load_data(self):
        """Load training and eval data."""
        self.train_data = []
        with open(self.config.train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.train_data.append(Data.from_json(line.strip()))
        self.logger.info(f"Loaded {len(self.train_data)} training episodes")
        
        self.eval_data = {}
        for path in self.config.eval_data_paths:
            if os.path.exists(path):
                bucket = os.path.splitext(os.path.basename(path))[0]
                episodes = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            episodes.append(Data.from_json(line.strip()))
                self.eval_data[bucket] = episodes
                self.logger.info(f"Loaded {len(episodes)} eval episodes for {bucket}")
    
    def generate_rollout(self, data: Data) -> Dict[str, Any]:
        """
        Generate one rollout by interacting with the environment.
        
        Returns dict with:
            actions: list of action strings
            observations: list of observation strings
            reward: total trajectory reward
            metrics: verifier metrics
            messages: full conversation for loss computation
        """
        obs = self.env.reset(data)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs},
        ]
        
        actions = []
        observations = [obs]
        done = False
        step = 0
        
        self.model.eval()
        
        while not done and step < self.config.max_steps_per_episode:
            # Generate action
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=4096
            ).to(self.model.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            action = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            action = action.split("\n")[0].strip()
            
            if not action:
                action = "FINAL_ANSWER I don't know"
            
            actions.append(action)
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            observations.append(obs)
            
            messages.append({"role": "assistant", "content": action})
            messages.append({"role": "user", "content": obs})
            
            step += 1
        
        # Get metrics
        metrics = self.verifier.verify_trajectory(self.env, data, actions)
        
        return {
            "actions": actions,
            "observations": observations,
            "reward": metrics["total_reward"],
            "metrics": metrics,
            "messages": messages,
        }
    
    def compute_grpo_loss(
        self,
        rollouts: List[Dict[str, Any]],
        data: Data,
    ) -> torch.Tensor:
        """
        Compute GRPO loss for a group of rollouts of the same episode.
        
        GRPO advantage: A_i = (r_i - mean(r)) / (std(r) + eps)
        Loss = -sum(A_i * log_prob(actions_i)) with KL penalty
        """
        rewards = [r["reward"] for r in rollouts]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
        advantages = [(r - mean_r) / std_r for r in rewards]
        
        losses = []
        
        self.model.train()
        
        for rollout, advantage in zip(rollouts, advantages):
            if abs(advantage) < 0.01:
                continue
            
            messages = rollout["messages"]
            
            # Get log probs for assistant actions
            for i, msg in enumerate(messages):
                if msg["role"] != "assistant":
                    continue
                
                # Build context up to this point
                context_msgs = messages[:i+1]
                text = self.tokenizer.apply_chat_template(
                    context_msgs, tokenize=False, add_generation_prompt=False
                )
                
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=4096
                ).to(self.model.device)
                
                # Get action tokens
                action_text = msg["content"]
                action_ids = self.tokenizer.encode(action_text, add_special_tokens=False)
                
                if len(action_ids) == 0:
                    continue
                
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                log_probs = -outputs.loss  # approximate per-token log prob
                
                # Weighted by advantage
                losses.append(-advantage * log_probs)
        
        if not losses:
            # No valid updates — return zero tensor with grad so backward() is safe
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)
        
        return torch.stack(losses).mean()
    
    def train_step(self, batch: List[Data]) -> Dict[str, float]:
        """One training step: generate rollouts + update."""
        batch_metrics = {
            "loss": 0.0,
            "mean_reward": 0.0,
            "max_reward": -float("inf"),
            "min_reward": float("inf"),
            "success_rate": 0.0,
            "mean_steps": 0.0,
            "mean_tools": 0.0,
            "mean_violations": 0.0,
        }
        
        total_rollouts = 0
        total_success = 0
        
        self.optimizer.zero_grad()
        
        for data in batch:
            # Generate G rollouts (no grad needed for generation)
            rollouts = []
            for g in range(self.config.num_generations):
                rollout = self.generate_rollout(data)
                rollouts.append(rollout)
            
            # Free generation cache
            torch.cuda.empty_cache()
            
            rewards = [r["reward"] for r in rollouts]
            batch_metrics["mean_reward"] += sum(rewards) / len(rewards)
            batch_metrics["max_reward"] = max(batch_metrics["max_reward"], max(rewards))
            batch_metrics["min_reward"] = min(batch_metrics["min_reward"], min(rewards))
            
            for r in rollouts:
                total_rollouts += 1
                if r["metrics"]["success"]:
                    total_success += 1
                batch_metrics["mean_steps"] += r["metrics"]["steps"]
                batch_metrics["mean_tools"] += r["metrics"]["tool_calls"]
                batch_metrics["mean_violations"] += r["metrics"]["policy_violations"]
            
            # Compute loss and backward immediately (saves VRAM vs accumulating)
            loss = self.compute_grpo_loss(rollouts, data)
            if loss.grad_fn is not None:
                (loss / len(batch)).backward()  # scale by batch size for proper averaging
            batch_metrics["loss"] += loss.item()
            
            # Free loss graph
            del rollouts, loss
            torch.cuda.empty_cache()
        
        # Gradient step
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        
        # Normalize metrics
        n = len(batch)
        batch_metrics["loss"] /= n
        batch_metrics["mean_reward"] /= n
        batch_metrics["success_rate"] = total_success / total_rollouts if total_rollouts else 0
        batch_metrics["mean_steps"] /= total_rollouts if total_rollouts else 1
        batch_metrics["mean_tools"] /= total_rollouts if total_rollouts else 1
        batch_metrics["mean_violations"] /= total_rollouts if total_rollouts else 1
        
        return batch_metrics
    
    def evaluate(self, eval_limit: int = None) -> Dict[str, Dict[str, float]]:
        """Run evaluation on all eval buckets."""
        eval_limit = eval_limit or self.config.eval_limit
        results = {}
        
        self.model.eval()
        
        for bucket, episodes in self.eval_data.items():
            subset = episodes[:eval_limit]
            total_success = 0
            total_reward = 0.0
            total_steps = 0
            total_tools = 0
            total_violations = 0
            
            for data in subset:
                rollout = self.generate_rollout(data)
                metrics = rollout["metrics"]
                total_success += int(metrics["success"])
                total_reward += metrics["total_reward"]
                total_steps += metrics["steps"]
                total_tools += metrics["tool_calls"]
                total_violations += metrics["policy_violations"]
            
            n = len(subset)
            results[bucket] = {
                "success_rate": total_success / n,
                "mean_reward": total_reward / n,
                "mean_steps": total_steps / n,
                "mean_tools": total_tools / n,
                "mean_violations": total_violations / n,
            }
            
            self.logger.info(
                f"  {bucket}: success={results[bucket]['success_rate']:.1%}, "
                f"reward={results[bucket]['mean_reward']:.3f}"
            )
        
        return results
    
    def train(self):
        """Main training loop."""
        self.setup_model()
        self.load_data()
        
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name or f"grpo_{datetime.now().strftime('%m%d_%H%M')}",
                    config=vars(self.config),
                )
            except ImportError:
                self.logger.warning("wandb not installed, logging to file only")
                self.config.use_wandb = False
        
        global_step = 0
        all_logs = []
        
        self.logger.info("=== Starting GRPO Training ===")
        self.logger.info(f"  Train episodes: {len(self.train_data)}")
        self.logger.info(f"  Batch size: {self.config.batch_size}")
        self.logger.info(f"  Num generations: {self.config.num_generations}")
        self.logger.info(f"  Epochs: {self.config.num_epochs}")
        
        for epoch in range(self.config.num_epochs):
            epoch_num = epoch + 1
            
            # Curriculum: filter data by max difficulty for this epoch
            if self.config.curriculum_schedule:
                max_diff = self.config.curriculum_schedule.get(
                    epoch_num,
                    max(self.config.curriculum_schedule.values())  # fallback to max
                )
                epoch_data = [d for d in self.train_data if d.difficulty <= max_diff]
                self.logger.info(
                    f"\n=== Epoch {epoch_num}/{self.config.num_epochs} "
                    f"(curriculum: d1–d{max_diff}, {len(epoch_data)} episodes) ==="
                )
            else:
                epoch_data = self.train_data
                self.logger.info(f"\n=== Epoch {epoch_num}/{self.config.num_epochs} ===")
            
            # Shuffle training data for this epoch
            random.shuffle(epoch_data)
            
            for i in range(0, len(epoch_data), self.config.batch_size):
                batch = epoch_data[i:i + self.config.batch_size]
                if len(batch) < self.config.batch_size:
                    continue
                
                t0 = time.time()
                metrics = self.train_step(batch)
                dt = time.time() - t0
                
                global_step += 1
                
                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "time": dt,
                    **metrics,
                }
                all_logs.append(log_entry)
                
                # Log to file
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                
                # Console logging
                if global_step % self.config.log_every == 0:
                    self.logger.info(
                        f"Step {global_step}: loss={metrics['loss']:.4f}, "
                        f"reward={metrics['mean_reward']:.3f}, "
                        f"success={metrics['success_rate']:.1%}, "
                        f"steps={metrics['mean_steps']:.1f}, "
                        f"time={dt:.1f}s"
                    )
                
                # Wandb logging
                if self.config.use_wandb:
                    try:
                        import wandb
                        wandb.log(log_entry, step=global_step)
                    except Exception:
                        pass
                
                # Evaluation
                if global_step % self.config.eval_every == 0:
                    self.logger.info(f"\n--- Eval at step {global_step} ---")
                    eval_results = self.evaluate()
                    
                    if self.config.use_wandb:
                        try:
                            import wandb
                            for bucket, res in eval_results.items():
                                for k, v in res.items():
                                    wandb.log({f"eval/{bucket}/{k}": v}, step=global_step)
                        except Exception:
                            pass
                
                # Save checkpoint
                if global_step % self.config.save_every == 0:
                    ckpt_dir = os.path.join(
                        self.config.output_dir, f"step_{global_step}"
                    )
                    self.save_checkpoint(ckpt_dir)
        
        # Final save
        self.save_checkpoint(os.path.join(self.config.output_dir, "final"))
        
        # Final evaluation
        self.logger.info("\n=== Final Evaluation ===")
        final_results = self.evaluate(eval_limit=50)
        
        # Save final results
        results_path = os.path.join(self.config.log_dir, "final_eval.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)
        
        self.logger.info(f"Training complete! Logs: {self.log_file}")
        self.logger.info(f"Final checkpoint: {self.config.output_dir}/final")
        
        if self.config.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        
        return all_logs
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.logger.info(f"Checkpoint saved: {path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-data", default="data/train.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--log-dir", default="logs/training")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    args = parser.parse_args()
    
    config = GRPOConfig(
        model_name=args.model_name,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        use_wandb=not args.no_wandb,
        log_dir=args.log_dir,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    
    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
