"""
GRPO training with vLLM for fast rollout generation.

Two-phase approach:
  Phase 1 (vLLM): Batch-generate rollouts through env interaction (fast, no grad)
  Phase 2 (HF):   Compute GRPO loss on collected rollouts and update LoRA weights

This is ~5-10x faster than grpo_train.py because vLLM uses PagedAttention
and optimized CUDA kernels for generation.

Usage:
    See train_notebook.ipynb
"""

import os
import sys
import json
import time
import random
import logging
import gc
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from copy import deepcopy

# vLLM environment config — MUST be set before importing vLLM
os.environ["VLLM_USE_V1"] = "0"  # use stable V0 engine
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"  # suppress vLLM info spam
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.data import Data
from env.metro_env import MetroViolationsEnv
from verifier.trajectory_verifier import MetroTrajectoryVerifier


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


@dataclass
class GRPOVLLMConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "checkpoints/grpo"

    # Training
    num_epochs: int = 3
    rollout_batch_size: int = 30   # episodes to generate rollouts for in one vLLM batch
    num_generations: int = 2       # rollouts per episode
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    train_batch_size: int = 4      # episodes per gradient step (during Phase 2)

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.7
    max_steps_per_episode: int = 8

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Logging
    log_dir: str = "logs/training"
    eval_every_rollout_batch: int = 3  # eval every N rollout batches
    save_every_rollout_batch: int = 5

    # Data
    train_data_path: str = "data/train.jsonl"
    eval_data_paths: list = field(default_factory=lambda: [
        "data/eval_d1.jsonl", "data/eval_d2.jsonl",
        "data/eval_d3.jsonl", "data/eval_d4.jsonl",
        "data/eval_d5.jsonl",
    ])
    eval_limit: int = 10
    train_subset_per_difficulty: int = 30  # max episodes per difficulty level

    # Curriculum
    curriculum_schedule: dict = field(default_factory=lambda: {
        1: 4, 2: 7, 3: 10,
    })

    # vLLM
    vllm_gpu_memory_utilization: float = 0.45
    vllm_max_model_len: int = 2048


class GRPOVLLMTrainer:
    """
    GRPO with vLLM-accelerated rollout generation.

    Workflow per rollout batch:
      1. Load vLLM → generate rollouts for N episodes (batched, fast)
      2. Unload vLLM → load HF model with LoRA
      3. Compute GRPO loss on rollouts → backward → optimizer step
      4. Save LoRA checkpoint → repeat with updated weights
    """

    def __init__(self, config: GRPOVLLMConfig):
        self.config = config
        self.env = MetroViolationsEnv()
        self.verifier = MetroTrajectoryVerifier()

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        self.log_file = os.path.join(
            config.log_dir, f"train_vllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self.lora_path = os.path.join(config.output_dir, "_latest_lora")

        self.logger = logging.getLogger("GRPOvLLM")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        self.logger.propagate = False  # Prevent duplicate logs from root logger

    def load_data(self):
        """Load and subsample training data."""
        all_data = []
        with open(self.config.train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_data.append(Data.from_json(line.strip()))

        # Subsample per difficulty
        self.train_data = []
        for d in range(1, 11):
            episodes_d = [e for e in all_data if e.difficulty == d]
            k = min(self.config.train_subset_per_difficulty, len(episodes_d))
            self.train_data.extend(random.sample(episodes_d, k))
        random.shuffle(self.train_data)
        self.logger.info(f"Training data: {len(self.train_data)} episodes (subsampled from {len(all_data)})")

        self.eval_data = {}
        for path in self.config.eval_data_paths:
            if os.path.exists(path):
                bucket = os.path.splitext(os.path.basename(path))[0]
                episodes = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            episodes.append(Data.from_json(line.strip()))
                self.eval_data[bucket] = episodes[:self.config.eval_limit]
                self.logger.info(f"Eval {bucket}: {len(self.eval_data[bucket])} episodes")

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 1: Generate rollouts with vLLM
    # ═══════════════════════════════════════════════════════════════

    def generate_rollouts_vllm(self, episodes: List[Data]) -> List[Dict]:
        """Generate rollouts for a batch of episodes using vLLM.
        
        All episodes are processed IN PARALLEL: at each step, we collect
        prompts from all active rollouts and call vLLM.generate() once.
        This gives ~Nx speedup where N = number of active rollouts.
        """
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.logger.info(f"Phase 1: Loading vLLM for {len(episodes)} episodes × {self.config.num_generations} rollouts...")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )

        # Load vLLM with LoRA if available
        lora_path = self.lora_path if os.path.exists(self.lora_path) else None

        if lora_path:
            llm = LLM(
                model=self.config.model_name,
                enable_lora=True,
                max_lora_rank=self.config.lora_r,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                max_model_len=self.config.vllm_max_model_len,
                trust_remote_code=True,
                enforce_eager=True,
                disable_custom_all_reduce=True,
                disable_log_stats=True,
                max_num_seqs=self.config.rollout_batch_size * 2,
            )
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("grpo_lora", 1, lora_path)
        else:
            llm = LLM(
                model=self.config.model_name,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                max_model_len=self.config.vllm_max_model_len,
                trust_remote_code=True,
                enforce_eager=True,
                disable_custom_all_reduce=True,
                disable_log_stats=True,
                max_num_seqs=self.config.rollout_batch_size * 2,
            )
            lora_request = None

        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=0.9,
        )

        # Initialize ALL rollouts at once
        # Each rollout is a dict tracking its state
        active_rollouts = []
        rollout_id = 0
        for ep in episodes:
            for g in range(self.config.num_generations):
                obs = self.env.reset(ep)
                active_rollouts.append({
                    "id": rollout_id,
                    "episode": ep,
                    "gen": g,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obs},
                    ],
                    "actions": [],
                    "observations": [obs],
                    "done": False,
                    "env_state": self.env._save_state() if hasattr(self.env, '_save_state') else None,
                })
                rollout_id += 1

        total_rollouts = len(active_rollouts)
        completed_rollouts = []

        self.logger.info(f"  {total_rollouts} rollouts initialized, starting parallel generation...")

        for step in range(self.config.max_steps_per_episode):
            # Collect prompts from all ACTIVE (not done) rollouts
            still_active = [r for r in active_rollouts if not r["done"]]
            if not still_active:
                break

            # Build prompts batch
            prompts = []
            for r in still_active:
                prompt = tokenizer.apply_chat_template(
                    r["messages"], tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)

            # ONE vLLM call for ALL active rollouts (progress_bar OFF)
            if lora_request:
                outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
            else:
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            # Process results
            for r, output in zip(still_active, outputs):
                action = output.outputs[0].text.strip()
                action = action.split("\n")[0].strip()
                if not action:
                    action = "FINAL_ANSWER I don't know"

                r["actions"].append(action)

                # Re-initialize env for this rollout and replay actions
                self.env.reset(r["episode"])
                obs = ""
                done = False
                for a in r["actions"]:
                    obs, reward, done, info = self.env.step(a)

                r["observations"].append(obs)
                r["messages"].append({"role": "assistant", "content": action})
                r["messages"].append({"role": "user", "content": obs})

                if done:
                    r["done"] = True

            # Mark remaining as done if this is the last step
            if step == self.config.max_steps_per_episode - 1:
                for r in still_active:
                    r["done"] = True

            n_active = sum(1 for r in active_rollouts if not r["done"])
            self.logger.info(
                f"  Step {step+1}/{self.config.max_steps_per_episode}: "
                f"generated {len(still_active)} actions, {n_active} still active"
            )

        # Verify all rollouts
        all_rollouts = []
        for r in active_rollouts:
            self.env.reset(r["episode"])
            for a in r["actions"]:
                self.env.step(a)
            metrics = self.verifier.verify_trajectory(self.env, r["episode"], r["actions"])

            all_rollouts.append({
                "episode": r["episode"],
                "actions": r["actions"],
                "observations": r["observations"],
                "reward": metrics["total_reward"],
                "metrics": metrics,
                "messages": r["messages"],
            })

        # Free vLLM aggressively
        del llm
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Give CUDA driver time to reclaim memory
        import time as _time
        _time.sleep(2)
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        self.logger.info(f"Phase 1 done: {len(all_rollouts)} rollouts, GPU free: {free_mem:.1f} GB")

        return all_rollouts

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 2: Compute loss and update with HF model
    # ═══════════════════════════════════════════════════════════════

    def train_on_rollouts(self, all_rollouts: List[Dict]) -> Dict[str, float]:
        """Load HF model, compute GRPO loss on pre-generated rollouts, update weights."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, PeftModel

        self.logger.info("Phase 2: Loading HF model for training...")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        if os.path.exists(self.lora_path):
            model = PeftModel.from_pretrained(model, self.lora_path, is_trainable=True)
            self.logger.info("Loaded existing LoRA weights")
        else:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            self.logger.info("Created new LoRA adapter")

        # Ensure LoRA params require grad
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        # Enable input embeddings grad (needed for loss.backward through model)
        model.enable_input_require_grads()
        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )

        # Group rollouts by episode
        episode_groups = {}
        for r in all_rollouts:
            qid = r["episode"].question_id
            episode_groups.setdefault(qid, []).append(r)

        total_loss = 0.0
        total_updates = 0
        episode_list = list(episode_groups.values())
        random.shuffle(episode_list)

        for batch_start in range(0, len(episode_list), self.config.train_batch_size):
            batch = episode_list[batch_start:batch_start + self.config.train_batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0

            for rollouts in batch:
                # Compute GRPO advantages
                rewards = [r["reward"] for r in rollouts]
                mean_r = sum(rewards) / len(rewards)
                std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
                advantages = [(r - mean_r) / std_r for r in rewards]
                
                # Detailed logging for the first question in the batch
                self.logger.info(f"  QID {rollouts[0]['episode'].question_id}: rewards={rewards}, mean={mean_r:.2f}, adv={['%.2f' % a for a in advantages]}")

                losses = []
                for idx, (rollout, advantage) in enumerate(zip(rollouts, advantages)):
                    self.logger.info(f"    Rollout {idx}: adv={advantage:.2f}, steps={len(rollout['actions'])}, succ={rollout['metrics']['success']}")
                    
                    if abs(advantage) < 0.01:
                        self.logger.info(f"      -> Skipping (neutral advantage)")
                        continue

                    for i, msg in enumerate(rollout["messages"]):
                        if msg["role"] != "assistant":
                            continue

                        context_msgs = rollout["messages"][:i + 1]
                        text = tokenizer.apply_chat_template(
                            context_msgs, tokenize=False, add_generation_prompt=False
                        )
                        inputs = tokenizer(
                            text, return_tensors="pt", truncation=True, max_length=1024
                        ).to(model.device)

                        with torch.enable_grad():
                            outputs = model(**inputs, labels=inputs["input_ids"])
                            if outputs.loss is not None and outputs.loss.grad_fn is not None:
                                step_loss = -advantage * (-outputs.loss)
                                losses.append(step_loss)
                                self.logger.info(f"      -> msg {i} loss={outputs.loss.item():.4f}, weighted={step_loss.item():.4f}")
                        
                        del inputs, outputs
                        torch.cuda.empty_cache()

                if losses:
                    loss = torch.stack(losses).mean() / len(batch)
                    if loss.grad_fn is not None:
                        loss.backward()
                        batch_loss += loss.item()
                    del losses, loss
                    torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            optimizer.step()
            total_loss += batch_loss
            total_updates += 1
            torch.cuda.empty_cache()

        # Save updated LoRA
        model.save_pretrained(self.lora_path)
        tokenizer.save_pretrained(self.lora_path)
        self.logger.info(f"LoRA saved to {self.lora_path}")

        # Free HF model
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

        # Compute aggregate metrics
        all_rewards = [r["reward"] for r in all_rollouts]
        all_success = [r["metrics"]["success"] for r in all_rollouts]

        return {
            "loss": total_loss / max(total_updates, 1),
            "mean_reward": sum(all_rewards) / len(all_rewards),
            "success_rate": sum(all_success) / len(all_success),
            "mean_steps": sum(r["metrics"]["steps"] for r in all_rollouts) / len(all_rollouts),
            "mean_tools": sum(r["metrics"]["tool_calls"] for r in all_rollouts) / len(all_rollouts),
            "num_rollouts": len(all_rollouts),
        }

    # ═══════════════════════════════════════════════════════════════
    #  EVAL (uses vLLM for speed)
    # ═══════════════════════════════════════════════════════════════

    def evaluate_vllm(self) -> Dict[str, Dict]:
        """Quick eval using vLLM."""
        eval_episodes = []
        for bucket, eps in self.eval_data.items():
            for ep in eps:
                eval_episodes.append((bucket, ep))

        if not eval_episodes:
            return {}

        # Generate rollouts (1 per episode)
        orig_gen = self.config.num_generations
        self.config.num_generations = 1
        rollouts = self.generate_rollouts_vllm([ep for _, ep in eval_episodes])
        self.config.num_generations = orig_gen

        # Aggregate by bucket
        results = {}
        bucket_metrics = {}
        for (bucket, ep), rollout in zip(eval_episodes, rollouts):
            bucket_metrics.setdefault(bucket, []).append(rollout["metrics"])

        for bucket, metrics_list in bucket_metrics.items():
            n = len(metrics_list)
            results[bucket] = {
                "success_rate": sum(m["success"] for m in metrics_list) / n,
                "mean_reward": sum(m["total_reward"] for m in metrics_list) / n,
                "mean_steps": sum(m["steps"] for m in metrics_list) / n,
            }
            self.logger.info(
                f"  {bucket}: success={results[bucket]['success_rate']:.1%}, "
                f"reward={results[bucket]['mean_reward']:.3f}"
            )

        return results

    # ═══════════════════════════════════════════════════════════════
    #  MAIN TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════

    def train(self):
        """Main training loop: alternating vLLM generation and HF training."""
        from tqdm.auto import tqdm
        self.load_data()
        all_logs = []
        rollout_batch_count = 0

        # Calculate total batches for progress bar
        total_batches = 0
        for epoch in range(self.config.num_epochs):
            max_diff = self.config.curriculum_schedule.get(epoch + 1, 10) if self.config.curriculum_schedule else 10
            epoch_data = [d for d in self.train_data if d.difficulty <= max_diff]
            total_batches += len(range(0, len(epoch_data), self.config.rollout_batch_size))

        pbar = tqdm(total=total_batches, desc='GRPO Training', unit='batch')

        for epoch in range(self.config.num_epochs):
            epoch_num = epoch + 1

            # Curriculum filter
            if self.config.curriculum_schedule:
                max_diff = self.config.curriculum_schedule.get(
                    epoch_num, max(self.config.curriculum_schedule.values())
                )
                epoch_data = [d for d in self.train_data if d.difficulty <= max_diff]
            else:
                epoch_data = list(self.train_data)

            random.shuffle(epoch_data)
            self.logger.info(
                f"\n{'='*60}\n"
                f"Epoch {epoch_num}/{self.config.num_epochs} "
                f"(d1-d{max_diff if self.config.curriculum_schedule else 10}, "
                f"{len(epoch_data)} episodes)\n{'='*60}"
            )

            # Process in rollout batches
            bs = self.config.rollout_batch_size
            for i in range(0, len(epoch_data), bs):
                batch_episodes = epoch_data[i:i + bs]
                rollout_batch_count += 1

                self.logger.info(
                    f"\n--- Rollout batch {rollout_batch_count}/{total_batches} "
                    f"({len(batch_episodes)} episodes) ---"
                )

                t0 = time.time()

                # Phase 1: Generate with vLLM
                rollouts = self.generate_rollouts_vllm(batch_episodes)

                t_gen = time.time() - t0

                # Phase 2: Train with HF
                metrics = self.train_on_rollouts(rollouts)

                t_total = time.time() - t0

                metrics["time_gen"] = t_gen
                metrics["time_total"] = t_total
                metrics["epoch"] = epoch_num
                metrics["rollout_batch"] = rollout_batch_count

                all_logs.append(metrics)
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                self.logger.info(
                    f"  loss={metrics['loss']:.4f}, reward={metrics['mean_reward']:.3f}, "
                    f"success={metrics['success_rate']:.1%}, "
                    f"gen={t_gen:.0f}s, total={t_total:.0f}s"
                )

                # Eval
                if rollout_batch_count % self.config.eval_every_rollout_batch == 0:
                    self.logger.info("\n--- Evaluation ---")
                    eval_results = self.evaluate_vllm()
                    with open(os.path.join(self.config.log_dir, "eval_log.jsonl"), "a") as f:
                        f.write(json.dumps({
                            "rollout_batch": rollout_batch_count, **eval_results
                        }) + "\n")

                # Save checkpoint
                if rollout_batch_count % self.config.save_every_rollout_batch == 0:
                    ckpt = os.path.join(self.config.output_dir, f"batch_{rollout_batch_count}")
                    os.makedirs(ckpt, exist_ok=True)
                    if os.path.exists(self.lora_path):
                        import shutil
                        shutil.copytree(self.lora_path, ckpt, dirs_exist_ok=True)
                    self.logger.info(f"Checkpoint: {ckpt}")

        # Final save
        final_path = os.path.join(self.config.output_dir, "final")
        if os.path.exists(self.lora_path):
            import shutil
            os.makedirs(final_path, exist_ok=True)
            shutil.copytree(self.lora_path, final_path, dirs_exist_ok=True)

        self.logger.info(f"\nTraining complete! Final model: {final_path}")
        self.logger.info(f"Logs: {self.log_file}")

        return all_logs
