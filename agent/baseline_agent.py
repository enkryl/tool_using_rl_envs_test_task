"""
Baseline agent using Qwen2.5-1.5B-Instruct with pure prompting (no fine-tuning).

This serves as the baseline for comparison with the GRPO-trained model.
Uses transformers for generation.
"""

import os
import sys
import json
import torch
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


class BaselineAgent:
    """
    Baseline agent that uses Qwen2.5-1.5B-Instruct with prompting.
    No fine-tuning — relies on the system prompt and few-shot examples.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Model loaded on {self.model.device}")

    def generate_action(self, messages: list) -> str:
        """Generate a single action given conversation history."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Take only the first line (avoid multi-action generation)
        lines = response.split("\n")
        action = lines[0].strip()

        return action if action else "FINAL_ANSWER I don't know"

    def run_episode(self, env, data, verbose: bool = False) -> dict:
        """
        Run a full episode: agent interacts with env until done or max_steps.

        Returns:
            dict with trajectory and metrics
        """
        obs = env.reset(data)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs},
        ]

        actions = []
        observations = []
        done = False
        step = 0

        while not done and step < data.max_steps:
            action = self.generate_action(messages)
            actions.append(action)

            if verbose:
                print(f"  Step {step + 1}: {action[:120]}")

            obs, reward, done, info = env.step(action)
            observations.append(obs)

            if verbose:
                print(f"    -> reward={reward:.3f}, done={done}")
                if obs:
                    print(f"    -> obs: {obs[:120]}")

            # Add to conversation
            messages.append({"role": "assistant", "content": action})
            messages.append({"role": "user", "content": obs})

            step += 1

        return {
            "question_id": data.question_id,
            "difficulty": data.difficulty,
            "task_type": data.task_type,
            "actions": actions,
            "observations": observations,
            "steps": step,
            "done": done,
        }
