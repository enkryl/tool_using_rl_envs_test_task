"""
Reward function for GRPO training.

Wraps the metro violations environment to compute trajectory-level rewards.
"""

import os
import sys
import json
import re
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.data import Data
from env.metro_env import MetroViolationsEnv
from verifier.trajectory_verifier import MetroTrajectoryVerifier


def compute_reward(
    data: Data,
    actions: List[str],
    env: MetroViolationsEnv = None,
    verifier: MetroTrajectoryVerifier = None,
) -> Dict[str, Any]:
    """
    Compute the total reward for a trajectory.

    This is the function used by GRPO to score rollouts.
    Returns metrics dict with 'total_reward' as the main signal.
    """
    if env is None:
        env = MetroViolationsEnv()
    if verifier is None:
        verifier = MetroTrajectoryVerifier()

    metrics = verifier.verify_trajectory(env, data, actions)
    return metrics


def reward_fn_for_grpo(
    prompts: List[str],
    completions: List[str],
    datas: List[Data],
    **kwargs,
) -> List[float]:
    """
    Batch reward function compatible with GRPO trainer.

    Each completion is a multi-step trajectory serialized as:
        action1\n---ACTION---\naction2\n---ACTION---\n...

    Args:
        prompts: list of prompt strings (initial observations)
        completions: list of completion strings (serialized trajectories)
        datas: list of Data objects (one per prompt)

    Returns:
        list of float rewards
    """
    env = MetroViolationsEnv()
    verifier = MetroTrajectoryVerifier()
    rewards = []

    for prompt, completion, data in zip(prompts, completions, datas):
        # Parse completion into action list
        actions = parse_trajectory(completion)

        if not actions:
            rewards.append(-1.0)
            continue

        metrics = compute_reward(data, actions, env, verifier)
        rewards.append(metrics["total_reward"])

    return rewards


def parse_trajectory(text: str) -> List[str]:
    """
    Parse a serialized trajectory string into a list of actions.

    Format: actions separated by '---ACTION---' delimiter.
    """
    if not text or not text.strip():
        return []

    # Split by delimiter
    parts = re.split(r'---ACTION---', text)
    actions = [p.strip() for p in parts if p.strip()]

    return actions


def serialize_trajectory(actions: List[str]) -> str:
    """Serialize a list of actions into a single string for GRPO."""
    return "\n---ACTION---\n".join(actions)
