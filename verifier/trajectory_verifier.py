"""
Trajectory verifier for the metro violations environment.

Replays a given action sequence through the environment and collects metrics.
"""

from typing import Any, Dict, List, Optional

from base.data import Data
from base.verifier import TrajectoryVerifier


class MetroTrajectoryVerifier(TrajectoryVerifier):
    """
    Verifier that evaluates a given action trajectory in MetroViolationsEnv.
    Does NOT call the agent — only replays provided actions.
    """

    def verify_trajectory(
        self,
        env,
        data: Data,
        actions: List[str],
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Replay the action list through env and collect metrics.

        Args:
            env: MetroViolationsEnv instance
            data: episode Data
            actions: list of agent action strings
            max_steps: override max steps (default: from data)

        Returns dict with keys:
            success, total_reward, steps, tool_calls, policy_violations,
            terminated_early, invalid_actions, info_trace
        """
        max_steps = max_steps or data.max_steps

        obs = env.reset(data)
        total_reward = 0.0
        tool_calls = 0
        policy_violations = 0
        invalid_actions = 0
        info_trace = []
        done = False
        steps_taken = 0
        last_info = {}

        for i, action in enumerate(actions):
            if done:
                break
            if max_steps and i >= max_steps:
                break

            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps_taken += 1

            if info.get("is_tool_call"):
                tool_calls += 1
            if info.get("policy_violation"):
                policy_violations += 1
            if info.get("invalid_action"):
                invalid_actions += 1

            info_trace.append({
                "step": i + 1,
                "action": action[:200],  # truncate for logging
                "observation": obs[:200],
                "reward": reward,
                "done": done,
                **{k: v for k, v in info.items()},
            })

            last_info = info

        success = last_info.get("success", False) if last_info else False

        return {
            "success": success,
            "total_reward": round(total_reward, 4),
            "steps": steps_taken,
            "tool_calls": tool_calls,
            "policy_violations": policy_violations,
            "terminated_early": not done,
            "invalid_actions": invalid_actions,
            "info_trace": info_trace,
        }
