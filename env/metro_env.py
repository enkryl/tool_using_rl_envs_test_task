"""
Metro Violations Environment — multi-step ToolEnv implementation.

The agent analyzes metro violation data using SQL tools and must follow
policy rules (confirmation before state changes, no hallucinated entities).
"""

import json
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from base.data import Data
from base.tool_env import ToolEnv
from env.db_generator import create_database, get_all_person_ids, get_all_station_ids, get_all_violation_codes
from env.tools import TOOL_REGISTRY, execute_tool, get_tools_description


# ─── Constants ──────────────────────────────────────────────────────

TOOL_CALL_PREFIX = "TOOL_CALL"
FINAL_ANSWER_PREFIX = "FINAL_ANSWER"

CONFIRMATION_PATTERNS = [
    r"\bconfirm\b", r"\byes\b", r"\bproceed\b", r"\bapproved?\b",
    r"\bgo ahead\b", r"\bdo it\b", r"\bagree\b",
]

SYSTEM_INTRO = """You are a metro violations analyst. You have access to a database with metro violation records.

Your task: {question}

{tools}

Action format:
  - To use a tool: TOOL_CALL {{"name": "<tool_name>", "args": {{...}}}}
  - To give a free-text message: just type your message
  - To submit your final answer: FINAL_ANSWER <your answer>

Rules:
  1. Before any state-changing action (like open_case), you MUST first propose it in free-text and receive confirmation.
  2. Only use entity IDs (person_id, station_id, etc.) that you have seen in tool outputs or the task description.
  3. Be concise and efficient — minimize unnecessary tool calls.
"""

MAX_STEPS_DEFAULT = 10


class MetroViolationsEnv(ToolEnv):
    """
    Multi-step environment for metro violations analysis.
    """

    def __init__(self):
        super().__init__(name="metro_violations")
        self._conn: Optional[sqlite3.Connection] = None
        self._data: Optional[Data] = None
        self._steps: int = 0
        self._tool_calls: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._user_confirmed: bool = False
        self._confirmation_proposed: bool = False
        self._known_entities: Set[str] = set()
        self._history: List[Dict[str, Any]] = []
        self._previous_queries: Set[str] = set()
        self._got_schema: bool = False
        self._policy_violations: int = 0
        self._invalid_actions: int = 0

    # ── ToolEnv interface ──────────────────────────────────────────

    def reset(self, data: Data) -> str:
        """Initialize episode and return initial observation."""
        self._data = data
        self._conn = create_database(data.db_seed, data.difficulty)
        self._steps = 0
        self._tool_calls = 0
        self._total_reward = 0.0
        self._done = False
        self._user_confirmed = False
        self._confirmation_proposed = False
        self._history = []
        self._previous_queries = set()
        self._got_schema = False
        self._policy_violations = 0
        self._invalid_actions = 0

        # Initialize known entities from question text
        self._known_entities = set()
        self._extract_entities_from_text(data.question_text)

        # Also add entities from metadata if present
        if "person_id" in data.metadata:
            self._known_entities.add(str(data.metadata["person_id"]))
        if "station_id" in data.metadata:
            self._known_entities.add(str(data.metadata["station_id"]))
        if "violation_code" in data.metadata:
            self._known_entities.add(str(data.metadata["violation_code"]))

        tools_desc = get_tools_description()
        observation = SYSTEM_INTRO.format(
            question=data.question_text,
            tools=tools_desc,
        )

        return observation

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Process one agent action.

        Returns: (observation, reward, done, info)
        """
        if self._done:
            return "Episode already finished.", 0.0, True, {"already_done": True}

        self._steps += 1
        max_steps = self._data.max_steps if self._data else MAX_STEPS_DEFAULT

        info: Dict[str, Any] = {
            "step": self._steps,
            "is_tool_call": False,
            "is_final_answer": False,
            "policy_violation": False,
            "policy_violation_type": None,
            "invalid_action": False,
            "success": False,
        }

        action = action.strip()
        reward = -0.02  # step penalty

        # ── Check max steps ────────────────────────────────────────
        if self._steps >= max_steps:
            self._done = True
            reward += -1.0  # fail — ran out of steps
            info["success"] = False
            info["max_steps_reached"] = True
            self._total_reward += reward
            self._history.append({"action": action, "info": info})
            return "Maximum steps reached. Episode failed.", reward, True, info

        # ── Parse action type ──────────────────────────────────────
        if action.startswith(FINAL_ANSWER_PREFIX):
            obs, r, info = self._handle_final_answer(action, info)
            reward += r

        elif action.startswith(TOOL_CALL_PREFIX):
            obs, r, info = self._handle_tool_call(action, info)
            reward += r

        else:
            obs, r, info = self._handle_free_text(action, info)
            reward += r

        self._total_reward += reward
        self._history.append({"action": action, "observation": obs, "info": info})
        return obs, reward, self._done, info

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs,
    ) -> List[Data]:
        """Procedurally generate episodes. Delegated to episode_generator."""
        from env.episode_generator import generate_episodes
        return generate_episodes(
            num_of_questions=num_of_questions,
            max_attempts=max_attempts,
            difficulty=difficulty,
            **kwargs,
        )

    # ── Internal handlers ──────────────────────────────────────────

    def _handle_final_answer(
        self, action: str, info: dict
    ) -> Tuple[str, float, dict]:
        """Handle FINAL_ANSWER action."""
        info["is_final_answer"] = True
        self._done = True

        answer_text = action[len(FINAL_ANSWER_PREFIX):].strip()

        # Compare with expected result
        success = self._check_answer(answer_text)
        info["success"] = success

        if success:
            return "Correct! Episode completed successfully.", 1.0, info
        else:
            return f"Incorrect answer. Expected: {self._data.expected_result}", -1.0, info

    def _handle_tool_call(
        self, action: str, info: dict
    ) -> Tuple[str, float, dict]:
        """Handle TOOL_CALL action."""
        info["is_tool_call"] = True
        self._tool_calls += 1
        reward = -0.01  # tool call penalty

        # Parse JSON
        json_str = action[len(TOOL_CALL_PREFIX):].strip()
        try:
            call = json.loads(json_str)
        except json.JSONDecodeError as e:
            info["invalid_action"] = True
            self._invalid_actions += 1
            return f"Error: Invalid JSON in TOOL_CALL: {e}", -0.1, info

        tool_name = call.get("name", "")
        tool_args = call.get("args", {})

        # Check tool exists
        if tool_name not in TOOL_REGISTRY:
            info["invalid_action"] = True
            self._invalid_actions += 1
            return (
                f"Error: Unknown tool '{tool_name}'. "
                f"Available tools: {', '.join(TOOL_REGISTRY.keys())}",
                -0.1,
                info,
            )

        tool_info = TOOL_REGISTRY[tool_name]

        # ── Policy: confirmation rule ──────────────────────────────
        if tool_info["is_write"] and not self._user_confirmed:
            info["policy_violation"] = True
            info["policy_violation_type"] = "confirmation_required"
            self._policy_violations += 1
            reward += -0.3
            return (
                f"Policy violation: '{tool_name}' is a state-changing tool. "
                f"You must propose this action in free-text and receive confirmation first.",
                reward,
                info,
            )

        # ── Policy: no hallucinated entities ───────────────────────
        hallucination = self._check_hallucinated_entities(tool_name, tool_args)
        if hallucination:
            info["policy_violation"] = True
            info["policy_violation_type"] = "hallucinated_entity"
            self._policy_violations += 1
            reward += -0.2
            # Still execute the tool but mark the violation
            # (the tool might return an error naturally)

        # ── Check for repeated queries ─────────────────────────────
        if tool_name == "run_sql":
            query = tool_args.get("query", "").strip()
            normalized = " ".join(query.upper().split())
            if normalized in self._previous_queries:
                reward += -0.05
                info["repeated_query"] = True
            self._previous_queries.add(normalized)

        # ── Execute tool ───────────────────────────────────────────
        result = execute_tool(self._conn, tool_name, tool_args)

        # Update known entities from result
        self._extract_entities_from_text(result)

        # Bonus for first get_schema call
        if tool_name == "get_schema" and not self._got_schema:
            self._got_schema = True
            reward += 0.05

        return f"[{tool_name}] {result}", reward, info

    def _handle_free_text(
        self, action: str, info: dict
    ) -> Tuple[str, float, dict]:
        """Handle free-text message."""
        # Check if this is a confirmation (simulated user response)
        if self._confirmation_proposed:
            is_confirm = any(
                re.search(pat, action, re.IGNORECASE) for pat in CONFIRMATION_PATTERNS
            )
            if is_confirm:
                self._user_confirmed = True
                return "Confirmed. You may proceed with the action.", 0.0, info

        # Check if agent is proposing a state-changing action
        propose_patterns = [
            r"open.?case", r"shall I", r"should I", r"can I proceed",
            r"I will open", r"I would like to open", r"propose",
            r"I recommend opening", r"let me open",
        ]
        is_proposal = any(
            re.search(pat, action, re.IGNORECASE) for pat in propose_patterns
        )
        if is_proposal:
            self._confirmation_proposed = True
            return (
                "I understand your proposal. Please confirm by saying 'yes' or 'confirm' "
                "to proceed with the state-changing action.",
                0.0,
                info,
            )

        return "Noted. Continue with your analysis.", 0.0, info

    # ── Verification helpers ───────────────────────────────────────

    def _check_answer(self, answer_text: str) -> bool:
        """Check if the final answer matches expected_result."""
        expected = self._data.expected_result

        if expected is None:
            return False

        answer_text = answer_text.strip()

        # Boolean check
        if isinstance(expected, bool):
            answer_lower = answer_text.lower()
            if expected:
                return answer_lower in ("true", "yes", "1")
            else:
                return answer_lower in ("false", "no", "0", "none", "no data", "no results")

        # Integer check
        if isinstance(expected, int):
            try:
                # Try to extract number from answer
                numbers = re.findall(r'-?\d+', answer_text)
                if numbers:
                    return int(numbers[0]) == expected
            except (ValueError, IndexError):
                pass
            return False

        # Float check
        if isinstance(expected, float):
            try:
                numbers = re.findall(r'-?\d+\.?\d*', answer_text)
                if numbers:
                    return abs(float(numbers[0]) - expected) < 0.01
            except (ValueError, IndexError):
                pass
            return False

        # List check — all expected items must be present in answer
        if isinstance(expected, list):
            answer_upper = answer_text.upper()
            found = 0
            for item in expected:
                item_str = str(item).upper()
                if item_str in answer_upper:
                    found += 1
            # At least 80% of items must be found
            return found >= len(expected) * 0.8 if expected else True

        # Dict check (for open_case tasks)
        if isinstance(expected, dict):
            if expected.get("case_opened"):
                # Check that a case was actually opened
                cur = self._conn.execute(
                    "SELECT COUNT(*) FROM cases WHERE person_id = ?",
                    (expected.get("person_id", ""),),
                )
                count = cur.fetchone()[0]
                return count > 0
            return False

        # String check — fuzzy containment
        if isinstance(expected, str):
            return expected.upper() in answer_text.upper()

        return str(expected).upper() in answer_text.upper()

    def _check_hallucinated_entities(self, tool_name: str, args: dict) -> Optional[str]:
        """
        Check if tool call arguments contain entity IDs not seen before.
        Returns the hallucinated entity string if found, else None.
        
        Note: run_sql is checked more loosely since IDs can be embedded in SQL.
        """
        # Don't check tools with no entity args
        if tool_name in ("get_schema", "get_table_sample"):
            return None

        # For run_sql, extract entity-like patterns from query
        if tool_name == "run_sql":
            query = args.get("query", "")
            # Extract person IDs like P123
            person_refs = re.findall(r"\bP\d+\b", query, re.IGNORECASE)
            for ref in person_refs:
                if ref.upper() not in {e.upper() for e in self._known_entities}:
                    return ref
            return None

        # For other tools, check specific args
        entity_args = {
            "lookup_station": ["station_id"],
            "lookup_violation": ["violation_code"],
            "get_case": ["person_id"],
            "open_case": ["person_id"],
        }

        check_args = entity_args.get(tool_name, [])
        for arg_name in check_args:
            if arg_name in args:
                val = str(args[arg_name])
                if val not in self._known_entities and val.upper() not in {
                    e.upper() for e in self._known_entities
                }:
                    return val
        return None

    def _extract_entities_from_text(self, text: str):
        """Extract entity-like IDs from text and add to known_entities."""
        # Person IDs: P1, P12, P123
        for m in re.finditer(r"\bP\d+\b", text):
            self._known_entities.add(m.group())

        # Station IDs: numbers that could be station IDs (from context)
        # We add all numbers found in tool outputs as potential station IDs
        for m in re.finditer(r"\b\d+\b", text):
            self._known_entities.add(m.group())

        # Violation codes: FARE, SMOK, VAND, etc.
        for m in re.finditer(r"\b[A-Z]{4}\b", text):
            self._known_entities.add(m.group())

        # Station names
        for m in re.finditer(r"\b[A-Z][a-z]+(?:_[A-Z]?)?\b", text):
            self._known_entities.add(m.group())
