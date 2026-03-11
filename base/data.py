"""
Data dataclass for metro violations environment episodes.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json


@dataclass
class Data:
    """
    Represents a single episode in the metro violations environment.
    
    Stores the question, difficulty, DB seed for reproducible generation,
    and hidden expected_result for verification (not shown to the agent).
    """
    question_id: str                    # unique episode ID
    question_text: str                  # task text shown to the agent
    difficulty: int                     # 1–10
    task_type: str                      # "count", "list_stations", "list_names", "bool", "open_case", ...
    db_seed: int                        # seed for reproducible SQLite DB generation
    expected_result: Any                # int / list / bool / dict — for verification
    needs_confirmation: bool = False    # whether episode requires confirmation before mutation
    max_steps: int = 10                 # max steps for this episode
    metadata: dict = field(default_factory=dict)  # extra fields (person_id, date_range, etc.)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "Data":
        """Deserialize from JSON string."""
        d = json.loads(json_str)
        return cls(**d)

    def to_dict(self) -> dict:
        """Convert to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Data":
        """Create from dict."""
        return cls(**d)
