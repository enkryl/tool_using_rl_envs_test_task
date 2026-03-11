"""
Episode generator for the metro violations environment.

Generates Data objects with questions, expected results, and DB seeds
for all 10 difficulty levels.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from base.data import Data
from env.db_generator import (
    create_database,
    get_all_person_ids,
    get_all_station_ids,
    get_all_violation_codes,
    get_gen_params,
    VIOLATION_TYPES,
    STATIONS,
    LINES,
)


# ─── Max steps by difficulty ───────────────────────────────────────

MAX_STEPS_BY_DIFFICULTY = {
    1: 5, 2: 6, 3: 7, 4: 8, 5: 10,
    6: 12, 7: 12, 8: 12, 9: 15, 10: 20,
}

# ─── Difficulty → task type mapping ────────────────────────────────

TASK_TYPES_BY_DIFFICULTY = {
    1: ["count_simple"],
    2: ["list_date_range"],
    3: ["list_with_names"],
    4: ["top_k", "aggregation"],
    5: ["multi_condition"],
    6: ["multi_step_explore"],
    7: ["ambiguous_query"],
    8: ["empty_result"],
    9: ["cross_entity"],
    10: ["full_case_workflow"],
}


def generate_episodes(
    num_of_questions: int = 100,
    max_attempts: int = 100,
    difficulty: Optional[int] = 1,
    **kwargs,
) -> List[Data]:
    """
    Generate episodes for a given difficulty level.

    Args:
        num_of_questions: number of episodes to generate
        max_attempts: max attempts per episode
        difficulty: 1–10
        **kwargs: override generation parameters
    """
    difficulty = difficulty or 1
    difficulty = max(1, min(10, difficulty))

    rng = random.Random(kwargs.get("seed", None))
    episodes = []

    task_types = TASK_TYPES_BY_DIFFICULTY[difficulty]

    for i in range(num_of_questions):
        for attempt in range(max_attempts):
            db_seed = rng.randint(1, 10_000_000)
            task_type = rng.choice(task_types)

            try:
                data = _generate_one_episode(
                    db_seed=db_seed,
                    difficulty=difficulty,
                    task_type=task_type,
                    rng=rng,
                )
                if data is not None:
                    episodes.append(data)
                    break
            except Exception:
                continue
        # If all attempts fail, skip this episode silently

    return episodes


def _generate_one_episode(
    db_seed: int,
    difficulty: int,
    task_type: str,
    rng: random.Random,
) -> Optional[Data]:
    """Generate a single episode. Returns None if invalid."""

    conn = create_database(db_seed, difficulty)
    person_ids = get_all_person_ids(conn)
    station_ids = get_all_station_ids(conn)
    violation_codes = get_all_violation_codes(conn)
    params = get_gen_params(difficulty)

    if not person_ids:
        conn.close()
        return None

    generators = {
        "count_simple": _gen_count_simple,
        "list_date_range": _gen_list_date_range,
        "list_with_names": _gen_list_with_names,
        "top_k": _gen_top_k,
        "aggregation": _gen_aggregation,
        "multi_condition": _gen_multi_condition,
        "multi_step_explore": _gen_multi_step_explore,
        "ambiguous_query": _gen_ambiguous_query,
        "empty_result": _gen_empty_result,
        "cross_entity": _gen_cross_entity,
        "full_case_workflow": _gen_full_case_workflow,
    }

    gen_fn = generators[task_type]
    result = gen_fn(conn, person_ids, station_ids, violation_codes, params, rng)
    conn.close()

    if result is None:
        return None

    question_text, expected_result, metadata, needs_confirmation = result

    return Data(
        question_id=str(uuid.uuid4())[:8],
        question_text=question_text,
        difficulty=difficulty,
        task_type=task_type,
        db_seed=db_seed,
        expected_result=expected_result,
        needs_confirmation=needs_confirmation,
        max_steps=MAX_STEPS_BY_DIFFICULTY[difficulty],
        metadata=metadata,
    )


# ─── d1: Simple count with one filter ──────────────────────────────

def _gen_count_simple(conn, person_ids, station_ids, violation_codes, params, rng):
    person = rng.choice(person_ids)
    cur = conn.execute(
        "SELECT COUNT(*) FROM violations WHERE person_id = ?", (person,)
    )
    count = cur.fetchone()[0]
    if count == 0:
        return None

    question = f"How many violations does person {person} have in total?"
    return question, count, {"person_id": person}, False


# ─── d2: List with date range ──────────────────────────────────────

def _gen_list_date_range(conn, person_ids, station_ids, violation_codes, params, rng):
    person = rng.choice(person_ids)
    base = datetime(2026, 1, 1)
    start_offset = rng.randint(0, max(0, params["num_days"] - 7))
    start_date = base + timedelta(days=start_offset)
    end_date = start_date + timedelta(days=rng.randint(3, 7))

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    cur = conn.execute(
        """
        SELECT DISTINCT station_id FROM violations
        WHERE person_id = ? AND date(event_ts) BETWEEN ? AND ?
        ORDER BY station_id
        """,
        (person, start_str, end_str),
    )
    stations = [row[0] for row in cur.fetchall()]
    if not stations:
        return None

    question = (
        f"Which stations (by station_id) did person {person} visit "
        f"between {start_str} and {end_str}?"
    )
    return question, stations, {"person_id": person, "date_range": [start_str, end_str]}, False


# ─── d3: List with station names (join) ────────────────────────────

def _gen_list_with_names(conn, person_ids, station_ids, violation_codes, params, rng):
    person = rng.choice(person_ids)
    cur = conn.execute(
        """
        SELECT DISTINCT s.station_name
        FROM violations v JOIN stations s ON v.station_id = s.station_id
        WHERE v.person_id = ?
        ORDER BY s.station_name
        """,
        (person,),
    )
    names = [row[0] for row in cur.fetchall()]
    if not names:
        return None

    question = (
        f"List the names of all stations where person {person} had violations. "
        f"Return station names, not IDs."
    )
    return question, names, {"person_id": person}, False


# ─── d4: Top-K aggregation ─────────────────────────────────────────

def _gen_top_k(conn, person_ids, station_ids, violation_codes, params, rng):
    person = rng.choice(person_ids)
    k = rng.choice([3, 5])

    cur = conn.execute(
        """
        SELECT s.station_name, COUNT(*) as cnt
        FROM violations v JOIN stations s ON v.station_id = s.station_id
        WHERE v.person_id = ?
        GROUP BY s.station_name
        ORDER BY cnt DESC
        LIMIT ?
        """,
        (person, k),
    )
    rows = cur.fetchall()
    if len(rows) < 2:
        return None

    names = [row[0] for row in rows]
    question = (
        f"What are the top {k} stations (by name) with the most violations "
        f"for person {person}? List them from most to least."
    )
    return question, names, {"person_id": person, "top_k": k}, False


def _gen_aggregation(conn, person_ids, station_ids, violation_codes, params, rng):
    person = rng.choice(person_ids)
    cur = conn.execute(
        "SELECT SUM(fine_amount) FROM violations WHERE person_id = ?",
        (person,),
    )
    total = cur.fetchone()[0]
    if total is None or total == 0:
        return None

    total = round(total, 2)
    question = (
        f"What is the total fine amount for person {person} across all violations?"
    )
    return question, total, {"person_id": person}, False


# ─── d5: Multi-condition query ─────────────────────────────────────

def _gen_multi_condition(conn, person_ids, station_ids, violation_codes, params, rng):
    person = rng.choice(person_ids)
    vcode = rng.choice(violation_codes)

    # Find a line that this person has visited
    cur = conn.execute(
        """
        SELECT DISTINCT l.color
        FROM violations v
        JOIN stations s ON v.station_id = s.station_id
        JOIN lines l ON s.line_id = l.line_id
        WHERE v.person_id = ?
        """,
        (person,),
    )
    colors = [row[0] for row in cur.fetchall()]
    if not colors:
        return None
    color = rng.choice(colors)

    # Count violations matching all conditions
    cur = conn.execute(
        """
        SELECT COUNT(*) FROM violations v
        JOIN stations s ON v.station_id = s.station_id
        JOIN lines l ON s.line_id = l.line_id
        WHERE v.person_id = ?
          AND v.violation_code = ?
          AND l.color = ?
          AND CAST(strftime('%w', v.event_ts) AS INTEGER) BETWEEN 1 AND 5
        """,
        (person, vcode, color),
    )
    count = cur.fetchone()[0]
    if count == 0:
        return None

    question = (
        f"How many violations of type '{vcode}' does person {person} have "
        f"on weekdays on the {color} line?"
    )
    return (
        question,
        count,
        {"person_id": person, "violation_code": vcode, "line_color": color},
        False,
    )


# ─── d6: Multi-step exploration ────────────────────────────────────

def _gen_multi_step_explore(conn, person_ids, station_ids, violation_codes, params, rng):
    station = rng.choice(station_ids)

    cur = conn.execute(
        """
        SELECT COUNT(DISTINCT device_id) FROM violations
        WHERE station_id = ?
        """,
        (station,),
    )
    count = cur.fetchone()[0]
    if count == 0:
        return None

    question = (
        f"How many unique devices have recorded violations at station "
        f"with station_id={station}? "
        f"You may need to explore the schema first to understand the data structure."
    )
    return question, count, {"station_id": station}, False


# ─── d7: Ambiguous query ──────────────────────────────────────────

def _gen_ambiguous_query(conn, person_ids, station_ids, violation_codes, params, rng):
    person = rng.choice(person_ids)

    # "Start of the month" = days 1–10
    cur = conn.execute(
        """
        SELECT s.station_name, COUNT(*) as cnt
        FROM violations v JOIN stations s ON v.station_id = s.station_id
        WHERE v.person_id = ?
          AND CAST(strftime('%d', v.event_ts) AS INTEGER) BETWEEN 1 AND 10
        GROUP BY s.station_name
        ORDER BY cnt DESC
        LIMIT 1
        """,
        (person,),
    )
    row = cur.fetchone()
    if row is None:
        return None

    question = (
        f"Where did person {person} appear most frequently at the start of the month? "
        f"Note: 'start of the month' means days 1 through 10. "
        f"Return the station name."
    )
    return question, row[0], {"person_id": person}, False


# ─── d8: Empty/null result ─────────────────────────────────────────

def _gen_empty_result(conn, person_ids, station_ids, violation_codes, params, rng):
    person = rng.choice(person_ids)
    vcode = rng.choice(violation_codes)
    station = rng.choice(station_ids)

    cur = conn.execute(
        """
        SELECT COUNT(*) FROM violations
        WHERE person_id = ? AND violation_code = ? AND station_id = ?
          AND shift_code = 'night'
        """,
        (person, vcode, station),
    )
    count = cur.fetchone()[0]

    # We want count == 0 for empty result task
    if count != 0:
        return None

    station_name = conn.execute(
        "SELECT station_name FROM stations WHERE station_id = ?", (station,)
    ).fetchone()[0]

    question = (
        f"Did person {person} have any '{vcode}' violations at station {station_name} "
        f"(station_id={station}) during the night shift? Answer 'yes' or 'no'."
    )
    return (
        question,
        False,
        {"person_id": person, "violation_code": vcode, "station_id": station},
        False,
    )


# ─── d9: Cross-entity comparison ──────────────────────────────────

def _gen_cross_entity(conn, person_ids, station_ids, violation_codes, params, rng):
    if len(person_ids) < 2:
        return None

    p1, p2 = rng.sample(person_ids, 2)

    cur = conn.execute(
        """
        SELECT DISTINCT s.station_name
        FROM violations v1
        JOIN violations v2 ON v1.station_id = v2.station_id
            AND date(v1.event_ts) = date(v2.event_ts)
        JOIN stations s ON v1.station_id = s.station_id
        WHERE v1.person_id = ? AND v2.person_id = ?
        ORDER BY s.station_name
        """,
        (p1, p2),
    )
    names = [row[0] for row in cur.fetchall()]
    if not names:
        return None

    question = (
        f"At which stations (by name) were both {p1} and {p2} present "
        f"on the same day? List all matching station names."
    )
    return question, names, {"person_id_1": p1, "person_id_2": p2}, False


# ─── d10: Full case workflow ──────────────────────────────────────

def _gen_full_case_workflow(conn, person_ids, station_ids, violation_codes, params, rng):
    # Find a person with enough violations
    for person in rng.sample(person_ids, min(len(person_ids), 20)):
        cur = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE person_id = ?", (person,)
        )
        count = cur.fetchone()[0]
        if count >= 5:
            threshold = rng.choice([3, 5])
            if count > threshold:
                question = (
                    f"Investigate person {person}: "
                    f"1) Find how many violations they have. "
                    f"2) If they have more than {threshold} violations, "
                    f"propose opening a case with reason "
                    f"'Multiple violations detected' and wait for confirmation. "
                    f"3) After confirmation, open the case. "
                    f"4) Submit FINAL_ANSWER with the total violation count."
                )
                expected = {
                    "case_opened": True,
                    "person_id": person,
                    "violation_count": count,
                }
                return (
                    question,
                    expected,
                    {"person_id": person, "threshold": threshold},
                    True,
                )
    return None
