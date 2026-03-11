"""
Tool implementations for the metro violations environment.

Each tool is a function that takes a SQLite connection (and args)
and returns a text observation string.
"""

import sqlite3
import json
import re
from typing import Any, Dict, Optional


def get_schema(conn: sqlite3.Connection) -> str:
    """Return the schema of all tables in the database."""
    cur = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    rows = cur.fetchall()
    parts = []
    for name, sql in rows:
        parts.append(f"-- Table: {name}")
        parts.append(sql)
        parts.append("")
    return "\n".join(parts)


def get_table_sample(conn: sqlite3.Connection, table_name: str, limit: int = 5) -> str:
    """Return first N rows of a table as formatted text."""
    # Validate table name to prevent injection
    valid_tables = {"lines", "stations", "devices", "violation_types", "violations", "cases"}
    if table_name not in valid_tables:
        return f"Error: Unknown table '{table_name}'. Valid tables: {', '.join(sorted(valid_tables))}"

    limit = min(max(1, limit), 20)  # clamp 1..20
    cur = conn.execute(f"SELECT * FROM {table_name} LIMIT ?", (limit,))
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()

    if not rows:
        return f"Table '{table_name}' is empty."

    # Format as text table
    lines = [" | ".join(columns)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        lines.append(" | ".join(str(v) for v in row))
    lines.append(f"\n({len(rows)} rows shown, limit={limit})")
    return "\n".join(lines)


def run_sql(conn: sqlite3.Connection, query: str) -> str:
    """
    Execute a SELECT query and return results as text.
    Only SELECT statements are allowed.
    """
    # Basic safety: only allow SELECT
    stripped = query.strip().upper()
    if not stripped.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed. INSERT/UPDATE/DELETE/DROP are forbidden."

    # Block dangerous patterns
    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH"]
    # Check for these as standalone keywords (not part of column names)
    for kw in dangerous:
        if re.search(rf'\b{kw}\b', stripped):
            if kw != "CREATE":  # CREATE might appear in subqueries as table name
                return f"Error: {kw} operations are not allowed."

    try:
        cur = conn.execute(query)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()

        if not rows:
            return "Query returned 0 rows."

        if len(rows) > 100:
            rows = rows[:100]
            truncated = True
        else:
            truncated = False

        lines = [" | ".join(columns)]
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))

        result = "\n".join(lines)
        if truncated:
            result += "\n\n(Results truncated to 100 rows)"
        else:
            result += f"\n\n({len(rows)} rows)"
        return result

    except Exception as e:
        return f"SQL Error: {str(e)}"


def lookup_station(conn: sqlite3.Connection, station_id: int) -> str:
    """Return detailed information about a station."""
    cur = conn.execute(
        """
        SELECT s.station_id, s.station_name, s.district, l.line_name, l.color
        FROM stations s JOIN lines l ON s.line_id = l.line_id
        WHERE s.station_id = ?
        """,
        (station_id,),
    )
    row = cur.fetchone()
    if row is None:
        return f"Error: Station with id={station_id} not found."
    return (
        f"Station ID: {row[0]}\n"
        f"Name: {row[1]}\n"
        f"District: {row[2]}\n"
        f"Line: {row[3]} ({row[4]})"
    )


def lookup_violation(conn: sqlite3.Connection, violation_code: str) -> str:
    """Return detailed information about a violation type."""
    cur = conn.execute(
        "SELECT * FROM violation_types WHERE violation_code = ?",
        (violation_code,),
    )
    row = cur.fetchone()
    if row is None:
        return f"Error: Violation code '{violation_code}' not found."
    return (
        f"Code: {row[0]}\n"
        f"Name: {row[1]}\n"
        f"Severity: {row[2]}\n"
        f"Requires review: {'yes' if row[3] else 'no'}"
    )


def get_case(conn: sqlite3.Connection, person_id: str) -> str:
    """Return case information for a person."""
    cur = conn.execute(
        "SELECT * FROM cases WHERE person_id = ? ORDER BY opened_at DESC",
        (person_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return f"No cases found for person '{person_id}'."

    parts = []
    for row in rows:
        parts.append(
            f"Case #{row[0]}: person={row[1]}, reason='{row[2]}', "
            f"status={row[3]}, opened_at={row[4]}"
        )
    return "\n".join(parts)


def open_case(
    conn: sqlite3.Connection,
    person_id: str,
    reason: str,
    linked_event_count: int,
) -> str:
    """
    Open a new case for a person. This is a STATE-CHANGING action
    that requires prior confirmation from the user.
    """
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    conn.execute(
        "INSERT INTO cases (person_id, reason, status, opened_at) VALUES (?, ?, 'open', ?)",
        (person_id, reason, now),
    )
    conn.commit()

    case_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return (
        f"Case #{case_id} opened successfully.\n"
        f"Person: {person_id}\n"
        f"Reason: {reason}\n"
        f"Linked events: {linked_event_count}\n"
        f"Status: open"
    )


# ─── Tool registry ─────────────────────────────────────────────────

# Maps tool name → (function, list of arg names, is_write_tool)
TOOL_REGISTRY = {
    "get_schema": {
        "fn": get_schema,
        "args": [],
        "is_write": False,
        "description": "Returns the SQL schema of all tables in the database.",
    },
    "get_table_sample": {
        "fn": get_table_sample,
        "args": ["table_name", "limit"],
        "is_write": False,
        "description": "Returns first N rows of a table. Args: table_name (str), limit (int, default 5).",
    },
    "run_sql": {
        "fn": run_sql,
        "args": ["query"],
        "is_write": False,
        "description": "Executes a SELECT SQL query and returns results. Args: query (str). Only SELECT allowed.",
    },
    "lookup_station": {
        "fn": lookup_station,
        "args": ["station_id"],
        "is_write": False,
        "description": "Returns detailed info about a station. Args: station_id (int).",
    },
    "lookup_violation": {
        "fn": lookup_violation,
        "args": ["violation_code"],
        "is_write": False,
        "description": "Returns detailed info about a violation type. Args: violation_code (str).",
    },
    "get_case": {
        "fn": get_case,
        "args": ["person_id"],
        "is_write": False,
        "description": "Returns case records for a person. Args: person_id (str).",
    },
    "open_case": {
        "fn": open_case,
        "args": ["person_id", "reason", "linked_event_count"],
        "is_write": True,
        "description": (
            "Opens a new investigation case for a person. "
            "Args: person_id (str), reason (str), linked_event_count (int). "
            "WARNING: This changes state. Must be confirmed by user first."
        ),
    },
}


def get_tools_description() -> str:
    """Return a formatted description of all available tools."""
    lines = ["Available tools:"]
    for name, info in TOOL_REGISTRY.items():
        write_tag = " [STATE-CHANGING]" if info["is_write"] else ""
        lines.append(f"  - {name}{write_tag}: {info['description']}")
    return "\n".join(lines)


def execute_tool(conn: sqlite3.Connection, tool_name: str, args: dict) -> str:
    """Execute a tool by name with given arguments."""
    if tool_name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{tool_name}'. Use get_schema() to see available tools."

    tool_info = TOOL_REGISTRY[tool_name]
    fn = tool_info["fn"]

    try:
        if not tool_info["args"]:
            return fn(conn)
        else:
            return fn(conn, **args)
    except TypeError as e:
        return f"Error calling {tool_name}: {str(e)}"
    except Exception as e:
        return f"Error in {tool_name}: {str(e)}"
