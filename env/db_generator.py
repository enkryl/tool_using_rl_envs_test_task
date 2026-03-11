"""
Synthetic metro violations database generator.

Creates an in-memory SQLite database with deterministic data based on a seed.
Tables: lines, stations, devices, violation_types, violations, cases.
"""

import sqlite3
import random
from datetime import datetime, timedelta
from typing import Optional


# ─── Fixed reference data (same for all seeds) ─────────────────────

LINES = [
    (1, "Sokolnicheskaya", "red"),
    (2, "Zamoskvoretskaya", "green"),
    (3, "Arbatsko-Pokrovskaya", "blue"),
    (4, "Filyovskaya", "light_blue"),
    (5, "Koltsevaya", "brown"),
    (6, "Kaluzhsko-Rizhskaya", "orange"),
    (7, "Tagansko-Krasnopresnenskaya", "purple"),
    (8, "Kalininskaya", "yellow"),
]

STATIONS = [
    # (station_id, station_name, line_id, district)
    (1, "Sokolniki", 1, "East"),
    (2, "Krasnoselskaya", 1, "East"),
    (3, "Komsomolskaya_R", 1, "Central"),
    (4, "Lubyanka", 1, "Central"),
    (5, "Okhotny_Ryad", 1, "Central"),
    (6, "Park_Kultury_R", 1, "South"),
    (7, "Universitet", 1, "South"),
    (8, "Novokuznetskaya", 2, "Central"),
    (9, "Teatralnaya", 2, "Central"),
    (10, "Tverskaya", 2, "Central"),
    (11, "Belorusskaya_Z", 2, "North"),
    (12, "Dinamo", 2, "North"),
    (13, "Aeroport", 2, "North"),
    (14, "Arbatskaya", 3, "Central"),
    (15, "Smolenskaya_A", 3, "West"),
    (16, "Kievskaya_A", 3, "West"),
    (17, "Park_Pobedy", 3, "West"),
    (18, "Fili", 4, "West"),
    (19, "Kutuzovskaya", 4, "West"),
    (20, "Studencheskaya", 4, "West"),
    (21, "Komsomolskaya_K", 5, "Central"),
    (22, "Kurskaya_K", 5, "Central"),
    (23, "Taganskaya_K", 5, "Central"),
    (24, "Paveletskaya_K", 5, "Central"),
    (25, "Oktyabrskaya_K", 5, "South"),
    (26, "Rizhskaya", 6, "North"),
    (27, "Alekseevskaya", 6, "North"),
    (28, "VDNH", 6, "North"),
    (29, "Tushinskaya", 7, "North-West"),
    (30, "Polezhaevskaya", 7, "North-West"),
]

VIOLATION_TYPES = [
    # (violation_code, violation_name, severity, requires_review)
    ("FARE", "Fare evasion", "low", False),
    ("SMOK", "Smoking in metro area", "medium", True),
    ("VAND", "Vandalism", "high", True),
    ("SAFE", "Safety rule violation", "medium", False),
    ("DIST", "Public disturbance", "medium", True),
    ("TRES", "Trespassing restricted area", "high", True),
    ("LITT", "Littering", "low", False),
    ("FAKE", "Using fake ticket/pass", "high", True),
]

DEVICE_TYPES = ["turnstile", "camera", "sensor", "manual_report"]

SHIFT_CODES = ["morning", "day", "evening", "night"]


# ─── Difficulty → generation parameters ────────────────────────────

def get_gen_params(difficulty: int) -> dict:
    """
    Return generation parameters based on difficulty level (1–10).
    Higher difficulty → more data, longer time range.
    """
    params = {
        1:  {"num_stations": 10, "num_persons": 20,  "num_violations": (50, 100),   "num_days": 7},
        2:  {"num_stations": 10, "num_persons": 20,  "num_violations": (60, 120),   "num_days": 10},
        3:  {"num_stations": 12, "num_persons": 25,  "num_violations": (100, 200),  "num_days": 14},
        4:  {"num_stations": 15, "num_persons": 30,  "num_violations": (150, 300),  "num_days": 21},
        5:  {"num_stations": 15, "num_persons": 35,  "num_violations": (200, 400),  "num_days": 30},
        6:  {"num_stations": 18, "num_persons": 40,  "num_violations": (250, 450),  "num_days": 30},
        7:  {"num_stations": 20, "num_persons": 50,  "num_violations": (350, 600),  "num_days": 45},
        8:  {"num_stations": 22, "num_persons": 60,  "num_violations": (400, 700),  "num_days": 60},
        9:  {"num_stations": 25, "num_persons": 70,  "num_violations": (500, 900),  "num_days": 75},
        10: {"num_stations": 28, "num_persons": 80,  "num_violations": (700, 1200), "num_days": 90},
    }
    return params.get(difficulty, params[1])


def create_database(seed: int, difficulty: int = 1) -> sqlite3.Connection:
    """
    Create an in-memory SQLite database with synthetic metro violation data.
    
    Args:
        seed: Random seed for reproducible generation.
        difficulty: 1–10, controls data volume and complexity.
    
    Returns:
        sqlite3.Connection to the in-memory database.
    """
    rng = random.Random(seed)
    params = get_gen_params(difficulty)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # ── Create tables ──────────────────────────────────────────────

    cur.executescript("""
        CREATE TABLE lines (
            line_id   INTEGER PRIMARY KEY,
            line_name TEXT NOT NULL,
            color     TEXT NOT NULL
        );
        CREATE TABLE stations (
            station_id   INTEGER PRIMARY KEY,
            station_name TEXT NOT NULL,
            line_id      INTEGER NOT NULL REFERENCES lines(line_id),
            district     TEXT NOT NULL
        );
        CREATE TABLE devices (
            device_id   INTEGER PRIMARY KEY,
            station_id  INTEGER NOT NULL REFERENCES stations(station_id),
            device_type TEXT NOT NULL
        );
        CREATE TABLE violation_types (
            violation_code TEXT PRIMARY KEY,
            violation_name TEXT NOT NULL,
            severity       TEXT NOT NULL,
            requires_review INTEGER NOT NULL
        );
        CREATE TABLE violations (
            event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            event_ts       TEXT NOT NULL,
            person_id      TEXT NOT NULL,
            station_id     INTEGER NOT NULL REFERENCES stations(station_id),
            device_id      INTEGER NOT NULL REFERENCES devices(device_id),
            violation_code TEXT NOT NULL REFERENCES violation_types(violation_code),
            fine_amount    REAL NOT NULL,
            shift_code     TEXT NOT NULL
        );
        CREATE TABLE cases (
            case_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id  TEXT NOT NULL,
            reason     TEXT NOT NULL,
            status     TEXT NOT NULL DEFAULT 'open',
            opened_at  TEXT NOT NULL
        );
    """)

    # ── Populate reference tables ──────────────────────────────────

    for line in LINES:
        cur.execute("INSERT INTO lines VALUES (?, ?, ?)", line)

    active_stations = STATIONS[:params["num_stations"]]
    for st in active_stations:
        cur.execute("INSERT INTO stations VALUES (?, ?, ?, ?)", st)

    # Generate devices (2–4 per station)
    device_id = 1
    device_ids_by_station = {}
    for st in active_stations:
        n_devices = rng.randint(2, 4)
        device_ids_by_station[st[0]] = []
        for _ in range(n_devices):
            dtype = rng.choice(DEVICE_TYPES)
            cur.execute("INSERT INTO devices VALUES (?, ?, ?)", (device_id, st[0], dtype))
            device_ids_by_station[st[0]].append(device_id)
            device_id += 1

    for vt in VIOLATION_TYPES:
        cur.execute(
            "INSERT INTO violation_types VALUES (?, ?, ?, ?)",
            (vt[0], vt[1], vt[2], 1 if vt[3] else 0),
        )

    # ── Generate persons ───────────────────────────────────────────

    person_ids = [f"P{i}" for i in range(1, params["num_persons"] + 1)]

    # ── Generate violations ────────────────────────────────────────

    base_date = datetime(2026, 1, 1)
    num_violations = rng.randint(*params["num_violations"])
    fine_map = {"low": (100, 500), "medium": (500, 2000), "high": (2000, 10000)}

    for _ in range(num_violations):
        person = rng.choice(person_ids)
        station = rng.choice(active_stations)
        station_id = station[0]
        device = rng.choice(device_ids_by_station[station_id])
        vtype = rng.choice(VIOLATION_TYPES)
        violation_code = vtype[0]
        severity = vtype[2]

        day_offset = rng.randint(0, params["num_days"] - 1)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        ts = base_date + timedelta(days=day_offset, hours=hour, minutes=minute)

        if hour < 6:
            shift = "night"
        elif hour < 12:
            shift = "morning"
        elif hour < 18:
            shift = "day"
        else:
            shift = "evening"

        fine_lo, fine_hi = fine_map[severity]
        fine = round(rng.uniform(fine_lo, fine_hi), 2)

        cur.execute(
            "INSERT INTO violations (event_ts, person_id, station_id, device_id, "
            "violation_code, fine_amount, shift_code) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts.strftime("%Y-%m-%d %H:%M"), person, station_id, device, violation_code, fine, shift),
        )

    conn.commit()
    return conn


def get_all_person_ids(conn: sqlite3.Connection) -> list:
    """Return all distinct person_ids in the violations table."""
    cur = conn.execute("SELECT DISTINCT person_id FROM violations ORDER BY person_id")
    return [row[0] for row in cur.fetchall()]


def get_all_station_ids(conn: sqlite3.Connection) -> list:
    """Return all station_ids in the stations table."""
    cur = conn.execute("SELECT station_id FROM stations ORDER BY station_id")
    return [row[0] for row in cur.fetchall()]


def get_all_violation_codes(conn: sqlite3.Connection) -> list:
    """Return all violation_codes in the violation_types table."""
    cur = conn.execute("SELECT violation_code FROM violation_types ORDER BY violation_code")
    return [row[0] for row in cur.fetchall()]
