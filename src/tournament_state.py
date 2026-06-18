"""
Manage the actual match results entered by the user.

Results are persisted in actual_results.json at the project root.
Schema:
{
  "group_results": {
    "A": [
      {"home": "Mexico", "away": "South Korea", "home_score": 2, "away_score": 1}
    ],
    ...
  },
  "knockout_results": {
    "r32": [
      {"team1": "Spain", "team2": "Morocco",
       "home_score": 2, "away_score": 0, "winner": "Spain"}
    ],
    "r16": [...],
    "quarter": [...],
    "semi": [...],
    "final": [...]
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from itertools import combinations
from typing import Optional

from .fixtures import GROUPS

RESULTS_FILE = Path(__file__).parent.parent / "actual_results.json"

KNOCKOUT_ROUNDS = ["r32", "r16", "quarter", "semi", "final"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_results() -> dict:
    """Load actual results from disk, or return an empty state."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"group_results": {}, "knockout_results": {}}


def save_results(state: dict) -> None:
    RESULTS_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Group helpers
# ---------------------------------------------------------------------------

def all_group_pairs(teams: list[str]) -> list[tuple[str, str]]:
    """All 6 (home, away) pairs for a 4-team round-robin."""
    return list(combinations(teams, 2))


def group_played_matches(group_letter: str, state: dict) -> list[dict]:
    return state.get("group_results", {}).get(group_letter, [])


def group_is_complete(group_letter: str, state: dict, groups: dict = None) -> bool:
    if groups is None:
        groups = GROUPS
    played = group_played_matches(group_letter, state)
    return len(played) == 6


def group_unplayed_pairs(group_letter: str, state: dict,
                          groups: dict = None) -> list[tuple[str, str]]:
    """Return pairs not yet recorded in actual results."""
    if groups is None:
        groups = GROUPS
    teams = groups[group_letter]
    played = group_played_matches(group_letter, state)

    played_keys = set()
    for m in played:
        played_keys.add((m["home"], m["away"]))
        played_keys.add((m["away"], m["home"]))

    remaining = []
    for t1, t2 in all_group_pairs(teams):
        if (t1, t2) not in played_keys and (t2, t1) not in played_keys:
            remaining.append((t1, t2))
    return remaining


def compute_partial_standings(group_letter: str, state: dict,
                               groups: dict = None) -> dict[str, dict]:
    """
    Compute points / GF / GA from actual results so far.
    Returns {team: {"pts": int, "gf": int, "ga": int}} for all 4 teams.
    """
    if groups is None:
        groups = GROUPS
    teams = groups[group_letter]
    record = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}

    for m in group_played_matches(group_letter, state):
        h, a = m["home"], m["away"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        if h not in record or a not in record:
            continue
        record[h]["gf"] += hs; record[h]["ga"] += as_
        record[a]["gf"] += as_; record[a]["ga"] += hs
        if hs > as_:   record[h]["pts"] += 3
        elif as_ > hs: record[a]["pts"] += 3
        else:          record[h]["pts"] += 1; record[a]["pts"] += 1

    return record


# ---------------------------------------------------------------------------
# Knockout helpers
# ---------------------------------------------------------------------------

def knockout_winner(round_name: str, team1: str, team2: str,
                    state: dict) -> Optional[str]:
    """
    If this exact knockout match has an actual result, return the winner.
    Match is identified by the unordered pair (team1, team2).
    """
    for m in state.get("knockout_results", {}).get(round_name, []):
        pair = {m["team1"], m["team2"]}
        if pair == {team1, team2}:
            return m["winner"]
    return None


def all_knockout_results_flat(state: dict) -> list[dict]:
    """Flat list of all actual knockout matches for model retraining."""
    matches = []
    for round_name, round_matches in state.get("knockout_results", {}).items():
        matches.extend(round_matches)
    return matches


# ---------------------------------------------------------------------------
# Convert actual results → rows for model retraining
# ---------------------------------------------------------------------------

def actual_results_as_dataframe(state: dict):
    """
    Return a pandas DataFrame of all actual match results,
    formatted for appending to the preprocessing DataFrame.
    Actual matches get the current date (= maximum time-decay weight).
    Also given a tournament weight of 1.0 (World Cup).
    """
    import pandas as pd
    import numpy as np
    from datetime import date

    today = str(date.today())
    rows = []

    # Group matches
    for _letter, matches in state.get("group_results", {}).items():
        for m in matches:
            rows.append({
                "home_team":    m["home"],
                "away_team":    m["away"],
                "home_score":   int(m["home_score"]),
                "away_score":   int(m["away_score"]),
                "neutral_bool": True,
                "date":         today,
            })

    # Knockout matches
    for round_name, matches in state.get("knockout_results", {}).items():
        for m in matches:
            rows.append({
                "home_team":    m["team1"],
                "away_team":    m["team2"],
                "home_score":   int(m.get("team1_score", 0)),
                "away_score":   int(m.get("team2_score", 0)),
                "neutral_bool": True,
                "date":         today,
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["days_ago"] = 0
    df["time_weight"] = 1.0
    df["tournament_weight"] = 1.0
    df["weight"] = 3.0  # extra boost so actual WC results dominate
    return df[["home_team", "away_team", "home_score", "away_score",
               "neutral_bool", "weight"]]
