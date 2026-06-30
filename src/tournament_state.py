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

# FIFA Annex C: maps frozenset of 8 qualifying 3rd-place group letters →
# {bracket_index: group_letter} assignments.
# Bracket indices refer to R32_BRACKET (0-based) after the 2026 tree reorder:
#   0=M74, 1=M77, 6=M81, 7=M82, 10=M79, 11=M80, 14=M85, 15=M87
# Add more combinations as needed; the backtracking solver is the fallback.
_ANNEX_C: dict[frozenset, dict[int, str]] = {
    frozenset("BDEFIJKL"): {
        0:  "D",   # M74: 1E vs 3D  (Germany vs Paraguay)
        1:  "F",   # M77: 1I vs 3F  (France vs Sweden)
        6:  "B",   # M81: 1D vs 3B  (USA vs Bosnia)
        7:  "I",   # M82: 1G vs 3I  (Belgium vs Senegal)
        10: "E",   # M79: 1A vs 3E  (Mexico vs Ecuador)
        11: "K",   # M80: 1L vs 3K  (England vs DR Congo)
        14: "J",   # M85: 1B vs 3J  (Switzerland vs Algeria)
        15: "L",   # M87: 1K vs 3L  (Colombia vs Ghana)
    },
}


def _solve_third_place(bracket, qualified_letters: list[str]) -> dict[int, str]:
    """
    Assign each 3rd-place group letter to its correct bracket slot using the
    FIFA Annex C table, falling back to a backtracking solver for unknown combos.
    Returns {bracket_index: group_letter}.
    """
    key = frozenset(qualified_letters)
    if key in _ANNEX_C:
        return _ANNEX_C[key]

    # Backtracking fallback for combinations not yet in _ANNEX_C
    slots = [(i, set(spec[1:])) for i, (_, spec) in enumerate(bracket) if spec.startswith("3")]
    result: dict[int, str] = {}

    def bt(si: int, remaining: list[str]) -> bool:
        if si == len(slots):
            return not remaining
        idx, valid = slots[si]
        for g in remaining:
            if g in valid:
                result[idx] = g
                if bt(si + 1, [x for x in remaining if x != g]):
                    return True
                del result[idx]
        return False

    bt(0, sorted(qualified_letters))
    return result


def build_r32_from_standings(all_standings: dict[str, list[dict]],
                              letters: list[str]) -> list[dict]:
    """
    Build the official R32 fixture list from group standings.
    all_standings: {letter: [1st_dict, 2nd_dict, 3rd_dict, 4th_dict]}
    Returns [{"t1": str, "t2": str, "label": str}, …] in bracket order.
    """
    from .fixtures import R32_BRACKET

    thirds = sorted(
        ((l, all_standings[l][2]) for l in letters),
        key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]),
        reverse=True,
    )
    qualified_letters = [l for l, _ in thirds[:8]]
    third_assignment = _solve_third_place(R32_BRACKET, qualified_letters)

    def resolve(spec: str) -> str:
        rank_idx = {"1": 0, "2": 1}[spec[0]]
        return all_standings[spec[1]][rank_idx]["team"]

    matches = []
    for i, (p1, p2) in enumerate(R32_BRACKET):
        t1 = resolve(p1)
        if p2.startswith("3"):
            g = third_assignment[i]
            t2 = all_standings[g][2]["team"]
            label = f"{p1} v 3{g}"
        else:
            t2 = resolve(p2)
            label = f"{p1} v {p2}"
        matches.append({"t1": t1, "t2": t2, "label": label})
    return matches


def _sort_standings(standings: list[dict], mresults: dict) -> list[dict]:
    """Sort group standings by pts → GD → GF → H2H pts → H2H GD."""
    standings.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    result = []
    i = 0
    while i < len(standings):
        j = i + 1
        while j < len(standings) and (
            standings[j]["pts"], standings[j]["gd"], standings[j]["gf"]
        ) == (standings[i]["pts"], standings[i]["gd"], standings[i]["gf"]):
            j += 1
        tied = standings[i:j]
        if len(tied) > 1:
            names = [t["team"] for t in tied]
            hth_pts = {n: 0 for n in names}
            hth_gd  = {n: 0 for n in names}
            for a, b in combinations(names, 2):
                if (a, b) in mresults:
                    sa, sb = mresults[(a, b)]
                elif (b, a) in mresults:
                    sb, sa = mresults[(b, a)]
                else:
                    continue
                hth_gd[a] += sa - sb; hth_gd[b] += sb - sa
                if sa > sb:   hth_pts[a] += 3
                elif sb > sa: hth_pts[b] += 3
                else:         hth_pts[a] += 1; hth_pts[b] += 1
            tied.sort(key=lambda x: (hth_pts[x["team"]], hth_gd[x["team"]]), reverse=True)
        result.extend(tied)
        i = j
    return result


def compute_group_final_standings(group_letter: str, state: dict,
                                   groups: dict = None) -> list[dict] | None:
    """
    Return sorted standings [{team, pts, gd, gf, ga}, …] 1st→4th
    for a completed group. Returns None if the group is not yet complete.
    """
    if groups is None:
        groups = GROUPS
    if not group_is_complete(group_letter, state, groups):
        return None
    teams = groups[group_letter]
    matches = group_played_matches(group_letter, state)

    record = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    mresults: dict[tuple, tuple] = {}
    for m in matches:
        h, a = m["home"], m["away"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        mresults[(h, a)] = (hs, as_)
        record[h]["gf"] += hs; record[h]["ga"] += as_
        record[a]["gf"] += as_; record[a]["ga"] += hs
        if hs > as_:   record[h]["pts"] += 3
        elif as_ > hs: record[a]["pts"] += 3
        else:          record[h]["pts"] += 1; record[a]["pts"] += 1

    standings = [
        {"team": t, "pts": r["pts"], "gf": r["gf"], "ga": r["ga"],
         "gd": r["gf"] - r["ga"]}
        for t, r in record.items()
    ]
    return _sort_standings(standings, mresults)


def compute_round_fixtures(round_name: str, state: dict,
                            groups: dict = None) -> list[dict] | None:
    """
    Return the actual fixtures for a knockout round as [{"t1", "t2", "label"}, …],
    derived from group standings and previous knockout results.
    Returns None if the required prior results are not yet complete.
    """
    if groups is None:
        groups = GROUPS

    if round_name == "r32":
        all_standings: dict[str, list[dict]] = {}
        for letter in groups:
            s = compute_group_final_standings(letter, state, groups)
            if s is None:
                return None
            all_standings[letter] = s
        return build_r32_from_standings(all_standings, list(groups.keys()))

    prev = {"r16": "r32", "quarter": "r16", "semi": "quarter", "final": "semi"}[round_name]
    prev_fixtures = compute_round_fixtures(prev, state, groups)
    if prev_fixtures is None:
        return None

    prev_results = state.get("knockout_results", {}).get(prev, [])
    if len(prev_results) < len(prev_fixtures):
        return None  # previous round not fully entered

    result_map = {
        frozenset({m["team1"], m["team2"]}): m["winner"]
        for m in prev_results
    }
    winners = []
    for f in prev_fixtures:
        key = frozenset({f["t1"], f["t2"]})
        if key not in result_map:
            return None
        winners.append(result_map[key])

    return [
        {"t1": winners[i], "t2": winners[i + 1], "label": f"Match {i // 2 + 1}"}
        for i in range(0, len(winners), 2)
    ]


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
