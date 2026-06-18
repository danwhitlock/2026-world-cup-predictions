"""Full 2026 World Cup tournament simulator (Monte Carlo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple

from .fixtures import GROUPS, ALL_TEAMS, DISPLAY_NAMES


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------

def _group_standings(teams: List[str], model, rng: np.random.Generator,
                     actual_matches: list = None) -> Tuple[List[dict], dict]:
    """
    Simulate a round-robin group.  Returns (standings, match_results).
    standings: list of 4 dicts {team, pts, gf, ga, gd}, ordered 1st→4th.
    match_results: {(t1, t2): (s1, s2)}

    actual_matches: list of already-played match dicts
      {"home": str, "away": str, "home_score": int, "away_score": int}
      These are used as-is; only unplayed pairs are simulated.
    """
    record = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    mresults: dict[tuple, tuple] = {}

    # Seed with actual results
    actual_played: set[frozenset] = set()
    for m in (actual_matches or []):
        h, a = m["home"], m["away"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        mresults[(h, a)] = (hs, as_)
        actual_played.add(frozenset({h, a}))
        record[h]["gf"] += hs; record[h]["ga"] += as_
        record[a]["gf"] += as_; record[a]["ga"] += hs
        if hs > as_:    record[h]["pts"] += 3
        elif as_ > hs:  record[a]["pts"] += 3
        else:           record[h]["pts"] += 1; record[a]["pts"] += 1

    # Simulate remaining pairs
    for t1, t2 in combinations(teams, 2):
        if frozenset({t1, t2}) in actual_played:
            continue
        s1, s2 = model.sample_score(t1, t2, neutral=True, rng=rng)
        mresults[(t1, t2)] = (s1, s2)
        record[t1]["gf"] += s1;  record[t1]["ga"] += s2
        record[t2]["gf"] += s2;  record[t2]["ga"] += s1
        if s1 > s2:
            record[t1]["pts"] += 3
        elif s2 > s1:
            record[t2]["pts"] += 3
        else:
            record[t1]["pts"] += 1
            record[t2]["pts"] += 1

    standings = [
        {"team": t, "pts": r["pts"],
         "gf": r["gf"], "ga": r["ga"], "gd": r["gf"] - r["ga"]}
        for t, r in record.items()
    ]
    standings = _sort_group(standings, mresults, rng)
    return standings, mresults


def _sort_group(standings: List[dict], mresults: dict,
                rng: np.random.Generator) -> List[dict]:
    """Sort by FIFA tiebreaker rules (pts → GD → GF → H2H → random)."""

    def sort_key(x):
        return (x["pts"], x["gd"], x["gf"])

    standings.sort(key=sort_key, reverse=True)

    result: List[dict] = []
    i = 0
    while i < len(standings):
        j = i + 1
        while j < len(standings) and sort_key(standings[j]) == sort_key(standings[i]):
            j += 1
        tied = standings[i:j]
        if len(tied) > 1:
            tied = _break_tie(tied, mresults, rng)
        result.extend(tied)
        i = j
    return result


def _break_tie(tied: List[dict], mresults: dict,
               rng: np.random.Generator) -> List[dict]:
    """Break a tie using head-to-head points/GD, then random."""
    names = [t["team"] for t in tied]
    hth_pts = {n: 0 for n in names}
    hth_gd  = {n: 0 for n in names}

    for a, b in combinations(names, 2):
        key = (a, b) if (a, b) in mresults else (b, a)
        if key not in mresults:
            continue
        sa, sb = mresults[key] if key == (a, b) else reversed(mresults[key])
        hth_gd[a] += sa - sb
        hth_gd[b] += sb - sa
        if sa > sb:   hth_pts[a] += 3
        elif sb > sa: hth_pts[b] += 3
        else:         hth_pts[a] += 1; hth_pts[b] += 1

    tied.sort(key=lambda x: (hth_pts[x["team"]], hth_gd[x["team"]]), reverse=True)

    # Any remaining exact tie → random
    _shuffle_exact_ties(tied, lambda x: (hth_pts[x["team"]], hth_gd[x["team"]]), rng)
    return tied


def _shuffle_exact_ties(lst: List[dict], key_fn, rng: np.random.Generator):
    i = 0
    while i < len(lst):
        j = i + 1
        while j < len(lst) and key_fn(lst[j]) == key_fn(lst[i]):
            j += 1
        if j - i > 1:
            chunk = lst[i:j]
            rng.shuffle(chunk)
            lst[i:j] = chunk
        i = j


# ---------------------------------------------------------------------------
# Knockout rounds
# ---------------------------------------------------------------------------

def _knockout_match(t1: str, t2: str, model,
                    rng: np.random.Generator) -> str:
    """Simulate a single knockout match (90 min + ET + pens if needed)."""
    s1, s2 = model.sample_score(t1, t2, neutral=True, rng=rng)
    if s1 != s2:
        return t1 if s1 > s2 else t2

    # Extra time: ~35 % of normal-time goal rate
    et1, et2 = model.sample_score(t1, t2, neutral=True, scale=0.35, rng=rng)
    total1, total2 = s1 + et1, s2 + et2
    if total1 != total2:
        return t1 if total1 > total2 else t2

    # Penalties: 50 / 50
    return t1 if rng.random() < 0.5 else t2


def _run_bracket(seeds: List[str], model,
                 stats: dict, stage_name: str,
                 rng: np.random.Generator) -> List[str]:
    """
    Eliminate half the teams in one round.
    Pairs: (seeds[0] vs seeds[1]), (seeds[2] vs seeds[3]), …
    Losers are recorded against stage_name.
    Returns list of winners in bracket order.
    """
    winners = []
    for i in range(0, len(seeds), 2):
        t1, t2 = seeds[i], seeds[i + 1]
        winner = _knockout_match(t1, t2, model, rng)
        loser  = t2 if winner == t1 else t1
        stats[loser][stage_name] += 1
        winners.append(winner)
    return winners


# ---------------------------------------------------------------------------
# Tournament simulation
# ---------------------------------------------------------------------------

def simulate_tournament(model, n_sims: int = 10_000,
                        groups: dict = None,
                        actual_state: dict = None,
                        verbose: bool = True) -> tuple:
    """
    Monte Carlo simulation of the full 2026 World Cup.

    actual_state: dict loaded from actual_results.json (via tournament_state.load_results).
      Played group matches and knockout results are fixed; only the remaining
      matches are simulated.

    Returns (stats, group_outcomes) where:
      stats: {team: {champion, final, semi, quarter, r16, r32, groups}}
      group_outcomes: {letter: {team: {1: count, 2: count, 3: count, 4: count}}}
    """
    if groups is None:
        groups = GROUPS
    if actual_state is None:
        actual_state = {"group_results": {}, "knockout_results": {}}

    all_teams = [t for ts in groups.values() for t in ts]
    stats = {
        t: {"champion": 0, "final": 0, "semi": 0,
            "quarter": 0, "r16": 0, "r32": 0, "groups": 0}
        for t in all_teams
    }
    group_outcomes = {
        letter: {team: {1: 0, 2: 0, 3: 0, 4: 0} for team in teams}
        for letter, teams in groups.items()
    }

    rng = np.random.default_rng(2026)

    iter_range = range(n_sims)
    if verbose:
        from tqdm import tqdm
        iter_range = tqdm(iter_range, desc="Simulating", unit="sim")

    for _ in iter_range:
        _single_sim(model, groups, stats, group_outcomes, actual_state, rng)

    return stats, group_outcomes


def _single_sim(model, groups: dict, stats: dict, group_outcomes: dict,
                actual_state: dict, rng: np.random.Generator) -> None:
    """Run one full tournament simulation, updating stats and group_outcomes in-place."""
    from .tournament_state import group_played_matches, knockout_winner

    group_letters = list(groups.keys())

    winners_by_group: dict[str, str]  = {}
    runners_by_group: dict[str, str]  = {}
    thirds: List[dict] = []

    # --- Group stage ---
    for letter, teams in groups.items():
        actual_matches = group_played_matches(letter, actual_state)
        standings, _ = _group_standings(teams, model, rng, actual_matches)
        for rank, s in enumerate(standings, start=1):
            group_outcomes[letter][s["team"]][rank] += 1
        winners_by_group[letter] = standings[0]["team"]
        runners_by_group[letter] = standings[1]["team"]
        third = {**standings[2], "group": letter}
        thirds.append(third)
        # 4th place → eliminated in groups
        stats[standings[3]["team"]]["groups"] += 1

    # --- Best 8 third-place teams ---
    thirds.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    best_thirds = [t["team"] for t in thirds[:8]]
    for t in thirds[8:]:
        stats[t["team"]]["groups"] += 1   # bottom 4 thirds → out

    # --- Build R32 bracket ---
    # 12 cross-group pairs (1A v 2B, 1B v 2A, …) + 4 pairs from best thirds
    r32: List[Tuple[str, str]] = []
    for i in range(0, len(group_letters), 2):
        g1, g2 = group_letters[i], group_letters[i + 1]
        r32.append((winners_by_group[g1], runners_by_group[g2]))
        r32.append((winners_by_group[g2], runners_by_group[g1]))

    # Shuffle best thirds to avoid predictable pairings
    shuffled_thirds = list(best_thirds)
    rng.shuffle(shuffled_thirds)
    for i in range(0, 8, 2):
        r32.append((shuffled_thirds[i], shuffled_thirds[i + 1]))

    r32_flat = [t for pair in r32 for t in pair]  # 32 teams in bracket order

    # --- Knockout rounds ---
    round_map = [
        (r32_flat,  "r32",     None),
    ]

    def run_round(seeds, stage_name, next_seeds=None):
        """Run one knockout round, using actual results where available."""
        winners = []
        for i in range(0, len(seeds), 2):
            t1, t2 = seeds[i], seeds[i + 1]
            actual = knockout_winner(stage_name, t1, t2, actual_state)
            if actual:
                winner = actual
                loser  = t2 if winner == t1 else t1
            else:
                winner = _knockout_match(t1, t2, model, rng)
                loser  = t2 if winner == t1 else t1
            stats[loser][stage_name] += 1
            winners.append(winner)
        return winners

    r16_seeds = run_round(r32_flat,  "r32")
    qf_seeds  = run_round(r16_seeds, "r16")
    sf_seeds  = run_round(qf_seeds,  "quarter")
    finalists = run_round(sf_seeds,  "semi")

    actual_final = knockout_winner("final", finalists[0], finalists[1], actual_state)
    if actual_final:
        champion  = actual_final
        runner_up = finalists[1] if champion == finalists[0] else finalists[0]
    else:
        champion  = _knockout_match(finalists[0], finalists[1], model, rng)
        runner_up = finalists[1] if champion == finalists[0] else finalists[0]

    stats[champion]["champion"] += 1
    stats[runner_up]["final"]   += 1


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_predictions(stats: dict, n_sims: int,
                      top_n: int = 48) -> None:
    """Print a formatted tournament prediction table."""
    from tabulate import tabulate

    rows = []
    for team, s in stats.items():
        display = DISPLAY_NAMES.get(team, team)
        rows.append({
            "Team":      display,
            "Win %":     f"{100 * s['champion'] / n_sims:.1f}%",
            "Final %":   f"{100 * (s['champion'] + s['final']) / n_sims:.1f}%",
            "Semi %":    f"{100 * (s['champion'] + s['final'] + s['semi']) / n_sims:.1f}%",
            "QF %":      f"{100 * (s['champion'] + s['final'] + s['semi'] + s['quarter']) / n_sims:.1f}%",
            "_sort":     s["champion"],
        })

    rows.sort(key=lambda x: x["_sort"], reverse=True)
    for r in rows:
        del r["_sort"]

    print(f"\n{'=' * 60}")
    print(f"  2026 FIFA WORLD CUP — {n_sims:,} SIMULATIONS")
    print(f"{'=' * 60}")
    print(tabulate(rows[:top_n], headers="keys", tablefmt="rounded_outline",
                   colalign=("left", "right", "right", "right", "right")))
    print()


def print_group_ratings(model, groups: dict = None) -> None:
    """Print model attack/defence ratings for the 48 WC teams."""
    from tabulate import tabulate

    if groups is None:
        groups = GROUPS

    import numpy as _np

    rows = []
    for letter, teams in groups.items():
        for team in teams:
            display = DISPLAY_NAMES.get(team, team)
            att = model.attack.get(team, 0.0)
            dfn = model.defense.get(team, 0.0)
            # xGD vs league-average opponent (att=0, def=0):
            #   goals scored ≈ exp(att), goals conceded ≈ exp(-def)
            xgd = _np.exp(att) - _np.exp(-dfn)
            rows.append({
                "Group":   letter,
                "Team":    display,
                "Attack":  f"{att:+.3f}",
                "Defence": f"{dfn:+.3f}",
                "xGD":     f"{xgd:+.2f}",
                "_sort":   xgd,
            })

    rows.sort(key=lambda x: x["_sort"], reverse=True)
    for r in rows:
        del r["_sort"]

    print(f"\n{'=' * 60}")
    print("  TEAM RATINGS  (xGD vs average opponent; att=def=0 is dataset average)")
    print(f"{'=' * 60}")
    print(tabulate(rows, headers="keys", tablefmt="rounded_outline"))
    print()
