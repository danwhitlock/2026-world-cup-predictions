"""
Interactive result entry for the 2026 World Cup.

Run:  python enter_results.py
      python enter_results.py --group A        # enter/update one group
      python enter_results.py --knockout r32   # enter knockout round
      python enter_results.py --show           # print current state
      python enter_results.py --clear          # wipe all results

After entering results, run:  python main.py --quick
to get updated predictions.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.fixtures import GROUPS, DISPLAY_NAMES
from src.tournament_state import (
    load_results, save_results,
    all_group_pairs, group_played_matches,
    KNOCKOUT_ROUNDS, RESULTS_FILE,
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def dn(team: str) -> str:
    return DISPLAY_NAMES.get(team, team)


def _show_state(state: dict) -> None:
    """Print a summary of all entered results."""
    print("\n=== ENTERED RESULTS ===\n")

    gr = state.get("group_results", {})
    if gr:
        print("GROUP STAGE:")
        for letter in sorted(gr):
            for m in gr[letter]:
                print(f"  Group {letter}: {dn(m['home'])} {m['home_score']}–{m['away_score']} {dn(m['away'])}")
    else:
        print("GROUP STAGE: (none entered)")

    kr = state.get("knockout_results", {})
    if kr:
        print("\nKNOCKOUT ROUNDS:")
        labels = {"r32": "Round of 32", "r16": "Round of 16",
                  "quarter": "Quarter-Final", "semi": "Semi-Final", "final": "Final"}
        for rnd in KNOCKOUT_ROUNDS:
            for m in kr.get(rnd, []):
                label = labels.get(rnd, rnd.upper())
                t1s = m.get("team1_score", "?")
                t2s = m.get("team2_score", "?")
                w   = m["winner"]
                print(f"  {label}: {dn(m['team1'])} {t1s}–{t2s} {dn(m['team2'])}  → winner: {dn(w)}")
    else:
        print("\nKNOCKOUT ROUNDS: (none entered)")

    print()


# ---------------------------------------------------------------------------
# Group result entry
# ---------------------------------------------------------------------------

def _parse_score(raw: str) -> tuple[int, int] | None:
    """Parse '2-1' or '2 1' → (2, 1). Returns None on bad input."""
    raw = raw.strip().replace(" ", "-")
    parts = raw.split("-")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None


def _enter_group(letter: str, state: dict) -> None:
    teams = GROUPS[letter]
    played = group_played_matches(letter, state)
    played_keys = {frozenset({m["home"], m["away"]}) for m in played}

    print(f"\n--- Group {letter}: {', '.join(dn(t) for t in teams)} ---")
    print("Enter score as  home-away  (e.g. 2-1) or press Enter to skip.\n")

    new_matches = []
    for t1, t2 in all_group_pairs(teams):
        key = frozenset({t1, t2})
        already = key in played_keys

        prompt = (f"  {dn(t1)} vs {dn(t2)}"
                  + (" [already entered — re-enter to overwrite or skip]: "
                     if already else ": "))
        raw = input(prompt).strip()
        if not raw:
            continue

        score = _parse_score(raw)
        while score is None:
            raw = input(f"    Bad format — try again (e.g. 2-1): ").strip()
            if not raw:
                break
            score = _parse_score(raw)

        if score is None:
            continue

        hs, as_ = score
        new_matches.append({"home": t1, "away": t2,
                             "home_score": hs, "away_score": as_})
        # Remove old entry for this pair if it existed
        played = [m for m in played
                  if frozenset({m["home"], m["away"]}) != key]
        played_keys.discard(key)

    if new_matches:
        state.setdefault("group_results", {})[letter] = played + new_matches
        save_results(state)
        print(f"\n  Saved {len(new_matches)} result(s) for Group {letter}.")
    else:
        print("  Nothing changed.")


# ---------------------------------------------------------------------------
# Knockout result entry
# ---------------------------------------------------------------------------

_ROUND_LABELS = {
    "r32":    "Round of 32",
    "r16":    "Round of 16",
    "quarter": "Quarter-Finals",
    "semi":   "Semi-Finals",
    "final":  "Final",
}


def _enter_knockout(round_name: str, state: dict) -> None:
    label = _ROUND_LABELS.get(round_name, round_name.upper())
    print(f"\n--- {label} ---")
    print("Enter: team1 score-score team2  (e.g.  Spain 2-0 Morocco)")
    print("Press Enter with no input when done.\n")

    existing = state.get("knockout_results", {}).get(round_name, [])

    new_matches = []
    while True:
        raw = input("  > ").strip()
        if not raw:
            break

        parts = raw.split()
        # Find the score token (contains a dash and two digits)
        score_idx = None
        for i, p in enumerate(parts):
            s = _parse_score(p)
            if s is not None:
                score_idx = i
                break

        if score_idx is None or score_idx == 0 or score_idx == len(parts) - 1:
            print("  Format: team1_name [multi word ok] score team2_name — try again.")
            continue

        team1 = " ".join(parts[:score_idx])
        team2 = " ".join(parts[score_idx + 1:])
        t1s, t2s = _parse_score(parts[score_idx])

        # Fuzzy-match team names against known fixtures
        team1_key = _fuzzy_match(team1)
        team2_key = _fuzzy_match(team2)

        if not team1_key or not team2_key:
            unknown = team1 if not team1_key else team2
            print(f"  Unrecognised team: '{unknown}'. Check spelling and try again.")
            continue

        winner = team1_key if t1s > t2s else (team2_key if t2s > t1s else None)
        if winner is None:
            pw = input(f"  Draw after 90 min — who won on penalties? "
                       f"[{dn(team1_key)} / {dn(team2_key)}]: ").strip()
            winner = _fuzzy_match(pw)
            if not winner:
                print("  Unrecognised team name — skipping match.")
                continue

        new_matches.append({
            "team1": team1_key, "team2": team2_key,
            "team1_score": t1s, "team2_score": t2s,
            "winner": winner,
        })
        print(f"  Recorded: {dn(team1_key)} {t1s}–{t2s} {dn(team2_key)}"
              f"  →  winner: {dn(winner)}")

    if new_matches:
        state.setdefault("knockout_results", {})[round_name] = (
            existing + new_matches
        )
        save_results(state)
        print(f"\n  Saved {len(new_matches)} result(s) for {label}.")
    else:
        print("  Nothing changed.")


# ---------------------------------------------------------------------------
# Fuzzy team name matching
# ---------------------------------------------------------------------------

def _all_known_teams() -> list[str]:
    return [t for teams in GROUPS.values() for t in teams]


def _fuzzy_match(name: str) -> str | None:
    """
    Match a user-typed team name to the canonical key used in results.csv.
    Tries: exact, display-name match, case-insensitive substring.
    """
    name_lower = name.strip().lower()
    all_teams = _all_known_teams()

    # Exact match on data key
    for t in all_teams:
        if t.lower() == name_lower:
            return t

    # Match on display name
    for t in all_teams:
        if dn(t).lower() == name_lower:
            return t

    # Prefix / substring match on data key or display name
    candidates = [t for t in all_teams
                  if name_lower in t.lower() or name_lower in dn(t).lower()]
    if len(candidates) == 1:
        return candidates[0]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enter 2026 WC actual results")
    parser.add_argument("--group",    metavar="LETTER",
                        help="Enter/update results for a specific group (e.g. A)")
    parser.add_argument("--knockout", metavar="ROUND",
                        choices=KNOCKOUT_ROUNDS,
                        help="Enter knockout results (r32/r16/quarter/semi/final)")
    parser.add_argument("--show",     action="store_true",
                        help="Print all currently entered results and exit")
    parser.add_argument("--clear",    action="store_true",
                        help="Wipe all entered results")
    args = parser.parse_args()

    state = load_results()

    if args.clear:
        confirm = input("Clear ALL entered results? [yes/N]: ").strip().lower()
        if confirm == "yes":
            save_results({"group_results": {}, "knockout_results": {}})
            print("Cleared.")
        else:
            print("Aborted.")
        return

    if args.show:
        _show_state(state)
        return

    if args.group:
        letter = args.group.upper()
        if letter not in GROUPS:
            print(f"Unknown group '{letter}'. Valid: {', '.join(GROUPS)}")
            sys.exit(1)
        _enter_group(letter, state)

    elif args.knockout:
        _enter_knockout(args.knockout, state)

    else:
        # Interactive menu
        print("\n=== 2026 World Cup Result Entry ===")
        print("Actual results are saved to:", RESULTS_FILE)
        print("After entering results, run:  python main.py --quick\n")

        while True:
            print("What would you like to enter?")
            print("  G A-L   — group stage (e.g. G A, G B, …)")
            print("  K r32   — knockout round (r32 / r16 / quarter / semi / final)")
            print("  show    — view all entered results")
            print("  q       — quit\n")
            cmd = input("> ").strip().lower()

            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "show":
                _show_state(state)
            elif cmd.startswith("g "):
                letter = cmd[2:].strip().upper()
                if letter in GROUPS:
                    _enter_group(letter, state)
                    state = load_results()  # reload after save
                else:
                    print(f"  Unknown group '{letter}'")
            elif cmd.startswith("k "):
                rnd = cmd[2:].strip().lower()
                if rnd in KNOCKOUT_ROUNDS:
                    _enter_knockout(rnd, state)
                    state = load_results()
                else:
                    print(f"  Unknown round '{rnd}'. Options: {', '.join(KNOCKOUT_ROUNDS)}")
            else:
                print("  Unrecognised command.")

        print("\nDone. Run  python main.py --quick  to update predictions.")


if __name__ == "__main__":
    main()
