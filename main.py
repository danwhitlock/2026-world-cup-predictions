"""
2026 FIFA World Cup prediction engine.

Usage:
    python main.py                 # 10 000 simulations
    python main.py --quick         # 1 000 sims, ~15s preview
    python main.py --ratings       # team ratings only, no simulation
    python main.py --sims 50000    # higher precision

Enter actual match results interactively:
    python enter_results.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import load_matches
from src.dixon_coles import DixonColesModel
from src.simulator import simulate_tournament, print_predictions, print_group_ratings
from src.fixtures import GROUPS, ALL_TEAMS
from src.tournament_state import load_results, actual_results_as_dataframe, RESULTS_FILE


def check_missing_teams(model: DixonColesModel) -> None:
    missing = [t for t in ALL_TEAMS if t not in model.teams]
    if missing:
        print(f"\nWarning: {len(missing)} WC team(s) not found in training data "
              f"(will use average-team defaults):")
        for t in missing:
            print(f"   - {t}")


def main():
    parser = argparse.ArgumentParser(description="2026 World Cup predictor")
    parser.add_argument("--sims",     type=int, default=10_000,
                        help="Number of Monte Carlo simulations")
    parser.add_argument("--quick",    action="store_true",
                        help="Run 1 000 simulations for a fast preview")
    parser.add_argument("--ratings",  action="store_true",
                        help="Print team ratings only (no simulation)")
    parser.add_argument("--xi",       type=float, default=0.004,
                        help="Time-decay constant (higher = faster decay)")
    parser.add_argument("--min-date", default="1993-01-01",
                        help="Ignore matches before this date (YYYY-MM-DD)")
    args = parser.parse_args()

    n_sims = 1_000 if args.quick else args.sims

    # --- Load historical data ---
    print(f"\nLoading match data (from {args.min_date}) …")
    matches = load_matches(xi=args.xi, min_date=args.min_date)

    # --- Append actual tournament results (if any) ---
    actual_state = load_results()
    actual_df = actual_results_as_dataframe(actual_state)
    if actual_df is not None:
        import pandas as pd
        n_actual = len(actual_df)
        matches = pd.concat([matches, actual_df], ignore_index=True)
        print(f"  + {n_actual} actual tournament result(s) added "
              f"(from {RESULTS_FILE.name})")
    else:
        print(f"  No actual results yet  (run: python enter_results.py)")

    n_teams = len(set(matches["home_team"]) | set(matches["away_team"]))
    print(f"  {len(matches):,} total matches across {n_teams} teams")

    # --- Fit model ---
    model = DixonColesModel()
    model.fit(matches, verbose=True)

    check_missing_teams(model)

    # --- Ratings ---
    print_group_ratings(model, GROUPS)

    if args.ratings:
        return

    # --- Simulate ---
    print(f"Running {n_sims:,} simulations …\n")
    stats, group_outcomes = simulate_tournament(
        model, n_sims=n_sims, groups=GROUPS,
        actual_state=actual_state, verbose=True,
    )

    print_predictions(stats, n_sims)

    from src.chart import generate_html
    html_path = Path(__file__).parent / "predictions.html"
    generate_html(model, stats, group_outcomes, GROUPS, n_sims, html_path,
                  actual_state=actual_state)
    print(f"\nWall chart saved to: {html_path}")


if __name__ == "__main__":
    main()
