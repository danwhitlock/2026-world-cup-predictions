"""Load and preprocess historical match data for model fitting."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Relative importance of different competition types
_TOURNAMENT_WEIGHTS = [
    ('FIFA World Cup',               1.00),
    ('UEFA Euro',                    0.90),
    ('Copa America',                 0.90),
    ('AFC Asian Cup',                0.85),
    ('Africa Cup of Nations',        0.85),
    ('CONCACAF Gold Cup',            0.85),
    ('OFC Nations Cup',              0.85),
    ('FIFA Confederations Cup',      0.85),
    ('World Cup qualification',      0.80),
    ('Euro qualification',           0.75),
    ('African Cup of Nations qual',  0.70),
    ('Friendly',                     0.50),
]

def _tournament_weight(name: str) -> float:
    for key, weight in _TOURNAMENT_WEIGHTS:
        if key.lower() in name.lower():
            return weight
    return 0.65


def _build_former_name_map() -> dict:
    """Map former team names to current names."""
    df = pd.read_csv(DATA_DIR / "former_names.csv")
    return dict(zip(df["former"], df["current"]))


def load_matches(xi: float = 0.004, min_date: str = "1993-01-01") -> pd.DataFrame:
    """
    Load results.csv, apply time decay and tournament weights.

    xi: exponential decay constant — 0.004 gives ~half-weight to matches ~5 years old
    min_date: discard matches before this date (pre-modern football era)
    """
    former_map = _build_former_name_map()

    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df = df[df["date"] >= min_date].copy()
    df = df.dropna(subset=["home_score", "away_score"])

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Normalise historical team names to current names
    df["home_team"] = df["home_team"].map(lambda x: former_map.get(x, x))
    df["away_team"] = df["away_team"].map(lambda x: former_map.get(x, x))

    # Neutral-venue flag as bool
    df["neutral_bool"] = df["neutral"].map(
        lambda x: x in (True, "TRUE", "True", "true", 1)
    )

    # Time decay: weight = exp(-xi * days_since_most_recent_match)
    latest = df["date"].max()
    df["days_ago"] = (latest - df["date"]).dt.days
    df["time_weight"] = np.exp(-xi * df["days_ago"])

    df["tournament_weight"] = df["tournament"].map(_tournament_weight)
    df["weight"] = df["time_weight"] * df["tournament_weight"]

    return df[
        ["date", "home_team", "away_team", "home_score", "away_score",
         "neutral_bool", "weight"]
    ].reset_index(drop=True)
