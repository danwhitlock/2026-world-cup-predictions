"""
Dixon-Coles model for football scoreline prediction.

Reference: Dixon & Coles (1997) "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market"
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Precompute log-factorials up to this number of goals
_MAX_GOALS_FIT = 20
_LOG_FACT = np.array([0.0] + list(np.cumsum(np.log(np.arange(1, _MAX_GOALS_FIT + 1)))))


def _tau(x: np.ndarray, y: np.ndarray,
         mu: np.ndarray, lam: np.ndarray, rho: float) -> np.ndarray:
    """Vectorised Dixon-Coles low-score correction."""
    t = np.ones(len(x))
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t[m00] = 1.0 - mu[m00] * lam[m00] * rho
    t[m01] = 1.0 + mu[m01] * rho
    t[m10] = 1.0 + lam[m10] * rho
    t[m11] = 1.0 - rho
    return t


class DixonColesModel:
    """
    Estimates per-team attack / defence parameters by maximising a
    weighted, time-decayed log-likelihood over historical results.
    """

    def __init__(self):
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_advantage: float = 0.0
        self.rho: float = 0.0
        self.fitted: bool = False
        self._score_cache: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, matches: pd.DataFrame,
            l2: float = 0.01, verbose: bool = True,
            ranking_priors: dict[str, float] | None = None) -> "DixonColesModel":
        """
        Fit the model to a DataFrame with columns:
          home_team, away_team, home_score, away_score, neutral_bool, weight

        l2: L2 regularisation strength (prevents overfitting for sparse teams).
        Identifiability is handled by a soft mean-zero constraint on attack params
        so that 0 represents the average team in the dataset, not an arbitrary ref.
        """
        home_teams = set(matches["home_team"])
        away_teams = set(matches["away_team"])
        self.teams = sorted(home_teams | away_teams)
        n = len(self.teams)

        # Pre-index teams in the DataFrame for vectorised LL
        team_idx = {t: i for i, t in enumerate(self.teams)}
        h_idx = matches["home_team"].map(team_idx).values
        a_idx = matches["away_team"].map(team_idx).values
        valid = (~np.isnan(h_idx.astype(float))) & (~np.isnan(a_idx.astype(float)))
        h_idx = h_idx[valid].astype(int)
        a_idx = a_idx[valid].astype(int)
        hs = matches["home_score"].values[valid].astype(int)
        aw = matches["away_score"].values[valid].astype(int)
        neutral = matches["neutral_bool"].values[valid].astype(bool)
        weights = matches["weight"].values[valid]

        # Clip scores for log-factorial lookup
        hs_c = np.clip(hs, 0, _MAX_GOALS_FIT)
        aw_c = np.clip(aw, 0, _MAX_GOALS_FIT)

        # Parameter layout: [attack (n), defense (n), home_adv, rho]
        # Identifiability: soft constraint pushes mean(attack) → 0
        n_params = 2 * n + 2
        x0 = np.zeros(n_params)
        x0[-1] = -0.1  # small initial rho

        # Warm-start attack params from FIFA ranking points for data-sparse teams.
        # Well-represented teams are overridden by the data anyway; the prior only
        # meaningfully shifts teams with few historical matches (e.g. Curaçao, Haiti).
        if ranking_priors:
            valid_pts = [v for v in ranking_priors.values() if v > 0]
            if valid_pts:
                median_pts = float(np.median(valid_pts))
                for i, team in enumerate(self.teams):
                    pts = ranking_priors.get(team, 0.0)
                    if pts > 0:
                        x0[i] = np.log(pts / median_pts)

        def neg_ll(params):
            att = params[:n]
            dfn = params[n:2 * n]
            hfa = params[-2]
            rho = params[-1]

            home_fac = np.where(neutral, 0.0, hfa)
            mu  = np.exp(att[h_idx] - dfn[a_idx] + home_fac)
            lam = np.exp(att[a_idx] - dfn[h_idx])

            log_p_h = hs * np.log(mu)  - mu  - _LOG_FACT[hs_c]
            log_p_a = aw * np.log(lam) - lam - _LOG_FACT[aw_c]

            t = _tau(hs, aw, mu, lam, rho)
            if np.any(t <= 0):
                return 1e12

            ll = weights * (np.log(t) + log_p_h + log_p_a)

            # L2 regularisation on attack and defence
            penalty = l2 * (np.sum(att ** 2) + np.sum(dfn ** 2))

            # Soft mean-zero constraint: keeps attack parameters centred so
            # that att=0 means "average team in the dataset" rather than
            # "whatever the arbitrary reference team is".
            centering = 500.0 * (np.mean(att) ** 2)

            return -(np.sum(ll)) + penalty + centering

        if verbose:
            print(f"Fitting Dixon-Coles on {valid.sum():,} matches, {n} teams …")

        res = minimize(neg_ll, x0, method="L-BFGS-B",
                       options={"maxiter": 3000, "maxfun": 500_000,
                                "ftol": 1e-10, "gtol": 1e-6})

        if verbose:
            status = "converged" if res.success else "DID NOT CONVERGE"
            print(f"Optimisation {status}: LL = {-res.fun:.1f}")

        att = res.x[:n]
        dfn = res.x[n:2 * n]

        self.attack         = dict(zip(self.teams, att))
        self.defense        = dict(zip(self.teams, dfn))
        self.home_advantage = float(res.x[-2])
        self.rho            = float(res.x[-1])
        self.fitted         = True
        self._score_cache   = {}

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _expected_goals(self, home: str, away: str,
                        neutral: bool = True, hfa_scale: float = 1.0):
        att_h = self.attack.get(home, 0.0)
        def_h = self.defense.get(home, 0.0)
        att_a = self.attack.get(away, 0.0)
        def_a = self.defense.get(away, 0.0)
        hfa = 0.0 if neutral else self.home_advantage * hfa_scale
        mu  = np.exp(att_h - def_a + hfa)
        lam = np.exp(att_a - def_h)
        return mu, lam

    def score_matrix(self, home: str, away: str,
                     neutral: bool = True, max_goals: int = 8,
                     hfa_scale: float = 1.0) -> np.ndarray:
        """Return P[i,j] = P(home scores i, away scores j)."""
        cache_key = (home, away, neutral, max_goals, hfa_scale)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        mu, lam = self._expected_goals(home, away, neutral, hfa_scale)
        goals = np.arange(max_goals + 1)
        p_h = poisson.pmf(goals, mu)
        p_a = poisson.pmf(goals, lam)
        m = np.outer(p_h, p_a)

        # Apply tau corrections to the four low-score cells
        rho = self.rho
        tau_vals = {
            (0, 0): 1 - mu * lam * rho,
            (0, 1): 1 + mu * rho,
            (1, 0): 1 + lam * rho,
            (1, 1): 1 - rho,
        }
        for (i, j), t in tau_vals.items():
            if i <= max_goals and j <= max_goals:
                m[i, j] *= max(t, 1e-12)

        m = np.maximum(m, 0)
        m /= m.sum()

        self._score_cache[cache_key] = m
        return m

    def result_probs(self, home: str, away: str, neutral: bool = True) -> dict:
        """Return P(home win), P(draw), P(away win)."""
        m = self.score_matrix(home, away, neutral)
        return {
            "home_win": float(np.sum(np.tril(m, -1))),
            "draw":     float(np.sum(np.diag(m))),
            "away_win": float(np.sum(np.triu(m, 1))),
        }

    def sample_score(self, home: str, away: str,
                     neutral: bool = True, scale: float = 1.0,
                     rng: np.random.Generator = None,
                     max_goals: int = 8,
                     hfa_scale: float = 1.0) -> tuple[int, int]:
        """
        Sample a scoreline from the model.
        scale < 1 reduces expected goals (used for extra-time simulation).
        hfa_scale < 1 applies a partial home advantage (e.g. host nations at a WC).
        """
        if rng is None:
            rng = np.random.default_rng()

        if scale == 1.0:
            m = self.score_matrix(home, away, neutral, max_goals, hfa_scale)
            flat = m.flatten()
        else:
            mu, lam = self._expected_goals(home, away, neutral, hfa_scale)
            mu  *= scale
            lam *= scale
            goals = np.arange(max_goals + 1)
            p_h = poisson.pmf(goals, mu)
            p_a = poisson.pmf(goals, lam)
            flat = np.outer(p_h, p_a).flatten()
            flat = np.maximum(flat, 0)
            flat /= flat.sum()

        idx = rng.choice(len(flat), p=flat)
        return divmod(idx, max_goals + 1)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def ratings(self) -> pd.DataFrame:
        """DataFrame of teams sorted by overall strength (attack − defence)."""
        rows = [
            {
                "team":    t,
                "attack":  self.attack.get(t, 0.0),
                "defense": self.defense.get(t, 0.0),
                # xGD vs a league-average opponent (att=0, def=0):
                #   goals_scored = exp(att), goals_conceded = exp(-def)
                "xGD":     np.exp(self.attack.get(t, 0.0))
                           - np.exp(-self.defense.get(t, 0.0)),
            }
            for t in self.teams
        ]
        return (pd.DataFrame(rows)
                  .sort_values("xGD", ascending=False)
                  .reset_index(drop=True))
