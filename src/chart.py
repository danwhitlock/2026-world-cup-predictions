"""
Generate a self-contained HTML wall chart from simulation results.

Layout:
  • Group stage grid (12 groups, 4 rows each with predicted finish %)
  • Knockout bracket (R32 → R16 → QF → SF → Final, split left/right)
  • Championship probability bar chart
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from .fixtures import DISPLAY_NAMES, GROUPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dn(team: str) -> str:
    """Display name for a team."""
    return DISPLAY_NAMES.get(team, team)


def _win_pct(model, home: str, away: str) -> float:
    """Return P(home wins) from the model (neutral venue)."""
    probs = model.result_probs(home, away, neutral=True)
    return probs["home_win"] + 0.5 * probs["draw"]


def _predicted_standings(group_letter: str, group_outcomes: dict, n_sims: int) -> list[dict]:
    """
    Return teams sorted by predicted finish position (most-likely 1st first).
    Each entry: {team, p1, p2, p3, p4}
    """
    rows = []
    for team, counts in group_outcomes[group_letter].items():
        rows.append({
            "team": team,
            "p1": counts[1] / n_sims,
            "p2": counts[2] / n_sims,
            "p3": counts[3] / n_sims,
            "p4": counts[4] / n_sims,
        })
    rows.sort(key=lambda x: x["p1"], reverse=True)
    return rows


def _most_likely_team(group_letter: str, rank: int,
                      group_outcomes: dict, n_sims: int) -> str:
    """Return the team most frequently finishing at `rank` in `group_letter`."""
    best, best_count = "", 0
    for team, counts in group_outcomes[group_letter].items():
        if counts[rank] > best_count:
            best_count = counts[rank]
            best = team
    return best


def _build_r32(group_outcomes: dict, n_sims: int, groups: dict) -> list[dict]:
    """
    Build the 16 R32 matchup dicts in bracket order.
    Returns [{"t1": str, "t2": str, "label": str}, …]
    Left half: groups A–F (indices 0–5)  → 12 matches (6 pairs)
    Right half: groups G–L (indices 6–11) → 12 matches (6 pairs)
    Third-place slots: appended to end (4 matches, deterministic approximation)
    """
    letters = list(groups.keys())
    matches = []

    for i in range(0, len(letters), 2):
        g1, g2 = letters[i], letters[i + 1]
        t1 = _most_likely_team(g1, 1, group_outcomes, n_sims)
        t2 = _most_likely_team(g2, 2, group_outcomes, n_sims)
        t3 = _most_likely_team(g2, 1, group_outcomes, n_sims)
        t4 = _most_likely_team(g1, 2, group_outcomes, n_sims)
        matches.append({"t1": t1, "t2": t2, "label": f"1{g1} v 2{g2}"})
        matches.append({"t1": t3, "t2": t4, "label": f"1{g2} v 2{g1}"})

    # Best 8 thirds (approximate: pick the 8 most likely 3rd-place teams)
    third_pool = []
    for letter in letters:
        best = _most_likely_team(letter, 3, group_outcomes, n_sims)
        p = group_outcomes[letter][best][3] / n_sims
        third_pool.append((best, p, letter))
    third_pool.sort(key=lambda x: x[1], reverse=True)
    thirds = [t[0] for t in third_pool[:8]]

    for i in range(0, 8, 2):
        matches.append({
            "t1": thirds[i],
            "t2": thirds[i + 1],
            "label": "Best 3rd",
        })

    return matches  # 16 matches total


def _simulate_bracket(model, r32: list[dict]) -> list[list[dict]]:
    """
    Deterministically propagate the bracket using head-to-head win probabilities.
    Returns list of rounds: [[{t1,t2,p1,p2,winner}, …], …]
    Teams are matched [0 v 1, 2 v 3, …] within each round.
    """
    rounds = []
    current = r32

    for _round_name in ["R32", "R16", "QF", "SF", "Final"]:
        round_matches = []
        next_teams = []
        for i in range(0, len(current), 2):
            m1 = current[i]
            m2 = current[i + 1] if i + 1 < len(current) else None

            # Get the two teams competing in this match
            if isinstance(m1, dict) and "winner" in m1:
                t1 = m1["winner"]
            elif isinstance(m1, dict) and "t1" in m1:
                p1 = _win_pct(model, m1["t1"], m1["t2"])
                m1 = {**m1, "p1": p1, "p2": 1 - p1,
                      "winner": m1["t1"] if p1 >= 0.5 else m1["t2"]}
                t1 = m1["winner"]
            else:
                t1 = m1

            if m2 is None:
                next_teams.append(t1)
                round_matches.append(m1)
                break

            if isinstance(m2, dict) and "winner" in m2:
                t2 = m2["winner"]
            elif isinstance(m2, dict) and "t1" in m2:
                p2 = _win_pct(model, m2["t1"], m2["t2"])
                m2 = {**m2, "p1": p2, "p2": 1 - p2,
                      "winner": m2["t1"] if p2 >= 0.5 else m2["t2"]}
                t2 = m2["winner"]
            else:
                t2 = m2

            round_matches.append(m1)
            round_matches.append(m2)

            # Next match for the next round
            p_next = _win_pct(model, t1, t2)
            next_teams.append({"t1": t1, "t2": t2, "p1": p_next, "p2": 1 - p_next,
                                "winner": t1 if p_next >= 0.5 else t2,
                                "label": "—"})

        rounds.append(round_matches)
        current = next_teams

        if len(current) == 1:
            break

    return rounds


def _color_for_pct(pct: float) -> str:
    """Green → amber → red gradient based on win probability."""
    if pct >= 0.60: return "#27ae60"
    if pct >= 0.45: return "#2ecc71"
    if pct >= 0.35: return "#f39c12"
    if pct >= 0.25: return "#e67e22"
    return "#e74c3c"


# ---------------------------------------------------------------------------
# HTML generation helpers
# ---------------------------------------------------------------------------

def _groups_html(groups: dict, group_outcomes: dict, n_sims: int) -> str:
    cards = []
    group_colors = {
        "A": "#c0392b", "B": "#8e44ad", "C": "#2980b9",
        "D": "#27ae60", "E": "#e67e22", "F": "#16a085",
        "G": "#d35400", "H": "#2c3e50", "I": "#7f8c8d",
        "J": "#c0392b", "K": "#8e44ad", "L": "#2980b9",
    }

    for letter, _teams in groups.items():
        standings = _predicted_standings(letter, group_outcomes, n_sims)
        color = group_colors.get(letter, "#333")

        rows_html = ""
        for rank, s in enumerate(standings, start=1):
            p1 = s["p1"] * 100
            p2 = s["p2"] * 100
            q_pct = p1 + p2  # qualify %

            if rank == 1:
                row_style = "background:rgba(255,215,0,0.15);font-weight:bold"
            elif rank == 2:
                row_style = "background:rgba(200,200,200,0.1);font-weight:bold"
            else:
                row_style = "color:#aaa"

            rows_html += f"""
              <tr style="{row_style}">
                <td style="padding:4px 6px;color:#888">{rank}</td>
                <td style="padding:4px 8px">{_dn(s['team'])}</td>
                <td style="padding:4px 6px;text-align:right;color:#aaa">{q_pct:.0f}%</td>
              </tr>"""

        cards.append(f"""
          <div style="background:#1a1f3c;border-radius:10px;overflow:hidden;min-width:170px">
            <div style="background:{color};padding:8px 12px;font-weight:bold;font-size:13px;letter-spacing:1px">
              GROUP {letter}
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:13px">
              {rows_html}
            </table>
          </div>""")

    return "\n".join(cards)


def _match_html(m: dict, is_winner: bool = False) -> str:
    """Render a single knockout match cell."""
    if not isinstance(m, dict) or "t1" not in m:
        return '<div style="height:52px"></div>'

    t1, t2 = _dn(m["t1"]), _dn(m.get("t2", "—"))
    p1 = m.get("p1", 0.5)
    p2 = m.get("p2", 0.5)
    winner = m.get("winner", "")

    def team_row(name, prob, is_w):
        bg = "background:rgba(255,215,0,0.25);font-weight:bold" if is_w else ""
        bar_w = int(prob * 50)
        return f"""
          <div style="display:flex;align-items:center;padding:3px 6px;{bg}">
            <span style="flex:1;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{name}</span>
            <span style="font-size:10px;color:#aaa;margin-left:4px;min-width:30px;text-align:right">{prob*100:.0f}%</span>
          </div>"""

    return f"""
      <div style="border:1px solid #334;border-radius:6px;overflow:hidden;margin:2px 0;background:#111827">
        {team_row(t1, p1, winner == m['t1'])}
        <div style="height:1px;background:#334"></div>
        {team_row(t2, p2, winner == m.get('t2', ''))}
      </div>"""


def _bracket_html(model, r32: list[dict], stats: dict, n_sims: int) -> str:
    """Build the full bracket HTML from R32 through Final."""

    # ---------- compute the bracket rounds ----------
    def get_team(m):
        return m["winner"] if isinstance(m, dict) and "winner" in m else m

    def make_match(t1, t2, label=""):
        p = _win_pct(model, t1, t2)
        return {"t1": t1, "t2": t2, "p1": p, "p2": 1 - p,
                "winner": t1 if p >= 0.5 else t2, "label": label}

    # Process R32
    r32_resolved = []
    for m in r32:
        p = _win_pct(model, m["t1"], m["t2"])
        r32_resolved.append({**m, "p1": p, "p2": 1 - p,
                              "winner": m["t1"] if p >= 0.5 else m["t2"]})

    rounds = [r32_resolved]
    current = r32_resolved
    round_names = ["Round of 32", "Round of 16", "Quarter-Finals",
                   "Semi-Finals", "Final"]

    for name in round_names[1:]:
        winners = [get_team(m) for m in current]
        next_round = [make_match(winners[i], winners[i + 1])
                      for i in range(0, len(winners), 2)]
        rounds.append(next_round)
        current = next_round
        if len(next_round) == 1:
            break

    # ---------- split into left / right halves ----------
    # Left: first 8 R32 matches (Groups A-F)
    # Right: next 8 R32 matches (Groups G-L + thirds)
    half = len(r32_resolved) // 2   # = 8

    left_r32  = r32_resolved[:half]
    right_r32 = r32_resolved[half:]

    def advance_half(half_r32):
        rounds_h = [half_r32]
        cur = half_r32
        while len(cur) > 1:
            winners = [get_team(m) for m in cur]
            nxt = [make_match(winners[i], winners[i + 1])
                   for i in range(0, len(winners), 2)]
            rounds_h.append(nxt)
            cur = nxt
        return rounds_h

    left_rounds  = advance_half(left_r32)
    right_rounds = advance_half(right_r32)

    # The final
    l_sf_winner = get_team(left_rounds[-1][0])
    r_sf_winner = get_team(right_rounds[-1][0])
    final_match = make_match(l_sf_winner, r_sf_winner)

    # ---------- render HTML ----------
    col_labels_left  = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals"]
    col_labels_right = col_labels_left[::-1]

    MAX_ROUNDS = max(len(left_rounds), len(right_rounds))

    def round_col(round_matches, round_idx, total_r32):
        """One round column, with spacing to vertically center matches."""
        n = len(round_matches)
        spacer = (2 ** round_idx - 1) * 8   # px between matches
        items = ""
        for m in round_matches:
            items += f"""
              <div style="margin-bottom:{spacer}px">
                {_match_html(m)}
              </div>"""
        return f"""<div style="display:flex;flex-direction:column;min-width:170px;padding:0 4px">{items}</div>"""

    left_cols = ""
    for i, rd in enumerate(left_rounds):
        label = col_labels_left[i] if i < len(col_labels_left) else ""
        left_cols += f"""
          <div>
            <div style="text-align:center;font-size:11px;color:#888;margin-bottom:6px;white-space:nowrap">{label}</div>
            {round_col(rd, i, half)}
          </div>"""

    right_cols = ""
    for i, rd in enumerate(reversed(right_rounds)):
        label = col_labels_right[i] if i < len(col_labels_right) else ""
        right_cols = f"""
          <div>
            <div style="text-align:center;font-size:11px;color:#888;margin-bottom:6px;white-space:nowrap">{label}</div>
            {round_col(rd, len(right_rounds) - 1 - i, half)}
          </div>""" + right_cols

    final_col = f"""
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:180px;padding:0 10px">
        <div style="text-align:center;font-size:11px;color:#f39c12;margin-bottom:6px;font-weight:bold">FINAL</div>
        {_match_html(final_match)}
        <div style="margin-top:12px;text-align:center">
          <div style="font-size:11px;color:#f39c12">PREDICTED CHAMPION</div>
          <div style="font-size:16px;font-weight:bold;color:gold;margin-top:4px">{_dn(final_match['winner'])}</div>
          <div style="font-size:11px;color:#888">{stats[final_match['winner']]['champion']/n_sims*100:.1f}% win probability</div>
        </div>
      </div>"""

    return f"""
      <div style="overflow-x:auto;padding:10px 0">
        <div style="display:inline-flex;align-items:flex-start;gap:0">
          {left_cols}
          {final_col}
          {right_cols}
        </div>
      </div>"""


def _prob_bars_html(stats: dict, n_sims: int, top_n: int = 24) -> str:
    items = sorted(stats.items(), key=lambda x: x[1]["champion"], reverse=True)[:top_n]
    max_pct = items[0][1]["champion"] / n_sims if items else 0.001

    rows = ""
    for team, s in items:
        pct = s["champion"] / n_sims
        bar_w = int(pct / max(max_pct, 0.001) * 300)
        color = _color_for_pct(pct)
        rows += f"""
          <div style="display:flex;align-items:center;margin:3px 0">
            <div style="min-width:145px;font-size:13px;text-align:right;padding-right:10px;color:#ddd">
              {_dn(team)}
            </div>
            <div style="width:{bar_w}px;height:18px;background:{color};border-radius:3px;transition:width 0.3s"></div>
            <div style="margin-left:8px;font-size:12px;color:#aaa">{pct*100:.1f}%</div>
          </div>"""
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_html(model, stats: dict, group_outcomes: dict,
                  groups: dict, n_sims: int, output_path: Path) -> None:
    """Write a self-contained HTML wall chart to output_path."""

    r32 = _build_r32(group_outcomes, n_sims, groups)

    groups_html   = _groups_html(groups, group_outcomes, n_sims)
    bracket_html  = _bracket_html(model, r32, stats, n_sims)
    prob_bars     = _prob_bars_html(stats, n_sims)

    predicted_champ_team = max(stats, key=lambda t: stats[t]["champion"])
    predicted_champ = _dn(predicted_champ_team)
    champ_pct = stats[predicted_champ_team]["champion"] / n_sims * 100

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>2026 FIFA World Cup — Predictions</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 0;
      background: #0d1117;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    h2 {{
      font-size: 16px; font-weight: 600; color: #58a6ff;
      border-bottom: 1px solid #30363d; padding-bottom: 8px; margin-top: 32px;
      text-transform: uppercase; letter-spacing: 1px;
    }}
    .container {{ max-width: 1600px; margin: 0 auto; padding: 20px 24px; }}
    .header {{
      text-align: center; padding: 32px 20px 24px;
      background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
      border-bottom: 1px solid #30363d;
    }}
    .header h1 {{ margin: 0; font-size: 28px; color: #f0f6fc; }}
    .header p  {{ margin: 8px 0 0; color: #8b949e; font-size: 14px; }}
    .badge {{
      display: inline-block; background: #21262d; border: 1px solid #30363d;
      border-radius: 20px; padding: 4px 14px; font-size: 12px; color: #8b949e; margin: 4px;
    }}
    .champ-box {{
      text-align:center; background: linear-gradient(135deg, #21262d, #161b22);
      border: 1px solid #f39c12; border-radius: 12px;
      padding: 20px; margin: 16px auto; max-width: 320px;
    }}
    .champ-box .label {{ font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }}
    .champ-box .name  {{ font-size: 26px; font-weight: bold; color: gold; margin: 6px 0; }}
    .champ-box .pct   {{ font-size: 13px; color: #8b949e; }}
    .groups-grid {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 12px; margin-top: 16px;
    }}
    @media (max-width: 1200px) {{
      .groups-grid {{ grid-template-columns: repeat(3, 1fr); }}
    }}
    @media (max-width: 700px) {{
      .groups-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .section-note {{ font-size: 12px; color: #8b949e; margin-bottom: 12px; }}
  </style>
</head>
<body>

<div class="header">
  <h1>&#x26BD; 2026 FIFA World Cup Predictions</h1>
  <p>Statistical model based on {n_sims:,} Monte Carlo simulations</p>
  <p>
    <span class="badge">Dixon-Coles Model</span>
    <span class="badge">30,000+ historical matches</span>
    <span class="badge">Time-decayed weighting</span>
    <span class="badge">All 48 teams</span>
  </p>
</div>

<div class="container">

  <div class="champ-box">
    <div class="label">Predicted Champion</div>
    <div class="name">{predicted_champ}</div>
    <div class="pct">{champ_pct:.1f}% win probability</div>
  </div>

  <h2>Group Stage Predictions</h2>
  <p class="section-note">Teams sorted by predicted finishing position. % = probability of qualifying (finishing 1st or 2nd).</p>
  <div class="groups-grid">
    {groups_html}
  </div>

  <h2>Knockout Bracket (Most Likely Progression)</h2>
  <p class="section-note">Each match shows the two most likely teams to appear and their head-to-head win probabilities. Gold = predicted winner.</p>
  {bracket_html}

  <h2>Championship Probability (Top 24)</h2>
  <p class="section-note">Probability of winning the tournament across all {n_sims:,} simulations.</p>
  <div style="margin-top:12px">
    {prob_bars}
  </div>

</div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
