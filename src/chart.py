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


def _mc_champion_prob(team: str, round_idx: int, stats: dict, n_sims: int) -> float:
    """
    P(team becomes champion | team reached round_idx), from Monte Carlo counts.
    Using this (rather than the conditional round-advance rate) ensures the bracket
    always propagates toward the overall tournament winner.
    round_idx: 0=R32, 1=R16, 2=QF, 3=SF, 4=Final
    """
    s = stats[team]
    reach_r32   = n_sims     - s['groups']
    reach_r16   = reach_r32  - s['r32']
    reach_qf    = reach_r16  - s['r16']
    reach_sf    = reach_qf   - s['quarter']
    reach_final = reach_sf   - s['semi']
    denominators = [reach_r32, reach_r16, reach_qf, reach_sf, reach_final]
    den = denominators[round_idx]
    return s['champion'] / den if den > 0 else 0.0


def _predicted_standings(group_letter: str, group_outcomes: dict, n_sims: int, stats: dict) -> list[dict]:
    """
    Return teams sorted by knockout qualification probability (highest first).
    p_qualify = 1 - P(eliminated in group stage), which includes the best-third pathway.
    Each entry: {team, p1, p2, p3, p4, p_qualify}
    """
    rows = []
    for team, counts in group_outcomes[group_letter].items():
        rows.append({
            "team": team,
            "p1": counts[1] / n_sims,
            "p2": counts[2] / n_sims,
            "p3": counts[3] / n_sims,
            "p4": counts[4] / n_sims,
            "p_qualify": 1.0 - stats[team]["groups"] / n_sims,
        })
    rows.sort(key=lambda x: x["p_qualify"], reverse=True)
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


def _build_r32(group_outcomes: dict, n_sims: int, groups: dict,
               actual_state: dict = None) -> list[dict]:
    """
    Build the 16 R32 matchup dicts in bracket order.
    Uses actual group standings when all groups are complete; falls back to
    simulation-based most-likely-team approximation otherwise.
    Returns [{"t1": str, "t2": str, "label": str}, …]
    """
    if actual_state is not None:
        from .tournament_state import compute_round_fixtures
        actual = compute_round_fixtures("r32", actual_state, groups)
        if actual is not None:
            return actual

    # Simulation-based fallback: use the official FIFA R32 bracket structure
    from .fixtures import R32_BRACKET
    from .tournament_state import _solve_third_place

    letters = list(groups.keys())

    # Approximate which 8 groups most often have a qualifying 3rd-place team
    third_pool = []
    for letter in letters:
        best = _most_likely_team(letter, 3, group_outcomes, n_sims)
        p = group_outcomes[letter][best][3] / n_sims
        third_pool.append((letter, best, p))
    third_pool.sort(key=lambda x: x[2], reverse=True)
    qualified_letters = [l for l, _, _ in third_pool[:8]]
    most_likely_3rd = {l: t for l, t, _ in third_pool}

    third_assignment = _solve_third_place(R32_BRACKET, qualified_letters)

    matches = []
    for i, (p1, p2) in enumerate(R32_BRACKET):
        rank1 = int(p1[0])
        t1 = _most_likely_team(p1[1], rank1, group_outcomes, n_sims)
        if p2.startswith("3"):
            g = third_assignment[i]
            t2 = most_likely_3rd[g]
            label = f"{p1} v 3{g}"
        else:
            rank2 = int(p2[0])
            t2 = _most_likely_team(p2[1], rank2, group_outcomes, n_sims)
            label = f"{p1} v {p2}"
        matches.append({"t1": t1, "t2": t2, "label": label})

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

            p_next = _win_pct(model, t1, t2)
            next_teams.append({"t1": t1, "t2": t2, "p1": p_next, "p2": 1 - p_next,
                                "winner": t1 if p_next >= 0.5 else t2,
                                "label": "—"})

        rounds.append(round_matches)
        current = next_teams

        if len(current) == 1:
            break

    return rounds


# ---------------------------------------------------------------------------
# HTML generation helpers
# ---------------------------------------------------------------------------

def _groups_html(groups: dict, group_outcomes: dict, n_sims: int, stats: dict) -> str:
    cards = []

    for letter, _teams in groups.items():
        standings = _predicted_standings(letter, group_outcomes, n_sims, stats)

        rows_html = ""
        for rank, s in enumerate(standings, start=1):
            q_pct = s["p_qualify"] * 100

            if rank == 1:
                row_cls = "qual-1"
            elif rank == 2:
                row_cls = "qual-2"
            elif rank == 3:
                row_cls = "qual-3"
            else:
                row_cls = "out"

            rows_html += f"""
              <tr class="{row_cls}">
                <td class="pos">{rank}</td>
                <td>{_dn(s['team'])}</td>
                <td class="pct">{q_pct:.0f}%</td>
              </tr>"""

        cards.append(f"""
          <div class="group-card">
            <div class="group-header">Group {letter}</div>
            <table class="group-table">
              {rows_html}
            </table>
          </div>""")

    return "\n".join(cards)


def _match_html(m: dict) -> str:
    """Render a single knockout match cell."""
    if not isinstance(m, dict) or "t1" not in m:
        return '<div style="height:52px"></div>'

    t1, t2 = _dn(m["t1"]), _dn(m.get("t2", "—"))
    p1 = m.get("p1", 0.5)
    p2 = m.get("p2", 0.5)
    winner = m.get("winner", "")

    def team_row(name, prob, is_w):
        cls = "match-row winner" if is_w else "match-row"
        return f"""<div class="{cls}"><span>{name}</span><span class="match-pct">{prob*100:.0f}%</span></div>"""

    return f"""<div class="match">
        {team_row(t1, p1, winner == m['t1'])}
        {team_row(t2, p2, winner == m.get('t2', ''))}
      </div>"""


def _bracket_html(model, r32: list[dict], stats: dict, n_sims: int) -> str:
    """Build the full bracket HTML from R32 through Final."""

    def get_team(m):
        return m["winner"] if isinstance(m, dict) and "winner" in m else m

    def make_match(t1, t2, round_idx, label=""):
        if round_idx == 4:
            # Final: compare raw champion counts so the bracket winner always
            # equals the MC leader shown in the hero box.
            p1_adv = stats[t1]["champion"]
            p2_adv = stats[t2]["champion"]
        else:
            p1_adv = _mc_champion_prob(t1, round_idx, stats, n_sims)
            p2_adv = _mc_champion_prob(t2, round_idx, stats, n_sims)
        total = p1_adv + p2_adv
        p1 = p1_adv / total if total > 0 else 0.5
        return {"t1": t1, "t2": t2, "p1": p1, "p2": 1 - p1,
                "winner": t1 if p1 >= 0.5 else t2, "label": label}

    # Process R32 (round_idx=0)
    r32_resolved = []
    for m in r32:
        r32_resolved.append(make_match(m["t1"], m["t2"], 0, m.get("label", "")))

    # Split into left/right halves
    half = len(r32_resolved) // 2  # = 8

    left_r32  = r32_resolved[:half]
    right_r32 = r32_resolved[half:]

    def advance_half(half_r32):
        rounds_h = [half_r32]
        cur = half_r32
        round_idx = 1  # first advance from R32 produces R16 matchups
        while len(cur) > 1:
            winners = [get_team(m) for m in cur]
            nxt = [make_match(winners[i], winners[i + 1], round_idx)
                   for i in range(0, len(winners), 2)]
            rounds_h.append(nxt)
            cur = nxt
            round_idx += 1
        return rounds_h

    left_rounds  = advance_half(left_r32)
    right_rounds = advance_half(right_r32)

    final_match = make_match(get_team(left_rounds[-1][0]), get_team(right_rounds[-1][0]), 4)

    col_labels_left  = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals"]
    col_labels_right = col_labels_left[::-1]

    def round_col(round_matches, round_idx):
        spacer = (2 ** round_idx - 1) * 8
        items = ""
        for m in round_matches:
            items += f'<div style="margin-bottom:{spacer}px">{_match_html(m)}</div>'
        return f'<div class="bracket-matches">{items}</div>'

    left_cols = ""
    for i, rd in enumerate(left_rounds):
        label = col_labels_left[i] if i < len(col_labels_left) else ""
        left_cols += f"""<div>
            <div class="bracket-col-label">{label}</div>
            {round_col(rd, i)}
          </div>"""

    right_cols = ""
    for i, rd in enumerate(reversed(right_rounds)):
        label = col_labels_right[i] if i < len(col_labels_right) else ""
        right_cols += f"""<div>
            <div class="bracket-col-label">{label}</div>
            {round_col(rd, len(right_rounds) - 1 - i)}
          </div>"""

    champ_name = _dn(final_match['winner'])
    champ_pct  = stats[final_match['winner']]['champion'] / n_sims * 100

    final_col = f"""<div class="bracket-final">
        <div class="final-label">Final</div>
        {_match_html(final_match)}
        <div class="final-champ">
          <div class="label">Predicted Champion</div>
          <div class="name">{champ_name}</div>
          <div class="pct">{champ_pct:.1f}% win probability</div>
        </div>
      </div>"""

    return f"""<div class="bracket-wrap">
        <div class="bracket">
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
        rows += f"""<div class="prob-row">
            <div class="prob-name">{_dn(team)}</div>
            <div class="prob-track"><div class="prob-fill" style="width:{bar_w}px"></div></div>
            <div class="prob-pct">{pct*100:.1f}%</div>
          </div>"""
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_CSS = """
    /* === Design system: warm charcoal dark mode === */
    :root {
      --bg:        #1C1A15;
      --surface:   #2A2620;
      --surface-2: #332E26;
      --border:    #3E3830;
      --border-em: #4E4840;
      --text:      #E8DFC8;
      --text-2:    #9A9080;
      --text-3:    #5E5848;
      --orange:    #D46030;
      --teal:      #2E9A8C;
      --mustard:   #C8A022;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Space Grotesk', system-ui, sans-serif;
    }

    /* Drafting-table grid overlay */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      background-image:
        linear-gradient(rgba(232,223,200,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(232,223,200,0.04) 1px, transparent 1px);
      background-size: 28px 28px;
    }

    /* --- Layout --- */
    .container {
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px;
      position: relative;
      z-index: 1;
    }

    /* --- Header --- */
    .header {
      text-align: center;
      padding: 52px 24px 40px;
      border-bottom: 1px solid var(--border);
      position: relative;
      z-index: 1;
    }

    .eyebrow {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      margin-bottom: 14px;
    }

    .eyebrow-line { width: 24px; height: 1.5px; background: var(--orange); flex-shrink: 0; }

    .eyebrow-text {
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--orange);
    }

    .header h1 {
      font-size: 34px;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text);
      line-height: 1.1;
    }

    .header .subtitle {
      margin-top: 10px;
      font-size: 14px;
      color: var(--text-2);
    }

    .badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: center;
      margin-top: 20px;
    }

    .badge {
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 5px 14px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--text-2);
    }

    /* --- Section headings --- */
    .section-head { margin-top: 48px; }

    .section-eyebrow {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
    }

    .section-eyebrow .line { width: 20px; height: 1.5px; background: var(--orange); flex-shrink: 0; }

    .section-eyebrow .label {
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--orange);
    }

    h2 {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: var(--text);
    }

    .section-note {
      margin-top: 6px;
      margin-bottom: 20px;
      font-size: 12px;
      color: var(--text-2);
    }

    /* --- Champion box --- */
    .champ-box {
      text-align: center;
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 28px 24px;
      margin: 28px auto;
      max-width: 300px;
      position: relative;
    }

    .champ-box::before { content: ''; position: absolute; top: 0; left: 0; width: 28px; height: 3px; background: var(--mustard); }
    .champ-box::after  { content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 28px; background: var(--mustard); }
    .champ-box .br { position: absolute; bottom: 0; right: 0; width: 28px; height: 28px; }
    .champ-box .br::before { content: ''; position: absolute; bottom: 0; right: 0; width: 28px; height: 3px; background: var(--mustard); }
    .champ-box .br::after  { content: ''; position: absolute; bottom: 0; right: 0; width: 3px; height: 28px; background: var(--mustard); }

    .champ-box .champ-label {
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--text-2);
    }

    .champ-box .champ-name {
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--mustard);
      margin: 10px 0 6px;
    }

    .champ-box .champ-pct { font-size: 13px; color: var(--text-2); }

    /* --- Groups grid --- */
    .groups-grid {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 12px;
      margin-top: 20px;
    }

    @media (max-width: 1200px) { .groups-grid { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 700px)  { .groups-grid { grid-template-columns: repeat(2, 1fr); } }

    .group-card {
      background: var(--surface);
      border: 1px solid var(--border);
      overflow: hidden;
      position: relative;
    }

    .group-card::before { content: ''; position: absolute; top: 0; left: 0; width: 20px; height: 2px; background: var(--orange); z-index: 1; }
    .group-card::after  { content: ''; position: absolute; top: 0; left: 0; width: 2px; height: 20px; background: var(--orange); z-index: 1; }

    .group-header {
      padding: 9px 12px 9px 16px;
      background: var(--surface-2);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--text-2);
      border-bottom: 1px solid var(--border);
    }

    .group-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    .group-table tr { border-bottom: 1px solid var(--border); }
    .group-table tr:last-child { border-bottom: none; }
    .group-table td { padding: 5px 8px; }
    .group-table td.pos { color: var(--text-3); width: 18px; padding-right: 4px; font-size: 11px; }
    .group-table td.pct { text-align: right; font-size: 12px; }

    .group-table tr.qual-1 { background: rgba(212,96,48,0.12); }
    .group-table tr.qual-1 td { color: var(--text); font-weight: 600; }
    .group-table tr.qual-1 td.pct { color: var(--orange); }

    .group-table tr.qual-2 { background: rgba(46,154,140,0.10); }
    .group-table tr.qual-2 td { color: var(--text); font-weight: 600; }
    .group-table tr.qual-2 td.pct { color: var(--teal); }

    .group-table tr.qual-3 { background: rgba(200,160,34,0.07); }
    .group-table tr.qual-3 td { color: var(--text-2); }
    .group-table tr.qual-3 td.pct { color: var(--mustard); }

    .group-table tr.out td { color: var(--text-2); }
    .group-table tr.out td.pct { color: var(--text-3); }

    /* --- Knockout bracket --- */
    .bracket-wrap { overflow-x: auto; padding: 12px 0; }
    .bracket { display: inline-flex; align-items: flex-start; gap: 0; }

    .bracket-col-label {
      text-align: center;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--text-3);
      margin-bottom: 8px;
      white-space: nowrap;
      padding: 0 4px;
    }

    .bracket-matches {
      display: flex;
      flex-direction: column;
      min-width: 170px;
      padding: 0 4px;
    }

    .match {
      border: 1px solid var(--border);
      overflow: hidden;
      margin: 2px 0;
      background: var(--surface);
    }

    .match-row {
      display: flex;
      align-items: center;
      padding: 4px 8px;
      font-size: 12px;
    }

    .match-row + .match-row { border-top: 1px solid var(--border); }

    .match-row.winner {
      background: rgba(212,96,48,0.15);
      font-weight: 700;
    }

    .match-row span { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    .match-pct {
      font-size: 10px;
      color: var(--text-2);
      margin-left: 6px;
      min-width: 28px;
      text-align: right;
    }

    .match-row.winner .match-pct { color: var(--orange); }

    /* Final column */
    .bracket-final {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-width: 180px;
      padding: 0 12px;
    }

    .final-label {
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--mustard);
      margin-bottom: 8px;
    }

    .final-champ { margin-top: 16px; text-align: center; }
    .final-champ .label {
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--mustard);
    }
    .final-champ .name {
      font-size: 18px;
      font-weight: 700;
      color: var(--mustard);
      letter-spacing: -0.01em;
      margin-top: 6px;
    }
    .final-champ .pct { font-size: 11px; color: var(--text-2); margin-top: 4px; }

    /* --- Probability bars --- */
    .prob-list { margin-top: 16px; }

    .prob-row {
      display: flex;
      align-items: center;
      margin: 5px 0;
    }

    .prob-name {
      min-width: 145px;
      font-size: 13px;
      text-align: right;
      padding-right: 14px;
      color: var(--text);
    }

    .prob-track {
      height: 3px;
      background: var(--border);
      position: relative;
      flex: 1;
      max-width: 320px;
    }

    .prob-fill { height: 100%; background: var(--orange); }

    .prob-pct {
      margin-left: 12px;
      font-size: 12px;
      font-weight: 700;
      color: var(--text-2);
      min-width: 38px;
    }

    /* --- Light mode overrides --- */
    html.light {
      --bg:        #F5F0E4;
      --surface:   #FDFAF3;
      --surface-2: #F0EAD8;
      --border:    #C8BEA8;
      --border-em: #A89E8A;
      --text:      #1E1E1E;
      --text-2:    #6B6458;
      --text-3:    #9A9080;
      --orange:    #C4501E;
      --teal:      #1A7A6E;
      --mustard:   #B08818;
    }

    html.light body::before {
      background-image:
        linear-gradient(rgba(30,30,30,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(30,30,30,0.04) 1px, transparent 1px);
    }

    html.light .group-table tr.qual-1 { background: rgba(196,80,30,0.10); }
    html.light .group-table tr.qual-2 { background: rgba(26,122,110,0.08); }
    html.light .group-table tr.qual-3 { background: rgba(176,136,24,0.08); }
    html.light .match-row.winner { background: rgba(196,80,30,0.10); }

    /* --- Theme toggle --- */
    .theme-toggle {
      position: fixed;
      top: 16px;
      right: 20px;
      z-index: 100;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-2);
      width: 36px;
      height: 36px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: color .15s, border-color .15s;
    }

    .theme-toggle:hover {
      color: var(--orange);
      border-color: var(--orange);
    }
"""


def generate_html(model, stats: dict, group_outcomes: dict,
                  groups: dict, n_sims: int, output_path: Path,
                  actual_state: dict = None) -> None:
    """Write a self-contained HTML wall chart to output_path."""

    r32 = _build_r32(group_outcomes, n_sims, groups, actual_state)

    groups_html   = _groups_html(groups, group_outcomes, n_sims, stats)
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
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet" />
  <style>{_CSS}</style>
</head>
<body>

<header class="header">
  <div class="eyebrow">
    <span class="eyebrow-line"></span>
    <span class="eyebrow-text">Monte Carlo Simulation &middot; {n_sims:,} Iterations</span>
    <span class="eyebrow-line"></span>
  </div>
  <h1>2026 FIFA World Cup Predictions</h1>
  <p class="subtitle">Statistical model based on {n_sims:,} Monte Carlo simulations</p>
  <div class="badges">
    <span class="badge">Dixon-Coles Model</span>
    <span class="badge">30,000+ historical matches</span>
    <span class="badge">Time-decayed weighting</span>
    <span class="badge">All 48 teams</span>
  </div>
</header>

<button class="theme-toggle" id="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark mode">
  <svg id="icon-moon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
  <svg id="icon-sun" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
</button>
<script>
(function(){{
  var stored = localStorage.getItem('wc-theme');
  var preferLight = window.matchMedia('(prefers-color-scheme: light)').matches;
  if (stored === 'light' || (!stored && preferLight)) {{
    document.documentElement.classList.add('light');
    document.getElementById('icon-moon').style.display = 'none';
    document.getElementById('icon-sun').style.display = '';
  }}
}})();
function toggleTheme() {{
  var isLight = document.documentElement.classList.toggle('light');
  localStorage.setItem('wc-theme', isLight ? 'light' : 'dark');
  document.getElementById('icon-moon').style.display = isLight ? 'none' : '';
  document.getElementById('icon-sun').style.display = isLight ? '' : 'none';
}}
</script>

<div class="container">

  <div class="champ-box">
    <span class="br"></span>
    <div class="champ-label">Predicted Champion</div>
    <div class="champ-name">{predicted_champ}</div>
    <div class="champ-pct">{champ_pct:.1f}% win probability</div>
  </div>

  <div class="section-head">
    <div class="section-eyebrow"><span class="line"></span><span class="label">Group Stage</span></div>
    <h2>Group Stage Predictions</h2>
  </div>
  <p class="section-note">Teams sorted by knockout qualification probability. % = probability of reaching the Round of 32, including via one of the 8 best third-place spots.</p>
  <div class="groups-grid">
    {groups_html}
  </div>

  <div class="section-head">
    <div class="section-eyebrow"><span class="line"></span><span class="label">Knockout Stage</span></div>
    <h2>Knockout Bracket (Most Likely Progression)</h2>
  </div>
  <p class="section-note">Each match shows the two most likely teams. Percentage = probability of winning the tournament from this point (given the team reached this round). Highlighted = predicted winner.</p>
  {bracket_html}

  <div class="section-head">
    <div class="section-eyebrow"><span class="line"></span><span class="label">Tournament Odds</span></div>
    <h2>Championship Probability (Top 24)</h2>
  </div>
  <p class="section-note">Probability of winning the tournament across all {n_sims:,} simulations.</p>
  <div class="prob-list">
    {prob_bars}
  </div>

</div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
