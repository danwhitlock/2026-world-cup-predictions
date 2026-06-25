"""
Web server for the 2026 World Cup prediction engine.

Run:
    python server.py              # full precision (10 000 sims)
    python server.py --quick      # 1 000 sims, faster first load
    python server.py --port 8080  # custom port

Open http://localhost:5000, then:
  📊 Live Tables  — group standings, fixtures and results
  📋 Enter Results — input actual match scores
  🔄 Regenerate   — refit model and update predictions
"""

import argparse
import json
import sys
import threading
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, Response, jsonify, request

from src.fixtures import DISPLAY_NAMES, GROUPS
from src.tournament_state import (
    KNOCKOUT_ROUNDS,
    all_group_pairs,
    load_results,
    save_results,
)

app = Flask(__name__)
_ROOT = Path(__file__).parent
_HTML_PATH = _ROOT / "predictions.html"
_DEFAULT_SIMS = 10_000

_sim_status: dict = {"running": False, "message": "idle", "error": None}
_sim_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _run_simulation(n_sims: int) -> None:
    import pandas as pd
    from src.chart import generate_html
    from src.dixon_coles import DixonColesModel
    from src.preprocessing import load_matches, load_ranking_priors
    from src.simulator import simulate_tournament
    from src.tournament_state import actual_results_as_dataframe

    _sim_status["message"] = "Loading match data…"
    matches = load_matches()
    actual_state = load_results()
    actual_df = actual_results_as_dataframe(actual_state)
    if actual_df is not None:
        matches = pd.concat([matches, actual_df], ignore_index=True)

    _sim_status["message"] = "Fitting model…"
    ranking_priors = load_ranking_priors()
    model = DixonColesModel()
    model.fit(matches, verbose=False, ranking_priors=ranking_priors or None)

    _sim_status["message"] = f"Simulating {n_sims:,} tournaments…"
    stats, group_outcomes = simulate_tournament(
        model, n_sims=n_sims, groups=GROUPS,
        actual_state=actual_state, verbose=False,
    )

    _sim_status["message"] = "Generating chart…"
    generate_html(model, stats, group_outcomes, GROUPS, n_sims, _HTML_PATH)


def _sim_worker(n_sims: int) -> None:
    try:
        _run_simulation(n_sims)
        _sim_status["message"] = "done"
    except Exception as exc:
        _sim_status["error"] = str(exc)
        _sim_status["message"] = "error"
    finally:
        _sim_status["running"] = False


# ---------------------------------------------------------------------------
# Team name fuzzy matching
# ---------------------------------------------------------------------------

def _all_teams() -> list[str]:
    return [t for ts in GROUPS.values() for t in ts]


def _fuzzy_match(name: str) -> str | None:
    name_lower = name.strip().lower()
    for t in _all_teams():
        if t.lower() == name_lower:
            return t
    for t in _all_teams():
        if DISPLAY_NAMES.get(t, t).lower() == name_lower:
            return t
    candidates = [
        t for t in _all_teams()
        if name_lower in t.lower() or name_lower in DISPLAY_NAMES.get(t, t).lower()
    ]
    return candidates[0] if len(candidates) == 1 else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if not _HTML_PATH.exists():
        return _no_predictions_page(), 200
    html = _HTML_PATH.read_text(encoding="utf-8")
    injection = _build_injection()
    if "</body>" in html:
        html = html.replace("</body>", injection + "\n</body>", 1)
    else:
        html += injection
    return Response(html, mimetype="text/html")


@app.route("/api/results", methods=["GET"])
def api_get_results():
    return jsonify(load_results())


@app.route("/api/results/group", methods=["POST"])
def api_save_group_result():
    data = request.json
    letter = data.get("letter", "").upper()
    if letter not in GROUPS:
        return jsonify({"error": f"Unknown group '{letter}'"}), 400
    match = {
        "home": data["home"], "away": data["away"],
        "home_score": int(data["home_score"]),
        "away_score": int(data["away_score"]),
    }
    state = load_results()
    state.setdefault("group_results", {})
    existing = state["group_results"].get(letter, [])
    pair = frozenset({match["home"], match["away"]})
    existing = [m for m in existing if frozenset({m["home"], m["away"]}) != pair]
    existing.append(match)
    state["group_results"][letter] = existing
    save_results(state)
    return jsonify({"ok": True})


@app.route("/api/results/group", methods=["DELETE"])
def api_delete_group_result():
    data = request.json
    letter = data.get("letter", "").upper()
    home, away = data.get("home"), data.get("away")
    state = load_results()
    existing = state.get("group_results", {}).get(letter, [])
    pair = frozenset({home, away})
    state.setdefault("group_results", {})[letter] = [
        m for m in existing if frozenset({m["home"], m["away"]}) != pair
    ]
    save_results(state)
    return jsonify({"ok": True})


@app.route("/api/results/knockout", methods=["POST"])
def api_save_knockout_result():
    data = request.json
    rnd = data.get("round", "").lower()
    if rnd not in KNOCKOUT_ROUNDS:
        return jsonify({"error": f"Unknown round '{rnd}'"}), 400

    team1 = _fuzzy_match(data.get("team1", ""))
    team2 = _fuzzy_match(data.get("team2", ""))
    if not team1 or not team2:
        bad = data.get("team1") if not team1 else data.get("team2")
        return jsonify({"error": f"Unrecognised team: '{bad}'"}), 400

    t1s, t2s = int(data["team1_score"]), int(data["team2_score"])
    winner_raw = data.get("winner", "")
    winner = _fuzzy_match(winner_raw) if winner_raw else None
    if winner is None:
        if t1s > t2s:     winner = team1
        elif t2s > t1s:   winner = team2
        else:
            return jsonify({"error": "Draw — specify penalty winner"}), 400

    match = {"team1": team1, "team2": team2,
             "team1_score": t1s, "team2_score": t2s, "winner": winner}
    state = load_results()
    state.setdefault("knockout_results", {})
    existing = state["knockout_results"].get(rnd, [])
    pair = frozenset({team1, team2})
    existing = [m for m in existing if frozenset({m["team1"], m["team2"]}) != pair]
    existing.append(match)
    state["knockout_results"][rnd] = existing
    save_results(state)
    return jsonify({"ok": True, "team1": team1, "team2": team2, "winner": winner})


@app.route("/api/results/knockout", methods=["DELETE"])
def api_delete_knockout_result():
    data = request.json
    rnd = data.get("round", "").lower()
    team1, team2 = data.get("team1"), data.get("team2")
    state = load_results()
    existing = state.get("knockout_results", {}).get(rnd, [])
    pair = frozenset({team1, team2})
    state.setdefault("knockout_results", {})[rnd] = [
        m for m in existing if frozenset({m["team1"], m["team2"]}) != pair
    ]
    save_results(state)
    return jsonify({"ok": True})


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    with _sim_lock:
        if _sim_status["running"]:
            return jsonify({"status": "already_running"})
        _sim_status.update(running=True, error=None, message="Starting…")
    n = (request.json or {}).get("n_sims", _DEFAULT_SIMS)
    threading.Thread(target=_sim_worker, args=(n,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/status")
def api_status():
    return jsonify(_sim_status)


@app.route("/api/live")
def api_live():
    """Compute live group standings and fixture lists from actual results."""
    state = load_results()
    result = {}

    for letter, teams in GROUPS.items():
        played = state.get("group_results", {}).get(letter, [])
        rec = {t: {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0}
               for t in teams}

        for m in played:
            h, a = m["home"], m["away"]
            hs, as_ = int(m["home_score"]), int(m["away_score"])
            if h not in rec or a not in rec:
                continue
            rec[h]["gf"] += hs; rec[h]["ga"] += as_; rec[h]["p"] += 1
            rec[a]["gf"] += as_; rec[a]["ga"] += hs; rec[a]["p"] += 1
            if hs > as_:
                rec[h]["pts"] += 3; rec[h]["w"] += 1; rec[a]["l"] += 1
            elif as_ > hs:
                rec[a]["pts"] += 3; rec[a]["w"] += 1; rec[h]["l"] += 1
            else:
                rec[h]["pts"] += 1; rec[a]["pts"] += 1
                rec[h]["d"] += 1;   rec[a]["d"] += 1

        standings_order = sorted(
            teams,
            key=lambda t: (rec[t]["pts"], rec[t]["gf"] - rec[t]["ga"], rec[t]["gf"]),
            reverse=True,
        )

        played_map = {}
        for m in played:
            played_map[frozenset({m["home"], m["away"]})] = m

        fixtures = []
        for t1, t2 in combinations(teams, 2):
            m = played_map.get(frozenset({t1, t2}))
            if m:
                s1 = m["home_score"] if m["home"] == t1 else m["away_score"]
                s2 = m["away_score"] if m["home"] == t1 else m["home_score"]
                fixtures.append({"t1": t1, "t2": t2, "s1": s1, "s2": s2, "played": True})
            else:
                fixtures.append({"t1": t1, "t2": t2, "played": False})

        result[letter] = {
            "standings": [
                {"team": t, "gd": rec[t]["gf"] - rec[t]["ga"], **rec[t]}
                for t in standings_order
            ],
            "fixtures": fixtures,
        }

    return jsonify(result)


# ---------------------------------------------------------------------------
# No-predictions fallback page
# ---------------------------------------------------------------------------

def _no_predictions_page() -> str:
    return """<!DOCTYPE html>
<html><head><title>2026 World Cup Predictions</title></head>
<body style="background:#111827;color:#f3f4f6;font-family:sans-serif;
  display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0">
<div style="text-align:center;max-width:400px">
  <h2 style="color:#f59e0b">2026 FIFA World Cup</h2>
  <p style="color:#9ca3af">No predictions generated yet.<br>
  Click below to run the simulation (~60s).</p>
  <button onclick="regen()" style="background:#f59e0b;color:#000;border:none;
    padding:12px 32px;border-radius:8px;font-size:15px;font-weight:700;cursor:pointer">
    Run Predictions</button>
  <p id="st" style="margin-top:14px;color:#9ca3af;font-size:13px"></p>
</div>
<script>
async function regen(){
  document.getElementById('st').textContent='Starting…';
  await fetch('/api/simulate',{method:'POST',
    headers:{'Content-Type':'application/json'},body:'{}'});
  poll();
}
async function poll(){
  const s=await(await fetch('/api/status')).json();
  document.getElementById('st').textContent=s.message;
  if(s.running)setTimeout(poll,2000);
  else if(s.message==='done')location.reload();
  else document.getElementById('st').style.color='#f87171';
}
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# HTML injection — results modal + live tables panel
# ---------------------------------------------------------------------------

_INJECTION_TEMPLATE = r"""
<style>
/* ── Design tokens ───────────────────────────────────────────── */
#wc-fabs,#wc-modal,#wc-live-panel,#wc-overlay{
  --bg:#1C1A15;--surface:#2A2620;--surface-2:#332E26;
  --border:#3E3830;--text:#E8DFC8;--text-2:#9A9080;--text-3:#5E5848;
  --orange:#D46030;--teal:#2E9A8C;
  font-family:'Space Grotesk',system-ui,sans-serif;
}
html.light #wc-fabs,html.light #wc-modal,
html.light #wc-live-panel,html.light #wc-overlay{
  --bg:#F5F0E4;--surface:#FDFAF3;--surface-2:#F0EAD8;
  --border:#C8BEA8;--text:#1E1E1E;--text-2:#6B6458;--text-3:#9A9080;
  --orange:#C4501E;--teal:#1A7A6E;
}
html.light .wc-si,html.light .wc-ko-row input,
html.light #wc-ko-rnd,html.light #wc-ko-pens input{
  background:#F5F0E4;
}
/* ── Light mode: main page overrides ───────────────────────── */
html.light{--bg:#F5F0E4;--surface:#FDFAF3;--surface-2:#F0EAD8;
  --border:#C8BEA8;--border-em:#A89E8A;--text:#1E1E1E;
  --text-2:#6B6458;--text-3:#9A9080;--orange:#C4501E;
  --teal:#1A7A6E;--mustard:#B08818}
html.light body{background:#F5F0E4;color:#1E1E1E}
html.light body::before{background-image:
  linear-gradient(rgba(30,30,30,0.04) 1px,transparent 1px),
  linear-gradient(90deg,rgba(30,30,30,0.04) 1px,transparent 1px)}
html.light .group-table tr.qual-1{background:rgba(196,80,30,0.10)}
html.light .group-table tr.qual-2{background:rgba(26,122,110,0.08)}
html.light .match-row.winner{background:rgba(196,80,30,0.10)}
#wc-theme-toggle:hover{color:#C4501E;border-color:#C4501E}

/* ── Shared / FABs ──────────────────────────────────────────── */
#wc-fabs{position:fixed;bottom:24px;right:24px;z-index:9000;
  display:flex;flex-direction:column;gap:8px;align-items:flex-end}
.wc-fab{border:none;padding:10px 20px;font-size:11px;font-weight:700;
  cursor:pointer;letter-spacing:0.12em;text-transform:uppercase;
  white-space:nowrap;transition:transform .15s,box-shadow .15s;
  box-shadow:0 4px 20px rgba(0,0,0,.6)}
.wc-fab:hover{transform:translate(-2px,-2px)}
#wc-fab-live{background:#2E9A8C;color:#fff}
#wc-fab-live:hover{box-shadow:3px 3px 0 #E8DFC8}
#wc-fab-results{background:#D46030;color:#fff}
#wc-fab-results:hover{box-shadow:3px 3px 0 #E8DFC8}

/* ── Overlays ───────────────────────────────────────────────── */
.wc-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:9001}

/* ── Results modal ──────────────────────────────────────────── */
#wc-modal{position:fixed;top:50%;left:50%;z-index:9002;
  transform:translate(-50%,-50%);background:var(--surface);color:var(--text);
  border:1px solid var(--border);box-shadow:0 24px 64px rgba(0,0,0,.8);
  width:min(700px,96vw);max-height:88vh;overflow:hidden;
  display:none;flex-direction:column}
#wc-modal.wc-open{display:flex}
#wc-mhdr{display:flex;align-items:center;justify-content:space-between;
  padding:15px 20px;background:var(--surface-2);
  border-bottom:1px solid var(--border);flex-shrink:0}
#wc-mhdr h2{margin:0;font-size:13px;font-weight:700;letter-spacing:0.14em;
  text-transform:uppercase;color:var(--orange)}
.wc-mclose{background:none;border:none;color:var(--text-3);font-size:20px;
  cursor:pointer;line-height:1;padding:0 4px}
.wc-mclose:hover{color:var(--text)}
#wc-tabs{display:flex;border-bottom:1px solid var(--border);
  background:var(--surface-2);flex-shrink:0}
.wc-tab{flex:1;padding:10px;background:none;border:none;color:var(--text-2);
  font-size:11px;cursor:pointer;font-weight:700;letter-spacing:0.1em;
  text-transform:uppercase;border-bottom:2px solid transparent}
.wc-tab.active{color:var(--orange);border-bottom-color:var(--orange)}
#wc-body{overflow-y:auto;flex:1;padding:16px 20px;background:var(--surface)}
#wc-gnav{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.wc-gbtn{padding:4px 12px;border:1px solid var(--border);
  background:var(--surface-2);color:var(--text-2);cursor:pointer;
  font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase}
.wc-gbtn.active{background:var(--orange);color:#fff;border-color:var(--orange)}
.wc-match{display:grid;grid-template-columns:1fr 44px 14px 44px 1fr 68px;
  align-items:center;gap:6px;margin-bottom:8px}
.wc-tn{font-size:12px;line-height:1.3;color:var(--text-2)}
.wc-tn.right{text-align:right}
.wc-si{width:38px;padding:5px 4px;text-align:center;
  border:1px solid var(--border);background:var(--bg, #1C1A15);
  color:var(--text);font-size:14px;font-weight:700;-moz-appearance:textfield}
.wc-si::-webkit-inner-spin-button,.wc-si::-webkit-outer-spin-button{display:none}
.wc-si:focus{outline:none;border-color:var(--orange)}
.wc-sep{text-align:center;color:var(--text-3);font-weight:700}
.wc-actions{display:flex;gap:4px}
.wc-save{padding:4px 10px;border:none;
  background:var(--surface-2);color:var(--text-2);cursor:pointer;
  font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase}
.wc-save:hover{background:var(--border);color:var(--text)}
.wc-save.saved{background:rgba(46,154,140,0.2);color:var(--teal)}
.wc-clr{padding:4px 6px;border:none;background:transparent;
  color:var(--text-3);cursor:pointer;font-size:13px}
.wc-clr:hover{color:#c05050}
.wc-match.has-r .wc-tn{color:var(--orange)}
#wc-ko-add{background:var(--surface-2);border:1px solid var(--border);
  padding:12px;margin-bottom:14px}
#wc-ko-add h4{margin:0 0 10px;font-size:10px;font-weight:700;
  letter-spacing:0.14em;text-transform:uppercase;color:var(--text-2)}
.wc-ko-row{display:grid;grid-template-columns:1fr 40px 16px 40px 1fr;
  gap:6px;align-items:center;margin-bottom:8px}
.wc-ko-row input{padding:6px 8px;border:1px solid var(--border);
  background:#1C1A15;color:var(--text);font-size:12px;width:100%;box-sizing:border-box}
.wc-ko-row input:focus{outline:none;border-color:var(--orange)}
.wc-ko-si{text-align:center!important}
#wc-ko-rnd{padding:7px 10px;border:1px solid var(--border);
  background:#1C1A15;color:var(--text);font-size:11px;font-weight:700;
  letter-spacing:0.08em;text-transform:uppercase;width:100%}
#wc-ko-pens{display:none;margin-top:6px}
#wc-ko-pens label{font-size:11px;color:var(--text-2)}
#wc-ko-pens input{padding:6px 8px;border:1px solid var(--border);
  background:#1C1A15;color:var(--text);font-size:12px;width:100%;
  margin-top:4px;box-sizing:border-box}
.wc-add-btn{margin-top:8px;padding:7px 16px;border:none;
  background:var(--teal);color:#fff;cursor:pointer;font-size:10px;
  font-weight:700;letter-spacing:0.1em;text-transform:uppercase}
.wc-add-btn:hover{background:#268a7e}
.wc-ko-entry{display:flex;align-items:center;justify-content:space-between;
  padding:8px 10px;background:var(--surface-2);border:1px solid var(--border);
  margin-bottom:6px}
.wc-ko-entry span{font-size:12px;color:var(--text-2)}
.wc-ko-rnd-lbl{font-size:10px;font-weight:700;letter-spacing:0.1em;
  text-transform:uppercase;color:var(--text-3);display:block;margin-bottom:2px}
.wc-del{background:none;border:none;color:var(--text-3);cursor:pointer;
  font-size:16px;padding:0 4px;flex-shrink:0}
.wc-del:hover{color:#c05050}
#wc-footer{padding:12px 20px;background:var(--surface-2);
  border-top:1px solid var(--border);
  display:flex;align-items:center;gap:10px;flex-shrink:0}
#wc-regen{background:var(--orange);color:#fff;border:none;
  padding:8px 18px;font-weight:700;cursor:pointer;font-size:10px;
  letter-spacing:0.12em;text-transform:uppercase;white-space:nowrap;flex-shrink:0;
  transition:transform .15s,box-shadow .15s}
#wc-regen:hover{transform:translate(-1px,-1px);box-shadow:2px 2px 0 #E8DFC8}
#wc-regen:disabled{background:var(--surface-2);color:var(--text-3);
  cursor:default;transform:none;box-shadow:none}
#wc-smsg{flex:1;font-size:12px;color:var(--text-2);min-width:0}

/* ── Live tables panel ──────────────────────────────────────── */
#wc-live-panel{position:fixed;inset:0;z-index:9002;background:var(--bg,#1C1A15);
  display:none;flex-direction:column;overflow:hidden}
#wc-live-panel.wc-open{display:flex}
#wc-live-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:14px 20px;background:var(--surface-2,#332E26);
  border-bottom:1px solid var(--border,#3E3830);flex-shrink:0}
#wc-live-hdr h2{margin:0;font-size:13px;font-weight:700;letter-spacing:0.14em;
  text-transform:uppercase;color:var(--teal)}
#wc-live-hdr .wc-live-sub{font-size:11px;color:var(--text-3);margin-left:10px}
#wc-live-body{overflow-y:auto;flex:1;padding:16px;
  display:grid;gap:14px;
  grid-template-columns:repeat(auto-fill,minmax(280px,1fr))}
.wc-gc{background:var(--surface,#2A2620);border:1px solid var(--border,#3E3830);overflow:hidden}
.wc-gc-hdr{background:var(--surface-2,#332E26);padding:9px 14px;
  font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
  color:var(--teal);display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid var(--border,#3E3830)}
.wc-gc-hdr span{font-size:10px;color:var(--text-3);font-weight:400;letter-spacing:0}
.wc-gt{width:100%;border-collapse:collapse;font-size:11px}
.wc-gt th{padding:5px 5px;text-align:center;color:var(--text-3);
  font-weight:700;font-size:9px;letter-spacing:0.1em;text-transform:uppercase;
  border-bottom:1px solid var(--border)}
.wc-gt th.left{text-align:left}
.wc-gt td{padding:6px 5px;text-align:center;color:var(--text-2)}
.wc-gt td.left{text-align:left}
.wc-gt tbody tr:not(:last-child) td{border-bottom:1px solid var(--border)}
.wc-gt tbody tr:hover td{background:rgba(232,223,200,.03)}
.wc-pos-q1{color:var(--orange)!important;font-weight:700}
.wc-pos-q3{color:var(--text-2)!important;font-weight:700}
.wc-pos-q4{color:var(--text-3)!important}
.wc-row-q1{background:rgba(212,96,48,.06)}
.wc-row-q3{background:rgba(232,223,200,.02)}
.wc-tname{max-width:110px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  color:var(--text)}
.wc-pts{font-weight:700;color:var(--text)!important}
.wc-fx-list{padding:8px 10px 10px;border-top:1px solid var(--border)}
.wc-fx{display:grid;grid-template-columns:1fr auto 1fr;
  gap:4px;align-items:center;padding:3px 0;font-size:11px}
.wc-fx-tn{color:var(--text-3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.wc-fx-tn.r{text-align:right}
.wc-fx-score{color:var(--orange);font-weight:700;font-size:12px;
  text-align:center;white-space:nowrap;padding:0 6px}
.wc-fx-vs{color:var(--text-3);font-size:10px;text-align:center;padding:0 6px}
.wc-fx.played .wc-fx-tn{color:var(--text-2)}
</style>

<!-- Theme toggle -->
<button id="wc-theme-toggle" onclick="wcToggleTheme()" title="Toggle light/dark mode" style="position:fixed;top:16px;right:20px;z-index:9000;width:36px;height:36px;display:flex;align-items:center;justify-content:center;cursor:pointer;background:var(--surface);border:1px solid var(--border);color:var(--text-2);transition:color .15s,border-color .15s">
  <svg id="wc-icon-moon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
  <svg id="wc-icon-sun" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
</button>

<!-- FAB buttons -->
<div id="wc-fabs">
  <button class="wc-fab" id="wc-fab-live" onclick="wcLiveOpen()">📊 Live Tables</button>
  <button class="wc-fab" id="wc-fab-results" onclick="wcOpen()">📋 Enter Results</button>
</div>

<!-- Results modal overlay -->
<div class="wc-overlay" id="wc-overlay" onclick="wcClose()"></div>

<!-- Results modal -->
<div id="wc-modal">
  <div id="wc-mhdr">
    <h2>Match Results</h2>
    <button class="wc-mclose" onclick="wcClose()">✕</button>
  </div>
  <div id="wc-tabs">
    <button class="wc-tab active" onclick="wcTab('groups',this)">Group Stage</button>
    <button class="wc-tab" onclick="wcTab('knockout',this)">Knockout Rounds</button>
  </div>
  <div id="wc-body">
    <div id="wc-pg">
      <div id="wc-gnav"></div>
      <div id="wc-gm"></div>
    </div>
    <div id="wc-pk" style="display:none">
      <div id="wc-ko-add">
        <h4>Add knockout result</h4>
        <div class="wc-ko-row">
          <input id="wc-kt1" placeholder="Team 1 name">
          <input id="wc-ks1" type="number" min="0" max="20" placeholder="0" class="wc-ko-si">
          <span class="wc-sep">–</span>
          <input id="wc-ks2" type="number" min="0" max="20" placeholder="0" class="wc-ko-si">
          <input id="wc-kt2" placeholder="Team 2 name">
        </div>
        <select id="wc-ko-rnd">
          <option value="r32">Round of 32</option>
          <option value="r16">Round of 16</option>
          <option value="quarter">Quarter-Finals</option>
          <option value="semi">Semi-Finals</option>
          <option value="final">Final</option>
        </select>
        <div id="wc-ko-pens">
          <label>Draw after 90 min — who won on penalties?</label>
          <input id="wc-kpw" placeholder="Penalty winner">
        </div>
        <button class="wc-add-btn" onclick="wcSaveKo()">Add Result</button>
      </div>
      <div id="wc-kl"></div>
    </div>
  </div>
  <div id="wc-footer">
    <div id="wc-smsg"></div>
    <button id="wc-regen" onclick="wcRegen()">🔄 Regenerate Predictions</button>
  </div>
</div>

<!-- Live tables full-screen panel -->
<div id="wc-live-panel">
  <div id="wc-live-hdr">
    <div style="display:flex;align-items:baseline;gap:8px">
      <h2>📊 Live Tables & Fixtures</h2>
      <span class="wc-live-sub" id="wc-live-sub"></span>
    </div>
    <button class="wc-mclose" onclick="wcLiveClose()">✕</button>
  </div>
  <div id="wc-live-body"></div>
</div>

<script>
(function(){
  var stored=localStorage.getItem('wc-theme');
  var preferLight=window.matchMedia('(prefers-color-scheme: light)').matches;
  if(stored==='light'||(stored===null&&preferLight)){
    document.documentElement.classList.add('light');
    document.getElementById('wc-icon-moon').style.display='none';
    document.getElementById('wc-icon-sun').style.display='';
  }
})();
function wcToggleTheme(){
  var isLight=document.documentElement.classList.toggle('light');
  localStorage.setItem('wc-theme',isLight?'light':'dark');
  document.getElementById('wc-icon-moon').style.display=isLight?'none':'';
  document.getElementById('wc-icon-sun').style.display=isLight?'':'none';
}
(function(){
const D=__WC_DATA__;
const N=__N_SIMS__;
const dn=t=>D.displayNames[t]||t;
let R={group_results:{},knockout_results:{}};
let CG='A';
let liveOpen=false;

// ── ESC to close ──────────────────────────────────────────────
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){
    if(liveOpen) wcLiveClose();
    else wcClose();
  }
});

// ── Results modal ─────────────────────────────────────────────
function wcOpen(){
  document.getElementById('wc-overlay').style.display='block';
  document.getElementById('wc-modal').classList.add('wc-open');
  wcLoad();
}
function wcClose(){
  document.getElementById('wc-overlay').style.display='none';
  document.getElementById('wc-modal').classList.remove('wc-open');
}
window.wcOpen=wcOpen; window.wcClose=wcClose;

function wcTab(name,btn){
  document.querySelectorAll('.wc-tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('wc-pg').style.display=name==='groups'?'':'none';
  document.getElementById('wc-pk').style.display=name==='knockout'?'':'none';
  if(name==='knockout') wcRenderKoList();
}
window.wcTab=wcTab;

async function wcLoad(){
  R=await(await fetch('/api/results')).json();
  wcBuildNav(); wcRenderGroup(CG);
}

function wcBuildNav(){
  const nav=document.getElementById('wc-gnav');
  nav.innerHTML='';
  Object.keys(D.groups).forEach(letter=>{
    const n=(R.group_results||{})[letter]?.length||0;
    const btn=document.createElement('button');
    btn.className='wc-gbtn'+(letter===CG?' active':'');
    btn.textContent=`${letter} ${n}/6`;
    btn.onclick=()=>{
      CG=letter;
      document.querySelectorAll('.wc-gbtn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      wcRenderGroup(letter);
    };
    nav.appendChild(btn);
  });
}

function wcRenderGroup(letter){
  const el=document.getElementById('wc-gm');
  const played=(R.group_results||{})[letter]||[];
  el.innerHTML='';
  D.pairs[letter].forEach(([t1,t2])=>{
    const m=played.find(x=>(x.home===t1&&x.away===t2)||(x.home===t2&&x.away===t1));
    let hs='',as_='';
    if(m){
      hs=m.home===t1?m.home_score:m.away_score;
      as_=m.home===t1?m.away_score:m.home_score;
    }
    const uid=(t1+t2).replace(/\W/g,'');
    const row=document.createElement('div');
    row.className='wc-match'+(m?' has-r':'');
    row.innerHTML=`
      <span class="wc-tn right">${dn(t1)}</span>
      <input class="wc-si" id="h${uid}" type="number" min="0" max="20" value="${hs}" placeholder="–">
      <span class="wc-sep">–</span>
      <input class="wc-si" id="a${uid}" type="number" min="0" max="20" value="${as_}" placeholder="–">
      <span class="wc-tn">${dn(t2)}</span>
      <div class="wc-actions">
        <button class="wc-save${m?' saved':''}" id="sb${uid}"
          onclick="wcSaveG('${letter}','${t1}','${t2}','h${uid}','a${uid}','sb${uid}')">
          ${m?'✓ Saved':'Save'}</button>
        <button class="wc-clr" title="Clear result"
          onclick="wcClearG('${letter}','${t1}','${t2}')">✕</button>
      </div>`;
    el.appendChild(row);
  });
}

async function wcSaveG(letter,t1,t2,hId,aId,btnId){
  const hs=document.getElementById(hId).value;
  const as_=document.getElementById(aId).value;
  if(hs===''||as_===''){alert('Enter both scores first.');return;}
  const r=await fetch('/api/results/group',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({letter,home:t1,away:t2,
      home_score:parseInt(hs),away_score:parseInt(as_)})});
  const d=await r.json();
  if(d.error){alert(d.error);return;}
  const btn=document.getElementById(btnId);
  btn.textContent='✓ Saved'; btn.classList.add('saved');
  btn.closest('.wc-match').classList.add('has-r');
  R=await(await fetch('/api/results')).json();
  wcBuildNav();
  if(liveOpen) wcLiveRefresh();
}
window.wcSaveG=wcSaveG;

async function wcClearG(letter,t1,t2){
  await fetch('/api/results/group',{method:'DELETE',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({letter,home:t1,away:t2})});
  R=await(await fetch('/api/results')).json();
  wcBuildNav(); wcRenderGroup(letter);
  if(liveOpen) wcLiveRefresh();
}
window.wcClearG=wcClearG;

// ── Knockout ──────────────────────────────────────────────────
['wc-ks1','wc-ks2'].forEach(id=>{
  document.getElementById(id).addEventListener('input',()=>{
    const s1=document.getElementById('wc-ks1').value;
    const s2=document.getElementById('wc-ks2').value;
    document.getElementById('wc-ko-pens').style.display=
      (s1!==''&&s2!==''&&parseInt(s1)===parseInt(s2))?'block':'none';
  });
});

async function wcSaveKo(){
  const t1=document.getElementById('wc-kt1').value.trim();
  const s1=document.getElementById('wc-ks1').value;
  const s2=document.getElementById('wc-ks2').value;
  const t2=document.getElementById('wc-kt2').value.trim();
  const rnd=document.getElementById('wc-ko-rnd').value;
  const pw=document.getElementById('wc-kpw').value.trim();
  if(!t1||!t2||s1===''||s2===''){alert('Fill in all fields.');return;}
  const body={round:rnd,team1:t1,team2:t2,
    team1_score:parseInt(s1),team2_score:parseInt(s2)};
  if(pw) body.winner=pw;
  const r=await fetch('/api/results/knockout',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();
  if(d.error){alert(d.error);return;}
  ['wc-kt1','wc-ks1','wc-ks2','wc-kt2','wc-kpw'].forEach(id=>{
    document.getElementById(id).value='';});
  document.getElementById('wc-ko-pens').style.display='none';
  R=await(await fetch('/api/results')).json();
  wcRenderKoList();
  if(liveOpen) wcLiveRefresh();
}
window.wcSaveKo=wcSaveKo;

function wcRenderKoList(){
  const el=document.getElementById('wc-kl');
  el.innerHTML='';
  const kr=R.knockout_results||{};
  const labels=D.knockoutLabels;
  let any=false;
  D.knockoutRounds.forEach(rnd=>{
    (kr[rnd]||[]).forEach(m=>{
      any=true;
      const div=document.createElement('div');
      div.className='wc-ko-entry';
      div.innerHTML=`<div>
        <span class="wc-ko-rnd-lbl">${labels[rnd]}</span>
        <span><strong>${dn(m.team1)} ${m.team1_score}–${m.team2_score} ${dn(m.team2)}</strong>
        → <span style="color:#f59e0b">${dn(m.winner)}</span></span>
      </div>
      <button class="wc-del" onclick="wcDelKo('${rnd}','${m.team1}','${m.team2}')">✕</button>`;
      el.appendChild(div);
    });
  });
  if(!any){
    el.innerHTML='<p style="color:#6b7280;font-size:12px;margin:0">No knockout results entered yet.</p>';
  }
}

async function wcDelKo(rnd,t1,t2){
  await fetch('/api/results/knockout',{method:'DELETE',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({round:rnd,team1:t1,team2:t2})});
  R=await(await fetch('/api/results')).json();
  wcRenderKoList();
  if(liveOpen) wcLiveRefresh();
}
window.wcDelKo=wcDelKo;

// ── Regenerate ────────────────────────────────────────────────
async function wcRegen(){
  const btn=document.getElementById('wc-regen');
  const msg=document.getElementById('wc-smsg');
  btn.disabled=true; msg.textContent='Starting…';
  await fetch('/api/simulate',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({n_sims:N})});
  wcPoll();
}
window.wcRegen=wcRegen;

async function wcPoll(){
  const msg=document.getElementById('wc-smsg');
  const btn=document.getElementById('wc-regen');
  const s=await(await fetch('/api/status')).json();
  msg.textContent=s.message;
  if(s.running) setTimeout(wcPoll,2000);
  else if(s.message==='done'){
    msg.textContent='Done! Reloading…';
    setTimeout(()=>location.reload(),700);
  } else {
    btn.disabled=false;
    if(s.error) msg.style.color='#f87171';
  }
}

// ── Live tables ───────────────────────────────────────────────
function wcLiveOpen(){
  liveOpen=true;
  document.getElementById('wc-live-panel').classList.add('wc-open');
  wcLiveRefresh();
}
function wcLiveClose(){
  liveOpen=false;
  document.getElementById('wc-live-panel').classList.remove('wc-open');
}
window.wcLiveOpen=wcLiveOpen; window.wcLiveClose=wcLiveClose;

async function wcLiveRefresh(){
  const data=await(await fetch('/api/live')).json();
  wcRenderLive(data);
}

function wcRenderLive(data){
  const body=document.getElementById('wc-live-body');
  body.innerHTML='';

  // Summary line: how many results entered total
  let total=0;
  Object.values(data).forEach(g=>g.fixtures.forEach(f=>{if(f.played)total++;}));
  document.getElementById('wc-live-sub').textContent=
    total ? `${total} result${total!==1?'s':''} entered` : 'No results entered yet';

  Object.entries(data).forEach(([letter,group])=>{
    const card=document.createElement('div');
    card.className='wc-gc';

    const played=group.fixtures.filter(f=>f.played).length;
    card.innerHTML=`<div class="wc-gc-hdr">GROUP ${letter}<span>${played}/6 played</span></div>`;

    // Standings table
    const tbl=document.createElement('table');
    tbl.className='wc-gt';
    tbl.innerHTML=`<thead><tr>
      <th style="width:20px"></th>
      <th class="left">Team</th>
      <th>P</th><th>W</th><th>D</th><th>L</th>
      <th>GF</th><th>GA</th><th>GD</th><th>Pts</th>
    </tr></thead>`;
    const tbody=document.createElement('tbody');
    group.standings.forEach((row,i)=>{
      const posCls=i<2?'wc-pos-q1':i===2?'wc-pos-q3':'wc-pos-q4';
      const rowCls=i<2?'wc-row-q1':i===2?'wc-row-q3':'';
      const gd=row.gd>0?`+${row.gd}`:row.gd;
      const tr=document.createElement('tr');
      if(rowCls) tr.className=rowCls;
      tr.innerHTML=`
        <td class="${posCls}">${i+1}</td>
        <td class="left"><span class="wc-tname" title="${dn(row.team)}">${dn(row.team)}</span></td>
        <td>${row.p}</td><td>${row.w}</td><td>${row.d}</td><td>${row.l}</td>
        <td>${row.gf}</td><td>${row.ga}</td><td>${gd}</td>
        <td class="wc-pts">${row.pts}</td>`;
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
    card.appendChild(tbl);

    // Fixtures
    const fxDiv=document.createElement('div');
    fxDiv.className='wc-fx-list';
    group.fixtures.forEach(f=>{
      const row=document.createElement('div');
      row.className='wc-fx'+(f.played?' played':'');
      if(f.played){
        row.innerHTML=`
          <span class="wc-fx-tn r">${dn(f.t1)}</span>
          <span class="wc-fx-score">${f.s1} – ${f.s2}</span>
          <span class="wc-fx-tn">${dn(f.t2)}</span>`;
      } else {
        row.innerHTML=`
          <span class="wc-fx-tn r">${dn(f.t1)}</span>
          <span class="wc-fx-vs">v</span>
          <span class="wc-fx-tn">${dn(f.t2)}</span>`;
      }
      fxDiv.appendChild(row);
    });
    card.appendChild(fxDiv);
    body.appendChild(card);
  });
}

// Resume poll if sim already running
(async()=>{
  const s=await(await fetch('/api/status')).json();
  if(s.running){
    document.getElementById('wc-regen').disabled=true;
    document.getElementById('wc-smsg').textContent=s.message;
    wcPoll();
  }
})();

})();
</script>
"""


def _build_injection() -> str:
    pairs = {
        letter: [[t1, t2] for t1, t2 in combinations(teams, 2)]
        for letter, teams in GROUPS.items()
    }
    wc_data = json.dumps({
        "groups":         {k: list(v) for k, v in GROUPS.items()},
        "pairs":          pairs,
        "displayNames":   DISPLAY_NAMES,
        "knockoutRounds": KNOCKOUT_ROUNDS,
        "knockoutLabels": {
            "r32": "Round of 32", "r16": "Round of 16",
            "quarter": "Quarter-Finals", "semi": "Semi-Finals", "final": "Final",
        },
    })
    return (_INJECTION_TEMPLATE
            .replace("__WC_DATA__", wc_data)
            .replace("__N_SIMS__", str(_DEFAULT_SIMS)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _DEFAULT_SIMS
    parser = argparse.ArgumentParser(description="2026 WC prediction web server")
    parser.add_argument("--quick", action="store_true",
                        help="Use 1 000 simulations (faster, less accurate)")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--host",  default="127.0.0.1")
    args = parser.parse_args()

    if args.quick:
        _DEFAULT_SIMS = 1_000

    print(f"\n  2026 World Cup Prediction Server")
    print(f"  Simulations per run: {_DEFAULT_SIMS:,}")
    print(f"  Open  http://{args.host}:{args.port}  in your browser")
    print(f"  Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
