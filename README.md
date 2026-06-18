# 2026 FIFA World Cup Predictions

A Monte Carlo tournament simulator for the 2026 FIFA World Cup, powered by a Dixon-Coles statistical model fitted on historical international results.

## How it works

1. **Model fitting** — loads `data/results.csv` (international matches from 1993 onward), applies exponential time decay and per-competition weights, then fits a Dixon-Coles Poisson model to estimate each team's attack and defence parameters.
2. **Simulation** — runs thousands of full tournament simulations (group stage through the final), sampling scorelines from the model. Group tiebreakers follow FIFA rules (points → GD → GF → H2H → random draw). Rather than computing exact probabilities analytically (which would be combinatorially enormous), each simulation plays the entire tournament end-to-end using weighted random draws. After 10,000 runs, probabilities are read directly from frequencies — if France wins in 1,847 simulations, their estimated win probability is 18.5%. For draws in knockout matches, extra time is simulated at 35% of the normal goal rate, then penalties as a 50/50 coin flip.
3. **Live updating** — as actual results come in, they are locked into the simulation so only future matches are simulated, keeping predictions current throughout the tournament.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## CLI usage

```bash
# Full run — 10 000 simulations (~60 s), saves predictions.html
python main.py

# Quick preview — 1 000 simulations
python main.py --quick

# Print team ratings only (no simulation)
python main.py --ratings

# Higher precision
python main.py --sims 50000
```

### Entering actual results (CLI)

```bash
python enter_results.py                   # interactive menu
python enter_results.py --group A         # enter/update one group
python enter_results.py --knockout r32    # enter a knockout round
python enter_results.py --show            # view all entered results
python enter_results.py --clear           # wipe all results
```

After entering results, re-run `python main.py --quick` to get updated predictions.

## Web UI

```bash
python server.py              # full precision (10 000 sims)
python server.py --quick      # 1 000 sims, faster first load
python server.py --port 8080  # custom port
```

Open `http://localhost:5000` in a browser. The UI provides:

- **Live Tables** — real-time group standings and fixtures as results are entered
- **Enter Results** — click-to-save score entry for group and knockout matches; draws prompt for a penalty winner
- **Regenerate Predictions** — refits the model and reruns simulations in the background, reloading the page when done

## Output

`main.py` prints a ranked table of win/final/semi/quarter-final probabilities for all 48 teams, and writes a self-contained `predictions.html` wall chart. The web server serves this chart with the live UI injected on top.

## Data

Historical match data lives in `data/`:

| File | Contents |
|---|---|
| `results.csv` | International results (source: [martj42/international-football-results](https://github.com/martj42/international-football-results)) |
| `former_names.csv` | Historical team name aliases mapped to current names |
| `goalscorers.csv` | Goalscorer data (not used by the model) |
| `shootouts.csv` | Penalty shootout data (not used by the model) |

Actual tournament results are stored locally in `actual_results.json`.

## Project structure

```
main.py              CLI entry point
server.py            Flask web server
enter_results.py     CLI result entry tool
src/
  dixon_coles.py     Dixon-Coles model (fitting + scoreline sampling)
  simulator.py       Monte Carlo tournament engine
  preprocessing.py   Data loading, time decay, tournament weights
  fixtures.py        2026 group definitions and team display names
  tournament_state.py  Load/save actual results, seed simulations
  chart.py           HTML wall chart generator
data/                Historical match data (CSV)
actual_results.json  Entered actual results (auto-created)
predictions.html     Generated wall chart (auto-created)
```
