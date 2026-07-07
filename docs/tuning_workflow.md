# MetalFish tuning workflow

A repeatable loop for making the engine *measurably* stronger and for turning
Lichess game data into validated default settings. Every change is tested
against the **current engine as the baseline** before it becomes a default.

The loop:

```
analyze games  ->  form hypothesis  ->  candidate config  ->  SPRT vs baseline
     ^                                                              |
     |                                                              v
 lock in default  <-  update defaults + CI  <-  statistically better? --no--> discard
```

## 1. Analyze the games

`tools/analyze_lichess_games.py` summarizes the audit logs the bot writes under
`results/lichess_audit/` (and the seek log `results/lichess_seek_audit.jsonl`).

```bash
# Aggregate every audit file, machine-readable
python3 tools/analyze_lichess_games.py --limit 5000 --json > /tmp/games.json
# Human summary incl. draw telemetry and score extremes
python3 tools/analyze_lichess_games.py --limit 5000 \
    --seek-audit results/lichess_seek_audit.jsonl
```

Each game record exposes `result`, `max_score_cp`/`min_score_cp` (engine eval
extremes), `decisive_score_samples`, draw offer/accept counts, and stream
health. That is enough to classify *why* points are lost:

- **Conversion failures** — `result == draw` while `max_score_cp` was large.
- **Drawing machine** — `result == draw` while the eval never left `±150cp`.
- **Blunders** — `result == loss` while `max_score_cp` was clearly winning.
- **Bot bugs** — non-empty `issues` (illegal moves, ponder failures, rejects).

### What the current corpus says (1024 games)

~87% of games are draws, and **566 of them stayed within `±50cp` the entire
game** — the engine reaches genuinely equal positions and is content to draw
them. It essentially never squanders a winning position (no drawn game exceeded
`+300cp`), and the 22 losses all drifted from *equal* to worse (max eval never
above `+93cp` before losing). The dominant Elo cap is therefore **contentment
with equality**, not tactical blunders. The two levers that follow attack that
directly.

## 2. The fighting levers

### `MCTSContempt` (engine, opt-in, default `0`)

The MCTS/hybrid search already backs up a draw-score term
(`GetQ(draw_score)` in `src/mcts/node.h`); `src/mcts/search.cpp` converts
`contempt` into `draw_score = -contempt / 10000`. `MCTSContempt` (registered in
`src/uci/engine.cpp`, wired in `make_mcts_config` in `src/uci/uci.cpp`) exposes
it for both the pure-MCTS and hybrid paths.

- Positive → avoid draws (penalize high-draw-probability nodes).
- `0` → historical draw-neutral behavior (the default; no behavior change).
- Rough scale: `contempt/10000 * draw_prob` in win-probability units, so
  `contempt=300` ≈ up to ~9cp against a fully-drawn line, `1000` ≈ ~30cp.

Per repo policy, this stays `0` until a candidate value is proven stronger by
the workflow below.

### Fighting draw policy (bot, `METALFISH_DRAW_FIGHTING`, default on)

`tools/lichess_bot.py` no longer offers or accepts draws in playable positions.
It only *accepts* when clearly losing or dead (insufficient material / forced
rule / tablebase loss-or-draw / mated), and only *offers* as a swindle when
already losing. Set `METALFISH_DRAW_FIGHTING=0` to restore the lenient policy.

## 3. Test a candidate against the baseline

`tools/tuning_workflow.py` plays each candidate config in a colour-balanced
match against the baseline (the same binary + default options, i.e. the current
engine) and applies SPRT.

```bash
# Fast search-parameter screening in Alpha-Beta (thousands of games/hour)
python3 tools/tuning_workflow.py --mode ab --movetime 100 \
    --candidates tools/tuning_candidates.json --max-games 400

# The hybrid engine the bot actually plays (slow; run overnight / lower budget)
python3 tools/tuning_workflow.py --mode hybrid --movetime 800 \
    --candidates tools/tuning_candidates.json --max-games 200

# A code change (new binary) vs the committed baseline binary
python3 tools/tuning_workflow.py --mode hybrid \
    --engine build/metalfish --candidate-engine build_candidate/metalfish \
    --candidates tools/tuning_candidates.json
```

Candidate configs are JSON (`tools/tuning_candidates.json` ships the
`MCTSContempt` sweep):

```json
{
  "elo0": 0.0, "elo1": 5.0,
  "candidates": [
    {"label": "contempt-300", "options": {"MCTSContempt": "300"}, "note": "..."}
  ]
}
```

Output is a ranked table with Elo ± 95% CI, W-D-L, draw ratio, and the SPRT
status, plus a machine-readable copy under `results/tuning/`. A candidate is
**recommended as the new default only if** it passes SPRT (`H1`) or its 95%
lower Elo bound is above `0` over a meaningful sample — otherwise the workflow
recommends keeping the current default.

### Methodology notes

- **SPRT / self-play.** Candidate vs a draw-neutral copy of itself is the
  standard first-order strength test (as on Fishtest). It measures whether the
  setting gains Elo head-to-head; vs-field Elo can differ, so confirm promising
  settings with a broader tournament (`tools/run_tournament_live.py`,
  `tools/run_cutechess_tournament.sh`) and, ultimately, live Lichess games.
- **Time control.** Prefer real TC (`--tc 10+0.1`) for time-management-sensitive
  changes; fixed `--movetime` is deterministic and best for search/eval params.
- **Cost.** Hybrid games run the GPU transformer (~1s+/move). Screen ideas in
  `--mode ab` first, then confirm the survivors in `--mode hybrid`.
- **Power.** ~200 games gives ±30–60 Elo error bars — enough to catch large
  regressions and screen candidates, not to resolve ±3 Elo. Report the sample
  size and CI honestly; do not claim a gain the interval does not support.

## 4. Lock in a validated default

When a candidate is validated:

1. Update the engine default (`Option(...)` in `src/uci/engine.cpp`) and/or the
   bot's `BASE_ENGINE_OPTIONS` in `tools/lichess_bot.py`.
2. Re-run the CI parity smokes locally (`tools/uci_smoke.py`) — the MCTS/hybrid
   fixed-node positions must still produce their expected best moves.
3. Commit with the results JSON referenced in the message, push to the PR
   branch, and confirm CI is green (`gh pr checks`).

## 5. Regression safety

- `tests/test_sprt_test.py` pins the SPRT harness (adjudication + Elo math).
- `tests/test_tuning_workflow.py` pins the candidate parsing, ranking, and
  recommendation logic (no games run).
- `tests/test_lichess_bot.py` covers the fighting draw policy in both modes.
- CI runs the MCTS/hybrid fixed-node smokes; `MCTSContempt=0` keeps them bit-
  for-bit identical, so shipping the option with a neutral default is safe.
