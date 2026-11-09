#!/usr/bin/env python3
"""Oracle study of hybrid arbitration: judge the AB move and the MCTS move on positions
where they disagreed, using strong reference engines (RQ6b of docs/paper_v2/02_evaluation_protocol.md,
with amendments §11.6 and §11.12).

Input: moves.jsonl from tools/paper/parse_hybrid_trace.py (records need fen, ab_move, mcts_move,
selected; `game` identifies the game for the clustered bootstrap). For each disagreement position
and each judge:
  1. unrestricted search  -> the judge's best move and score (mover's perspective, cp)
  2. a score for each candidate. Time-limited judges (Stockfish): one restricted search
     `searchmoves ab mcts` with MultiPV=2, which scores both candidates at equal depth.
     Node-limited judges (Lc0): the position after each candidate searched with the same node
     limit from a fresh tree (ucinewgame), negated -- equal effort per candidate (§11.12d).
Loss of a candidate = best_cp - candidate_cp, best_cp being the largest of the three scores.
Expected-score loss uses a logistic with --cp-scale (250 cp per unit of log-odds, §11.12d).

Output: per-position JSONL plus <out>.summary.json with, per judge, the fraction of positions where
AB's / MCTS's move is better / tie (|delta| <= --tie-cp), the mean loss of the AB move, the MCTS
move, the realised selection and the oracle arbiter (min of the two), the realised arbiter's
accuracy (its move better or tied), and the oracle-arbiter upper bound
mean(loss_selected) - mean(min loss) with its projected Elo (the logistic Elo of a score of
0.5 + bound, per disagreement position); with several judges, their agreement (Cohen's kappa for
two) and the bound on the positions where all judges find the move not selected better, positions
where they differ counted as undecided; and, in each direction between two judges (§11.14), the
cross-judge results: on the positions where the selecting judge rates the MCTS move better by more
than --tie-cp, the share on which the scoring judge does too, and the arbiter bound when the
selecting judge picks the candidate (its smaller loss; the realised selection on an exact tie) and
the scoring judge's losses score that pick against the realised selection, which removes the
optimism of a judge that both selects and scores. Estimates are means over strata weighted by the
recorded stratum population counts (a share among flagged positions is a ratio of two such means);
95% intervals are percentile bootstraps resampling games as clusters within strata (§11.12g), and
the number of games contributing positions is reported. A projected Elo is a labelled conversion
of the per-position bound, not a prediction of game results.

Sampling (protocol 02 §5 + amendment §11.6): disagreement positions are stratified by game
phase (opening = fullmove <= 15; endgame = at most 6 non-pawn, non-king pieces; else middlegame)
and by the tercile of the MCTS root visit count at decision time; allocation is proportional to
stratum size; `--no-stratify` gives a plain seeded random sample. The strata, cut points and
counts are written to <out>.summary.json.

Usage:
    python3 tools/paper/oracle_probe.py moves.jsonl --out oracle.jsonl \
        --judge sf19=reference/stockfish-sf19/src/stockfish --judge-option sf19:Threads=8 \
        --judge-option sf19:Hash=2048 --judge-option sf19:SyzygyPath=syzygy --movetime 60000 \
        [--judge lc0=/opt/homebrew/bin/lc0 --judge-arg lc0:--backend=metal --judge-arg lc0:--weights=... --nodes lc0:30000] \
        --sample 600 --seed 1
    # rewrite the summary from the judged rows already in --out (no engines):
    python3 tools/paper/oracle_probe.py moves.jsonl --out oracle.jsonl --judge sf19=- --judge lc0=- \
        --nodes lc0:30000 --summarise-only
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import logging
import math
import os
import pathlib
import random
import signal
import sys
import threading
import time
from typing import Callable, Dict, List, Optional

import chess
import chess.engine
import gpu_errors

KEEP_FIELDS = (
    "fen",
    "ab_move",
    "mcts_move",
    "selected",
    "reason",
    "mcts_playouts",
    "mcts_root_visits",
    "mcts_evals",
    "source",
    "engine",
    "game",
    "opening",
)

# Amendment 16 (b): a bound is within the tie band when the upper limit of its 95% interval is at
# most this, and beyond it when the lower limit exceeds it: the expected-score difference of the
# registered 10 cp tie threshold at an even position (10/250 x 1/4, logistic with 250 cp per unit
# of log-odds).
TIE_BAND_ES = 0.01


def source_cell(rec: Dict) -> str:
    """The cell a disagreement record came from, the directory of its source log
    (results/paper_v2/RQ1a/engines.log -> RQ1a), or "unknown" (amendment 16 (a))."""
    src = rec.get("source")
    return pathlib.PurePosixPath(src).parent.name if src else "unknown"


def logistic(cp: float, scale: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / scale))


def logistic_elo(es_gain: Optional[float]) -> Optional[float]:
    """Projected Elo of an expected-score gain: the logistic Elo of a score of 0.5 + gain
    (§11.12d). Per disagreement position, not per game."""
    if es_gain is None or not -0.5 < es_gain < 0.5:
        return None
    s = 0.5 + es_gain
    return 400.0 * math.log10(s / (1.0 - s))


def cp_of(score: chess.engine.PovScore) -> float:
    return float(score.relative.score(mate_score=30000))


def cp_after_move(
    eng, board: chess.Board, move: chess.Move, limit: chess.engine.Limit
) -> float:
    """Score of `move` for the side to move in `board`: the position after the move searched
    with `limit` from a fresh tree (a new game object makes python-chess send ucinewgame), from
    the mover's point of view. Checkmate and draws on the board are scored without a search.
    """
    child = board.copy(stack=False)
    child.push(move)
    outcome = child.outcome()
    if outcome is not None:
        if outcome.winner is None:
            return 0.0
        return 30000.0 if outcome.winner == board.turn else -30000.0
    info = eng.analyse(child, limit, game=object())
    return float(info["score"].pov(board.turn).score(mate_score=30000))


def judge_position(
    eng,
    board: chess.Board,
    cands: List[str],
    limit: chess.engine.Limit,
    per_candidate: bool = False,
) -> Dict:
    """The judge's best move and score, and a score for each candidate: with `per_candidate`,
    the position after each candidate searched with the same limit from a fresh tree (equal
    effort); otherwise one restricted multi-PV search over the candidates (equal depth).
    """
    best = eng.analyse(board, limit, game=object() if per_candidate else None)
    best_cp = cp_of(best["score"])
    best_move = best["pv"][0].uci() if best.get("pv") else None
    moves = [
        chess.Move.from_uci(m)
        for m in cands
        if chess.Move.from_uci(m) in board.legal_moves
    ]
    cand_cp: Dict[str, float] = {}
    if per_candidate:
        for m in moves:
            cand_cp[m.uci()] = cp_after_move(eng, board, m, limit)
    else:
        infos = eng.analyse(board, limit, multipv=len(moves), root_moves=moves)
        if isinstance(infos, dict):
            infos = [infos]
        for info in infos:
            if info.get("pv"):
                cand_cp[info["pv"][0].uci()] = cp_of(info["score"])
    return {
        "best_move": best_move,
        "best_cp": best_cp,
        "cand_cp": cand_cp,
        "depth": best.get("depth"),
        "nodes": best.get("nodes"),
    }


def score_candidates(j: Dict, r: Dict, tie_cp: float, cp_scale: float) -> Dict:
    """A judgment with each candidate's loss against the judge's best, in cp and expected
    score, and the verdict; an error when the judge did not score both candidates."""
    ab_cp = j["cand_cp"].get(r["ab_move"])
    mc_cp = j["cand_cp"].get(r["mcts_move"])
    if ab_cp is None or mc_cp is None:
        return {"error": "candidate not scored", **j}
    best = max(j["best_cp"], ab_cp, mc_cp)
    j.update(
        {
            "ab_cp": ab_cp,
            "mcts_cp": mc_cp,
            "delta_cp": ab_cp - mc_cp,
            "loss_ab_cp": best - ab_cp,
            "loss_mcts_cp": best - mc_cp,
            "loss_ab_es": logistic(best, cp_scale) - logistic(ab_cp, cp_scale),
            "loss_mcts_es": logistic(best, cp_scale) - logistic(mc_cp, cp_scale),
            "verdict": (
                "ab"
                if ab_cp - mc_cp > tie_cp
                else ("mcts" if mc_cp - ab_cp > tie_cp else "tie")
            ),
        }
    )
    return j


def write_rows(path: str, rows: List[Dict]) -> None:
    """Replace `path` with `rows` (JSONL) in one step, so it never holds a partial file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


STDERR_GRACE_S = 0.5  # after a search, time for the engine's stderr to arrive


class StderrWatch(logging.Handler):
    """Echoes the judges' stderr (python-chess logs it on `chess.engine`) and counts, per engine
    process, the lines that report a GPU error."""

    def __init__(self) -> None:
        super().__init__()
        self.by_engine: Dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        print(msg, file=sys.stderr)
        if gpu_errors.is_gpu_error(msg):
            args = record.args if isinstance(record.args, tuple) else ()
            who = str(args[0]) if args else ""
            self.by_engine[who] = self.by_engine.get(who, 0) + 1


def close_engine(eng) -> None:
    """Quit an engine; one that has died cannot, which is not an error here."""
    try:
        eng.quit()
    except Exception:  # noqa: BLE001
        try:
            eng.close()
        except Exception:  # noqa: BLE001
            pass


def kill_engine(eng) -> None:
    try:
        os.kill(eng.transport.get_pid(), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass


class Judge:
    """A judge engine, started again from scratch after it dies, overruns or reports a GPU error
    during a search (amendment 15 (i)). `spawn` starts and configures one engine process.
    """

    def __init__(self, name: str, spawn: Callable, watch: StderrWatch) -> None:
        self.name, self.spawn, self.watch = name, spawn, watch
        self.eng = spawn()

    def gpu_errors(self) -> int:
        return self.watch.by_engine.get(str(getattr(self.eng, "protocol", self.eng)), 0)

    def restart(self) -> None:
        close_engine(self.eng)
        self.eng = self.spawn()

    def close(self) -> None:
        close_engine(self.eng)


def judge_with_retries(
    judge: Judge,
    board: chess.Board,
    cands: List[str],
    limit: chess.engine.Limit,
    per_candidate: bool,
    retries: int,
    timeout: Optional[float] = None,
) -> Dict:
    """judge_position with up to `retries` further attempts, each on a fresh engine process, when
    the engine dies, takes longer than `timeout` seconds, or reports a GPU error on stderr during
    the search. Returns the judgment, or the last error, with the number of attempts made.
    """
    error = None
    for attempt in range(1, retries + 2):
        before = judge.gpu_errors()
        timer = None
        if timeout:
            timer = threading.Timer(timeout, kill_engine, (judge.eng,))
            timer.start()
        try:
            j = judge_position(judge.eng, board, cands, limit, per_candidate)
            judge.eng.ping()
            time.sleep(STDERR_GRACE_S)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        else:
            if judge.gpu_errors() == before:
                j["attempts"] = attempt
                return j
            error = "GPU error reported during the search"
        finally:
            if timer:
                timer.cancel()
        judge.restart()
    return {"error": error, "attempts": retries + 1}


def judge_config(path: str, jargs: List[str], options: List, limit: str) -> str:
    """The judge as configured: engine, arguments, UCI options and limit. Recorded in every
    judgment; a resumed run keeps only judgments made with the same configuration."""
    return " ".join(
        [os.path.basename(path), *jargs, *(f"{k}={v}" for k, v in options), limit]
    )


def cluster_key(rec: Dict) -> Optional[str]:
    """The game a trace record came from (source log, engine, game ordinal), or None."""
    if rec.get("game") is None:
        return None
    return f"{rec.get('source')}|{rec.get('engine')}|{rec['game']}"


def stratum_weights(rows: List[Dict], sampling: Dict) -> Dict[str, float]:
    """Weight of each stratum present in `rows`: its recorded population count, normalised over
    the strata present; without recorded strata, each stratum's share of the rows."""
    present = sorted({r.get("stratum") or "all" for r in rows})
    pop = sampling.get("strata_population") or {}
    if pop and all(s in pop for s in present):
        total = sum(pop[s] for s in present)
        return {s: pop[s] / total for s in present}
    counts = {
        s: sum(1 for r in rows if (r.get("stratum") or "all") == s) for s in present
    }
    return {s: counts[s] / len(rows) for s in present}


def cluster_units(rows: List[Dict], values: Dict[str, List[float]]) -> Dict[str, List]:
    """{stratum: [[n_positions, [sum of each statistic]], ...]}, one entry per game (a row
    without a game is its own cluster), in the order of `values`."""
    names = list(values)
    strata: Dict[str, Dict[str, List]] = {}
    for i, r in enumerate(rows):
        unit = strata.setdefault(r.get("stratum") or "all", {}).setdefault(
            r.get("cluster") or f"position:{i}", [0, [0.0] * len(names)]
        )
        unit[0] += 1
        for j, name in enumerate(names):
            unit[1][j] += values[name][i]
    return {s: list(units.values()) for s, units in strata.items()}


def weighted_means(
    units: Dict[str, List], weights: Dict[str, float], k: int
) -> List[float]:
    est = [0.0] * k
    for s, clusters in units.items():
        n = sum(c[0] for c in clusters)
        for j in range(k):
            est[j] += weights[s] * sum(c[1][j] for c in clusters) / n
    return est


def stratified_estimates(
    rows: List[Dict],
    values: Dict[str, List[float]],
    weights: Dict[str, float],
    n_boot: int,
    seed: int,
) -> Dict[str, Dict]:
    """Stratum-weighted mean of each statistic with a 95% percentile interval from a bootstrap
    that resamples clusters (games) with replacement within each stratum."""
    names = list(values)
    units = cluster_units(rows, values)
    point = weighted_means(units, weights, len(names))
    rng = random.Random(seed)
    reps: List[List[float]] = [[] for _ in names]
    for _ in range(n_boot):
        drawn = {s: rng.choices(cl, k=len(cl)) for s, cl in sorted(units.items())}
        for j, v in enumerate(weighted_means(drawn, weights, len(names))):
            reps[j].append(v)
    out: Dict[str, Dict] = {}
    for j, name in enumerate(names):
        r = sorted(reps[j])
        ci = (
            [r[int(0.025 * n_boot)], r[min(n_boot - 1, int(0.975 * n_boot))]]
            if r
            else None
        )
        out[name] = {
            "estimate": point[j],
            "ci95": ci,
            "sample_mean": sum(values[name]) / len(rows),
        }
    return out


def stratified_ratio(
    rows: List[Dict],
    num: List[float],
    den: List[float],
    weights: Dict[str, float],
    n_boot: int,
    seed: int,
) -> Dict:
    """Ratio of two stratum-weighted means (a share among the positions `den` flags) with a 95%
    percentile interval from the same cluster bootstrap; a replicate that draws no flagged
    position has no ratio and is left out (the number of replicates used is reported).
    """
    units = cluster_units(rows, {"num": num, "den": den})
    n, d = weighted_means(units, weights, 2)
    rng = random.Random(seed)
    reps: List[float] = []
    for _ in range(n_boot):
        drawn = {s: rng.choices(cl, k=len(cl)) for s, cl in sorted(units.items())}
        bn, bd = weighted_means(drawn, weights, 2)
        if bd > 0:
            reps.append(bn / bd)
    reps.sort()
    k = len(reps)
    return {
        "estimate": n / d if d > 0 else None,
        "ci95": (
            [reps[int(0.025 * k)], reps[min(k - 1, int(0.975 * k))]] if reps else None
        ),
        "replicates": k,
        "sample": {"flagged": int(sum(den)), "also": int(sum(num))},
    }


def cohen_kappa(pairs: List[tuple], labels=("ab", "mcts", "tie")) -> Optional[float]:
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum(
        (sum(1 for a, _ in pairs if a == lab) / n)
        * (sum(1 for _, b in pairs if b == lab) / n)
        for lab in labels
    )
    return None if pe >= 1.0 else (po - pe) / (1.0 - pe)


def with_elo(stat: Dict) -> Dict:
    """Projected logistic Elo of an expected-score bound and of its interval (§11.12d)."""
    ci = stat.get("ci95")
    return {
        "estimate": logistic_elo(stat["estimate"]),
        "ci95": [logistic_elo(ci[0]), logistic_elo(ci[1])] if ci else None,
        "label": "projected Elo: logistic Elo of a score of 0.5 + bound, per disagreement position; "
        "a conversion, not a prediction of game results",
    }


def summarise(
    rows: List[Dict],
    judge_names: List[str],
    sampling: Dict,
    n_boot: int = 10000,
    seed: int = 1,
) -> Dict:
    summary: Dict = {
        "positions": len(rows),
        "games": len({r.get("cluster") or f"position:{i}" for i, r in enumerate(rows)}),
        "cluster_unit": (
            "game" if rows and all(r.get("cluster") for r in rows) else "position"
        ),
        "sampling": sampling,
        "method": {
            "estimates": "means over strata weighted by the recorded stratum population counts",
            "ci95": f"percentile bootstrap, {n_boot} resamples of games as clusters within strata",
        },
        "judges": {},
    }
    bounds: Dict[str, Dict[int, float]] = {}
    for name in judge_names:
        ok = [r for r in rows if "verdict" in r["judges"].get(name, {})]
        if not ok:
            continue
        js = [r["judges"][name] for r in ok]
        sel_ab = [r["selected"] == r["ab_move"] for r in ok]
        loss_sel = [
            j["loss_ab_es"] if a else j["loss_mcts_es"] for j, a in zip(js, sel_ab)
        ]
        loss_min = [min(j["loss_ab_es"], j["loss_mcts_es"]) for j in js]
        bound = [s - m for s, m in zip(loss_sel, loss_min)]
        bounds[name] = {id(r): b for r, b in zip(ok, bound)}
        values = {
            "frac_ab_better": [float(j["verdict"] == "ab") for j in js],
            "frac_mcts_better": [float(j["verdict"] == "mcts") for j in js],
            "frac_tie": [float(j["verdict"] == "tie") for j in js],
            "mean_cp_delta_ab_minus_mcts": [j["delta_cp"] for j in js],
            "loss_es_ab": [j["loss_ab_es"] for j in js],
            "loss_es_mcts": [j["loss_mcts_es"] for j in js],
            "loss_es_selected": loss_sel,
            "loss_es_oracle_arbiter": loss_min,
            "loss_es_ab_minus_mcts": [j["loss_ab_es"] - j["loss_mcts_es"] for j in js],
            "arbiter_upper_bound_es": bound,
            "realised_accuracy": [
                float(j["verdict"] in ("tie", "ab" if a else "mcts"))
                for j, a in zip(js, sel_ab)
            ],
            "selected_was_ab": [float(a) for a in sel_ab],
        }
        stats = stratified_estimates(
            ok, values, stratum_weights(ok, sampling), n_boot, seed
        )
        summary["judges"][name] = {
            "n": len(ok),
            "games": len(
                {r.get("cluster") or f"position:{i}" for i, r in enumerate(ok)}
            ),
            "errors": sum(1 for r in rows if "error" in r["judges"].get(name, {})),
            "configs": dict(
                collections.Counter(j.get("config", "unrecorded") for j in js)
            ),
            "judged_again": sum(1 for j in js if j.get("attempts", 1) > 1),
            "selected_neither_candidate": sum(
                1 for r in ok if r["selected"] not in (r["ab_move"], r["mcts_move"])
            ),
            "stats": stats,
            "projected_elo_of_bound": with_elo(stats["arbiter_upper_bound_es"]),
        }
    names = [n for n in judge_names if n in summary["judges"]]
    if len(names) >= 2:
        both = [
            r for r in rows if all("verdict" in r["judges"].get(n, {}) for n in names)
        ]
        if both:
            verdicts = [[r["judges"][n]["verdict"] for n in names] for r in both]
            other = ["mcts" if r["selected"] == r["ab_move"] else "ab" for r in both]
            agree = [len(set(v)) == 1 for v in verdicts]
            values = {
                "frac_judges_agree": [float(a) for a in agree],
                "frac_consensus_ab_better": [
                    float(a and v[0] == "ab") for a, v in zip(agree, verdicts)
                ],
                "frac_consensus_mcts_better": [
                    float(a and v[0] == "mcts") for a, v in zip(agree, verdicts)
                ],
                "frac_consensus_tie": [
                    float(a and v[0] == "tie") for a, v in zip(agree, verdicts)
                ],
                "frac_undecided": [float(not a) for a in agree],
                "consensus_bound_es": [
                    min(bounds[n][id(r)] for n in names) if a and v[0] == o else 0.0
                    for r, a, v, o in zip(both, agree, verdicts, other)
                ],
            }
            stats = stratified_estimates(
                both, values, stratum_weights(both, sampling), n_boot, seed
            )
            agreement: Dict = {
                "judges": names,
                "n": len(both),
                "strata_covered": sorted({r.get("stratum") or "all" for r in both}),
                "games": len(
                    {r.get("cluster") or f"position:{i}" for i, r in enumerate(both)}
                ),
                "consensus_rule": "all judges name the same candidate better (or all tie); "
                "otherwise undecided; the bound counts a position only when the consensus names "
                "the move not selected, at the smallest of the judges' bounds",
                "stats": stats,
                "projected_elo_of_consensus_bound": with_elo(
                    stats["consensus_bound_es"]
                ),
            }
            if len(names) == 2:
                pairs = [tuple(v) for v in verdicts]
                agreement["confusion"] = {
                    f"{a}/{b}": sum(1 for p in pairs if p == (a, b))
                    for a in ("ab", "mcts", "tie")
                    for b in ("ab", "mcts", "tie")
                }
                agreement["cohen_kappa_sample"] = cohen_kappa(pairs)
            summary["agreement"] = agreement
            summary["cross_judge"] = cross_judge(both, names, sampling, n_boot, seed)
    return summary


def cross_judge(
    both: List[Dict],
    names: List[str],
    sampling: Dict,
    n_boot: int = 10000,
    seed: int = 1,
) -> Dict[str, Dict]:
    """Amendment §11.14, for each ordered pair (selector, scorer) of judges on the positions both
    judged: the share of the selector's MCTS-better positions (by more than the tie margin) that
    the scorer also rates MCTS-better, and the arbiter bound when the selector picks the candidate
    and the scorer's losses score that pick against the realised selection."""
    weights = stratum_weights(both, sampling)
    sel_ab = [r["selected"] == r["ab_move"] for r in both]
    out: Dict[str, Dict] = {}
    for a, b in itertools.permutations(names, 2):
        ja = [r["judges"][a] for r in both]
        jb = [r["judges"][b] for r in both]
        flagged = [float(x["verdict"] == "mcts") for x in ja]
        also = [f * float(y["verdict"] == "mcts") for f, y in zip(flagged, jb)]
        # the selector's oracle arbiter picks its smaller loss (the selection on an exact tie)
        pick_ab = [
            x["loss_ab_es"] < x["loss_mcts_es"]
            or (x["loss_ab_es"] == x["loss_mcts_es"] and s)
            for x, s in zip(ja, sel_ab)
        ]
        gain = [
            (y["loss_ab_es"] if s else y["loss_mcts_es"])
            - (y["loss_ab_es"] if p else y["loss_mcts_es"])
            for y, s, p in zip(jb, sel_ab, pick_ab)
        ]
        stats = stratified_estimates(
            both, {"cross_bound_es": gain}, weights, n_boot, seed
        )
        out[f"{a}->{b}"] = {
            "selector": a,
            "scorer": b,
            "n": len(both),
            "rule": f"of the positions where {a} rates the MCTS move better by more than the tie "
            f"margin, the share that {b} also rates MCTS-better; the bound when {a} picks the "
            f"candidate with its smaller loss (the realised selection on an exact tie) and {b}'s "
            "losses score the pick against the realised selection",
            "scorer_verdicts_on_flagged": {
                v: sum(1 for f, y in zip(flagged, jb) if f and y["verdict"] == v)
                for v in ("ab", "mcts", "tie")
            },
            "confirmation_rate": stratified_ratio(
                both, also, flagged, weights, n_boot, seed
            ),
            "picks_not_selected": sum(1 for p, s in zip(pick_ab, sel_ab) if p != s),
            "stats": stats,
            "projected_elo_of_cross_bound": with_elo(stats["cross_bound_es"]),
        }
    return out


def tie_band(stat: Optional[Dict], band: float = TIE_BAND_ES) -> Optional[str]:
    """Amendment 16 (b): "within" the tie band when the 95% interval's upper limit is at most
    `band`, "beyond" when its lower limit exceeds it, "neither" otherwise; None without an
    interval."""
    ci = (stat or {}).get("ci95")
    if not ci:
        return None
    lo, hi = ci
    if hi <= band:
        return "within"
    if lo > band:
        return "beyond"
    return "neither"


def rescored(rows: List[Dict], cp_scale: float) -> List[Dict]:
    """Copies of `rows` whose judgments' expected-score losses use another logistic constant
    (amendment 16 (g)), from the centipawns each judgment keeps; verdicts, decided in centipawns,
    are unchanged."""
    out = []
    for r in rows:
        judges = {}
        for name, j in r["judges"].items():
            j = dict(j)
            if "verdict" in j:
                best = max(j["best_cp"], j["ab_cp"], j["mcts_cp"])
                j["loss_ab_es"] = logistic(best, cp_scale) - logistic(
                    j["ab_cp"], cp_scale
                )
                j["loss_mcts_es"] = logistic(best, cp_scale) - logistic(
                    j["mcts_cp"], cp_scale
                )
            judges[name] = j
        out.append(dict(r, judges=judges))
    return out


def outcome_class(
    summary: Dict,
    strata_sampled: Dict[str, int],
    n_sampled: int,
    tie_band_es: float = TIE_BAND_ES,
) -> Dict:
    """Amendment 16 (b), (c): the class of oracle outcome, decided by the consensus bound and the
    two cross-judge bounds against the tie band: "little_to_gain" (all within), "room_under_both_
    judges" (all beyond) or "judge_dependent_or_unresolved". An incomplete run (fewer positions with
    both verdicts than were sampled) gets a class only when every sampled stratum has a position
    with both verdicts. The per-judge bounds do not enter."""
    ag = summary.get("agreement")
    cj = summary.get("cross_judge") or {}
    out: Dict = {"tie_band_es": tie_band_es, "sampled": n_sampled}
    if not ag or len(cj) != 2:
        out.update(
            {"class": None, "reason": "no two judges with verdicts on common positions"}
        )
        return out
    out["positions_with_both_verdicts"] = ag["n"]
    out["complete"] = ag["n"] >= n_sampled
    covered = set(ag.get("strata_covered") or [])
    missing = sorted(s for s, k in strata_sampled.items() if k > 0 and s not in covered)
    if not out["complete"] and missing:
        out.update(
            {
                "class": None,
                "reason": "incomplete run: a sampled stratum has no position with both verdicts",
                "strata_missing": missing,
            }
        )
        return out
    bands = {"consensus": tie_band(ag["stats"].get("consensus_bound_es"), tie_band_es)}
    bands.update(
        {
            k: tie_band(v["stats"].get("cross_bound_es"), tie_band_es)
            for k, v in sorted(cj.items())
        }
    )
    out["bands"] = bands
    found = set(bands.values())
    out["class"] = (
        "little_to_gain"
        if found == {"within"}
        else (
            "room_under_both_judges"
            if found == {"beyond"}
            else "judge_dependent_or_unresolved"
        )
    )
    return out


def with_source_cells(
    summary: Dict,
    rows: List[Dict],
    sample: List[Dict],
    judge_names: List[str],
    sampling: Dict,
    n_boot: int,
    seed: int,
    tie_cp: float = 10.0,
    cp_scale: float = 250.0,
) -> Dict:
    """Amendment 16 (a)-(c), (g): every statistic again for each source cell's positions,
    reweighted by that cell's own stratum population; the class of outcome, pooled and per cell;
    and the class again with the conversion constant halved and doubled, the tie band scaled
    with it (tie_cp at an even position)."""
    by_pop = sampling.get("strata_population_by_source")
    if by_pop:
        summary["by_source_cell"] = {}
        for cell in sorted({source_cell(r) for r in sample}):
            cell_rows = [r for r in rows if source_cell(r) == cell]
            if not cell_rows:
                continue
            cell_sampling = dict(
                sampling,
                strata_population=by_pop.get(cell, {}),
                strata_sample=(sampling.get("strata_sample_by_source") or {}).get(
                    cell, {}
                ),
            )
            cs = summarise(cell_rows, judge_names, cell_sampling, n_boot, seed)
            cs["outcome"] = outcome_class(
                cs,
                cell_sampling["strata_sample"],
                sum(cell_sampling["strata_sample"].values()),
                tie_cp / cp_scale / 4,
            )
            summary["by_source_cell"][cell] = cs
    summary["outcome"] = outcome_class(
        summary,
        sampling.get("strata_sample") or {},
        len(sample),
        tie_cp / cp_scale / 4,
    )
    groups = [("pooled", rows, sampling, len(sample))]
    for cell, cs in sorted((summary.get("by_source_cell") or {}).items()):
        groups.append(
            (
                cell,
                [r for r in rows if source_cell(r) == cell],
                cs["sampling"],
                cs["outcome"]["sampled"],
            )
        )
    sens: Dict[str, Dict] = {}
    cp_fields = all(
        all(k in j for k in ("best_cp", "ab_cp", "mcts_cp"))
        for r in rows
        for j in r["judges"].values()
        if "verdict" in j
    )
    for scale in (cp_scale / 2, cp_scale * 2) if cp_fields else ():
        entry: Dict = {"tie_band_es": tie_cp / scale / 4}
        for name, group_rows, group_sampling, n in groups:
            gs = summarise(
                rescored(group_rows, scale), judge_names, group_sampling, n_boot, seed
            )
            entry[name] = outcome_class(
                gs, group_sampling.get("strata_sample") or {}, n, tie_cp / scale / 4
            )
        sens[f"{scale:g}"] = entry
    summary["conversion_sensitivity"] = sens or {
        "computed": False,
        "reason": "judgments without their centipawn scores",
    }
    return summary


def phase_of(fen: str, opening_moves: int = 15, endgame_pieces: int = 6) -> str:
    """Game phase of a FEN (protocol §11.6): 'opening' up to fullmove `opening_moves`,
    'endgame' when both sides together have at most `endgame_pieces` non-pawn, non-king
    pieces, otherwise 'middlegame'."""
    parts = fen.split()
    try:
        fullmove = int(parts[5])
    except (IndexError, ValueError):
        fullmove = 1
    pieces = sum(1 for ch in parts[0] if ch in "NBRQnbrq")
    if fullmove <= opening_moves:
        return "opening"
    if pieces <= endgame_pieces:
        return "endgame"
    return "middlegame"


def visit_count(rec: Dict) -> Optional[float]:
    """MCTS root visits at decision time, falling back to playouts, then evals."""
    for key in ("mcts_root_visits", "mcts_playouts", "mcts_evals"):
        v = rec.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def tercile_cuts(values: List[float]) -> List[float]:
    s = sorted(values)
    if not s:
        return []
    return [s[len(s) // 3], s[(2 * len(s)) // 3]]


def visit_bin(v: Optional[float], cuts: List[float]) -> str:
    if v is None:
        return "visits_unknown"
    if len(cuts) < 2:
        return "visits_all"
    if v < cuts[0]:
        return "visits_low"
    if v < cuts[1]:
        return "visits_mid"
    return "visits_high"


def stratified_sample(
    dis: List[Dict],
    n: int,
    seed: int,
    opening_moves: int = 15,
    endgame_pieces: int = 6,
):
    """Proportional stratified sample of `n` disagreement records over game phase x MCTS
    visit tercile (largest-remainder allocation; seeded shuffle within each stratum; the
    final order is shuffled too so a `--limit` prefix is still a random subset). Each
    sampled record gets a `_stratum` key. Returns (sample, sampling_info)."""
    cuts = tercile_cuts([v for v in (visit_count(r) for r in dis) if v is not None])
    strata: Dict[str, List[Dict]] = {}
    for r in dis:
        key = (
            f"{phase_of(r['fen'], opening_moves, endgame_pieces)}/"
            f"{visit_bin(visit_count(r), cuts)}"
        )
        strata.setdefault(key, []).append(r)
    total = len(dis)
    n = min(n, total)
    alloc: Dict[str, int] = {}
    remainders = []
    assigned = 0
    for key in sorted(strata):
        exact = n * len(strata[key]) / total if total else 0.0
        alloc[key] = int(exact)
        assigned += alloc[key]
        remainders.append((exact - alloc[key], key))
    for _, key in sorted(remainders, reverse=True):
        if assigned >= n:
            break
        if alloc[key] < len(strata[key]):
            alloc[key] += 1
            assigned += 1
    rng = random.Random(seed)
    sample: List[Dict] = []
    for key in sorted(strata):
        pool = list(strata[key])
        rng.shuffle(pool)
        for r in pool[: alloc[key]]:
            r["_stratum"] = key
            sample.append(r)
    rng.shuffle(sample)
    # amendment 16 (a): each source cell's own stratum population and sample, for its reweighting
    by_source: Dict[str, Dict[str, int]] = {}
    for key in sorted(strata):
        for r in strata[key]:
            cell = by_source.setdefault(source_cell(r), {})
            cell[key] = cell.get(key, 0) + 1
    sample_by_source: Dict[str, Dict[str, int]] = {}
    for r in sample:
        cell = sample_by_source.setdefault(source_cell(r), {})
        cell[r["_stratum"]] = cell.get(r["_stratum"], 0) + 1
    info = {
        "method": "stratified",
        "seed": seed,
        "requested": n,
        "visit_metric": "mcts_root_visits (fallback playouts, evals)",
        "visit_cuts": cuts,
        "phase_rule": {
            "opening_moves": opening_moves,
            "endgame_pieces": endgame_pieces,
        },
        "strata_population": {k: len(v) for k, v in sorted(strata.items())},
        "strata_sample": {k: alloc[k] for k in sorted(strata)},
        "strata_population_by_source": {
            c: dict(sorted(v.items())) for c, v in sorted(by_source.items())
        },
        "strata_sample_by_source": {
            c: dict(sorted(v.items())) for c, v in sorted(sample_by_source.items())
        },
    }
    return sample, info


def shard_slice(items: List, spec: str) -> List:
    """The I-th of K contiguous slices of `items` for `spec` = "I/K" (0-based). The K slices
    concatenated in shard order give `items` back, so positions judged on separate runners and
    joined in shard order keep the row order of a single run (protocol amendment 15)."""
    i, k = (int(x) for x in spec.split("/"))
    if not 0 <= i < k:
        raise ValueError(f"shard {spec}: need 0 <= I < K")
    n = len(items)
    return items[i * n // k : (i + 1) * n // k]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("moves_jsonl", nargs="+", help="one or more moves.jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--judge", action="append", required=True, help="NAME=PATH (repeatable)"
    )
    ap.add_argument(
        "--judge-option", action="append", default=[], help="NAME:OPTION=VALUE"
    )
    ap.add_argument("--judge-arg", action="append", default=[], help="NAME:ARG")
    ap.add_argument(
        "--movetime",
        type=int,
        default=60000,
        help="ms per search for time-limited judges",
    )
    ap.add_argument(
        "--nodes",
        action="append",
        default=[],
        help="NAME:NODES for node-limited judges (e.g. lc0)",
    )
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--no-stratify",
        action="store_true",
        help="plain seeded random sample instead of the protocol's stratified sample",
    )
    ap.add_argument(
        "--opening-moves",
        type=int,
        default=15,
        help="fullmove number up to which a position counts as opening (default 15)",
    )
    ap.add_argument(
        "--endgame-pieces",
        type=int,
        default=6,
        help="non-pawn, non-king pieces (both sides) at or below which a position is endgame",
    )
    ap.add_argument("--tie-cp", type=float, default=10.0)
    ap.add_argument("--cp-scale", type=float, default=250.0)
    ap.add_argument(
        "--limit", type=int, help="smoke: judge only the first N sampled positions"
    )
    ap.add_argument(
        "--shard",
        help="I/K: judge only the I-th of K contiguous slices of the sample (0-based) and write "
        "no summary; concatenate the K outputs in shard order and run --summarise-only on them",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="bootstrap resamples (default 10000)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="keep the rows already in --out and judge only the positions not yet judged",
    )
    ap.add_argument(
        "--summarise-only",
        action="store_true",
        help="read the judged rows from --out and rewrite the summary; no engine is started",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="further attempts, each on a fresh engine process, when a judge dies, overruns "
        "--judge-timeout or reports a GPU error during a search (default 2)",
    )
    ap.add_argument(
        "--judge-timeout",
        type=float,
        help="seconds one judge may take over a position before its engine is killed and the "
        "position judged again",
    )
    ap.add_argument(
        "--keep-unrecorded",
        action="append",
        default=[],
        help="NAME: with --resume, keep this judge's judgments that record no configuration "
        "(made before configurations were recorded); others are kept only when made with "
        "the same configuration",
    )
    args = ap.parse_args(argv)

    # stream the trace (hundreds of MB): keep only disagreement records and the fields used here
    n_recs = 0
    dis = []
    for path in args.moves_jsonl:
        with open(path) as fh:
            for line in fh:
                n_recs += 1
                r = json.loads(line)
                if (
                    r.get("fen")
                    and r.get("ab_move")
                    and r.get("mcts_move")
                    and r["ab_move"] != r["mcts_move"]
                    and r.get("reason") != "ab_fallback_time_safety"
                ):
                    dis.append({k: r[k] for k in KEEP_FIELDS if k in r})
    if args.no_stratify:
        rng = random.Random(args.seed)
        rng.shuffle(dis)
        sample = dis[: args.sample]
        sampling: Dict = {
            "method": "random",
            "seed": args.seed,
            "requested": args.sample,
        }
    else:
        sample, sampling = stratified_sample(
            dis, args.sample, args.seed, args.opening_moves, args.endgame_pieces
        )
    if args.limit:
        sample = sample[: args.limit]
    if args.shard:
        sample = shard_slice(sample, args.shard)
    print(
        f"{n_recs} records, {len(dis)} disagreements, judging {len(sample)} "
        f"({sampling['method']})",
        file=sys.stderr,
    )

    node_limits = dict(
        (kv.split(":", 1)[0], int(kv.split(":", 1)[1])) for kv in args.nodes
    )
    judge_names = [spec.split("=", 1)[0] for spec in args.judge]
    judging = {
        name: (
            {
                "limit": f"nodes {node_limits[name]}",
                "candidates": "position after each candidate, same node limit, fresh tree",
            }
            if name in node_limits
            else {
                "limit": f"movetime {args.movetime} ms",
                "candidates": "restricted multi-PV search over both candidates",
            }
        )
        for name in judge_names
    }

    def row_key(row: Dict) -> tuple:
        return (row["fen"], row["ab_move"], row["mcts_move"], row.get("cluster"))

    def sample_key(r: Dict) -> tuple:
        return (r["fen"], r["ab_move"], r["mcts_move"], cluster_key(r))

    done: List[Dict] = []
    if args.resume or args.summarise_only:
        try:
            with open(args.out) as fh:
                done = [json.loads(l) for l in fh if l.strip()]
        except FileNotFoundError:
            done = []
    if args.summarise_only:
        rows = done
        incomplete = 0
    else:
        specs = [spec.split("=", 1) for spec in args.judge]
        jargs = {
            name: [
                a.split(":", 1)[1] for a in args.judge_arg if a.split(":", 1)[0] == name
            ]
            for name, _ in specs
        }
        jopts = {
            name: [
                o.split(":", 1)[1].split("=", 1)
                for o in args.judge_option
                if o.split(":", 1)[0] == name
            ]
            for name, _ in specs
        }
        configs = {
            name: judge_config(
                path,
                jargs[name],
                jopts[name],
                (
                    f"nodes={node_limits[name]}"
                    if name in node_limits
                    else f"movetime={args.movetime}"
                ),
            )
            for name, path in specs
        }

        def usable(name: str, entry: Optional[Dict]) -> bool:
            """A judgment a resumed run keeps: no error, and made with this configuration (or
            made before configurations were recorded, for a judge named in --keep-unrecorded).
            """
            if not entry or "error" in entry:
                return False
            if "config" in entry:
                return entry["config"] == configs[name]
            return name in args.keep_unrecorded

        by_key = {row_key(r): r for r in done}
        in_sample = {sample_key(r) for r in sample}

        def complete(r: Dict) -> bool:
            row = by_key.get(sample_key(r))
            return bool(row) and all(
                usable(n, row["judges"].get(n)) for n in judge_names
            )

        def ordered() -> List[Dict]:
            """Rows in sample order, then any rows of --out that are not in the sample."""
            return [
                by_key[sample_key(r)] for r in sample if sample_key(r) in by_key
            ] + [r for r in done if row_key(r) not in in_sample]

        todo = [r for r in sample if not complete(r)]
        if done:
            print(
                f"resuming: {len(done)} rows, {len(todo)} positions to judge",
                file=sys.stderr,
            )
        watch = StderrWatch()
        engine_log = logging.getLogger("chess.engine")
        engine_log.addHandler(watch)
        engine_log.propagate = False

        def spawner(name: str, path: str) -> Callable:
            def spawn():
                eng = chess.engine.SimpleEngine.popen_uci([path, *jargs[name]])
                for k, v in jopts[name]:
                    try:
                        eng.configure({k: v})
                    except Exception as exc:  # noqa: BLE001
                        # a judge without its configured options is not the judge the
                        # protocol names (e.g. Lc0 with smart pruning on): stop here
                        close_engine(eng)
                        raise SystemExit(
                            f"error: {name} option {k}={v} rejected: {exc}"
                        )
                return eng

            return spawn

        judges: Dict[str, Judge] = {}
        try:
            for name, path in specs if todo else []:
                judges[name] = Judge(name, spawner(name, path), watch)
            for k, r in enumerate(todo, 1):
                board = chess.Board(r["fen"])
                row = by_key.get(sample_key(r)) or {
                    "fen": r["fen"],
                    "ab_move": r["ab_move"],
                    "mcts_move": r["mcts_move"],
                    "selected": r.get("selected"),
                    "reason": r.get("reason"),
                    "mcts_playouts": r.get("mcts_playouts"),
                    "mcts_root_visits": r.get("mcts_root_visits"),
                    "stratum": r.get("_stratum"),
                    "cluster": cluster_key(r),
                    "opening": r.get("opening"),
                    "judges": {},
                }
                for name, judge in judges.items():
                    prev = row["judges"].get(name)
                    if usable(name, prev):
                        continue
                    limit = (
                        chess.engine.Limit(nodes=node_limits[name])
                        if name in node_limits
                        else chess.engine.Limit(time=args.movetime / 1000.0)
                    )
                    j = judge_with_retries(
                        judge,
                        board,
                        [r["ab_move"], r["mcts_move"]],
                        limit,
                        name in node_limits,
                        args.retries,
                        args.judge_timeout,
                    )
                    if prev and "error" in prev and prev.get("config") == configs[name]:
                        j["attempts"] += prev.get("attempts", 1)
                    j["config"] = configs[name]
                    if "error" not in j:
                        j = score_candidates(j, r, args.tie_cp, args.cp_scale)
                    row["judges"][name] = j
                by_key[sample_key(r)] = row
                write_rows(args.out, ordered())
                print(
                    f"position {k}/{len(todo)}: "
                    + ", ".join(
                        f"{n} {'error' if 'error' in row['judges'][n] else 'judged'}"
                        f" after {row['judges'][n].get('attempts', 1)} attempt(s)"
                        for n in judges
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            for judge in judges.values():
                judge.close()
        rows = ordered()
        incomplete = sum(1 for r in sample if not complete(r))
        if incomplete:
            print(
                f"{incomplete} positions still lack a judgment from every judge; resume to judge them",
                file=sys.stderr,
            )

    if args.shard:
        print(f"shard {args.shard}: {len(rows)} rows in {args.out}", file=sys.stderr)
        return 3 if incomplete else 0
    summary = summarise(rows, judge_names, sampling, args.bootstrap, args.seed)
    summary = with_source_cells(
        summary,
        rows,
        sample,
        judge_names,
        sampling,
        args.bootstrap,
        args.seed,
        args.tie_cp,
        args.cp_scale,
    )
    summary["judging"] = judging
    summary["method"].update(
        {
            "expected_score": f"logistic, {args.cp_scale:g} cp per unit of log-odds",
            "tie_cp": args.tie_cp,
        }
    )
    with open(args.out + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 3 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
