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
where they differ counted as undecided. Estimates are means over strata weighted by the recorded
stratum population counts; 95% intervals are percentile bootstraps resampling games as clusters
within strata (§11.12g), and the number of games contributing positions is reported.

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
import json
import math
import random
import sys
from typing import Dict, List, Optional

import chess
import chess.engine

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
        "label": "projected Elo: logistic Elo of a score of 0.5 + bound, per disagreement position",
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
    }
    return sample, info


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

    done: List[Dict] = []
    if args.resume or args.summarise_only:
        try:
            with open(args.out) as fh:
                done = [json.loads(l) for l in fh if l.strip()]
        except FileNotFoundError:
            done = []
    if args.summarise_only:
        rows = done
    else:
        judged = {row_key(r) for r in done}
        todo = [
            r
            for r in sample
            if (r["fen"], r["ab_move"], r["mcts_move"], cluster_key(r)) not in judged
        ]
        if done:
            print(f"resuming: {len(done)} judged, {len(todo)} to go", file=sys.stderr)
        judges = {}
        for spec in args.judge if todo else []:
            name, path = spec.split("=", 1)
            jargs = [
                a.split(":", 1)[1] for a in args.judge_arg if a.split(":", 1)[0] == name
            ]
            eng = chess.engine.SimpleEngine.popen_uci([path, *jargs])
            for o in args.judge_option:
                jn, kv = o.split(":", 1)
                if jn == name:
                    k, v = kv.split("=", 1)
                    try:
                        eng.configure({k: v})
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"warning: {name} option {k}={v} rejected: {exc}",
                            file=sys.stderr,
                        )
            judges[name] = eng

        rows = list(done)
        try:
            with open(args.out, "a" if done else "w") as out:
                for r in todo:
                    board = chess.Board(r["fen"])
                    row = {
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
                    for name, eng in judges.items():
                        limit = (
                            chess.engine.Limit(nodes=node_limits[name])
                            if name in node_limits
                            else chess.engine.Limit(time=args.movetime / 1000.0)
                        )
                        try:
                            j = judge_position(
                                eng,
                                board,
                                [r["ab_move"], r["mcts_move"]],
                                limit,
                                per_candidate=name in node_limits,
                            )
                        except Exception as exc:  # noqa: BLE001
                            row["judges"][name] = {"error": str(exc)}
                            continue
                        ab_cp = j["cand_cp"].get(r["ab_move"])
                        mc_cp = j["cand_cp"].get(r["mcts_move"])
                        if ab_cp is None or mc_cp is None:
                            row["judges"][name] = {"error": "candidate not scored", **j}
                            continue
                        best = max(j["best_cp"], ab_cp, mc_cp)
                        j.update(
                            {
                                "ab_cp": ab_cp,
                                "mcts_cp": mc_cp,
                                "delta_cp": ab_cp - mc_cp,
                                "loss_ab_cp": best - ab_cp,
                                "loss_mcts_cp": best - mc_cp,
                                "loss_ab_es": logistic(best, args.cp_scale)
                                - logistic(ab_cp, args.cp_scale),
                                "loss_mcts_es": logistic(best, args.cp_scale)
                                - logistic(mc_cp, args.cp_scale),
                                "verdict": (
                                    "ab"
                                    if ab_cp - mc_cp > args.tie_cp
                                    else (
                                        "mcts" if mc_cp - ab_cp > args.tie_cp else "tie"
                                    )
                                ),
                            }
                        )
                        row["judges"][name] = j
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
        finally:
            for eng in judges.values():
                eng.quit()

    summary = summarise(rows, judge_names, sampling, args.bootstrap, args.seed)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
