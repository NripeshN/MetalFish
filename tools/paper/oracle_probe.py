#!/usr/bin/env python3
"""Oracle study of hybrid arbitration: judge the AB move and the MCTS move on positions
where they disagreed, using strong reference engines (RQ6b of docs/paper_v2/02_evaluation_protocol.md).

Input: moves.jsonl from tools/paper/parse_hybrid_trace.py (records need fen, ab_move, mcts_move,
selected). For each disagreement position and each judge:
  1. unrestricted search  -> oracle best move and score (mover's perspective, cp)
  2. restricted search `searchmoves ab mcts` with MultiPV=2 -> scores of both candidates at equal depth
Loss of a candidate = oracle_best_cp - candidate_cp (>= 0 up to search noise). Expected-score loss uses a
logistic with scale --cp-scale (default 250 cp per unit log-odds, a conventional approximation).

Output: per-position JSONL plus a summary with paired bootstrap CIs:
  frac_ab_better / frac_mcts_better / frac_tie (|delta| <= --tie-cp)
  mean loss of AB move, MCTS move, the realised selection, and the oracle arbiter (min of the two)
  upper bound of any arbiter = mean(loss_selected) - mean(min loss)

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


def expected_score(cp: float, scale: float) -> float:
    return 1.0 / (
        1.0 + 10.0 ** (-cp / (scale * 4.0 / math.log(10.0) * math.log(10.0) / 4.0))
    )


def logistic(cp: float, scale: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / scale))


def cp_of(score: chess.engine.PovScore) -> float:
    return float(score.relative.score(mate_score=30000))


def judge_position(
    eng, board: chess.Board, cands: List[str], limit: chess.engine.Limit
) -> Dict:
    best = eng.analyse(board, limit)
    best_cp = cp_of(best["score"])
    best_move = best["pv"][0].uci() if best.get("pv") else None
    moves = [
        chess.Move.from_uci(m)
        for m in cands
        if chess.Move.from_uci(m) in board.legal_moves
    ]
    infos = eng.analyse(board, limit, multipv=len(moves), root_moves=moves)
    if isinstance(infos, dict):
        infos = [infos]
    cand_cp: Dict[str, float] = {}
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


def bootstrap_mean(values: List[float], n: int = 10000, seed: int = 1):
    if not values:
        return None
    rng = random.Random(seed)
    k = len(values)
    means = []
    for _ in range(n):
        s = 0.0
        for _ in range(k):
            s += values[rng.randrange(k)]
        means.append(s / k)
    means.sort()
    return [means[int(0.025 * n)], means[int(0.975 * n)]]


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
    ap.add_argument("moves_jsonl")
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
    args = ap.parse_args(argv)

    recs = [json.loads(l) for l in open(args.moves_jsonl)]
    dis = [
        r
        for r in recs
        if r.get("fen")
        and r.get("ab_move")
        and r.get("mcts_move")
        and r["ab_move"] != r["mcts_move"]
        and r.get("reason") != "ab_fallback_time_safety"
    ]
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
        f"{len(recs)} records, {len(dis)} disagreements, judging {len(sample)} "
        f"({sampling['method']})",
        file=sys.stderr,
    )

    judges = {}
    node_limits = dict(
        (kv.split(":", 1)[0], int(kv.split(":", 1)[1])) for kv in args.nodes
    )
    for spec in args.judge:
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

    rows = []
    try:
        with open(args.out, "w") as out:
            for r in sample:
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
                            eng, board, [r["ab_move"], r["mcts_move"]], limit
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
                                else "mcts" if mc_cp - ab_cp > args.tie_cp else "tie"
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

    summary: Dict = {"positions": len(rows), "sampling": sampling, "judges": {}}
    for name in judges:
        ok = [r for r in rows if "verdict" in r["judges"].get(name, {})]
        if not ok:
            continue
        v = [r["judges"][name]["verdict"] for r in ok]
        loss_ab = [r["judges"][name]["loss_ab_es"] for r in ok]
        loss_mc = [r["judges"][name]["loss_mcts_es"] for r in ok]
        loss_sel = [
            (
                r["judges"][name]["loss_ab_es"]
                if r["selected"] == r["ab_move"]
                else r["judges"][name]["loss_mcts_es"]
            )
            for r in ok
        ]
        loss_min = [min(a, b) for a, b in zip(loss_ab, loss_mc)]
        diff = [a - b for a, b in zip(loss_ab, loss_mc)]
        summary["judges"][name] = {
            "n": len(ok),
            "frac_ab_better": v.count("ab") / len(ok),
            "frac_mcts_better": v.count("mcts") / len(ok),
            "frac_tie": v.count("tie") / len(ok),
            "mean_cp_delta_ab_minus_mcts": sum(
                r["judges"][name]["delta_cp"] for r in ok
            )
            / len(ok),
            "mean_loss_es": {
                "ab": sum(loss_ab) / len(ok),
                "mcts": sum(loss_mc) / len(ok),
                "selected": sum(loss_sel) / len(ok),
                "oracle_arbiter": sum(loss_min) / len(ok),
            },
            "arbiter_upper_bound_es": sum(loss_sel) / len(ok) - sum(loss_min) / len(ok),
            "ci95_loss_ab_minus_mcts_es": bootstrap_mean(diff, seed=args.seed),
            "ci95_arbiter_upper_bound_es": bootstrap_mean(
                [s - m for s, m in zip(loss_sel, loss_min)], seed=args.seed
            ),
            "selected_was_ab": sum(1 for r in ok if r["selected"] == r["ab_move"])
            / len(ok),
        }
    with open(args.out + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
