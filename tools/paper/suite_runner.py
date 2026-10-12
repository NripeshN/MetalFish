#!/usr/bin/env python3
"""Run EPD test suites (bm/am) and Lichess puzzle samples through any UCI engine.

Outputs per-position JSONL and a summary with Wilson 95% intervals. Movetime-based
(``go movetime``); a position is solved when the engine's bestmove matches a ``bm`` move
(any of them) and is not an ``am`` move. Puzzles require the whole solution line.

Examples:
    python3 tools/paper/suite_runner.py --engine reference/stockfish-sf19/src/stockfish \
        --option Threads=8 --option Hash=2048 --epd reference/testsuites/eret/eret.epd \
        --movetime 15000 --out results/paper_v2/suites/eret_sf19.jsonl
    python3 tools/paper/suite_runner.py --engine build/metalfish --option Threads=8 \
        --option UseHybridSearch=true --option NNWeights=$PWD/networks/BT4-1024x15x32h-swa-6147500.pb \
        --option TransformerLowTimeFallbackMs=0 --option TransformerMinMoveBudgetMs=0 \
        --puzzles reference/lichess_puzzles/lichess_db_puzzle.csv --sample 1000 --seed 1 \
        --min-rating 1200 --max-rating 2800 --min-popularity 70 --movetime 1000 --out out.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def parse_epd_line(line: str) -> Optional[Dict]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(" ")
    if len(parts) < 4:
        return None
    fen = " ".join(parts[:4]) + " 0 1"
    ops = " ".join(parts[4:])
    rec: Dict = {"fen": fen, "bm": [], "am": [], "id": ""}
    for op in ops.split(";"):
        op = op.strip()
        if not op:
            continue
        key, _, val = op.partition(" ")
        val = val.strip().strip('"')
        if key == "bm":
            rec["bm"] = val.split()
        elif key == "am":
            rec["am"] = val.split()
        elif key == "id":
            rec["id"] = val
    return rec


def san_to_uci(board: chess.Board, san: str) -> Optional[str]:
    try:
        return board.parse_san(san).uci()
    except Exception:  # noqa: BLE001
        try:
            return board.parse_uci(san).uci()
        except Exception:  # noqa: BLE001
            return None


def open_engine(path: str, options: Dict[str, str], args: List[str]):
    eng = chess.engine.SimpleEngine.popen_uci([path, *args])
    for k, v in options.items():
        try:
            eng.configure({k: v})
        except Exception as exc:  # noqa: BLE001
            print(f"warning: option {k}={v} rejected: {exc}", file=sys.stderr)
    return eng


def run_epd(eng, positions: List[Dict], movetime_ms: int, out, label: str) -> Dict:
    solved = 0
    n = 0
    for rec in positions:
        board = chess.Board(rec["fen"])
        bm = [m for m in (san_to_uci(board, s) for s in rec["bm"]) if m]
        am = [m for m in (san_to_uci(board, s) for s in rec["am"]) if m]
        t0 = time.time()
        info = eng.play(
            board,
            chess.engine.Limit(time=movetime_ms / 1000.0),
            info=chess.engine.INFO_ALL,
        )
        elapsed = time.time() - t0
        mv = info.move.uci() if info.move else None
        ok = (mv in bm if bm else True) and (mv not in am)
        n += 1
        solved += int(ok)
        row = {
            "suite": label,
            "id": rec["id"],
            "fen": rec["fen"],
            "bm": bm,
            "am": am,
            "move": mv,
            "solved": ok,
            "elapsed_s": round(elapsed, 3),
            "depth": info.info.get("depth"),
            "nodes": info.info.get("nodes"),
            "score_cp": (
                info.info.get("score").relative.score(mate_score=100000)
                if info.info.get("score")
                else None
            ),
        }
        out.write(json.dumps(row) + "\n")
        out.flush()
    lo, hi = wilson(solved, n)
    return {
        "suite": label,
        "positions": n,
        "solved": solved,
        "rate": solved / n if n else 0.0,
        "wilson95": [lo, hi],
    }


def sample_puzzles(
    path: str, sample: int, seed: int, min_rating: int, max_rating: int, min_pop: int
) -> List[Dict]:
    rng = random.Random(seed)
    chosen: List[Dict] = []
    seen = 0
    # reservoir sampling over eligible rows
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rating = int(row["Rating"])
                pop = int(row["Popularity"])
            except ValueError:
                continue
            if not (min_rating <= rating <= max_rating) or pop < min_pop:
                continue
            seen += 1
            rec = {
                "id": row["PuzzleId"],
                "fen": row["FEN"],
                "moves": row["Moves"].split(),
                "rating": rating,
                "themes": row["Themes"],
            }
            if len(chosen) < sample:
                chosen.append(rec)
            else:
                j = rng.randrange(seen)
                if j < sample:
                    chosen[j] = rec
    return chosen


def run_puzzles(eng, puzzles: List[Dict], movetime_ms: int, out, label: str) -> Dict:
    solved = 0
    n = 0
    by_band: Dict[str, List[int]] = {}
    for pz in puzzles:
        board = chess.Board(pz["fen"])
        moves = pz["moves"]
        board.push_uci(moves[0])  # the opponent's setup move
        ok = True
        played: List[str] = []
        for i in range(1, len(moves), 2):
            expected = moves[i]
            res = eng.play(board, chess.engine.Limit(time=movetime_ms / 1000.0))
            mv = res.move.uci() if res.move else None
            played.append(mv or "")
            if mv != expected:
                # Lichess accepts alternative mates on the final move
                if i == len(moves) - 1 and mv:
                    b2 = board.copy()
                    b2.push_uci(mv)
                    if b2.is_checkmate():
                        break
                ok = False
                break
            board.push_uci(expected)
            if i + 1 < len(moves):
                board.push_uci(moves[i + 1])
        n += 1
        solved += int(ok)
        band = f"{(pz['rating'] // 400) * 400}-{(pz['rating'] // 400) * 400 + 399}"
        by_band.setdefault(band, [0, 0])
        by_band[band][0] += int(ok)
        by_band[band][1] += 1
        out.write(
            json.dumps(
                {
                    "suite": label,
                    "id": pz["id"],
                    "rating": pz["rating"],
                    "themes": pz["themes"],
                    "expected": moves[1::2],
                    "played": played,
                    "solved": ok,
                }
            )
            + "\n"
        )
        out.flush()
    lo, hi = wilson(solved, n)
    return {
        "suite": label,
        "positions": n,
        "solved": solved,
        "rate": solved / n if n else 0.0,
        "wilson95": [lo, hi],
        "by_rating_band": {
            k: {"solved": v[0], "n": v[1], "rate": v[0] / v[1]}
            for k, v in sorted(by_band.items())
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--engine", required=True)
    ap.add_argument(
        "--arg",
        action="append",
        default=[],
        help="engine command-line argument (repeatable)",
    )
    ap.add_argument(
        "--option",
        action="append",
        default=[],
        help="UCI option NAME=VALUE (repeatable)",
    )
    ap.add_argument("--epd", help="EPD suite file")
    ap.add_argument("--puzzles", help="Lichess puzzle CSV")
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-rating", type=int, default=1200)
    ap.add_argument("--max-rating", type=int, default=2800)
    ap.add_argument("--min-popularity", type=int, default=70)
    ap.add_argument("--movetime", type=int, required=True, help="milliseconds per move")
    ap.add_argument(
        "--limit", type=int, help="only the first N positions (smoke tests)"
    )
    ap.add_argument("--label", default=None)
    ap.add_argument(
        "--out",
        required=True,
        help="per-position JSONL; summary goes to <out>.summary.json",
    )
    args = ap.parse_args(argv)
    options = dict(o.split("=", 1) for o in args.option)
    eng = open_engine(args.engine, options, args.arg)
    label = args.label or (args.epd or args.puzzles or "suite").split("/")[-1]
    try:
        with open(args.out, "w") as out:
            if args.epd:
                positions = [
                    r
                    for r in (
                        parse_epd_line(l)
                        for l in open(args.epd, encoding="utf-8", errors="replace")
                    )
                    if r
                ]
                if args.limit:
                    positions = positions[: args.limit]
                summary = run_epd(eng, positions, args.movetime, out, label)
            elif args.puzzles:
                puzzles = sample_puzzles(
                    args.puzzles,
                    args.sample,
                    args.seed,
                    args.min_rating,
                    args.max_rating,
                    args.min_popularity,
                )
                if args.limit:
                    puzzles = puzzles[: args.limit]
                summary = run_puzzles(eng, puzzles, args.movetime, out, label)
            else:
                print("need --epd or --puzzles", file=sys.stderr)
                return 2
    finally:
        eng.quit()
    summary.update(
        {
            "engine": args.engine,
            "options": options,
            "args": args.arg,
            "movetime_ms": args.movetime,
            "seed": args.seed,
        }
    )
    with open(args.out + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
