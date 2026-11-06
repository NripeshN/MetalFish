#!/usr/bin/env python3
"""Time a node-limited oracle judge on the runner: timings only, no verdict (protocol amendment 15).

Runs the oracle's judge procedure (oracle_probe.judge_position: the position, then each candidate
from a fresh tree, all at the node limit) on N disagreement positions that are NOT in the registered
600-position sample, and counts engine crashes and the GPU errors the engine reports on stderr.
Prints wall time and the root search's nodes per position; no score, move or verdict is printed or
written. With --fen, times one search of each given position instead.

    python3 tools/paper/judge_probe.py disagreements.jsonl --engine "$(brew --prefix)/bin/lc0" \
        --arg=--backend=blas --arg=--weights=networks/BT4.pb --option SmartPruningFactor=0 \
        --nodes 30000 --positions 4 --seed 101
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import random
import statistics
import sys
import time
from typing import Dict, List, Optional

import chess
import chess.engine

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import oracle_probe  # noqa: E402


def outside_sample(path: str, n: int, seed: int) -> List[Dict]:
    """N disagreement records not in the registered sample (oracle_probe's filter, sample size 600,
    seed 1), drawn with `seed`."""
    dis = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if (
                r.get("fen")
                and r.get("ab_move")
                and r.get("mcts_move")
                and r["ab_move"] != r["mcts_move"]
                and r.get("reason") != "ab_fallback_time_safety"
            ):
                dis.append({k: r[k] for k in oracle_probe.KEEP_FIELDS if k in r})
    sample, _ = oracle_probe.stratified_sample(dis, 600, 1)
    taken = {(r["fen"], r["ab_move"], r["mcts_move"]) for r in sample}
    rest = [r for r in dis if (r["fen"], r["ab_move"], r["mcts_move"]) not in taken]
    random.Random(seed).shuffle(rest)
    return rest[:n]


def start(engine: str, args: List[str], options: Dict[str, str]):
    eng = chess.engine.SimpleEngine.popen_uci([engine, *args])
    for k, v in options.items():
        eng.configure({k: v})
    return eng


def stop(eng) -> None:
    try:
        eng.quit()
    except Exception:  # noqa: BLE001 - a dead engine cannot quit
        pass


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("disagreements", help="rq6b_disagreements.jsonl (gunzipped)")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--arg", action="append", default=[], help="engine argument")
    ap.add_argument("--option", action="append", default=[], help="UCI NAME=VALUE")
    ap.add_argument("--nodes", type=int, default=30000)
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--fen", action="append", default=[], help="time one search of FEN")
    args = ap.parse_args(argv)

    watch = oracle_probe.StderrWatch()
    log = logging.getLogger("chess.engine")
    log.addHandler(watch)
    log.propagate = False
    options = dict(o.split("=", 1) for o in args.option)
    limit = chess.engine.Limit(nodes=args.nodes)
    eng = start(args.engine, args.arg, options)
    crashes = 0
    times = []
    try:
        for fen in args.fen:
            t0 = time.time()
            info = eng.analyse(chess.Board(fen), limit, game=object())
            dt = time.time() - t0
            print(
                f"one search, {args.nodes} nodes: {dt:.1f} s, nodes {info.get('nodes')}, "
                f"nps {info.get('nps')}",
                flush=True,
            )
        for i, r in enumerate(
            []
            if args.fen
            else outside_sample(args.disagreements, args.positions, args.seed)
        ):
            before = sum(watch.by_engine.values())
            t0 = time.time()
            try:
                j = oracle_probe.judge_position(
                    eng,
                    chess.Board(r["fen"]),
                    [r["ab_move"], r["mcts_move"]],
                    limit,
                    per_candidate=True,
                )
                nodes = j.get("nodes")
            except Exception as exc:  # noqa: BLE001
                crashes += 1
                nodes = f"crash ({type(exc).__name__})"
                stop(eng)
                eng = start(args.engine, args.arg, options)
            dt = time.time() - t0
            times.append(dt)
            print(
                f"position {i}: {dt / 60:.1f} min for the position and both candidates; "
                f"root nodes {nodes}; GPU error lines {sum(watch.by_engine.values()) - before}",
                flush=True,
            )
    finally:
        stop(eng)
    if times:
        print(
            f"{len(times)} positions: mean {statistics.mean(times) / 60:.1f} min, "
            f"median {statistics.median(times) / 60:.1f} min, max {max(times) / 60:.1f} min; "
            f"crashes {crashes}; GPU error lines {sum(watch.by_engine.values())}",
            flush=True,
        )
    return 1 if crashes or any(watch.by_engine.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
