#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import bk_parity  # noqa: E402

from tools import lc0_coreml_position_parity as parity  # noqa: E402


def model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        weights=args.weights,
        batch_size=args.batch_size,
        candidate_compute_unit=args.compute_unit,
        candidate_precision=args.precision,
        candidate_value_head_fp32=args.value_head_fp32,
        candidate_policy_head_fp32=args.policy_head_fp32,
    )


def select_positions(selection: str) -> list[tuple[str, list[str], str]]:
    if selection.strip().lower() == "all":
        return bk_parity.BK_POSITIONS
    return bk_parity.select_positions(selection)


def expected_moves(fen: str, expected_sans: Sequence[str]) -> set[str]:
    return set(bk_parity.expected_uci_moves(fen, expected_sans))


def wdl_value(wdl: np.ndarray) -> float:
    flat = np.asarray(wdl, dtype=np.float32).reshape(-1)
    if flat.size != 3:
        raise ValueError(f"WDL tensor must have 3 values, got {flat.size}")
    return float(flat[0] - flat[2])


def score_child_for_root(
    board: chess.Board, move: chess.Move, child_wdl: np.ndarray
) -> float:
    child = board.copy(stack=False)
    child.push(move)
    if child.is_checkmate():
        return 1.0
    if child.is_stalemate() or child.is_insufficient_material():
        return 0.0
    return -wdl_value(child_wdl)


def best_expected_rank(
    ranked: Sequence[dict[str, Any]], expected: set[str]
) -> int | None:
    ranks = [int(item["rank"]) for item in ranked if item["move"] in expected]
    return min(ranks) if ranks else None


def rank_bucket(rank: int | None, limit: int) -> int:
    return 1 if rank is not None and rank <= limit else 0


def predict_child_wdls(
    model: Any, child_fens: Sequence[str], batch_size: int
) -> list[np.ndarray]:
    if not child_fens:
        return []
    wdls: list[np.ndarray] = []
    for offset in range(0, len(child_fens), batch_size):
        chunk = list(child_fens[offset : offset + batch_size])
        while len(chunk) < batch_size:
            chunk.append(chunk[-1])
        planes = np.concatenate(
            [parity.encode_fen_classical_112(fen) for fen in chunk], axis=0
        )
        outputs = model.predict({"x": planes})
        wdl, _policy = parity.split_heads(outputs)
        for idx in range(min(batch_size, len(child_fens) - offset)):
            wdls.append(np.asarray(wdl[idx], dtype=np.float32))
    return wdls


def evaluate_position(
    model: Any,
    fen: str,
    expected_sans: Sequence[str],
    bk_id: str,
    batch_size: int,
    top: int,
) -> tuple[dict[str, Any], float]:
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    child_fens: list[str] = []
    model_moves: list[chess.Move] = []
    ranked: list[dict[str, Any]] = []

    start = time.perf_counter()
    for move in moves:
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            ranked.append({"move": move.uci(), "score": 1.0, "terminal": "mate"})
        elif child.is_stalemate() or child.is_insufficient_material():
            ranked.append({"move": move.uci(), "score": 0.0, "terminal": "draw"})
        else:
            model_moves.append(move)
            child_fens.append(child.fen())

    child_wdls = predict_child_wdls(model, child_fens, batch_size)
    for move, child_wdl in zip(model_moves, child_wdls):
        ranked.append(
            {
                "move": move.uci(),
                "score": score_child_for_root(board, move, child_wdl),
                "child_wdl": [float(x) for x in child_wdl.reshape(-1).tolist()],
                "terminal": "",
            }
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    ranked.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank

    expected = expected_moves(fen, expected_sans)
    rank = best_expected_rank(ranked, expected)
    return (
        {
            "id": bk_id,
            "fen": fen,
            "expected_san": list(expected_sans),
            "expected_uci": sorted(expected),
            "expected_rank": rank,
            "top_move": ranked[0]["move"] if ranked else "0000",
            "top": ranked[:top],
            "legal_moves": len(moves),
            "prediction_ms": elapsed_ms,
        },
        elapsed_ms,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    positions = select_positions(args.positions)
    build_start = time.perf_counter()
    model = parity.build_model(model_args(args), "heads", True)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    if positions:
        sample = parity.encode_fen_classical_112(positions[0][0])
        sample = np.concatenate([sample] * args.batch_size, axis=0)
        for _ in range(args.warmup):
            model.predict({"x": sample})

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for fen, expected_sans, bk_id in positions:
        row, elapsed_ms = evaluate_position(
            model, fen, expected_sans, bk_id, args.batch_size, args.top
        )
        rows.append(row)
        latencies.append(elapsed_ms)

    total = len(rows)
    top1 = sum(rank_bucket(row["expected_rank"], 1) for row in rows)
    top3 = sum(rank_bucket(row["expected_rank"], 3) for row in rows)
    top5 = sum(rank_bucket(row["expected_rank"], 5) for row in rows)
    top10 = sum(rank_bucket(row["expected_rank"], 10) for row in rows)
    ranked_values = [row["expected_rank"] for row in rows if row["expected_rank"]]
    return {
        "weights": str(Path(args.weights).resolve()),
        "compute_unit": args.compute_unit,
        "precision": args.precision,
        "value_head_fp32": args.value_head_fp32,
        "policy_head_fp32": args.policy_head_fp32,
        "batch_size": args.batch_size,
        "positions": rows,
        "summary": {
            "total": total,
            "top1": top1,
            "top3": top3,
            "top5": top5,
            "top10": top10,
            "median_expected_rank": (
                statistics.median(ranked_values) if ranked_values else None
            ),
            "mean_prediction_ms": statistics.fmean(latencies) if latencies else 0.0,
            "median_prediction_ms": statistics.median(latencies) if latencies else 0.0,
            "build_ms": build_ms,
        },
    }


def comma_list(values: Iterable[str]) -> str:
    return ", ".join(values)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank Bratko-Kopec expected moves by Core ML T1 child-value search."
    )
    parser.add_argument("weights", help="Lc0 T1 .pb or .pb.gz weights")
    parser.add_argument(
        "--compute-unit", choices=["all", "cpu", "cpu-gpu", "cpu-ne"], default="cpu-ne"
    )
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument(
        "--value-head-fp32", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--policy-head-fp32", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--positions", default="all")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def print_human(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print("MetalFish Core ML T1 BK root child-value probe")
    print(f"  Weights:   {result['weights']}")
    print(
        f"  Candidate: {result['compute_unit']} {result['precision']} "
        f"batch={result['batch_size']}"
    )
    print(
        "  Coverage:  "
        f"top1={summary['top1']}/{summary['total']} "
        f"top3={summary['top3']}/{summary['total']} "
        f"top5={summary['top5']}/{summary['total']} "
        f"top10={summary['top10']}/{summary['total']}"
    )
    print(
        "  Latency:   "
        f"median={summary['median_prediction_ms']:.3f} ms "
        f"mean={summary['mean_prediction_ms']:.3f} ms "
        f"build={summary['build_ms']:.1f} ms"
    )
    print(f"  Median expected rank: {summary['median_expected_rank']}")
    for row in result["positions"]:
        top_moves = comma_list(item["move"] for item in row["top"])
        rank = row["expected_rank"] if row["expected_rank"] is not None else "-"
        print(
            f"  {row['id']}: rank={rank!s:>2} expected={row['expected_uci']} "
            f"top={top_moves}"
        )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = run(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
