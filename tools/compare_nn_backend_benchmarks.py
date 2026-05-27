#!/usr/bin/env python3
"""Compare NN backend batch benchmark lines from test_nn_comparison logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BATCH_RE = re.compile(r"\bb(?P<size>\d+)=(?P<batch>[0-9.]+)ms/(?P<eval>[0-9.]+)ms_eval")
GRAPH_REUSE_RE = re.compile(r"\bb(?P<size>\d+)\b")
EXECUTOR_RE = re.compile(r"executor=([^,()]+(?:\([^)]*\))?)")


@dataclass(frozen=True)
class BatchTiming:
    batch_size: int
    batch_ms: float
    eval_ms: float

    @property
    def positions_per_second(self) -> float:
        return 1000.0 / self.eval_ms


@dataclass(frozen=True)
class BenchmarkLog:
    path: str
    label: str
    backend_line: str | None
    backend_after_line: str | None
    warmups: int | None
    batches: dict[int, BatchTiming]
    graph_reuse_batches: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-log", required=True)
    parser.add_argument("--actual-log", required=True)
    parser.add_argument("--expected-label")
    parser.add_argument("--actual-label")
    parser.add_argument("--summary-out")
    parser.add_argument("--min-common-batches", type=int, default=1)
    parser.add_argument("--require-graph-reuse", action="store_true")
    parser.add_argument(
        "--max-eval-ms-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional hard ceiling for actual/expected eval-ms ratio on every "
            "common batch. A value of 0 records data without failing on speed."
        ),
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite(value: float, *, label: str) -> float:
    require(math.isfinite(value) and value > 0.0, f"{label} must be positive")
    return value


def find_prefixed_line(lines: list[str], prefix: str) -> str | None:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped
    return None


def parse_executor(line: str | None) -> str | None:
    if not line:
        return None
    match = EXECUTOR_RE.search(line)
    return match.group(1).strip() if match else None


def parse_warmups(line: str | None) -> int | None:
    if line is None:
        return None
    _, _, value = line.partition(":")
    try:
        return int(value.strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid benchmark_warmups line: {line!r}") from exc


def parse_batches(line: str | None, *, path: str) -> dict[int, BatchTiming]:
    require(line is not None, f"{path}: missing batches line")
    batches: dict[int, BatchTiming] = {}
    for match in BATCH_RE.finditer(line):
        batch_size = int(match.group("size"))
        batch_ms = finite(float(match.group("batch")), label=f"b{batch_size} batch_ms")
        eval_ms = finite(float(match.group("eval")), label=f"b{batch_size} eval_ms")
        batches[batch_size] = BatchTiming(batch_size, batch_ms, eval_ms)
    require(bool(batches), f"{path}: no parseable batch timings")
    return batches


def parse_graph_reuse(line: str | None) -> list[int]:
    if line is None:
        return []
    return [int(match.group("size")) for match in GRAPH_REUSE_RE.finditer(line)]


def parse_log(path: str | Path, *, label: str) -> BenchmarkLog:
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")
    lines = text.splitlines()
    backend_line = find_prefixed_line(lines, "backend:")
    backend_after_line = find_prefixed_line(lines, "backend_after:")
    if label:
        require(
            label in text,
            f"{path_obj}: expected backend label {label!r} was not found",
        )
    warmups = parse_warmups(find_prefixed_line(lines, "benchmark_warmups:"))
    batches = parse_batches(find_prefixed_line(lines, "batches:"), path=str(path_obj))
    graph_reuse_batches = parse_graph_reuse(
        find_prefixed_line(lines, "graph_reuse_probe:")
    )
    return BenchmarkLog(
        path=str(path_obj),
        label=label,
        backend_line=backend_line,
        backend_after_line=backend_after_line,
        warmups=warmups,
        batches=batches,
        graph_reuse_batches=graph_reuse_batches,
    )


def timing_record(timing: BatchTiming) -> dict[str, float | int]:
    return {
        "batch_size": timing.batch_size,
        "batch_ms": timing.batch_ms,
        "eval_ms": timing.eval_ms,
        "positions_per_second": timing.positions_per_second,
    }


def best_batch(batches: dict[int, BatchTiming]) -> BatchTiming:
    return min(batches.values(), key=lambda timing: timing.eval_ms)


def compare_logs(
    expected: BenchmarkLog,
    actual: BenchmarkLog,
    args: argparse.Namespace,
) -> dict[str, Any]:
    common_sizes = sorted(set(expected.batches) & set(actual.batches))
    require(
        len(common_sizes) >= args.min_common_batches,
        f"only {len(common_sizes)} common batch timing(s), expected at least "
        f"{args.min_common_batches}",
    )
    if args.require_graph_reuse:
        require(
            bool(expected.graph_reuse_batches),
            f"{expected.path}: missing graph_reuse_probe",
        )
        require(
            bool(actual.graph_reuse_batches),
            f"{actual.path}: missing graph_reuse_probe",
        )

    common: list[dict[str, float | int]] = []
    for batch_size in common_sizes:
        left = expected.batches[batch_size]
        right = actual.batches[batch_size]
        eval_ratio = right.eval_ms / left.eval_ms
        batch_ratio = right.batch_ms / left.batch_ms
        if args.max_eval_ms_ratio > 0.0:
            require(
                eval_ratio <= args.max_eval_ms_ratio,
                f"b{batch_size} eval-ms ratio {eval_ratio:.6g} exceeds "
                f"{args.max_eval_ms_ratio}",
            )
        common.append(
            {
                "batch_size": batch_size,
                "expected_batch_ms": left.batch_ms,
                "actual_batch_ms": right.batch_ms,
                "batch_ms_ratio": batch_ratio,
                "expected_eval_ms": left.eval_ms,
                "actual_eval_ms": right.eval_ms,
                "eval_ms_ratio": eval_ratio,
                "actual_speedup_vs_expected": 1.0 / eval_ratio,
                "expected_positions_per_second": left.positions_per_second,
                "actual_positions_per_second": right.positions_per_second,
            }
        )

    expected_best = best_batch(expected.batches)
    actual_best = best_batch(actual.batches)
    return {
        "expected": {
            "path": expected.path,
            "label": expected.label,
            "backend_line": expected.backend_line,
            "backend_after_line": expected.backend_after_line,
            "executor_before": parse_executor(expected.backend_line),
            "executor_after": parse_executor(expected.backend_after_line),
            "warmups": expected.warmups,
            "batches": {
                str(size): timing_record(timing)
                for size, timing in sorted(expected.batches.items())
            },
            "graph_reuse_batches": expected.graph_reuse_batches,
            "best_batch": timing_record(expected_best),
        },
        "actual": {
            "path": actual.path,
            "label": actual.label,
            "backend_line": actual.backend_line,
            "backend_after_line": actual.backend_after_line,
            "executor_before": parse_executor(actual.backend_line),
            "executor_after": parse_executor(actual.backend_after_line),
            "warmups": actual.warmups,
            "batches": {
                str(size): timing_record(timing)
                for size, timing in sorted(actual.batches.items())
            },
            "graph_reuse_batches": actual.graph_reuse_batches,
            "best_batch": timing_record(actual_best),
        },
        "common_batches": common,
        "common_batch_count": len(common),
        "best_common_actual": min(common, key=lambda row: float(row["actual_eval_ms"])),
        "worst_eval_ms_ratio": max(float(row["eval_ms_ratio"]) for row in common),
    }


def main() -> int:
    args = parse_args()
    expected = parse_log(args.expected_log, label=args.expected_label or "")
    actual = parse_log(args.actual_log, label=args.actual_label or "")
    summary = compare_logs(expected, actual, args)

    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    best = summary["best_common_actual"]
    actual_best = summary["actual"]["best_batch"]
    print(
        "NN backend benchmark compare: PASS "
        f"common_batches={summary['common_batch_count']} "
        f"best_common=b{best['batch_size']} "
        f"actual_eval_ms={best['actual_eval_ms']:.4f} "
        f"expected_eval_ms={best['expected_eval_ms']:.4f} "
        f"speedup_vs_expected={best['actual_speedup_vs_expected']:.3f} "
        f"actual_best=b{actual_best['batch_size']} "
        f"actual_best_eval_ms={actual_best['eval_ms']:.4f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"NN backend benchmark compare: FAIL: {exc}")
        raise SystemExit(1)
