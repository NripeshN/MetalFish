#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "tests"))

from bk_parity import BK_POSITIONS, san_to_uci  # noqa: E402

PERF_POSITIONS: List[Tuple[str, str]] = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    (
        "sicilian",
        "r1bqkb1r/pp3ppp/2n2n2/2ppP3/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 2 6",
    ),
    (
        "bk07",
        "1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - - 0 1",
    ),
    (
        "middlegame",
        "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 6 7",
    ),
    ("endgame", "8/5pk1/5p1p/2R5/pp6/1P4PP/P4PK1/2r5 w - - 0 36"),
]


@dataclass
class SearchResult:
    position: str
    bestmove: str
    expected: List[str] = field(default_factory=list)
    correct: Optional[bool] = None
    nodes: int = 0
    nps: int = 0
    time_ms: int = 0
    elapsed_ms: int = 0
    error: str = ""


@dataclass
class EngineRun:
    label: str
    run_index: int
    tactical: List[SearchResult] = field(default_factory=list)
    performance: List[SearchResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def tactical_score(self) -> int:
        return sum(1 for result in self.tactical if result.correct)

    @property
    def tactical_completed(self) -> int:
        return sum(1 for result in self.tactical if not result.error)


@dataclass
class EngineSummary:
    scores: List[int]
    completed: List[int]
    score_mean: float
    score_min: int
    score_max: int
    nps_median: int
    nodes_median: int
    elapsed_median_ms: int
    errors: List[str]


class UCISession:
    def __init__(
        self,
        engine: pathlib.Path,
        label: str,
        cwd: pathlib.Path,
        timeout: float,
    ) -> None:
        self.engine = engine
        self.label = label
        self.cwd = cwd
        self.timeout = timeout
        self.output_tail: List[str] = []
        self.proc = subprocess.Popen(
            [str(engine)],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.wait_for("uciok", 120)

    def send(self, command: str) -> None:
        if not self.proc.stdin:
            raise RuntimeError(f"{self.label}: engine stdin closed")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def wait_for(self, prefix: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        if not self.proc.stdout:
            raise RuntimeError(f"{self.label}: engine stdout closed")
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.label}: process died while waiting for {prefix} "
                    f"(exit={self.proc.returncode}); tail={self.tail()}"
                )
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            text = line.strip()
            self.remember(text)
            if text.startswith(prefix):
                return text
        raise TimeoutError(
            f"{self.label}: timeout waiting for {prefix}; tail={self.tail()}"
        )

    def remember(self, line: str) -> None:
        self.output_tail.append(line)
        if len(self.output_tail) > 80:
            self.output_tail.pop(0)

    def tail(self) -> str:
        return " | ".join(self.output_tail[-12:])

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def ready(self) -> None:
        self.send("isready")
        self.wait_for("readyok", 120)

    def search(self, name: str, fen: str, movetime_ms: int) -> SearchResult:
        self.send("ucinewgame")
        self.ready()
        self.send(f"position fen {fen}")
        self.send(f"go movetime {movetime_ms}")

        result = SearchResult(position=name, bestmove="0000")
        deadline = time.monotonic() + max(90.0, movetime_ms / 1000.0 + self.timeout)
        start = time.monotonic()
        if not self.proc.stdout:
            raise RuntimeError(f"{self.label}: engine stdout closed")

        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.label}: process died during {name} "
                    f"(exit={self.proc.returncode}); tail={self.tail()}"
                )
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            text = line.strip()
            self.remember(text)
            if text.startswith("bestmove"):
                parts = text.split()
                if len(parts) > 1:
                    result.bestmove = parts[1]
                break
            if text.startswith("info "):
                parse_info_line(text, result)
        else:
            raise TimeoutError(f"{self.label}: search timeout on {name}; {self.tail()}")

        result.elapsed_ms = int((time.monotonic() - start) * 1000)
        self.ready()
        return result

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=5)


def parse_info_line(line: str, result: SearchResult) -> None:
    parts = line.split()
    for idx, token in enumerate(parts[:-1]):
        next_token = parts[idx + 1]
        if token == "nodes":
            result.nodes = parse_int(next_token, result.nodes)
        elif token == "nps":
            result.nps = parse_int(next_token, result.nps)
        elif token == "time":
            result.time_ms = parse_int(next_token, result.time_ms)


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def engine_options(weights: pathlib.Path, threads: int, hash_mb: int) -> Dict[str, str]:
    return {
        "Threads": str(threads),
        "Hash": str(hash_mb),
        "Move Overhead": "500",
        "UseMCTS": "false",
        "UseHybridSearch": "true",
        "NNWeights": str(weights),
        "MCTSAddDirichletNoise": "false",
        "HybridTrace": "false",
    }


def configure_engine(
    session: UCISession, weights: pathlib.Path, threads: int, hash_mb: int
) -> None:
    for name, value in engine_options(weights, threads, hash_mb).items():
        session.setoption(name, value)
    session.ready()


def select_bk_positions(selector: str) -> List[Tuple[str, List[str], str]]:
    if selector == "all":
        return BK_POSITIONS
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [pos for pos in BK_POSITIONS if pos[2] in requested]
    missing = requested.difference({pos[2] for pos in selected})
    if missing:
        raise ValueError(f"Unknown BK positions: {', '.join(sorted(missing))}")
    return selected


def run_engine_once(
    label: str,
    run_index: int,
    engine: pathlib.Path,
    cwd: pathlib.Path,
    weights: pathlib.Path,
    threads: int,
    hash_mb: int,
    bk_positions: Sequence[Tuple[str, List[str], str]],
    bk_movetime_ms: int,
    perf_movetime_ms: int,
    timeout_margin_s: float,
) -> EngineRun:
    run = EngineRun(label=label, run_index=run_index)
    session = UCISession(engine, label, cwd, timeout_margin_s)
    try:
        configure_engine(session, weights, threads, hash_mb)
        session.search("warmup", PERF_POSITIONS[0][1], min(1000, bk_movetime_ms))

        for fen, expected_san, name in bk_positions:
            expected = [san_to_uci(fen, san) for san in expected_san]
            try:
                result = session.search(name, fen, bk_movetime_ms)
                result.expected = expected
                result.correct = result.bestmove in expected
            except Exception as exc:
                result = SearchResult(
                    position=name,
                    bestmove="0000",
                    expected=expected,
                    correct=False,
                    error=str(exc),
                )
                run.errors.append(f"{name}: {exc}")
            run.tactical.append(result)

        for name, fen in PERF_POSITIONS:
            try:
                run.performance.append(session.search(name, fen, perf_movetime_ms))
            except Exception as exc:
                run.performance.append(
                    SearchResult(position=name, bestmove="0000", error=str(exc))
                )
                run.errors.append(f"perf/{name}: {exc}")
    finally:
        session.close()
    return run


def run_repeated(args: argparse.Namespace, label: str, engine: pathlib.Path, cwd: pathlib.Path) -> List[EngineRun]:
    runs = []
    bk_positions = select_bk_positions(args.positions)
    for run_index in range(args.repeat):
        print(f"\n[{label}] run {run_index + 1}/{args.repeat}", flush=True)
        run = run_engine_once(
            label,
            run_index,
            engine,
            cwd,
            args.weights.resolve(),
            args.threads_resolved,
            args.hash_mb,
            bk_positions,
            args.bk_movetime,
            args.perf_movetime,
            args.timeout_margin,
        )
        print(
            f"  tactical {run.tactical_score}/{len(bk_positions)} "
            f"completed={run.tactical_completed}/{len(bk_positions)} "
            f"errors={len(run.errors)}",
            flush=True,
        )
        runs.append(run)
    return runs


def summarize_engine(runs: Sequence[EngineRun]) -> EngineSummary:
    scores = [run.tactical_score for run in runs]
    completed = [run.tactical_completed for run in runs]
    perf = [result for run in runs for result in run.performance if not result.error]
    nps_values = [result.nps for result in perf if result.nps > 0]
    node_values = [result.nodes for result in perf if result.nodes > 0]
    elapsed_values = [result.elapsed_ms for result in perf if result.elapsed_ms > 0]
    errors = [error for run in runs for error in run.errors]
    return EngineSummary(
        scores=scores,
        completed=completed,
        score_mean=statistics.mean(scores) if scores else 0.0,
        score_min=min(scores) if scores else 0,
        score_max=max(scores) if scores else 0,
        nps_median=int(statistics.median(nps_values)) if nps_values else 0,
        nodes_median=int(statistics.median(node_values)) if node_values else 0,
        elapsed_median_ms=int(statistics.median(elapsed_values)) if elapsed_values else 0,
        errors=errors,
    )


def compare_summaries(
    baseline: EngineSummary,
    candidate: EngineSummary,
    total_positions: int,
    args: argparse.Namespace,
) -> List[str]:
    failures: List[str] = []
    if baseline.errors:
        failures.append(f"baseline had {len(baseline.errors)} errors")
    if candidate.errors:
        failures.append(f"candidate had {len(candidate.errors)} errors")
    if min(baseline.completed or [0]) < total_positions:
        failures.append("baseline did not complete all tactical positions")
    if min(candidate.completed or [0]) < total_positions:
        failures.append("candidate did not complete all tactical positions")

    if candidate.score_mean + 1e-9 < baseline.score_mean - args.max_bk_mean_drop:
        failures.append(
            f"candidate BK mean {candidate.score_mean:.2f} below baseline "
            f"{baseline.score_mean:.2f} by more than {args.max_bk_mean_drop:.2f}"
        )
    if candidate.score_min < baseline.score_min - args.max_bk_min_drop:
        failures.append(
            f"candidate BK worst run {candidate.score_min} below baseline worst "
            f"{baseline.score_min} by more than {args.max_bk_min_drop}"
        )
    if args.min_candidate_bk_score > 0 and candidate.score_min < args.min_candidate_bk_score:
        failures.append(
            f"candidate BK worst run {candidate.score_min}/{total_positions} below "
            f"absolute floor {args.min_candidate_bk_score}/{total_positions}"
        )

    if baseline.nps_median > 0 and candidate.nps_median > 0:
        floor = baseline.nps_median * (1.0 - args.max_perf_regression)
        if candidate.nps_median < floor:
            failures.append(
                f"candidate median NPS {candidate.nps_median} below baseline "
                f"{baseline.nps_median} with {args.max_perf_regression:.0%} tolerance"
            )
    elif candidate.nps_median <= 0:
        failures.append("candidate did not report usable NPS")

    return failures


def resolve_threads(value: str) -> int:
    if value != "auto":
        return max(3, int(value))
    return max(3, min(8, os.cpu_count() or 8))


def write_report(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-engine", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-engine", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-cwd", type=pathlib.Path, default=PROJ)
    parser.add_argument("--candidate-cwd", type=pathlib.Path, default=PROJ)
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument("--threads", default="auto")
    parser.add_argument("--hash-mb", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--positions", default="all")
    parser.add_argument("--bk-movetime", type=int, default=5000)
    parser.add_argument("--perf-movetime", type=int, default=2000)
    parser.add_argument("--timeout-margin", type=float, default=90.0)
    parser.add_argument("--max-bk-mean-drop", type=float, default=0.67)
    parser.add_argument("--max-bk-min-drop", type=int, default=1)
    parser.add_argument("--min-candidate-bk-score", type=int, default=0)
    parser.add_argument("--max-perf-regression", type=float, default=0.25)
    parser.add_argument("--json-out", type=pathlib.Path)
    return parser.parse_args(argv)


def validate_paths(args: argparse.Namespace) -> None:
    for path in [args.baseline_engine, args.candidate_engine, args.weights]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.max_perf_regression < 0 or args.max_perf_regression >= 1:
        raise ValueError("--max-perf-regression must be in [0, 1)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.threads_resolved = resolve_threads(args.threads)
    validate_paths(args)

    print("Hybrid regression comparison")
    print(f"  baseline:  {args.baseline_engine}")
    print(f"  candidate: {args.candidate_engine}")
    print(f"  weights:   {args.weights}")
    print(
        f"  repeat={args.repeat} threads={args.threads_resolved} "
        f"bk_movetime={args.bk_movetime} perf_movetime={args.perf_movetime}",
        flush=True,
    )

    baseline_runs = run_repeated(
        args, "baseline-main", args.baseline_engine.resolve(), args.baseline_cwd.resolve()
    )
    candidate_runs = run_repeated(
        args,
        "candidate-pr",
        args.candidate_engine.resolve(),
        args.candidate_cwd.resolve(),
    )

    baseline_summary = summarize_engine(baseline_runs)
    candidate_summary = summarize_engine(candidate_runs)
    total_positions = len(select_bk_positions(args.positions))
    failures = compare_summaries(
        baseline_summary, candidate_summary, total_positions, args
    )

    payload = {
        "config": {
            "repeat": args.repeat,
            "threads": args.threads_resolved,
            "hash_mb": args.hash_mb,
            "positions": args.positions,
            "bk_movetime_ms": args.bk_movetime,
            "perf_movetime_ms": args.perf_movetime,
            "max_bk_mean_drop": args.max_bk_mean_drop,
            "max_bk_min_drop": args.max_bk_min_drop,
            "min_candidate_bk_score": args.min_candidate_bk_score,
            "max_perf_regression": args.max_perf_regression,
        },
        "baseline": {
            "summary": asdict(baseline_summary),
            "runs": [asdict(run) for run in baseline_runs],
        },
        "candidate": {
            "summary": asdict(candidate_summary),
            "runs": [asdict(run) for run in candidate_runs],
        },
        "failures": failures,
    }
    if args.json_out:
        write_report(args.json_out, payload)
        print(f"Report: {args.json_out}")

    print("\nSummary")
    print(f"  baseline scores:  {baseline_summary.scores}")
    print(f"  candidate scores: {candidate_summary.scores}")
    print(
        f"  baseline median NPS={baseline_summary.nps_median} "
        f"candidate median NPS={candidate_summary.nps_median}"
    )

    if failures:
        print("\nRegression failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nRegression gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
