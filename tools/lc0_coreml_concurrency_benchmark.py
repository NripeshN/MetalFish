#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import lc0_coreml_position_parity as parity  # noqa: E402


DEFAULT_FEN = parity.DEFAULT_FENS[1]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def latency_summary(latencies: list[float], batch_size: int) -> dict[str, Any]:
    if not latencies:
        raise RuntimeError("no latency samples recorded")
    median_ms = statistics.median(latencies)
    return {
        "iterations": len(latencies),
        "batch_size": batch_size,
        "median_ms": median_ms,
        "median_positions_per_second": 1000.0 * batch_size / median_ms,
        "mean_ms": statistics.fmean(latencies),
        "p90_ms": percentile(latencies, 0.90),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


def parse_probe_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{"):
            return json.loads(stripped)
    raise RuntimeError("Metal probe did not emit JSON")


def model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        weights=args.weights,
        batch_size=args.batch_size,
        candidate_compute_unit=args.coreml_compute_unit,
        candidate_precision=args.coreml_precision,
        candidate_value_head_fp32=args.coreml_value_head_fp32,
        candidate_policy_head_fp32=args.coreml_policy_head_fp32,
    )


def make_planes(fen: str, batch_size: int) -> np.ndarray:
    planes = parity.encode_fen_classical_112(fen)
    if batch_size > 1:
        planes = np.repeat(planes, batch_size, axis=0)
    return planes


def measure_coreml(
    model: Any, planes: np.ndarray, warmup: int, iterations: int, batch_size: int
) -> dict[str, Any]:
    for _ in range(warmup):
        model.predict({"x": planes})
    latencies: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.predict({"x": planes})
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latency_summary(latencies, batch_size)


def metal_probe_command(
    args: argparse.Namespace,
    iterations: int,
    ready_file: Path | None = None,
    start_file: Path | None = None,
) -> list[str]:
    command = [
        str(Path(args.metal_probe)),
        "--weights",
        str(Path(args.weights)),
        "--fen",
        args.fen,
        "--top",
        "0",
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(iterations),
        "--batch-size",
        str(args.batch_size),
    ]
    if ready_file is not None:
        command.extend(["--ready-file", str(ready_file)])
    if start_file is not None:
        command.extend(["--start-file", str(start_file)])
    return command


def run_metal_probe(args: argparse.Namespace, iterations: int) -> dict[str, Any]:
    command = metal_probe_command(args, iterations)
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"failed to run Metal probe {args.metal_probe}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        raise RuntimeError(
            f"Metal probe failed with exit code {exc.returncode}: {stderr or stdout}"
        ) from exc
    return parse_probe_json(completed.stdout)


def start_metal_probe(
    args: argparse.Namespace,
    iterations: int,
    ready_file: Path | None = None,
    start_file: Path | None = None,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        metal_probe_command(args, iterations, ready_file, start_file),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def wait_for_probe_ready(
    process: subprocess.Popen[str], ready_file: Path, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_file.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                "Metal probe exited before ready signal: "
                f"{stderr.strip() or stdout.strip()}"
            )
        time.sleep(0.001)
    process.kill()
    process.communicate()
    raise RuntimeError("Metal probe did not become ready before timeout")


def extract_latency(probe_result: dict[str, Any]) -> dict[str, Any]:
    latency = probe_result.get("latency")
    if not isinstance(latency, dict):
        raise RuntimeError("Metal probe JSON did not include latency")
    return latency


def slowdown(concurrent: dict[str, Any], solo: dict[str, Any]) -> float:
    return float(concurrent["median_ms"]) / float(solo["median_ms"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be positive")
    if args.iterations < 1 or args.metal_iterations < 1:
        raise RuntimeError("--iterations and --metal-iterations must be positive")
    if not Path(args.metal_probe).exists():
        raise RuntimeError(f"Metal probe not found: {args.metal_probe}")

    planes = make_planes(args.fen, args.batch_size)
    build_start = time.perf_counter()
    model = parity.build_model(model_args(args), "heads", True)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    coreml_solo = measure_coreml(
        model, planes, args.warmup, args.iterations, args.batch_size
    )
    metal_solo = extract_latency(run_metal_probe(args, args.metal_iterations))

    with tempfile.TemporaryDirectory(prefix="metalfish-ane-overlap-") as tmp:
        ready_file = Path(tmp) / "metal.ready"
        start_file = Path(tmp) / "metal.start"
        process = start_metal_probe(
            args, args.metal_iterations, ready_file, start_file
        )
        wait_for_probe_ready(process, ready_file, args.timeout)
        start_file.touch()
        try:
            coreml_concurrent = measure_coreml(
                model, planes, args.warmup, args.iterations, args.batch_size
            )
            stdout, stderr = process.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate()
            raise RuntimeError("concurrent Metal probe timed out") from exc
        except Exception:
            process.kill()
            process.communicate()
            raise
    if process.returncode != 0:
        raise RuntimeError(
            "concurrent Metal probe failed with exit code "
            f"{process.returncode}: {stderr.strip() or stdout.strip()}"
        )
    metal_concurrent = extract_latency(parse_probe_json(stdout))

    return {
        "weights": str(Path(args.weights).resolve()),
        "fen": args.fen,
        "batch_size": args.batch_size,
        "coreml": {
            "compute_unit": args.coreml_compute_unit,
            "precision": args.coreml_precision,
            "value_head_fp32": args.coreml_value_head_fp32,
            "policy_head_fp32": args.coreml_policy_head_fp32,
            "build_ms": build_ms,
            "solo": coreml_solo,
            "concurrent": coreml_concurrent,
            "slowdown": slowdown(coreml_concurrent, coreml_solo),
        },
        "metal": {
            "probe": str(Path(args.metal_probe).resolve()),
            "solo": metal_solo,
            "concurrent": metal_concurrent,
            "slowdown": slowdown(metal_concurrent, metal_solo),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Core ML T1 heads while MetalFish Metal inference runs concurrently."
    )
    parser.add_argument("weights", help="Lc0 T1-256 .pb or .pb.gz weights")
    parser.add_argument("--metal-probe", required=True, help="Path to build/metalfish_nn_probe")
    parser.add_argument("--fen", default=DEFAULT_FEN)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coreml-compute-unit", choices=["all", "cpu", "cpu-gpu", "cpu-ne"], default="cpu-ne")
    parser.add_argument("--coreml-precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--coreml-value-head-fp32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coreml-policy-head-fp32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--metal-iterations", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def print_human(result: dict[str, Any]) -> None:
    print("MetalFish Core ML/Metal concurrency benchmark")
    print(f"  Weights:    {result['weights']}")
    print(f"  Batch size: {result['batch_size']}")
    print(f"  Core ML:    {result['coreml']['compute_unit']} {result['coreml']['precision']}")
    print(f"  Build:      {result['coreml']['build_ms']:.1f} ms")
    core_solo = result["coreml"]["solo"]
    core_con = result["coreml"]["concurrent"]
    metal_solo = result["metal"]["solo"]
    metal_con = result["metal"]["concurrent"]
    print(
        "  Core ML solo:       "
        f"{core_solo['median_ms']:.3f} ms, "
        f"{core_solo['median_positions_per_second']:.1f} pos/s"
    )
    print(
        "  Core ML concurrent: "
        f"{core_con['median_ms']:.3f} ms, "
        f"{core_con['median_positions_per_second']:.1f} pos/s "
        f"({result['coreml']['slowdown']:.2f}x)"
    )
    print(
        "  Metal solo:         "
        f"{metal_solo['median_ms']:.3f} ms, "
        f"{metal_solo['median_positions_per_second']:.1f} pos/s"
    )
    print(
        "  Metal concurrent:   "
        f"{metal_con['median_ms']:.3f} ms, "
        f"{metal_con['median_positions_per_second']:.1f} pos/s "
        f"({result['metal']['slowdown']:.2f}x)"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
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
