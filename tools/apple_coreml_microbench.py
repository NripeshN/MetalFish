#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def load_coremltools() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "coremltools and numpy are required. Use a temporary venv, for "
            "example: python3.11 -m venv /tmp/metalfish-coremltools-venv && "
            "/tmp/metalfish-coremltools-venv/bin/python -m pip install "
            "coremltools==9.0"
        ) from exc
    return np, ct, mb, types


def compute_unit_from_name(ct: Any, name: str) -> Any:
    units = {
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
    }
    return units[name]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def build_model(
    np: Any,
    ct: Any,
    mb: Any,
    types: Any,
    batch: int,
    inputs: int,
    outputs: int,
    compute_unit: Any,
    seed: int,
) -> tuple[Any, Any, Any]:
    rng = np.random.default_rng(seed)
    weight = rng.normal(0.0, 0.02, size=(outputs, inputs)).astype(np.float32)
    bias = rng.normal(0.0, 0.01, size=(outputs,)).astype(np.float32)

    @mb.program(input_specs=[mb.TensorSpec(shape=(batch, inputs), dtype=types.fp32)])
    def program(x):  # type: ignore[no-untyped-def]
        w = mb.const(val=weight)
        b = mb.const(val=bias)
        return mb.linear(x=x, weight=w, bias=b)

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_units=compute_unit,
    )
    return model, weight, bias


def benchmark_predict(model: Any, sample: Any, warmup: int, iterations: int) -> dict[str, Any]:
    for _ in range(warmup):
        model.predict({"x": sample})

    latencies_ms: list[float] = []
    last_output = None
    for _ in range(iterations):
        start = time.perf_counter()
        last_output = model.predict({"x": sample})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "iterations": iterations,
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p90_ms": percentile(latencies_ms, 0.90),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "output_name": next(iter(last_output.keys())) if last_output else "",
    }


def benchmark_numpy(
    np: Any,
    sample: Any,
    weight: Any,
    bias: Any,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        sample @ weight.T + bias

    latencies_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        sample @ weight.T + bias
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "iterations": iterations,
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p90_ms": percentile(latencies_ms, 0.90),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    np, ct, mb, types = load_coremltools()
    compute_unit = compute_unit_from_name(ct, args.compute_unit)
    sample = np.random.default_rng(args.seed + 1).normal(
        0.0, 1.0, size=(args.batch, args.inputs)
    ).astype(np.float32)

    build_start = time.perf_counter()
    model, weight, bias = build_model(
        np,
        ct,
        mb,
        types,
        args.batch,
        args.inputs,
        args.outputs,
        compute_unit,
        args.seed,
    )
    build_ms = (time.perf_counter() - build_start) * 1000.0

    package_path = ""
    if args.save_model:
        package_path = str(Path(args.save_model).resolve())
        model.save(package_path)
    else:
        with tempfile.TemporaryDirectory(prefix="metalfish-coreml-microbench-") as tmp:
            package_path = str(Path(tmp) / "DenseMicrobench.mlpackage")
            model.save(package_path)

    return {
        "batch": args.batch,
        "inputs": args.inputs,
        "outputs": args.outputs,
        "compute_unit": args.compute_unit,
        "build_ms": build_ms,
        "model_package": package_path if args.save_model else "",
        "coreml": benchmark_predict(model, sample, args.warmup, args.iterations),
        "numpy": benchmark_numpy(np, sample, weight, bias, args.warmup, args.iterations),
    }


def print_human(result: dict[str, Any]) -> None:
    print("MetalFish Core ML dense microbenchmark")
    print(
        f"  Shape:        batch={result['batch']} "
        f"inputs={result['inputs']} outputs={result['outputs']}"
    )
    print(f"  Compute unit: {result['compute_unit']}")
    print(f"  Build time:   {result['build_ms']:.3f} ms")
    coreml = result["coreml"]
    numpy = result["numpy"]
    print(
        "  Core ML:      "
        f"median={coreml['median_ms']:.4f} ms "
        f"mean={coreml['mean_ms']:.4f} ms "
        f"p90={coreml['p90_ms']:.4f} ms"
    )
    print(
        "  NumPy:        "
        f"median={numpy['median_ms']:.4f} ms "
        f"mean={numpy['mean_ms']:.4f} ms "
        f"p90={numpy['p90_ms']:.4f} ms"
    )
    if result["model_package"]:
        print(f"  Saved model:  {result['model_package']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark tiny Core ML dense kernels for Apple accelerator experiments."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--inputs", type=int, default=32)
    parser.add_argument("--outputs", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--compute-unit",
        choices=["all", "cpu", "cpu-gpu", "cpu-ne"],
        default="cpu-ne",
    )
    parser.add_argument("--save-model", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_benchmark(args)
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
