#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import lc0_coreml_value_export as exporter  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_percentile_uses_sorted_values() -> None:
    expect("empty", exporter.percentile([], 0.5) == 0.0)
    expect("p0", exporter.percentile([3.0, 1.0, 2.0], 0.0) == 1.0)
    expect("p90", exporter.percentile([3.0, 1.0, 2.0], 0.9) == 3.0)


def test_compute_unit_mapping() -> None:
    fake_ct = types.SimpleNamespace(
        ComputeUnit=types.SimpleNamespace(
            ALL="all", CPU_ONLY="cpu", CPU_AND_GPU="cpu-gpu", CPU_AND_NE="cpu-ne"
        )
    )
    expect("cpu", exporter.compute_unit_from_name(fake_ct, "cpu") == "cpu")
    expect("cpu-ne", exporter.compute_unit_from_name(fake_ct, "cpu-ne") == "cpu-ne")


def test_precision_mapping() -> None:
    fake_ct = types.SimpleNamespace(
        precision=types.SimpleNamespace(FLOAT16="fp16", FLOAT32="fp32")
    )
    expect("fp16", exporter.precision_from_name(fake_ct, "fp16") == "fp16")
    expect("fp32", exporter.precision_from_name(fake_ct, "fp32") == "fp32")


def test_parse_args() -> None:
    args = exporter.parse_args(
        [
            "networks/t1-256x10-distilled-swa-2432500.pb.gz",
            "--compute-unit",
            "cpu-ne",
            "--precision",
            "fp32",
            "--benchmark",
            "--warmup",
            "2",
            "--iterations",
            "7",
        ]
    )
    expect("weights parsed", args.weights.endswith(".pb.gz"))
    expect("compute parsed", args.compute_unit == "cpu-ne")
    expect("precision parsed", args.precision == "fp32")
    expect("benchmark parsed", args.benchmark)
    expect("warmup parsed", args.warmup == 2)
    expect("iterations parsed", args.iterations == 7)


def main() -> int:
    test_percentile_uses_sorted_values()
    test_compute_unit_mapping()
    test_precision_mapping()
    test_parse_args()
    print("Lc0 Core ML value exporter tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
