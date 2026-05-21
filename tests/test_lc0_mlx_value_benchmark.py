#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import lc0_mlx_value_benchmark as bench  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_device_mapping() -> None:
    fake_mx = types.SimpleNamespace(cpu="cpu-device", gpu="gpu-device")
    expect("cpu maps", bench.device_from_name(fake_mx, "cpu") == "cpu-device")
    expect("gpu maps", bench.device_from_name(fake_mx, "gpu") == "gpu-device")


def test_parse_args() -> None:
    args = bench.parse_args(
        [
            "networks/t1-256x10-distilled-swa-2432500.pb.gz",
            "--device",
            "cpu",
            "--precision",
            "fp16",
            "--benchmark",
            "--warmup",
            "2",
            "--iterations",
            "5",
        ]
    )
    expect("weights parsed", args.weights.endswith(".pb.gz"))
    expect("device parsed", args.device == "cpu")
    expect("precision parsed", args.precision == "fp16")
    expect("benchmark parsed", args.benchmark)
    expect("warmup parsed", args.warmup == 2)
    expect("iterations parsed", args.iterations == 5)


def test_percentile_reused() -> None:
    expect("p50", bench.percentile([5.0, 1.0, 3.0], 0.5) == 3.0)


def main() -> int:
    test_device_mapping()
    test_parse_args()
    test_percentile_reused()
    print("Lc0 MLX value benchmark tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
