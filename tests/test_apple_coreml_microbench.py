#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import apple_coreml_microbench as microbench  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_percentile_uses_sorted_values() -> None:
    values = [10.0, 1.0, 5.0, 3.0]

    expect("p0", microbench.percentile(values, 0.0) == 1.0)
    expect("p90", microbench.percentile(values, 0.90) == 10.0)
    expect("empty percentile", microbench.percentile([], 0.5) == 0.0)


def test_parse_transformer_args() -> None:
    args = microbench.parse_args(
        [
            "--model",
            "transformer",
            "--batch",
            "2",
            "--tokens",
            "64",
            "--channels",
            "128",
            "--heads",
            "8",
            "--ffn-mult",
            "4",
            "--layers",
            "2",
            "--compute-unit",
            "cpu-ne",
        ]
    )

    expect("model parsed", args.model == "transformer")
    expect("tokens parsed", args.tokens == 64)
    expect("channels parsed", args.channels == 128)
    expect("heads parsed", args.heads == 8)
    expect("layers parsed", args.layers == 2)
    expect("compute unit parsed", args.compute_unit == "cpu-ne")


def test_compute_unit_mapping_uses_expected_names() -> None:
    fake_ct = types.SimpleNamespace(
        ComputeUnit=types.SimpleNamespace(
            ALL="all", CPU_ONLY="cpu", CPU_AND_GPU="cpu-gpu", CPU_AND_NE="cpu-ne"
        )
    )

    expect(
        "cpu-ne maps",
        microbench.compute_unit_from_name(fake_ct, "cpu-ne") == "cpu-ne",
    )
    expect("all maps", microbench.compute_unit_from_name(fake_ct, "all") == "all")


def main() -> int:
    test_percentile_uses_sorted_values()
    test_parse_transformer_args()
    test_compute_unit_mapping_uses_expected_names()
    print("Apple Core ML microbench tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
