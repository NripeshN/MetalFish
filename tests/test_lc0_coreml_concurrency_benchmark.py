#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import lc0_coreml_concurrency_benchmark as bench  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_parse_probe_json_uses_last_json_line() -> None:
    parsed = bench.parse_probe_json(
        'noise\n{"old": 1}\nmore\n{"latency": {"median_ms": 2.5}}\n'
    )
    expect("last json", parsed["latency"]["median_ms"] == 2.5)


def test_latency_summary_reports_batch_throughput() -> None:
    summary = bench.latency_summary([4.0, 2.0, 3.0], 8)
    expect("median", summary["median_ms"] == 3.0)
    expect("pps", summary["median_positions_per_second"] == 1000.0 * 8 / 3.0)
    expect("p90", summary["p90_ms"] == 4.0)


def test_parse_args_defaults() -> None:
    args = bench.parse_args(["net.pb.gz", "--metal-probe", "build/metalfish_nn_probe"])
    expect("batch default", args.batch_size == 16)
    expect("metal weights default", args.metal_weights is None)
    expect("coreml unit", args.coreml_compute_unit == "cpu-ne")
    expect("coreml precision", args.coreml_precision == "fp16")
    expect("value head fp32", args.coreml_value_head_fp32)
    expect("policy head fp32 off", not args.coreml_policy_head_fp32)
    expect("metal iterations", args.metal_iterations == 60)


def test_metal_probe_command_contains_batch_and_iterations() -> None:
    args = bench.parse_args(
        [
            "net.pb.gz",
            "--metal-probe",
            "build/metalfish_nn_probe",
            "--batch-size",
            "4",
            "--iterations",
            "7",
            "--metal-iterations",
            "11",
        ]
    )
    command = bench.metal_probe_command(args, 11)
    expect("probe path", command[0] == "build/metalfish_nn_probe")
    expect("batch", command[command.index("--batch-size") + 1] == "4")
    expect("iterations", command[command.index("--iterations") + 1] == "11")


def test_metal_probe_command_can_use_separate_metal_weights() -> None:
    args = bench.parse_args(
        [
            "t1.pb.gz",
            "--metal-probe",
            "build/metalfish_nn_probe",
            "--metal-weights",
            "bt4.pb",
        ]
    )
    command = bench.metal_probe_command(args, 5)
    expect(
        "separate metal weights", command[command.index("--weights") + 1] == "bt4.pb"
    )


def test_metal_probe_command_can_add_signal_files() -> None:
    args = bench.parse_args(["net.pb.gz", "--metal-probe", "build/metalfish_nn_probe"])
    command = bench.metal_probe_command(
        args,
        3,
        pathlib.Path("/tmp/ready"),
        pathlib.Path("/tmp/start"),
    )
    expect("ready file", command[command.index("--ready-file") + 1] == "/tmp/ready")
    expect("start file", command[command.index("--start-file") + 1] == "/tmp/start")


def test_slowdown() -> None:
    expect(
        "ratio",
        bench.slowdown({"median_ms": 3.0}, {"median_ms": 2.0}) == 1.5,
    )


def main() -> int:
    test_parse_probe_json_uses_last_json_line()
    test_latency_summary_reports_batch_throughput()
    test_parse_args_defaults()
    test_metal_probe_command_contains_batch_and_iterations()
    test_metal_probe_command_can_use_separate_metal_weights()
    test_metal_probe_command_can_add_signal_files()
    test_slowdown()
    print("Lc0 Core ML concurrency benchmark tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
