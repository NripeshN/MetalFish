#!/usr/bin/env python3
from __future__ import annotations

import io
import pathlib
import sys
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import apple_accelerator_probe as probe  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_parse_key_value_lines() -> None:
    values = probe.parse_key_value_lines(
        "coreml=available\n"
        "metal_device=Apple M2 Max\n"
        "ignored line\n"
        "metal_has_unified_memory=true\n"
    )

    expect("coreml parsed", values["coreml"] == "available")
    expect("device parsed", values["metal_device"] == "Apple M2 Max")
    expect("ignored malformed line", "ignored line" not in values)


def test_collect_probe_skip_swift_has_stable_shape() -> None:
    result = probe.collect_probe(skip_swift=True)

    expect("result has system", isinstance(result["system"], str))
    expect("result has machine", isinstance(result["machine"], str))
    expect("swift skipped", result["swift_probe"]["stage"] == "skipped")
    expect("ane candidate boolean", isinstance(result["ane_candidate"], bool))
    expect(
        "mpsgraph candidate boolean",
        isinstance(result["mpsgraph_candidate"], bool),
    )


def test_human_output_names_candidates() -> None:
    result = {
        "system": "Darwin",
        "machine": "arm64",
        "apple_silicon": True,
        "swiftc": "/usr/bin/swiftc",
        "coremlcompiler": "/usr/bin/coremlcompiler",
        "ane_candidate": True,
        "mpsgraph_candidate": True,
        "swift_probe": {
            "ok": True,
            "stage": "run",
            "stdout": "",
            "stderr": "",
            "capabilities": {
                "coreml": "available",
                "coreml_cpu_and_neural_engine": "available",
                "metal_device": "Apple M2 Max",
                "metal_has_unified_memory": "true",
                "mpsgraph": "available",
            },
        },
    }

    out = io.StringIO()
    with redirect_stdout(out):
        probe.print_human(result)
    text = out.getvalue()

    expect("prints ANE candidate", "ANE candidate:   True" in text)
    expect("prints MPSGraph candidate", "MPSGraph candidate: True" in text)


def main() -> int:
    test_parse_key_value_lines()
    test_collect_probe_skip_swift_has_stable_shape()
    test_human_output_names_candidates()
    print("Apple accelerator probe tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
