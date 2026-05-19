#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


SWIFT_PROBE = r"""
import CoreML
import Metal
import MetalPerformanceShadersGraph

let config = MLModelConfiguration()
config.computeUnits = .all
print("coreml=available")
print("coreml_compute_units_all=available")

if #available(macOS 14.0, *) {
    config.computeUnits = .cpuAndNeuralEngine
    print("coreml_cpu_and_neural_engine=available")
} else {
    print("coreml_cpu_and_neural_engine=unavailable")
}

if let device = MTLCreateSystemDefaultDevice() {
    print("metal_device=\(device.name)")
    #if os(macOS)
    print("metal_has_unified_memory=\(device.hasUnifiedMemory)")
    #endif
} else {
    print("metal_device=none")
}

let graph = MPSGraph()
let shape: [NSNumber] = [1]
let a = graph.placeholder(shape: shape, dataType: .float32, name: "a")
_ = graph.addition(a, a, name: "b")
print("mpsgraph=available")
"""


def run_command(args: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def parse_key_value_lines(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            values[key.strip()] = value.strip()
    return values


def find_xcrun_tool(tool: str) -> str | None:
    xcrun = shutil.which("xcrun")
    if xcrun:
        result = run_command([xcrun, "--find", tool], timeout=10.0)
        if result.returncode == 0:
            path = result.stdout.strip()
            if path:
                return path
    return shutil.which(tool)


def swiftc_command() -> list[str] | None:
    xcrun = shutil.which("xcrun")
    if xcrun and find_xcrun_tool("swiftc"):
        return [xcrun, "swiftc"]
    swiftc = shutil.which("swiftc")
    if swiftc:
        return [swiftc]
    return None


def run_swift_probe(swiftc: list[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="metalfish-apple-accel-") as tmp:
        tmpdir = Path(tmp)
        source = tmpdir / "probe.swift"
        binary = tmpdir / "probe"
        source.write_text(SWIFT_PROBE, encoding="utf-8")

        compile_result = run_command([*swiftc, str(source), "-o", str(binary)])
        if compile_result.returncode != 0:
            return {
                "ok": False,
                "stage": "compile",
                "stdout": compile_result.stdout,
                "stderr": compile_result.stderr,
                "capabilities": {},
            }

        run_result = run_command([str(binary)])
        return {
            "ok": run_result.returncode == 0,
            "stage": "run",
            "stdout": run_result.stdout,
            "stderr": run_result.stderr,
            "capabilities": parse_key_value_lines(run_result.stdout),
        }


def collect_probe(skip_swift: bool = False) -> dict[str, Any]:
    swiftc = find_xcrun_tool("swiftc")
    swiftc_launcher = swiftc_command()
    coremlcompiler = find_xcrun_tool("coremlcompiler")
    machine = platform.machine()
    system = platform.system()
    result: dict[str, Any] = {
        "system": system,
        "machine": machine,
        "apple_silicon": system == "Darwin" and machine == "arm64",
        "swiftc": swiftc,
        "coremlcompiler": coremlcompiler,
        "swift_probe": {
            "ok": False,
            "stage": "skipped" if skip_swift else "unavailable",
            "stdout": "",
            "stderr": "",
            "capabilities": {},
        },
    }

    if swiftc_launcher and not skip_swift and system == "Darwin":
        result["swift_probe"] = run_swift_probe(swiftc_launcher)

    capabilities = result["swift_probe"].get("capabilities", {})
    result["ane_candidate"] = bool(
        result["apple_silicon"]
        and capabilities.get("coreml") == "available"
        and capabilities.get("coreml_cpu_and_neural_engine") == "available"
    )
    result["mpsgraph_candidate"] = bool(
        result["apple_silicon"]
        and capabilities.get("mpsgraph") == "available"
        and capabilities.get("metal_device")
        and capabilities.get("metal_device") != "none"
    )
    return result


def print_human(result: dict[str, Any]) -> None:
    capabilities = result["swift_probe"].get("capabilities", {})
    print("MetalFish Apple accelerator probe")
    print(f"  System:          {result['system']} {result['machine']}")
    print(f"  Apple Silicon:   {result['apple_silicon']}")
    print(f"  swiftc:          {result['swiftc'] or 'not found'}")
    print(f"  coremlcompiler:  {result['coremlcompiler'] or 'not found'}")
    print(f"  Core ML:         {capabilities.get('coreml', 'unknown')}")
    print(
        "  ANE compute:     "
        f"{capabilities.get('coreml_cpu_and_neural_engine', 'unknown')}"
    )
    print(f"  Metal device:    {capabilities.get('metal_device', 'unknown')}")
    print(f"  Unified memory:  {capabilities.get('metal_has_unified_memory', 'unknown')}")
    print(f"  MPSGraph:        {capabilities.get('mpsgraph', 'unknown')}")
    print(f"  ANE candidate:   {result['ane_candidate']}")
    print(f"  MPSGraph candidate: {result['mpsgraph_candidate']}")
    if not result["swift_probe"].get("ok") and result["swift_probe"].get("stderr"):
        print()
        print(textwrap.indent(result["swift_probe"]["stderr"].strip(), "  swift: "))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Apple Core ML/ANE and MPSGraph availability for experimental "
            "MetalFish backend work."
        )
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument(
        "--skip-swift",
        action="store_true",
        help="skip compiling the Swift framework probe",
    )
    parser.add_argument(
        "--require-ane",
        action="store_true",
        help="return non-zero when Core ML cpuAndNeuralEngine is unavailable",
    )
    args = parser.parse_args(argv)

    result = collect_probe(skip_swift=args.skip_swift)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)

    if args.require_ane and not result["ane_candidate"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
