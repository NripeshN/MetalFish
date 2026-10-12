#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = ROOT / "build" / "metalfish"
WEIGHTS = ROOT / "networks" / "BT4-1024x15x32h-swa-6147500.pb"


def run_precision(enabled: bool) -> None:
    expected = "FP16" if enabled else "FP32"
    rejected = "FP32" if enabled else "FP16"
    command = [
        sys.executable,
        str(ROOT / "tools" / "uci_smoke.py"),
        "--engine",
        str(ENGINE),
        "--timeout",
        "120",
        "--setoption",
        "UseHybridSearch=false",
        "--setoption",
        "UseMCTS=true",
        "--setoption",
        f"NNWeights={WEIGHTS}",
        "--setoption",
        "NNBackend=metal",
        "--setoption",
        f"NNMetalFP16={'true' if enabled else 'false'}",
        "--setoption",
        "MCTSMaxThreads=1",
        "--setoption",
        "TransformerLowTimeFallbackMs=0",
        "--go",
        "nodes 1",
        "--expect-output",
        f"Precision: {expected}",
        "--reject-output",
        f"Precision: {rejected}",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    if sys.platform != "darwin":
        print("Metal precision option test: SKIP (requires macOS)")
        return 0
    if not ENGINE.is_file() or not WEIGHTS.is_file():
        raise FileNotFoundError("Build the engine and download BT4 weights first")

    run_precision(False)
    run_precision(True)
    print("Metal precision option test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
