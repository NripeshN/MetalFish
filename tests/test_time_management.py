#!/usr/bin/env python3
"""
Time management audit: verifies the engine allocates time correctly
under various time control scenarios.

Usage:
    python3 tests/test_time_management.py

Requires: ./build/metalfish binary and networks/BT4-1024x15x32h-swa-6147500.pb.gz
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
import queue
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = ROOT / "build" / "metalfish"
NETWORK = ROOT / "networks" / "BT4-1024x15x32h-swa-6147500.pb.gz"


def find_network() -> str:
    if NETWORK.exists():
        return str(NETWORK)
    # fallback: search for any .pb or .pb.gz in networks/
    for pattern in ("*.pb.gz", "*.pb"):
        nets = list((ROOT / "networks").glob(pattern))
        if nets:
            return str(nets[0])
    print("ERROR: No network file found in networks/", file=sys.stderr)
    sys.exit(1)


class UCIEngine:
    """Minimal UCI engine wrapper with threaded stdout reader."""

    def __init__(self, binary: str):
        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._reader, daemon=True
        )
        self._reader_thread.start()

    def _reader(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._queue.put(line.rstrip("\n\r"))
        self._queue.put(None)

    def send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read_until(self, token: str, timeout: float = 120.0) -> tuple[list[str], float]:
        lines: list[str] = []
        start = time.monotonic()
        deadline = start + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for '{token}' after {timeout:.1f}s. "
                    f"Got {len(lines)} lines so far."
                )
            try:
                line = self._queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                raise EOFError("Engine process closed stdout unexpectedly")
            lines.append(line)
            if line.startswith(token):
                elapsed = time.monotonic() - start
                return lines, elapsed

    def uci_init(self, network: str) -> None:
        self.send("uci")
        self.read_until("uciok")
        self.send("setoption name UseHybridSearch value true")
        self.send(f"setoption name NNWeights value {network}")
        self.send("setoption name HybridMCTSRootReject value true")
        self.send("setoption name Threads value 4")
        self.send("setoption name Hash value 256")
        self.send("isready")
        self.read_until("readyok", timeout=30.0)

    def go_timed(
        self,
        position_cmd: str,
        go_cmd: str,
        timeout: float = 120.0,
    ) -> tuple[str, float, list[str]]:
        self.send(position_cmd)
        self.send("isready")
        self.read_until("readyok", timeout=10.0)
        self.send(go_cmd)
        lines, elapsed = self.read_until("bestmove", timeout=timeout)
        bestmove_line = lines[-1]
        info_lines = [l for l in lines if l.startswith("info")]
        return bestmove_line, elapsed, info_lines

    def quit(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# Test scenarios (as specified)
# ---------------------------------------------------------------------------

class Scenario:
    def __init__(self, name, go, min_time, max_time, description):
        self.name = name
        self.position = "position startpos"
        self.go = go
        self.min_time = min_time
        self.max_time = max_time
        self.description = description


SCENARIOS = [
    Scenario(
        name="Bullet 1+0",
        go="go wtime 60000 btime 60000 winc 0 binc 0",
        min_time=1.0,
        max_time=4.0,
        description="60s each, no increment. Expected: ~2-4s.",
    ),
    Scenario(
        name="Blitz 3+2",
        go="go wtime 180000 btime 180000 winc 2000 binc 2000",
        min_time=3.0,
        max_time=10.0,
        description="180s + 2s increment. Expected: ~5-10s.",
    ),
    Scenario(
        name="Rapid 10+5",
        go="go wtime 600000 btime 600000 winc 5000 binc 5000",
        min_time=8.0,
        max_time=25.0,
        description="600s + 5s increment. Expected: ~15-25s.",
    ),
    Scenario(
        name="Low time panic (5s left)",
        go="go wtime 5000 btime 60000 winc 0 binc 0",
        min_time=0.0,
        max_time=1.0,
        description="5s left, no increment. Expected: <1s.",
    ),
    Scenario(
        name="Very low time (1s left)",
        go="go wtime 1000 btime 60000 winc 0 binc 0",
        min_time=0.0,
        max_time=0.5,
        description="1s left, no increment. Expected: <0.5s.",
    ),
    Scenario(
        name="Increment only (3s + 3s inc)",
        go="go wtime 3000 btime 60000 winc 3000 binc 3000",
        min_time=0.1,
        max_time=3.0,
        description="3s left + 3s increment buffer. Expected: ~2-3s.",
    ),
]


def extract_time_budget_info(info_lines: list[str]) -> str | None:
    """Extract 'Time budget: Xms' from info string lines."""
    for line in info_lines:
        if "Time budget:" in line:
            return line
    return None


def run_audit() -> None:
    network = find_network()
    print("=" * 72)
    print("MetalFish Time Management Audit")
    print("=" * 72)
    print(f"Engine:  {ENGINE}")
    print(f"Network: {network}")
    print(f"Config:  UseHybridSearch=true, HybridMCTSRootReject=true,")
    print(f"         Threads=4, Hash=256")
    print("=" * 72)
    print()

    if not ENGINE.exists():
        print("ERROR: Engine binary not found. Build first.", file=sys.stderr)
        sys.exit(1)

    engine = UCIEngine(str(ENGINE))
    try:
        print("Initializing engine (UCI handshake + options)...")
        engine.uci_init(network)
        print("Engine ready.\n")

        results: list[dict] = []
        passes = 0
        failures = 0

        for i, sc in enumerate(SCENARIOS, 1):
            print(f"--- Test {i}/{len(SCENARIOS)}: {sc.name} ---")
            print(f"  {sc.description}")
            print(f"  Command: {sc.go}")
            print(f"  Expected range: {sc.min_time:.1f}s - {sc.max_time:.1f}s")

            # Reset state
            engine.send("ucinewgame")
            engine.send("isready")
            engine.read_until("readyok", timeout=15.0)

            try:
                bestmove, elapsed, info_lines = engine.go_timed(
                    sc.position,
                    sc.go,
                    timeout=max(sc.max_time * 3, 60.0),
                )
            except TimeoutError as e:
                print(f"  TIMEOUT: {e}")
                results.append({
                    "name": sc.name,
                    "status": "TIMEOUT",
                    "elapsed": None,
                })
                failures += 1
                print()
                continue

            in_range = sc.min_time <= elapsed <= sc.max_time
            status = "PASS" if in_range else "FAIL"

            if in_range:
                passes += 1
            else:
                failures += 1

            budget_info = extract_time_budget_info(info_lines)
            results.append({
                "name": sc.name,
                "status": status,
                "elapsed": elapsed,
                "min": sc.min_time,
                "max": sc.max_time,
                "bestmove": bestmove,
                "budget_info": budget_info,
            })

            marker = "PASS" if in_range else "**FAIL**"
            print(f"  Actual time: {elapsed:.3f}s  [{marker}]")
            if budget_info:
                print(f"  Engine reported: {budget_info}")
            print(f"  {bestmove}")
            if not in_range:
                if elapsed < sc.min_time:
                    print(f"  ISSUE: Too fast! Expected >= {sc.min_time:.1f}s")
                else:
                    print(f"  ISSUE: Too slow! Expected <= {sc.max_time:.1f}s")
            print()

        # Summary table
        print("=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(f"{'Scenario':<32} {'Expected':>14} {'Actual':>10} {'Status':>8}")
        print("-" * 72)
        for r in results:
            if r["elapsed"] is not None:
                time_str = f"{r['elapsed']:.2f}s"
                range_str = f"{r['min']:.1f}-{r['max']:.1f}s"
            else:
                time_str = "TIMEOUT"
                range_str = "N/A"
            print(f"{r['name']:<32} {range_str:>14} {time_str:>10} {r['status']:>8}")
        print("-" * 72)
        print(f"Passed: {passes}/{len(SCENARIOS)}  |  Failed: {failures}/{len(SCENARIOS)}")
        print("=" * 72)

        if failures > 0:
            print("\nISSUES DETECTED:")
            for r in results:
                if r["status"] != "PASS":
                    if r["elapsed"] is not None:
                        if r["elapsed"] > r["max"]:
                            print(f"  - {r['name']}: took {r['elapsed']:.2f}s "
                                  f"(limit {r['max']:.1f}s) - risk of flagging")
                        elif r["elapsed"] < r["min"]:
                            print(f"  - {r['name']}: took {r['elapsed']:.2f}s "
                                  f"(min {r['min']:.1f}s) - underusing time")
                    else:
                        print(f"  - {r['name']}: TIMEOUT")
            sys.exit(1)
        else:
            print("\nAll tests PASSED.")

    finally:
        engine.quit()


if __name__ == "__main__":
    run_audit()
