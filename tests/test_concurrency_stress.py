#!/usr/bin/env python3
"""Concurrency stress tests for MetalFish Hybrid.

Tests rapid position changes and ponderhit transitions under high load:
  1. Rapid position cycling: start/stop searches quickly across many positions
  2. Ponderhit continuation: verify ponderhit extends a ponder search correctly

Fails on: crash, hang (>15s), or missing bestmove.
"""

import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
import time

import chess

PROJ = pathlib.Path(__file__).resolve().parent.parent
ENGINE = PROJ / "build" / "metalfish"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"

TIMEOUT_BESTMOVE = 15.0

# 20 different positions for rapid cycling
RAPID_POSITIONS = [
    "position startpos",
    "position startpos moves e2e4",
    "position startpos moves e2e4 e7e5",
    "position startpos moves d2d4",
    "position startpos moves d2d4 d7d5",
    "position startpos moves g1f3",
    "position startpos moves e2e4 c7c5",
    "position startpos moves e2e4 e7e6",
    "position startpos moves d2d4 g8f6",
    "position startpos moves c2c4",
    "position startpos moves e2e4 e7e5 g1f3",
    "position startpos moves e2e4 e7e5 g1f3 b8c6",
    "position startpos moves d2d4 d7d5 c2c4",
    "position startpos moves d2d4 d7d5 c2c4 e7e6",
    "position startpos moves e2e4 c7c5 g1f3",
    "position startpos moves e2e4 c7c5 g1f3 d7d6",
    "position startpos moves g1f3 d7d5 d2d4",
    "position startpos moves e2e4 e7e5 f1c4",
    "position startpos moves d2d4 g8f6 c2c4 g7g6",
    "position startpos moves e2e4 d7d6 d2d4 g8f6",
]


def main():
    if not ENGINE.exists():
        print(f"ERROR: Engine not found at {ENGINE}")
        return 1
    if not WEIGHTS.exists():
        print(f"ERROR: Weights not found at {WEIGHTS}")
        return 1

    # --- Set up engine process ---
    stderr_fd = tempfile.NamedTemporaryFile(
        prefix="metalfish_concurrency_", suffix=".stderr", delete=False, mode="w"
    )

    proc = subprocess.Popen(
        [str(ENGINE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_fd,
        text=True,
        bufsize=1,
    )

    stdout_lines = queue.Queue()

    def pump_stdout():
        for line in proc.stdout:
            stdout_lines.put(line.strip())
        stdout_lines.put(None)

    stdout_thread = threading.Thread(target=pump_stdout, daemon=True)
    stdout_thread.start()

    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def read_until(prefix, timeout=TIMEOUT_BESTMOVE):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None and stdout_lines.empty():
                return None
            remaining = max(0.0, min(0.1, deadline - time.time()))
            try:
                line = stdout_lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                return None
            if line.startswith(prefix):
                return line
        return "TIMEOUT"

    def alive():
        return proc.poll() is None

    def drain_queue():
        """Drain any remaining lines from the queue."""
        while not stdout_lines.empty():
            try:
                stdout_lines.get_nowait()
            except queue.Empty:
                break

    # --- Initialize engine ---
    send("uci")
    r = read_until("uciok", 30)
    if not r or r == "TIMEOUT":
        print("ERROR: uciok timeout")
        proc.kill()
        proc.wait(timeout=5)
        stderr_fd.close()
        return 1

    send("setoption name UseMCTS value false")
    send("setoption name UseHybridSearch value true")
    send(f"setoption name NNWeights value {WEIGHTS}")
    send("setoption name Threads value 4")
    send("setoption name Hash value 256")
    send("setoption name HybridMCTSRootReject value true")
    send("setoption name HybridMCTSThreads value 1")
    send("setoption name HybridABThreads value 3")
    send("setoption name MCTSAddDirichletNoise value false")
    send("setoption name Ponder value true")
    send("isready")
    r = read_until("readyok", 120)
    if not r or r == "TIMEOUT":
        print("ERROR: readyok timeout")
        proc.kill()
        proc.wait(timeout=5)
        stderr_fd.close()
        return 1

    results = {
        "rapid_cycling_ok": 0,
        "rapid_cycling_total": 0,
        "ponderhit_continuation_ok": False,
        "crashes": 0,
        "timeouts": 0,
        "hangs": 0,
    }

    # =========================================================================
    # TEST 1: Rapid position change cycling (20 iterations)
    # =========================================================================
    print("=" * 60)
    print("TEST 1: Rapid position change cycling")
    print("  Start go infinite, stop after 500ms, switch position")
    print("  20 iterations")
    print("=" * 60)

    for i in range(20):
        pos_cmd = RAPID_POSITIONS[i % len(RAPID_POSITIONS)]
        results["rapid_cycling_total"] += 1

        # Send position and start infinite search
        send(pos_cmd)
        send("go infinite")

        # Wait 500ms
        time.sleep(0.5)

        if not alive():
            print(f"  [{i+1}/20] CRASH during search (exit {proc.returncode})")
            results["crashes"] += 1
            break

        # Stop and collect bestmove
        send("stop")
        r = read_until("bestmove", TIMEOUT_BESTMOVE)

        if not alive():
            print(f"  [{i+1}/20] CRASH after stop (exit {proc.returncode})")
            results["crashes"] += 1
            break

        if r is None:
            print(f"  [{i+1}/20] ENGINE DIED - no bestmove")
            results["crashes"] += 1
            break

        if r == "TIMEOUT":
            print(f"  [{i+1}/20] HANG - no bestmove within {TIMEOUT_BESTMOVE}s")
            results["hangs"] += 1
            break

        if not r.startswith("bestmove"):
            print(f"  [{i+1}/20] UNEXPECTED response: {r}")
            results["timeouts"] += 1
            break

        results["rapid_cycling_ok"] += 1

        # Print progress every 5 iterations
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/20] OK - got: {r}")

    rapid_passed = results["rapid_cycling_ok"] == 20
    if rapid_passed:
        print(f"  PASS: All 20 rapid cycles completed successfully")
    else:
        print(
            f"  FAIL: Only {results['rapid_cycling_ok']}/20 cycles completed"
        )

    # =========================================================================
    # TEST 2: Ponderhit continuation
    # =========================================================================
    print()
    print("=" * 60)
    print("TEST 2: Ponderhit continuation")
    print("  go ponder movetime 10000, then ponderhit after 1s")
    print("  Engine should continue and stop on its own")
    print("=" * 60)

    if alive() and results["crashes"] == 0:
        drain_queue()

        # Start a ponder search with movetime 10000
        send("position startpos")
        send("go ponder movetime 10000")

        # Wait 1 second then send ponderhit
        time.sleep(1.0)

        if not alive():
            print(f"  CRASH during ponder (exit {proc.returncode})")
            results["crashes"] += 1
        else:
            # Send ponderhit - engine should continue searching for the remaining
            # movetime and then emit bestmove on its own
            send("ponderhit")

            # Wait for bestmove - should come within ~9s (10s movetime minus the 1s
            # we already waited, plus some margin)
            r = read_until("bestmove", 12.0)

            if not alive():
                print(f"  CRASH after ponderhit (exit {proc.returncode})")
                results["crashes"] += 1
            elif r is None:
                print("  ENGINE DIED after ponderhit")
                results["crashes"] += 1
            elif r == "TIMEOUT":
                print("  HANG - no bestmove within 12s after ponderhit")
                results["hangs"] += 1
                # Try to recover
                send("stop")
                read_until("bestmove", 5)
            else:
                parts = r.split()
                if len(parts) >= 2 and parts[0] == "bestmove":
                    best = parts[1]
                    # Validate it is a legal move from startpos
                    board = chess.Board()
                    try:
                        move = chess.Move.from_uci(best)
                        if move in board.legal_moves:
                            results["ponderhit_continuation_ok"] = True
                            print(f"  PASS: Received legal bestmove after ponderhit: {r}")
                        else:
                            print(f"  FAIL: Illegal bestmove: {best}")
                            results["timeouts"] += 1
                    except ValueError:
                        print(f"  FAIL: Invalid move format: {best}")
                        results["timeouts"] += 1
                else:
                    print(f"  FAIL: Unexpected response: {r}")
                    results["timeouts"] += 1
    else:
        print("  SKIPPED (engine crashed in previous test)")

    # =========================================================================
    # Cleanup
    # =========================================================================
    if alive():
        send("quit")
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    stderr_fd.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Rapid cycling:        {results['rapid_cycling_ok']}/20 OK")
    print(f"  Ponderhit continuation: {'PASS' if results['ponderhit_continuation_ok'] else 'FAIL'}")
    print(f"  Crashes:              {results['crashes']}")
    print(f"  Timeouts:             {results['timeouts']}")
    print(f"  Hangs:                {results['hangs']}")
    print()

    all_passed = (
        rapid_passed
        and results["ponderhit_continuation_ok"]
        and results["crashes"] == 0
        and results["timeouts"] == 0
        and results["hangs"] == 0
    )

    if all_passed:
        print("OVERALL: PASS - All concurrency stress tests passed")
        return 0
    else:
        print("OVERALL: FAIL - One or more tests failed")
        stderr_path = pathlib.Path(stderr_fd.name)
        if stderr_path.exists():
            print(f"\nEngine stderr log: {stderr_path}")
            content = stderr_path.read_text(errors="replace").splitlines()
            if content:
                print("Last 30 lines of stderr:")
                for line in content[-30:]:
                    print(f"  {line}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
