#!/usr/bin/env python3
"""Stress test: rapidly start/stop ponder searches to verify no crashes.

Sends go ponder / stop / go / stop sequences to the hybrid engine
many times in quick succession. If the engine segfaults, the process
dies and this test reports failure.
"""

import subprocess
import sys
import time
import pathlib

PROJ = pathlib.Path(__file__).resolve().parent.parent
ENGINE = PROJ / "build" / "metalfish"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"

ITERATIONS = 50
POSITIONS = [
    "startpos",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2",
    "rnbqkb1r/pp2pppp/2p2n2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
]


def main():
    if not ENGINE.exists():
        print(f"ERROR: Engine not found at {ENGINE}")
        return 1
    if not WEIGHTS.exists():
        print(f"ERROR: Weights not found at {WEIGHTS}")
        return 1

    print(f"Ponder stress test: {ITERATIONS} iterations")
    print(f"Engine: {ENGINE}")
    print()

    proc = subprocess.Popen(
        [str(ENGINE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def wait_for(prefix, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                return None
            line = proc.stdout.readline().strip()
            if line.startswith(prefix):
                return line
        return None

    def check_alive():
        if proc.poll() is not None:
            print(f"\n  ENGINE CRASHED (exit code {proc.returncode})")
            return False
        return True

    send("uci")
    if wait_for("uciok") is None:
        print("ERROR: Engine didn't respond to uci")
        return 1

    send("setoption name UseHybridSearch value true")
    send(f"setoption name NNWeights value {WEIGHTS}")
    send("setoption name Threads value 4")
    send("setoption name Hash value 256")
    send("setoption name HybridMCTSThreads value 1")
    send("setoption name HybridABThreads value 3")
    send("setoption name Ponder value true")
    send("isready")
    if wait_for("readyok", timeout=120) is None:
        print("ERROR: Engine didn't become ready")
        return 1

    print("  Engine ready. Starting stress test...\n")
    crashes = 0

    for i in range(ITERATIONS):
        pos = POSITIONS[i % len(POSITIONS)]
        fen_cmd = f"position fen {pos}" if pos != "startpos" else "position startpos"

        # Pattern 1: go ponder → stop → go → bestmove
        send(fen_cmd)
        send("go ponder")
        time.sleep(0.05 + (i % 5) * 0.02)

        if not check_alive():
            crashes += 1
            break

        send("stop")
        result = wait_for("bestmove", timeout=10)
        if result is None:
            if not check_alive():
                crashes += 1
                break
            print(f"  [{i}] Timeout waiting for bestmove after ponder stop")
            continue

        # Pattern 2: immediate go (real search) after stop
        send(fen_cmd + " moves e2e4" if pos == "startpos" else fen_cmd)
        send("go movetime 100")

        if not check_alive():
            crashes += 1
            break

        result = wait_for("bestmove", timeout=15)
        if result is None:
            if not check_alive():
                crashes += 1
                break
            print(f"  [{i}] Timeout waiting for bestmove after real search")
            continue

        # Pattern 3: rapid stop during real search
        send(fen_cmd)
        send("go infinite")
        time.sleep(0.02)
        send("stop")

        if not check_alive():
            crashes += 1
            break

        result = wait_for("bestmove", timeout=10)
        if result is None:
            if not check_alive():
                crashes += 1
                break

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{ITERATIONS}] OK")

    if check_alive():
        send("quit")
        proc.wait(timeout=5)

    print()
    if crashes == 0:
        print(f"PASS: {ITERATIONS} ponder stress iterations without crash")
        return 0
    else:
        print(f"FAIL: Engine crashed after {crashes} failure(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
