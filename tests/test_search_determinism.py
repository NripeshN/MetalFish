#!/usr/bin/env python3
"""Search determinism test for MetalFish hybrid engine.

Runs 5 positions 8 times each with 'go movetime 5000' and checks
whether the engine produces consistent bestmove choices across runs.
Each run launches a fresh engine process for true independence.
"""

import os
import subprocess
import sys
import time
from collections import Counter

ENGINE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "metalfish"
)
NETWORK = "networks/BT4-1024x15x32h-swa-6147500.pb.gz"
RUNS = 8
MOVETIME = 5000
STABILITY_THRESHOLD = 0.60  # >60% same move = stable

POSITIONS = [
    ("Starting position", "position startpos"),
    (
        "Rb4 blunder position",
        "position fen 2kr4/1q1r1p2/4p3/p3P3/2Pn1PP1/2R5/1Q3B1P/5BK1 w - - 4 34",
    ),
    (
        "Sicilian Najdorf",
        "position fen rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    ),
    (
        "Sharp tactical",
        "position fen r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
    ),
    ("Quiet endgame", "position fen 8/pp3pkp/2p3p1/8/4P3/2P3PP/PP3PK1/8 w - - 0 1"),
]

UCI_OPTIONS = [
    "setoption name UseHybridSearch value true",
    f"setoption name NNWeights value {NETWORK}",
    "setoption name HybridMCTSRootReject value true",
    "setoption name Threads value 4",
    "setoption name Hash value 256",
]


def run_engine_once(position_cmd):
    """Launch engine, set options, run search, return bestmove string."""
    proc = subprocess.Popen(
        [ENGINE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def wait_for(token, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            if token in line:
                return line.strip()
        raise TimeoutError(f"Timed out waiting for '{token}'")

    try:
        # UCI handshake
        send("uci")
        wait_for("uciok")

        # Set options
        for opt in UCI_OPTIONS:
            send(opt)

        send("isready")
        wait_for("readyok")

        # New game + ready
        send("ucinewgame")
        send("isready")
        wait_for("readyok")

        # Position and go
        send(position_cmd)
        send(f"go movetime {MOVETIME}")

        # Wait for bestmove (generous timeout: movetime + 30s buffer)
        deadline = time.time() + MOVETIME / 1000 + 30
        bestmove = None
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith("bestmove"):
                parts = line.strip().split()
                bestmove = parts[1] if len(parts) >= 2 else None
                break

        return bestmove

    finally:
        # Kill engine process for true independence
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def main():
    print("=" * 70)
    print("MetalFish Search Determinism Test")
    print(f"Runs per position: {RUNS}, movetime: {MOVETIME}ms")
    print(f"Stability threshold: >{STABILITY_THRESHOLD*100:.0f}% same move")
    print("=" * 70)

    all_stable = True

    for pos_name, pos_cmd in POSITIONS:
        print(f"\n{'─' * 70}")
        print(f"Position: {pos_name}")
        print(f"Command:  {pos_cmd}")
        print(f"{'─' * 70}")

        moves = []
        for i in range(RUNS):
            print(f"  Run {i+1}/{RUNS}...", end=" ", flush=True)
            try:
                move = run_engine_once(pos_cmd)
                moves.append(move)
                print(f"bestmove: {move}")
            except Exception as e:
                print(f"ERROR: {e}")
                moves.append(None)

        # Analyze results
        valid_moves = [m for m in moves if m is not None]
        counter = Counter(valid_moves)

        print(f"\n  Results ({len(valid_moves)}/{RUNS} valid runs):")
        for move, count in counter.most_common():
            pct = count / len(valid_moves) * 100 if valid_moves else 0
            print(f"    {move}: {count}/{len(valid_moves)} ({pct:.1f}%)")

        if valid_moves:
            most_common_move, most_common_count = counter.most_common(1)[0]
            stability_pct = most_common_count / len(valid_moves)
            stable = stability_pct > STABILITY_THRESHOLD

            print(
                f"\n  Most common move: {most_common_move} ({stability_pct*100:.1f}%)"
            )
            print(
                f"  Stable: {'YES' if stable else 'NO'} (threshold: >{STABILITY_THRESHOLD*100:.0f}%)"
            )

            if not stable:
                all_stable = False

            # Special check for Rb4 blunder position
            if pos_name == "Rb4 blunder position":
                blunder_count = counter.get("b2b4", 0)
                if blunder_count > 0:
                    print(
                        f"\n  WARNING: Engine chose the blunder b2b4 {blunder_count}/{len(valid_moves)} times!"
                    )
                else:
                    print(f"\n  GOOD: Engine never chose the blunder move b2b4")
        else:
            print("  ERROR: No valid runs completed")
            all_stable = False

    print(f"\n{'=' * 70}")
    print(
        f"OVERALL: {'ALL POSITIONS STABLE' if all_stable else 'SOME POSITIONS UNSTABLE'}"
    )
    print(f"{'=' * 70}")

    return 0 if all_stable else 1


if __name__ == "__main__":
    sys.exit(main())
