#!/usr/bin/env python3
"""
Test MCTS vs AB vs Hybrid parity on 10 critical positions.

Launches the engine fresh for each mode/position combination,
captures bestmove and score, then flags disagreements.
"""

import subprocess
import sys
import os
import re
import time

ENGINE_PATH = os.path.join(os.path.dirname(__file__), "..", "build", "metalfish")
ENGINE_PATH = os.path.abspath(ENGINE_PATH)
NN_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "networks", "BT4-1024x15x32h-swa-6147500.pb.gz")
NN_WEIGHTS = os.path.abspath(NN_WEIGHTS)

MOVETIME = 5000

POSITIONS = [
    ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("French Advance", "r1bqkbnr/pppppppp/2n5/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"),
    ("Italian Game", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("d4 systems", "rnbqkb1r/pp2pppp/5n2/2pp4/3P4/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 0 4"),
    ("QGD setup", "r1bq1rk1/ppp1bppp/2n2n2/3pp3/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 8"),
    ("Blunder trap (avoid Rb4)", "2kr4/1q1r1p2/4p3/p3P3/2Pn1PP1/2R5/1Q3B1P/5BK1 w - - 4 34"),
    ("Middlegame", "r2qk2r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("KPK endgame (must win)", "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"),
    ("KRK endgame (must mate)", "8/5k2/8/8/8/2K5/8/4R3 w - - 0 1"),
    ("French Defense", "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"),
]

MODES = {
    "AB": [
        "setoption name UseHybridSearch value false",
        "setoption name UseMCTS value false",
    ],
    "MCTS": [
        "setoption name UseMCTS value true",
        "setoption name UseHybridSearch value false",
        f"setoption name NNWeights value {NN_WEIGHTS}",
    ],
    "Hybrid": [
        "setoption name UseHybridSearch value true",
        "setoption name HybridMCTSRootReject value true",
        f"setoption name NNWeights value {NN_WEIGHTS}",
    ],
}


def send(proc, cmd):
    """Send a command to the engine."""
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def read_until(proc, target, timeout=60):
    """Read engine output until a line starts with target or timeout."""
    lines = []
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        lines.append(line)
        if line.startswith(target):
            return lines
    return lines


def extract_score(info_lines):
    """Extract the last reported score from info lines."""
    score_cp = None
    score_mate = None
    for line in info_lines:
        # Look for "info depth ... score cp X" or "score mate X"
        m = re.search(r"\bscore cp (-?\d+)", line)
        if m:
            score_cp = int(m.group(1))
            score_mate = None
        m = re.search(r"\bscore mate (-?\d+)", line)
        if m:
            score_mate = int(m.group(1))
            score_cp = None
    if score_mate is not None:
        return f"mate {score_mate}"
    if score_cp is not None:
        return f"{score_cp} cp"
    return "N/A"


def run_search(fen, mode_name):
    """Launch engine, configure mode, run search, return (bestmove, score, raw_cp)."""
    proc = subprocess.Popen(
        [ENGINE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # Initialize UCI
        send(proc, "uci")
        read_until(proc, "uciok", timeout=10)

        # Set mode options
        for opt in MODES[mode_name]:
            send(proc, opt)

        # Sync
        send(proc, "isready")
        read_until(proc, "readyok", timeout=10)

        # New game
        send(proc, "ucinewgame")
        send(proc, "isready")
        read_until(proc, "readyok", timeout=10)

        # Set position
        send(proc, f"position fen {fen}")

        # Go
        send(proc, f"go movetime {MOVETIME}")

        # Read until bestmove
        lines = read_until(proc, "bestmove", timeout=MOVETIME // 1000 + 30)

        # Extract bestmove
        bestmove = "N/A"
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    bestmove = parts[1]
                break

        # Extract score from info lines
        score = extract_score(lines)

        # Extract raw cp for comparison
        raw_cp = None
        for line in reversed(lines):
            m = re.search(r"\bscore cp (-?\d+)", line)
            if m:
                raw_cp = int(m.group(1))
                break
            m = re.search(r"\bscore mate (-?\d+)", line)
            if m:
                raw_cp = 30000 * (1 if int(m.group(1)) > 0 else -1)
                break

        return bestmove, score, raw_cp

    finally:
        try:
            send(proc, "quit")
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait()


def main():
    print("=" * 80)
    print("METALFISH MODE PARITY TEST")
    print(f"Engine: {ENGINE_PATH}")
    print(f"NN Weights: {NN_WEIGHTS}")
    print(f"Movetime: {MOVETIME}ms per position per mode")
    print("=" * 80)
    print()

    results = []  # (pos_name, fen, {mode: (bestmove, score, raw_cp)})
    disagreements = []

    for pos_idx, (pos_name, fen) in enumerate(POSITIONS):
        print(f"[{pos_idx + 1}/{len(POSITIONS)}] {pos_name}")
        print(f"  FEN: {fen}")

        pos_results = {}
        for mode_name in ["AB", "MCTS", "Hybrid"]:
            sys.stdout.write(f"  {mode_name:8s}: searching... ")
            sys.stdout.flush()
            try:
                bestmove, score, raw_cp = run_search(fen, mode_name)
                pos_results[mode_name] = (bestmove, score, raw_cp)
                print(f"bestmove={bestmove:8s}  score={score}")
            except Exception as e:
                pos_results[mode_name] = ("ERROR", str(e), None)
                print(f"ERROR: {e}")

        results.append((pos_name, fen, pos_results))

        # Check for disagreements
        ab = pos_results.get("AB", ("N/A", "N/A", None))
        mcts = pos_results.get("MCTS", ("N/A", "N/A", None))
        hybrid = pos_results.get("Hybrid", ("N/A", "N/A", None))

        if ab[0] != mcts[0]:
            gap = None
            if ab[2] is not None and mcts[2] is not None:
                gap = abs(ab[2] - mcts[2])
            if gap is not None and gap > 100:
                msg = (f"  ** DISAGREEMENT: AB={ab[0]} ({ab[1]}) vs "
                       f"MCTS={mcts[0]} ({mcts[1]}), gap={gap}cp")
                print(msg)
                disagreements.append((pos_name, "AB vs MCTS", ab[0], mcts[0], gap))
            elif ab[0] != mcts[0]:
                print(f"  (minor disagreement: AB={ab[0]} vs MCTS={mcts[0]}, "
                      f"gap={'N/A' if gap is None else f'{gap}cp'})")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Position':<30s} {'AB':<10s} {'MCTS':<10s} {'Hybrid':<10s} {'AB Score':<12s} {'MCTS Score':<12s} {'Hybrid Score':<12s}")
    print("-" * 96)
    for pos_name, fen, pos_results in results:
        ab = pos_results.get("AB", ("N/A", "N/A", None))
        mcts = pos_results.get("MCTS", ("N/A", "N/A", None))
        hybrid = pos_results.get("Hybrid", ("N/A", "N/A", None))
        print(f"{pos_name:<30s} {ab[0]:<10s} {mcts[0]:<10s} {hybrid[0]:<10s} {ab[1]:<12s} {mcts[1]:<12s} {hybrid[1]:<12s}")

    print()
    if disagreements:
        print(f"FLAGGED DISAGREEMENTS (>100cp gap between AB and MCTS): {len(disagreements)}")
        for pos_name, comparison, move1, move2, gap in disagreements:
            print(f"  - {pos_name}: {comparison} -> {move1} vs {move2} (gap={gap}cp)")
    else:
        print("No significant disagreements (>100cp) between AB and MCTS.")

    print()
    print("Test complete.")
    return 0 if not disagreements else 1


if __name__ == "__main__":
    sys.exit(main())
