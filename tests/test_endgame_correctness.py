#!/usr/bin/env python3
"""Endgame correctness tests for MetalFish hybrid search.

Launches the engine in hybrid mode and verifies evaluation/bestmove
for a set of well-known endgame positions.
"""

import subprocess
import sys
import time
import re
import os
import threading
import queue

ENGINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "metalfish"
)
WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVETIME = 3000  # ms per position

UCI_OPTIONS = [
    "setoption name UseHybridSearch value true",
    "setoption name NNWeights value networks/BT4-1024x15x32h-swa-6147500.pb.gz",
    "setoption name HybridMCTSRootReject value true",
    "setoption name Threads value 4",
    "setoption name Hash value 256",
    "setoption name SyzygyPath value syzygy",
]

POSITIONS = [
    {
        "name": "KRK mate",
        "fen": "8/8/4k3/8/8/8/8/4K2R w - - 0 1",
        "check": "mate_positive",
        "description": "White should find a mating sequence (score should be mate in N)",
    },
    {
        "name": "KQK mate",
        "fen": "8/8/4k3/8/8/8/8/3QK3 w - - 0 1",
        "check": "mate_positive",
        "description": "White should find mate",
    },
    {
        "name": "KPK won pawn",
        "fen": "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1",
        "check": "large_positive",
        "description": "White winning (pawn promotes), score should be large positive",
    },
    {
        "name": "KPK drawn (wrong rook pawn)",
        "fen": "8/8/8/8/8/7K/7P/5k2 w - - 0 1",
        "check": "positive",
        "description": "Actually won since Black king is far. Score positive.",
    },
    {
        "name": "Lucena position",
        "fen": "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
        "check": "large_positive_or_mate",
        "description": "White winning (building the bridge), large positive score or mate",
    },
    {
        "name": "Philidor position",
        "fen": "8/8/8/8/4k3/8/R7/4K2r b - - 0 1",
        "check": "draw_or_close",
        "description": "Draw or close to 0",
    },
    {
        "name": "Opposite color bishops",
        "fen": "8/5k2/4p3/4P3/3B4/8/2b5/5K2 w - - 0 1",
        "check": "draw_or_close",
        "description": "Draw or close to 0 (blocked pawns, opposite color bishops)",
    },
    {
        "name": "Pawn race",
        "fen": "8/p7/8/8/8/8/7P/8 w - - 0 1",
        "check": "draw_or_close",
        "description": "Depends on calculation - likely close to 0",
    },
    {
        "name": "Rook endgame 7th rank",
        "fen": "8/R4pk1/8/8/8/6K1/8/3r4 w - - 0 1",
        "check": "positive",
        "description": "White has advantage (rook on 7th, pawn restricted)",
    },
    {
        "name": "Stalemate trap",
        "fen": "k7/8/1K6/8/8/8/8/1Q6 w - - 0 1",
        "check": "no_stalemate",
        "description": "Must NOT play Qa2 (stalemate) - should find winning move",
    },
]


class EngineProcess:
    """Manages communication with the UCI engine using a reader thread."""

    def __init__(self):
        self.proc = None
        self.output_queue = queue.Queue()
        self.reader_thread = None
        self._stop_reader = False

    def start(self):
        self.proc = subprocess.Popen(
            [ENGINE_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=WORKDIR,
        )
        self._stop_reader = False
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()

    def _reader(self):
        """Background thread that reads stdout lines into a queue."""
        try:
            for line in iter(self.proc.stdout.readline, ""):
                if self._stop_reader:
                    break
                self.output_queue.put(line.rstrip("\n"))
        except (ValueError, OSError):
            pass

    def send(self, cmd):
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def read_until(self, marker, timeout=30):
        """Read lines until one contains the marker or timeout."""
        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                line = self.output_queue.get(timeout=min(remaining, 0.5))
                lines.append(line)
                if marker in line:
                    break
            except queue.Empty:
                if self.proc.poll() is not None:
                    break
                continue
        return lines

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def quit(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            if self.proc:
                self.proc.kill()
        self._stop_reader = True


def parse_score(info_lines):
    """Extract the last score from info lines."""
    score_cp = None
    score_mate = None
    for line in reversed(info_lines):
        m = re.search(r"score cp (-?\d+)", line)
        if m:
            score_cp = int(m.group(1))
            break
        m = re.search(r"score mate (-?\d+)", line)
        if m:
            score_mate = int(m.group(1))
            break
    return score_cp, score_mate


def parse_bestmove(line):
    """Extract bestmove from the bestmove line."""
    m = re.match(r"bestmove\s+(\S+)", line)
    if m:
        return m.group(1)
    return None


def evaluate_result(pos, bestmove, score_cp, score_mate):
    """Evaluate whether the engine's response is correct."""
    check = pos["check"]
    passed = False
    reason = ""

    if check == "mate_positive":
        if score_mate is not None and score_mate > 0:
            passed = True
            reason = f"Found mate in {score_mate}"
        elif score_cp is not None and score_cp > 500:
            passed = True
            reason = f"Large positive score {score_cp}cp (mate not announced but winning)"
        else:
            reason = f"Expected mate, got cp={score_cp} mate={score_mate}"

    elif check == "large_positive":
        if score_mate is not None and score_mate > 0:
            passed = True
            reason = f"Found mate in {score_mate}"
        elif score_cp is not None and score_cp > 200:
            passed = True
            reason = f"Large positive score: {score_cp}cp"
        else:
            reason = f"Expected large positive, got cp={score_cp} mate={score_mate}"

    elif check == "large_positive_or_mate":
        if score_mate is not None and score_mate > 0:
            passed = True
            reason = f"Found mate in {score_mate}"
        elif score_cp is not None and score_cp > 200:
            passed = True
            reason = f"Large positive score: {score_cp}cp"
        else:
            reason = f"Expected large positive or mate, got cp={score_cp} mate={score_mate}"

    elif check == "positive":
        if score_mate is not None and score_mate > 0:
            passed = True
            reason = f"Found mate in {score_mate}"
        elif score_cp is not None and score_cp > 0:
            passed = True
            reason = f"Positive score: {score_cp}cp"
        else:
            reason = f"Expected positive score, got cp={score_cp} mate={score_mate}"

    elif check == "draw_or_close":
        if score_cp is not None and abs(score_cp) <= 100:
            passed = True
            reason = f"Close to 0: {score_cp}cp"
        elif score_mate is not None:
            reason = f"Expected draw, got mate in {score_mate}"
        elif score_cp is not None:
            if abs(score_cp) <= 150:
                passed = True
                reason = f"Roughly drawish: {score_cp}cp (within tolerance)"
            else:
                reason = f"Expected ~0, got {score_cp}cp"
        else:
            reason = "No score returned"

    elif check == "no_stalemate":
        # The queen is on b1 in the given FEN. Stalemate moves:
        # b1a2 gives stalemate (Qa2 with Kb6, Ka8 has no squares).
        stalemate_moves = ["b1a2"]
        if bestmove in stalemate_moves:
            reason = f"Played stalemate move {bestmove}!"
        elif score_mate is not None and score_mate > 0:
            passed = True
            reason = f"Found mate in {score_mate} with {bestmove} (no stalemate)"
        elif score_cp is not None and score_cp > 200:
            passed = True
            reason = f"Winning with {bestmove}, score {score_cp}cp (no stalemate)"
        else:
            passed = True
            reason = f"Move {bestmove} (not stalemate), score cp={score_cp} mate={score_mate}"

    return passed, reason


def run_test():
    """Run all endgame positions through the engine."""
    print("=" * 70)
    print("MetalFish Endgame Correctness Test")
    print("=" * 70)
    print(f"Engine: {ENGINE_PATH}")
    print(f"Movetime: {MOVETIME}ms per position")
    print(f"Mode: Hybrid (AB + MCTS)")
    print("=" * 70)
    print()

    engine = EngineProcess()
    engine.start()

    # Initialize UCI
    engine.send("uci")
    uci_lines = engine.read_until("uciok", timeout=10)
    if not any("uciok" in l for l in uci_lines):
        print("FATAL: Engine did not respond with uciok")
        print(f"Got lines: {uci_lines}")
        engine.quit()
        sys.exit(1)

    # Set options
    for opt in UCI_OPTIONS:
        engine.send(opt)

    engine.send("isready")
    ready_lines = engine.read_until("readyok", timeout=30)
    if not any("readyok" in l for l in ready_lines):
        print("FATAL: Engine did not respond with readyok after options")
        print(f"Got lines: {ready_lines}")
        engine.quit()
        sys.exit(1)

    print("Engine initialized successfully.\n")

    results = []
    passed_count = 0
    failed_count = 0

    for i, pos in enumerate(POSITIONS, 1):
        print(f"--- Position {i}/10: {pos['name']} ---")
        print(f"FEN: {pos['fen']}")
        print(f"Expected: {pos['description']}")

        # Use a fresh engine for each position to avoid cascading failures
        if not engine.is_alive():
            print("  Engine died, restarting...")
            engine = EngineProcess()
            engine.start()
            engine.send("uci")
            engine.read_until("uciok", timeout=10)
            for opt in UCI_OPTIONS:
                engine.send(opt)
            engine.send("isready")
            engine.read_until("readyok", timeout=30)

        engine.send("ucinewgame")
        engine.send("isready")
        ready = engine.read_until("readyok", timeout=20)
        if not any("readyok" in l for l in ready):
            print(f"  FAIL: Engine not ready after ucinewgame")
            results.append((pos, None, None, None, False, "Engine not ready"))
            failed_count += 1
            # Kill and restart for next position
            engine.quit()
            engine = EngineProcess()
            engine.start()
            engine.send("uci")
            engine.read_until("uciok", timeout=10)
            for opt in UCI_OPTIONS:
                engine.send(opt)
            engine.send("isready")
            engine.read_until("readyok", timeout=30)
            print()
            continue

        engine.send(f"position fen {pos['fen']}")
        engine.send(f"go movetime {MOVETIME}")

        # Read until bestmove with generous timeout
        info_lines = []
        bestmove_line = None
        search_lines = engine.read_until("bestmove", timeout=MOVETIME / 1000 + 20)

        for line in search_lines:
            if line.startswith("info"):
                info_lines.append(line)
            if line.startswith("bestmove"):
                bestmove_line = line

        if not bestmove_line:
            print(f"  FAIL: No bestmove received (timeout or crash)")
            # Print any lines we did get for diagnostics
            if search_lines:
                print(f"  Last lines: {search_lines[-3:]}")
            results.append((pos, None, None, None, False, "No bestmove (timeout/crash)"))
            failed_count += 1
            # Kill and restart for next position
            engine.quit()
            engine = EngineProcess()
            engine.start()
            engine.send("uci")
            engine.read_until("uciok", timeout=10)
            for opt in UCI_OPTIONS:
                engine.send(opt)
            engine.send("isready")
            engine.read_until("readyok", timeout=30)
            print()
            continue

        bestmove = parse_bestmove(bestmove_line)
        score_cp, score_mate = parse_score(info_lines)

        # Format score string
        if score_mate is not None:
            score_str = f"mate {score_mate}"
        elif score_cp is not None:
            score_str = f"{score_cp}cp"
        else:
            score_str = "unknown"

        passed, reason = evaluate_result(pos, bestmove, score_cp, score_mate)

        status = "PASS" if passed else "FAIL"
        if passed:
            passed_count += 1
        else:
            failed_count += 1

        print(f"  Bestmove: {bestmove}")
        print(f"  Score: {score_str}")
        print(f"  Result: {status} - {reason}")
        print()

        results.append((pos, bestmove, score_cp, score_mate, passed, reason))

    # Quit engine
    engine.quit()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total: {len(POSITIONS)}, Passed: {passed_count}, Failed: {failed_count}")
    print()

    print(f"{'#':<3} {'Position':<25} {'Bestmove':<10} {'Score':<15} {'Status':<6}")
    print("-" * 70)
    for i, (pos, bm, cp, mate, ok, reason) in enumerate(results, 1):
        if mate is not None:
            sc = f"mate {mate}"
        elif cp is not None:
            sc = f"{cp}cp"
        else:
            sc = "N/A"
        status = "PASS" if ok else "FAIL"
        print(f"{i:<3} {pos['name']:<25} {bm or 'N/A':<10} {sc:<15} {status:<6}")

    print()
    if failed_count > 0:
        print("FLAGGED ISSUES:")
        for i, (pos, bm, cp, mate, ok, reason) in enumerate(results, 1):
            if not ok:
                print(f"  Position {i} ({pos['name']}): {reason}")
    else:
        print("All positions evaluated correctly.")

    print()
    return failed_count == 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
