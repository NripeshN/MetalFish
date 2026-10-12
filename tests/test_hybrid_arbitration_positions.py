#!/usr/bin/env python3
"""
Test hybrid arbitration on tricky positions where AB and MCTS might disagree.

Positions selected for tactical sharpness or where positional vs tactical
assessments diverge: zugzwang, fortress, exchange sac, quiet middlegame,
pawn endgame technique, king safety, opposite-side castling, and trapped
piece detection.

Usage:
    python3 tests/test_hybrid_arbitration_positions.py
"""

from __future__ import annotations

import pathlib
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PROJ = pathlib.Path(__file__).resolve().parent.parent
ENGINE = PROJ / "build" / "metalfish"
NETWORK = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb.gz"

MOVETIME_MS = 8000
SEARCH_TIMEOUT = 90.0  # seconds beyond movetime to wait

# --- Test positions ---
# (name, fen, description, acceptable_moves or None, score_range or None)
POSITIONS: List[
    Tuple[str, str, str, Optional[List[str]], Optional[Tuple[int, int]]]
] = [
    (
        "zugzwang",
        "8/8/p1p5/1p5p/1P5k/8/PP3KPP/8 w - - 0 1",
        "White should NOT rush; best is Kf1/Ke3/Kg1 or similar waiting move",
        None,  # many reasonable waiting moves
        None,
    ),
    (
        "fortress_draw",
        "8/8/8/p7/Pk6/1P6/1K6/8 w - - 0 1",
        "Should be ~0cp draw (fortress)",
        None,
        (-50, 50),  # score should be near 0
    ),
    (
        "exchange_sacrifice",
        "r1b1k2r/2qnbppp/p2ppn2/1p4B1/3NPP2/2N2Q2/PPP3PP/2KR1B1R w kq - 0 1",
        "Sharp; should find Bxf6 or similar tactical blow",
        ["g5f6", "d4b5", "f3h3"],  # Bxf6, Nb5, Qh3 are all thematic
        None,
    ),
    (
        "quiet_qgd",
        "r1bq1rk1/ppp2ppp/2n1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQ - 0 1",
        "Standard QGD; should play normal developing moves, not blunder",
        None,
        (-50, 600),  # White has strong central play; cxd5 wins a pawn
    ),
    (
        "pawn_endgame",
        "8/8/1p6/p1p5/P1P3k1/1P6/5K2/8 w - - 0 1",
        "Pawn endgame requiring precision",
        None,
        None,
    ),
    (
        "king_safety_italian",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "Italian Game; should castle (e1g1) or play d3",
        ["e1g1", "d2d3", "c2c3"],  # O-O, d3, c3
        (-30, 50),
    ),
    (
        "opposite_side_castling",
        "r1b1k2r/pp1pqppp/2n2n2/2p1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 1",
        "Should play h3 or Be3 to address bishop pin",
        ["h2h3", "c1e3", "e1g1", "b1d2", "c1g5"],  # h3, Be3, O-O, Nd2, Bg5
        (-30, 200),
    ),
    (
        "trapped_piece_detection",
        "rnb1kbnr/ppppqppp/8/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "Should NOT fall for traps; safe developing moves",
        None,
        (-30, 200),  # d4 is strong, engine may see significant advantage
    ),
    (
        "lichess_hc4_queen_blunder",
        "2kr4/1q1r1p2/4p3/p3P3/2Pn1PP1/2R5/1Q3B1P/5BK1 w - - 4 34",
        "Historical hC4lRCc0 regression: avoid MCTS override Qb6??",
        ["b2a3"],
        (-100, 100),
    ),
]


@dataclass
class PositionResult:
    name: str
    fen: str
    description: str
    bestmove: str = ""
    score_cp: Optional[int] = None
    score_mate: Optional[int] = None
    hybrid_reason: str = ""
    ab_move: str = ""
    mcts_move: str = ""
    ab_score: Optional[int] = None
    mcts_q: Optional[float] = None
    mcts_playouts: int = 0
    info_lines: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    error: str = ""


_READ_TIMEOUT = object()


class UCIEngine:
    """Manages a persistent UCI engine process with threaded stdout reading."""

    def __init__(self, engine_path: pathlib.Path, cwd: pathlib.Path) -> None:
        self.engine_path = engine_path
        self.cwd = cwd
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self.output_log: List[str] = []
        self.proc = subprocess.Popen(
            [str(engine_path)],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(
            target=self._read_stdout, daemon=True, name="uci-stdout"
        )
        self._reader.start()

    def _read_stdout(self) -> None:
        try:
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self._queue.put(line.rstrip("\n"))
        finally:
            self._queue.put(None)

    def send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read_line(self, timeout: float = 30.0):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _READ_TIMEOUT
            if self.proc.poll() is not None and self._queue.empty():
                return None
            try:
                line = self._queue.get(timeout=min(0.25, remaining))
                if line is not None:
                    self.output_log.append(line)
                return line
            except queue.Empty:
                continue

    def wait_for(self, prefix: str, timeout: float = 120.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"Engine died while waiting for '{prefix}' "
                    f"(exit={self.proc.returncode})"
                )
            line = self.read_line(timeout=deadline - time.monotonic())
            if line is None:
                raise RuntimeError(f"stdout closed waiting for '{prefix}'")
            if line is _READ_TIMEOUT:
                break
            if line.startswith(prefix):
                return line
        raise TimeoutError(
            f"Timeout waiting for '{prefix}'; last output: "
            + " | ".join(self.output_log[-8:])
        )

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def isready(self) -> None:
        self.send("isready")
        self.wait_for("readyok", 120.0)

    def quit(self) -> None:
        self.send("quit")
        self.proc.wait(timeout=10)


def parse_hybrid_trace(line: str) -> Dict[str, str]:
    """Parse a HybridTrace info string into key=value pairs."""
    result = {}
    # The format is: info string HybridTrace: key=value key=value ...
    trace_part = line.split("HybridTrace: ", 1)
    if len(trace_part) < 2:
        return result
    content = trace_part[1]
    # Parse key=value pairs (values may contain spaces in lists)
    for match in re.finditer(r"(\w+)=(\S+)", content):
        result[match.group(1)] = match.group(2)
    return result


def run_position(
    engine: UCIEngine, name: str, fen: str, movetime_ms: int
) -> PositionResult:
    """Run a single position search and collect results."""
    result = PositionResult(name=name, fen=fen, description="")

    engine.send("ucinewgame")
    engine.isready()
    engine.send(f"position fen {fen}")
    engine.output_log.clear()
    engine.send(f"go movetime {movetime_ms}")

    deadline = time.monotonic() + SEARCH_TIMEOUT + (movetime_ms / 1000.0)
    last_score_cp = None
    last_score_mate = None

    while time.monotonic() < deadline:
        line = engine.read_line(timeout=deadline - time.monotonic())
        if line is None:
            result.error = "Engine process died during search"
            return result
        if line is _READ_TIMEOUT:
            result.error = "Search timed out"
            return result
        if line == "":
            continue

        # Collect info lines
        if line.startswith("info "):
            result.info_lines.append(line)
            # Parse score from depth info lines
            if " score " in line and " depth " in line:
                cp_match = re.search(r" score cp (-?\d+)", line)
                mate_match = re.search(r" score mate (-?\d+)", line)
                if cp_match:
                    last_score_cp = int(cp_match.group(1))
                if mate_match:
                    last_score_mate = int(mate_match.group(1))

            # Parse HybridTrace
            if "HybridTrace:" in line:
                trace = parse_hybrid_trace(line)
                result.hybrid_reason = trace.get("reason", "")
                result.ab_move = trace.get("ABMove", "")
                result.mcts_move = trace.get("MCTSMove", "")
                ab_score_str = trace.get("ABScore", "")
                if ab_score_str:
                    try:
                        result.ab_score = int(ab_score_str)
                    except ValueError:
                        pass
                mcts_q_str = trace.get("MCTSQ", "")
                if mcts_q_str:
                    try:
                        result.mcts_q = float(mcts_q_str)
                    except ValueError:
                        pass
                playouts_str = trace.get("MCTSPlayouts", "")
                if playouts_str:
                    try:
                        result.mcts_playouts = int(playouts_str)
                    except ValueError:
                        pass

        elif line.startswith("bestmove"):
            parts = line.split()
            if len(parts) > 1:
                result.bestmove = parts[1]
            break

    result.score_cp = last_score_cp
    result.score_mate = last_score_mate
    return result


def classify_result(
    result: PositionResult,
    acceptable_moves: Optional[List[str]],
    score_range: Optional[Tuple[int, int]],
) -> None:
    """Flag suspicious results."""
    # Check if move is acceptable
    if acceptable_moves and result.bestmove:
        if result.bestmove not in acceptable_moves:
            result.flags.append(
                f"UNEXPECTED_MOVE: {result.bestmove} not in {acceptable_moves}"
            )

    # Check score range
    if score_range and result.score_cp is not None:
        lo, hi = score_range
        if result.score_cp < lo or result.score_cp > hi:
            result.flags.append(
                f"SCORE_OUT_OF_RANGE: {result.score_cp}cp not in [{lo}, {hi}]"
            )

    # Flag very extreme scores that suggest a blunder
    if result.score_cp is not None:
        if result.score_cp < -200:
            result.flags.append(f"VERY_LOW_SCORE: {result.score_cp}cp")
        elif result.score_cp > 500 and result.name not in (
            "exchange_sacrifice",
            "zugzwang",
            "pawn_endgame",
        ):
            result.flags.append(f"SUSPICIOUSLY_HIGH_SCORE: {result.score_cp}cp")

    # Flag if hybrid trace is missing
    if not result.hybrid_reason:
        result.flags.append("NO_HYBRID_TRACE")

    # Flag if AB and MCTS disagree but no override happened
    if (
        result.ab_move
        and result.mcts_move
        and result.ab_move != result.mcts_move
        and result.hybrid_reason == "engines_agree"
    ):
        result.flags.append(
            "ENGINES_AGREE_BUT_MOVES_DIFFER: "
            f"AB={result.ab_move} MCTS={result.mcts_move}"
        )


def print_result(result: PositionResult, idx: int) -> None:
    """Pretty-print a single position result."""
    print(f"\n{'='*70}")
    print(f"Position {idx}: {result.name}")
    print(f"  FEN: {result.fen}")
    print(f"  Description: {result.description}")
    print(f"  {'-'*60}")

    if result.error:
        print(f"  ERROR: {result.error}")
        return

    # Move and score
    score_str = ""
    if result.score_mate is not None:
        score_str = f"mate {result.score_mate}"
    elif result.score_cp is not None:
        score_str = f"{result.score_cp} cp"
    else:
        score_str = "unknown"

    print(f"  Bestmove: {result.bestmove}")
    print(f"  Score:    {score_str}")

    # Hybrid arbitration
    reason_display = result.hybrid_reason or "(not available)"
    print(f"  Hybrid reason: {reason_display}")
    if result.ab_move or result.mcts_move:
        print(f"  AB move:   {result.ab_move}")
        print(f"  MCTS move: {result.mcts_move}")
        agree = result.ab_move == result.mcts_move
        print(f"  Agreement: {'YES' if agree else 'NO -- engines disagree'}")
    if result.ab_score is not None:
        print(f"  AB score:  {result.ab_score} cp")
    if result.mcts_q is not None:
        print(f"  MCTS Q:    {result.mcts_q:.3f}")
    if result.mcts_playouts > 0:
        print(f"  MCTS playouts: {result.mcts_playouts}")

    # Flags
    if result.flags:
        print(f"  {'~'*60}")
        for flag in result.flags:
            print(f"  [FLAG] {flag}")


def main() -> int:
    if not ENGINE.exists():
        print(f"ERROR: Engine not found at {ENGINE}", file=sys.stderr)
        print("  Build with: cmake --build build --target metalfish", file=sys.stderr)
        return 1

    if not NETWORK.exists():
        print(f"ERROR: Network not found at {NETWORK}", file=sys.stderr)
        print(
            "  Download with: python3 tools/download_engine_networks.py --dest networks",
            file=sys.stderr,
        )
        return 1

    print("=" * 70)
    print("MetalFish Hybrid Arbitration Test")
    print(f"Engine:   {ENGINE}")
    print(f"Network:  {NETWORK}")
    print(f"Movetime: {MOVETIME_MS}ms")
    print("=" * 70)

    # Start engine
    print("\nStarting engine...")
    engine = UCIEngine(ENGINE, cwd=PROJ / "build")

    # UCI init
    engine.send("uci")
    engine.wait_for("uciok", 30.0)

    # Configure hybrid mode
    engine.setoption("UseHybridSearch", "true")
    engine.setoption("NNWeights", str(NETWORK))
    engine.setoption("Threads", "4")
    engine.setoption("Hash", "256")
    engine.setoption("HybridMCTSRootReject", "true")
    engine.setoption("HybridTrace", "true")
    engine.isready()
    print("Engine ready in hybrid mode.\n")

    # Run positions
    results: List[PositionResult] = []
    for idx, (name, fen, desc, acceptable, score_range) in enumerate(POSITIONS, 1):
        print(f"Searching position {idx}/{len(POSITIONS)}: {name} ...", flush=True)
        result = run_position(engine, name, fen, MOVETIME_MS)
        result.description = desc
        classify_result(result, acceptable, score_range)
        results.append(result)
        print_result(result, idx)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total = len(results)
    errors = sum(1 for r in results if r.error)
    flagged = sum(1 for r in results if r.flags)
    agree_count = sum(1 for r in results if r.hybrid_reason == "engines_agree")
    ab_default_count = sum(1 for r in results if r.hybrid_reason == "ab_default")
    mcts_override_count = sum(
        1
        for r in results
        if r.hybrid_reason and r.hybrid_reason not in ("engines_agree", "ab_default")
    )

    print(f"  Total positions:     {total}")
    print(f"  Errors:              {errors}")
    print(f"  Flagged:             {flagged}")
    print(f"  engines_agree:       {agree_count}")
    print(f"  ab_default:          {ab_default_count}")
    print(f"  mcts_override:       {mcts_override_count}")

    if flagged > 0:
        print(f"\n  Flagged positions:")
        for r in results:
            if r.flags:
                print(f"    - {r.name}: {'; '.join(r.flags)}")

    print()

    # Cleanup
    engine.quit()

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
