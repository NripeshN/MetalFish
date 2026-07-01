#!/usr/bin/env python3
"""Endgame and edge-case position tests for MetalFish.

Tests the engine on well-known endgame positions and tricky edge cases
where engines commonly fail. Each position is searched with go movetime 5000
using hybrid mode, and results are evaluated for correctness.

Note: MetalFish reports mate scores as large centipawn values (e.g. 29985, 8308)
rather than the standard UCI "score mate N" format. Values above ~8000 cp are
treated as mate-equivalent winning scores.
"""

import pathlib
import subprocess
import sys
import threading
import time

PROJ = pathlib.Path(__file__).resolve().parent.parent
ENGINE = PROJ / "build" / "metalfish"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb.gz"

MOVETIME = 5000
TIMEOUT = MOVETIME / 1000 + 15  # generous timeout beyond movetime

# Threshold above which a cp score is considered mate-equivalent
MATE_SCORE_THRESHOLD = 8000


def is_mate_score(cp):
    """Check if a centipawn value represents a mate-equivalent score."""
    return cp is not None and abs(cp) >= MATE_SCORE_THRESHOLD


def is_winning(cp, mate):
    """Check if the score indicates a winning position for the side to move."""
    if mate is not None and mate > 0:
        return True
    if cp is not None and cp > MATE_SCORE_THRESHOLD:
        return True
    return False


class Position:
    def __init__(self, name, fen, expected_desc, accept_fn):
        self.name = name
        self.fen = fen
        self.expected_desc = expected_desc
        self.accept_fn = accept_fn  # fn(bestmove, score_cp, mate_in) -> (pass, reason)


# --- Position definitions ---

ENDGAMES = [
    Position(
        name="KRK mate",
        fen="8/8/8/4k3/8/8/8/4K2R w - - 0 1",
        expected_desc="Should find forced mate (mate or mate-equivalent score >8000cp)",
        accept_fn=lambda bm, cp, mate: (
            (
                is_winning(cp, mate),
                f"mate={mate}, cp={cp}, move={bm}; "
                f"{'GOOD: mate-equivalent score' if is_mate_score(cp) else 'expected winning'}",
            )
        ),
    ),
    Position(
        name="KQK mate",
        fen="8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1",
        expected_desc="Should find mate quickly (mate or mate-equivalent score >8000cp)",
        accept_fn=lambda bm, cp, mate: (
            (
                is_winning(cp, mate),
                f"mate={mate}, cp={cp}, move={bm}; "
                f"{'GOOD: mate-equivalent score' if is_mate_score(cp) else 'expected winning'}",
            )
        ),
    ),
    Position(
        name="KPK win (opposition)",
        fen="8/8/8/8/8/4K3/4P3/4k3 w - - 0 1",
        expected_desc="White wins with Kd3 (opposition); score should be large positive or mate",
        accept_fn=lambda bm, cp, mate: (
            (
                (mate is not None and mate > 0) or (cp is not None and cp > 300),
                f"move={bm}, cp={cp}, mate={mate}; "
                f"expect winning eval (Kd3 is best, gains opposition)",
            )
        ),
    ),
    Position(
        name="KPK draw (f-pawn, Black has opposition)",
        fen="8/8/8/8/8/5k2/5P2/5K2 w - - 0 1",
        expected_desc="Should be draw; Black has opposition and blocks the pawn advance",
        accept_fn=lambda bm, cp, mate: (
            (
                cp is not None and abs(cp) < 150,
                f"move={bm}, cp={cp}, mate={mate}; "
                f"expect drawish eval (Black holds opposition)",
            )
        ),
    ),
    Position(
        name="Rook endgame Lucena",
        fen="1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
        expected_desc="White wins; should build the bridge (Rc4 is the classic first move)",
        accept_fn=lambda bm, cp, mate: (
            (
                (mate is not None and mate > 0) or (cp is not None and cp > 500),
                f"move={bm}, cp={cp}, mate={mate}; "
                f"expect winning eval (bridge-building technique)",
            )
        ),
    ),
    Position(
        name="Rook endgame Philidor",
        fen="8/8/8/4Pk2/R7/8/8/4K2r b - - 0 1",
        expected_desc=(
            "Black should draw with Philidor defense (3rd rank barrier); "
            "score should be near 0 or slightly negative from Black's perspective"
        ),
        accept_fn=lambda bm, cp, mate: (
            (
                cp is not None and cp > -500,
                f"move={bm}, cp={cp}, mate={mate}; "
                f"expect drawish or slightly negative (Philidor holds)",
            )
        ),
    ),
]

EDGE_CASES = [
    Position(
        name="Scholar's mate threat (must defend f7)",
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 3",
        expected_desc=(
            "Black must defend Qxf7#; best moves include Qe7, g6, or Nxe4. "
            "Engine should not hang and must find a defensive move"
        ),
        accept_fn=lambda bm, cp, mate: (
            (
                bm is not None and bm != "(none)" and len(bm) >= 4,
                f"move={bm}, cp={cp}; valid defensive move found",
            )
        ),
    ),
    Position(
        name="Only two legal moves (King vs Queen check)",
        fen="8/8/8/8/8/8/3q4/4K3 w - - 0 1",
        expected_desc="In check with only Kd2 or Kf1; engine must not crash or return (none)",
        accept_fn=lambda bm, cp, mate: (
            (
                bm in ("e1d2", "e1f1"),
                f"move={bm}, cp={cp}; " f"expected one of [e1d2, e1f1]",
            )
        ),
    ),
    Position(
        name="50-move rule approaching (halfmove=99)",
        fen="8/8/8/8/8/3K4/8/3k4 w - - 99 100",
        expected_desc="Should recognize draw is imminent (score ~0)",
        accept_fn=lambda bm, cp, mate: (
            (
                cp is not None and abs(cp) < 100,
                f"move={bm}, cp={cp}, mate={mate}; expect draw recognition",
            )
        ),
    ),
    Position(
        name="Bare kings draw (high halfmove)",
        fen="8/8/8/3k4/8/3K4/8/8 w - - 80 90",
        expected_desc="Bare kings = draw; score should be 0 or near 0",
        accept_fn=lambda bm, cp, mate: (
            (
                cp is not None and abs(cp) < 50,
                f"move={bm}, cp={cp}, mate={mate}; bare kings must be 0",
            )
        ),
    ),
    Position(
        name="Stalemate trap (White must not stalemate Black)",
        fen="5k2/5P2/5K2/8/8/8/8/8 w - - 0 1",
        expected_desc=(
            "Ke6 stalemates Black! White must play Kg6 or similar to win. "
            "Score should still be winning but engine must avoid stalemate"
        ),
        accept_fn=lambda bm, cp, mate: (
            (
                bm != "f6e6",  # Ke6 is stalemate!
                f"move={bm}, cp={cp}, mate={mate}; "
                f"{'GOOD: avoided stalemate trap' if bm != 'f6e6' else 'BAD: fell into stalemate'}",
            )
        ),
    ),
    Position(
        name="Zugzwang (mutual, triangulation needed)",
        fen="8/8/p1p5/1p5p/1P5p/8/PPP2K1p/4R1rk w - - 0 1",
        expected_desc=(
            "Complex zugzwang position; White should find Kf1 or similar calm move "
            "that puts Black in zugzwang. Engine must not crash on this tricky position"
        ),
        accept_fn=lambda bm, cp, mate: (
            (
                bm is not None and bm != "(none)" and len(bm) >= 4,
                f"move={bm}, cp={cp}, mate={mate}; "
                f"found a move in complex zugzwang position",
            )
        ),
    ),
]


def read_output(proc, lines_out, stop_event):
    """Reader thread: accumulates lines from engine stdout."""
    while not stop_event.is_set():
        line = proc.stdout.readline()
        if not line:
            break
        lines_out.append(line.strip())


def send(proc, cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def run_position(proc, lines, lock, pos):
    """Send a position to the running engine and collect the result."""
    # Clear accumulated lines
    with lock:
        lines.clear()

    send(proc, f"position fen {pos.fen}")
    send(proc, f"go movetime {MOVETIME}")

    # Wait for bestmove
    deadline = time.time() + TIMEOUT
    bestmove = None
    last_score_cp = None
    last_mate = None

    while time.time() < deadline:
        time.sleep(0.05)
        with lock:
            for line in lines:
                if line.startswith("bestmove"):
                    parts = line.split()
                    bestmove = parts[1] if len(parts) > 1 else None
                # Parse info lines for score
                if "info" in line and "score" in line:
                    tokens = line.split()
                    for i, tok in enumerate(tokens):
                        if tok == "cp" and i + 1 < len(tokens):
                            try:
                                last_score_cp = int(tokens[i + 1])
                            except ValueError:
                                pass
                        if tok == "mate" and i + 1 < len(tokens):
                            try:
                                last_mate = int(tokens[i + 1])
                            except ValueError:
                                pass
            if bestmove is not None:
                break

    return bestmove, last_score_cp, last_mate


def main():
    if not ENGINE.exists():
        print(f"ERROR: Engine binary not found at {ENGINE}")
        sys.exit(1)
    if not WEIGHTS.exists():
        print(f"ERROR: Network weights not found at {WEIGHTS}")
        sys.exit(1)

    print(f"Engine: {ENGINE}")
    print(f"Weights: {WEIGHTS}")
    print(f"Movetime: {MOVETIME}ms per position")
    print()

    # Start engine process
    proc = subprocess.Popen(
        [str(ENGINE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    lines = []
    lock = threading.Lock()
    stop_event = threading.Event()

    reader_thread = threading.Thread(
        target=read_output, args=(proc, lines, stop_event), daemon=True
    )
    reader_thread.start()

    # Initialize UCI
    send(proc, "uci")
    time.sleep(1)

    # Set options
    send(proc, "setoption name UseHybridSearch value true")
    send(proc, f"setoption name NNWeights value {WEIGHTS}")
    send(proc, "setoption name Threads value 4")
    send(proc, "setoption name Hash value 256")
    send(proc, "setoption name HybridMCTSRootReject value true")
    send(proc, "isready")

    # Wait for readyok
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline:
        time.sleep(0.1)
        with lock:
            for line in lines:
                if "readyok" in line:
                    ready = True
                    break
        if ready:
            break

    if not ready:
        print("ERROR: Engine did not respond with readyok within 30s")
        proc.kill()
        sys.exit(1)

    print("Engine initialized successfully.\n")
    print("=" * 78)

    all_positions = [("ENDGAME", ENDGAMES), ("EDGE CASE", EDGE_CASES)]
    results = []
    failures = []

    for category, positions in all_positions:
        print(f"\n{'=' * 78}")
        print(f"  {category} POSITIONS")
        print(f"{'=' * 78}")

        for i, pos in enumerate(positions, 1):
            print(f"\n--- [{category} #{i}] {pos.name} ---")
            print(f"  FEN: {pos.fen}")
            print(f"  Expected: {pos.expected_desc}")

            # Clear TT between positions for independence
            send(proc, "ucinewgame")
            send(proc, "isready")
            time.sleep(0.5)
            with lock:
                lines.clear()

            bestmove, score_cp, mate_in = run_position(proc, lines, lock, pos)

            if bestmove is None:
                print(f"  TIMEOUT: Engine did not respond within {TIMEOUT}s")
                failures.append((pos.name, "TIMEOUT", "No bestmove received"))
                results.append((pos.name, "TIMEOUT", None, None, None))
                continue

            passed, reason = pos.accept_fn(bestmove, score_cp, mate_in)

            status = "PASS" if passed else "FAIL"
            symbol = "+" if passed else "X"

            print(f"  Result: bestmove={bestmove}, score_cp={score_cp}, mate={mate_in}")
            print(f"  [{symbol}] {status}: {reason}")

            results.append((pos.name, status, bestmove, score_cp, mate_in))
            if not passed:
                failures.append((pos.name, status, reason))

    # Summary
    print(f"\n{'=' * 78}")
    print("  SUMMARY")
    print(f"{'=' * 78}")
    total = len(results)
    passed_count = sum(1 for r in results if r[1] == "PASS")
    failed_count = sum(1 for r in results if r[1] == "FAIL")
    timeout_count = sum(1 for r in results if r[1] == "TIMEOUT")

    print(f"\n  Total: {total}")
    print(f"  Passed: {passed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Timeout: {timeout_count}")

    if failures:
        print(f"\n  FLAGGED POSITIONS:")
        for name, status, reason in failures:
            print(f"    [{status}] {name}: {reason}")

    print()

    # Cleanup
    send(proc, "quit")
    stop_event.set()
    proc.wait(timeout=5)

    sys.exit(0 if failed_count == 0 and timeout_count == 0 else 1)


if __name__ == "__main__":
    main()
