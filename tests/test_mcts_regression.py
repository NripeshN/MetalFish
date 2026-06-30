#!/usr/bin/env python3
"""
MCTS-specific regression test: verifies NN evaluation and policy correctness
by running the engine in MCTS-only, AB-only, and Hybrid modes on a set of
positions, then flags disagreements that could indicate blunder potential.

Usage:
    python3 tests/test_mcts_regression.py
    python3 tests/test_mcts_regression.py --movetime 3000
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
BUILD_DIR = PROJECT_ROOT / "build"
ENGINE_BIN = BUILD_DIR / "metalfish"
NN_WEIGHTS = PROJECT_ROOT / "networks" / "BT4-1024x15x32h-swa-6147500.pb.gz"

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Test positions
# ---------------------------------------------------------------------------
POSITIONS: List[Dict[str, str]] = [
    {
        "name": "Starting position",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "notes": "Should be roughly equal, sensible opening moves (e4, d4, Nf3, c4).",
    },
    {
        "name": "Ruy Lopez (after 1.e4 e5 2.Nf3 Nc6 3.Bb5)",
        "fen": "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "notes": "Well-known opening, eval ~equal, Black to move (a6 or Nf6 expected).",
    },
    {
        "name": "Sicilian Najdorf (after 5...a6)",
        "fen": "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
        "notes": "Sharp position, White slightly better. Bg5/Be2/Be3/f3 expected.",
    },
    {
        "name": "Winning for White (up a rook)",
        "fen": "r1bqkbnr/pppppppp/2n5/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # Actually use a clearly won position:
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 1",
        # Let's use a genuinely up-a-rook position:
    },
    {
        "name": "Losing for White (down a queen)",
        "fen": "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "notes": "White is missing the queen. Should show large negative eval.",
    },
    {
        "name": "KRK endgame (White winning)",
        "fen": "8/8/8/4k3/8/8/8/4K2R w - - 0 1",
        "notes": "King + Rook vs King. White should win; eval very positive.",
    },
    {
        "name": "Drawn: opposite-colored bishops",
        "fen": "8/5k2/4b3/8/8/4B3/5K2/8 w - - 0 1",
        "notes": "Drawn endgame. Eval should be near zero.",
    },
    {
        "name": "Tactical fork position",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "notes": "Scholar's mate threat (Qxf7#). White should find Qxf7#.",
    },
    {
        "name": "Quiet positional (Carlsbad structure)",
        "fen": "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2NBPN2/PPP2PPP/R1BQ1RK1 w - - 0 9",
        "notes": "Quiet middlegame. Positional moves expected, eval ~equal.",
    },
    {
        "name": "French Winawer (blunder-prone: before Qg4)",
        "fen": "rnbqk2r/pp3ppp/4p3/2ppP3/3P4/P1P5/2P2PPP/R1BQKBNR w KQkq - 0 7",
        "notes": "The specific position from the blunder game. Qg4 is strong here.",
    },
]

# Fix the winning/losing positions properly
POSITIONS[3] = {
    "name": "Winning for White (up a rook)",
    "fen": "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/R1BQKB1R w KQkq - 0 1",
    # Better: use a position where White has an extra rook cleanly
    "fen": "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
}

# Use cleaner FENs for the material-up/down cases:
POSITIONS[3] = {
    "name": "Winning for White (up a rook, clean)",
    "fen": "4k3/8/8/8/8/8/PPPPPPPP/R3K2R w KQ - 0 1",
    "notes": "White has king + 2 rooks + 8 pawns vs lone king. Massive advantage.",
}

POSITIONS[4] = {
    "name": "Losing for White (down a queen)",
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
    "notes": "White is missing the queen entirely. Eval should be very negative for White.",
}


# ---------------------------------------------------------------------------
# Engine interaction
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    bestmove: str = ""
    score_cp: Optional[int] = None
    score_mate: Optional[int] = None
    depth: int = 0
    nodes: int = 0
    pv: str = ""
    info_lines: List[str] = field(default_factory=list)
    raw_output: List[str] = field(default_factory=list)


def run_engine_search(
    fen: str,
    movetime: int,
    options: Dict[str, str],
    timeout: int = 60,
) -> SearchResult:
    """Run the engine with given options on a position and return results."""
    result = SearchResult()

    # Build UCI command sequence
    commands = ["uci"]
    # Wait for uciok implicitly by just sending all commands
    for name, value in options.items():
        commands.append(f"setoption name {name} value {value}")
    commands.append("isready")
    commands.append(f"position fen {fen}")
    commands.append(f"go movetime {movetime}")

    uci_input = "\n".join(commands) + "\n"

    try:
        proc = subprocess.Popen(
            [str(ENGINE_BIN)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(input=uci_input, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        result.bestmove = "TIMEOUT"
        return result
    except Exception as e:
        result.bestmove = f"ERROR: {e}"
        return result

    lines = stdout.strip().split("\n")
    result.raw_output = lines

    # Parse output
    last_info = ""
    for line in lines:
        line = line.strip()
        if line.startswith("info "):
            result.info_lines.append(line)
            last_info = line
        elif line.startswith("bestmove "):
            parts = line.split()
            result.bestmove = parts[1] if len(parts) > 1 else ""

    # Parse the last info line for score/depth/nodes
    if last_info:
        tokens = last_info.split()
        for i, tok in enumerate(tokens):
            if tok == "depth" and i + 1 < len(tokens):
                try:
                    result.depth = int(tokens[i + 1])
                except ValueError:
                    pass
            elif tok == "score" and i + 1 < len(tokens):
                if tokens[i + 1] == "cp" and i + 2 < len(tokens):
                    try:
                        result.score_cp = int(tokens[i + 2])
                    except ValueError:
                        pass
                elif tokens[i + 1] == "mate" and i + 2 < len(tokens):
                    try:
                        result.score_mate = int(tokens[i + 2])
                    except ValueError:
                        pass
            elif tok == "nodes" and i + 1 < len(tokens):
                try:
                    result.nodes = int(tokens[i + 1])
                except ValueError:
                    pass
            elif tok == "pv" and i + 1 < len(tokens):
                result.pv = " ".join(tokens[i + 1 :])

    return result


def score_string(r: SearchResult) -> str:
    """Human-readable score string."""
    if r.score_mate is not None:
        return f"mate {r.score_mate}"
    elif r.score_cp is not None:
        return f"{r.score_cp:+d}cp"
    else:
        return "N/A"


def score_numeric(r: SearchResult) -> Optional[int]:
    """Return numeric centipawn value (mate converted to large number)."""
    if r.score_mate is not None:
        # Mate in N -> +/- 30000 - N*100 so closer mates are bigger
        sign = 1 if r.score_mate > 0 else -1
        return sign * (30000 - abs(r.score_mate) * 100)
    return r.score_cp


# ---------------------------------------------------------------------------
# Main test logic
# ---------------------------------------------------------------------------
def run_all_tests(movetime: int = 5000, verbose: bool = False) -> bool:
    """Run the full regression suite. Returns True if all tests pass."""

    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}  MetalFish MCTS Regression Test{RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}")
    print(f"  Engine:    {ENGINE_BIN}")
    print(f"  NN:        {NN_WEIGHTS}")
    print(f"  Movetime:  {movetime}ms per position per mode")
    print(f"  Positions: {len(POSITIONS)}")
    print(f"{'=' * 72}\n")

    if not ENGINE_BIN.exists():
        print(f"{RED}ERROR: Engine binary not found at {ENGINE_BIN}{RESET}")
        print("Build with: cmake --build build --target metalfish")
        return False

    if not NN_WEIGHTS.exists():
        print(f"{RED}ERROR: NN weights not found at {NN_WEIGHTS}{RESET}")
        print("Download with: python3 tools/download_engine_networks.py --dest networks")
        return False

    # UCI option sets for each mode
    mcts_options = {
        "UseMCTS": "true",
        "UseHybridSearch": "false",
        "NNWeights": str(NN_WEIGHTS),
        "Threads": "1",
    }

    ab_options = {
        "UseMCTS": "false",
        "UseHybridSearch": "false",
        "Threads": "4",
        "Hash": "256",
    }

    hybrid_options = {
        "UseHybridSearch": "true",
        "NNWeights": str(NN_WEIGHTS),
        "Threads": "4",
        "Hash": "256",
        "HybridMCTSRootReject": "true",
    }

    # Increased timeout for MCTS (it can be slow on first init)
    timeout = movetime // 1000 + 30

    results: List[Dict] = []
    disagreements: List[Dict] = []
    failures: List[str] = []

    for idx, pos in enumerate(POSITIONS):
        name = pos["name"]
        fen = pos["fen"]
        notes = pos.get("notes", "")

        print(f"{CYAN}{BOLD}[{idx + 1}/{len(POSITIONS)}] {name}{RESET}")
        print(f"  FEN: {DIM}{fen}{RESET}")
        if notes:
            print(f"  Expected: {DIM}{notes}{RESET}")
        print()

        # --- MCTS-only ---
        print(f"  {BOLD}MCTS-only{RESET} ... ", end="", flush=True)
        t0 = time.time()
        mcts_result = run_engine_search(fen, movetime, mcts_options, timeout=timeout)
        mcts_time = time.time() - t0
        if mcts_result.bestmove and not mcts_result.bestmove.startswith("ERROR"):
            print(
                f"{GREEN}OK{RESET}  move={mcts_result.bestmove}  "
                f"score={score_string(mcts_result)}  "
                f"nodes={mcts_result.nodes}  "
                f"({mcts_time:.1f}s)"
            )
        else:
            print(f"{RED}FAIL{RESET}  {mcts_result.bestmove}")
            failures.append(f"MCTS on '{name}': {mcts_result.bestmove}")
            if verbose:
                for line in mcts_result.raw_output[-10:]:
                    print(f"    {DIM}{line}{RESET}")

        # --- AB-only ---
        print(f"  {BOLD}AB-only{RESET}   ... ", end="", flush=True)
        t0 = time.time()
        ab_result = run_engine_search(fen, movetime, ab_options, timeout=timeout)
        ab_time = time.time() - t0
        if ab_result.bestmove and not ab_result.bestmove.startswith("ERROR"):
            print(
                f"{GREEN}OK{RESET}  move={ab_result.bestmove}  "
                f"score={score_string(ab_result)}  "
                f"depth={ab_result.depth}  "
                f"({ab_time:.1f}s)"
            )
        else:
            print(f"{RED}FAIL{RESET}  {ab_result.bestmove}")
            failures.append(f"AB on '{name}': {ab_result.bestmove}")
            if verbose:
                for line in ab_result.raw_output[-10:]:
                    print(f"    {DIM}{line}{RESET}")

        # --- Hybrid ---
        print(f"  {BOLD}Hybrid{RESET}    ... ", end="", flush=True)
        t0 = time.time()
        hybrid_result = run_engine_search(fen, movetime, hybrid_options, timeout=timeout)
        hybrid_time = time.time() - t0
        if hybrid_result.bestmove and not hybrid_result.bestmove.startswith("ERROR"):
            # Try to detect which engine was chosen from info lines
            chooser_info = ""
            for info_line in hybrid_result.info_lines:
                if "hybrid" in info_line.lower() or "chosen" in info_line.lower():
                    chooser_info = info_line
            print(
                f"{GREEN}OK{RESET}  move={hybrid_result.bestmove}  "
                f"score={score_string(hybrid_result)}  "
                f"depth={hybrid_result.depth}  "
                f"({hybrid_time:.1f}s)"
            )
            if chooser_info:
                print(f"         chooser: {DIM}{chooser_info}{RESET}")
        else:
            print(f"{RED}FAIL{RESET}  {hybrid_result.bestmove}")
            failures.append(f"Hybrid on '{name}': {hybrid_result.bestmove}")
            if verbose:
                for line in hybrid_result.raw_output[-10:]:
                    print(f"    {DIM}{line}{RESET}")

        # --- Check disagreement ---
        mcts_score = score_numeric(mcts_result)
        ab_score = score_numeric(ab_result)

        disagreement = None
        if mcts_score is not None and ab_score is not None:
            diff = abs(mcts_score - ab_score)
            if diff > 200:
                disagreement = {
                    "position": name,
                    "fen": fen,
                    "mcts_move": mcts_result.bestmove,
                    "mcts_score": mcts_score,
                    "ab_move": ab_result.bestmove,
                    "ab_score": ab_score,
                    "diff": diff,
                }
                disagreements.append(disagreement)
                print(
                    f"  {YELLOW}{BOLD}WARNING: MCTS/AB disagree by {diff}cp "
                    f"(MCTS={score_string(mcts_result)}, AB={score_string(ab_result)}){RESET}"
                )

        results.append(
            {
                "name": name,
                "fen": fen,
                "mcts": mcts_result,
                "ab": ab_result,
                "hybrid": hybrid_result,
                "disagreement": disagreement,
            }
        )
        print()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{'=' * 72}\n")

    # Results table
    print(
        f"  {'Position':<40} {'MCTS':<12} {'AB':<12} {'Hybrid':<12} {'Diff':<8}"
    )
    print(f"  {'-' * 84}")
    for r in results:
        name = r["name"][:38]
        mcts_s = score_string(r["mcts"])
        ab_s = score_string(r["ab"])
        hyb_s = score_string(r["hybrid"])
        ms = score_numeric(r["mcts"])
        ab_n = score_numeric(r["ab"])
        diff_s = ""
        if ms is not None and ab_n is not None:
            d = abs(ms - ab_n)
            diff_s = f"{d}cp"
            if d > 200:
                diff_s = f"{YELLOW}{d}cp{RESET}"
        print(f"  {name:<40} {mcts_s:<12} {ab_s:<12} {hyb_s:<12} {diff_s:<8}")

    # Disagreements
    if disagreements:
        print(f"\n  {YELLOW}{BOLD}DISAGREEMENTS (>200cp difference):{RESET}")
        for d in disagreements:
            print(
                f"    - {d['position']}: MCTS={d['mcts_score']:+d}cp ({d['mcts_move']}) "
                f"vs AB={d['ab_score']:+d}cp ({d['ab_move']}) -- diff={d['diff']}cp"
            )
        print(
            f"\n  {YELLOW}These positions may have blunder potential in hybrid mode.{RESET}"
        )
    else:
        print(f"\n  {GREEN}No major MCTS/AB disagreements (all within 200cp).{RESET}")

    # Failures
    if failures:
        print(f"\n  {RED}{BOLD}FAILURES:{RESET}")
        for f in failures:
            print(f"    - {RED}{f}{RESET}")
        print(f"\n  {RED}Some engine modes failed to produce results.{RESET}")
    else:
        print(f"\n  {GREEN}All engine modes produced results successfully.{RESET}")

    # Sanity checks
    sanity_issues = []

    # Check 1: KRK endgame should be strongly positive for White
    krk = results[5]  # KRK endgame
    krk_mcts = score_numeric(krk["mcts"])
    krk_ab = score_numeric(krk["ab"])
    if krk_mcts is not None and krk_mcts < 200:
        sanity_issues.append(
            f"KRK endgame: MCTS eval too low ({krk_mcts}cp) -- expected >200cp"
        )
    if krk_ab is not None and krk_ab < 200:
        sanity_issues.append(
            f"KRK endgame: AB eval too low ({krk_ab}cp) -- expected >200cp"
        )

    # Check 2: Queen-down position should be negative
    q_down = results[4]  # Down a queen
    q_mcts = score_numeric(q_down["mcts"])
    q_ab = score_numeric(q_down["ab"])
    if q_ab is not None and q_ab > -200:
        sanity_issues.append(
            f"Queen-down: AB eval too high ({q_ab}cp) -- expected < -200cp"
        )

    # Check 3: Drawn position should be near zero for AB
    drawn = results[6]  # Opposite-colored bishops
    drawn_ab = score_numeric(drawn["ab"])
    if drawn_ab is not None and abs(drawn_ab) > 150:
        sanity_issues.append(
            f"Drawn position: AB eval too extreme ({drawn_ab}cp) -- expected near 0"
        )

    # Check 4: Scholar's mate position -- should find mate or winning move
    fork = results[7]  # Fork/mate threat
    fork_mcts = fork["mcts"]
    fork_ab = fork["ab"]
    if fork_ab.score_mate is None and (fork_ab.score_cp is not None and fork_ab.score_cp < 300):
        sanity_issues.append(
            f"Scholar's mate: AB didn't find mate or big advantage "
            f"(got {score_string(fork_ab)})"
        )

    if sanity_issues:
        print(f"\n  {YELLOW}{BOLD}SANITY CHECK WARNINGS:{RESET}")
        for issue in sanity_issues:
            print(f"    - {YELLOW}{issue}{RESET}")
    else:
        print(f"\n  {GREEN}All sanity checks passed.{RESET}")

    print(f"\n{'=' * 72}\n")

    # Return success only if no hard failures
    return len(failures) == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MetalFish MCTS regression test suite"
    )
    parser.add_argument(
        "--movetime",
        type=int,
        default=5000,
        help="Search time per position per mode in ms (default: 5000)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show raw engine output on failure"
    )
    args = parser.parse_args()

    success = run_all_tests(movetime=args.movetime, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
