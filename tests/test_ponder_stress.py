#!/usr/bin/env python3
"""Ponder lifecycle stress test for MetalFish Hybrid.

Tests the full ponderhit pathway repeatedly:
  1. go ponder (with clock info)
  2. ponderhit (opponent played predicted move)
  3. bestmove received
  4. Follow-up go movetime (verify engine is still responsive)

Also tests the stop-during-ponder path and rapid cycling.
Fails on: crash, hang (>15s), or invalid follow-up search.
"""

import argparse
import json
import os
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
ANE_WEIGHTS = PROJ / "networks" / "t1-512x15x8h-distilled-swa-3395000.pb.gz"
ANE_MODEL = PROJ / "build" / "coreml" / "compiled" / "t1-512-heads-b8.mlmodelc"

ITERATIONS = 20
TIMEOUT_BESTMOVE = 15.0
TIMEOUT_FOLLOWUP = 10.0

POSITIONS = [
    ("startpos", "e2e4 e7e5"),
    ("startpos", "d2d4 d7d5"),
    ("startpos", "e2e4 c7c5"),
    ("rnbqkb1r/pp2pppp/2p2n2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4", "e4e5 f6d7"),
    (
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "d2d3 d7d6",
    ),
]

PONDER_LEGALITY_REGRESSIONS = [
    (
        "ZcBZ0qcT-ply30",
        "r2q2k1/ppp2ppp/3n1b2/3p1b2/3P1B2/2P1Q3/PP1N1PPP/R4BK1 w - - 4 16",
    ),
    (
        "ZcBZ0qcT-ply32",
        "r2q2k1/p1p2ppp/1p1n1b2/3p1b2/3P1B2/1NP1Q3/PP3PPP/R4BK1 w - - 0 17",
    ),
    (
        "OD8KhLoq-ply18",
        "r1bqkb1r/3n1pp1/p2p1n2/1p1Np2p/4P3/6PP/PPP1NP2/R1BQKB1R w KQkq - 2 10",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            f"Number of core ponder cycles to run "
            f"(default: {ITERATIONS}; smoke default: 1)"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a fast core UCI ponder/stop regression check and skip "
            "extended probes"
        ),
    )
    parser.add_argument(
        "--stderr-log",
        type=pathlib.Path,
        help="Write engine stderr to this path instead of a temporary file",
    )
    parser.add_argument(
        "--engine",
        type=pathlib.Path,
        default=ENGINE,
        help=f"Engine binary to test (default: {ENGINE})",
    )
    parser.add_argument(
        "--keep-stderr",
        action="store_true",
        help="Keep the temporary stderr log after the test finishes",
    )
    parser.add_argument(
        "--transformer-min-move-budget-ms",
        type=int,
        help="Override TransformerMinMoveBudgetMs for stress reproduction",
    )
    parser.add_argument(
        "--ane-ponder-smoke",
        action="store_true",
        help=(
            "Also require the real Core ML/ANE root probe to complete during "
            "speculative ponder and be reused on ponderhit."
        ),
    )
    parser.add_argument(
        "--ane-weights",
        type=pathlib.Path,
        default=ANE_WEIGHTS,
        help=f"ANE T1 weights for --ane-ponder-smoke (default: {ANE_WEIGHTS})",
    )
    parser.add_argument(
        "--ane-model",
        type=pathlib.Path,
        default=ANE_MODEL,
        help=f"Compiled Core ML model for --ane-ponder-smoke (default: {ANE_MODEL})",
    )
    return parser.parse_args()


def tail_file(path: pathlib.Path, lines: int = 40) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"<unable to read stderr log {path}: {exc}>"
    if not content:
        return "<stderr was empty>"
    return "\n".join(content[-lines:])


def open_stderr_log(args):
    if args.stderr_log:
        args.stderr_log.parent.mkdir(parents=True, exist_ok=True)
        return args.stderr_log.open("w"), args.stderr_log, False

    fd, raw_path = tempfile.mkstemp(prefix="metalfish_ponder_", suffix=".stderr")
    os.close(fd)
    path = pathlib.Path(raw_path)
    return path.open("w"), path, not args.keep_stderr


def main():
    args = parse_args()
    default_iterations = 1 if args.smoke else ITERATIONS
    iterations = max(1, args.iterations or default_iterations)
    run_extended_probes = not args.smoke

    engine_path = args.engine

    if not engine_path.exists():
        print(f"ERROR: Engine not found at {engine_path}")
        return 1
    if not WEIGHTS.exists():
        print(f"ERROR: Weights not found at {WEIGHTS}")
        return 1
    if args.ane_ponder_smoke:
        if not args.ane_weights.exists():
            print(f"ERROR: ANE weights not found at {args.ane_weights}")
            return 1
        if not args.ane_model.exists():
            print(f"ERROR: ANE Core ML model not found at {args.ane_model}")
            return 1

    stderr_file, stderr_path, remove_stderr = open_stderr_log(args)

    proc = subprocess.Popen(
        [str(engine_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
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
        line, _ = read_until_collect(prefix, timeout)
        return line

    def read_until_collect(prefix, timeout=TIMEOUT_BESTMOVE):
        deadline = time.time() + timeout
        seen = []
        while time.time() < deadline:
            if proc.poll() is not None and stdout_lines.empty():
                return None, seen
            remaining = max(0.0, min(0.1, deadline - time.time()))
            try:
                line = stdout_lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                return None, seen
            seen.append(line)
            if line.startswith(prefix):
                return line, seen
        return "TIMEOUT", seen

    def mcts_playouts_from_info(lines):
        for line in reversed(lines):
            marker = "MCTSPlayouts="
            if marker not in line:
                continue
            tail = line.split(marker, 1)[1]
            token = tail.split()[0]
            try:
                return int(token)
            except ValueError:
                return None
        return None

    def read_bestmove_for(duration):
        deadline = time.time() + duration
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
            if line.startswith("bestmove"):
                return line
        return None

    def bestmove_ponder_legal(fen, line):
        parts = line.split()
        if len(parts) < 2:
            return False, "missing bestmove token"

        best = parts[1]
        try:
            board = chess.Board(fen)
            best_move = chess.Move.from_uci(best)
        except ValueError as exc:
            return False, str(exc)

        if best_move not in board.legal_moves:
            return False, f"bestmove {best} is illegal"

        ponder = parts[3] if len(parts) > 3 and parts[2] == "ponder" else None
        if not ponder:
            return True, ""

        board.push(best_move)
        try:
            ponder_move = chess.Move.from_uci(ponder)
        except ValueError as exc:
            return False, str(exc)
        if ponder_move not in board.legal_moves:
            return False, f"ponder {ponder} is illegal after {best}"
        return True, ""

    def alive():
        if proc.poll() is not None:
            return False
        return True

    def close_failed_startup():
        if alive():
            proc.kill()
            proc.wait(timeout=5)
        stderr_file.close()
        print()
        print("Engine stderr tail:")
        print(tail_file(stderr_path))
        if remove_stderr:
            try:
                stderr_path.unlink()
            except OSError:
                pass

    send("uci")
    r = read_until("uciok", 30)
    if not r or r == "TIMEOUT":
        print("ERROR: uciok timeout")
        close_failed_startup()
        return 1

    send("setoption name UseMCTS value false")
    send("setoption name UseHybridSearch value true")
    send(f"setoption name NNWeights value {WEIGHTS}")
    send("setoption name Threads value 4")
    send("setoption name MultiPV value 1")
    send("setoption name HybridMCTSThreads value 1")
    send("setoption name HybridABThreads value 3")
    send("setoption name MCTSAddDirichletNoise value false")
    send("setoption name HybridABPolicyWeight value 0.0")
    hybrid_trace = "true" if args.ane_ponder_smoke else "false"
    send(f"setoption name HybridTrace value {hybrid_trace}")
    if args.ane_ponder_smoke:
        send("setoption name HybridANERootProbe value true")
        send(f"setoption name HybridANEWeights value {args.ane_weights}")
        send(f"setoption name HybridANEModelPath value {args.ane_model}")
        send("setoption name HybridANEComputeUnits value cpu-ne")
        send("setoption name HybridANERootHintWaitMs value 1000")
        send("setoption name HybridANEMinBudgetMs value 1000")
    if args.transformer_min_move_budget_ms is not None:
        budget = max(0, min(5000, args.transformer_min_move_budget_ms))
        send(f"setoption name TransformerMinMoveBudgetMs value {budget}")
    send("setoption name Ponder value true")
    send("setoption name Hash value 256")
    send("isready")
    r = read_until("readyok", 120)
    if not r or r == "TIMEOUT":
        print("ERROR: readyok timeout")
        close_failed_startup()
        return 1

    mode = "smoke" if args.smoke else "full"
    print(f"Ponder stress test: {iterations} iterations ({mode})")
    print(f"Engine: {engine_path.name}")
    print()

    stats = {
        "ponderhit_ok": 0,
        "stop_ok": 0,
        "followup_ok": 0,
        "pure_mcts_ponder_ok": 0,
        "early_bestmove": 0,
        "ponder_mcts_active": 0,
        "ponder_legality_ok": 0,
        "low_clock_ponder_mcts_ok": 0,
        "nbcejrue_regression_ok": 0,
        "filqkzru_regression_ok": 0,
        "ane_ponder_probe_ok": 0,
        "ane_ponder_reuse_ok": 0,
        "crashes": 0,
        "timeouts": 0,
        "total": 0,
    }

    if args.ane_ponder_smoke:
        send("position startpos moves " "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6")
        send("go ponder wtime 60000 btime 60000 winc 1000 binc 1000")
        r = read_until("info string Hybrid: ANE root probe completed", 10)
        if not alive():
            print(
                "  [ANE ponder] CRASH during speculative probe "
                f"(exit {proc.returncode})"
            )
            stats["crashes"] += 1
        elif r is None or r == "TIMEOUT":
            print("  [ANE ponder] ANE root probe did not complete before ponderhit")
            stats["timeouts"] += 1
            send("stop")
            read_until("bestmove", 5)
        else:
            stats["ane_ponder_probe_ok"] = 1
            send("ponderhit")
            r = read_until(
                "info string Hybrid: ANE root probe skipped: already ready", 5
            )
            if not alive():
                print(
                    "  [ANE ponder] CRASH after ponderhit " f"(exit {proc.returncode})"
                )
                stats["crashes"] += 1
            elif r is None or r == "TIMEOUT":
                print("  [ANE ponder] ANE root probe was not reused on ponderhit")
                stats["timeouts"] += 1
                send("stop")
                read_until("bestmove", 5)
            else:
                stats["ane_ponder_reuse_ok"] = 1
                r = read_until("bestmove", TIMEOUT_BESTMOVE)
                if r is None or r == "TIMEOUT":
                    print("  [ANE ponder] TIMEOUT waiting for bestmove")
                    stats["timeouts"] += 1
                    send("stop")
                    read_until("bestmove", 5)

    if stats["crashes"] or stats["timeouts"]:
        close_failed_startup()
        return 1

    print("Ponder legality regressions:")
    for label, fen in PONDER_LEGALITY_REGRESSIONS:
        send(f"position fen {fen}")
        send("go ponder movetime 1000")
        time.sleep(0.25)

        early = read_bestmove_for(0.05)
        if early:
            print(f"  {label}: EARLY bestmove before ponderhit: {early}")
            stats["early_bestmove"] += 1
            break

        send("ponderhit")
        r = read_until("bestmove", TIMEOUT_BESTMOVE)
        if not alive():
            print(f"  {label}: CRASH after ponderhit (exit {proc.returncode})")
            stats["crashes"] += 1
            break
        if r is None or r == "TIMEOUT":
            print(f"  {label}: TIMEOUT after ponderhit")
            stats["timeouts"] += 1
            send("stop")
            read_until("bestmove", 5)
            continue

        ok, reason = bestmove_ponder_legal(fen, r)
        if not ok:
            print(f"  {label}: ILLEGAL {reason}: {r}")
            stats["timeouts"] += 1
            break

        stats["ponder_legality_ok"] += 1
        print(f"  {label}: OK {r}")

    if (
        stats["ponder_legality_ok"] != len(PONDER_LEGALITY_REGRESSIONS)
        or stats["crashes"]
        or stats["timeouts"]
        or stats["early_bestmove"]
    ):
        close_failed_startup()
        return 1

    for i in range(iterations):
        pos_fen, moves = POSITIONS[i % len(POSITIONS)]
        pos_cmd = (
            f"position fen {pos_fen} moves {moves}"
            if pos_fen != "startpos"
            else f"position startpos moves {moves}"
        )

        stats["total"] += 1

        send(pos_cmd)
        send("go ponder wtime 60000 btime 60000 winc 1000 binc 1000")
        time.sleep(0.3 + (i % 3) * 0.1)

        if not alive():
            print(f"  [{i}] CRASH during ponder (exit {proc.returncode})")
            stats["crashes"] += 1
            break

        early = read_bestmove_for(0.05)
        if early:
            print(f"  [{i}] EARLY bestmove before ponderhit: {early}")
            stats["early_bestmove"] += 1
            break

        send("ponderhit")
        r = read_until("bestmove", TIMEOUT_BESTMOVE)
        if not alive():
            print(f"  [{i}] CRASH after ponderhit (exit {proc.returncode})")
            stats["crashes"] += 1
            break
        if r is None or r == "TIMEOUT":
            print(f"  [{i}] TIMEOUT after ponderhit")
            stats["timeouts"] += 1
            # Try to recover
            send("stop")
            read_until("bestmove", 5)
            continue
        stats["ponderhit_ok"] += 1

        send(pos_cmd)
        send("go movetime 200")
        r = read_until("bestmove", TIMEOUT_FOLLOWUP)
        if not alive():
            print(f"  [{i}] CRASH during follow-up (exit {proc.returncode})")
            stats["crashes"] += 1
            break
        if r is None or r == "TIMEOUT":
            print(f"  [{i}] TIMEOUT during follow-up")
            stats["timeouts"] += 1
            send("stop")
            read_until("bestmove", 5)
            continue
        stats["followup_ok"] += 1

        send(pos_cmd)
        send("go ponder wtime 60000 btime 60000")
        time.sleep(0.6)
        send("stop")
        r, stop_lines = read_until_collect("bestmove", TIMEOUT_BESTMOVE)
        if not alive():
            print(f"  [{i}] CRASH after stop-ponder (exit {proc.returncode})")
            stats["crashes"] += 1
            break
        if r is None or r == "TIMEOUT":
            print(f"  [{i}] TIMEOUT after stop-ponder")
            stats["timeouts"] += 1
            continue
        mcts_playouts = mcts_playouts_from_info(stop_lines)
        if not mcts_playouts or mcts_playouts <= 0:
            print(f"  [{i}] No MCTS playouts during stop-ponder")
            stats["timeouts"] += 1
            continue
        stats["ponder_mcts_active"] += 1
        stats["stop_ok"] += 1

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{iterations}] OK")

    if alive() and stats["crashes"] == 0 and stats["timeouts"] == 0:
        send("position startpos moves d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2")
        send("go ponder wtime 305930 btime 293540 winc 3000 binc 3000")
        send("ponderhit")
        r, regression_lines = read_until_collect("bestmove", TIMEOUT_BESTMOVE)
        if not alive():
            print(f"  [FILqKzru] CRASH after book ponderhit (exit {proc.returncode})")
            stats["crashes"] += 1
        elif r is None or r == "TIMEOUT":
            print("  [FILqKzru] TIMEOUT after book ponderhit")
            stats["timeouts"] += 1
            send("stop")
            read_until("bestmove", 5)
        else:
            parts = r.split()
            best = parts[1] if len(parts) > 1 else "0000"
            if best == "0000":
                print("  [FILqKzru] invalid bestmove after book ponderhit")
                stats["timeouts"] += 1
            else:
                stats["filqkzru_regression_ok"] = 1

    if (
        run_extended_probes
        and alive()
        and stats["crashes"] == 0
        and stats["timeouts"] == 0
    ):
        send("position startpos")
        send("go ponder wtime 100 btime 100 winc 0 binc 0")
        time.sleep(0.6)
        send("stop")
        r, low_clock_lines = read_until_collect("bestmove", TIMEOUT_BESTMOVE)
        if r is None or r == "TIMEOUT":
            print("  [low-clock ponder] TIMEOUT after stop")
            stats["timeouts"] += 1
        else:
            mcts_playouts = mcts_playouts_from_info(low_clock_lines)
            if not mcts_playouts or mcts_playouts <= 0:
                print("  [low-clock ponder] no MCTS playouts")
                stats["timeouts"] += 1
            else:
                stats["low_clock_ponder_mcts_ok"] = 1

    if (
        run_extended_probes
        and alive()
        and stats["crashes"] == 0
        and stats["timeouts"] == 0
    ):
        fen = "3r1k2/1pr1n3/p3R1p1/N2n1p1p/8/1BP5/PP4PP/1K2R3 b - - 10 29"
        send(f"position fen {fen}")
        send("go ponder wtime 60000 btime 60000 winc 1000 binc 1000")
        time.sleep(1.0)
        early = read_bestmove_for(0.05)
        if early:
            print(f"  [NBCejRUE] EARLY bestmove before ponderhit: {early}")
            stats["early_bestmove"] += 1
        else:
            send("ponderhit")
            r, regression_lines = read_until_collect("bestmove", TIMEOUT_BESTMOVE)
            if r is None or r == "TIMEOUT":
                print("  [NBCejRUE] TIMEOUT after ponderhit")
                stats["timeouts"] += 1
                send("stop")
                read_until("bestmove", 5)
            else:
                parts = r.split()
                best = parts[1] if len(parts) > 1 else "0000"
                mcts_playouts = mcts_playouts_from_info(regression_lines)
                if best == "d5f4":
                    print("  [NBCejRUE] regression: rejected blunder returned")
                    stats["timeouts"] += 1
                elif not mcts_playouts or mcts_playouts <= 0:
                    print("  [NBCejRUE] no MCTS playouts during regression")
                    stats["timeouts"] += 1
                else:
                    stats["nbcejrue_regression_ok"] = 1

    if (
        run_extended_probes
        and alive()
        and stats["crashes"] == 0
        and stats["timeouts"] == 0
    ):
        send("setoption name UseHybridSearch value false")
        send("setoption name UseMCTS value true")
        send("setoption name MCTSMaxThreads value 1")
        send("setoption name MCTSMinibatchSize value 0")
        send("ucinewgame")
        send("isready")
        r = read_until("readyok", 120)
        if r is None or r == "TIMEOUT":
            print("  [pure-mcts] TIMEOUT waiting for readyok")
            stats["timeouts"] += 1
        else:
            send("position startpos")
            send("go ponder wtime 60000 btime 60000 winc 1000 binc 1000")
            early = read_bestmove_for(7.0)
            if early:
                print(f"  [pure-mcts] EARLY bestmove before ponderhit: {early}")
                stats["timeouts"] += 1
            else:
                send("ponderhit")
                r = read_until("bestmove", TIMEOUT_BESTMOVE)
                if r is None or r == "TIMEOUT":
                    print("  [pure-mcts] TIMEOUT after ponderhit")
                    stats["timeouts"] += 1
                    send("stop")
                    read_until("bestmove", 5)
                else:
                    stats["pure_mcts_ponder_ok"] = 1
                    print("  [pure-mcts] OK")

    if alive():
        send("quit")
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    stderr_file.close()

    print()
    print(f"Results: {json.dumps(stats, indent=2)}")
    print()

    core_passed = (
        stats["crashes"] == 0
        and stats["timeouts"] == 0
        and stats["ponderhit_ok"] == iterations
        and stats["followup_ok"] == iterations
        and stats["stop_ok"] == iterations
        and stats["early_bestmove"] == 0
        and stats["ponder_mcts_active"] == iterations
    )
    extended_passed = args.smoke or (
        stats["pure_mcts_ponder_ok"] == 1
        and stats["low_clock_ponder_mcts_ok"] == 1
        and stats["nbcejrue_regression_ok"] == 1
    )
    regression_passed = stats["filqkzru_regression_ok"] == 1
    ane_passed = not args.ane_ponder_smoke or (
        stats["ane_ponder_probe_ok"] == 1 and stats["ane_ponder_reuse_ok"] == 1
    )
    passed = core_passed and extended_passed and regression_passed and ane_passed

    if not passed or args.keep_stderr or args.stderr_log:
        print(f"Engine stderr log: {stderr_path}")
    if not passed:
        print()
        print("Engine stderr tail:")
        print(tail_file(stderr_path))

    if remove_stderr:
        try:
            stderr_path.unlink()
        except OSError:
            pass

    if passed:
        if args.smoke:
            print(f"PASS: {iterations} smoke ponder cycle(s) without issues")
        else:
            print(f"PASS: {iterations} full ponder cycles without issues")
        return 0
    elif stats["crashes"] > 0:
        print(f"FAIL: {stats['crashes']} crash(es)")
        return 1
    elif stats["timeouts"] > 0:
        print(f"PARTIAL: {stats['timeouts']} timeout(s) but no crashes")
        print("Ponderhit works but may be too slow for some patterns")
        return 2
    else:
        print(f"FAIL: unexpected state")
        return 1


if __name__ == "__main__":
    sys.exit(main())
