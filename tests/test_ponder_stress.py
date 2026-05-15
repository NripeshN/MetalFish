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
import subprocess
import sys
import tempfile
import time

PROJ = pathlib.Path(__file__).resolve().parent.parent
ENGINE = PROJ / "build" / "metalfish"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=ITERATIONS,
        help=f"Number of full ponder cycles to run (default: {ITERATIONS})",
    )
    parser.add_argument(
        "--stderr-log",
        type=pathlib.Path,
        help="Write engine stderr to this path instead of a temporary file",
    )
    parser.add_argument(
        "--keep-stderr",
        action="store_true",
        help="Keep the temporary stderr log after the test finishes",
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
    iterations = max(1, args.iterations)

    if not ENGINE.exists():
        print(f"ERROR: Engine not found at {ENGINE}")
        return 1
    if not WEIGHTS.exists():
        print(f"ERROR: Weights not found at {WEIGHTS}")
        return 1

    stderr_file, stderr_path, remove_stderr = open_stderr_log(args)

    proc = subprocess.Popen(
        [str(ENGINE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        text=True,
        bufsize=1,
    )

    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def read_until(prefix, timeout=TIMEOUT_BESTMOVE):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                return None
            line = proc.stdout.readline().strip()
            if line.startswith(prefix):
                return line
        return "TIMEOUT"

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

    # Init
    send("uci")
    r = read_until("uciok", 30)
    if not r or r == "TIMEOUT":
        print("ERROR: uciok timeout")
        close_failed_startup()
        return 1

    send("setoption name UseHybridSearch value true")
    send(f"setoption name NNWeights value {WEIGHTS}")
    send("setoption name Threads value 4")
    send("setoption name HybridMCTSThreads value 1")
    send("setoption name HybridABThreads value 3")
    send("setoption name Ponder value true")
    send("setoption name Hash value 256")
    send("isready")
    r = read_until("readyok", 120)
    if not r or r == "TIMEOUT":
        print("ERROR: readyok timeout")
        close_failed_startup()
        return 1

    print(f"Ponder stress test: {iterations} iterations")
    print(f"Engine: {ENGINE.name}")
    print()

    stats = {
        "ponderhit_ok": 0,
        "stop_ok": 0,
        "followup_ok": 0,
        "crashes": 0,
        "timeouts": 0,
        "total": 0,
    }

    for i in range(iterations):
        pos_fen, moves = POSITIONS[i % len(POSITIONS)]
        pos_cmd = (
            f"position fen {pos_fen} moves {moves}"
            if pos_fen != "startpos"
            else f"position startpos moves {moves}"
        )

        stats["total"] += 1

        # === Pattern A: go ponder -> ponderhit -> bestmove ===
        send(pos_cmd)
        send("go ponder wtime 60000 btime 60000 winc 1000 binc 1000")
        time.sleep(0.3 + (i % 3) * 0.1)

        if not alive():
            print(f"  [{i}] CRASH during ponder")
            stats["crashes"] += 1
            break

        send("ponderhit")
        r = read_until("bestmove", TIMEOUT_BESTMOVE)
        if not alive():
            print(f"  [{i}] CRASH after ponderhit")
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

        # === Pattern B: follow-up search must work ===
        send(pos_cmd)
        send("go movetime 200")
        r = read_until("bestmove", TIMEOUT_FOLLOWUP)
        if not alive():
            print(f"  [{i}] CRASH during follow-up")
            stats["crashes"] += 1
            break
        if r is None or r == "TIMEOUT":
            print(f"  [{i}] TIMEOUT during follow-up")
            stats["timeouts"] += 1
            send("stop")
            read_until("bestmove", 5)
            continue
        stats["followup_ok"] += 1

        # === Pattern C: go ponder -> stop (not ponderhit) -> bestmove ===
        send(pos_cmd)
        send("go ponder wtime 60000 btime 60000")
        time.sleep(0.2)
        send("stop")
        r = read_until("bestmove", TIMEOUT_BESTMOVE)
        if not alive():
            print(f"  [{i}] CRASH after stop-ponder")
            stats["crashes"] += 1
            break
        if r is None or r == "TIMEOUT":
            print(f"  [{i}] TIMEOUT after stop-ponder")
            stats["timeouts"] += 1
            continue
        stats["stop_ok"] += 1

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{iterations}] OK")

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

    passed = (
        stats["crashes"] == 0
        and stats["timeouts"] == 0
        and stats["ponderhit_ok"] == iterations
        and stats["followup_ok"] == iterations
        and stats["stop_ok"] == iterations
    )

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
