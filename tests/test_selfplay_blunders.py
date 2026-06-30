#!/usr/bin/env python3
"""Self-play blunder detection test for MetalFish Hybrid.

Plays N games of the engine against itself in hybrid mode, then analyzes
every position for evaluation drops > 100cp between consecutive moves,
which indicate potential blunders.

Usage:
    python3 tests/test_selfplay_blunders.py
    python3 tests/test_selfplay_blunders.py --games 3 --movetime 3000
"""

import argparse
import pathlib
import queue
import subprocess
import sys
import threading
import time

import chess

PROJ = pathlib.Path(__file__).resolve().parent.parent
ENGINE = PROJ / "build" / "metalfish"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb.gz"

NUM_GAMES = 5
MOVETIME_MS = 5000
BLUNDER_THRESHOLD_CP = 100
MAX_MOVES_PER_GAME = 200


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=int, default=NUM_GAMES, help="Number of self-play games"
    )
    parser.add_argument(
        "--movetime", type=int, default=MOVETIME_MS, help="Movetime in ms per move"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=BLUNDER_THRESHOLD_CP,
        help="Eval drop threshold in centipawns to flag as blunder",
    )
    parser.add_argument(
        "--engine", type=pathlib.Path, default=ENGINE, help="Engine binary path"
    )
    return parser.parse_args()


class UCIEngine:
    """Manages a single UCI engine process with non-blocking stdout reads."""

    def __init__(self, engine_path, name, hash_mb=256):
        self.name = name
        self.engine_path = engine_path
        self.hash_mb = hash_mb
        self.proc = None
        self.stdout_queue = queue.Queue()
        self._reader_thread = None

    def start(self):
        self.proc = subprocess.Popen(
            [str(self.engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader_thread.start()

    def _pump_stdout(self):
        for line in self.proc.stdout:
            self.stdout_queue.put(line.strip())
        self.stdout_queue.put(None)

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix, timeout=30.0):
        """Read lines until one starts with prefix. Returns that line."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                remaining = max(0.1, deadline - time.time())
                line = self.stdout_queue.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"[{self.name}] Timed out waiting for '{prefix}' "
                    f"(timeout={timeout}s)"
                )
            if line is None:
                raise RuntimeError(f"[{self.name}] Engine process terminated")
            if line.startswith(prefix):
                return line
        raise TimeoutError(
            f"[{self.name}] Timed out waiting for '{prefix}' (timeout={timeout}s)"
        )

    def read_until_bestmove(self, timeout=30.0):
        """Read lines until bestmove, collecting info lines. Returns (bestmove, score)."""
        deadline = time.time() + timeout
        last_score_cp = None
        last_score_mate = None

        while time.time() < deadline:
            try:
                remaining = max(0.1, deadline - time.time())
                line = self.stdout_queue.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"[{self.name}] Timed out waiting for bestmove "
                    f"(timeout={timeout}s)"
                )
            if line is None:
                raise RuntimeError(f"[{self.name}] Engine process terminated")

            if line.startswith("info") and " score " in line:
                parts = line.split()
                if "score" in parts:
                    idx = parts.index("score")
                    if idx + 2 < len(parts):
                        if parts[idx + 1] == "cp":
                            try:
                                last_score_cp = int(parts[idx + 2])
                                last_score_mate = None
                            except ValueError:
                                pass
                        elif parts[idx + 1] == "mate":
                            try:
                                last_score_mate = int(parts[idx + 2])
                                last_score_cp = None
                            except ValueError:
                                pass

            if line.startswith("bestmove"):
                parts = line.split()
                move_str = parts[1] if len(parts) > 1 else None
                score = None
                if last_score_cp is not None:
                    score = last_score_cp
                elif last_score_mate is not None:
                    # Convert mate scores to large cp values
                    score = 10000 * (1 if last_score_mate > 0 else -1)
                return move_str, score

        raise TimeoutError(
            f"[{self.name}] Timed out waiting for bestmove (timeout={timeout}s)"
        )

    def init_uci(self):
        self.send("uci")
        self.read_until("uciok")

    def configure_hybrid(self):
        self.send("setoption name UseHybridSearch value true")
        self.send(f"setoption name NNWeights value {WEIGHTS}")
        self.send("setoption name Threads value 4")
        self.send(f"setoption name Hash value {self.hash_mb}")
        self.send("setoption name HybridMCTSRootReject value true")

    def new_game(self):
        self.send("ucinewgame")
        self.send("isready")
        self.read_until("readyok")

    def go_movetime(self, movetime_ms, timeout=None):
        if timeout is None:
            timeout = (movetime_ms / 1000.0) + 20.0
        self.send(f"go movetime {movetime_ms}")
        return self.read_until_bestmove(timeout=timeout)

    def set_position(self, moves):
        if moves:
            moves_str = " ".join(moves)
            self.send(f"position startpos moves {moves_str}")
        else:
            self.send("position startpos")

    def quit(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_game(engine_white, engine_black, movetime_ms, game_num):
    """Play one self-play game. Returns list of (fen, move_uci, score_before, score_after, side)."""
    board = chess.Board()
    moves = []
    move_records = []  # (fen, move_uci, score_from_white_pov)

    engine_white.new_game()
    engine_black.new_game()

    print(f"\n--- Game {game_num} ---")
    move_num = 0

    while not board.is_game_over() and move_num < MAX_MOVES_PER_GAME:
        fen_before = board.fen()
        is_white = board.turn == chess.WHITE
        engine = engine_white if is_white else engine_black

        engine.set_position(moves)

        timeout = (movetime_ms / 1000.0) + 20.0
        move_str, score = engine.go_movetime(movetime_ms, timeout=timeout)

        if move_str is None or move_str == "(none)":
            print(f"  Engine returned no move at ply {move_num}")
            break

        try:
            move = chess.Move.from_uci(move_str)
        except ValueError:
            print(f"  Invalid move string: {move_str}")
            break

        if move not in board.legal_moves:
            print(f"  Illegal move {move_str} in position {fen_before}")
            break

        # Normalize score to white's perspective
        score_white = score if is_white else (-score if score is not None else None)

        move_records.append(
            {
                "fen": fen_before,
                "move": move_str,
                "score_white_pov": score_white,
                "side": "white" if is_white else "black",
                "ply": move_num,
            }
        )

        board.push(move)
        moves.append(move_str)
        move_num += 1

        if move_num % 10 == 0:
            print(f"  Ply {move_num}, eval: {score_white} cp (white pov)")

    result = board.result() if board.is_game_over() else "1/2-1/2"
    termination = ""
    if board.is_checkmate():
        termination = "checkmate"
    elif board.is_stalemate():
        termination = "stalemate"
    elif board.is_insufficient_material():
        termination = "insufficient material"
    elif board.is_fifty_moves():
        termination = "50-move rule"
    elif board.is_repetition():
        termination = "repetition"
    elif move_num >= MAX_MOVES_PER_GAME:
        termination = "max moves reached"

    print(f"  Result: {result} ({termination}) after {move_num} plies")
    return move_records, result


def detect_blunders(move_records, threshold_cp):
    """Find evaluation drops exceeding threshold between consecutive moves.

    A blunder for white is when score_white drops by > threshold after white's move.
    A blunder for black is when score_white rises by > threshold after black's move
    (i.e., black's eval from their own perspective dropped).
    """
    blunders = []

    for i in range(1, len(move_records)):
        prev = move_records[i - 1]
        curr = move_records[i]

        if prev["score_white_pov"] is None or curr["score_white_pov"] is None:
            continue

        score_before = prev["score_white_pov"]
        score_after = curr["score_white_pov"]

        # The move at index i-1 is what caused the eval change seen at index i.
        # If white moved (prev record), and score dropped for white, white blundered.
        # If black moved (prev record), and score rose for white, black blundered.

        mover = prev["side"]
        if mover == "white":
            # White just moved; eval after should be similar or better for white.
            # A drop means white blundered.
            drop = score_before - score_after
            if drop > threshold_cp:
                blunders.append(
                    {
                        "fen": prev["fen"],
                        "move": prev["move"],
                        "side": "white",
                        "eval_before_cp": score_before,
                        "eval_after_cp": score_after,
                        "drop_cp": drop,
                        "ply": prev["ply"],
                    }
                )
        else:
            # Black just moved; from black's perspective, eval should stay or improve.
            # A rise in white's score means black's eval dropped.
            drop = score_after - score_before
            if drop > threshold_cp:
                blunders.append(
                    {
                        "fen": prev["fen"],
                        "move": prev["move"],
                        "side": "black",
                        "eval_before_cp": -score_before,  # from black's POV
                        "eval_after_cp": -score_after,  # from black's POV
                        "drop_cp": drop,
                        "ply": prev["ply"],
                    }
                )

    return blunders


def main():
    args = parse_args()

    if not args.engine.exists():
        print(f"ERROR: Engine not found at {args.engine}")
        return 1
    if not WEIGHTS.exists():
        print(f"ERROR: Weights not found at {WEIGHTS}")
        return 1

    print("=" * 70)
    print("MetalFish Self-Play Blunder Detection Test")
    print("=" * 70)
    print(f"  Games:     {args.games}")
    print(f"  Movetime:  {args.movetime} ms")
    print(f"  Threshold: {args.threshold} cp")
    print(f"  Engine:    {args.engine}")
    print(f"  Weights:   {WEIGHTS}")
    print("=" * 70)

    # Use separate hash sizes to avoid sharing transposition tables
    engine_white = UCIEngine(args.engine, "White", hash_mb=256)
    engine_black = UCIEngine(args.engine, "Black", hash_mb=256)

    engine_white.start()
    engine_black.start()

    engine_white.init_uci()
    engine_black.init_uci()

    engine_white.configure_hybrid()
    engine_black.configure_hybrid()

    # Verify engines are ready
    engine_white.send("isready")
    engine_white.read_until("readyok")
    engine_black.send("isready")
    engine_black.read_until("readyok")

    print("\nEngines initialized successfully in hybrid mode.\n")

    all_blunders = []
    all_records = []

    for game_num in range(1, args.games + 1):
        try:
            records, result = play_game(
                engine_white, engine_black, args.movetime, game_num
            )
            all_records.append(records)

            blunders = detect_blunders(records, args.threshold)
            if blunders:
                print(f"\n  ** {len(blunders)} blunder(s) detected in game {game_num}:")
                for b in blunders:
                    print(f"     Ply {b['ply']}: {b['side']} played {b['move']}")
                    print(f"       FEN:    {b['fen']}")
                    print(
                        f"       Eval:   {b['eval_before_cp']} cp -> "
                        f"{b['eval_after_cp']} cp "
                        f"(drop: {b['drop_cp']} cp)"
                    )
            else:
                print(f"  No blunders detected in game {game_num}.")

            all_blunders.extend(
                [{**b, "game": game_num} for b in blunders]
            )

        except (TimeoutError, RuntimeError) as exc:
            print(f"\n  ERROR in game {game_num}: {exc}")
            # Try to recover engines for next game
            try:
                engine_white.quit()
            except Exception:
                pass
            try:
                engine_black.quit()
            except Exception:
                pass
            engine_white = UCIEngine(args.engine, "White", hash_mb=256)
            engine_black = UCIEngine(args.engine, "Black", hash_mb=256)
            engine_white.start()
            engine_black.start()
            engine_white.init_uci()
            engine_black.init_uci()
            engine_white.configure_hybrid()
            engine_black.configure_hybrid()
            engine_white.send("isready")
            engine_white.read_until("readyok")
            engine_black.send("isready")
            engine_black.read_until("readyok")

    # Final report
    print("\n" + "=" * 70)
    print("BLUNDER DETECTION SUMMARY")
    print("=" * 70)

    total_moves = sum(len(r) for r in all_records)
    print(f"\nTotal games played: {len(all_records)}")
    print(f"Total moves analyzed: {total_moves}")
    print(f"Total blunders found (>{args.threshold} cp drop): {len(all_blunders)}")

    if all_blunders:
        print(f"\nBlunder rate: {len(all_blunders)}/{total_moves} moves "
              f"({100.0 * len(all_blunders) / max(1, total_moves):.2f}%)")
        print("\nAll blunders:")
        print("-" * 70)
        for i, b in enumerate(all_blunders, 1):
            print(f"\n  Blunder #{i} (Game {b['game']}, Ply {b['ply']}):")
            print(f"    Side:   {b['side']}")
            print(f"    FEN:    {b['fen']}")
            print(f"    Move:   {b['move']}")
            print(f"    Eval before: {b['eval_before_cp']} cp")
            print(f"    Eval after:  {b['eval_after_cp']} cp")
            print(f"    Drop:        {b['drop_cp']} cp")
    else:
        print("\nNo blunders detected across all games. Engine play was consistent.")

    print("\n" + "=" * 70)

    # Cleanup
    engine_white.quit()
    engine_black.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
