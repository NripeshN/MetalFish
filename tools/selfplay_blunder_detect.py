#!/usr/bin/env python3
"""Self-play blunder detection: plays N games between two engine instances,
tracking eval and flagging blunders (eval drops > 150cp between consecutive
moves from the same side)."""

import os
import subprocess
import sys
import time

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

ENGINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "metalfish"
)
NETWORK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "networks",
    "BT4-1024x15x32h-swa-6147500.pb.gz",
)

NUM_GAMES = 3
MOVETIME = 3000
MAX_MOVES = 80
DECISIVE_THRESHOLD = 2000
BLUNDER_THRESHOLD = 150

ENGINE_OPTIONS = [
    ("UseHybridSearch", "true"),
    ("NNWeights", NETWORK_PATH),
    ("HybridMCTSRootReject", "true"),
    ("Threads", "4"),
    ("Hash", "256"),
]


class UCIEngine:
    def __init__(self, name):
        self.name = name
        self.proc = subprocess.Popen(
            [ENGINE_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        for opt_name, opt_val in ENGINE_OPTIONS:
            self._send(f"setoption name {opt_name} value {opt_val}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name}: engine process died")
            if token in line:
                return line.strip()

    def go(self, position_moves):
        """Send position and go command. Returns (bestmove, score_from_engine_pov)."""
        if position_moves:
            self._send(f"position startpos moves {' '.join(position_moves)}")
        else:
            self._send("position startpos")
        self._send("isready")
        self._wait_for("readyok")
        self._send(f"go movetime {MOVETIME}")

        last_score = 0
        found_score = False
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name}: engine process died during search")
            line = line.strip()
            if line.startswith("info") and "score" in line:
                score = self._parse_score(line)
                if score is not None:
                    last_score = score
                    found_score = True
            if line.startswith("bestmove"):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else None
                return bestmove, last_score if found_score else 0

    def _parse_score(self, info_line):
        """Parse score from info string. Returns centipawns from engine's POV."""
        parts = info_line.split()
        for i, part in enumerate(parts):
            if part == "score":
                if i + 1 < len(parts):
                    if parts[i + 1] == "cp" and i + 2 < len(parts):
                        try:
                            return int(parts[i + 2])
                        except ValueError:
                            return None
                    elif parts[i + 1] == "mate" and i + 2 < len(parts):
                        try:
                            mate_in = int(parts[i + 2])
                            return 30000 if mate_in > 0 else -30000
                        except ValueError:
                            return None
        return None

    def newgame(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_game(game_num):
    """Play a single self-play game and return results."""
    print(f"\n{'='*60}")
    print(f"  GAME {game_num}")
    print(f"{'='*60}")

    white = UCIEngine("White")
    black = UCIEngine("Black")

    moves = []
    scores_white_pov = []  # all scores from white's perspective
    move_records = []  # (ply, move, score_white_pov)

    white.newgame()
    black.newgame()

    result = "draw by move count"
    ply = 0

    try:
        while ply < MAX_MOVES:
            is_white_turn = ply % 2 == 0
            engine = white if is_white_turn else black

            bestmove, score_engine_pov = engine.go(moves)

            if bestmove is None or bestmove == "(none)":
                result = "no legal moves"
                break

            # Convert score to white's perspective
            # Engine reports from its own POV, so negate for black
            score_white_pov = score_engine_pov if is_white_turn else -score_engine_pov

            moves.append(bestmove)
            scores_white_pov.append(score_white_pov)
            move_records.append((ply + 1, bestmove, score_white_pov))

            side = "W" if is_white_turn else "B"
            move_num = (ply // 2) + 1
            print(
                f"  {move_num:3d}{'.' if is_white_turn else '...'} {bestmove:6s}  "
                f"eval={score_white_pov:+5d}cp (from white POV)"
            )

            ply += 1

            # Check for decisive
            if abs(score_white_pov) > DECISIVE_THRESHOLD:
                result = "decisive (eval > 2000cp)"
                break
            if abs(score_engine_pov) >= 30000:
                result = "mate found"
                break

    except Exception as e:
        result = f"error: {e}"
    finally:
        white.quit()
        black.quit()

    # Detect blunders
    # A blunder is when the eval from one side's perspective drops by >150cp
    # between their consecutive moves.
    # White moves are at ply 1, 3, 5, ... (indices 0, 2, 4, ...)
    # Black moves are at ply 2, 4, 6, ... (indices 1, 3, 5, ...)
    blunders = []

    # White's consecutive moves
    white_indices = list(range(0, len(scores_white_pov), 2))
    for i in range(1, len(white_indices)):
        prev_idx = white_indices[i - 1]
        curr_idx = white_indices[i]
        # White wants score to be high; a drop means blunder
        drop = scores_white_pov[prev_idx] - scores_white_pov[curr_idx]
        if drop > BLUNDER_THRESHOLD:
            blunders.append(
                {
                    "side": "White",
                    "ply": curr_idx + 1,
                    "move": moves[curr_idx],
                    "score_before": scores_white_pov[prev_idx],
                    "score_after": scores_white_pov[curr_idx],
                    "drop": drop,
                }
            )

    # Black's consecutive moves (from black's perspective, score should be low/negative from white POV)
    black_indices = list(range(1, len(scores_white_pov), 2))
    for i in range(1, len(black_indices)):
        prev_idx = black_indices[i - 1]
        curr_idx = black_indices[i]
        # Black wants white's score to decrease; an increase means blunder for black
        # From black's POV: prev = -scores_white_pov[prev_idx], curr = -scores_white_pov[curr_idx]
        # Drop for black = prev_black - curr_black = (-scores_white_pov[prev_idx]) - (-scores_white_pov[curr_idx])
        #                 = scores_white_pov[curr_idx] - scores_white_pov[prev_idx]
        drop = scores_white_pov[curr_idx] - scores_white_pov[prev_idx]
        if drop > BLUNDER_THRESHOLD:
            blunders.append(
                {
                    "side": "Black",
                    "ply": curr_idx + 1,
                    "move": moves[curr_idx],
                    "score_before": scores_white_pov[prev_idx],
                    "score_after": scores_white_pov[curr_idx],
                    "drop": drop,
                }
            )

    # Sort blunders by ply
    blunders.sort(key=lambda b: b["ply"])

    # Find worst eval swing (absolute change between any consecutive evals)
    worst_swing = 0
    worst_swing_ply = 0
    for i in range(1, len(scores_white_pov)):
        swing = abs(scores_white_pov[i] - scores_white_pov[i - 1])
        if swing > worst_swing:
            worst_swing = swing
            worst_swing_ply = i + 1

    return {
        "game_num": game_num,
        "total_moves": len(moves),
        "result": result,
        "blunders": blunders,
        "worst_swing": worst_swing,
        "worst_swing_ply": worst_swing_ply,
        "moves": moves,
        "scores": scores_white_pov,
    }


def print_report(game_result):
    g = game_result
    print(f"\n{'─'*60}")
    print(f"  GAME {g['game_num']} REPORT")
    print(f"{'─'*60}")
    print(f"  Total moves played: {g['total_moves']}")
    print(f"  Final result: {g['result']}")
    print(f"  Worst eval swing: {g['worst_swing']}cp at ply {g['worst_swing_ply']}")
    print()

    if g["blunders"]:
        print(f"  BLUNDERS DETECTED ({len(g['blunders'])}):")
        for b in g["blunders"]:
            print(
                f"    Ply {b['ply']:3d} ({b['side']:5s}): {b['move']:6s} "
                f"| before={b['score_before']:+5d}cp -> after={b['score_after']:+5d}cp "
                f"| drop={b['drop']}cp"
            )
    else:
        print("  No blunders detected (threshold: >150cp drop)")

    print()
    print(f"  UCI moves: {' '.join(g['moves'])}")
    print()


def main():
    print("MetalFish Self-Play Blunder Detection")
    print(f"Engine: {ENGINE_PATH}")
    print(f"Network: {NETWORK_PATH}")
    print(f"Games: {NUM_GAMES} | Movetime: {MOVETIME}ms | Max moves: {MAX_MOVES}")
    print(f"Blunder threshold: >{BLUNDER_THRESHOLD}cp drop")
    print(f"Decisive threshold: |eval| > {DECISIVE_THRESHOLD}cp")

    results = []
    for i in range(1, NUM_GAMES + 1):
        result = play_game(i)
        results.append(result)
        print_report(result)

    # Summary
    print(f"\n{'='*60}")
    print("  OVERALL SUMMARY")
    print(f"{'='*60}")
    total_blunders = sum(len(r["blunders"]) for r in results)
    total_moves_all = sum(r["total_moves"] for r in results)
    print(f"  Games played: {NUM_GAMES}")
    print(f"  Total moves (all games): {total_moves_all}")
    print(f"  Total blunders detected: {total_blunders}")
    for r in results:
        print(
            f"    Game {r['game_num']}: {r['total_moves']} moves, "
            f"{len(r['blunders'])} blunders, result={r['result']}"
        )


if __name__ == "__main__":
    main()
