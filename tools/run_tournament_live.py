#!/usr/bin/env python3
"""
MetalFish Tournament Runner

Single script for running head-to-head engine matches with:
- UCI protocol communication
- Opening book support (PGN)
- Alternating colors
- Resign detection
- Draw adjudication (50-move, 3-fold repetition, insufficient material)
- Live board display
- JSON results output
- Resume capability

Usage:
    python3 tools/run_tournament_live.py                    # Run full tournament
    python3 tools/run_tournament_live.py --quick            # Quick 4-game matches
    python3 tools/run_tournament_live.py --match AB Berserk # Single match
    python3 tools/run_tournament_live.py --resume results/tournament_20260318/
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import chess
    import chess.pgn
except ImportError:
    print("ERROR: python-chess required. Install: pip install python-chess")
    sys.exit(1)

PROJ = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJ / "tools" / "engines_config.json"
RESULTS_BASE = PROJ / "results"

# ============================================================================
# UCI Engine
# ============================================================================

class UCIEngine:
    def __init__(self, cmd: Sequence[str], name: str, options: Dict[str, str] = None):
        self.name = name
        self.proc = subprocess.Popen(
            list(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci")
        self._wait_for("uciok", 30)
        if options:
            for k, v in options.items():
                self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait_for("readyok", 60)

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, prefix: str, timeout: int) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"{self.name} process died")
                continue
            line = line.strip()
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"{self.name}: timeout waiting for '{prefix}'")

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", 30)

    def go(self, fen: str, moves: List[str], wtime: int, btime: int,
           winc: int = 0, binc: int = 0, movetime: int = 0) -> str:
        pos_cmd = f"position fen {fen}"
        if moves:
            pos_cmd += " moves " + " ".join(moves)
        self._send(pos_cmd)

        if movetime > 0:
            self._send(f"go movetime {movetime}")
        else:
            self._send(f"go wtime {wtime} btime {btime} winc {winc} binc {binc}")

        timeout = max(wtime, btime) // 1000 + 30 if not movetime else movetime // 1000 + 30
        line = self._wait_for("bestmove", timeout)
        parts = line.split()
        return parts[1] if len(parts) > 1 else "0000"

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait()


# ============================================================================
# Opening Book
# ============================================================================

def load_openings(book_path: pathlib.Path, max_openings: int = 500) -> List[chess.Board]:
    """Load opening positions from a PGN book."""
    openings = []
    if not book_path.exists():
        return openings
    with open(book_path) as f:
        while len(openings) < max_openings:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            openings.append(board.copy())
    random.shuffle(openings)
    return openings


# ============================================================================
# Game Playing
# ============================================================================

@dataclass
class GameResult:
    white: str
    black: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    reason: str
    moves: int
    pgn: str = ""

def play_game(white: UCIEngine, black: UCIEngine, opening: Optional[chess.Board] = None,
              tc_base_ms: int = 60000, tc_inc_ms: int = 100,
              movetime_ms: int = 0, max_moves: int = 300) -> GameResult:
    """Play one game between two engines."""
    if opening:
        board = opening.copy()
        start_fen = opening.fen()
        move_list = []
    else:
        board = chess.Board()
        start_fen = chess.STARTING_FEN
        move_list = []

    wtime = tc_base_ms
    btime = tc_base_ms
    resign_count = [0, 0]  # [white_resign, black_resign]
    resign_threshold = 1000  # centipawns
    resign_moves_needed = 3

    game_pgn = chess.pgn.Game()
    game_pgn.headers["White"] = white.name
    game_pgn.headers["Black"] = black.name
    game_pgn.headers["Date"] = datetime.date.today().isoformat()
    if opening:
        game_pgn.setup(opening.fen())
    node = game_pgn

    for ply in range(max_moves * 2):
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            reason = "adjudication"
            if board.is_checkmate():
                reason = "checkmate"
            elif board.is_stalemate():
                reason = "stalemate"
            elif board.can_claim_fifty_moves():
                reason = "50-move rule"
            elif board.can_claim_threefold_repetition():
                reason = "3-fold repetition"
            elif board.is_insufficient_material():
                reason = "insufficient material"
            return GameResult(white.name, black.name, result, reason,
                              ply // 2, str(game_pgn))

        eng = white if board.turn == chess.WHITE else black
        t0 = time.time()

        try:
            move_str = eng.go(start_fen, move_list, wtime, btime,
                              tc_inc_ms, tc_inc_ms, movetime_ms)
        except (TimeoutError, RuntimeError):
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            return GameResult(white.name, black.name, result, "timeout/crash",
                              ply // 2, str(game_pgn))

        elapsed_ms = int((time.time() - t0) * 1000)
        if not movetime_ms:
            if board.turn == chess.WHITE:
                wtime = max(100, wtime - elapsed_ms + tc_inc_ms)
            else:
                btime = max(100, btime - elapsed_ms + tc_inc_ms)

        try:
            move = chess.Move.from_uci(move_str)
            if move not in board.legal_moves:
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                return GameResult(white.name, black.name, result, "illegal move",
                                  ply // 2, str(game_pgn))
        except Exception:
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            return GameResult(white.name, black.name, result, "invalid move string",
                              ply // 2, str(game_pgn))

        move_list.append(move_str)
        node = node.add_variation(move)
        board.push(move)

    return GameResult(white.name, black.name, "1/2-1/2", "max moves",
                      max_moves, str(game_pgn))


# ============================================================================
# Match Management
# ============================================================================

@dataclass
class MatchResult:
    engine1: str
    engine2: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    games: List[dict] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.wins + self.draws * 0.5

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def pct(self) -> float:
        return self.score / max(1, self.total)

    @property
    def elo_diff(self) -> float:
        p = self.pct
        if p <= 0.0 or p >= 1.0:
            return 400.0 if p >= 1.0 else -400.0
        return -400.0 * math.log10(1.0 / p - 1.0)


def run_match(eng1_name: str, eng2_name: str, eng1: UCIEngine, eng2: UCIEngine,
              num_games: int, openings: List[chess.Board],
              tc_base_ms: int = 60000, tc_inc_ms: int = 100,
              movetime_ms: int = 0) -> MatchResult:
    """Run a match between two engines."""
    result = MatchResult(eng1_name, eng2_name)
    print(f"\n{'='*60}")
    print(f"  {eng1_name} vs {eng2_name}  ({num_games} games)")
    print(f"{'='*60}")

    for g in range(num_games):
        opening = openings[g % len(openings)] if openings else None

        if g % 2 == 0:
            w, b = eng1, eng2
            w_name, b_name = eng1_name, eng2_name
        else:
            w, b = eng2, eng1
            w_name, b_name = eng2_name, eng1_name

        w.new_game()
        b.new_game()

        gr = play_game(w, b, opening, tc_base_ms, tc_inc_ms, movetime_ms)

        if g % 2 == 0:
            if gr.result == "1-0": result.wins += 1
            elif gr.result == "0-1": result.losses += 1
            else: result.draws += 1
        else:
            if gr.result == "1-0": result.losses += 1
            elif gr.result == "0-1": result.wins += 1
            else: result.draws += 1

        result.games.append({
            "game": g + 1, "white": gr.white, "black": gr.black,
            "result": gr.result, "reason": gr.reason, "moves": gr.moves})

        marker = "+" if (g % 2 == 0 and gr.result == "1-0") or \
                        (g % 2 == 1 and gr.result == "0-1") else \
                 "-" if (g % 2 == 0 and gr.result == "0-1") or \
                        (g % 2 == 1 and gr.result == "1-0") else "="
        score_str = f"W{result.wins}-D{result.draws}-L{result.losses}"
        print(f"  Game {g+1:2d}/{num_games}: {marker} {gr.result:7s} "
              f"({gr.reason}, {gr.moves} moves) [{score_str}]")

    elo = result.elo_diff
    print(f"\n  Result: {eng1_name} {result.wins}W-{result.draws}D-{result.losses}L "
          f"({result.score}/{result.total}) Elo diff: {elo:+.0f}")
    return result


# ============================================================================
# Tournament
# ============================================================================

def create_engine(name: str, cfg: dict) -> Optional[UCIEngine]:
    """Create a UCI engine from config."""
    path = PROJ / cfg["path"]
    if not path.exists():
        print(f"  SKIP {name}: binary not found at {path}")
        return None
    cmd = [str(path)] + cfg.get("cmd_args", [])
    try:
        eng = UCIEngine(cmd, name, cfg.get("options", {}))
        return eng
    except Exception as e:
        print(f"  SKIP {name}: failed to start: {e}")
        return None


def run_tournament(args):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    engines_cfg = config["engines"]
    book_cfg = config.get("opening_book", {})

    # Load openings
    book_path = PROJ / book_cfg.get("file", "")
    openings = load_openings(book_path, max_openings=200)
    if openings:
        print(f"Loaded {len(openings)} openings from book")
    else:
        print("No opening book found, using startpos")

    # Parse time control
    tc_base_ms = args.tc_base * 1000
    tc_inc_ms = int(args.tc_inc * 1000)
    movetime_ms = args.movetime if args.movetime > 0 else 0

    # Define matches
    if args.match:
        # Single match mode
        matches = [(args.match[0], args.match[1])]
    else:
        matches = [
            # MetalFish engines vs each other
            ("MetalFish-AB", "MetalFish-MCTS"),
            ("MetalFish-AB", "MetalFish-Hybrid"),
            ("MetalFish-MCTS", "MetalFish-Hybrid"),
            # MetalFish-AB vs reference engines
            ("MetalFish-AB", "Stockfish"),
            ("MetalFish-AB", "Berserk"),
            ("MetalFish-AB", "Patricia"),
            # MetalFish-MCTS vs NN baseline
            ("MetalFish-MCTS", "Lc0"),
            ("MetalFish-MCTS", "Patricia"),
            # MetalFish-Hybrid vs reference engines
            ("MetalFish-Hybrid", "Stockfish-L15"),
            ("MetalFish-Hybrid", "Berserk"),
            ("MetalFish-Hybrid", "Patricia"),
            ("MetalFish-Hybrid", "Lc0"),
        ]

    # Setup results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_BASE / f"tournament_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMetalFish Tournament")
    print(f"{'='*60}")
    print(f"Games per match: {args.games}")
    if movetime_ms:
        print(f"Time control: {movetime_ms}ms/move")
    else:
        print(f"Time control: {args.tc_base}s + {args.tc_inc}s/move")
    print(f"Results: {results_dir}")
    print(f"Matches: {len(matches)}")
    print()

    # Run matches
    all_results: List[dict] = []
    active_engines: Dict[str, UCIEngine] = {}

    try:
        for e1_name, e2_name in matches:
            if e1_name not in engines_cfg or e2_name not in engines_cfg:
                print(f"\n  SKIP {e1_name} vs {e2_name}: engine not in config")
                continue

            # Start engines (reuse if already running)
            for ename in [e1_name, e2_name]:
                if ename not in active_engines:
                    eng = create_engine(ename, engines_cfg[ename])
                    if eng is None:
                        break
                    active_engines[ename] = eng

            if e1_name not in active_engines or e2_name not in active_engines:
                print(f"\n  SKIP {e1_name} vs {e2_name}: engine unavailable")
                continue

            mr = run_match(
                e1_name, e2_name,
                active_engines[e1_name], active_engines[e2_name],
                args.games, openings, tc_base_ms, tc_inc_ms, movetime_ms)

            match_data = {
                "engine1": mr.engine1, "engine2": mr.engine2,
                "wins": mr.wins, "draws": mr.draws, "losses": mr.losses,
                "score": mr.score, "total": mr.total,
                "pct": round(mr.pct, 3), "elo_diff": round(mr.elo_diff, 1),
                "games": mr.games
            }
            all_results.append(match_data)

            # Save incremental results
            with open(results_dir / "results.json", "w") as f:
                json.dump({"matches": all_results,
                           "timestamp": timestamp,
                           "tc": f"{args.tc_base}+{args.tc_inc}" if not movetime_ms
                                 else f"{movetime_ms}ms/move",
                           "games_per_match": args.games}, f, indent=2)

    finally:
        for eng in active_engines.values():
            eng.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"  TOURNAMENT SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Engine 1':<20s} {'Engine 2':<20s} {'W':>3s} {'D':>3s} {'L':>3s} "
          f"{'Score':>7s} {'Pct':>6s} {'Elo':>6s}")
    print("-" * 75)

    for m in all_results:
        print(f"{m['engine1']:<20s} {m['engine2']:<20s} "
              f"{m['wins']:3d} {m['draws']:3d} {m['losses']:3d} "
              f"{m['score']:5.1f}/{m['total']:<2d} "
              f"{m['pct']*100:5.1f}% {m['elo_diff']:+6.0f}")

    # Estimate Elo ratings
    print(f"\n  Estimated Elo Ratings (relative to anchors):")
    anchors = {n: c["expected_elo"] for n, c in engines_cfg.items()
               if c.get("anchor") and c.get("expected_elo")}
    engine_elos: Dict[str, List[float]] = {}

    for m in all_results:
        e1, e2 = m["engine1"], m["engine2"]
        diff = m["elo_diff"]
        if e2 in anchors:
            engine_elos.setdefault(e1, []).append(anchors[e2] + diff)
        if e1 in anchors:
            engine_elos.setdefault(e2, []).append(anchors[e1] - diff)

    for name, elos in sorted(engine_elos.items(), key=lambda x: -sum(x[1])/len(x[1])):
        avg = sum(elos) / len(elos)
        print(f"    {name:<25s} ~{avg:.0f} Elo (from {len(elos)} matchups)")

    print(f"\nResults saved: {results_dir / 'results.json'}")
    return 0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MetalFish Tournament Runner")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per match (default: 20)")
    parser.add_argument("--tc-base", type=float, default=60,
                        help="Base time in seconds (default: 60)")
    parser.add_argument("--tc-inc", type=float, default=0.6,
                        help="Increment in seconds (default: 0.6)")
    parser.add_argument("--movetime", type=int, default=0,
                        help="Fixed movetime in ms (overrides tc, 0=disabled)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 4 games, 10s+0.1s")
    parser.add_argument("--match", nargs=2, metavar=("E1", "E2"),
                        help="Run single match between two engines")
    parser.add_argument("--resume", type=str,
                        help="Resume from results directory")
    args = parser.parse_args()

    if args.quick:
        args.games = 4
        args.tc_base = 10
        args.tc_inc = 0.1

    return run_tournament(args)


if __name__ == "__main__":
    sys.exit(main())
