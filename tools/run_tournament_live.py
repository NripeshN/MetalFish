#!/usr/bin/env python3
"""
MetalFish Live Tournament - streams every move to the board display.

Runs UCI engines directly (no cutechess-cli) so we can show each move
as it happens. Uses ChessBoardVisualizer for the board display.

Usage: python3 tools/run_tournament_live.py
"""

import subprocess, sys, os, time, threading, signal
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

try:
    import chess, chess.pgn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-chess", "-q"])
    import chess, chess.pgn

from elo_tournament import ChessBoardVisualizer

# ── Config ──
DIR = Path(__file__).resolve().parent.parent
NNWEIGHTS = DIR / "networks/BT4-1024x15x32h-swa-6147500.pb"
METALFISH = str(DIR / "build/metalfish")
STOCKFISH = str(DIR / "reference/stockfish/src/stockfish")
BERSERK = str(DIR / "reference/berserk/src/berserk")
PATRICIA = str(DIR / "reference/Patricia/engine/patricia")
BOOK = DIR / "reference/books/8moves_v3.pgn"

TC_BASE = 300  # seconds
TC_INC = 3     # seconds per move
GAMES = 20
THREADS = 4
HASH = 256

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = DIR / f"results/tournament_{TIMESTAMP}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

C = "\033[36m"; W = "\033[1;37m"; G = "\033[32m"; R = "\033[31m"
Y = "\033[33m"; D = "\033[2m"; B = "\033[1m"; N = "\033[0m"


class UCIEngine:
    """Manages a UCI engine process."""

    def __init__(self, cmd, name, options=None):
        self.name = name
        self.proc = subprocess.Popen(
            [cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        self._send("uci")
        self._wait_for("uciok")
        # Set options
        for key, val in (options or {}).items():
            self._send(f"setoption name {key} value {val}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _readline(self):
        return self.proc.stdout.readline().strip()

    def _wait_for(self, token):
        while True:
            line = self._readline()
            if line.startswith(token):
                return line

    def newgame(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def go(self, position_cmd, wtime, btime, winc, binc):
        """Send position and go, return (bestmove, info_lines)."""
        self._send(position_cmd)
        self._send(f"go wtime {wtime} btime {btime} winc {winc} binc {binc}")
        info_lines = []
        while True:
            line = self._readline()
            if line.startswith("bestmove"):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else None
                return bestmove, info_lines
            elif line.startswith("info") and "depth" in line and "score" in line:
                info_lines.append(line)

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except:
            self.proc.kill()


def load_openings(book_path, count=100):
    """Load opening positions from a PGN book."""
    openings = []
    try:
        with open(book_path) as f:
            while len(openings) < count:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                moves = []
                for move in game.mainline_moves():
                    moves.append(move.uci())
                    board.push(move)
                if moves:
                    openings.append(moves)
    except:
        pass
    return openings if openings else [["e2e4"], ["d2d4"], ["g1f3"], ["c2c4"]]


def parse_score(info_lines):
    """Extract score from the last info line."""
    if not info_lines:
        return ""
    last = info_lines[-1]
    # Find "score cp X" or "score mate X"
    parts = last.split()
    for i, p in enumerate(parts):
        if p == "score" and i + 2 < len(parts):
            if parts[i+1] == "cp":
                cp = int(parts[i+2])
                return f"{cp/100:+.2f}"
            elif parts[i+1] == "mate":
                return f"M{parts[i+2]}"
    return ""


def parse_depth(info_lines):
    """Extract depth from the last info line."""
    if not info_lines:
        return ""
    last = info_lines[-1]
    parts = last.split()
    for i, p in enumerate(parts):
        if p == "depth" and i + 1 < len(parts):
            return parts[i+1]
    return ""


def parse_nps(info_lines):
    """Extract NPS from the last info line."""
    if not info_lines:
        return ""
    last = info_lines[-1]
    parts = last.split()
    for i, p in enumerate(parts):
        if p == "nps" and i + 1 < len(parts):
            nps = int(parts[i+1])
            if nps > 1000000:
                return f"{nps/1000000:.1f}M"
            elif nps > 1000:
                return f"{nps/1000:.0f}K"
            return str(nps)
    return ""


def play_game(eng_w, eng_b, opening_moves, viz, game_num, total_games,
              match_name, match_score_w, match_score_b):
    """Play one game between two engines with live display."""
    board = chess.Board()
    viz.reset()
    viz.white_player = eng_w.name
    viz.black_player = eng_b.name

    eng_w.newgame()
    eng_b.newgame()

    moves_uci = []
    wtime = TC_BASE * 1000
    btime = TC_BASE * 1000
    winc = TC_INC * 1000
    binc = TC_INC * 1000
    last_score = ""
    last_depth = ""
    last_nps = ""
    result = "*"

    # Apply opening moves
    for uci_move in opening_moves:
        move = chess.Move.from_uci(uci_move)
        if move in board.legal_moves:
            board.push(move)
            moves_uci.append(uci_move)
            viz.apply_move(uci_move)

    # Show opening position
    os.system("clear")
    move_num = len(board.move_stack) // 2 + 1
    side = "White" if board.turn == chess.WHITE else "Black"
    print(viz.render(
        last_move=f"Game {game_num}/{total_games} | {side} to move (opening)",
        white_score=match_score_w,
        black_score=match_score_b
    ))
    print(f"  {D}{match_name}{N}")

    # Game loop
    while not board.is_game_over(claim_draw=True):
        # Build position command
        if moves_uci:
            pos_cmd = "position startpos moves " + " ".join(moves_uci)
        else:
            pos_cmd = "position startpos"

        # Determine which engine to move
        if board.turn == chess.WHITE:
            engine = eng_w
            t_start = time.time()
            bestmove, info = engine.go(pos_cmd, wtime, btime, winc, binc)
            elapsed_ms = int((time.time() - t_start) * 1000)
            wtime = max(100, wtime - elapsed_ms + winc)
        else:
            engine = eng_b
            t_start = time.time()
            bestmove, info = engine.go(pos_cmd, wtime, btime, winc, binc)
            elapsed_ms = int((time.time() - t_start) * 1000)
            btime = max(100, btime - elapsed_ms + binc)

        if not bestmove or bestmove == "(none)":
            break

        # Apply move
        try:
            move = chess.Move.from_uci(bestmove)
            if move not in board.legal_moves:
                break
        except:
            break

        # Get SAN before pushing
        san = board.san(move)
        move_num = board.fullmove_number
        is_white = board.turn == chess.WHITE

        board.push(move)
        moves_uci.append(bestmove)
        viz.apply_move(bestmove)

        last_score = parse_score(info)
        last_depth = parse_depth(info)
        last_nps = parse_nps(info)

        # Format move display
        if is_white:
            move_str = f"{move_num}. {san}"
        else:
            move_str = f"{move_num}... {san}"

        time_str = f"{elapsed_ms/1000:.1f}s"
        info_str = f"{move_str}  {last_score}  d{last_depth}  {last_nps}  {time_str}"

        # Format clocks
        wm, ws = divmod(wtime // 1000, 60)
        bm, bs = divmod(btime // 1000, 60)
        clock_str = f"W {wm}:{ws:02d}  B {bm}:{bs:02d}"

        # Update display
        os.system("clear")
        print(viz.render(
            last_move=f"Game {game_num}/{total_games} | {info_str}",
            white_score=match_score_w,
            black_score=match_score_b
        ))
        print(f"  {D}{match_name}  |  {clock_str}  |  Ply {len(board.move_stack)}{N}")

        # Check time forfeit
        if wtime <= 0:
            result = "0-1"
            break
        if btime <= 0:
            result = "1-0"
            break

        # Simple resign detection (score > 10.0 for 5 moves)
        # (simplified - just let games play out)

    # Determine result
    if result == "*":
        if board.is_checkmate():
            result = "0-1" if board.turn == chess.WHITE else "1-0"
        elif board.is_stalemate() or board.is_insufficient_material() or \
             board.can_claim_draw() or board.is_fifty_moves():
            result = "1/2-1/2"
        else:
            result = "1/2-1/2"  # Default to draw for abnormal endings

    # Show final position
    if result == "1-0":
        res_display = f"{G}1-0 {viz.white_player} wins{N}"
    elif result == "0-1":
        res_display = f"{R}0-1 {viz.black_player} wins{N}"
    else:
        res_display = f"{Y}1/2-1/2 Draw{N}"

    os.system("clear")
    print(viz.render(
        last_move=f"Game {game_num}/{total_games} FINISHED: {result}",
        white_score=match_score_w + (1 if result == "1-0" else 0.5 if result == "1/2-1/2" else 0),
        black_score=match_score_b + (1 if result == "0-1" else 0.5 if result == "1/2-1/2" else 0)
    ))
    print(f"  {res_display}  {D}({len(board.move_stack)} plies){N}")
    print()

    return result, board, moves_uci


def save_pgn(pgn_path, white, black, result, moves_uci, game_num, opening_len):
    """Append a game to a PGN file."""
    game = chess.pgn.Game()
    game.headers["Event"] = "MetalFish Tournament"
    game.headers["Site"] = "Apple M2 Max"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_num)
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Result"] = result
    game.headers["TimeControl"] = f"{TC_BASE}+{TC_INC}"

    node = game
    board = chess.Board()
    for uci in moves_uci:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)

    with open(pgn_path, "a") as f:
        f.write(str(game) + "\n\n")


def run_match(name1, cmd1, opts1, name2, cmd2, opts2, match_num, total_matches, openings):
    """Run a full match between two engines."""
    viz = ChessBoardVisualizer()
    pgn_path = RESULTS_DIR / f"{name1}_vs_{name2}.pgn"
    match_name = f"Match {match_num}/{total_matches}: {name1} vs {name2}"

    score_1 = 0.0  # First engine's score
    score_2 = 0.0
    results = []

    for g in range(GAMES):
        game_num = g + 1
        opening = openings[g % len(openings)] if openings else []

        # Alternate colors every 2 games (same opening, swapped colors)
        if g % 2 == 0:
            # Engine 1 is white
            eng_w = UCIEngine(cmd1, name1, opts1)
            eng_b = UCIEngine(cmd2, name2, opts2)
            result, board, moves = play_game(
                eng_w, eng_b, opening, viz, game_num, GAMES,
                match_name, score_1, score_2
            )
            save_pgn(pgn_path, name1, name2, result, moves, game_num, len(opening))
            if result == "1-0":
                score_1 += 1
            elif result == "0-1":
                score_2 += 1
            else:
                score_1 += 0.5; score_2 += 0.5
        else:
            # Engine 2 is white (color swap)
            eng_w = UCIEngine(cmd2, name2, opts2)
            eng_b = UCIEngine(cmd1, name1, opts1)
            result, board, moves = play_game(
                eng_w, eng_b, opening, viz, game_num, GAMES,
                match_name, score_2, score_1
            )
            save_pgn(pgn_path, name2, name1, result, moves, game_num, len(opening))
            if result == "1-0":
                score_2 += 1
            elif result == "0-1":
                score_1 += 1
            else:
                score_1 += 0.5; score_2 += 0.5

        eng_w.quit()
        eng_b.quit()
        results.append(result)

        # Brief pause between games
        time.sleep(1)

    # Match summary
    w = sum(1 for r in results if r == "1-0")
    d = sum(1 for r in results if r == "1/2-1/2")
    l = sum(1 for r in results if r == "0-1")
    print(f"\n  {B}Match result ({name1} perspective): {G}+{w}{N} {Y}={d}{N} {R}-{l}{N}")
    print(f"  Score: {score_1}/{GAMES}")
    print()

    return score_1, score_2


def main():
    # Check engines
    for path, name in [(METALFISH, "MetalFish"), (STOCKFISH, "Stockfish"),
                        (BERSERK, "Berserk"), (PATRICIA, "Patricia")]:
        if not Path(path).exists():
            print(f"{R}MISSING: {path}{N}")
            sys.exit(1)

    openings = load_openings(BOOK, count=50)
    print(f"  Loaded {len(openings)} openings from book")

    matches = [
        ("MetalFish-AB", METALFISH, {"Threads": THREADS, "Hash": HASH},
         "Patricia", PATRICIA, {"Threads": THREADS, "Hash": HASH}),
        ("MetalFish-AB", METALFISH, {"Threads": THREADS, "Hash": HASH},
         "Stockfish-L10", STOCKFISH, {"Threads": THREADS, "Hash": HASH, "Skill Level": 10}),
        ("MetalFish-AB", METALFISH, {"Threads": THREADS, "Hash": HASH},
         "Stockfish-L15", STOCKFISH, {"Threads": THREADS, "Hash": HASH, "Skill Level": 15}),
        ("MetalFish-AB", METALFISH, {"Threads": THREADS, "Hash": HASH},
         "Berserk", BERSERK, {"Threads": THREADS, "Hash": HASH}),
        ("MetalFish-AB", METALFISH, {"Threads": THREADS, "Hash": HASH},
         "Stockfish-Full", STOCKFISH, {"Threads": THREADS, "Hash": HASH}),
        ("MetalFish-Hybrid", METALFISH,
         {"Threads": THREADS, "Hash": HASH, "UseHybridSearch": "true",
          "NNWeights": str(NNWEIGHTS)},
         "Patricia", PATRICIA, {"Threads": THREADS, "Hash": HASH}),
        ("MetalFish-Hybrid", METALFISH,
         {"Threads": THREADS, "Hash": HASH, "UseHybridSearch": "true",
          "NNWeights": str(NNWEIGHTS)},
         "Stockfish-L10", STOCKFISH, {"Threads": THREADS, "Hash": HASH, "Skill Level": 10}),
        ("MetalFish-Hybrid", METALFISH,
         {"Threads": THREADS, "Hash": HASH, "UseHybridSearch": "true",
          "NNWeights": str(NNWEIGHTS)},
         "Stockfish-L15", STOCKFISH, {"Threads": THREADS, "Hash": HASH, "Skill Level": 15}),
        ("MetalFish-AB", METALFISH, {"Threads": THREADS, "Hash": HASH},
         "MetalFish-Hybrid", METALFISH,
         {"Threads": THREADS, "Hash": HASH, "UseHybridSearch": "true",
          "NNWeights": str(NNWEIGHTS)}),
    ]

    start = time.time()
    all_results = []

    for i, (n1, c1, o1, n2, c2, o2) in enumerate(matches):
        s1, s2 = run_match(n1, c1, o1, n2, c2, o2, i+1, len(matches), openings)
        all_results.append((n1, n2, s1, s2))

    # Final summary
    elapsed = int(time.time() - start)
    os.system("clear")
    print(f"\n  {C}{B}{'='*45}{N}")
    print(f"  {C}{B}  TOURNAMENT COMPLETE  ({elapsed//3600}h{(elapsed%3600)//60:02d}m){N}")
    print(f"  {C}{B}{'='*45}{N}\n")

    for n1, n2, s1, s2 in all_results:
        total = int(s1 + s2)
        print(f"  {n1} vs {n2}: {G}{s1}{N} - {R}{s2}{N} / {total}")

    print(f"\n  {D}PGN files: {RESULTS_DIR}{N}")
    print(f"  {G}{B}Done!{N}\n")


if __name__ == "__main__":
    main()
