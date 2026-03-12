#!/usr/bin/env python3
"""
MetalFish Overnight Tournament with Live Board Display.

Shows a live chess board for every game using the ChessBoardVisualizer
from elo_tournament.py. Runs cutechess-cli matches and parses output
in real-time.

Usage: python3 tools/run_tournament_live.py
"""

import subprocess, sys, os, re, time, shutil
from pathlib import Path
from datetime import datetime, timedelta

# Add tools dir to path so we can import from elo_tournament
sys.path.insert(0, str(Path(__file__).parent))

try:
    import chess
except ImportError:
    print("Installing python-chess...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-chess", "-q"])
    import chess

# Import the visualizer from elo_tournament
from elo_tournament import ChessBoardVisualizer

# ── Configuration ──
DIR = Path(__file__).resolve().parent.parent
CUTECHESS = DIR / "reference/cutechess/build/cutechess-cli"
BOOK = DIR / "reference/books/8moves_v3.pgn"
NNWEIGHTS = DIR / "networks/BT4-1024x15x32h-swa-6147500.pb"
METALFISH = DIR / "build/metalfish"
STOCKFISH = DIR / "reference/stockfish/src/stockfish"
BERSERK = DIR / "reference/berserk/src/berserk"
PATRICIA = DIR / "reference/Patricia/engine/patricia"

TC = "300+3"
GAMES = 20
THREADS = 4
HASH = 256

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = DIR / f"results/tournament_{TIMESTAMP}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Colors
C = "\033[36m"; W = "\033[1;37m"; G = "\033[32m"; R = "\033[31m"
Y = "\033[33m"; M = "\033[35m"; D = "\033[2m"; B = "\033[1m"
N = "\033[0m"


def clear():
    os.system("clear" if os.name != "nt" else "cls")


class TournamentRunner:
    def __init__(self):
        self.viz = ChessBoardVisualizer()
        self.total_matches = 0
        self.current_match = 0
        self.games_played = 0
        self.start_time = time.time()
        self.match_results = []  # list of (name1, name2, w, d, l)

    def elapsed(self):
        s = int(time.time() - self.start_time)
        return f"{s//3600}h{(s%3600)//60:02d}m"

    def banner(self):
        print()
        print(f"  {C}{B}MetalFish Overnight ELO Tournament{N}")
        print(f"  {C}{'='*40}{N}")
        print(f"  TC: {W}{TC}{N}  Threads: {W}{THREADS}{N}  Hash: {W}{HASH}MB{N}")
        print(f"  Games/pair: {W}{GAMES}{N}  Started: {W}{datetime.now().strftime('%H:%M')}{N}")
        print()

    def run_match(self, name1, cmd1, opts1, name2, cmd2, opts2):
        """Run a single match with live board display."""
        self.current_match += 1
        pgn_file = RESULTS_DIR / f"{name1}_vs_{name2}.pgn"

        print()
        print(f"  {C}{B}Match {self.current_match}/{self.total_matches}: "
              f"{W}{name1}{C} vs {W}{name2}{N}")
        print(f"  {D}Elapsed: {self.elapsed()} | {GAMES} games at TC {TC}{N}")
        print(f"  {D}Each game ~5-10 min. Live board shown below.{N}")
        print()

        # Build cutechess command
        cmd = [
            str(CUTECHESS),
            "-engine", f"cmd={cmd1}", f"name={name1}", "proto=uci",
            f"option.Threads={THREADS}", f"option.Hash={HASH}",
        ]
        for opt in opts1:
            cmd.append(opt)
        cmd += [
            "-engine", f"cmd={cmd2}", f"name={name2}", "proto=uci",
            f"option.Threads={THREADS}", f"option.Hash={HASH}",
        ]
        for opt in opts2:
            cmd.append(opt)
        cmd += [
            "-each", f"tc={TC}",
            "-games", str(GAMES),
            "-rounds", str(GAMES // 2),
            "-repeat", "-recover",
            "-concurrency", "1",
            "-openings", f"file={BOOK}", "format=pgn", "order=random",
            "-resign", "movecount=5", "score=1000",
            "-draw", "movenumber=40", "movecount=8", "score=10",
            "-pgnout", str(pgn_file),
        ]

        # Track match score
        match_w, match_d, match_l = 0, 0, 0

        # Reset board for first game
        self.viz.reset()
        self.viz.white_player = name1
        self.viz.black_player = name2
        current_game = 0
        is_white_turn = True
        move_list = []

        # Run cutechess and parse output
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )

        for line in proc.stdout:
            line = line.strip()

            # Game started
            m = re.match(r"Started game (\d+)", line)
            if m:
                current_game = int(m.group(1))
                self.viz.reset()
                move_list = []
                is_white_turn = True

                # Alternate player names for color swap
                if current_game % 2 == 0:
                    self.viz.white_player = name2
                    self.viz.black_player = name1
                else:
                    self.viz.white_player = name1
                    self.viz.black_player = name2

                # Show board
                clear()
                self.banner()
                print(f"  {B}Match {self.current_match}/{self.total_matches}: "
                      f"{name1} vs {name2}{N}")
                print(f"  {D}Score: +{match_w} ={match_d} -{match_l}{N}")
                print(self.viz.render(
                    last_move=f"Game {current_game}/{GAMES} - starting",
                    white_score=match_w + match_d * 0.5,
                    black_score=match_l + match_d * 0.5
                ))

            # Move played - parse from PGN-style notation in cutechess output
            # cutechess outputs moves like: "1. e4 {+0.30/20 1.2s} e5 {-0.25/18 0.8s}"
            # But we get them from the "Finished game" line with the moves in the PGN
            # Actually cutechess doesn't output individual moves to stdout.
            # We need to watch the PGN file for changes.

            # Game finished
            m_fin = re.match(r"Finished game (\d+).*: (.+)", line)
            if m_fin:
                game_num = int(m_fin.group(1))
                result_str = m_fin.group(2)
                self.games_played += 1

                # Parse result
                if "1-0" in result_str:
                    match_w += 1
                    result_display = f"{G}1-0{N}"
                elif "0-1" in result_str:
                    match_l += 1
                    result_display = f"{R}0-1{N}"
                elif "1/2" in result_str:
                    match_d += 1
                    result_display = f"{Y}1/2-1/2{N}"
                else:
                    result_display = result_str

                # Try to replay the game from PGN to show final position
                self._replay_last_game(pgn_file, game_num)

                reason = ""
                m_reason = re.search(r"\{(.+?)\}", line)
                if m_reason:
                    reason = m_reason.group(1)

                # Show final board
                clear()
                self.banner()
                print(f"  {B}Match {self.current_match}/{self.total_matches}: "
                      f"{name1} vs {name2}{N}")
                print(f"  {D}Score: +{match_w} ={match_d} -{match_l}{N}")
                print(self.viz.render(
                    last_move=f"Game {game_num}/{GAMES}: {result_str}",
                    white_score=match_w + match_d * 0.5,
                    black_score=match_l + match_d * 0.5
                ))
                print(f"  Game {game_num}: {result_display}  {D}{reason}{N}")
                print()

            # Score line
            if line.startswith("Score of"):
                print(f"  {B}{line}{N}")
            if line.startswith("Elo diff"):
                print(f"  {C}{line}{N}")

        proc.wait()

        # Record match result
        self.match_results.append((name1, name2, match_w, match_d, match_l))
        print(f"\n  {B}Match result: {G}+{match_w}{N} {Y}={match_d}{N} {R}-{match_l}{N}")

    def _replay_last_game(self, pgn_file, game_num):
        """Replay the latest game from PGN to show the final position."""
        try:
            with open(pgn_file, "r") as f:
                content = f.read()

            # Parse all games and get the last one
            import io
            pgn_io = io.StringIO(content)
            game = None
            for _ in range(game_num):
                g = chess.pgn.read_game(pgn_io)
                if g:
                    game = g

            if game:
                self.viz.reset()
                # Set player names from PGN headers
                self.viz.white_player = game.headers.get("White", "White")
                self.viz.black_player = game.headers.get("Black", "Black")
                # Replay all moves
                board = game.board()
                for move in game.mainline_moves():
                    self.viz.apply_move(move.uci())
        except Exception:
            pass  # If replay fails, just keep the last board state

    def final_summary(self):
        elapsed_s = int(time.time() - self.start_time)
        print()
        print(f"  {C}{B}{'='*40}{N}")
        print(f"  {C}{B}TOURNAMENT COMPLETE{N}")
        print(f"  {C}{B}{'='*40}{N}")
        print(f"  Time: {W}{elapsed_s//3600}h{(elapsed_s%3600)//60:02d}m{N}")
        print(f"  Games: {W}{self.games_played}{N}")
        print()
        print(f"  {B}Results:{N}")
        print()
        for name1, name2, w, d, l in self.match_results:
            total = w + d + l
            score = w + d * 0.5
            print(f"  {name1} vs {name2}: "
                  f"{G}+{w}{N} {Y}={d}{N} {R}-{l}{N}  ({score}/{total})")
        print()
        print(f"  {D}PGN files: {RESULTS_DIR}{N}")
        print(f"  {G}{B}Done!{N}")
        print()


def main():
    # Pre-flight checks
    for binary, name in [(METALFISH, "MetalFish"), (STOCKFISH, "Stockfish"),
                          (BERSERK, "Berserk"), (PATRICIA, "Patricia"),
                          (CUTECHESS, "cutechess-cli")]:
        if not binary.exists():
            print(f"  {R}MISSING: {binary}{N}")
            sys.exit(1)

    runner = TournamentRunner()

    # Define matches
    matches = [
        # Round 1: MetalFish-AB gauntlet
        ("MetalFish-AB", str(METALFISH), [],
         "Patricia", str(PATRICIA), []),
        ("MetalFish-AB", str(METALFISH), [],
         "Stockfish-L10", str(STOCKFISH), ["option.Skill Level=10"]),
        ("MetalFish-AB", str(METALFISH), [],
         "Stockfish-L15", str(STOCKFISH), ["option.Skill Level=15"]),
        ("MetalFish-AB", str(METALFISH), [],
         "Berserk", str(BERSERK), []),
        ("MetalFish-AB", str(METALFISH), [],
         "Stockfish-Full", str(STOCKFISH), []),
        # Round 2: MetalFish-Hybrid gauntlet
        ("MetalFish-Hybrid", str(METALFISH),
         ["option.UseHybridSearch=true", f"option.NNWeights={NNWEIGHTS}"],
         "Patricia", str(PATRICIA), []),
        ("MetalFish-Hybrid", str(METALFISH),
         ["option.UseHybridSearch=true", f"option.NNWeights={NNWEIGHTS}"],
         "Stockfish-L10", str(STOCKFISH), ["option.Skill Level=10"]),
        ("MetalFish-Hybrid", str(METALFISH),
         ["option.UseHybridSearch=true", f"option.NNWeights={NNWEIGHTS}"],
         "Stockfish-L15", str(STOCKFISH), ["option.Skill Level=15"]),
        # Round 3: Head-to-head
        ("MetalFish-AB", str(METALFISH), [],
         "MetalFish-Hybrid", str(METALFISH),
         ["option.UseHybridSearch=true", f"option.NNWeights={NNWEIGHTS}"]),
    ]

    runner.total_matches = len(matches)

    clear()
    runner.banner()
    print(f"  {B}Engines verified. {len(matches)} matches, "
          f"{len(matches) * GAMES} total games.{N}")
    print(f"  {Y}Starting in 3 seconds...{N}")
    time.sleep(3)

    for name1, cmd1, opts1, name2, cmd2, opts2 in matches:
        runner.run_match(name1, cmd1, opts1, name2, cmd2, opts2)

    runner.final_summary()


if __name__ == "__main__":
    main()
