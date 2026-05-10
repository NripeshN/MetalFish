#!/usr/bin/env python3
"""Lichess Bot: runs MetalFish Hybrid on Lichess via the Bot API.

Optimized for M2 Max (12 cores, 38-core GPU, 32GB unified memory).
Continuously accepts and plays challenges until interrupted.

Usage:
    python3 tools/lichess_bot.py --seek --no-casual
    python3 tools/lichess_bot.py --seek --tc "5+3" --no-casual
    python3 tools/lichess_bot.py --seek --rotate --no-casual
"""

import json
import os
import random
import subprocess
import sys
import threading
import time
import pathlib
import argparse
import traceback

import requests
import chess

PROJ = pathlib.Path(__file__).resolve().parent.parent
ENGINE = PROJ / "build" / "metalfish"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"
LICHESS_API = "https://lichess.org/api"
EXPLORER_API = "https://explorer.lichess.ovh"

ENGINE_OPTIONS = {
    "Threads": "11",
    "Hash": "8192",
    "UseHybridSearch": "true",
    "NNWeights": str(WEIGHTS),
    "HybridMCTSThreads": "1",
    "HybridABThreads": "10",
    "Move Overhead": "200",
    "MCTSMinibatchSize": "256",
    "MCTSPolicySoftmaxTemp": "1.36",
}

ROTATION_TCS = [
    (180, 2),    # 3+2 blitz
    (300, 0),    # 5+0 blitz
    (300, 3),    # 5+3 blitz
    (600, 0),    # 10+0 rapid
    (600, 5),    # 10+5 rapid
    (900, 10),   # 15+10 rapid
    (120, 1),    # 2+1 bullet
    (60, 1),     # 1+1 bullet
]

ACCEPTED_SPEEDS = {"bullet", "blitz", "rapid", "classical", "correspondence"}

SEEK_RETRY_DELAY = 10
CHALLENGE_TIMEOUT = 30


def parse_tc(tc_str: str) -> tuple[int, int]:
    parts = tc_str.replace(" ", "").split("+")
    limit = int(float(parts[0]) * 60)
    inc = int(parts[1]) if len(parts) > 1 else 0
    return (limit, inc)


def load_api_key() -> str:
    key = os.environ.get("LICHESS_API_KEY")
    if key:
        return key
    env_file = PROJ / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("LICHESS_API_KEY="):
                return line.split("=", 1)[1].strip()
    print("ERROR: No LICHESS_API_KEY found in .env or environment")
    sys.exit(1)


class UCIEngine:
    def __init__(self, path: pathlib.Path, options: dict):
        self.proc = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._send("uci")
        self._wait_for("uciok")
        for name, value in options.items():
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok", timeout=120)

    def _send(self, cmd: str):
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _wait_for(self, prefix: str, timeout: float = 60) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("Engine process died")
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Engine process closed stdout")
            line = line.strip()
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"Timeout waiting for '{prefix}'")

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", timeout=120)

    def set_position(self, initial_fen: str, moves: list[str]):
        if initial_fen == "startpos":
            cmd = "position startpos"
        else:
            cmd = f"position fen {initial_fen}"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)

    def go(self, *, wtime=None, btime=None, winc=None, binc=None,
           movetime=None, movestogo=None) -> str:
        self._send("isready")
        self._wait_for("readyok")

        cmd = "go"
        if movetime is not None:
            cmd += f" movetime {movetime}"
        else:
            if wtime is not None:
                cmd += f" wtime {wtime}"
            if btime is not None:
                cmd += f" btime {btime}"
            if winc is not None:
                cmd += f" winc {winc}"
            if binc is not None:
                cmd += f" binc {binc}"
            if movestogo is not None:
                cmd += f" movestogo {movestogo}"

        self._send(cmd)
        line = self._wait_for("bestmove", timeout=600)
        parts = line.split()
        return parts[1] if len(parts) > 1 else "0000"

    def alive(self) -> bool:
        return self.proc.poll() is None

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


class OpeningBook:
    def __init__(self, api_key: str, min_games: int = 5):
        self.min_games = min_games
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self._cache: dict[str, str | None] = {}

    def lookup(self, fen: str) -> str | None:
        if fen in self._cache:
            return self._cache[fen]
        move = self._query_masters(fen) or self._query_lichess(fen)
        self._cache[fen] = move
        return move

    def _query_masters(self, fen: str) -> str | None:
        try:
            r = requests.get(f"{EXPLORER_API}/masters", params={
                "fen": fen, "topGames": 0, "recentGames": 0,
            }, headers=self.headers, timeout=3)
            if r.status_code == 200:
                return self._pick_best_move(r.json())
        except Exception:
            pass
        return None

    def _query_lichess(self, fen: str) -> str | None:
        try:
            r = requests.get(f"{EXPLORER_API}/lichess", params={
                "fen": fen, "ratings": "2200,2500",
                "speeds": "blitz,rapid,classical",
                "topGames": 0, "recentGames": 0,
            }, headers=self.headers, timeout=3)
            if r.status_code == 200:
                return self._pick_best_move(r.json())
        except Exception:
            pass
        return None

    def _pick_best_move(self, data: dict) -> str | None:
        moves = data.get("moves", [])
        if not moves:
            return None
        best, best_score = None, -1.0
        for m in moves:
            games = m.get("white", 0) + m.get("draws", 0) + m.get("black", 0)
            if games < self.min_games:
                continue
            wins = m.get("white", 0) + m.get("draws", 0) * 0.5
            score = (wins / games) * (games ** 0.3) if games > 0 else 0
            if score > best_score:
                best_score = score
                best = m.get("uci")
        return best


class LichessBot:
    def __init__(self, api_key: str, args):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.args = args
        self.active_games: dict[str, threading.Thread] = {}
        self.bot_id = ""
        self.username = ""
        self._rotation_idx = 0
        self._pending_challenge: str | None = None
        self._challenge_sent_at: float = 0
        self.book = OpeningBook(api_key=api_key, min_games=5)

    def api_get(self, path: str, **kwargs):
        return requests.get(f"{LICHESS_API}{path}", headers=self.headers, **kwargs)

    def api_post(self, path: str, **kwargs):
        return requests.post(f"{LICHESS_API}{path}", headers=self.headers, **kwargs)

    def get_profile(self) -> dict:
        r = self.api_get("/account")
        r.raise_for_status()
        return r.json()

    def accept_challenge(self, challenge_id: str):
        r = self.api_post(f"/challenge/{challenge_id}/accept")
        if r.status_code != 200:
            print(f"  Could not accept {challenge_id}: {r.status_code}")

    def decline_challenge(self, challenge_id: str, reason: str = "generic"):
        self.api_post(f"/challenge/{challenge_id}/decline", json={"reason": reason})

    def make_move(self, game_id: str, move: str) -> bool:
        r = self.api_post(f"/bot/game/{game_id}/move/{move}")
        if r.status_code != 200:
            print(f"  [{game_id}] Move {move} failed: {r.status_code}")
            return False
        return True

    def resign(self, game_id: str):
        self.api_post(f"/bot/game/{game_id}/resign")

    # ---- Seeking / Challenging ----

    def seek_game(self):
        if self._pending_challenge and (time.time() - self._challenge_sent_at < CHALLENGE_TIMEOUT):
            return  # still waiting for previous challenge response

        self._pending_challenge = None
        limit, inc = self._next_tc()
        tc_label = f"{limit//60}+{inc}"

        try:
            r = self.api_get("/bot/online", params={"nb": 100})
            if r.status_code != 200:
                print(f"  Failed to list bots: {r.status_code}")
                return

            bots = []
            for line in r.text.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    bot = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bot_id = bot.get("id", "")
                if bot_id and bot_id != self.bot_id:
                    bots.append(bot_id)

            if not bots:
                print("  No online bots found")
                return

            random.shuffle(bots)

            for target in bots[:5]:
                rated = self.args.accept_rated
                r = self.api_post(f"/challenge/{target}", json={
                    "rated": rated,
                    "clock.limit": limit,
                    "clock.increment": inc,
                })
                if r.status_code == 200:
                    resp = r.json()
                    cid = resp.get("challenge", {}).get("id", "")
                    self._pending_challenge = cid
                    self._challenge_sent_at = time.time()
                    print(f"  Challenged {target} ({tc_label}, {'rated' if rated else 'casual'})")
                    return
                elif r.status_code == 429:
                    print("  Rate limited, backing off...")
                    time.sleep(60)
                    return

            print(f"  All challenge attempts failed")
        except Exception as e:
            print(f"  Seek error: {e}")

    def _next_tc(self) -> tuple[int, int]:
        if self.args.tc:
            return parse_tc(self.args.tc)
        if self.args.rotate:
            tc = ROTATION_TCS[self._rotation_idx % len(ROTATION_TCS)]
            self._rotation_idx += 1
            return tc
        return (300, 3)

    def _should_seek(self) -> bool:
        return self.args.seek and len(self.active_games) < self.args.max_games

    # ---- Challenge acceptance ----

    def should_accept(self, challenge: dict) -> bool:
        ch = challenge.get("challenge", challenge)

        # Ignore our own outgoing challenges
        challenger_id = ch.get("challenger", {}).get("id", "")
        if challenger_id == self.bot_id:
            return False

        variant = ch.get("variant", {}).get("key", "standard")
        if variant != "standard":
            return False

        rated = ch.get("rated", False)
        if rated and not self.args.accept_rated:
            return False
        if not rated and not self.args.accept_casual:
            return False

        speed = ch.get("speed", "")
        if speed not in ACCEPTED_SPEEDS:
            return False

        if len(self.active_games) >= self.args.max_games:
            return False

        return True

    # ---- Game play ----

    def play_game(self, game_id: str):
        print(f"  [{game_id}] Starting...")
        engine = None
        try:
            engine = UCIEngine(ENGINE, ENGINE_OPTIONS)
            engine.new_game()
            self._game_loop(game_id, engine)
        except Exception as e:
            print(f"  [{game_id}] Error: {e}")
            traceback.print_exc()
        finally:
            if engine:
                engine.quit()
            self.active_games.pop(game_id, None)
            print(f"  [{game_id}] Finished.")
            if self._should_seek():
                time.sleep(3)
                self.seek_game()

    def _game_loop(self, game_id: str, engine: UCIEngine):
        with self.api_get(f"/bot/game/stream/{game_id}", stream=True) as r:
            if r.status_code != 200:
                print(f"  [{game_id}] Stream failed: {r.status_code}")
                return
            game_info = {}
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")

                if etype == "gameFull":
                    game_info = event
                    my_color = self._get_my_color(event)
                    initial_fen = event.get("initialFen", "startpos")
                    state = event.get("state", {})
                    moves = state.get("moves", "").split() if state.get("moves") else []
                    status = state.get("status", "started")
                    if status != "started":
                        break
                    self._try_move(game_id, engine, initial_fen, moves, my_color, state)

                elif etype == "gameState":
                    status = event.get("status", "started")
                    if status != "started":
                        break
                    initial_fen = game_info.get("initialFen", "startpos")
                    my_color = self._get_my_color(game_info)
                    moves = event.get("moves", "").split() if event.get("moves") else []
                    self._try_move(game_id, engine, initial_fen, moves, my_color, event)

                elif etype == "chatLine":
                    pass  # ignore chat

    def _get_my_color(self, game_full: dict) -> str:
        white_id = game_full.get("white", {}).get("id", "")
        return "white" if white_id == self.bot_id else "black"

    def _try_move(self, game_id: str, engine: UCIEngine,
                  initial_fen: str, moves: list[str],
                  my_color: str, state: dict):
        is_white_turn = len(moves) % 2 == 0
        is_my_turn = (is_white_turn and my_color == "white") or \
                     (not is_white_turn and my_color == "black")
        if not is_my_turn:
            return

        # Opening book lookup (instant, no clock cost)
        board = chess.Board(initial_fen) if initial_fen != "startpos" else chess.Board()
        for m in moves:
            try:
                board.push_uci(m)
            except ValueError:
                break

        book_move = self.book.lookup(board.fen())
        if book_move:
            # Verify it's legal
            try:
                chess.Move.from_uci(book_move)
                if chess.Move.from_uci(book_move) in board.legal_moves:
                    print(f"  [{game_id}] Book: {book_move}")
                    self.make_move(game_id, book_move)
                    return
            except ValueError:
                pass

        # Engine search
        if not engine.alive():
            print(f"  [{game_id}] Engine died, resigning")
            self.resign(game_id)
            return

        engine.set_position(initial_fen, moves)

        wtime = state.get("wtime", 60000)
        btime = state.get("btime", 60000)
        winc = state.get("winc", 0)
        binc = state.get("binc", 0)

        if not isinstance(wtime, int):
            wtime = 300000
        if not isinstance(btime, int):
            btime = 300000

        try:
            move = engine.go(wtime=wtime, btime=btime, winc=winc, binc=binc)
            if move == "0000" or move == "(none)":
                return
            self.make_move(game_id, move)
        except (TimeoutError, RuntimeError) as e:
            print(f"  [{game_id}] Engine error: {e}, resigning")
            self.resign(game_id)

    # ---- Main event loop ----

    def run(self):
        profile = self.get_profile()
        self.bot_id = profile.get("id", "")
        self.username = profile.get("username", self.bot_id)
        title = profile.get("title", "")

        if title != "BOT":
            print(f"ERROR: '{self.username}' is not a BOT account.")
            print("Upgrade at: https://lichess.org/account/bot")
            sys.exit(1)

        tc_mode = "rotate" if self.args.rotate else (self.args.tc or "5+3")
        print("=" * 60)
        print(f"  MetalFish Lichess Bot")
        print(f"  Account:  {self.username}")
        print(f"  Engine:   Hybrid (AB 10T + MCTS 1T + GPU)")
        print(f"  Hash:     8192 MB | Network: BT4-1024x15x32h")
        print(f"  Rated:    {self.args.accept_rated} | Casual: {self.args.accept_casual}")
        print(f"  Seek:     {self.args.seek} | TC: {tc_mode}")
        print(f"  Max games: {self.args.max_games}")
        print("=" * 60)
        print("\nListening for challenges... (Ctrl+C to stop)\n")

        self._event_loop()

    def _event_loop(self):
        # Send initial seek before entering event stream
        if self._should_seek():
            self.seek_game()

        while True:
            try:
                with self.api_get("/stream/event", stream=True, timeout=None) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            # Keep-alive ping — check if we should seek
                            if self._should_seek() and self._pending_challenge is None:
                                self.seek_game()
                            elif self._pending_challenge and \
                                 time.time() - self._challenge_sent_at > CHALLENGE_TIMEOUT:
                                print("  Challenge timed out, trying another...")
                                self._pending_challenge = None
                                if self._should_seek():
                                    self.seek_game()
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._handle_event(event)

            except requests.exceptions.ConnectionError:
                print("  Connection lost, reconnecting in 5s...")
                time.sleep(5)
            except requests.exceptions.ChunkedEncodingError:
                print("  Stream interrupted, reconnecting in 3s...")
                time.sleep(3)
            except requests.exceptions.ReadTimeout:
                print("  Read timeout, reconnecting...")
                time.sleep(2)
            except KeyboardInterrupt:
                print("\n\nShutting down...")
                for gid in list(self.active_games.keys()):
                    print(f"  Resigning {gid}...")
                    self.resign(gid)
                break
            except Exception as e:
                print(f"  Unexpected error in event loop: {e}")
                traceback.print_exc()
                time.sleep(5)

    def _handle_event(self, event: dict):
        etype = event.get("type", "")

        if etype == "challenge":
            ch = event["challenge"]
            challenger = ch.get("challenger", {}).get("name", "?")
            challenger_id = ch.get("challenger", {}).get("id", "")
            tc = ch.get("timeControl", {})
            if tc.get("type") == "clock":
                tc_str = f"{tc.get('limit', 0)//60}+{tc.get('increment', 0)}"
            else:
                tc_str = tc.get("type", "?")
            rated = "rated" if ch.get("rated") else "casual"

            # Skip our own outgoing challenges showing up as events
            if challenger_id == self.bot_id:
                return

            if self.should_accept(event):
                print(f"  Accepting {rated} from {challenger} ({tc_str})")
                self.accept_challenge(ch["id"])
            else:
                print(f"  Declining from {challenger} ({tc_str}, {rated})")
                self.decline_challenge(ch["id"])

        elif etype == "gameStart":
            game_id = event["game"]["gameId"]
            self._pending_challenge = None
            if game_id not in self.active_games:
                t = threading.Thread(target=self.play_game, args=(game_id,), daemon=True)
                self.active_games[game_id] = t
                t.start()

        elif etype == "gameFinish":
            game_id = event.get("game", {}).get("gameId", "")
            self.active_games.pop(game_id, None)

        elif etype == "challengeDeclined":
            self._pending_challenge = None
            if self._should_seek():
                time.sleep(3)
                self.seek_game()

        elif etype == "challengeCanceled":
            self._pending_challenge = None
            if self._should_seek():
                time.sleep(3)
                self.seek_game()


def main():
    parser = argparse.ArgumentParser(
        description="MetalFish Lichess Bot — plays continuously until stopped")
    parser.add_argument("--accept-rated", action="store_true", default=True)
    parser.add_argument("--no-rated", dest="accept_rated", action="store_false")
    parser.add_argument("--accept-casual", action="store_true", default=True)
    parser.add_argument("--no-casual", dest="accept_casual", action="store_false")
    parser.add_argument("--max-games", type=int, default=1,
                        help="Max concurrent games (default: 1)")
    parser.add_argument("--seek", action="store_true", default=False,
                        help="Actively challenge online bots")
    parser.add_argument("--tc", type=str, default=None,
                        help="Fixed time control, e.g. '5+3', '3+2', '10+0'")
    parser.add_argument("--rotate", action="store_true", default=False,
                        help="Cycle through blitz/rapid/bullet time controls")
    args = parser.parse_args()

    if not ENGINE.exists():
        print(f"ERROR: Engine not found at {ENGINE}")
        print("Build with: cd build && cmake .. && make -j8")
        sys.exit(1)
    if not WEIGHTS.exists():
        print(f"ERROR: Weights not found at {WEIGHTS}")
        sys.exit(1)

    api_key = load_api_key()
    bot = LichessBot(api_key, args)
    bot.run()


if __name__ == "__main__":
    main()
