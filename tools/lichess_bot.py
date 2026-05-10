#!/usr/bin/env python3
"""Lichess Bot: runs MetalFish Hybrid on Lichess via the Bot API.

Features:
  - Opening book from Lichess masters database (instant moves)
  - Pondering (thinks on opponent's time)
  - Aggressive challenge seeking with timeout/retry
  - Rate limit awareness
  - Engine crash recovery

Usage:
    python3 tools/lichess_bot.py --seek --no-casual
    python3 tools/lichess_bot.py --seek --tc "5+3" --no-casual
    python3 tools/lichess_bot.py --seek --rotate --no-casual
"""

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import threading
import time
import traceback

import chess
import requests

PROJ = pathlib.Path(__file__).resolve().parent.parent
ENGINE = PROJ / "build" / "metalfish"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"
LICHESS_API = "https://lichess.org/api"
EXPLORER_API = "https://explorer.lichess.ovh"

ENGINE_OPTIONS = {
    "Threads": "11",
    "Hash": "8192",
    "Ponder": "true",
    "UseHybridSearch": "true",
    "NNWeights": str(WEIGHTS),
    "HybridMCTSThreads": "1",
    "HybridABThreads": "10",
    "Move Overhead": "150",
    "MCTSMinibatchSize": "256",
    "MCTSPolicySoftmaxTemp": "1.36",
}

ROTATION_TCS = [
    (180, 2),  # 3+2 blitz
    (300, 0),  # 5+0 blitz
    (300, 3),  # 5+3 blitz
    (600, 0),  # 10+0 rapid
    (600, 5),  # 10+5 rapid
    (900, 10),  # 15+10 rapid
    (120, 1),  # 2+1 bullet
]

ACCEPTED_SPEEDS = {"bullet", "blitz", "rapid", "classical", "correspondence"}
CHALLENGE_TIMEOUT = 20
MAX_CHALLENGE_RETRIES = 3


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
        self._send("uci")
        self._wait_for("uciok")
        for name, value in options.items():
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok", timeout=120)
        self._pondering = False
        self._ponder_move: str | None = None

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
                raise RuntimeError("Engine closed stdout")
            line = line.strip()
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"Timeout waiting for '{prefix}'")

    def new_game(self):
        self.stop_pondering()
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

    def go(
        self,
        *,
        wtime=None,
        btime=None,
        winc=None,
        binc=None,
        movetime=None,
        movestogo=None,
    ) -> tuple[str, str | None]:
        """Returns (bestmove, ponder_move_or_None)."""
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
        best = parts[1] if len(parts) > 1 else "0000"
        ponder = parts[3] if len(parts) > 3 and parts[2] == "ponder" else None
        return best, ponder

    def start_pondering(self, initial_fen: str, moves: list[str], ponder_move: str):
        """Start thinking on opponent's time (go ponder)."""
        self.stop_pondering()
        all_moves = moves + [ponder_move]
        self.set_position(initial_fen, all_moves)
        self._send("go ponder")
        self._pondering = True
        self._ponder_move = ponder_move

    def ponderhit(self) -> tuple[str, str | None]:
        """Opponent played the predicted move — convert ponder to real search."""
        if not self._pondering:
            return "0000", None
        self._send("ponderhit")
        self._pondering = False
        line = self._wait_for("bestmove", timeout=600)
        parts = line.split()
        best = parts[1] if len(parts) > 1 else "0000"
        ponder = parts[3] if len(parts) > 3 and parts[2] == "ponder" else None
        return best, ponder

    def stop_pondering(self):
        """Stop pondering if active."""
        if self._pondering:
            self._send("stop")
            try:
                self._wait_for("bestmove", timeout=5)
            except (TimeoutError, RuntimeError):
                pass
            self._pondering = False
            self._ponder_move = None

    @property
    def ponder_move(self) -> str | None:
        return self._ponder_move if self._pondering else None

    def alive(self) -> bool:
        return self.proc.poll() is None

    def quit(self):
        self.stop_pondering()
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
            r = requests.get(
                f"{EXPLORER_API}/masters",
                params={
                    "fen": fen,
                    "topGames": 0,
                    "recentGames": 0,
                },
                headers=self.headers,
                timeout=3,
            )
            if r.status_code == 200:
                return self._pick_best_move(r.json())
        except Exception:
            pass
        return None

    def _query_lichess(self, fen: str) -> str | None:
        try:
            r = requests.get(
                f"{EXPLORER_API}/lichess",
                params={
                    "fen": fen,
                    "ratings": "2200,2500",
                    "speeds": "blitz,rapid,classical",
                    "topGames": 0,
                    "recentGames": 0,
                },
                headers=self.headers,
                timeout=3,
            )
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
            score = (wins / games) * (games**0.3) if games > 0 else 0
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
        self._challenge_retries = 0
        self.book = OpeningBook(api_key=api_key, min_games=5)
        self._seek_timer: threading.Timer | None = None

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

    # ---- Seeking ----

    def seek_game(self):
        if self._pending_challenge and (
            time.time() - self._challenge_sent_at < CHALLENGE_TIMEOUT
        ):
            return

        self._pending_challenge = None
        limit, inc = self._next_tc()
        tc_label = f"{limit//60}+{inc}"

        try:
            r = self.api_get("/bot/online", params={"nb": 100})
            if r.status_code != 200:
                self._schedule_retry()
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
                print("  No online bots, retrying in 30s...")
                self._schedule_retry(30)
                return

            random.shuffle(bots)

            for target in bots[:8]:
                rated = self.args.accept_rated
                r = self.api_post(
                    f"/challenge/{target}",
                    json={
                        "rated": rated,
                        "clock.limit": limit,
                        "clock.increment": inc,
                    },
                )
                if r.status_code == 200:
                    self._pending_challenge = target
                    self._challenge_sent_at = time.time()
                    self._challenge_retries = 0
                    print(
                        f"  Challenged {target} ({tc_label}, {'rated' if rated else 'casual'})"
                    )
                    self._schedule_challenge_timeout()
                    return
                elif r.status_code == 429:
                    print("  Rate limited, waiting 60s...")
                    self._schedule_retry(60)
                    return
                # 400 = bot doesn't accept this TC, try next

            print(f"  No bot accepted, retrying in 10s...")
            self._schedule_retry(10)
        except Exception as e:
            print(f"  Seek error: {e}")
            self._schedule_retry(15)

    def _schedule_retry(self, delay: float = 10):
        if self._seek_timer:
            self._seek_timer.cancel()
        self._seek_timer = threading.Timer(delay, self._retry_seek)
        self._seek_timer.daemon = True
        self._seek_timer.start()

    def _retry_seek(self):
        if self._should_seek():
            self.seek_game()

    def _schedule_challenge_timeout(self):
        if self._seek_timer:
            self._seek_timer.cancel()
        self._seek_timer = threading.Timer(CHALLENGE_TIMEOUT, self._challenge_timed_out)
        self._seek_timer.daemon = True
        self._seek_timer.start()

    def _challenge_timed_out(self):
        if self._pending_challenge:
            print(f"  Challenge to {self._pending_challenge} timed out")
            # Cancel the outgoing challenge
            if self._pending_challenge:
                self.api_post(f"/challenge/{self._pending_challenge}/cancel")
            self._pending_challenge = None
            self._challenge_retries += 1
            if self._challenge_retries < MAX_CHALLENGE_RETRIES and self._should_seek():
                self.seek_game()
            elif self._should_seek():
                self._challenge_retries = 0
                self._schedule_retry(15)

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
                self._schedule_retry(3)

    def _game_loop(self, game_id: str, engine: UCIEngine):
        with self.api_get(f"/bot/game/stream/{game_id}", stream=True, timeout=30) as r:
            if r.status_code != 200:
                print(f"  [{game_id}] Stream failed: {r.status_code}")
                return
            game_info = {}
            my_color = "white"
            initial_fen = "startpos"

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
                    status = state.get("status", "started")
                    if status != "started":
                        break
                    moves = state.get("moves", "").split() if state.get("moves") else []
                    self._try_move(game_id, engine, initial_fen, moves, my_color, state)

                elif etype == "gameState":
                    status = event.get("status", "started")
                    if status != "started":
                        break
                    moves = event.get("moves", "").split() if event.get("moves") else []
                    self._try_move(game_id, engine, initial_fen, moves, my_color, event)

                elif etype == "chatLine":
                    pass

    def _get_my_color(self, game_full: dict) -> str:
        white_id = game_full.get("white", {}).get("id", "")
        return "white" if white_id == self.bot_id else "black"

    def _try_move(
        self,
        game_id: str,
        engine: UCIEngine,
        initial_fen: str,
        moves: list[str],
        my_color: str,
        state: dict,
    ):
        is_white_turn = len(moves) % 2 == 0
        is_my_turn = (is_white_turn and my_color == "white") or (
            not is_white_turn and my_color == "black"
        )

        if not is_my_turn:
            # Opponent's turn — start pondering if we have a prediction
            last_opponent_move = moves[-1] if moves else None
            if engine.ponder_move and last_opponent_move == engine.ponder_move:
                # Opponent played our predicted move — ponderhit!
                best, ponder = engine.ponderhit()
                if best and best != "0000":
                    print(f"  [{game_id}] Ponderhit! {best}")
                    if self.make_move(game_id, best) and ponder:
                        engine.start_pondering(initial_fen, moves + [best], ponder)
            else:
                engine.stop_pondering()
            return

        # My turn — check if ponderhit already handled it
        # (ponderhit path above sends the move, so we only get here if no ponderhit)

        # Opening book (instant, zero clock cost)
        board = chess.Board(initial_fen) if initial_fen != "startpos" else chess.Board()
        for m in moves:
            try:
                board.push_uci(m)
            except ValueError:
                break

        book_move = self.book.lookup(board.fen())
        if book_move:
            try:
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

        engine.stop_pondering()
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
            best, ponder = engine.go(wtime=wtime, btime=btime, winc=winc, binc=binc)
            if best == "0000" or best == "(none)":
                return
            if self.make_move(game_id, best) and ponder:
                engine.start_pondering(initial_fen, moves + [best], ponder)
        except (TimeoutError, RuntimeError) as e:
            print(f"  [{game_id}] Engine error: {e}, resigning")
            self.resign(game_id)

    # ---- Main loop ----

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
        print(f"  Engine:   Hybrid (AB 10T + MCTS 1T + GPU + Ponder)")
        print(f"  Hash:     8192 MB | Network: BT4-1024x15x32h")
        print(f"  Book:     Lichess Masters DB (instant moves)")
        print(
            f"  Rated:    {self.args.accept_rated} | Casual: {self.args.accept_casual}"
        )
        print(f"  Seek:     {self.args.seek} | TC: {tc_mode}")
        print(f"  Max games: {self.args.max_games}")
        print("=" * 60)
        print("\nListening... (Ctrl+C to stop)\n")

        self._event_loop()

    def _event_loop(self):
        if self._should_seek():
            self.seek_game()

        while True:
            try:
                with self.api_get("/stream/event", stream=True, timeout=(10, 30)) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._handle_event(event)

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ):
                print("  Connection lost, reconnecting in 3s...")
                time.sleep(3)
            except requests.exceptions.ReadTimeout:
                pass  # normal — just reconnect
            except KeyboardInterrupt:
                print("\n\nShutting down...")
                if self._seek_timer:
                    self._seek_timer.cancel()
                for gid in list(self.active_games.keys()):
                    print(f"  Resigning {gid}...")
                    self.resign(gid)
                break
            except Exception as e:
                print(f"  Event loop error: {e}")
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
            if self._seek_timer:
                self._seek_timer.cancel()
            if game_id not in self.active_games:
                t = threading.Thread(
                    target=self.play_game, args=(game_id,), daemon=True
                )
                self.active_games[game_id] = t
                t.start()

        elif etype == "gameFinish":
            game_id = event.get("game", {}).get("gameId", "")
            self.active_games.pop(game_id, None)

        elif etype in ("challengeDeclined", "challengeCanceled"):
            self._pending_challenge = None
            if self._should_seek():
                self._schedule_retry(5)


def main():
    parser = argparse.ArgumentParser(
        description="MetalFish Lichess Bot — plays continuously until stopped"
    )
    parser.add_argument("--accept-rated", action="store_true", default=True)
    parser.add_argument("--no-rated", dest="accept_rated", action="store_false")
    parser.add_argument("--accept-casual", action="store_true", default=True)
    parser.add_argument("--no-casual", dest="accept_casual", action="store_false")
    parser.add_argument(
        "--max-games", type=int, default=1, help="Max concurrent games (default: 1)"
    )
    parser.add_argument(
        "--seek",
        action="store_true",
        default=False,
        help="Actively challenge online bots",
    )
    parser.add_argument(
        "--tc",
        type=str,
        default=None,
        help="Fixed time control, e.g. '5+3', '3+2', '10+0'",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        default=False,
        help="Cycle through blitz/rapid/bullet time controls",
    )
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
