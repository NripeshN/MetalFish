#!/usr/bin/env python3
"""Lichess Bot: runs MetalFish Hybrid on Lichess via the Bot API.

Features:
  - Bounded opening book from Lichess masters database
  - Pondering by default, with hard stop budgets for clock safety
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
import queue
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

LOGICAL_CORES = os.cpu_count() or 4
SEARCH_WORKERS = max(3, LOGICAL_CORES - 1)
HYBRID_MCTS_THREADS = 1
HYBRID_AB_THREADS = max(1, SEARCH_WORKERS - HYBRID_MCTS_THREADS)
BOOK_MAX_PLY = 10
BOOK_MIN_CLOCK_MS = 30_000
BOOK_TIMEOUT_S = 0.25
ENGINE_STOP_GRACE_S = 1.0
PONDER_STOP_TIMEOUT_S = 1.0
PONDER_HIT_TIMEOUT_S = 2.0
MIN_SEARCH_TIMEOUT_S = 0.2
MAX_SEARCH_TIMEOUT_S = 30.0
CLOCK_SAFETY_MS = 1_500


def machine_memory_mb() -> int:
    if hasattr(os, "sysconf"):
        try:
            return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // (
                1024 * 1024
            )
        except (OSError, ValueError):
            pass
    return 0


def local_hash_mb() -> int:
    memory_mb = machine_memory_mb()
    if memory_mb <= 0:
        return 8192
    target = (memory_mb * 3) // 8
    return max(1024, min(16384, (target // 1024) * 1024))


ENGINE_OPTIONS = {
    "Threads": str(SEARCH_WORKERS),
    "Hash": str(local_hash_mb()),
    "Ponder": "false",
    "UseHybridSearch": "true",
    "NNWeights": str(WEIGHTS),
    "HybridMCTSThreads": str(HYBRID_MCTS_THREADS),
    "HybridABThreads": str(HYBRID_AB_THREADS),
    "MCTSMaxThreads": str(HYBRID_MCTS_THREADS),
    "Move Overhead": "500",
    "MCTSMinibatchSize": "0",
    "MCTSPolicySoftmaxTemp": "1.359",
    "SyzygyPath": str(PROJ / "syzygy"),
    "SyzygyProbeDepth": "2",
    "SyzygyProbeLimit": "6",
}

ROTATION_TCS = [
    (900, 10),  # 15+10 rapid
    (600, 5),  # 10+5 rapid
    (300, 3),  # 5+3 blitz
    (180, 2),  # 3+2 blitz
]

ZERO_INCREMENT_ROTATION_TCS = [
    (600, 0),  # 10+0 rapid
    (300, 0),  # 5+0 blitz
]

BULLET_ROTATION_TCS = [
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
    def __init__(
        self,
        path: pathlib.Path,
        options: dict,
        *,
        preload_transformer: bool = False,
    ):
        self.path = path
        self.options = dict(options)
        self.preload_transformer = preload_transformer
        self.proc: subprocess.Popen | None = None
        self._output: queue.Queue[str | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._pondering = False
        self._ponder_move: str | None = None
        self._launch()

    def _launch(self):
        env = os.environ.copy()
        if self.preload_transformer:
            env["METALFISH_PRELOAD_TRANSFORMER"] = "1"
        self.proc = subprocess.Popen(
            [str(self.path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        self._output = queue.Queue()
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()
        self._send("uci")
        self._wait_for("uciok")
        for name, value in self.options.items():
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok", timeout=120)
        self._pondering = False
        self._ponder_move: str | None = None

    def _read_stdout(self):
        try:
            if self.proc is None or self.proc.stdout is None:
                return
            for line in self.proc.stdout:
                self._output.put(line.strip())
        finally:
            self._output.put(None)

    def _send(self, cmd: str):
        try:
            if self.proc is None or self.proc.stdin is None:
                return
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _wait_for(self, prefix: str, timeout: float = 60) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is None:
                raise RuntimeError("Engine process is not running")
            remaining = max(0.0, deadline - time.time())
            try:
                line = self._output.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError("Engine process died")
                continue
            if line is None:
                raise RuntimeError("Engine closed stdout")
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"Timeout waiting for '{prefix}'")

    def _drain_available_output(self):
        while True:
            try:
                self._output.get_nowait()
            except queue.Empty:
                return

    def restart(self):
        self._pondering = False
        self._ponder_move = None
        if self.proc is not None:
            try:
                self._send("quit")
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
                except Exception:
                    pass
        self._launch()
        self.new_game()

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
        timeout: float = 600,
    ) -> tuple[str, str | None]:
        """Returns (bestmove, ponder_move_or_None)."""
        self._send("isready")
        self._wait_for("readyok")
        self._drain_available_output()

        self._send(
            self._go_command(
                wtime=wtime,
                btime=btime,
                winc=winc,
                binc=binc,
                movetime=movetime,
                movestogo=movestogo,
            )
        )
        try:
            line = self._wait_for("bestmove", timeout=timeout)
        except TimeoutError:
            self._send("stop")
            line = self._wait_for("bestmove", timeout=ENGINE_STOP_GRACE_S)
        parts = line.split()
        best = parts[1] if len(parts) > 1 else "0000"
        ponder = parts[3] if len(parts) > 3 and parts[2] == "ponder" else None
        return best, ponder

    def start_pondering(
        self,
        initial_fen: str,
        moves: list[str],
        ponder_move: str,
        *,
        wtime=None,
        btime=None,
        winc=None,
        binc=None,
    ):
        """Start thinking on opponent's time (go ponder)."""
        self.stop_pondering()
        all_moves = moves + [ponder_move]
        self.set_position(initial_fen, all_moves)
        self._drain_available_output()
        self._send(
            self._go_command(
                ponder=True,
                wtime=wtime,
                btime=btime,
                winc=winc,
                binc=binc,
            )
        )
        self._pondering = True
        self._ponder_move = ponder_move

    def _go_command(
        self,
        *,
        ponder: bool = False,
        wtime=None,
        btime=None,
        winc=None,
        binc=None,
        movetime=None,
        movestogo=None,
    ) -> str:
        parts = ["go"]
        if ponder:
            parts.append("ponder")
        if movetime is not None:
            parts.extend(["movetime", str(movetime)])
        else:
            if wtime is not None:
                parts.extend(["wtime", str(wtime)])
            if btime is not None:
                parts.extend(["btime", str(btime)])
            if winc is not None:
                parts.extend(["winc", str(winc)])
            if binc is not None:
                parts.extend(["binc", str(binc)])
            if movestogo is not None:
                parts.extend(["movestogo", str(movestogo)])
        return " ".join(parts)

    def ponderhit(self, timeout: float = PONDER_HIT_TIMEOUT_S) -> tuple[str, str | None]:
        """Opponent played the predicted move; use the current ponder result."""
        if not self._pondering:
            return "0000", None
        # The hybrid coordinator does not reliably finish promptly on UCI
        # ponderhit, so stop and use the best move found on opponent time.
        self._send("stop")
        try:
            line = self._wait_for("bestmove", timeout=timeout)
        finally:
            self._pondering = False
            self._ponder_move = None
        parts = line.split()
        best = parts[1] if len(parts) > 1 else "0000"
        ponder = parts[3] if len(parts) > 3 and parts[2] == "ponder" else None
        return best, ponder

    def stop_pondering(
        self,
        timeout: float = PONDER_STOP_TIMEOUT_S,
        *,
        restart_on_failure: bool = True,
    ) -> bool:
        """Stop pondering if active."""
        if not self._pondering:
            return True

        ok = True
        self._send("stop")
        try:
            self._wait_for("bestmove", timeout=timeout)
        except (TimeoutError, RuntimeError):
            ok = False
        self._pondering = False
        self._ponder_move = None

        if not ok and restart_on_failure:
            try:
                self.restart()
            except Exception:
                pass
        return ok

    @property
    def ponder_move(self) -> str | None:
        return self._ponder_move if self._pondering else None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def quit(self):
        self.stop_pondering(restart_on_failure=False)
        try:
            self._send("quit")
            if self.proc is not None:
                self.proc.wait(timeout=5)
        except Exception:
            try:
                if self.proc is not None:
                    self.proc.kill()
            except Exception:
                pass


class OpeningBook:
    def __init__(self, api_key: str, min_games: int = 5, timeout: float = 1.0):
        self.min_games = min_games
        self.timeout = timeout
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
                timeout=self.timeout,
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
                timeout=self.timeout,
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
        self._pending_challenge_id: str | None = None
        self._pending_challenge_target: str | None = None
        self._challenge_sent_at: float = 0
        self._challenge_retries = 0
        self.book = OpeningBook(
            api_key=api_key, min_games=5, timeout=BOOK_TIMEOUT_S
        )
        self._seek_timer: threading.Timer | None = None
        self._warm_engine: UCIEngine | None = None
        self._elo_widen_steps = 0
        self._declined_cooldown: dict[str, float] = {}  # bot_id -> timestamp
        self._rate_limit_count = 0
        self._tc_failures = 0
        self._seek_lock = threading.Lock()
        self._draining = threading.Event()
        self._shutdown = threading.Event()

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
            detail = r.text.strip().replace("\n", " ")[:200]
            suffix = f": {detail}" if detail else ""
            print(f"  [{game_id}] Move {move} failed: {r.status_code}{suffix}")
            return False
        return True

    def resign(self, game_id: str):
        self.api_post(f"/bot/game/{game_id}/resign")

    def abort_game(self, game_id: str) -> bool:
        r = self.api_post(f"/bot/game/{game_id}/abort")
        if r.status_code != 200:
            detail = r.text.strip().replace("\n", " ")[:200]
            suffix = f": {detail}" if detail else ""
            print(f"  [{game_id}] Abort failed: {r.status_code}{suffix}")
            return False
        return True

    def abort_or_resign(self, game_id: str):
        if not self.abort_game(game_id):
            self.resign(game_id)

    def _create_engine(self, *, preload_transformer: bool) -> UCIEngine:
        options = dict(ENGINE_OPTIONS)
        options["Ponder"] = "true" if self.args.ponder else "false"
        return UCIEngine(
            ENGINE,
            options,
            preload_transformer=preload_transformer,
        )

    def _prepare_warm_engine(self):
        if not self.args.prewarm_engine or self._warm_engine is not None:
            return
        print("  Preparing engine: transformer preload + NNUE replicas...")
        start = time.time()
        try:
            self._warm_engine = self._create_engine(preload_transformer=True)
        except Exception as e:
            print(f"  Engine warmup failed, will initialize on game start: {e}")
            self._warm_engine = None
            return
        elapsed = time.time() - start
        print(f"  Engine ready ({elapsed:.1f}s warmup)")

    def _acquire_engine(self) -> UCIEngine:
        if self._warm_engine is not None:
            engine = self._warm_engine
            self._warm_engine = None
            return engine
        return self._create_engine(preload_transformer=False)

    def _close_warm_engine(self):
        if self._warm_engine is None:
            return
        self._warm_engine.quit()
        self._warm_engine = None

    def _reserved_games(self) -> int:
        pending = 1 if self._pending_challenge_id else 0
        return len(self.active_games) + pending

    def _challenge_id_from_response(self, response: requests.Response) -> str | None:
        try:
            data = response.json()
        except ValueError:
            return None

        challenge = data.get("challenge") if isinstance(data, dict) else None
        if isinstance(challenge, dict) and challenge.get("id"):
            return str(challenge["id"])
        if isinstance(data, dict) and data.get("id"):
            return str(data["id"])
        return None

    def _clear_pending_challenge(self):
        self._pending_challenge_id = None
        self._pending_challenge_target = None
        self._challenge_sent_at = 0

    def _cancel_pending_challenge(self, reason: str):
        challenge_id = self._pending_challenge_id
        target = self._pending_challenge_target or challenge_id
        if not challenge_id:
            return

        if reason == "game started":
            self._clear_pending_challenge()
            return

        print(f"  Canceling challenge to {target} ({reason})")
        try:
            self.api_post(f"/challenge/{challenge_id}/cancel")
        except Exception as e:
            print(f"  Could not cancel challenge {challenge_id}: {e}")
        self._clear_pending_challenge()

    def _enter_drain_mode(self):
        if self._draining.is_set():
            return
        self._draining.set()
        print("\n  Drain requested: no new games will be started.")
        if self._seek_timer:
            self._seek_timer.cancel()
            self._seek_timer = None
        self._cancel_pending_challenge("drain requested")
        self._close_warm_engine()
        if not self.active_games:
            self._shutdown.set()

    def _start_stdin_watcher(self):
        if not sys.stdin.isatty():
            return

        def watch_stdin():
            try:
                while not self._shutdown.is_set():
                    ch = sys.stdin.read(1)
                    if ch == "":
                        self._enter_drain_mode()
                        return
            except Exception:
                return

        t = threading.Thread(target=watch_stdin, daemon=True)
        t.start()

    # ---- Seeking ----

    def seek_game(self):
        if not self._seek_lock.acquire(blocking=False):
            return
        try:
            self._seek_game_once()
        finally:
            self._seek_lock.release()

    def _seek_game_once(self):
        if not self._should_seek():
            return

        if self._pending_challenge_id and (
            time.time() - self._challenge_sent_at < CHALLENGE_TIMEOUT
        ):
            return

        if self._pending_challenge_id:
            target = self._pending_challenge_target
            self._cancel_pending_challenge("expired")
            self._cooldown_bot(target)

        limit, inc = self._next_tc()
        tc_label = f"{limit//60}+{inc}"
        speed = self._tc_to_speed(limit, inc)

        try:
            r = self.api_get("/bot/online", params={"nb": 100})
            if r.status_code != 200:
                self._schedule_retry()
                return

            bots: list[dict] = []
            now = time.time()
            for line in r.text.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    bot = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bot_id = bot.get("id", "")
                if not bot_id or bot_id == self.bot_id:
                    continue
                # Skip bots on cooldown
                cooldown_until = self._declined_cooldown.get(bot_id, 0)
                if now < cooldown_until:
                    continue
                bots.append(bot)

            if not bots:
                print("  No eligible bots online, retrying in 30s...")
                self._schedule_retry(30)
                return

            candidates = self._filter_bots_by_elo(bots, speed)
            if not candidates:
                candidates = [b.get("id", "") for b in bots]
                random.shuffle(candidates)

            # Challenge ONE bot at a time, wait for response
            target = candidates[0]
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
                challenge_id = self._challenge_id_from_response(r)
                if not challenge_id:
                    print(f"  Could not read challenge id for {target}")
                    self._cooldown_bot(target, duration=60)
                    self._schedule_retry(5)
                    return
                self._pending_challenge_id = challenge_id
                self._pending_challenge_target = target
                self._challenge_sent_at = time.time()
                self._challenge_retries = 0
                self._rate_limit_count = 0
                print(
                    f"  Challenged {target} ({tc_label}, {'rated' if rated else 'casual'})"
                )
                self._schedule_challenge_timeout()
            elif r.status_code == 429:
                print("  Rate limited, backing off...")
                self._rate_limit_count += 1
                if self._rate_limit_count >= 3:
                    # Likely hit daily challenge cap — back off aggressively
                    backoff = 900  # 15 minutes
                    print(f"  Possible daily cap hit. Waiting {backoff//60}min before retrying.")
                else:
                    backoff = 90  # Lichess docs say "wait a full minute"
                    print(f"  Waiting {backoff}s...")
                self._schedule_retry(backoff)
            else:
                # Bot doesn't accept — cooldown and try another
                self._cooldown_bot(target, duration=300)
                self._schedule_retry(2)
        except Exception as e:
            print(f"  Seek error: {e}")
            self._schedule_retry(15)

    def _cooldown_bot(self, bot_id: str | None, duration: int = 600):
        """Put a bot on cooldown so we don't re-challenge it."""
        if bot_id:
            self._declined_cooldown[bot_id] = time.time() + duration

    def _cleanup_cooldowns(self):
        """Remove expired cooldowns."""
        now = time.time()
        self._declined_cooldown = {
            k: v for k, v in self._declined_cooldown.items() if v > now
        }

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
        if self._pending_challenge_id:
            target = self._pending_challenge_target or self._pending_challenge_id
            print(f"  Challenge to {target} timed out")
            self._cooldown_bot(target, duration=600)
            self._cancel_pending_challenge("timeout")
            self._challenge_retries += 1
            self._tc_failures += 1
            if self._tc_failures >= 3 and self.args.rotate:
                print(f"  TC not working, trying next format...")
                self._advance_rotation()
                self._tc_failures = 0
                self._challenge_retries = 0
                self._cleanup_cooldowns()
            if self._challenge_retries < MAX_CHALLENGE_RETRIES and self._should_seek():
                self.seek_game()
            elif self._should_seek():
                self._challenge_retries = 0
                self._cleanup_cooldowns()
                self._schedule_retry(15)

    def _next_tc(self) -> tuple[int, int]:
        if self.args.tc:
            return parse_tc(self.args.tc)
        if self.args.rotate:
            rotation_tcs = self._rotation_tcs()
            tc = rotation_tcs[self._rotation_idx % len(rotation_tcs)]
            return tc
        return (300, 3)

    def _rotation_tcs(self) -> list[tuple[int, int]]:
        rotation_tcs = list(ROTATION_TCS)
        if self.args.include_zero_increment:
            rotation_tcs.extend(ZERO_INCREMENT_ROTATION_TCS)
        if self.args.include_bullet:
            rotation_tcs.extend(BULLET_ROTATION_TCS)
        return rotation_tcs

    def _advance_rotation(self):
        """Advance to next TC. Call only after a game starts successfully."""
        rotation_tcs = self._rotation_tcs()
        self._rotation_idx = (self._rotation_idx + 1) % len(rotation_tcs)

    def _tc_to_speed(self, limit: int, inc: int) -> str:
        estimated_duration = limit + 40 * inc
        if estimated_duration < 120:
            return "bullet"
        elif estimated_duration < 480:
            return "blitz"
        elif estimated_duration < 1500:
            return "rapid"
        return "classical"

    def _bot_plays_speed(self, bot: dict, speed: str) -> bool:
        """Check if bot has played games in this speed (non-provisional)."""
        perfs = bot.get("perfs", {})
        perf = perfs.get(speed, {})
        games = perf.get("games", 0)
        return games >= 5 and not perf.get("prov", False)

    def _get_bot_rating(self, bot: dict, speed: str) -> int | None:
        perfs = bot.get("perfs", {})
        perf = perfs.get(speed, {})
        rating = perf.get("rating")
        if rating and not perf.get("prov", False):
            return rating
        # Fall back: try any available rating
        for s in ("blitz", "rapid", "bullet", "classical"):
            p = perfs.get(s, {})
            if p.get("rating") and not p.get("prov", False):
                return p["rating"]
        return None

    def _filter_bots_by_elo(self, bots: list[dict], speed: str) -> list[str]:
        """Filter bots that actually play the target speed, sorted by Elo proximity."""
        if not self.args.elo_seek:
            # Still filter by speed activity even without elo-seek
            active = [b for b in bots if self._bot_plays_speed(b, speed)]
            if not active:
                active = bots
            ids = [b.get("id", "") for b in active]
            random.shuffle(ids)
            return ids

        our_rating = self._our_rating(speed)
        elo_range = self._current_elo_range()

        scored: list[tuple[int, str]] = []
        for bot in bots:
            bot_id = bot.get("id", "")
            if not bot_id:
                continue
            if not self._bot_plays_speed(bot, speed):
                continue
            rating = self._get_bot_rating(bot, speed)
            if rating is None:
                continue
            diff = abs(rating - our_rating)
            if diff <= elo_range:
                scored.append((diff, bot_id))

        if not scored:
            self._widen_elo_range()
            active = [b for b in bots if self._bot_plays_speed(b, speed)]
            if not active:
                active = bots
            ids = [b.get("id", "") for b in active]
            random.shuffle(ids)
            return ids

        scored.sort(key=lambda x: x[0])
        # Slight randomization among close-rated bots
        top = scored[:15]
        random.shuffle(top)
        return [bot_id for _, bot_id in top]

    def _our_rating(self, speed: str) -> int:
        """Get our bot's rating for the given speed."""
        if not hasattr(self, "_cached_ratings"):
            self._cached_ratings = {}
            try:
                profile = self.get_profile()
                perfs = profile.get("perfs", {})
                for s in ("bullet", "blitz", "rapid", "classical"):
                    p = perfs.get(s, {})
                    if p.get("rating"):
                        self._cached_ratings[s] = p["rating"]
            except Exception:
                pass
        return self._cached_ratings.get(speed, self.args.elo_target or 1500)

    def _current_elo_range(self) -> int:
        """Returns the current Elo range, widening after failed attempts."""
        base = self.args.elo_range if self.args.elo_range else 200
        return base + self._elo_widen_steps * 100

    def _widen_elo_range(self):
        self._elo_widen_steps += 1
        new_range = self._current_elo_range()
        print(f"  Widening Elo range to ±{new_range}")

    def _reset_elo_range(self):
        self._elo_widen_steps = 0

    def _should_seek(self) -> bool:
        return (
            self.args.seek
            and not self._draining.is_set()
            and self._reserved_games() < self.args.max_games
        )

    # ---- Challenge acceptance ----

    def should_accept(self, challenge: dict) -> bool:
        if self._draining.is_set():
            return False
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
        tc = ch.get("timeControl", {})
        if tc.get("type") == "clock":
            increment = int(tc.get("increment", 0) or 0)
            if speed == "bullet" and not self.args.include_bullet:
                return False
            if increment == 0 and not self.args.include_zero_increment:
                return False
        if self._reserved_games() >= self.args.max_games:
            return False
        return True

    # ---- Game play ----

    def play_game(self, game_id: str):
        print(f"  [{game_id}] Starting...")
        engine = None
        try:
            engine = self._acquire_engine()
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
            if self._draining.is_set():
                if not self.active_games:
                    self._shutdown.set()
            elif self._should_seek():
                self._prepare_warm_engine()
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
                        print(f"  [{game_id}] Game already ended: {status}")
                        break
                    moves = state.get("moves", "").split() if state.get("moves") else []
                    self._try_move(game_id, engine, initial_fen, moves, my_color, state)

                elif etype == "gameState":
                    status = event.get("status", "started")
                    if status != "started":
                        moves = event.get("moves", "").split() if event.get("moves") else []
                        winner = event.get("winner", "")
                        result = f"{status}"
                        if winner:
                            result += f" ({winner} wins)"
                        print(f"  [{game_id}] Game over: {result}, {len(moves)} moves")
                        break
                    moves = event.get("moves", "").split() if event.get("moves") else []
                    self._try_move(game_id, engine, initial_fen, moves, my_color, event)

                elif etype == "chatLine":
                    pass

    def _get_my_color(self, game_full: dict) -> str:
        white_id = game_full.get("white", {}).get("id", "")
        return "white" if white_id == self.bot_id else "black"

    def _last_move_matches_ponder(
        self, game_id: str, initial_fen: str, moves: list[str], ponder_move: str
    ) -> bool:
        if not moves:
            return False
        if moves[-1] == ponder_move:
            return True
        previous = self._build_board(game_id, initial_fen, moves[:-1])
        if previous is None:
            return False
        parsed = self._normalize_uci_move(moves[-1], previous)
        return parsed is not None and parsed.uci() == ponder_move

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
            # Keep any active ponder search running while the opponent is on move.
            return

        board = self._build_board(game_id, initial_fen, moves)
        if board is None:
            return

        wtime, btime, winc, binc = self._clock_values(state)
        search_timeout = self._search_timeout_seconds(
            my_color, wtime, btime, winc, binc
        )
        ponder_stop_timeout = self._ponder_stop_timeout_seconds(
            my_color, wtime, btime, winc, binc
        )

        # My turn: if the opponent played the predicted ponder move, convert that
        # search before doing any new book/engine work.
        if engine.ponder_move:
            if self._last_move_matches_ponder(
                game_id, initial_fen, moves, engine.ponder_move
            ):
                try:
                    best, ponder = engine.ponderhit(timeout=ponder_stop_timeout)
                except (TimeoutError, RuntimeError) as e:
                    print(f"  [{game_id}] Ponder stop failed: {e}; restarting")
                    engine.restart()
                    best, ponder = "0000", None
                if best and best not in ("0000", "(none)"):
                    parsed = self._parse_legal_move(game_id, best, board, "ponder")
                    if parsed is None:
                        print(
                            f"  [{game_id}] Restarting after rejected ponder move"
                        )
                        engine.restart()
                    elif self.make_move(game_id, parsed.uci()):
                        move_uci = parsed.uci()
                        print(f"  [{game_id}] Ponderhit! {move_uci}")
                        self._start_pondering_if_legal(
                            game_id,
                            engine,
                            initial_fen,
                            moves,
                            board,
                            move_uci,
                            ponder,
                            wtime=wtime,
                            btime=btime,
                            winc=winc,
                            binc=binc,
                        )
                        return
                    else:
                        return
            else:
                if not engine.stop_pondering(timeout=ponder_stop_timeout):
                    print(f"  [{game_id}] Ponder stop timed out; engine restarted")

        # Opening book. Explorer calls are remote and count against our clock on
        # Lichess, so only use them early while there is enough clock cushion.
        if self._should_query_book(board, my_color, wtime, btime):
            book_move = self.book.lookup(board.fen())
            if book_move:
                parsed = self._parse_legal_move(
                    game_id, book_move, board, "book"
                )
                if parsed is not None:
                    if self.make_move(game_id, parsed.uci()):
                        move_uci = parsed.uci()
                        print(f"  [{game_id}] Book: {move_uci}")
                        self._start_book_pondering(
                            game_id,
                            engine,
                            initial_fen,
                            moves,
                            board,
                            move_uci,
                            wtime=wtime,
                            btime=btime,
                            winc=winc,
                            binc=binc,
                        )
                    return

        # Engine search
        if not engine.alive():
            print(f"  [{game_id}] Engine died, restarting")
            try:
                engine.restart()
            except Exception as e:
                print(f"  [{game_id}] Engine restart failed: {e}")
                fallback = self._fallback_move(board)
                if fallback and self.make_move(game_id, fallback):
                    print(f"  [{game_id}] Fallback legal move: {fallback}")
                return

        if not engine.stop_pondering(timeout=ponder_stop_timeout):
            print(f"  [{game_id}] Ponder stop timed out; engine restarted")
        engine.set_position(initial_fen, moves)

        try:
            search_start = time.time()
            best, ponder = engine.go(
                wtime=wtime,
                btime=btime,
                winc=winc,
                binc=binc,
                timeout=search_timeout,
            )
            search_ms = int((time.time() - search_start) * 1000)
            if best == "0000" or best == "(none)":
                fallback = self._fallback_move(board)
                if not fallback:
                    return
                best, ponder = fallback, None
            if not self._is_legal_move(best, board):
                remaining_wtime, remaining_btime = self._after_search_clock(
                    my_color, wtime, btime, search_ms
                )
                best, ponder = self._recover_engine_move(
                    game_id,
                    engine,
                    initial_fen,
                    moves,
                    board,
                    best,
                    my_color=my_color,
                    wtime=remaining_wtime,
                    btime=remaining_btime,
                )
                if not best:
                    return
            parsed = self._parse_legal_move(game_id, best, board, "engine")
            if parsed is not None and self.make_move(game_id, parsed.uci()):
                move_uci = parsed.uci()
                print(f"  [{game_id}] Engine: {move_uci}")
                ponder_wtime, ponder_btime = self._after_search_clock(
                    my_color, wtime, btime, search_ms
                )
                self._start_pondering_if_legal(
                    game_id,
                    engine,
                    initial_fen,
                    moves,
                    board,
                    move_uci,
                    ponder,
                    wtime=ponder_wtime,
                    btime=ponder_btime,
                    winc=winc,
                    binc=binc,
                )
        except (TimeoutError, RuntimeError) as e:
            print(f"  [{game_id}] Engine error: {e}; restarting and recovering")
            best, _ = self._recover_engine_move(
                game_id,
                engine,
                initial_fen,
                moves,
                board,
                f"error: {e}",
                my_color=my_color,
                wtime=wtime,
                btime=btime,
            )
            if not best:
                return
            parsed = self._parse_legal_move(game_id, best, board, "recovery")
            if parsed is not None and self.make_move(game_id, parsed.uci()):
                print(f"  [{game_id}] Recovery: {parsed.uci()}")

    def _build_board(
        self, game_id: str, initial_fen: str, moves: list[str]
    ) -> chess.Board | None:
        try:
            board = (
                chess.Board(initial_fen)
                if initial_fen != "startpos"
                else chess.Board()
            )
        except ValueError as e:
            print(f"  [{game_id}] Bad initial FEN from stream: {e}")
            return None

        for ply, move in enumerate(moves, start=1):
            parsed = self._normalize_uci_move(move, board)
            if parsed is None:
                print(f"  [{game_id}] Bad stream move {move} at ply {ply}")
                return None
            board.push(parsed)
        return board

    def _normalize_uci_move(
        self, move: str, board: chess.Board
    ) -> chess.Move | None:
        try:
            parsed = chess.Move.from_uci(move)
        except ValueError:
            return None
        if parsed in board.legal_moves:
            return self._canonical_castle_move(parsed, board)

        return self._normalize_lichess_castle(parsed, board)

    def _canonical_castle_move(
        self, move: chess.Move, board: chess.Board
    ) -> chess.Move:
        if not board.is_castling(move):
            return move

        kingside = move.to_square > move.from_square
        for candidate in board.legal_moves:
            if candidate.from_square != move.from_square:
                continue
            if not board.is_castling(candidate):
                continue
            if (candidate.to_square > candidate.from_square) == kingside:
                return candidate
        return move

    def _normalize_lichess_castle(
        self, move: chess.Move, board: chess.Board
    ) -> chess.Move | None:
        piece = board.piece_at(move.from_square)
        target = board.piece_at(move.to_square)
        if not piece or piece.piece_type != chess.KING or piece.color != board.turn:
            return None
        if (
            not target
            or target.piece_type != chess.ROOK
            or target.color != board.turn
        ):
            return None
        if chess.square_rank(move.from_square) != chess.square_rank(move.to_square):
            return None

        kingside = move.to_square > move.from_square
        for candidate in board.legal_moves:
            if candidate.from_square != move.from_square:
                continue
            if not board.is_castling(candidate):
                continue
            if (candidate.to_square > candidate.from_square) == kingside:
                return candidate
        return None

    def _is_legal_move(self, move: str, board: chess.Board) -> bool:
        return self._normalize_uci_move(move, board) is not None

    def _parse_legal_move(
        self, game_id: str, move: str, board: chess.Board, source: str
    ) -> chess.Move | None:
        parsed = self._normalize_uci_move(move, board)
        if parsed is None:
            try:
                chess.Move.from_uci(move)
            except ValueError:
                print(f"  [{game_id}] Ignoring malformed {source} move {move}")
                return None
            print(
                f"  [{game_id}] Ignoring illegal {source} move {move} "
                f"for FEN: {board.fen()}"
            )
            return None
        return parsed

    def _clock_values(self, state: dict) -> tuple[int, int, int, int]:
        wtime = state.get("wtime", 60000)
        btime = state.get("btime", 60000)
        winc = state.get("winc", 0)
        binc = state.get("binc", 0)

        if not isinstance(wtime, int):
            wtime = 300000
        if not isinstance(btime, int):
            btime = 300000
        if not isinstance(winc, int):
            winc = 0
        if not isinstance(binc, int):
            binc = 0
        return wtime, btime, winc, binc

    def _should_query_book(
        self, board: chess.Board, my_color: str, wtime: int, btime: int
    ) -> bool:
        if board.ply() >= BOOK_MAX_PLY:
            return False
        our_time = wtime if my_color == "white" else btime
        return our_time >= BOOK_MIN_CLOCK_MS

    def _after_search_clock(
        self, my_color: str, wtime: int, btime: int, elapsed_ms: int
    ) -> tuple[int, int]:
        if my_color == "white":
            wtime = max(1, wtime - elapsed_ms)
        else:
            btime = max(1, btime - elapsed_ms)
        return wtime, btime

    def _our_clock(
        self, my_color: str, wtime: int, btime: int, winc: int, binc: int
    ) -> tuple[int, int]:
        if my_color == "white":
            return wtime, winc
        return btime, binc

    def _search_timeout_seconds(
        self, my_color: str, wtime: int, btime: int, winc: int, binc: int
    ) -> float:
        our_time, our_inc = self._our_clock(my_color, wtime, btime, winc, binc)
        if our_time <= 0:
            return MIN_SEARCH_TIMEOUT_S

        reserve = CLOCK_SAFETY_MS if our_time >= 15_000 else 2_000
        spendable = max(MIN_SEARCH_TIMEOUT_S * 1000, our_time - reserve)

        if our_inc > 0:
            budget = our_time / 30.0 + our_inc * 0.75
        else:
            budget = our_time / 22.0

        if our_time < 10_000:
            budget = min(budget, our_time * 0.35)
        elif our_time < 60_000:
            budget = min(budget, our_time * 0.25)
        else:
            budget = min(budget, our_time * 0.14)

        budget = min(budget, spendable, MAX_SEARCH_TIMEOUT_S * 1000)
        return max(MIN_SEARCH_TIMEOUT_S, budget / 1000.0)

    def _ponder_stop_timeout_seconds(
        self, my_color: str, wtime: int, btime: int, winc: int, binc: int
    ) -> float:
        our_time, _ = self._our_clock(my_color, wtime, btime, winc, binc)
        spendable = max(MIN_SEARCH_TIMEOUT_S, (our_time - CLOCK_SAFETY_MS) / 1000.0)
        return min(PONDER_HIT_TIMEOUT_S, spendable)

    def _start_pondering_if_legal(
        self,
        game_id: str,
        engine: UCIEngine,
        initial_fen: str,
        moves: list[str],
        board: chess.Board,
        best: str,
        ponder: str | None,
        *,
        wtime: int,
        btime: int,
        winc: int,
        binc: int,
    ):
        if not ponder:
            return
        if not self.args.ponder:
            return

        parsed_best = self._normalize_uci_move(best, board)
        if parsed_best is None:
            return
        board_after = board.copy(stack=False)
        board_after.push(parsed_best)
        parsed_ponder = self._normalize_uci_move(ponder, board_after)
        if parsed_ponder is None:
            print(
                f"  [{game_id}] Ignoring illegal ponder move {ponder} "
                f"after {parsed_best.uci()}"
            )
            return
        print(f"  [{game_id}] Ponder: {parsed_ponder.uci()}")
        engine.start_pondering(
            initial_fen,
            moves + [parsed_best.uci()],
            parsed_ponder.uci(),
            wtime=wtime,
            btime=btime,
            winc=winc,
            binc=binc,
        )

    def _start_book_pondering(
        self,
        game_id: str,
        engine: UCIEngine,
        initial_fen: str,
        moves: list[str],
        board: chess.Board,
        best: str,
        *,
        wtime: int,
        btime: int,
        winc: int,
        binc: int,
    ):
        parsed_best = self._normalize_uci_move(best, board)
        if parsed_best is None:
            return
        board_after = board.copy(stack=False)
        board_after.push(parsed_best)
        if not self.args.ponder:
            return
        reply = self.book.lookup(board_after.fen())
        parsed_reply = self._normalize_uci_move(reply, board_after) if reply else None
        if parsed_reply is None:
            return
        print(f"  [{game_id}] Book ponder: {parsed_reply.uci()}")
        engine.start_pondering(
            initial_fen,
            moves + [parsed_best.uci()],
            parsed_reply.uci(),
            wtime=wtime,
            btime=btime,
            winc=winc,
            binc=binc,
        )

    def _recover_engine_move(
        self,
        game_id: str,
        engine: UCIEngine,
        initial_fen: str,
        moves: list[str],
        board: chess.Board,
        illegal_move: str,
        *,
        my_color: str,
        wtime: int,
        btime: int,
    ) -> tuple[str | None, str | None]:
        print(
            f"  [{game_id}] Engine returned illegal move {illegal_move} "
            f"for FEN: {board.fen()}; restarting engine and retrying once"
        )
        try:
            engine.restart()
            engine.set_position(initial_fen, moves)
            our_time = wtime if my_color == "white" else btime
            movetime = max(100, min(1000, our_time // 20))
            best, ponder = engine.go(
                movetime=movetime,
                timeout=movetime / 1000.0 + ENGINE_STOP_GRACE_S,
            )
            parsed = self._normalize_uci_move(best, board)
            if best not in ("0000", "(none)") and parsed is not None:
                print(f"  [{game_id}] Retry move: {parsed.uci()}")
                return parsed.uci(), ponder
            print(f"  [{game_id}] Retry also returned illegal move: {best}")
        except (TimeoutError, RuntimeError) as e:
            print(f"  [{game_id}] Engine retry failed: {e}")

        fallback = self._fallback_move(board)
        if fallback:
            print(f"  [{game_id}] Fallback legal move: {fallback}")
            return fallback, None
        return None, None

    def _fallback_move(self, board: chess.Board) -> str | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }

        def score(move: chess.Move) -> int:
            total = 0
            if move.promotion:
                total += piece_values.get(move.promotion, 0)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim:
                    total += piece_values.get(victim.piece_type, 0)
                if attacker:
                    total -= piece_values.get(attacker.piece_type, 0) // 10
            board.push(move)
            if board.is_checkmate():
                total += 100000
            elif board.is_check():
                total += 50
            board.pop()
            return total

        return max(legal_moves, key=score).uci()

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

        if self.args.rotate:
            tc_mode = "rotate " + ", ".join(
                f"{limit // 60}+{inc}" for limit, inc in self._rotation_tcs()
            )
        else:
            tc_mode = self.args.tc or "5+3"
        hash_mb = ENGINE_OPTIONS["Hash"]
        print("=" * 60)
        print(f"  MetalFish Lichess Bot")
        print(f"  Account:  {self.username}")
        print(
            f"  Engine:   Hybrid (AB {HYBRID_AB_THREADS}T + "
            f"MCTS {HYBRID_MCTS_THREADS}T + GPU"
            f"{' + Ponder' if self.args.ponder else ''})"
        )
        print(
            f"  Workers:  {SEARCH_WORKERS} search + 1 coordinator "
            f"| CPU: {LOGICAL_CORES} logical"
        )
        print(f"  Hash:     {hash_mb} MB | Network: BT4-1024x15x32h")
        print(
            f"  Clock:    Move overhead {ENGINE_OPTIONS['Move Overhead']} ms "
            f"| hard cap {MAX_SEARCH_TIMEOUT_S:.0f}s"
        )
        print(
            f"  Book:     Lichess Masters DB "
            f"(ply < {BOOK_MAX_PLY}, clock >= {BOOK_MIN_CLOCK_MS // 1000}s)"
        )
        print(
            f"  Rated:    {self.args.accept_rated} | Casual: {self.args.accept_casual}"
        )
        print(f"  Seek:     {self.args.seek} | TC: {tc_mode}")
        print(f"  Max games: {self.args.max_games}")
        print("=" * 60)

        self._prepare_warm_engine()
        self._start_stdin_watcher()
        print("\nListening... (Ctrl+C to stop, Ctrl+D to drain after current game)\n")

        self._event_loop()

    def _event_loop(self):
        if self._should_seek():
            self.seek_game()

        while not self._shutdown.is_set():
            try:
                with self.api_get("/stream/event", stream=True, timeout=(10, 30)) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if self._shutdown.is_set():
                            break
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._handle_event(event)
                        if self._shutdown.is_set():
                            break

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ):
                if not self._shutdown.is_set():
                    print("  Connection lost, reconnecting in 3s...")
                    time.sleep(3)
            except requests.exceptions.ReadTimeout:
                pass  # normal — just reconnect
            except KeyboardInterrupt:
                print("\n\nShutting down...")
                if self._seek_timer:
                    self._seek_timer.cancel()
                    self._seek_timer = None
                self._cancel_pending_challenge("shutdown")
                self._close_warm_engine()
                for gid in list(self.active_games.keys()):
                    print(f"  Resigning {gid}...")
                    self.resign(gid)
                break
            except Exception as e:
                if not self._shutdown.is_set():
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
            if self._seek_timer:
                self._seek_timer.cancel()
                self._seek_timer = None
            if game_id in self.active_games:
                return
            self._cancel_pending_challenge("game started")
            self._reset_elo_range()
            self._tc_failures = 0
            if self._draining.is_set():
                print(f"  [{game_id}] Drain mode active, aborting new game")
                self.abort_or_resign(game_id)
                if not self.active_games:
                    self._shutdown.set()
                return
            if len(self.active_games) >= self.args.max_games:
                print(f"  [{game_id}] Max games reached, aborting overflow game")
                self.abort_or_resign(game_id)
                return
            self._advance_rotation()
            t = threading.Thread(target=self.play_game, args=(game_id,), daemon=True)
            self.active_games[game_id] = t
            t.start()

        elif etype == "gameFinish":
            if self._draining.is_set() and not self.active_games:
                self._shutdown.set()

        elif etype in ("challengeDeclined", "challengeCanceled"):
            target = self._pending_challenge_target
            self._cooldown_bot(target, duration=600)
            self._clear_pending_challenge()
            self._tc_failures += 1
            if self._tc_failures >= 5 and self.args.rotate:
                print(f"  TC getting too many declines, trying next format...")
                self._advance_rotation()
                self._tc_failures = 0
                self._cleanup_cooldowns()
            if self._should_seek():
                self._schedule_retry(2)


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
        help="Cycle through strength-first increment time controls",
    )
    parser.add_argument(
        "--include-zero-increment",
        action="store_true",
        default=False,
        help="Also rotate into 10+0 and 5+0 (weaker due flag risk)",
    )
    parser.add_argument(
        "--include-bullet",
        action="store_true",
        default=False,
        help="Also rotate into 2+1 bullet (weakest clock profile)",
    )
    parser.add_argument(
        "--no-prewarm",
        dest="prewarm_engine",
        action="store_false",
        default=True,
        help="Skip startup transformer preload (faster startup, weaker first search)",
    )
    parser.add_argument(
        "--ponder",
        dest="ponder",
        action="store_true",
        default=True,
        help="Enable UCI pondering (default)",
    )
    parser.add_argument(
        "--no-ponder",
        dest="ponder",
        action="store_false",
        help="Disable pondering",
    )
    parser.add_argument(
        "--elo-seek",
        action="store_true",
        default=False,
        help="Filter opponents by Elo proximity (starts tight, widens if no match)",
    )
    parser.add_argument(
        "--elo-range",
        type=int,
        default=200,
        help="Initial Elo range ± for opponent filtering (default: 200)",
    )
    parser.add_argument(
        "--elo-target",
        type=int,
        default=None,
        help="Target Elo to seek around (default: use bot's own rating)",
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
