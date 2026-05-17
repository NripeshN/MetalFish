#!/usr/bin/env python3
from __future__ import annotations

import io
import pathlib
import queue
import sys
import threading
import types
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.modules.setdefault(
    "requests",
    types.SimpleNamespace(get=None, post=None, Response=object),
)

from tools import lichess_bot  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_reader_uses_launch_queue() -> None:
    old_queue: "queue.Queue[str | None]" = queue.Queue()
    new_queue: "queue.Queue[str | None]" = queue.Queue()

    lichess_bot.UCIEngine._read_stdout(io.StringIO("uciok\n"), old_queue)

    expect("old line", old_queue.get_nowait() == "uciok")
    expect("old sentinel", old_queue.get_nowait() is None)
    expect("new queue untouched", new_queue.empty())


def test_live_defaults_avoid_crash_prone_resources() -> None:
    options, profile = lichess_bot.build_engine_options()

    expect("syzygy default disabled", "SyzygyPath" not in options)
    expect("resource reserve", lichess_bot.RESOURCE_RESERVE_MB >= 1024)
    expect("thread lower bound", int(options["Threads"]) >= 3)
    expect(
        "thread upper bound",
        int(options["Threads"]) <= lichess_bot.MAX_SEARCH_WORKERS,
    )
    expect("hash lower bound", int(options["Hash"]) >= 512)
    expect(
        "profile mirrors threads",
        int(profile["threads"]) == int(options["Threads"]),
    )


def test_engine_configure_applies_changed_options() -> None:
    engine = object.__new__(lichess_bot.UCIEngine)
    engine.options = {"Hash": "1024", "Threads": "8"}
    sent: list[str] = []

    def send(command: str) -> None:
        sent.append(command)

    def wait_for(prefix: str, timeout: float = 60) -> str:
        expect("configure ready wait", prefix == "readyok")
        return "readyok"

    engine.stop_pondering = lambda: True
    engine._send = send
    engine._wait_for = wait_for

    engine.configure({"Hash": "2048", "Threads": "8", "Ponder": "false"})

    expect(
        "changed options sent",
        sent
        == [
            "setoption name Hash value 2048",
            "setoption name Ponder value false",
            "isready",
        ],
    )
    expect("options updated", engine.options["Hash"] == "2048")


def test_ponderhit_stops_after_timeout() -> None:
    engine = object.__new__(lichess_bot.UCIEngine)
    engine._pondering = True
    engine._ponder_move = "e2e4"
    sent: list[str] = []
    calls = 0

    def send(command: str) -> None:
        sent.append(command)

    def wait_for(prefix: str, timeout: float = 60) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("ponderhit timeout")
        return "bestmove e7e5 ponder g1f3"

    engine._send = send
    engine._wait_for = wait_for

    best, ponder = engine.ponderhit(timeout=0.01)

    expect("ponderhit best", best == "e7e5")
    expect("ponderhit ponder", ponder == "g1f3")
    expect("ponderhit sends stop", sent == ["ponderhit", "stop"])
    expect("ponder state cleared", not engine._pondering)
    expect("ponder move cleared", engine._ponder_move is None)


def test_submitted_turn_guard() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    bot._submitted_turns = {}
    bot._submitted_turns_lock = threading.Lock()

    moves = ["e2e4", "c7c5"]
    expect("fresh turn", not bot._already_submitted_for_turn("game", moves))
    bot._record_submitted_turn("game", moves)
    expect("duplicate turn", bot._already_submitted_for_turn("game", moves))
    expect(
        "new turn",
        not bot._already_submitted_for_turn("game", moves + ["g1f3"]),
    )


def test_duplicate_game_state_does_not_resubmit() -> None:
    class Args:
        ponder = False

    class Book:
        def lookup(self, fen: str) -> None:
            return None

    class Engine:
        ponder_move = None

        def alive(self) -> bool:
            return True

        def stop_pondering(self, timeout: float = 0) -> bool:
            return True

        def set_position(self, initial_fen: str, moves: list[str]) -> None:
            return None

        def go(self, **kwargs) -> tuple[str, None]:
            return "e2e4", None

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.book = Book()
    bot._submitted_turns = {}
    bot._submitted_turns_lock = threading.Lock()
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()
    submitted: list[str] = []

    def make_move(game_id: str, move: str) -> bool:
        submitted.append(move)
        return True

    bot.make_move = make_move
    state = {"wtime": 900000, "btime": 900000, "winc": 10000, "binc": 10000}

    with redirect_stdout(io.StringIO()):
        bot._try_move("game", Engine(), "startpos", [], "white", state)
        bot._try_move("game", Engine(), "startpos", [], "white", state)

    expect("single submit", submitted == ["e2e4"])


def test_ponder_game_circuit_breaker() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()

    expect("ponder initially allowed", bot._ponder_allowed_for_game("game"))
    with redirect_stdout(io.StringIO()):
        bot._disable_ponder_for_game("game", "test")
    expect("ponder disabled", not bot._ponder_allowed_for_game("game"))


def test_quit_after_games_limit() -> None:
    class Args:
        quit_after_games = 1
        seek = True
        max_games = 1

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot._completed_games = 0
    bot._draining = threading.Event()
    bot._pending_challenge_id = None
    bot.active_games = {}

    expect("seek before completed limit", bot._should_seek())
    bot._completed_games = 1
    expect("no seek after completed limit", not bot._should_seek())


def main() -> int:
    test_reader_uses_launch_queue()
    test_live_defaults_avoid_crash_prone_resources()
    test_engine_configure_applies_changed_options()
    test_ponderhit_stops_after_timeout()
    test_submitted_turn_guard()
    test_duplicate_game_state_does_not_resubmit()
    test_ponder_game_circuit_breaker()
    test_quit_after_games_limit()
    print("Lichess bot tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
