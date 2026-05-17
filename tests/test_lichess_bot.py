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

    if "SyzygyPath" in options:
        expect(
            "syzygy path validated",
            lichess_bot.syzygy_path_is_safe(pathlib.Path(options["SyzygyPath"])),
        )
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


def test_engine_send_fails_fast_for_dead_process() -> None:
    engine = object.__new__(lichess_bot.UCIEngine)
    engine.proc = None
    engine._stderr_tail = []

    try:
        engine._send("isready")
    except RuntimeError as exc:
        expect("dead process diagnostic", "not running" in str(exc))
    else:
        raise AssertionError("dead process should raise")

    engine._send("quit", ignore_errors=True)


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
    later_moves = moves + ["g1f3", "d7d6"]
    bot._record_submitted_turn("game", later_moves)
    expect("later turn recorded", bot._already_submitted_for_turn("game", later_moves))
    expect("old turn retained", bot._already_submitted_for_turn("game", moves))


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


def test_stale_stream_ply_does_not_search_old_position() -> None:
    class Args:
        ponder = False

    class Book:
        def lookup(self, fen: str) -> None:
            return None

    class Engine:
        ponder_move = None

        def __init__(self) -> None:
            self.go_calls = 0

        def alive(self) -> bool:
            return True

        def stop_pondering(self, timeout: float = 0) -> bool:
            return True

        def set_position(self, initial_fen: str, moves: list[str]) -> None:
            return None

        def go(self, **kwargs) -> tuple[str, None]:
            self.go_calls += 1
            return "e2e4", None

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.book = Book()
    bot._submitted_turns = {}
    bot._max_seen_ply = {"game": 4}
    bot._submitted_turns_lock = threading.Lock()
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()
    submitted: list[str] = []

    bot.make_move = lambda game_id, move: submitted.append(move) or True
    engine = Engine()
    state = {"wtime": 900000, "btime": 900000, "winc": 10000, "binc": 10000}

    with redirect_stdout(io.StringIO()):
        bot._try_move("game", engine, "startpos", [], "white", state)

    expect("stale stream did not search", engine.go_calls == 0)
    expect("stale stream did not submit", submitted == [])


def test_stale_move_rejection_suppresses_duplicate_turn() -> None:
    class Args:
        ponder = False

    class Book:
        def lookup(self, fen: str) -> None:
            return None

    class Engine:
        ponder_move = None

        def __init__(self) -> None:
            self.go_calls = 0

        def alive(self) -> bool:
            return True

        def stop_pondering(self, timeout: float = 0) -> bool:
            return True

        def set_position(self, initial_fen: str, moves: list[str]) -> None:
            return None

        def go(self, **kwargs) -> tuple[str, None]:
            self.go_calls += 1
            return "e2e4", None

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.book = Book()
    bot._submitted_turns = {}
    bot._submitted_turns_lock = threading.Lock()
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()
    bot._last_move_failure_detail = ""
    submitted: list[str] = []

    def make_move(game_id: str, move: str) -> bool:
        submitted.append(move)
        bot._last_move_failure_detail = '400: {"error":"Not your turn"}'
        return False

    bot.make_move = make_move
    engine = Engine()
    state = {"wtime": 900000, "btime": 900000, "winc": 10000, "binc": 10000}

    with redirect_stdout(io.StringIO()):
        bot._try_move("game", engine, "startpos", [], "white", state)
        bot._try_move("game", engine, "startpos", [], "white", state)

    expect("single stale submit attempt", submitted == ["e2e4"])
    expect("duplicate stale turn skipped", engine.go_calls == 1)
    expect("stale rejected turn recorded", bot._already_submitted_for_turn("game", []))


def test_transient_move_rejection_remains_retryable() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    bot._submitted_turns = {}
    bot._submitted_turns_lock = threading.Lock()
    bot._last_move_failure_detail = "503: service unavailable"
    submitted: list[str] = []

    def make_move(game_id: str, move: str) -> bool:
        submitted.append(move)
        return False

    bot.make_move = make_move

    with redirect_stdout(io.StringIO()):
        ok = bot._submit_move("game", [], "e2e4")

    expect("transient submit failed", not ok)
    expect("transient attempted once", submitted == ["e2e4"])
    expect(
        "transient turn not recorded", not bot._already_submitted_for_turn("game", [])
    )


def test_ponder_game_circuit_breaker() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()

    expect("ponder initially allowed", bot._ponder_allowed_for_game("game"))
    with redirect_stdout(io.StringIO()):
        bot._disable_ponder_for_game("game", "test")
    expect("ponder disabled", not bot._ponder_allowed_for_game("game"))


def test_stale_challenge_event_preserves_current_pending() -> None:
    class Args:
        seek = True
        max_games = 1
        rotate = False

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.bot_id = "metalfish"
    bot.active_games = {}
    bot._completed_games = 0
    bot._pending_challenge_id = "new-challenge"
    bot._pending_challenge_target = "newbot"
    bot._challenge_sent_at = 0
    bot._tc_failures = 0
    bot._declined_cooldown = {}
    bot._draining = threading.Event()

    scheduled: list[float] = []
    bot._schedule_retry = lambda delay=10: scheduled.append(delay)

    event = {
        "type": "challengeCanceled",
        "challenge": {"id": "old-challenge", "destUser": {"id": "oldbot"}},
    }
    bot._handle_event(event)

    expect("current pending preserved", bot._pending_challenge_id == "new-challenge")
    expect("stale target cooled down", "oldbot" in bot._declined_cooldown)
    expect("stale event does not retry", scheduled == [])
    expect("stale event does not count tc failure", bot._tc_failures == 0)


def test_matching_challenge_event_clears_pending() -> None:
    class Args:
        seek = True
        max_games = 1
        rotate = False

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.bot_id = "metalfish"
    bot.active_games = {}
    bot._completed_games = 0
    bot._pending_challenge_id = "current"
    bot._pending_challenge_target = "targetbot"
    bot._challenge_sent_at = 0
    bot._tc_failures = 0
    bot._declined_cooldown = {}
    bot._draining = threading.Event()

    scheduled: list[float] = []
    bot._schedule_retry = lambda delay=10: scheduled.append(delay)

    event = {
        "type": "challengeDeclined",
        "challenge": {"id": "current", "destUser": {"id": "targetbot"}},
    }
    bot._handle_event(event)

    expect("pending cleared", bot._pending_challenge_id is None)
    expect("target cooled down", "targetbot" in bot._declined_cooldown)
    expect("retry scheduled", scheduled == [2])
    expect("tc failure counted", bot._tc_failures == 1)


def test_unrelated_challenge_event_without_pending_is_ignored() -> None:
    class Args:
        seek = True
        max_games = 1
        rotate = False

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.bot_id = "metalfish"
    bot.active_games = {}
    bot._completed_games = 0
    bot._pending_challenge_id = None
    bot._pending_challenge_target = None
    bot._challenge_sent_at = 0
    bot._tc_failures = 0
    bot._declined_cooldown = {}
    bot._draining = threading.Event()

    scheduled: list[float] = []
    bot._schedule_retry = lambda delay=10: scheduled.append(delay)

    event = {
        "type": "challengeCanceled",
        "challenge": {"id": "incoming", "challenger": {"id": "otherbot"}},
    }
    bot._handle_event(event)

    expect("no pending remains", bot._pending_challenge_id is None)
    expect("unrelated target cooled down", "otherbot" in bot._declined_cooldown)
    expect("unrelated event does not retry", scheduled == [])
    expect("unrelated event does not count tc failure", bot._tc_failures == 0)


def test_challenge_event_identity_accepts_string_users() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    bot.bot_id = "metalfish"

    challenge_id, target = bot._challenge_event_identity(
        {
            "type": "challengeDeclined",
            "challenge": {"id": "c1", "destUser": "targetbot"},
        }
    )

    expect("string challenge id parsed", challenge_id == "c1")
    expect("string target parsed", target == "targetbot")


def test_game_start_claims_slot_and_clears_pending() -> None:
    class Args:
        seek = True
        max_games = 1
        include_zero_increment = False
        include_bullet = False

    class Timer:
        canceled = False

        def cancel(self) -> None:
            self.canceled = True

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.active_games = {}
    bot._pending_challenge_id = "challenge"
    bot._pending_challenge_target = "targetbot"
    bot._challenge_sent_at = 123.0
    bot._rotation_idx = 0
    bot._tc_failures = 3
    bot._draining = threading.Event()
    bot._shutdown = threading.Event()
    bot._seek_timer = Timer()

    posted: list[str] = []
    played: list[str] = []
    bot.api_post = lambda path, **kwargs: posted.append(path)
    bot.play_game = lambda game_id: played.append(game_id)

    with redirect_stdout(io.StringIO()):
        bot._handle_event({"type": "gameStart", "game": {"gameId": "g1"}})

    thread = bot.active_games.get("g1")
    if thread is not None:
        thread.join(timeout=1)

    expect("pending cleared", bot._pending_challenge_id is None)
    expect("seek timer canceled", bot._seek_timer is None)
    expect("no challenge cancel for accepted game", posted == [])
    expect("game slot reserved", "g1" in bot.active_games)
    expect("game thread ran", played == ["g1"])
    expect("tc failures reset", bot._tc_failures == 0)
    expect("seek blocked by active game", not bot._should_seek())


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
    test_engine_send_fails_fast_for_dead_process()
    test_ponderhit_stops_after_timeout()
    test_submitted_turn_guard()
    test_duplicate_game_state_does_not_resubmit()
    test_stale_stream_ply_does_not_search_old_position()
    test_stale_move_rejection_suppresses_duplicate_turn()
    test_transient_move_rejection_remains_retryable()
    test_ponder_game_circuit_breaker()
    test_stale_challenge_event_preserves_current_pending()
    test_matching_challenge_event_clears_pending()
    test_unrelated_challenge_event_without_pending_is_ignored()
    test_challenge_event_identity_accepts_string_users()
    test_game_start_claims_slot_and_clears_pending()
    test_quit_after_games_limit()
    print("Lichess bot tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
