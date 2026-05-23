#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import pathlib
import queue
import sys
import tempfile
import threading
import time
import types
from contextlib import redirect_stderr, redirect_stdout

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


def test_runtime_ane_options_are_explicitly_opt_in() -> None:
    base_options, _ = lichess_bot.build_engine_options()
    disabled = types.SimpleNamespace(ponder=True, hybrid_ane_root_probe=False)
    enabled = types.SimpleNamespace(
        ponder=True,
        hybrid_ane_root_probe=True,
        hybrid_ane_weights=pathlib.Path("networks/t1.pb.gz"),
        hybrid_ane_model_path=pathlib.Path("build/coreml/t1.mlmodelc"),
        hybrid_ane_compute_units="cpu-ne",
        hybrid_ane_root_hint_count=10,
        hybrid_ane_root_hint_wait_ms=0,
        hybrid_ane_min_budget_ms=1000,
    )

    no_ane = lichess_bot.apply_runtime_engine_options(base_options, disabled)
    with_ane = lichess_bot.apply_runtime_engine_options(base_options, enabled)

    expect("ANE probe is opt-in", "HybridANERootProbe" not in no_ane)
    expect("ANE probe option enabled", with_ane["HybridANERootProbe"] == "true")
    expect("ANE weights passed", with_ane["HybridANEWeights"] == "networks/t1.pb.gz")
    expect(
        "ANE model passed", with_ane["HybridANEModelPath"] == "build/coreml/t1.mlmodelc"
    )
    expect("ANE wait uses retained profile", with_ane["HybridANERootHintWaitMs"] == "0")
    expect(
        "ANE min budget uses retained profile",
        with_ane["HybridANEMinBudgetMs"] == "1000",
    )


def test_verbose_runtime_enables_trace_without_forcing_ane_hints() -> None:
    base_options, _ = lichess_bot.build_engine_options()
    args = types.SimpleNamespace(
        ponder=True,
        verbose=True,
        hybrid_ane_root_probe=True,
        hybrid_ane_weights=pathlib.Path("networks/t1.pb.gz"),
        hybrid_ane_model_path=pathlib.Path("build/coreml/t1.mlmodelc"),
        hybrid_ane_compute_units="cpu-ne",
        hybrid_ane_root_hint_count=10,
        hybrid_ane_root_hint_wait_ms=0,
        hybrid_ane_min_budget_ms=1000,
    )

    options = lichess_bot.apply_runtime_engine_options(base_options, args)

    expect("verbose enables HybridTrace", options["HybridTrace"] == "true")
    expect("verbose does not force ANE root hints", "HybridANERootHints" not in options)
    expect("ANE root probe stays enabled", options["HybridANERootProbe"] == "true")


def test_verbose_path_enables_trace_and_resolves_log_path() -> None:
    base_options, _ = lichess_bot.build_engine_options()
    args = types.SimpleNamespace(
        ponder=True,
        verbose="results/lichess_verbose/run.log",
        hybrid_ane_root_probe=False,
    )

    options = lichess_bot.apply_runtime_engine_options(base_options, args)
    path = lichess_bot.verbose_log_path(args)

    expect("verbose path enables HybridTrace", options["HybridTrace"] == "true")
    expect("verbose path resolved", path is not None)
    expect("verbose path filename", path.name == "run.log")
    expect("verbose path absolute", path.is_absolute())


def test_verbose_log_tees_stdout_and_stderr() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "verbose.log"
        args = types.SimpleNamespace(verbose=str(path))

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            lichess_bot.install_verbose_log(args)
            try:
                print("stdout marker", flush=True)
                sys.stderr.write("stderr marker\n")
                sys.stderr.flush()
            finally:
                args._verbose_log_close()

        text = path.read_text()
        expect("stdout tee captured", "stdout marker" in text)
        expect("stderr tee captured", "stderr marker" in text)
        expect("verbose log announced", "Verbose log:" in text)


def test_verbose_uci_option_tracking_filters_useful_options() -> None:
    engine = object.__new__(lichess_bot.UCIEngine)
    engine.verbose = True
    engine.label = "test"
    engine._uci_options_seen = set()

    out = io.StringIO()
    with redirect_stdout(out):
        engine._record_uci_option(
            "option name HybridANERootProbe type check default false"
        )
        engine._verbose_engine_line(
            "option name HybridANERootProbe type check default false"
        )
        engine._debug_ane_uci_support()

    text = out.getvalue()
    expect("verbose prints useful option", "HybridANERootProbe" in text)
    expect("verbose reports missing ANE options", "ANE UCI options missing" in text)


def test_ane_runtime_values_are_clamped_to_uci_bounds() -> None:
    args = types.SimpleNamespace(
        hybrid_ane_root_hint_count=999,
        hybrid_ane_root_hint_wait_ms=999999,
        hybrid_ane_min_budget_ms=999999,
    )
    lichess_bot.normalize_ane_args(args)

    expect("ANE hint count upper bound", args.hybrid_ane_root_hint_count == 32)
    expect("ANE wait upper bound", args.hybrid_ane_root_hint_wait_ms == 1000)
    expect("ANE budget upper bound", args.hybrid_ane_min_budget_ms == 30000)

    args = types.SimpleNamespace(
        hybrid_ane_root_hint_count=-5,
        hybrid_ane_root_hint_wait_ms=-5,
        hybrid_ane_min_budget_ms=-5,
    )
    lichess_bot.normalize_ane_args(args)

    expect("ANE hint count lower bound", args.hybrid_ane_root_hint_count == 1)
    expect("ANE wait lower bound", args.hybrid_ane_root_hint_wait_ms == 0)
    expect("ANE budget lower bound", args.hybrid_ane_min_budget_ms == 0)


def test_ane_config_validation_requires_existing_files() -> None:
    args = types.SimpleNamespace(
        seek=False,
        accept_rated=False,
        accept_casual=True,
        elo_seek=False,
        seek_highest_rated=False,
        elo_range=200,
        min_rated_opponent_elo=2200,
        hybrid_ane_root_probe=True,
        hybrid_ane_weights=pathlib.Path("/missing/t1.pb.gz"),
        hybrid_ane_model_path=pathlib.Path("/missing/t1.mlmodelc"),
    )

    errors = lichess_bot.validate_bot_config(args)
    expect(
        "missing ANE weights rejected",
        any("ANE weights not found" in e for e in errors),
    )
    expect(
        "missing ANE model rejected",
        any("Core ML model not found" in e for e in errors),
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        args.hybrid_ane_weights = root / "t1.pb.gz"
        args.hybrid_ane_model_path = root / "t1.mlmodelc"
        args.hybrid_ane_weights.write_bytes(b"weights")
        args.hybrid_ane_model_path.mkdir()
        errors = lichess_bot.validate_bot_config(args)
        expect("existing ANE files accepted", errors == [])


def test_load_adjusted_workers_preserves_floor() -> None:
    expect(
        "idle workers unchanged",
        lichess_bot.load_adjusted_workers(8, 0.0) == 8,
    )
    expect(
        "busy load trims workers",
        lichess_bot.load_adjusted_workers(8, lichess_bot.LOAD_THROTTLE_HIGH) == 6,
    )
    expect(
        "extreme load halves workers",
        lichess_bot.load_adjusted_workers(8, lichess_bot.LOAD_THROTTLE_EXTREME) == 4,
    )
    expect(
        "worker floor preserved",
        lichess_bot.load_adjusted_workers(3, 99.0) == 3,
    )


def test_pre_game_resource_prep_runs_cleanup_before_allocation() -> None:
    old_resource_prep = lichess_bot.PRE_GAME_RESOURCE_PREP
    old_available = lichess_bot.available_memory_mb
    old_should_purge = lichess_bot.should_run_pre_game_purge
    old_purge = lichess_bot.run_pre_game_memory_purge
    old_collect = lichess_bot.gc.collect

    calls: list[str] = []
    memory_values = iter([1024, 4096])

    try:
        lichess_bot.PRE_GAME_RESOURCE_PREP = True
        lichess_bot.available_memory_mb = lambda: next(memory_values)

        def should_purge(available_mb: int) -> bool:
            calls.append(f"should:{available_mb}")
            return True

        def purge() -> bool:
            calls.append("purge")
            return True

        lichess_bot.should_run_pre_game_purge = should_purge
        lichess_bot.run_pre_game_memory_purge = purge
        lichess_bot.gc.collect = lambda: calls.append("gc") or 7

        bot = object.__new__(lichess_bot.LichessBot)
        out = io.StringIO()
        with redirect_stdout(out):
            bot._prepare_game_resources("g1")
    finally:
        lichess_bot.PRE_GAME_RESOURCE_PREP = old_resource_prep
        lichess_bot.available_memory_mb = old_available
        lichess_bot.should_run_pre_game_purge = old_should_purge
        lichess_bot.run_pre_game_memory_purge = old_purge
        lichess_bot.gc.collect = old_collect

    expect("cleanup order", calls == ["gc", "should:1024", "purge"])
    text = out.getvalue()
    expect("resource prep logs memory delta", "1024 MB -> 4096 MB" in text)
    expect("resource prep logs purge", "purge=ok" in text)


def test_bot_instance_lock_blocks_second_holder() -> None:
    if lichess_bot.fcntl is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        lock_path = pathlib.Path(tmp) / "lichess_bot.lock"
        first = lichess_bot.BotInstanceLock(lock_path)
        second = lichess_bot.BotInstanceLock(lock_path)
        first.acquire()
        try:
            blocked = False
            try:
                second.acquire()
            except RuntimeError as exc:
                blocked = "already running" in str(exc)
            expect("second bot instance blocked", blocked)
        finally:
            first.release()

        second.acquire()
        second.release()


def test_seek_rating_requires_rated_only_mode() -> None:
    class Args:
        accept_rated = False
        accept_casual = True

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    expect("casual default seek", not bot._seek_is_rated())

    bot.args.accept_rated = True
    expect("mixed mode still seeks casual", not bot._seek_is_rated())

    bot.args.accept_casual = False
    expect("rated-only mode seeks rated", bot._seek_is_rated())


def test_seek_and_rated_policy_labels_are_explicit() -> None:
    class Args:
        seek = True
        accept_rated = False
        accept_casual = True
        min_rated_opponent_elo = 2200
        elo_seek = False
        seek_highest_rated = False
        elo_range = 200

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot._elo_widen_steps = 0

    expect("casual seek label", bot._seek_policy_label() == "casual")
    expect("rated disabled label", bot._rated_policy_label() == "disabled")

    bot.args.accept_rated = True
    bot.args.elo_seek = True
    expect("mixed mode seeks casual", bot._seek_policy_label() == "casual")
    expect(
        "incoming rated label includes elo",
        bot._rated_policy_label() == ">= 2200, within +/-200 Elo",
    )

    bot.args.accept_casual = False
    expect(
        "rated seek label includes filters",
        bot._seek_policy_label() == "rated (>= 2200, within +/-200 Elo)",
    )

    bot.args.elo_seek = False
    bot.args.seek_highest_rated = True
    expect(
        "highest-rated seek label",
        bot._seek_policy_label() == "rated (>= 2200, highest-rated first)",
    )


def test_bot_config_rejects_dangerous_rated_seek() -> None:
    args = types.SimpleNamespace(
        seek=True,
        accept_rated=True,
        accept_casual=False,
        elo_seek=False,
        seek_highest_rated=False,
        elo_range=200,
        min_rated_opponent_elo=2200,
    )
    errors = lichess_bot.validate_bot_config(args)
    expect(
        "rated seek requires bounded mode",
        any("--elo-seek or --seek-highest-rated" in e for e in errors),
    )

    args.seek_highest_rated = True
    args.min_rated_opponent_elo = 0
    errors = lichess_bot.validate_bot_config(args)
    expect("rated seek requires floor", any("opponent-elo > 0" in e for e in errors))

    args.min_rated_opponent_elo = 2200
    errors = lichess_bot.validate_bot_config(args)
    expect("safe rated seek accepted", errors == [])


def test_bot_config_rejects_invalid_common_modes() -> None:
    args = types.SimpleNamespace(
        seek=False,
        accept_rated=False,
        accept_casual=False,
        elo_seek=False,
        seek_highest_rated=False,
        elo_range=200,
        min_rated_opponent_elo=2200,
    )
    errors = lichess_bot.validate_bot_config(args)
    expect("no game type rejected", any("No game type" in e for e in errors))

    args.accept_casual = True
    args.elo_seek = True
    args.elo_range = 0
    errors = lichess_bot.validate_bot_config(args)
    expect("bad elo range rejected", any("--elo-range" in e for e in errors))


def test_config_check_prints_without_api_or_engine_launch() -> None:
    args = types.SimpleNamespace(
        seek=True,
        accept_rated=False,
        accept_casual=True,
        elo_seek=False,
        elo_range=200,
        min_rated_opponent_elo=2200,
        ponder=True,
    )

    out = io.StringIO()
    with redirect_stdout(out):
        lichess_bot.print_config_check(args)

    text = out.getvalue()
    expect("config check title", "MetalFish Lichess Bot config check" in text)
    expect("config check seek", "Seek: casual" in text)
    expect("config check resources", "Resources:" in text)
    expect("config check syzygy", "Syzygy:" in text)


def test_should_seek_respects_resource_gate() -> None:
    class Args:
        quit_after_games = 0
        seek = True
        max_games = 1

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot._completed_games = 0
    bot._draining = threading.Event()
    bot._pending_challenge_id = None
    bot.active_games = {}
    expect("seek allowed with resources", bot._should_seek())

    scheduled: list[float] = []
    bot._schedule_retry = lambda delay=10: scheduled.append(delay)
    bot._resources_allow_new_game = lambda: False
    with redirect_stdout(io.StringIO()):
        bot._seek_game_once()
    expect("seek still eligible before resource gate", bot._should_seek())
    expect("busy resources reschedule seek", scheduled == [30])


def test_seek_dry_run_does_not_post_challenge() -> None:
    class Args:
        seek = True
        accept_rated = False
        accept_casual = True
        elo_seek = False
        tc = "5+3"
        rotate = False
        ponder = True

    class Response:
        status_code = 200
        text = '{"id":"targetbot","perfs":{"blitz":{"games":20,"rating":2400}}}\n'

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.bot_id = "metalfish"
    bot._declined_cooldown = {}
    bot._resources_allow_new_game = lambda: True
    bot.api_get = lambda path, params=None: Response()

    def api_post(path: str, **kwargs):
        raise AssertionError("dry-run must not post a challenge")

    bot.api_post = api_post

    out = io.StringIO()
    with redirect_stdout(out):
        ok = bot.preview_seek_once()

    expect("dry-run found target", ok)
    text = out.getvalue()
    expect("dry-run prints config first", text.startswith("MetalFish Lichess Bot"))
    expect("dry-run names target", "would challenge targetbot" in text)


def test_seek_dry_run_reports_rating_floor_block() -> None:
    class Args:
        seek = True
        accept_rated = True
        accept_casual = False
        elo_seek = False
        tc = "15+10"
        rotate = False
        min_rated_opponent_elo = 2200
        ponder = True

    class Response:
        status_code = 200
        text = '{"id":"lowbot","perfs":{"rapid":{"games":20,"rating":1500}}}\n'

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.bot_id = "metalfish"
    bot._declined_cooldown = {}
    bot._resources_allow_new_game = lambda: True
    bot.api_get = lambda path, params=None: Response()

    out = io.StringIO()
    with redirect_stdout(out):
        ok = bot.preview_seek_once()

    expect("dry-run blocks low rated target", not ok)
    text = out.getvalue()
    expect("dry-run prints config first", text.startswith("MetalFish Lichess Bot"))
    expect("dry-run reports rating floor", "rating floor" in text)


def test_seek_candidates_tolerate_malformed_perf_records() -> None:
    class Args:
        elo_seek = False

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bots = [
        {"id": "bad-perfs", "perfs": "not-a-dict"},
        {"id": "bad-speed", "perfs": {"rapid": "not-a-dict"}},
        {
            "id": "good",
            "perfs": {"rapid": {"games": "6", "rating": "2300", "prov": False}},
        },
    ]

    candidates, reason = bot._seek_candidates(bots, "rapid", rated=False)

    expect("malformed perf records skipped", reason is None)
    expect("valid active speed retained", candidates == ["good"])


def test_seek_candidates_filter_cached_cooldowns_case_insensitively() -> None:
    class Args:
        elo_seek = False
        seek_highest_rated = False
        avoid_repeat_format = False

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot._declined_cooldown = {}
    bot._cooldown_bot("TargetBot", duration=600)
    bots = [
        {
            "id": "targetbot",
            "perfs": {"rapid": {"games": 20, "rating": 2300, "prov": False}},
        },
        {
            "id": "otherbot",
            "perfs": {"rapid": {"games": 20, "rating": 2300, "prov": False}},
        },
    ]

    candidates, reason = bot._seek_candidates(bots, "rapid", rated=False)

    expect("cooldown candidate skipped", reason is None)
    expect("non-cooldown candidate remains", candidates == ["otherbot"])


def test_seek_candidates_filter_time_control_cooldown_by_speed_only() -> None:
    class Args:
        elo_seek = False
        seek_highest_rated = True
        avoid_repeat_format = False
        min_rated_opponent_elo = 0

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot._declined_cooldown = {}
    bot._speed_declined_cooldown = {}
    bot._cooldown_bot_speed("TargetBot", "rapid", duration=600)
    bots = [
        {"id": "targetbot", "perfs": {"rapid": {"games": 20, "rating": 2800}}},
        {"id": "freshbot", "perfs": {"rapid": {"games": 20, "rating": 2600}}},
    ]

    candidates, reason = bot._seek_candidates(bots, "rapid", rated=False)
    expect("rapid speed cooldown filters target", reason is None)
    expect("fresh rapid remains after speed cooldown", candidates == ["freshbot"])

    bots[0]["perfs"]["blitz"] = {"games": 20, "rating": 2800}
    bots[1]["perfs"]["blitz"] = {"games": 20, "rating": 2600}
    candidates, reason = bot._seek_candidates(bots, "blitz", rated=False)
    expect("speed cooldown does not filter blitz", reason is None)
    expect("target returns for different speed", candidates[0] == "targetbot")


def test_speed_challenge_cooldowns_persist_between_runs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "speed-cooldowns.json"
        old_path = lichess_bot.SPEED_CHALLENGE_COOLDOWN_PATH
        try:
            lichess_bot.SPEED_CHALLENGE_COOLDOWN_PATH = path

            bot = object.__new__(lichess_bot.LichessBot)
            bot._persist_challenge_cooldowns = True
            bot._speed_declined_cooldown = {}
            bot._cooldown_bot_speed("TargetBot", "rapid", duration=600)

            loaded_bot = object.__new__(lichess_bot.LichessBot)
            loaded = loaded_bot._load_speed_challenge_cooldowns()
            file_created = path.exists()
        finally:
            lichess_bot.SPEED_CHALLENGE_COOLDOWN_PATH = old_path

    expect("speed cooldown file created", file_created)
    expect("rapid cooldown persisted", "targetbot" in loaded.get("rapid", {}))
    expect("global speeds not invented", "blitz" not in loaded)


def test_highest_rated_seek_orders_candidates_descending() -> None:
    class Args:
        elo_seek = False
        seek_highest_rated = True
        avoid_repeat_format = False
        min_rated_opponent_elo = 2200

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot._declined_cooldown = {}
    bots = [
        {"id": "mid", "perfs": {"rapid": {"games": 20, "rating": 2500}}},
        {"id": "top", "perfs": {"rapid": {"games": 20, "rating": 2850}}},
        {"id": "low", "perfs": {"rapid": {"games": 20, "rating": 2250}}},
        {"id": "inactive", "perfs": {"rapid": {"games": 1, "rating": 3000}}},
    ]

    candidates, reason = bot._seek_candidates(bots, "rapid", rated=True)

    expect("highest-rated seek has candidates", reason is None)
    expect("highest-rated order", candidates == ["top", "mid", "low"])


def test_avoid_repeat_format_filters_only_matching_speed() -> None:
    class Args:
        elo_seek = False
        seek_highest_rated = True
        avoid_repeat_format = True
        min_rated_opponent_elo = 0

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot._declined_cooldown = {}
    bot._played_by_speed = {"rapid": {"targetbot": 1.0}}
    bot._persist_played_format_history = False
    bots = [
        {"id": "targetbot", "perfs": {"rapid": {"games": 20, "rating": 2800}}},
        {"id": "freshbot", "perfs": {"rapid": {"games": 20, "rating": 2600}}},
    ]

    candidates, reason = bot._seek_candidates(bots, "rapid", rated=False)
    expect("repeat rapid filtered", reason is None)
    expect("fresh rapid remains", candidates == ["freshbot"])

    bots[0]["perfs"]["blitz"] = {"games": 20, "rating": 2800}
    bots[1]["perfs"]["blitz"] = {"games": 20, "rating": 2600}
    candidates, reason = bot._seek_candidates(bots, "blitz", rated=False)
    expect("same bot allowed in new format", reason is None)
    expect("blitz highest still includes target", candidates[0] == "targetbot")


def test_online_bots_uses_documented_fetch_limit() -> None:
    class Response:
        status_code = 200
        text = '{"id":"targetbot","perfs":{"rapid":{"games":20,"rating":2400}}}\n'

    bot = object.__new__(lichess_bot.LichessBot)
    bot.bot_id = "metalfish"
    bot._online_bots_cache = None
    calls: list[dict] = []

    def api_get(path: str, **kwargs):
        calls.append({"path": path, **kwargs})
        return Response()

    bot.api_get = api_get

    bots, status = bot._online_bots()

    expect("online bots request ok", status is None)
    expect("online bot parsed", [bot["id"] for bot in bots or []] == ["targetbot"])
    expect(
        "online bots fetches max documented page",
        calls
        == [
            {
                "path": "/bot/online",
                "params": {"nb": lichess_bot.BOT_ONLINE_FETCH_LIMIT},
            }
        ],
    )


def test_challenge_failure_ratelimit_sets_long_cooldown() -> None:
    class Response:
        status_code = 400
        text = '{"ratelimit":{"seconds":50207}}'

        def json(self) -> dict:
            return {"ratelimit": {"seconds": 50207}}

    bot = object.__new__(lichess_bot.LichessBot)

    cooldown = bot._challenge_failure_cooldown_seconds(Response())

    expect("ratelimit cooldown includes server seconds", cooldown == 50267)


def test_retry_after_can_use_structured_ratelimit_body_for_challenges() -> None:
    class Response:
        headers = {}

        def json(self) -> dict:
            return {"ratelimit": {"seconds": 50207}}

    expect(
        "generic 429 keeps minimum backoff",
        int(lichess_bot._retry_after_seconds(Response()))
        == int(lichess_bot.LICHESS_429_BACKOFF_S),
    )
    expect(
        "challenge 429 honors structured ratelimit seconds",
        int(lichess_bot._retry_after_seconds(Response(), include_body=True)) == 50207,
    )


def test_retry_after_prefers_header_when_larger_than_body() -> None:
    class Response:
        headers = {"retry-after": "120"}

        def json(self) -> dict:
            return {"ratelimit": {"seconds": 30}}

    expect(
        "lower-case retry-after header is honored",
        int(lichess_bot._retry_after_seconds(Response(), include_body=True)) == 120,
    )


def test_seek_rate_limit_uses_server_backoff_body() -> None:
    class Response:
        status_code = 429
        headers = {}
        text = '{"ratelimit":{"seconds":600}}'

        def json(self) -> dict:
            return {"ratelimit": {"seconds": 600}}

    bot = object.__new__(lichess_bot.LichessBot)
    bot._rate_limit_count = 0
    bot._pending_challenge_id = None
    bot._challenge_sent_at = 0.0
    bot._resources_allow_new_game = lambda: True
    bot._should_seek = lambda: True
    bot._next_tc = lambda: (300, 3)
    bot._tc_to_speed = lambda limit, inc: "blitz"
    bot._seek_is_rated = lambda: False
    bot._online_bots = lambda: ([{"id": "targetbot"}], None)
    bot._seek_candidates = lambda bots, speed, rated: (["targetbot"], None)
    bot.api_post = lambda path, **kwargs: Response()
    scheduled: list[float] = []
    bot._schedule_retry = lambda delay=10: scheduled.append(delay)

    with redirect_stdout(io.StringIO()):
        bot._seek_game_once()

    expect("seek 429 schedules server body backoff", scheduled == [600])


def test_opening_book_does_not_send_bot_token_to_explorer() -> None:
    class Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "moves": [
                    {"uci": "e2e4", "white": 10, "draws": 0, "black": 0},
                ]
            }

    class FakeRequests:
        calls: list[dict] = []

        @staticmethod
        def get(url: str, **kwargs):
            FakeRequests.calls.append({"url": url, **kwargs})
            return Response()

    old_requests = lichess_bot.requests
    try:
        lichess_bot.requests = FakeRequests
        book = lichess_bot.OpeningBook(
            api_key="secret-token",
            min_games=1,
            timeout=0,
            allow_online=True,
        )
        expect("book move found", book.lookup("startpos") == "e2e4")
    finally:
        lichess_bot.requests = old_requests

    headers = FakeRequests.calls[0].get("headers", {})
    expect("explorer request has user agent", "User-Agent" in headers)
    expect("explorer request has no auth", "Authorization" not in headers)


def test_opening_book_scores_for_side_to_move() -> None:
    book = lichess_bot.OpeningBook(min_games=1)
    data = {
        "moves": [
            {"uci": "e7e5", "white": 60, "draws": 20, "black": 20},
            {"uci": "c7c5", "white": 30, "draws": 20, "black": 50},
        ]
    }

    expect(
        "black book uses black score",
        book._pick_best_move(
            data,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        )
        == "c7c5",
    )


def test_opening_book_local_reader_uses_entry_move_attribute() -> None:
    class Entry:
        move = lichess_bot.chess.Move.from_uci("e2e4")
        weight = 7

    class Reader:
        def find_all(self, board):
            return [Entry()]

    book = lichess_bot.OpeningBook(min_games=1)
    book._readers = [Reader()]

    expect(
        "local polyglot move found",
        book.lookup(lichess_bot.chess.Board().fen()) == "e2e4",
    )


def test_online_bots_request_is_cached_briefly() -> None:
    class Response:
        status_code = 200
        text = '{"id":"target","perfs":{"blitz":{"games":20,"rating":2400}}}\n'

    bot = object.__new__(lichess_bot.LichessBot)
    bot.bot_id = "metalfish"
    bot._declined_cooldown = {}
    calls = 0

    def api_get(path: str, params=None):
        nonlocal calls
        calls += 1
        return Response()

    bot.api_get = api_get

    bots1, status1 = bot._online_bots()
    bots2, status2 = bot._online_bots()

    expect("first online fetch ok", status1 is None and bots1 is not None)
    expect("cached online fetch ok", status2 is None and bots2 == bots1)
    expect("online endpoint called once", calls == 1)


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


def test_set_position_stops_active_search() -> None:
    engine = object.__new__(lichess_bot.UCIEngine)
    engine._pondering = False
    engine._ponder_move = None
    engine._active_search = "search"
    engine._uci_lock = threading.RLock()
    engine._output = queue.Queue()
    sent: list[str] = []

    def wait_for(prefix: str, timeout: float = 60) -> str:
        expect("position stop waits for bestmove", prefix == "bestmove")
        return "bestmove e2e4"

    engine._send = lambda command: sent.append(command)
    engine._wait_for = wait_for

    engine.set_position("startpos", ["e2e4"])

    expect("position stopped active search", sent[0] == "stop")
    expect(
        "position command sent after stop", sent[-1] == "position startpos moves e2e4"
    )
    expect("active search cleared", engine._active_search is None)


def test_go_tracks_active_search_until_bestmove() -> None:
    engine = object.__new__(lichess_bot.UCIEngine)
    engine._pondering = False
    engine._ponder_move = None
    engine._active_search = None
    engine._uci_lock = threading.RLock()
    engine._output = queue.Queue()
    sent: list[str] = []

    def wait_for(prefix: str, timeout: float = 60) -> str:
        if prefix == "readyok":
            return "readyok"
        expect("go marks active before bestmove", engine._active_search == "search")
        return "bestmove e2e4 ponder e7e5"

    engine._send = lambda command: sent.append(command)
    engine._wait_for = wait_for

    best, ponder = engine.go(movetime=100, timeout=1)

    expect("go best", best == "e2e4")
    expect("go ponder", ponder == "e7e5")
    expect("go command sent", sent == ["isready", "go movetime 100"])
    expect("go clears active search", engine._active_search is None)


def test_go_stop_timeout_keeps_active_search_for_recovery() -> None:
    engine = object.__new__(lichess_bot.UCIEngine)
    engine._pondering = False
    engine._ponder_move = None
    engine._active_search = None
    engine._uci_lock = threading.RLock()
    engine._output = queue.Queue()
    sent: list[str] = []
    bestmove_waits = 0

    def wait_for(prefix: str, timeout: float = 60) -> str:
        nonlocal bestmove_waits
        if prefix == "readyok":
            return "readyok"
        bestmove_waits += 1
        raise TimeoutError("still searching")

    engine._send = lambda command: sent.append(command)
    engine._wait_for = wait_for

    try:
        engine.go(movetime=100, timeout=0.01)
    except TimeoutError:
        pass
    else:
        raise AssertionError("go should propagate stop timeout")

    expect("go sent stop after timeout", sent == ["isready", "go movetime 100", "stop"])
    expect("go tried bestmove twice", bestmove_waits == 2)
    expect(
        "active search retained after failed stop", engine._active_search == "search"
    )


def test_stop_pondering_stops_nonponder_active_search() -> None:
    engine = object.__new__(lichess_bot.UCIEngine)
    engine._pondering = False
    engine._ponder_move = None
    engine._active_search = "search"
    engine._uci_lock = threading.RLock()
    engine._output = queue.Queue()
    sent: list[str] = []

    engine._send = lambda command: sent.append(command)
    engine._wait_for = lambda prefix, timeout=60: "bestmove e2e4"

    ok = engine.stop_pondering(timeout=1)

    expect("active nonponder stop ok", ok)
    expect("active nonponder stop sent", sent == ["stop"])
    expect("active nonponder cleared", engine._active_search is None)


def test_sync_position_records_exact_position_key() -> None:
    engine = object.__new__(lichess_bot.UCIEngine)
    engine._pondering = False
    engine._ponder_move = None
    engine._active_search = None
    engine._position_key = None
    engine._uci_lock = threading.RLock()
    engine._output = queue.Queue()
    sent: list[str] = []

    def wait_for(prefix: str, timeout: float = 60) -> str:
        expect("sync waits ready", prefix == "readyok")
        return "readyok"

    engine._send = lambda command: sent.append(command)
    engine._wait_for = wait_for

    ok = engine.sync_position("startpos", ["e2e4", "e7e5"])

    expect("sync position ok", ok)
    expect(
        "sync position commands",
        sent == ["position startpos moves e2e4 e7e5", "isready"],
    )
    expect(
        "sync position key match", engine.position_matches("startpos", ["e2e4", "e7e5"])
    )
    expect(
        "sync position key mismatch", not engine.position_matches("startpos", ["e2e4"])
    )


def test_recovery_requires_position_sync() -> None:
    class Engine:
        def __init__(self) -> None:
            self.restart_calls = 0
            self.go_calls = 0

        def restart(self) -> None:
            self.restart_calls += 1

        def sync_position(
            self, initial_fen: str, moves: list[str], timeout: float = 0
        ) -> bool:
            return False

        def go(self, **kwargs):
            self.go_calls += 1
            raise AssertionError("recovery must not search after failed sync")

    bot = object.__new__(lichess_bot.LichessBot)
    bot._audit_enabled = False
    board = lichess_bot.chess.Board()
    engine = Engine()

    with redirect_stdout(io.StringIO()):
        best, ponder = bot._recover_engine_move(
            "g1",
            engine,
            "startpos",
            [],
            board,
            "a1a1",
            my_color="white",
            wtime=60_000,
            btime=60_000,
        )

    expect("failed sync returns no move", best is None and ponder is None)
    expect("failed sync restarted once", engine.restart_calls == 1)
    expect("failed sync did not search", engine.go_calls == 0)


def test_recovery_retries_after_position_sync() -> None:
    class Engine:
        def __init__(self) -> None:
            self.restart_calls = 0
            self.synced = False

        def restart(self) -> None:
            self.restart_calls += 1

        def sync_position(
            self, initial_fen: str, moves: list[str], timeout: float = 0
        ) -> bool:
            self.synced = (initial_fen, moves) == ("startpos", [])
            return self.synced

        def go(self, **kwargs):
            expect("recovery searched only after sync", self.synced)
            return "e2e4", "e7e5"

    bot = object.__new__(lichess_bot.LichessBot)
    bot._audit_enabled = False
    board = lichess_bot.chess.Board()
    engine = Engine()

    with redirect_stdout(io.StringIO()):
        best, ponder = bot._recover_engine_move(
            "g1",
            engine,
            "startpos",
            [],
            board,
            "a1a1",
            my_color="white",
            wtime=60_000,
            btime=60_000,
        )

    expect("synced recovery best", best == "e2e4")
    expect("synced recovery ponder", ponder == "e7e5")
    expect("synced recovery restarted once", engine.restart_calls == 1)


def test_fallback_prefers_immediate_mate() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    board = lichess_bot.chess.Board("7k/6Q1/5K2/8/8/8/8/8 w - - 0 1")

    move = bot._fallback_move(board)

    expect("fallback found a move", move is not None)
    board.push(lichess_bot.chess.Move.from_uci(move))
    expect("fallback move mates", board.is_checkmate())


def test_fallback_avoids_hanging_queen_capture() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    board = lichess_bot.chess.Board("4k3/3p4/8/8/8/8/8/3QK3 w - - 0 1")

    move = bot._fallback_move(board)

    expect("fallback found safe alternative", move is not None)
    expect("fallback avoids loose queen capture", move != "d1d7")


def test_ponder_start_failure_is_nonfatal() -> None:
    class Args:
        ponder = True

    class Engine:
        def __init__(self) -> None:
            self.restart_calls = 0

        def start_pondering(self, *args, **kwargs) -> None:
            raise RuntimeError("dead ponder")

        def restart(self) -> None:
            self.restart_calls += 1

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()

    board = lichess_bot.chess.Board()
    engine = Engine()

    with redirect_stdout(io.StringIO()):
        bot._start_pondering_if_legal(
            "game",
            engine,
            "startpos",
            [],
            board,
            "e2e4",
            "e7e5",
            wtime=60_000,
            btime=60_000,
            winc=1_000,
            binc=1_000,
        )

    expect(
        "ponder disabled after failed start", not bot._ponder_allowed_for_game("game")
    )
    expect("engine restarted after failed ponder", engine.restart_calls == 1)


def test_book_ponder_start_failure_is_nonfatal() -> None:
    class Args:
        ponder = True

    class Book:
        def lookup(self, fen: str) -> str:
            return "e7e5"

    class Engine:
        def __init__(self) -> None:
            self.restart_calls = 0

        def start_pondering(self, *args, **kwargs) -> None:
            raise RuntimeError("dead book ponder")

        def restart(self) -> None:
            self.restart_calls += 1

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.book = Book()
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()

    board = lichess_bot.chess.Board()
    engine = Engine()

    with redirect_stdout(io.StringIO()):
        bot._start_book_pondering(
            "game",
            engine,
            "startpos",
            [],
            board,
            "e2e4",
            wtime=60_000,
            btime=60_000,
            winc=1_000,
            binc=1_000,
        )

    expect(
        "book ponder disabled after failed start",
        not bot._ponder_allowed_for_game("game"),
    )
    expect("engine restarted after failed book ponder", engine.restart_calls == 1)


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

    def make_move(game_id: str, move: str, **kwargs) -> bool:
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

    bot.make_move = lambda game_id, move, **kwargs: submitted.append(move) or True
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

    def make_move(game_id: str, move: str, **kwargs) -> bool:
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


def test_pre_submit_active_check_skips_inactive_draw_state() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    bot._submitted_turns = {}
    bot._submitted_turns_lock = threading.Lock()
    records: list[dict] = []
    submitted: list[str] = []

    def audit(game_id: str, event: str, **fields) -> None:
        records.append({"event": event, **fields})

    bot._audit = audit
    bot._pre_submit_active_check_needed = lambda board: True
    bot._game_active_status = lambda game_id: False
    bot.make_move = lambda game_id, move, **kwargs: submitted.append(move) or True

    with redirect_stdout(io.StringIO()):
        result = bot._submit_move_if_active(
            "game", [], "e2e4", lichess_bot.chess.Board()
        )

    expect("inactive draw state not submitted", not result)
    expect("move API not called", submitted == [])
    expect(
        "turn recorded after inactive skip",
        bot._already_submitted_for_turn("game", []),
    )
    expect(
        "active check audited",
        any(
            record["event"] == "pre_submit_active_check"
            and record.get("active") is False
            for record in records
        ),
    )
    expect(
        "inactive skip audited",
        any(record["event"] == "move_submit_skipped_inactive" for record in records),
    )


def test_pre_submit_active_check_is_not_used_for_normal_state() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    bot._submitted_turns = {}
    bot._submitted_turns_lock = threading.Lock()
    bot._last_move_failure_detail = ""
    records: list[dict] = []
    active_checks: list[str] = []

    bot._audit = lambda game_id, event, **fields: records.append(
        {"event": event, **fields}
    )
    bot._pre_submit_active_check_needed = lambda board: False
    bot._game_active_status = lambda game_id: active_checks.append(game_id) or True
    bot.make_move = lambda game_id, move, **kwargs: True

    result = bot._submit_move_if_active("game", [], "e2e4", lichess_bot.chess.Board())

    expect("normal state submitted", result)
    expect("active endpoint not called", active_checks == [])
    expect(
        "normal submit audited",
        any(
            record["event"] == "move_submit" and record["result"] == "accepted"
            for record in records
        ),
    )


def test_submit_skips_locally_completed_game_without_api_call() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    bot._submitted_turns = {}
    bot._submitted_turns_lock = threading.Lock()
    records: list[dict] = []
    active_checks: list[str] = []
    submitted: list[str] = []

    bot._audit = lambda game_id, event, **fields: records.append(
        {"event": event, **fields}
    )
    bot._pre_submit_active_check_needed = lambda board: False
    bot._game_active_status = lambda game_id: active_checks.append(game_id) or True
    bot.make_move = lambda game_id, move, **kwargs: submitted.append(move) or True
    bot._mark_game_completed("game")

    with redirect_stdout(io.StringIO()):
        result = bot._submit_move_if_active(
            "game", [], "e2e4", lichess_bot.chess.Board()
        )

    expect("completed game submit skipped", not result)
    expect("completed game did not call move API", submitted == [])
    expect("completed game did not call active API", active_checks == [])
    expect(
        "completed skip recorded submitted turn",
        bot._already_submitted_for_turn("game", []),
    )
    expect(
        "completed skip audited",
        any(
            record["event"] == "move_submit_skipped_inactive"
            and record.get("source") == "local_completed"
            for record in records
        ),
    )


def test_make_move_can_offer_draw_on_same_request() -> None:
    class Response:
        status_code = 200
        text = ""

    bot = object.__new__(lichess_bot.LichessBot)
    calls: list[dict] = []

    def api_post(path: str, **kwargs):
        calls.append({"path": path, **kwargs})
        return Response()

    bot.api_post = api_post

    expect("draw offer move accepted", bot.make_move("game", "e2e4", offering_draw=True))
    expect(
        "draw offer uses move endpoint",
        calls == [
            {
                "path": "/bot/game/game/move/e2e4",
                "params": {"offeringDraw": "true"},
            }
        ],
    )


def test_draw_offer_reason_requires_claim_and_tablebase_draw() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    board = lichess_bot.chess.Board()

    bot._draw_claim_available = lambda candidate: True
    bot._tablebase_wdl = lambda candidate: 0
    expect("tb draw claim offers draw", bot._draw_offer_reason(board) == "tb_draw_claim")

    bot._tablebase_wdl = lambda candidate: 2
    expect("tb win does not offer draw", bot._draw_offer_reason(board) is None)

    bot._draw_claim_available = lambda candidate: False
    bot._tablebase_wdl = lambda candidate: 0
    expect("no claim does not offer draw", bot._draw_offer_reason(board) is None)


def test_draw_offer_caps_search_and_marks_move_request() -> None:
    class Engine:
        ponder_move = None

        def __init__(self) -> None:
            self.timeout: float | None = None

        def alive(self) -> bool:
            return True

        def stop_pondering(self, timeout: float) -> bool:
            return True

        def set_position(self, initial_fen: str, moves: list[str]) -> None:
            return None

        def go(self, **kwargs):
            self.timeout = kwargs.get("timeout")
            return "e2e4", None

    bot = object.__new__(lichess_bot.LichessBot)
    bot._submitted_turns = {}
    bot._max_seen_ply = {}
    bot._submitted_turns_lock = threading.Lock()
    bot._completed_game_ids = set()
    bot._completed_game_order = []
    records: list[dict] = []
    bot._audit = lambda game_id, event, **fields: records.append(
        {"event": event, **fields}
    )
    bot._should_query_book = lambda board, my_color, wtime, btime: False
    bot._draw_claim_available = lambda board: True
    bot._tablebase_wdl = lambda board: 0
    bot._pre_submit_active_check_needed = lambda board: False

    submitted: list[dict] = []

    def make_move(game_id: str, move: str, *, offering_draw: bool = False) -> bool:
        submitted.append({"move": move, "offering_draw": offering_draw})
        return True

    bot.make_move = make_move
    engine = Engine()
    state = {"wtime": 600000, "btime": 600000, "winc": 10000, "binc": 10000}

    with redirect_stdout(io.StringIO()):
        bot._try_move("game", engine, "startpos", [], "white", state)

    expect(
        "draw offer search capped",
        engine.timeout is not None
        and engine.timeout <= lichess_bot.DRAW_OFFER_SEARCH_CAP_MS / 1000.0,
    )
    expect(
        "draw offer submitted on move request",
        submitted == [{"move": "e2e4", "offering_draw": True}],
    )
    expect(
        "draw state audited",
        any(
            record["event"] == "draw_state"
            and record.get("draw_offer_eligible") is True
            for record in records
        ),
    )


def test_ponderhit_audit_records_elapsed_ms() -> None:
    class Args:
        ponder = True

    class Book:
        def lookup(self, fen: str) -> None:
            return None

    class Engine:
        ponder_move = "e2e4"

        def ponderhit(self, timeout: float = 0) -> tuple[str, None]:
            return "e7e5", None

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.book = Book()
    bot._submitted_turns = {}
    bot._submitted_turns_lock = threading.Lock()
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()
    bot._last_move_failure_detail = ""
    records: list[dict] = []

    def audit(game_id: str, event: str, **fields) -> None:
        records.append({"event": event, **fields})

    bot._audit = audit
    bot.make_move = lambda game_id, move, **kwargs: True
    state = {"wtime": 900000, "btime": 900000, "winc": 10000, "binc": 10000}

    with redirect_stdout(io.StringIO()):
        bot._try_move("game", Engine(), "startpos", ["e2e4"], "black", state)

    result = next(record for record in records if record["event"] == "ponderhit_result")
    expect("ponderhit elapsed recorded", isinstance(result.get("elapsed_ms"), int))
    expect("ponderhit elapsed non-negative", result["elapsed_ms"] >= 0)


def test_elo_seek_does_not_fallback_to_random_low_rated_bots() -> None:
    class Args:
        elo_seek = True
        elo_range = 200
        elo_target = None

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot._elo_widen_steps = 0
    bot._cached_ratings = {"rapid": 2900}
    bots = [
        {
            "id": "low",
            "perfs": {"rapid": {"games": 50, "rating": 1500, "prov": False}},
        },
        {
            "id": "very-low",
            "perfs": {"rapid": {"games": 50, "rating": 900, "prov": False}},
        },
    ]

    with redirect_stdout(io.StringIO()):
        filtered = bot._filter_bots_by_elo(bots, "rapid")

    expect("no random fallback", filtered == [])
    expect("range widened for retry", bot._elo_widen_steps == 1)


def test_rated_seek_rating_floor() -> None:
    class Args:
        min_rated_opponent_elo = 2200

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    low = {"perfs": {"rapid": {"games": 10, "rating": 1500, "prov": False}}}
    high = {"perfs": {"rapid": {"games": 10, "rating": 2400, "prov": False}}}

    expect("low rated rejected", not bot._rated_opponent_allowed(low, "rapid"))
    expect("high rated accepted", bot._rated_opponent_allowed(high, "rapid"))


def test_rated_challenge_rating_floor_uses_direct_rating() -> None:
    class Args:
        min_rated_opponent_elo = 2200

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()

    low = {"challenger": {"id": "low", "rating": 1500}}
    high = {"challenger": {"id": "high", "rating": "2400"}}
    sparse = {"challenger": {"id": "unknown"}}

    expect(
        "low direct challenge rejected",
        not bot._rated_challenge_allowed(low, "rapid"),
    )
    expect(
        "high direct challenge accepted",
        bot._rated_challenge_allowed(high, "rapid"),
    )
    expect(
        "missing rated challenge rejected",
        not bot._rated_challenge_allowed(sparse, "rapid"),
    )


def test_should_accept_handles_sparse_challenge_users() -> None:
    class Args:
        accept_rated = False
        accept_casual = True
        include_bullet = False
        include_zero_increment = False
        max_games = 1

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.bot_id = "metalfish"
    bot._draining = threading.Event()
    bot.active_games = {}
    bot._pending_challenge_id = None
    bot._resources_allow_new_game = lambda: True

    challenge = {
        "challenge": {
            "challenger": "otherbot",
            "variant": "standard",
            "rated": False,
            "speed": "rapid",
            "timeControl": {"type": "clock", "increment": 10},
        }
    }
    expect(
        "string challenger accepted when otherwise valid",
        bot.should_accept(challenge),
    )

    challenge["challenge"]["challenger"] = "metalfish"
    expect("self string challenger rejected", not bot.should_accept(challenge))


def test_should_accept_respects_resource_gate() -> None:
    class Args:
        accept_rated = False
        accept_casual = True
        include_bullet = False
        include_zero_increment = False
        max_games = 1

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.bot_id = "metalfish"
    bot._draining = threading.Event()
    bot.active_games = {}
    bot._pending_challenge_id = None
    bot._resources_allow_new_game = lambda: False

    challenge = {
        "challenge": {
            "challenger": {"id": "otherbot"},
            "variant": {"key": "standard"},
            "rated": False,
            "speed": "rapid",
            "timeControl": {"type": "clock", "increment": 10},
        }
    }

    expect("busy resources reject challenge", not bot.should_accept(challenge))
    expect(
        "busy resources reason",
        bot._challenge_reject_reason(challenge) == "resources busy",
    )


def test_rated_challenge_respects_elo_seek_range() -> None:
    class Args:
        accept_rated = True
        accept_casual = True
        elo_seek = True
        elo_range = 200
        elo_target = None
        min_rated_opponent_elo = 2200
        include_bullet = False
        include_zero_increment = False
        max_games = 1

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.bot_id = "metalfish"
    bot._cached_ratings = {"rapid": 2900}
    bot._elo_widen_steps = 0
    bot._draining = threading.Event()
    bot.active_games = {}
    bot._pending_challenge_id = None
    bot._resources_allow_new_game = lambda: True

    challenge = {
        "challenge": {
            "challenger": {"id": "near", "rating": 2840},
            "variant": {"key": "standard"},
            "rated": True,
            "speed": "rapid",
            "timeControl": {"type": "clock", "increment": 10},
        }
    }
    expect("near rated challenge accepted", bot.should_accept(challenge))

    challenge["challenge"]["challenger"] = {"id": "far", "rating": 2400}
    expect("far rated challenge rejected", not bot.should_accept(challenge))
    expect(
        "far rated challenge reason",
        bot._challenge_reject_reason(challenge) == "rated elo range",
    )


def test_casual_challenge_ignores_elo_seek_range() -> None:
    class Args:
        accept_rated = True
        accept_casual = True
        elo_seek = True
        elo_range = 200
        elo_target = None
        min_rated_opponent_elo = 2200
        include_bullet = False
        include_zero_increment = False
        max_games = 1

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.bot_id = "metalfish"
    bot._cached_ratings = {"rapid": 2900}
    bot._elo_widen_steps = 0
    bot._draining = threading.Event()
    bot.active_games = {}
    bot._pending_challenge_id = None
    bot._resources_allow_new_game = lambda: True

    challenge = {
        "challenge": {
            "challenger": {"id": "casual-low", "rating": 900},
            "variant": {"key": "standard"},
            "rated": False,
            "speed": "rapid",
            "timeControl": {"type": "clock", "increment": 10},
        }
    }
    expect("casual challenge accepted despite elo seek", bot.should_accept(challenge))


def test_challenge_decline_logs_reason() -> None:
    class Args:
        accept_rated = False
        accept_casual = True
        include_bullet = False
        include_zero_increment = False
        max_games = 1

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.bot_id = "metalfish"
    bot._draining = threading.Event()
    bot.active_games = {}
    bot._pending_challenge_id = None
    bot._resources_allow_new_game = lambda: True
    declined: list[str] = []
    bot.decline_challenge = lambda challenge_id: declined.append(challenge_id)

    event = {
        "type": "challenge",
        "challenge": {
            "id": "c1",
            "challenger": {"id": "otherbot", "name": "OtherBot"},
            "variant": {"key": "standard"},
            "rated": False,
            "speed": "rapid",
            "timeControl": {"type": "clock", "limit": 300, "increment": 0},
        },
    }

    out = io.StringIO()
    with redirect_stdout(out):
        bot._handle_event(event)

    expect("challenge declined", declined == ["c1"])
    expect("decline reason logged", "zero increment disabled" in out.getvalue())


def test_challenge_handler_tolerates_malformed_time_control() -> None:
    class Args:
        accept_rated = False
        accept_casual = True
        include_bullet = False
        include_zero_increment = False
        max_games = 1

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.bot_id = "metalfish"
    bot._draining = threading.Event()
    bot.active_games = {}
    bot._pending_challenge_id = None
    bot._resources_allow_new_game = lambda: True
    declined: list[str] = []
    bot.decline_challenge = lambda challenge_id: declined.append(challenge_id)

    event = {
        "type": "challenge",
        "challenge": {
            "challenger": {"id": "otherbot", "name": "OtherBot"},
            "variant": {"key": "standard"},
            "rated": False,
            "speed": "rapid",
            "timeControl": "corrupt",
        },
    }

    out = io.StringIO()
    with redirect_stdout(out):
        bot._handle_event(event)

    expect("missing id not declined", declined == [])
    expect(
        "malformed time control logged", "unsupported time control" in out.getvalue()
    )


def test_transient_move_rejection_remains_retryable() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    bot._submitted_turns = {}
    bot._submitted_turns_lock = threading.Lock()
    bot._last_move_failure_detail = "503: service unavailable"
    submitted: list[str] = []

    def make_move(game_id: str, move: str, **kwargs) -> bool:
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
    with redirect_stdout(io.StringIO()):
        bot._handle_event(event)

    expect("pending cleared", bot._pending_challenge_id is None)
    expect("target cooled down", "targetbot" in bot._declined_cooldown)
    expect("retry scheduled", scheduled == [2])
    expect("tc failure counted", bot._tc_failures == 1)


def test_time_control_decline_uses_speed_cooldown() -> None:
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
    bot._pending_challenge_speed = "rapid"
    bot._challenge_sent_at = 0
    bot._tc_failures = 0
    bot._declined_cooldown = {}
    bot._speed_declined_cooldown = {}
    bot._draining = threading.Event()

    scheduled: list[float] = []
    bot._schedule_retry = lambda delay=10: scheduled.append(delay)

    event = {
        "type": "challengeDeclined",
        "challenge": {
            "id": "current",
            "destUser": {"id": "targetbot"},
            "reason": "This time control is too slow for me.",
        },
    }
    with redirect_stdout(io.StringIO()):
        bot._handle_event(event)

    expect("pending cleared", bot._pending_challenge_id is None)
    expect("target not globally cooled", "targetbot" not in bot._declined_cooldown)
    expect(
        "target cooled only for rapid",
        "targetbot" in bot._speed_declined_cooldown.get("rapid", {}),
    )
    expires = bot._speed_declined_cooldown["rapid"]["targetbot"]
    expect(
        "time control cooldown is long enough",
        expires - time.time() >= lichess_bot.TIME_CONTROL_DECLINE_COOLDOWN_S - 5,
    )
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


def test_challenge_timeout_cancels_server_side_challenge() -> None:
    class Args:
        seek = False
        rotate = False
        max_games = 1

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.active_games = {}
    bot._pending_challenge_id = "challenge"
    bot._pending_challenge_target = "targetbot"
    bot._pending_challenge_speed = "rapid"
    bot._challenge_retries = 0
    bot._tc_failures = 0
    bot._draining = threading.Event()
    bot._declined_cooldown = {}
    bot._persist_challenge_cooldowns = False

    posted: list[str] = []
    bot.api_post = lambda path, **kwargs: posted.append(path)

    with redirect_stdout(io.StringIO()):
        bot._challenge_timed_out()

    expect("timeout clears pending challenge", bot._pending_challenge_id is None)
    expect("timeout posts cancel", posted == ["/challenge/challenge/cancel"])
    expect("timeout cools target", "targetbot" in bot._declined_cooldown)
    expect("timeout counts tc failure", bot._tc_failures == 1)


def test_game_start_claims_slot_and_clears_pending() -> None:
    class Args:
        seek = True
        max_games = 1
        include_zero_increment = False
        include_bullet = False
        avoid_repeat_format = True

    class Timer:
        canceled = False

        def cancel(self) -> None:
            self.canceled = True

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.active_games = {}
    bot._pending_challenge_id = "g1"
    bot._pending_challenge_target = "targetbot"
    bot._pending_challenge_speed = "rapid"
    bot._challenge_sent_at = 123.0
    bot._rotation_idx = 0
    bot._tc_failures = 3
    bot._draining = threading.Event()
    bot._shutdown = threading.Event()
    bot._seek_timer = Timer()
    bot._played_by_speed = {}
    bot._persist_played_format_history = False

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
    expect(
        "target marked as played in speed",
        "targetbot" in bot._played_by_speed.get("rapid", {}),
    )


def test_game_start_cancels_unrelated_pending_challenge() -> None:
    class Args:
        seek = True
        max_games = 1
        include_zero_increment = False
        include_bullet = False
        avoid_repeat_format = False

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.active_games = {}
    bot._pending_challenge_id = "oldchallenge"
    bot._pending_challenge_target = "oldbot"
    bot._pending_challenge_speed = "rapid"
    bot._challenge_sent_at = 123.0
    bot._rotation_idx = 0
    bot._tc_failures = 2
    bot._draining = threading.Event()
    bot._shutdown = threading.Event()
    bot._seek_timer = None
    bot._seek_lock = threading.Lock()

    posted: list[str] = []
    played: list[str] = []
    bot.api_post = lambda path, **kwargs: posted.append(path)
    bot.play_game = lambda game_id: played.append(game_id)

    with redirect_stdout(io.StringIO()):
        bot._handle_event({"type": "gameStart", "game": {"gameId": "g1"}})

    thread = bot.active_games.get("g1")
    if thread is not None:
        thread.join(timeout=1)

    expect("unrelated pending canceled", posted == ["/challenge/oldchallenge/cancel"])
    expect("unrelated pending cleared", bot._pending_challenge_id is None)
    expect("game still started", played == ["g1"])
    expect("seek blocked after race", not bot._should_seek())


def test_malformed_global_events_are_ignored() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    bot.active_games = {}
    bot._draining = threading.Event()
    bot._shutdown = threading.Event()

    with redirect_stdout(io.StringIO()):
        bot._handle_event(None)
        bot._handle_event([])
        bot._handle_event({"type": "gameStart"})
        bot._handle_event({"type": "gameFinish", "game": "corrupt"})

    expect("malformed events did not start games", bot.active_games == {})
    expect("malformed events did not shut down", not bot._shutdown.is_set())


def test_game_finish_marks_game_completed() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    bot.active_games = {}
    bot._draining = threading.Event()
    bot._shutdown = threading.Event()

    with redirect_stdout(io.StringIO()):
        bot._handle_event({"type": "gameFinish", "game": {"gameId": "g1"}})

    expect("game finish remembered", bot._game_was_completed("g1"))


def test_stale_game_start_for_completed_game_is_ignored() -> None:
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
    timer = Timer()
    bot._seek_timer = timer
    bot._mark_game_completed("g1")

    played: list[str] = []
    bot.play_game = lambda game_id: played.append(game_id)

    with redirect_stdout(io.StringIO()):
        bot._handle_event({"type": "gameStart", "game": {"gameId": "g1"}})

    expect("stale completed game not started", played == [])
    expect("stale completed game not active", bot.active_games == {})
    expect("pending challenge preserved", bot._pending_challenge_id == "challenge")
    expect("seek timer preserved", bot._seek_timer is timer and not timer.canceled)


def test_completed_game_history_is_bounded() -> None:
    bot = object.__new__(lichess_bot.LichessBot)
    for i in range(lichess_bot.COMPLETED_GAME_HISTORY_LIMIT + 3):
        bot._mark_game_completed(f"g{i}")

    expect("old completed game expired", not bot._game_was_completed("g0"))
    expect("recent completed game retained", bot._game_was_completed("g512"))
    expect(
        "completed game history bounded",
        len(bot._completed_game_order) == lichess_bot.COMPLETED_GAME_HISTORY_LIMIT,
    )


def test_play_game_stream_failure_is_not_completed() -> None:
    class Args:
        quit_after_games = 0
        seek = True
        max_games = 1

    class Engine:
        def new_game(self) -> None:
            return None

        def quit(self) -> None:
            return None

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.active_games = {"g1": threading.current_thread()}
    bot._submitted_turns = {"g1": [()]}
    bot._max_seen_ply = {"g1": 1}
    bot._submitted_turns_lock = threading.Lock()
    bot._ponder_disabled_games = {"g1"}
    bot._ponder_disabled_lock = threading.Lock()
    bot._completed_games = 0
    bot._completed_game_ids = set()
    bot._completed_game_order = []
    bot._draining = threading.Event()
    bot._shutdown = threading.Event()
    bot._acquire_engine = lambda game_id: Engine()

    calls: list[str] = []
    bot._game_loop = lambda game_id, engine: calls.append(game_id) or False

    old_retries = lichess_bot.GAME_STREAM_RETRIES
    old_delay = lichess_bot.GAME_STREAM_RETRY_DELAY_S
    lichess_bot.GAME_STREAM_RETRIES = 1
    lichess_bot.GAME_STREAM_RETRY_DELAY_S = 0.0
    try:
        with redirect_stdout(io.StringIO()):
            bot.play_game("g1")
    finally:
        lichess_bot.GAME_STREAM_RETRIES = old_retries
        lichess_bot.GAME_STREAM_RETRY_DELAY_S = old_delay

    expect("stream failure retried once", calls == ["g1", "g1"])
    expect("stream failure not counted", bot._completed_games == 0)
    expect("stream failure not marked complete", not bot._game_was_completed("g1"))
    expect("stream failure shuts down", bot._shutdown.is_set())
    expect("stream failure clears active slot", bot.active_games == {})


def test_play_game_stream_inactive_status_is_completed() -> None:
    class Args:
        quit_after_games = 0
        seek = False
        max_games = 1

    class Engine:
        def new_game(self) -> None:
            return None

        def quit(self) -> None:
            return None

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.active_games = {"g1": threading.current_thread()}
    bot._submitted_turns = {}
    bot._max_seen_ply = {}
    bot._submitted_turns_lock = threading.Lock()
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()
    bot._completed_games = 0
    bot._completed_game_ids = set()
    bot._completed_game_order = []
    bot._draining = threading.Event()
    bot._shutdown = threading.Event()
    bot._acquire_engine = lambda game_id: Engine()
    bot._game_loop = lambda game_id, engine: False
    bot._game_active_status = lambda game_id: False

    with redirect_stdout(io.StringIO()):
        bot.play_game("g1")

    expect("inactive stream counted", bot._completed_games == 1)
    expect("inactive stream marked complete", bot._game_was_completed("g1"))
    expect("inactive stream does not force shutdown", not bot._shutdown.is_set())


def test_play_game_stream_active_status_extends_reconnects() -> None:
    class Args:
        quit_after_games = 0
        seek = True
        max_games = 1

    class Engine:
        def new_game(self) -> None:
            return None

        def quit(self) -> None:
            return None

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.active_games = {"g1": threading.current_thread()}
    bot._submitted_turns = {}
    bot._max_seen_ply = {}
    bot._submitted_turns_lock = threading.Lock()
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()
    bot._completed_games = 0
    bot._completed_game_ids = set()
    bot._completed_game_order = []
    bot._draining = threading.Event()
    bot._shutdown = threading.Event()
    bot._acquire_engine = lambda game_id: Engine()

    calls: list[str] = []
    statuses = iter([True, True, None])
    bot._game_loop = lambda game_id, engine: calls.append(game_id) or False
    bot._game_active_status = lambda game_id: next(statuses)

    old_retries = lichess_bot.GAME_STREAM_RETRIES
    old_active_retries = lichess_bot.GAME_STREAM_ACTIVE_RETRIES
    old_delay = lichess_bot.GAME_STREAM_RETRY_DELAY_S
    lichess_bot.GAME_STREAM_RETRIES = 0
    lichess_bot.GAME_STREAM_ACTIVE_RETRIES = 2
    lichess_bot.GAME_STREAM_RETRY_DELAY_S = 0.0
    try:
        with redirect_stdout(io.StringIO()):
            bot.play_game("g1")
    finally:
        lichess_bot.GAME_STREAM_RETRIES = old_retries
        lichess_bot.GAME_STREAM_ACTIVE_RETRIES = old_active_retries
        lichess_bot.GAME_STREAM_RETRY_DELAY_S = old_delay

    expect("active status extends reconnects", calls == ["g1", "g1", "g1"])
    expect("active dropped stream not counted", bot._completed_games == 0)
    expect("active dropped stream shuts down after budget", bot._shutdown.is_set())


def test_play_game_transient_unknown_status_extends_reconnects() -> None:
    class Args:
        quit_after_games = 0
        seek = True
        max_games = 1

    class Engine:
        def new_game(self) -> None:
            return None

        def quit(self) -> None:
            return None

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.active_games = {"g1": threading.current_thread()}
    bot._submitted_turns = {}
    bot._max_seen_ply = {}
    bot._submitted_turns_lock = threading.Lock()
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()
    bot._completed_games = 0
    bot._completed_game_ids = set()
    bot._completed_game_order = []
    bot._draining = threading.Event()
    bot._shutdown = threading.Event()
    bot._acquire_engine = lambda game_id: Engine()
    bot._game_active_status = lambda game_id: None

    calls: list[str] = []

    def game_loop(game_id: str, engine) -> bool:
        calls.append(game_id)
        bot._last_game_stream_status = 502
        bot._last_game_stream_error = ""
        return False

    bot._game_loop = game_loop

    old_retries = lichess_bot.GAME_STREAM_RETRIES
    old_active_retries = lichess_bot.GAME_STREAM_ACTIVE_RETRIES
    old_delay = lichess_bot.GAME_STREAM_RETRY_DELAY_S
    lichess_bot.GAME_STREAM_RETRIES = 0
    lichess_bot.GAME_STREAM_ACTIVE_RETRIES = 2
    lichess_bot.GAME_STREAM_RETRY_DELAY_S = 0.0
    try:
        with redirect_stdout(io.StringIO()):
            bot.play_game("g1")
    finally:
        lichess_bot.GAME_STREAM_RETRIES = old_retries
        lichess_bot.GAME_STREAM_ACTIVE_RETRIES = old_active_retries
        lichess_bot.GAME_STREAM_RETRY_DELAY_S = old_delay

    expect("transient unknown active extends reconnects", calls == ["g1", "g1", "g1"])
    expect("transient dropped stream not counted", bot._completed_games == 0)
    expect("transient dropped stream shuts down after budget", bot._shutdown.is_set())


def test_play_game_finished_stream_is_completed() -> None:
    class Args:
        quit_after_games = 0
        seek = False
        max_games = 1

    class Engine:
        def new_game(self) -> None:
            return None

        def quit(self) -> None:
            return None

    bot = object.__new__(lichess_bot.LichessBot)
    bot.args = Args()
    bot.active_games = {"g1": threading.current_thread()}
    bot._submitted_turns = {}
    bot._max_seen_ply = {}
    bot._submitted_turns_lock = threading.Lock()
    bot._ponder_disabled_games = set()
    bot._ponder_disabled_lock = threading.Lock()
    bot._completed_games = 0
    bot._completed_game_ids = set()
    bot._completed_game_order = []
    bot._draining = threading.Event()
    bot._shutdown = threading.Event()
    bot._acquire_engine = lambda game_id: Engine()
    bot._game_loop = lambda game_id, engine: True

    with redirect_stdout(io.StringIO()):
        bot.play_game("g1")

    expect("finished stream counted", bot._completed_games == 1)
    expect("finished stream marked complete", bot._game_was_completed("g1"))
    expect("finished stream does not force shutdown", not bot._shutdown.is_set())


def test_game_active_status_parses_now_playing_ids() -> None:
    class Response:
        status_code = 200

        def __init__(self, data: dict) -> None:
            self.data = data

        def json(self) -> dict:
            return self.data

    bot = object.__new__(lichess_bot.LichessBot)
    bot.api_get = lambda path, timeout=0: Response(
        {
            "nowPlaying": [
                {"gameId": "other"},
                {"fullId": "NBCejRUEblack"},
            ]
        }
    )

    expect("full id matches game id", bot._game_active_status("NBCejRUE") is True)
    expect("missing active game is false", bot._game_active_status("missing") is False)


def test_game_loop_uses_long_read_timeout_tuple() -> None:
    class Response:
        status_code = 502

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    bot = object.__new__(lichess_bot.LichessBot)
    bot._audit_enabled = False
    calls: list[dict] = []

    def api_get(path: str, **kwargs):
        calls.append({"path": path, **kwargs})
        return Response()

    bot.api_get = api_get

    with redirect_stdout(io.StringIO()):
        finished = bot._game_loop("g1", object())

    expect("failed stream not finished", not finished)
    expect(
        "game stream uses connect/read timeout tuple",
        calls
        == [
            {
                "path": "/bot/game/stream/g1",
                "stream": True,
                "timeout": (10, lichess_bot.GAME_STREAM_TIMEOUT_S),
            }
        ],
    )


def test_game_loop_skips_malformed_stream_frames() -> None:
    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def iter_lines(self):
            yield b"[]"
            yield b'{"type":"gameFull","state":"corrupt"}'
            yield b'{"type":"gameState","status":"started","moves":["bad"]}'
            yield b'{"type":"gameState","status":"mate","winner":"white","moves":["bad"]}'

    bot = object.__new__(lichess_bot.LichessBot)
    bot.bot_id = "metalfish"
    bot.api_get = lambda path, stream=False, timeout=0: Response()

    calls: list[list[str]] = []
    bot._try_move = (
        lambda game_id, engine, initial_fen, moves, my_color, state: calls.append(moves)
    )

    with redirect_stdout(io.StringIO()):
        finished = bot._game_loop("g1", object())

    expect("malformed stream frames still reach terminal state", finished)
    expect("malformed move fields become empty move lists", calls == [[], []])
    expect("terminal stream frame marks game completed", bot._game_was_completed("g1"))


def test_audit_writes_bounded_jsonl() -> None:
    old_dir = lichess_bot.LICHESS_AUDIT_DIR
    old_limit = lichess_bot.LICHESS_AUDIT_EVENT_LIMIT
    old_field_limit = lichess_bot.LICHESS_AUDIT_FIELD_LIMIT

    try:
        with tempfile.TemporaryDirectory() as tmp:
            lichess_bot.LICHESS_AUDIT_DIR = pathlib.Path(tmp)
            lichess_bot.LICHESS_AUDIT_EVENT_LIMIT = 2
            lichess_bot.LICHESS_AUDIT_FIELD_LIMIT = 16

            bot = object.__new__(lichess_bot.LichessBot)
            bot._audit_enabled = True
            bot._audit_lock = threading.Lock()
            bot._audit_counts = {}

            bot._audit("game/1", "start", long_text="x" * 64)
            bot._audit("game/1", "move", moves=list(range(40)))
            bot._audit("game/1", "dropped")

            path = pathlib.Path(tmp) / "game_1.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
    finally:
        lichess_bot.LICHESS_AUDIT_DIR = old_dir
        lichess_bot.LICHESS_AUDIT_EVENT_LIMIT = old_limit
        lichess_bot.LICHESS_AUDIT_FIELD_LIMIT = old_field_limit

    expect("audit event limit enforced", len(records) == 2)
    expect("audit event serialized", records[0]["event"] == "start")
    expect("audit field truncated", records[0]["long_text"].endswith("..."))
    expect("audit list bounded", len(records[1]["moves"]) == 32)


def test_submit_move_writes_audit_result() -> None:
    old_dir = lichess_bot.LICHESS_AUDIT_DIR

    try:
        with tempfile.TemporaryDirectory() as tmp:
            lichess_bot.LICHESS_AUDIT_DIR = pathlib.Path(tmp)

            bot = object.__new__(lichess_bot.LichessBot)
            bot._audit_enabled = True
            bot._audit_lock = threading.Lock()
            bot._audit_counts = {}
            bot._submitted_turns = {}
            bot._submitted_turns_lock = threading.Lock()
            bot._last_move_failure_detail = ""
            calls = iter([True, False])

            def make_move(game_id: str, move: str, **kwargs) -> bool:
                ok = next(calls)
                if not ok:
                    bot._last_move_failure_detail = '400: {"error":"Not your turn"}'
                return ok

            bot.make_move = make_move
            with redirect_stdout(io.StringIO()):
                bot._submit_move("g1", ["e2e4"], "e7e5")
                bot._submit_move("g1", ["e2e4", "e7e5"], "g1f3")

            path = pathlib.Path(tmp) / "g1.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
    finally:
        lichess_bot.LICHESS_AUDIT_DIR = old_dir

    expect("submit audit accepted", records[0]["result"] == "accepted")
    expect("submit audit rejected", records[1]["result"] == "rejected")
    expect("submit audit stale rejection", records[1]["stale"])


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
    test_runtime_ane_options_are_explicitly_opt_in()
    test_verbose_runtime_enables_trace_without_forcing_ane_hints()
    test_verbose_path_enables_trace_and_resolves_log_path()
    test_verbose_log_tees_stdout_and_stderr()
    test_verbose_uci_option_tracking_filters_useful_options()
    test_ane_runtime_values_are_clamped_to_uci_bounds()
    test_ane_config_validation_requires_existing_files()
    test_load_adjusted_workers_preserves_floor()
    test_pre_game_resource_prep_runs_cleanup_before_allocation()
    test_bot_instance_lock_blocks_second_holder()
    test_seek_rating_requires_rated_only_mode()
    test_seek_and_rated_policy_labels_are_explicit()
    test_bot_config_rejects_dangerous_rated_seek()
    test_bot_config_rejects_invalid_common_modes()
    test_config_check_prints_without_api_or_engine_launch()
    test_should_seek_respects_resource_gate()
    test_seek_dry_run_does_not_post_challenge()
    test_seek_dry_run_reports_rating_floor_block()
    test_seek_candidates_tolerate_malformed_perf_records()
    test_seek_candidates_filter_cached_cooldowns_case_insensitively()
    test_seek_candidates_filter_time_control_cooldown_by_speed_only()
    test_speed_challenge_cooldowns_persist_between_runs()
    test_highest_rated_seek_orders_candidates_descending()
    test_avoid_repeat_format_filters_only_matching_speed()
    test_online_bots_uses_documented_fetch_limit()
    test_challenge_failure_ratelimit_sets_long_cooldown()
    test_retry_after_can_use_structured_ratelimit_body_for_challenges()
    test_retry_after_prefers_header_when_larger_than_body()
    test_seek_rate_limit_uses_server_backoff_body()
    test_opening_book_does_not_send_bot_token_to_explorer()
    test_opening_book_scores_for_side_to_move()
    test_opening_book_local_reader_uses_entry_move_attribute()
    test_online_bots_request_is_cached_briefly()
    test_engine_configure_applies_changed_options()
    test_engine_send_fails_fast_for_dead_process()
    test_ponderhit_stops_after_timeout()
    test_set_position_stops_active_search()
    test_go_tracks_active_search_until_bestmove()
    test_go_stop_timeout_keeps_active_search_for_recovery()
    test_stop_pondering_stops_nonponder_active_search()
    test_sync_position_records_exact_position_key()
    test_recovery_requires_position_sync()
    test_recovery_retries_after_position_sync()
    test_fallback_prefers_immediate_mate()
    test_fallback_avoids_hanging_queen_capture()
    test_ponder_start_failure_is_nonfatal()
    test_book_ponder_start_failure_is_nonfatal()
    test_submitted_turn_guard()
    test_duplicate_game_state_does_not_resubmit()
    test_stale_stream_ply_does_not_search_old_position()
    test_stale_move_rejection_suppresses_duplicate_turn()
    test_pre_submit_active_check_skips_inactive_draw_state()
    test_pre_submit_active_check_is_not_used_for_normal_state()
    test_ponderhit_audit_records_elapsed_ms()
    test_elo_seek_does_not_fallback_to_random_low_rated_bots()
    test_rated_seek_rating_floor()
    test_rated_challenge_rating_floor_uses_direct_rating()
    test_should_accept_handles_sparse_challenge_users()
    test_should_accept_respects_resource_gate()
    test_rated_challenge_respects_elo_seek_range()
    test_casual_challenge_ignores_elo_seek_range()
    test_challenge_decline_logs_reason()
    test_challenge_handler_tolerates_malformed_time_control()
    test_transient_move_rejection_remains_retryable()
    test_ponder_game_circuit_breaker()
    test_submit_skips_locally_completed_game_without_api_call()
    test_make_move_can_offer_draw_on_same_request()
    test_draw_offer_reason_requires_claim_and_tablebase_draw()
    test_draw_offer_caps_search_and_marks_move_request()
    test_stale_challenge_event_preserves_current_pending()
    test_matching_challenge_event_clears_pending()
    test_time_control_decline_uses_speed_cooldown()
    test_unrelated_challenge_event_without_pending_is_ignored()
    test_challenge_event_identity_accepts_string_users()
    test_challenge_timeout_cancels_server_side_challenge()
    test_game_start_claims_slot_and_clears_pending()
    test_game_start_cancels_unrelated_pending_challenge()
    test_malformed_global_events_are_ignored()
    test_game_finish_marks_game_completed()
    test_stale_game_start_for_completed_game_is_ignored()
    test_completed_game_history_is_bounded()
    test_play_game_stream_failure_is_not_completed()
    test_play_game_stream_inactive_status_is_completed()
    test_play_game_stream_active_status_extends_reconnects()
    test_play_game_transient_unknown_status_extends_reconnects()
    test_play_game_finished_stream_is_completed()
    test_game_active_status_parses_now_playing_ids()
    test_game_loop_uses_long_read_timeout_tuple()
    test_game_loop_skips_malformed_stream_frames()
    test_audit_writes_bounded_jsonl()
    test_submit_move_writes_audit_result()
    test_quit_after_games_limit()
    print("Lichess bot tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
