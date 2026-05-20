#!/usr/bin/env python3
"""Lichess Bot: runs MetalFish Hybrid on Lichess via the Bot API.

Features:
  - Bounded local Polyglot opening book with optional Explorer fallback
  - Pondering by default, with hard stop budgets for clock safety
  - Aggressive challenge seeking with timeout/retry
  - Rate limit awareness
  - Engine crash recovery

Usage:
    python3 tools/lichess_bot.py --seek --no-casual
    python3 tools/lichess_bot.py --seek --tc "5+3" --no-casual
    python3 tools/lichess_bot.py --seek --rotate --no-casual
"""

from __future__ import annotations

import argparse
import email.utils
import gc
import json
import os
import pathlib
import queue
import random
import re
import subprocess
import sys
import threading
import time
import traceback

import chess
import chess.polyglot

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

try:
    import requests
except ModuleNotFoundError:
    requests = None

PROJ = pathlib.Path(__file__).resolve().parent.parent
ENGINE = PROJ / "build" / "metalfish"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"
LICHESS_API = "https://lichess.org/api"
EXPLORER_API = "https://explorer.lichess.ovh"
HTTP_USER_AGENT = os.environ.get(
    "METALFISH_HTTP_USER_AGENT",
    "MetalFishBot/1.0 (+https://github.com/NripeshN/MetalFish)",
)

LOGICAL_CORES = os.cpu_count() or 4


def apple_performance_cores() -> int | None:
    if sys.platform != "darwin":
        return None
    for key in ("hw.perflevel0.physicalcpu_max", "hw.physicalcpu_max"):
        try:
            value = subprocess.check_output(
                ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
            ).strip()
            cores = int(value)
            if cores > 0:
                return cores
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_bool_string(name: str, default: bool) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return "true" if default else "false"
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "true"
    if value in {"0", "false", "no", "off"}:
        return "false"
    return "true" if default else "false"


PERFORMANCE_CORES = apple_performance_cores()
MAX_SEARCH_WORKERS = max(3, PERFORMANCE_CORES or (LOGICAL_CORES - 1))
REQUESTED_SEARCH_WORKERS = max(
    0,
    min(
        MAX_SEARCH_WORKERS,
        env_int("METALFISH_SEARCH_WORKERS", env_int("METALFISH_THREADS", 0)),
    ),
)
HYBRID_MCTS_THREADS = max(
    0,
    min(MAX_SEARCH_WORKERS - 1, env_int("METALFISH_HYBRID_MCTS_THREADS", 0)),
)
HYBRID_AB_THREADS = max(
    0, min(MAX_SEARCH_WORKERS, env_int("METALFISH_HYBRID_AB_THREADS", 0))
)
HYBRID_AUTO_AB_THREADS_CAP = max(
    0, min(MAX_SEARCH_WORKERS, env_int("METALFISH_HYBRID_AUTO_AB_THREADS_CAP", 0))
)
HYBRID_MCTS_KLD = max(0.0, min(1.0, env_float("METALFISH_HYBRID_MCTS_KLD", 0.0)))
HYBRID_MCTS_ROOT_REJECT = env_bool_string("METALFISH_HYBRID_MCTS_ROOT_REJECT", True)
HYBRID_MCTS_SHARED_TT = env_bool_string("METALFISH_HYBRID_MCTS_SHARED_TT", False)
HYBRID_MCTS_AB_ROOT_HINTS = env_bool_string("METALFISH_HYBRID_MCTS_AB_ROOT_HINTS", True)
HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS = max(
    0, min(1000, env_int("METALFISH_HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS", 25))
)
HYBRID_MCTS_AB_ROOT_HINT_COUNT = max(
    1, min(16, env_int("METALFISH_HYBRID_MCTS_AB_ROOT_HINT_COUNT", 8))
)
HYBRID_AB_CANDIDATE_VERIFY_MS = max(
    0, min(1000, env_int("METALFISH_HYBRID_AB_CANDIDATE_VERIFY_MS", 120))
)
HYBRID_AB_CANDIDATE_VERIFY_COUNT = max(
    1, min(10, env_int("METALFISH_HYBRID_AB_CANDIDATE_VERIFY_COUNT", 4))
)
HYBRID_AB_POLICY_WEIGHT = max(
    0.0, min(1.0, env_float("METALFISH_HYBRID_AB_POLICY_WEIGHT", 0.0))
)
HYBRID_ROOT_PAWN_LEVER_TIEBREAK = env_bool_string(
    "METALFISH_HYBRID_ROOT_PAWN_LEVER_TIEBREAK", True
)
HYBRID_TRACE = env_bool_string("METALFISH_HYBRID_TRACE", False)
HYBRID_MCTS_MINIBATCH = max(0, min(4096, env_int("METALFISH_HYBRID_MCTS_MINIBATCH", 0)))
TRANSFORMER_LOW_TIME_FALLBACK_MS = max(
    0, min(30000, env_int("METALFISH_TRANSFORMER_LOW_TIME_FALLBACK_MS", 3000))
)
TRANSFORMER_MIN_MOVE_BUDGET_MS = max(
    0, min(5000, env_int("METALFISH_TRANSFORMER_MIN_MOVE_BUDGET_MS", 400))
)
DEFAULT_SYZYGY_PATH = PROJ / "syzygy"


def syzygy_path_is_safe(path: pathlib.Path) -> bool:
    if not path.is_dir():
        return False
    if not any(path.glob("*.rtbw")) or not any(path.glob("*.rtbz")):
        return False
    try:
        import chess.syzygy

        with chess.syzygy.open_tablebase(str(path)) as tablebase:
            for fen in (
                "7k/8/8/8/8/8/QRR5/K7 w - - 0 1",
                "7k/8/8/8/8/8/6R1/K7 w - - 0 1",
            ):
                board = chess.Board(fen)
                tablebase.probe_wdl(board)
                tablebase.probe_dtz(board)
            if (path / "KPPvKPP.rtbw").exists():
                board = chess.Board("7k/6pp/8/8/8/8/1PP5/K7 w - - 0 1")
                tablebase.probe_wdl(board)
        return True
    except Exception:
        return False


def resolve_syzygy_path() -> str:
    explicit = os.environ.get("METALFISH_SYZYGY_PATH", "").strip()
    candidates = [pathlib.Path(explicit)] if explicit else [DEFAULT_SYZYGY_PATH]
    for path in candidates:
        if syzygy_path_is_safe(path):
            return str(path)
    return ""


SYZYGY_PATH = resolve_syzygy_path()
RESOURCE_RESERVE_MB = max(1024, env_int("METALFISH_RESOURCE_RESERVE_MB", 2048))
LOAD_THROTTLE_START = max(0.0, env_float("METALFISH_LOAD_THROTTLE_START", 0.75))
LOAD_THROTTLE_HIGH = max(
    LOAD_THROTTLE_START, env_float("METALFISH_LOAD_THROTTLE_HIGH", 1.10)
)
LOAD_THROTTLE_EXTREME = max(
    LOAD_THROTTLE_HIGH, env_float("METALFISH_LOAD_THROTTLE_EXTREME", 1.50)
)
DEFER_SEEK_LOAD = max(
    LOAD_THROTTLE_EXTREME, env_float("METALFISH_DEFER_SEEK_LOAD", 1.75)
)
BOOK_MAX_PLY = max(0, env_int("METALFISH_BOOK_MAX_PLY", 20))
BOOK_MIN_CLOCK_MS = 30_000
SUBMITTED_TURN_HISTORY_LIMIT = 512
COMPLETED_GAME_HISTORY_LIMIT = 512
MIN_RATED_OPPONENT_ELO = max(0, env_int("METALFISH_MIN_RATED_OPPONENT_ELO", 2200))
BOOK_TIMEOUT_S = max(0.05, min(5.0, env_float("METALFISH_BOOK_TIMEOUT_S", 1.0)))
BOOK_CACHE_PATH = pathlib.Path(
    os.environ.get(
        "METALFISH_BOOK_CACHE",
        str(PROJ / "results" / "lichess_book_cache.json"),
    )
)
BOOK_CACHE_LIMIT = max(0, env_int("METALFISH_BOOK_CACHE_LIMIT", 50000))
DEFAULT_BOOK_DIR = PROJ / "books"
BOOK_ALLOW_ONLINE = env_bool_string("METALFISH_BOOK_ALLOW_ONLINE", False) == "true"
BOT_LOCK_PATH = pathlib.Path(
    os.environ.get(
        "METALFISH_BOT_LOCK",
        str(PROJ / "results" / "lichess_bot.lock"),
    )
)
ALLOW_CONCURRENT_BOT = (
    env_bool_string("METALFISH_ALLOW_CONCURRENT_BOT", False) == "true"
)
ENGINE_STOP_GRACE_S = 3.0
PONDER_STOP_TIMEOUT_S = 4.0
PONDER_HIT_TIMEOUT_S = 6.0
MIN_SEARCH_TIMEOUT_S = 0.2
MAX_SEARCH_TIMEOUT_S = 30.0
GAME_STREAM_TIMEOUT_S = max(
    30.0, min(1800.0, env_float("METALFISH_GAME_STREAM_TIMEOUT_S", 180.0))
)
GAME_STREAM_RETRIES = max(2, env_int("METALFISH_GAME_STREAM_RETRIES", 6))
GAME_STREAM_ACTIVE_RETRIES = max(
    8, env_int("METALFISH_GAME_STREAM_ACTIVE_RETRIES", 24)
)
GAME_STREAM_RETRY_DELAY_S = max(
    0.0, env_float("METALFISH_GAME_STREAM_RETRY_DELAY_S", 1.0)
)
GAME_STREAM_TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}
EVENT_STREAM_READ_TIMEOUT_S = env_float("METALFISH_EVENT_STREAM_READ_TIMEOUT_S", 300.0)
EVENT_STREAM_RECONNECT_DELAY_S = env_float(
    "METALFISH_EVENT_STREAM_RECONNECT_DELAY_S", 1.0
)
LICHESS_API_MIN_INTERVAL_S = env_float("METALFISH_LICHESS_API_MIN_INTERVAL_S", 0.35)
EXPLORER_API_MIN_INTERVAL_S = env_float("METALFISH_EXPLORER_API_MIN_INTERVAL_S", 0.25)
LICHESS_429_BACKOFF_S = env_float("METALFISH_LICHESS_429_BACKOFF_S", 65.0)
BOT_ONLINE_FETCH_LIMIT = max(
    1, min(512, env_int("METALFISH_BOT_ONLINE_FETCH_LIMIT", 512))
)
BOT_ONLINE_CACHE_TTL_S = env_float("METALFISH_BOT_ONLINE_CACHE_TTL_S", 20.0)
PLAYING_STATUS_CACHE_TTL_S = env_float("METALFISH_PLAYING_STATUS_CACHE_TTL_S", 5.0)
CHALLENGE_COOLDOWN_PATH = pathlib.Path(
    os.environ.get(
        "METALFISH_CHALLENGE_COOLDOWN_FILE",
        str(PROJ / "results" / "lichess_challenge_cooldowns.json"),
    )
)
PLAYED_FORMAT_HISTORY_PATH = pathlib.Path(
    os.environ.get(
        "METALFISH_PLAYED_FORMAT_FILE",
        str(PROJ / "results" / "lichess_played_formats.json"),
    )
)
PLAYED_FORMAT_HISTORY_LIMIT = max(
    128, min(20000, env_int("METALFISH_PLAYED_FORMAT_HISTORY_LIMIT", 4096))
)
PRE_GAME_RESOURCE_PREP = (
    env_bool_string("METALFISH_PRE_GAME_RESOURCE_PREP", True) == "true"
)
PRE_GAME_PURGE_MODE = os.environ.get("METALFISH_PRE_GAME_PURGE", "auto").strip().lower()
PRE_GAME_PURGE_TIMEOUT_S = max(
    0.1, min(10.0, env_float("METALFISH_PRE_GAME_PURGE_TIMEOUT_S", 2.0))
)
PRE_GAME_PURGE_THRESHOLD_MB = max(
    RESOURCE_RESERVE_MB,
    env_int("METALFISH_PRE_GAME_PURGE_THRESHOLD_MB", RESOURCE_RESERVE_MB * 2),
)
LICHESS_AUDIT_ENABLED = env_bool_string("METALFISH_LICHESS_AUDIT", True) == "true"
LICHESS_AUDIT_DIR = pathlib.Path(
    os.environ.get(
        "METALFISH_LICHESS_AUDIT_DIR",
        str(PROJ / "results" / "lichess_audit"),
    )
)
LICHESS_AUDIT_EVENT_LIMIT = max(
    128, min(20000, env_int("METALFISH_LICHESS_AUDIT_EVENT_LIMIT", 4096))
)
LICHESS_AUDIT_FIELD_LIMIT = max(
    64, min(4096, env_int("METALFISH_LICHESS_AUDIT_FIELD_LIMIT", 512))
)
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


def available_memory_mb() -> int:
    if sys.platform == "darwin":
        try:
            page_size = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.pagesize"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
            out = subprocess.check_output(
                ["vm_stat"], text=True, stderr=subprocess.DEVNULL
            )
            pages: dict[str, int] = {}
            for line in out.splitlines():
                match = re.match(r"Pages ([^:]+):\s+([0-9]+)\.", line.strip())
                if match:
                    pages[match.group(1)] = int(match.group(2))
            free_pages = pages.get("free", 0)
            speculative_pages = pages.get("speculative", 0)
            purgeable_pages = pages.get("purgeable", 0)
            inactive_pages = pages.get("inactive", 0)
            available_pages = (
                free_pages + speculative_pages + purgeable_pages + inactive_pages // 2
            )
            return (available_pages * page_size) // (1024 * 1024)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    meminfo = pathlib.Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
        except (OSError, ValueError):
            pass

    return 0


def system_load_ratio() -> float:
    try:
        return os.getloadavg()[0] / max(1, LOGICAL_CORES)
    except (AttributeError, OSError):
        return 0.0


def load_adjusted_workers(workers: int, load_ratio: float) -> int:
    if load_ratio >= LOAD_THROTTLE_EXTREME:
        workers = workers // 2
    elif load_ratio >= LOAD_THROTTLE_HIGH:
        workers -= 2
    elif load_ratio >= LOAD_THROTTLE_START:
        workers -= 1
    return max(3, workers)


def dynamic_search_workers(active_peer_engines: int = 0) -> int:
    if REQUESTED_SEARCH_WORKERS > 0:
        return REQUESTED_SEARCH_WORKERS

    workers = MAX_SEARCH_WORKERS
    available_mb = available_memory_mb()

    if active_peer_engines > 0:
        workers = max(3, workers // (active_peer_engines + 1))

    if available_mb and available_mb < RESOURCE_RESERVE_MB:
        workers -= 2
    elif available_mb and available_mb < RESOURCE_RESERVE_MB * 2:
        workers -= 1

    workers = load_adjusted_workers(workers, system_load_ratio())

    return max(3, min(MAX_SEARCH_WORKERS, workers))


def should_run_pre_game_purge(available_mb: int) -> bool:
    if not PRE_GAME_RESOURCE_PREP:
        return False
    if PRE_GAME_PURGE_MODE in {"0", "false", "no", "off", "never"}:
        return False
    if sys.platform != "darwin" or not pathlib.Path("/usr/bin/purge").exists():
        return False
    if PRE_GAME_PURGE_MODE in {"1", "true", "yes", "on", "always"}:
        return True
    return bool(available_mb and available_mb < PRE_GAME_PURGE_THRESHOLD_MB)


def run_pre_game_memory_purge(timeout_s: float = PRE_GAME_PURGE_TIMEOUT_S) -> bool:
    try:
        result = subprocess.run(
            ["/usr/bin/purge"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return False


def local_hash_mb(active_peer_engines: int = 0) -> int:
    requested = env_int("METALFISH_HASH_MB", 0)
    if requested > 0:
        return max(128, min(32768, requested))

    memory_mb = machine_memory_mb()
    if memory_mb <= 0:
        return 4096

    available_mb = available_memory_mb()
    engine_count = max(1, active_peer_engines + 1)
    if available_mb > 0:
        target = (available_mb - RESOURCE_RESERVE_MB) // engine_count
    else:
        target = (memory_mb * 3 // 8) // engine_count

    target = min(target, (memory_mb * 3) // 4, 32768)

    if target >= 1024:
        return max(1024, (target // 1024) * 1024)
    return max(512, (target // 256) * 256)


def live_resource_profile(active_peer_engines: int = 0) -> dict[str, float | int]:
    return {
        "threads": dynamic_search_workers(active_peer_engines),
        "hash_mb": local_hash_mb(active_peer_engines),
        "available_mb": available_memory_mb(),
        "total_mb": machine_memory_mb(),
        "load_ratio": system_load_ratio(),
    }


BASE_ENGINE_OPTIONS = {
    "Ponder": "true",
    "UseMCTS": "false",
    "UseHybridSearch": "true",
    "NNWeights": str(WEIGHTS),
    "MultiPV": "1",
    "HybridMCTSThreads": str(HYBRID_MCTS_THREADS),
    "HybridABThreads": str(HYBRID_AB_THREADS),
    "HybridAutoABThreadsCap": str(HYBRID_AUTO_AB_THREADS_CAP),
    "TransformerLowTimeFallbackMs": str(TRANSFORMER_LOW_TIME_FALLBACK_MS),
    "TransformerMinMoveBudgetMs": str(TRANSFORMER_MIN_MOVE_BUDGET_MS),
    "MCTSMaxThreads": str(HYBRID_MCTS_THREADS),
    "MCTSParityPreset": "false",
    "MCTSAddDirichletNoise": "false",
    "HybridMCTSMinimumKLDGainPerNode": str(HYBRID_MCTS_KLD),
    "HybridMCTSRootReject": HYBRID_MCTS_ROOT_REJECT,
    "HybridMCTSUseSharedTT": HYBRID_MCTS_SHARED_TT,
    "HybridMCTSABRootHints": HYBRID_MCTS_AB_ROOT_HINTS,
    "HybridMCTSABRootHintDelayMs": str(HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS),
    "HybridMCTSABRootHintCount": str(HYBRID_MCTS_AB_ROOT_HINT_COUNT),
    "HybridABCandidateVerifyMs": str(HYBRID_AB_CANDIDATE_VERIFY_MS),
    "HybridABCandidateVerifyCount": str(HYBRID_AB_CANDIDATE_VERIFY_COUNT),
    "HybridABPolicyWeight": str(HYBRID_AB_POLICY_WEIGHT),
    "HybridRootPawnLeverTieBreak": HYBRID_ROOT_PAWN_LEVER_TIEBREAK,
    "HybridTrace": HYBRID_TRACE,
    "Move Overhead": "500",
    "MCTSMinibatchSize": str(HYBRID_MCTS_MINIBATCH),
    "MCTSMinimumKLDGainPerNode": "0.00005",
    "MCTSPolicySoftmaxTemp": "1.359",
    "MCTSSmartPruningFactor": "1.33",
    "MCTSCacheHistoryLength": "0",
    "MCTSSolidTreeThreshold": "100",
}

if SYZYGY_PATH:
    BASE_ENGINE_OPTIONS["SyzygyPath"] = SYZYGY_PATH
    BASE_ENGINE_OPTIONS["SyzygyProbeDepth"] = "2"
    BASE_ENGINE_OPTIONS["SyzygyProbeLimit"] = "6"


def build_engine_options(active_peer_engines: int = 0) -> tuple[dict[str, str], dict]:
    profile = live_resource_profile(active_peer_engines)
    threads = int(profile["threads"])
    options = dict(BASE_ENGINE_OPTIONS)
    options["Threads"] = str(threads)
    options["Hash"] = str(int(profile["hash_mb"]))

    if HYBRID_MCTS_THREADS > 0:
        mcts_threads = min(HYBRID_MCTS_THREADS, max(1, threads - 1))
        options["HybridMCTSThreads"] = str(mcts_threads)
        options["MCTSMaxThreads"] = str(mcts_threads)
    else:
        options["HybridMCTSThreads"] = "0"
        options["MCTSMaxThreads"] = "0"

    if HYBRID_AB_THREADS > 0:
        options["HybridABThreads"] = str(min(HYBRID_AB_THREADS, threads))
    else:
        options["HybridABThreads"] = "0"

    if HYBRID_AUTO_AB_THREADS_CAP > 0:
        options["HybridAutoABThreadsCap"] = str(
            min(HYBRID_AUTO_AB_THREADS_CAP, threads)
        )
    else:
        options["HybridAutoABThreadsCap"] = "0"

    return options, profile


ENGINE_OPTIONS, RESOURCE_PROFILE = build_engine_options()

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
CHALLENGE_TIMEOUT = max(20.0, env_float("METALFISH_CHALLENGE_TIMEOUT_S", 21.0))
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


def args_seek_is_rated(args) -> bool:
    return bool(args.seek and args.accept_rated and not args.accept_casual)


def validate_bot_config(args) -> list[str]:
    errors: list[str] = []
    if not args.accept_rated and not args.accept_casual:
        errors.append("No game type enabled. Use --accept-rated or allow casual games.")

    floor = getattr(args, "min_rated_opponent_elo", MIN_RATED_OPPONENT_ELO)
    if floor < 0:
        errors.append("--min-rated-opponent-elo must be >= 0.")

    if getattr(args, "elo_seek", False) and getattr(args, "elo_range", 0) <= 0:
        errors.append("--elo-range must be > 0 when --elo-seek is enabled.")

    if args_seek_is_rated(args):
        if not (
            getattr(args, "elo_seek", False)
            or getattr(args, "seek_highest_rated", False)
        ):
            errors.append(
                "Rated outgoing seeks require --elo-seek or --seek-highest-rated."
            )
        if floor <= 0:
            errors.append("Rated outgoing seeks require --min-rated-opponent-elo > 0.")
    return errors


def print_config_check(args) -> None:
    checker = object.__new__(LichessBot)
    checker.args = args
    checker._elo_widen_steps = 0
    options, profile = build_engine_options()

    available_mb = int(profile.get("available_mb", 0))
    total_mb = int(profile.get("total_mb", 0))
    memory = f"{available_mb}/{total_mb} MB" if available_mb and total_mb else "unknown"
    syzygy = SYZYGY_PATH if SYZYGY_PATH else "disabled"

    print("MetalFish Lichess Bot config check")
    print(f"  Engine path: {ENGINE}")
    print(f"  Engine present: {ENGINE.exists()}")
    print(f"  Weights present: {WEIGHTS.exists()}")
    print(f"  Accepts: rated={args.accept_rated} | casual={args.accept_casual}")
    print(f"  Seek: {checker._seek_policy_label()}")
    print(f"  Incoming rated: {checker._rated_policy_label()}")
    print(
        "  Repeat seek: "
        f"{'avoid same bot/speed' if getattr(args, 'avoid_repeat_format', False) else 'allowed'}"
    )
    print(
        f"  Resources: {int(profile['threads'])} search threads, "
        f"Hash {int(profile['hash_mb'])} MB, Free {memory}, "
        f"load {float(profile['load_ratio']):.2f}"
    )
    print(f"  Resource gate allows new game: {checker._resources_allow_new_game()}")
    print(f"  Syzygy: {syzygy}")
    paths = configured_book_paths()
    if paths:
        print(
            f"  Book: {len(paths)} local book(s), online fallback={BOOK_ALLOW_ONLINE}"
        )
    else:
        print(f"  Book: no local book, online fallback={BOOK_ALLOW_ONLINE}")
    print(
        f"  Audit: {'enabled' if LICHESS_AUDIT_ENABLED else 'disabled'} "
        f"({LICHESS_AUDIT_DIR})"
    )
    print(f"  Ponder: {args.ponder}")


class BotInstanceLock:
    def __init__(self, path: pathlib.Path, enabled: bool = True):
        self.path = pathlib.Path(path)
        self.enabled = enabled and not ALLOW_CONCURRENT_BOT and fcntl is not None
        self._file = None

    def acquire(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.seek(0)
            owner = lock_file.read().strip()
            lock_file.close()
            detail = f" ({owner})" if owner else ""
            raise RuntimeError(
                f"another MetalFish Lichess bot is already running{detail}"
            ) from exc

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started": int(time.time()),
                    "argv": sys.argv,
                },
                sort_keys=True,
            )
        )
        lock_file.flush()
        self._file = lock_file

    def release(self) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


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
        self._stderr_tail: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._pondering = False
        self._ponder_move: str | None = None
        self._active_search: str | None = None
        self._position_key: tuple[str, tuple[str, ...]] | None = None
        self._uci_lock = threading.RLock()
        self._launch()

    def _launch(self):
        env = os.environ.copy()
        env["METALFISH_PRELOAD_TRANSFORMER"] = "1" if self.preload_transformer else "0"
        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                [str(self.path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
            output: queue.Queue[str | None] = queue.Queue()
            stderr_tail: list[str] = []
            self.proc = proc
            self._output = output
            self._stderr_tail = stderr_tail
            self._reader_thread = threading.Thread(
                target=self._read_stdout,
                args=(proc.stdout, output),
                daemon=True,
            )
            self._reader_thread.start()
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                args=(proc.stderr, stderr_tail),
                daemon=True,
            )
            self._stderr_thread.start()
            self._send("uci")
            self._wait_for("uciok")
            for name, value in self.options.items():
                self._send(f"setoption name {name} value {value}")
            self._send("isready")
            self._wait_for("readyok", timeout=120)
            self._pondering = False
            self._ponder_move: str | None = None
            self._active_search = None
            self._position_key = None
        except Exception:
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
            if proc is not None:
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    try:
                        if stream:
                            stream.close()
                    except Exception:
                        pass
            self.proc = None
            raise

    @staticmethod
    def _read_stdout(stream, output: "queue.Queue[str | None]"):
        try:
            if stream is None:
                return
            for line in stream:
                output.put(line.strip())
        finally:
            output.put(None)

    @staticmethod
    def _read_stderr(stream, tail: list[str]):
        if stream is None:
            return
        for line in stream:
            tail.append(line.strip())
            if len(tail) > 20:
                tail.pop(0)

    def _diagnostic_tail(self) -> str:
        parts = []
        if self.proc is not None and self.proc.poll() is not None:
            parts.append(f"exit={self.proc.returncode}")
        if self._stderr_tail:
            parts.append("stderr=" + " | ".join(self._stderr_tail[-3:]))
        return " (" + "; ".join(parts) + ")" if parts else ""

    def _ensure_state(self):
        if not hasattr(self, "_uci_lock"):
            self._uci_lock = threading.RLock()
        if not hasattr(self, "_pondering"):
            self._pondering = False
        if not hasattr(self, "_ponder_move"):
            self._ponder_move = None
        if not hasattr(self, "_active_search"):
            self._active_search = "ponder" if self._pondering else None
        if not hasattr(self, "_position_key"):
            self._position_key = None

    def _send(self, cmd: str, *, ignore_errors: bool = False):
        try:
            if self.proc is None or self.proc.stdin is None:
                if ignore_errors:
                    return
                raise RuntimeError("Engine process is not running")
            if self.proc.poll() is not None:
                if ignore_errors:
                    return
                raise RuntimeError(f"Engine process died{self._diagnostic_tail()}")
            assert self.proc.stdin is not None
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            if ignore_errors:
                return
            raise RuntimeError(
                f"Engine stdin write failed{self._diagnostic_tail()}"
            ) from exc

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
                    raise RuntimeError(f"Engine process died{self._diagnostic_tail()}")
                continue
            if line is None:
                raise RuntimeError(f"Engine closed stdout{self._diagnostic_tail()}")
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"Timeout waiting for '{prefix}'")

    def _drain_available_output(self):
        while True:
            try:
                self._output.get_nowait()
            except queue.Empty:
                return

    def _parse_bestmove(self, line: str) -> tuple[str, str | None]:
        parts = line.split()
        best = parts[1] if len(parts) > 1 else "0000"
        ponder = parts[3] if len(parts) > 3 and parts[2] == "ponder" else None
        return best, ponder

    def _clear_active_search(self):
        self._active_search = None
        self._pondering = False
        self._ponder_move = None

    def _make_position_key(
        self, initial_fen: str, moves: list[str]
    ) -> tuple[str, tuple[str, ...]]:
        return (str(initial_fen), tuple(str(move) for move in moves))

    def restart(self):
        self._ensure_state()
        with self._uci_lock:
            self._pondering = False
            self._ponder_move = None
            self._active_search = None
            self._position_key = None
            if self.proc is not None:
                try:
                    self._send("quit", ignore_errors=True)
                    self.proc.wait(timeout=2)
                except Exception:
                    try:
                        self.proc.kill()
                        self.proc.wait(timeout=2)
                    except Exception:
                        pass
                finally:
                    for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                        try:
                            if stream:
                                stream.close()
                        except Exception:
                            pass
            self._launch()
            self.new_game()

    def configure(self, options: dict[str, str]):
        self._ensure_state()
        with self._uci_lock:
            self.stop_pondering()
            changed = False
            for name, value in options.items():
                value = str(value)
                if self.options.get(name) == value:
                    continue
                self._send(f"setoption name {name} value {value}")
                changed = True
            self.options = dict(options)
            if changed:
                self._send("isready")
                self._wait_for("readyok", timeout=120)

    def new_game(self):
        self._ensure_state()
        with self._uci_lock:
            self.stop_pondering()
            self._send("ucinewgame")
            self._send("isready")
            self._wait_for("readyok", timeout=120)
            self._position_key = None

    def set_position(self, initial_fen: str, moves: list[str]):
        self._ensure_state()
        with self._uci_lock:
            self.stop_pondering()
            if initial_fen == "startpos":
                cmd = "position startpos"
            else:
                cmd = f"position fen {initial_fen}"
            if moves:
                cmd += " moves " + " ".join(moves)
            self._send(cmd)
            self._position_key = self._make_position_key(initial_fen, moves)

    def position_matches(self, initial_fen: str, moves: list[str]) -> bool:
        self._ensure_state()
        return self._position_key == self._make_position_key(initial_fen, moves)

    def sync_position(
        self, initial_fen: str, moves: list[str], timeout: float = 30
    ) -> bool:
        self._ensure_state()
        with self._uci_lock:
            self.set_position(initial_fen, moves)
            self._send("isready")
            self._wait_for("readyok", timeout=timeout)
            return self.position_matches(initial_fen, moves)

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
        self._ensure_state()
        with self._uci_lock:
            self.stop_pondering()
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
            self._active_search = "search"
            line = None
            try:
                line = self._wait_for("bestmove", timeout=timeout)
            except TimeoutError:
                self._send("stop")
                line = self._wait_for("bestmove", timeout=ENGINE_STOP_GRACE_S)
            finally:
                if line is not None:
                    self._clear_active_search()
            return self._parse_bestmove(line)

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
        self._ensure_state()
        with self._uci_lock:
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
            self._active_search = "ponder"

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

    def ponderhit(
        self, timeout: float = PONDER_HIT_TIMEOUT_S
    ) -> tuple[str, str | None]:
        """Opponent played the predicted move; use the current ponder result."""
        self._ensure_state()
        with self._uci_lock:
            if not self._pondering:
                return "0000", None
            line = None
            try:
                self._send("ponderhit")
                try:
                    line = self._wait_for("bestmove", timeout=timeout)
                except TimeoutError:
                    self._send("stop")
                    line = self._wait_for("bestmove", timeout=ENGINE_STOP_GRACE_S)
            finally:
                if line is not None:
                    self._clear_active_search()
            return self._parse_bestmove(line)

    def stop_pondering(
        self,
        timeout: float = PONDER_STOP_TIMEOUT_S,
        *,
        restart_on_failure: bool = True,
    ) -> bool:
        self._ensure_state()
        with self._uci_lock:
            if not self._pondering and self._active_search is None:
                return True

            ok = True
            try:
                self._send("stop")
                self._wait_for("bestmove", timeout=timeout)
            except (TimeoutError, RuntimeError):
                ok = False
            if ok:
                self._clear_active_search()
                self._drain_available_output()
            else:
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
        self._ensure_state()
        with self._uci_lock:
            self.stop_pondering(restart_on_failure=False)
            try:
                self._send("quit", ignore_errors=True)
                if self.proc is not None:
                    self.proc.wait(timeout=5)
            except Exception:
                try:
                    if self.proc is not None:
                        self.proc.kill()
                except Exception:
                    pass
            finally:
                self._clear_active_search()
                if self.proc is not None:
                    for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                        try:
                            if stream:
                                stream.close()
                        except Exception:
                            pass


def _retry_after_seconds(response) -> float:
    header = getattr(response, "headers", {}).get("Retry-After")
    if not header:
        return LICHESS_429_BACKOFF_S
    try:
        return max(LICHESS_429_BACKOFF_S, float(header))
    except (TypeError, ValueError):
        pass
    try:
        retry_time = email.utils.parsedate_to_datetime(header)
        return max(LICHESS_429_BACKOFF_S, retry_time.timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        return LICHESS_429_BACKOFF_S


class SerializedHttpClient:
    def __init__(
        self,
        *,
        min_interval_s: float,
        auth_token: str | None = None,
    ):
        self.min_interval_s = max(0.0, min_interval_s)
        self.headers = {"User-Agent": HTTP_USER_AGENT}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        self._lock = threading.Lock()
        self._next_request_at = 0.0
        self._blocked_until = 0.0
        self._session = (
            requests.Session()
            if requests is not None and hasattr(requests, "Session")
            else None
        )

    def request(self, method: str, url: str, **kwargs):
        headers = dict(self.headers)
        if kwargs.get("headers"):
            headers.update(kwargs.pop("headers"))
        with self._lock:
            now = time.monotonic()
            wait_s = max(self._next_request_at - now, self._blocked_until - now)
            if wait_s > 0:
                time.sleep(wait_s)
            requester = None
            if self._session is not None:
                requester = self._session.request
            elif requests is not None:
                requester = getattr(requests, method.lower())
            if requester is None:
                raise RuntimeError("Python package 'requests' is required")
            if self._session is not None:
                response = requester(method, url, headers=headers, **kwargs)
            else:
                response = requester(url, headers=headers, **kwargs)
            self._next_request_at = time.monotonic() + self.min_interval_s
            if getattr(response, "status_code", None) == 429:
                self._blocked_until = time.monotonic() + _retry_after_seconds(response)
            return response

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)


def configured_book_paths() -> list[pathlib.Path]:
    raw = os.environ.get("METALFISH_BOOK_PATH", "").strip()
    if raw:
        candidates = [
            pathlib.Path(part).expanduser()
            for part in raw.split(os.pathsep)
            if part.strip()
        ]
    else:
        candidates = []
        for pattern in ("*.bin", "*.book", "*.polyglot"):
            candidates.extend(sorted(DEFAULT_BOOK_DIR.glob(pattern)))
    return [path for path in candidates if path.is_file()]


class OpeningBook:
    def __init__(
        self,
        api_key: str = "",
        min_games: int = 5,
        timeout: float = 1.0,
        *,
        allow_online: bool = False,
        cache_path: pathlib.Path | None = None,
        book_paths: list[pathlib.Path] | None = None,
    ):
        self.min_games = min_games
        self.timeout = timeout
        self.allow_online = allow_online
        self.cache_path = cache_path
        self.http = SerializedHttpClient(min_interval_s=EXPLORER_API_MIN_INTERVAL_S)
        self._cache: dict[str, str] = self._load_cache()
        self._miss_cache: set[str] = set()
        self._readers: list[chess.polyglot.MemoryMappedReader] = []
        self._book_paths: list[pathlib.Path] = []
        for path in book_paths or []:
            try:
                self._readers.append(chess.polyglot.open_reader(str(path)))
                self._book_paths.append(path)
            except Exception:
                pass

    def lookup(self, fen: str) -> str | None:
        local = self._lookup_local(fen)
        if local:
            return local
        if fen in self._cache:
            return self._cache[fen]
        if fen in self._miss_cache:
            return None
        if not self.allow_online:
            return None
        move = self._query_masters(fen) or self._query_lichess(fen)
        if move:
            self._remember(fen, move)
        else:
            self._miss_cache.add(fen)
        return move

    def description(self) -> str:
        if self._book_paths:
            names = ", ".join(path.name for path in self._book_paths[:3])
            suffix = "..." if len(self._book_paths) > 3 else ""
            source = f"{len(self._book_paths)} local ({names}{suffix})"
        else:
            source = "no local book"
        fallback = "online fallback on" if self.allow_online else "online fallback off"
        return f"{source}, {fallback}"

    def close(self) -> None:
        for reader in self._readers:
            try:
                reader.close()
            except Exception:
                pass
        self._readers = []

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(fen): str(move)
            for fen, move in data.items()
            if isinstance(fen, str) and isinstance(move, str)
        }

    def _remember(self, fen: str, move: str) -> None:
        if BOOK_CACHE_LIMIT <= 0:
            return
        self._cache[fen] = move
        while len(self._cache) > BOOK_CACHE_LIMIT:
            self._cache.pop(next(iter(self._cache)))
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._cache, sort_keys=True))
            tmp.replace(self.cache_path)
        except OSError:
            pass

    def _lookup_local(self, fen: str) -> str | None:
        if not self._readers:
            return None
        try:
            board = chess.Board(fen)
        except ValueError:
            return None
        weights: dict[str, int] = {}
        for reader in self._readers:
            try:
                entries = list(reader.find_all(board))
            except Exception:
                continue
            for entry in entries:
                move_attr = entry.move
                move = move_attr() if callable(move_attr) else move_attr
                if move in board.legal_moves:
                    weights[move.uci()] = weights.get(move.uci(), 0) + entry.weight
        if not weights:
            return None
        return max(weights.items(), key=lambda item: (item[1], item[0]))[0]

    def _query_masters(self, fen: str) -> str | None:
        try:
            r = self.http.get(
                f"{EXPLORER_API}/masters",
                params={
                    "fen": fen,
                    "topGames": 0,
                    "recentGames": 0,
                },
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return self._pick_best_move(r.json(), fen)
        except Exception:
            pass
        return None

    def _query_lichess(self, fen: str) -> str | None:
        try:
            r = self.http.get(
                f"{EXPLORER_API}/lichess",
                params={
                    "fen": fen,
                    "ratings": "2200,2500",
                    "speeds": "blitz,rapid,classical",
                    "topGames": 0,
                    "recentGames": 0,
                },
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return self._pick_best_move(r.json(), fen)
        except Exception:
            pass
        return None

    def _pick_best_move(self, data: dict, fen: str) -> str | None:
        moves = data.get("moves", [])
        if not moves:
            return None
        fen_parts = fen.split()
        black_to_move = len(fen_parts) > 1 and fen_parts[1] == "b"
        best, best_score = None, -1.0
        for m in moves:
            games = m.get("white", 0) + m.get("draws", 0) + m.get("black", 0)
            if games < self.min_games:
                continue
            wins = m.get("black" if black_to_move else "white", 0)
            wins += m.get("draws", 0) * 0.5
            score = (wins / games) * (games**0.3) if games > 0 else 0
            if score > best_score:
                best_score = score
                best = m.get("uci")
        return best


class LichessBot:
    def __init__(self, api_key: str, args):
        self.api_key = api_key
        self.http = SerializedHttpClient(
            min_interval_s=LICHESS_API_MIN_INTERVAL_S,
            auth_token=api_key,
        )
        self.headers = dict(self.http.headers)
        self.args = args
        self.active_games: dict[str, threading.Thread] = {}
        self.bot_id = ""
        self.username = ""
        self._rotation_idx = 0
        self._pending_challenge_id: str | None = None
        self._pending_challenge_target: str | None = None
        self._pending_challenge_speed: str | None = None
        self._challenge_sent_at: float = 0
        self._challenge_retries = 0
        self.book = OpeningBook(
            api_key=api_key,
            min_games=5,
            timeout=BOOK_TIMEOUT_S,
            allow_online=BOOK_ALLOW_ONLINE,
            cache_path=BOOK_CACHE_PATH if BOOK_ALLOW_ONLINE else None,
            book_paths=configured_book_paths(),
        )
        self._seek_timer: threading.Timer | None = None
        self._warm_engine: UCIEngine | None = None
        self._elo_widen_steps = 0
        self._persist_challenge_cooldowns = True
        self._declined_cooldown: dict[str, float] = self._load_challenge_cooldowns()
        self._persist_played_format_history = True
        self._played_by_speed: dict[str, dict[str, float]] = (
            self._load_played_format_history()
        )
        self._rate_limit_count = 0
        self._tc_failures = 0
        self._completed_games = 0
        self._completed_game_ids: set[str] = set()
        self._completed_game_order: list[str] = []
        self._seek_lock = threading.Lock()
        self._submitted_turns: dict[str, list[tuple[str, ...]]] = {}
        self._max_seen_ply: dict[str, int] = {}
        self._submitted_turns_lock = threading.Lock()
        self._ponder_disabled_games: set[str] = set()
        self._ponder_disabled_lock = threading.Lock()
        self._last_resource_profile: dict[str, float | int] | None = None
        self._last_move_failure_detail = ""
        self._online_bots_cache: tuple[float, list[dict]] | None = None
        self._playing_status_cache: dict[str, tuple[float, bool | None]] = {}
        self._last_game_stream_status: int | None = None
        self._last_game_stream_error = ""
        self._audit_enabled = LICHESS_AUDIT_ENABLED
        self._audit_lock = threading.Lock()
        self._audit_counts: dict[str, int] = {}
        self._draining = threading.Event()
        self._shutdown = threading.Event()

    def api_get(self, path: str, **kwargs):
        return self.http.get(f"{LICHESS_API}{path}", **kwargs)

    def api_post(self, path: str, **kwargs):
        return self.http.post(f"{LICHESS_API}{path}", **kwargs)

    def _audit_path(self, game_id: str) -> pathlib.Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(game_id))[:64]
        return LICHESS_AUDIT_DIR / f"{safe_id or 'game'}.jsonl"

    def _audit_value(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > LICHESS_AUDIT_FIELD_LIMIT:
                return value[: LICHESS_AUDIT_FIELD_LIMIT - 3] + "..."
            return value
        if isinstance(value, (list, tuple)):
            return [self._audit_value(v) for v in list(value)[:32]]
        if isinstance(value, dict):
            out = {}
            for idx, (key, item) in enumerate(value.items()):
                if idx >= 32:
                    break
                out[str(key)] = self._audit_value(item)
            return out
        text = str(value)
        if len(text) > LICHESS_AUDIT_FIELD_LIMIT:
            text = text[: LICHESS_AUDIT_FIELD_LIMIT - 3] + "..."
        return text

    def _audit_context(self, moves: list[str]) -> dict:
        return {"ply": len(moves), "moves_tail": moves[-8:]}

    def _audit(self, game_id: str, event: str, **fields) -> None:
        if not game_id or not getattr(self, "_audit_enabled", False):
            return
        if not hasattr(self, "_audit_lock"):
            self._audit_lock = threading.Lock()
            self._audit_counts = {}

        with self._audit_lock:
            count = self._audit_counts.get(game_id, 0)
            if count >= LICHESS_AUDIT_EVENT_LIMIT:
                return
            record = {
                "ts": round(time.time(), 3),
                "event": event,
            }
            for key, value in fields.items():
                record[str(key)] = self._audit_value(value)
            try:
                LICHESS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
                with self._audit_path(game_id).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                self._audit_counts[game_id] = count + 1
            except OSError:
                return

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
        self._last_move_failure_detail = ""
        r = self.api_post(f"/bot/game/{game_id}/move/{move}")
        if r.status_code != 200:
            detail = r.text.strip().replace("\n", " ")[:200]
            suffix = f": {detail}" if detail else ""
            self._last_move_failure_detail = f"{r.status_code}{suffix}"
            print(f"  [{game_id}] Move {move} failed: {r.status_code}{suffix}")
            return False
        return True

    def _same_game_id(self, lhs: str | None, rhs: str | None) -> bool:
        if not lhs or not rhs:
            return False
        lhs = str(lhs)
        rhs = str(rhs)
        if lhs == rhs:
            return True
        return min(len(lhs), len(rhs)) >= 8 and (
            lhs.startswith(rhs) or rhs.startswith(lhs)
        )

    def _game_active_status(self, game_id: str) -> bool | None:
        now = time.time()
        cache = getattr(self, "_playing_status_cache", {})
        cached = cache.get(game_id)
        if cached is not None and cached[0] > now:
            return cached[1]

        result: bool | None = None
        try:
            r = self.api_get("/account/playing", timeout=10)
            if r.status_code != 200:
                cache[game_id] = (now + PLAYING_STATUS_CACHE_TTL_S, result)
                self._playing_status_cache = cache
                return result
            data = r.json()
        except Exception:
            cache[game_id] = (now + PLAYING_STATUS_CACHE_TTL_S, result)
            self._playing_status_cache = cache
            return result

        games = data.get("nowPlaying", []) if isinstance(data, dict) else []
        if not isinstance(games, list):
            cache[game_id] = (now + PLAYING_STATUS_CACHE_TTL_S, result)
            self._playing_status_cache = cache
            return result
        for game in games:
            if not isinstance(game, dict):
                continue
            for key in ("gameId", "id", "fullId"):
                if self._same_game_id(game.get(key), game_id):
                    result = True
                    cache[game_id] = (now + PLAYING_STATUS_CACHE_TTL_S, result)
                    self._playing_status_cache = cache
                    return result
        result = False
        cache[game_id] = (now + PLAYING_STATUS_CACHE_TTL_S, result)
        self._playing_status_cache = cache
        return result

    def _stream_moves(self, state: dict) -> list[str]:
        moves = state.get("moves", "") if isinstance(state, dict) else ""
        return moves.split() if isinstance(moves, str) and moves else []

    def _already_submitted_for_turn(self, game_id: str, moves: list[str]) -> bool:
        key = tuple(moves)
        with self._submitted_turns_lock:
            return key in self._submitted_turns.get(game_id, [])

    def _remember_stream_ply(self, game_id: str, moves: list[str]) -> bool:
        ply = len(moves)
        with self._submitted_turns_lock:
            if not hasattr(self, "_max_seen_ply"):
                self._max_seen_ply = {}
            max_seen = self._max_seen_ply.get(game_id, -1)
            if ply < max_seen:
                return False
            self._max_seen_ply[game_id] = ply
            return True

    def _record_submitted_turn(self, game_id: str, moves: list[str]) -> None:
        key = tuple(moves)
        with self._submitted_turns_lock:
            history = self._submitted_turns.setdefault(game_id, [])
            if key in history:
                return
            history.append(key)
            overflow = len(history) - SUBMITTED_TURN_HISTORY_LIMIT
            if overflow > 0:
                del history[:overflow]

    def _move_failure_looks_stale(self) -> bool:
        detail = self._last_move_failure_detail.lower()
        return any(
            marker in detail
            for marker in (
                "not your turn",
                "game already over",
                "cannot move",
                "no piece",
                "illegal move",
            )
        )

    def _submit_move(self, game_id: str, moves: list[str], move: str) -> bool:
        if self.make_move(game_id, move):
            self._record_submitted_turn(game_id, moves)
            self._audit(
                game_id,
                "move_submit",
                move=move,
                result="accepted",
                **self._audit_context(moves),
            )
            return True

        if self._move_failure_looks_stale():
            self._record_submitted_turn(game_id, moves)
            print(
                f"  [{game_id}] Suppressing retries for stale turn after "
                f"rejected move {move}"
            )
        self._audit(
            game_id,
            "move_submit",
            move=move,
            result="rejected",
            detail=self._last_move_failure_detail,
            stale=self._move_failure_looks_stale(),
            **self._audit_context(moves),
        )
        return False

    def _ponder_allowed_for_game(self, game_id: str) -> bool:
        with self._ponder_disabled_lock:
            return game_id not in self._ponder_disabled_games

    def _disable_ponder_for_game(self, game_id: str, reason: str) -> None:
        with self._ponder_disabled_lock:
            already_disabled = game_id in self._ponder_disabled_games
            self._ponder_disabled_games.add(game_id)
        if not already_disabled:
            print(f"  [{game_id}] Disabling ponder for this game: {reason}")
            self._audit(game_id, "ponder_disabled", reason=reason)

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

    def _active_peer_engines(self) -> int:
        return max(0, len(self.active_games) - 1)

    def _live_engine_options(self) -> tuple[dict[str, str], dict]:
        options, profile = build_engine_options(self._active_peer_engines())
        options["Ponder"] = "true" if self.args.ponder else "false"
        return options, profile

    def _print_resource_profile(self, game_id: str | None, profile: dict) -> None:
        prefix = f"  [{game_id}] " if game_id else "  "
        available_mb = int(profile.get("available_mb", 0))
        total_mb = int(profile.get("total_mb", 0))
        load_ratio = float(profile.get("load_ratio", 0.0))
        memory = (
            f"{available_mb}/{total_mb} MB" if available_mb and total_mb else "unknown"
        )
        print(
            f"{prefix}Resources: {int(profile['threads'])} search threads, "
            f"Hash {int(profile['hash_mb'])} MB, "
            f"memory available {memory}, reserve {RESOURCE_RESERVE_MB} MB, "
            f"load {load_ratio:.2f}"
        )

    def _prepare_game_resources(self, game_id: str | None = None) -> None:
        if not PRE_GAME_RESOURCE_PREP:
            return

        prefix = f"  [{game_id}] " if game_id else "  "
        before_mb = available_memory_mb()
        collected = gc.collect()
        purge_status = "skipped"
        if should_run_pre_game_purge(before_mb):
            purge_status = "ok" if run_pre_game_memory_purge() else "failed"
        after_mb = available_memory_mb()

        if purge_status != "skipped" or before_mb != after_mb:
            before = f"{before_mb} MB" if before_mb else "unknown"
            after = f"{after_mb} MB" if after_mb else "unknown"
            print(
                f"{prefix}Resource prep: gc={collected}, purge={purge_status}, "
                f"available {before} -> {after}"
            )

    def _create_engine(self, *, preload_transformer: bool) -> UCIEngine:
        options, profile = self._live_engine_options()
        self._last_resource_profile = profile
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
        profile = self._last_resource_profile or RESOURCE_PROFILE
        print(
            f"  Engine ready ({elapsed:.1f}s warmup, "
            f"{int(profile['threads'])}T, Hash {int(profile['hash_mb'])} MB)"
        )

    def _acquire_engine(self, game_id: str | None = None) -> UCIEngine:
        options, profile = self._live_engine_options()
        if self._warm_engine is not None:
            engine = self._warm_engine
            self._warm_engine = None
            try:
                engine.configure(options)
            except Exception:
                engine.quit()
                engine = UCIEngine(
                    ENGINE,
                    options,
                    preload_transformer=True,
                )
            self._last_resource_profile = profile
            self._print_resource_profile(game_id, profile)
            return engine
        self._last_resource_profile = profile
        self._print_resource_profile(game_id, profile)
        return UCIEngine(ENGINE, options, preload_transformer=False)

    def _close_warm_engine(self):
        if self._warm_engine is None:
            return
        self._warm_engine.quit()
        self._warm_engine = None

    def _validate_self_test_move(
        self, label: str, move: str, board: chess.Board
    ) -> None:
        if move in ("0000", "(none)", ""):
            raise RuntimeError(f"{label}: engine returned no move")
        parsed = self._normalize_uci_move(move, board)
        if parsed is None:
            raise RuntimeError(f"{label}: illegal move {move} for {board.fen()}")

    def _run_engine_self_test(self) -> bool:
        print("  Engine self-test: UCI, search, ponder, and recovery probes...")
        engine = self._warm_engine
        created_engine = False
        if engine is None:
            try:
                engine = self._create_engine(preload_transformer=True)
                created_engine = True
            except Exception as e:
                print(f"  Engine self-test failed during startup: {e}")
                return False

        try:
            engine.new_game()

            board = chess.Board()
            engine.set_position("startpos", [])
            best, ponder = engine.go(movetime=150, timeout=10)
            self._validate_self_test_move("quick search", best, board)
            print(f"  Engine self-test: quick search {best}")

            engine.start_pondering(
                "startpos",
                [],
                "e2e4",
                wtime=60_000,
                btime=60_000,
                winc=1_000,
                binc=1_000,
            )
            time.sleep(0.2)
            if not engine.stop_pondering(timeout=PONDER_STOP_TIMEOUT_S):
                raise RuntimeError("ponder stop did not return bestmove")
            print("  Engine self-test: ponder stop OK")

            ponder_board = chess.Board()
            ponder_board.push(chess.Move.from_uci("e2e4"))
            engine.start_pondering(
                "startpos",
                [],
                "e2e4",
                wtime=60_000,
                btime=60_000,
                winc=1_000,
                binc=1_000,
            )
            time.sleep(0.2)
            best, ponder = engine.ponderhit(timeout=10)
            self._validate_self_test_move("ponderhit", best, ponder_board)
            print(f"  Engine self-test: ponderhit {best}")

            if SYZYGY_PATH:
                tb_board = chess.Board("8/8/8/8/8/8/P1k5/K7 w - - 0 1")
                engine.set_position(tb_board.fen(), [])
                best, _ = engine.go(movetime=100, timeout=10)
                self._validate_self_test_move("syzygy probe", best, tb_board)
                if self._normalize_uci_move(best, tb_board).uci() != "a2a4":
                    raise RuntimeError(
                        f"syzygy probe: expected tablebase win a2a4, got {best}"
                    )
                print("  Engine self-test: Syzygy root ranking OK")
            else:
                print("  Engine self-test: Syzygy probe skipped (disabled)")

            engine.new_game()
            print("  Engine self-test: OK")
            return True
        except Exception as e:
            diagnostic = engine._diagnostic_tail() if engine is not None else ""
            print(f"  Engine self-test failed: {e}{diagnostic}")
            if engine is self._warm_engine:
                self._close_warm_engine()
            return False
        finally:
            if created_engine and engine is not None:
                engine.quit()

    def _reserved_games(self) -> int:
        pending = 1 if self._pending_challenge_id else 0
        return len(self.active_games) + pending

    def _completed_limit_reached(self) -> bool:
        limit = getattr(self.args, "quit_after_games", 0) or 0
        return limit > 0 and self._completed_games >= limit

    def _mark_game_completed(self, game_id: str | None) -> None:
        if not game_id:
            return
        if not hasattr(self, "_completed_game_ids"):
            self._completed_game_ids = set()
            self._completed_game_order = []
        if game_id in self._completed_game_ids:
            return
        self._completed_game_ids.add(game_id)
        self._completed_game_order.append(game_id)
        overflow = len(self._completed_game_order) - COMPLETED_GAME_HISTORY_LIMIT
        if overflow > 0:
            for old_game_id in self._completed_game_order[:overflow]:
                self._completed_game_ids.discard(old_game_id)
            del self._completed_game_order[:overflow]

    def _game_was_completed(self, game_id: str | None) -> bool:
        if not game_id or not hasattr(self, "_completed_game_ids"):
            return False
        return game_id in self._completed_game_ids

    def _event_game_id(self, event: dict) -> str:
        game = event.get("game", {})
        if isinstance(game, dict):
            game_id = game.get("gameId") or game.get("id")
            if game_id:
                return str(game_id)
        game_id = event.get("gameId") or event.get("id")
        return str(game_id) if game_id else ""

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

    def _challenge_event_identity(self, event: dict) -> tuple[str | None, str | None]:
        ch = event.get("challenge") if isinstance(event.get("challenge"), dict) else {}
        challenge_id = ch.get("id") or event.get("challengeId") or event.get("id")
        target = None
        for key in ("destUser", "challenger"):
            user = ch.get(key)
            if isinstance(user, dict):
                user_id = user.get("id")
            elif isinstance(user, str):
                user_id = user
            else:
                user_id = None
            if user_id and self._cooldown_key(user_id) != self._cooldown_key(
                self.bot_id
            ):
                target = user_id
                break
        return (str(challenge_id) if challenge_id else None, target)

    def _challenge_event_reason(self, event: dict) -> str:
        ch = event.get("challenge") if isinstance(event.get("challenge"), dict) else {}
        for source in (event, ch):
            if not isinstance(source, dict):
                continue
            for key in ("reason", "declineReason", "status"):
                value = source.get(key)
                if value:
                    return str(value)
        return ""

    def _clear_pending_challenge(self):
        self._pending_challenge_id = None
        self._pending_challenge_target = None
        self._pending_challenge_speed = None
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

    def seek_game(self):
        if not self._seek_lock.acquire(blocking=False):
            return
        try:
            self._seek_game_once()
        finally:
            self._seek_lock.release()

    def _seek_is_rated(self) -> bool:
        return bool(self.args.accept_rated and not self.args.accept_casual)

    def _rated_policy_label(self, *, outgoing: bool = False) -> str:
        if not self.args.accept_rated:
            return "disabled"
        floor = getattr(self.args, "min_rated_opponent_elo", 0) or 0
        parts = [f">= {floor}" if floor > 0 else "no floor"]
        if outgoing and getattr(self.args, "seek_highest_rated", False):
            parts.append("highest-rated first")
        elif getattr(self.args, "elo_seek", False):
            parts.append(f"within +/-{self._current_elo_range()} Elo")
        return ", ".join(parts)

    def _seek_policy_label(self) -> str:
        if not self.args.seek:
            return "disabled"
        if self._seek_is_rated():
            return f"rated ({self._rated_policy_label(outgoing=True)})"
        return "casual"

    def _cooldown_key(self, bot_id: str | None) -> str | None:
        if not bot_id:
            return None
        key = str(bot_id).strip().lower()
        return key or None

    def _load_challenge_cooldowns(self) -> dict[str, float]:
        try:
            data = json.loads(CHALLENGE_COOLDOWN_PATH.read_text())
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        now = time.time()
        cooldowns: dict[str, float] = {}
        for key, value in data.items():
            norm_key = self._cooldown_key(str(key))
            if not norm_key:
                continue
            try:
                expires = float(value)
            except (TypeError, ValueError):
                continue
            if expires > now:
                cooldowns[norm_key] = expires
        return cooldowns

    def _save_challenge_cooldowns(self) -> None:
        if not getattr(self, "_persist_challenge_cooldowns", False):
            return
        cooldowns = getattr(self, "_declined_cooldown", {})
        now = time.time()
        data = {
            key: expires
            for key, expires in sorted(cooldowns.items())
            if key and expires > now
        }
        try:
            CHALLENGE_COOLDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
            CHALLENGE_COOLDOWN_PATH.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_played_format_history(self) -> dict[str, dict[str, float]]:
        try:
            data = json.loads(PLAYED_FORMAT_HISTORY_PATH.read_text())
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}

        history: dict[str, dict[str, float]] = {}
        for speed, entries in data.items():
            if speed not in ACCEPTED_SPEEDS or not isinstance(entries, dict):
                continue
            clean_entries: dict[str, float] = {}
            for bot_id, played_at in entries.items():
                key = self._cooldown_key(str(bot_id))
                if not key:
                    continue
                try:
                    clean_entries[key] = float(played_at)
                except (TypeError, ValueError):
                    clean_entries[key] = 0.0
            if clean_entries:
                history[speed] = clean_entries
        return history

    def _save_played_format_history(self) -> None:
        if not getattr(self, "_persist_played_format_history", False):
            return
        history = getattr(self, "_played_by_speed", {})
        if not isinstance(history, dict):
            return

        data: dict[str, dict[str, float]] = {}
        remaining = PLAYED_FORMAT_HISTORY_LIMIT
        for speed in sorted(history):
            entries = history.get(speed, {})
            if speed not in ACCEPTED_SPEEDS or not isinstance(entries, dict):
                continue
            ordered = sorted(entries.items(), key=lambda item: item[1], reverse=True)
            limited = ordered[: max(0, remaining)]
            if limited:
                data[speed] = {key: played_at for key, played_at in limited}
                remaining -= len(limited)
            if remaining <= 0:
                break

        try:
            PLAYED_FORMAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            PLAYED_FORMAT_HISTORY_PATH.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _bot_on_cooldown(self, bot_id: str | None, now: float | None = None) -> bool:
        key = self._cooldown_key(bot_id)
        if not key:
            return False
        cooldowns = getattr(self, "_declined_cooldown", {})
        return (now or time.time()) < cooldowns.get(key, 0)

    def _bot_played_in_speed(self, bot_id: str | None, speed: str) -> bool:
        if not getattr(self.args, "avoid_repeat_format", False):
            return False
        key = self._cooldown_key(bot_id)
        if not key:
            return False
        history = getattr(self, "_played_by_speed", {})
        entries = history.get(speed, {}) if isinstance(history, dict) else {}
        return key in entries

    def _mark_seek_opponent_played(self, bot_id: str | None, speed: str | None):
        if not getattr(self.args, "avoid_repeat_format", False):
            return
        key = self._cooldown_key(bot_id)
        if not key or speed not in ACCEPTED_SPEEDS:
            return
        if not hasattr(self, "_played_by_speed") or not isinstance(
            self._played_by_speed, dict
        ):
            self._played_by_speed = {}
        self._played_by_speed.setdefault(str(speed), {})[key] = time.time()
        self._save_played_format_history()

    def _online_bots_from_ndjson(self, text: str) -> list[dict]:
        bots: list[dict] = []
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                bot = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(bot, dict):
                continue
            bot_id = str(bot.get("id") or bot.get("name") or "")
            if not bot_id or self._cooldown_key(bot_id) == self._cooldown_key(
                self.bot_id
            ):
                continue
            bots.append(bot)
        return bots

    def _online_bots(
        self, *, force_refresh: bool = False
    ) -> tuple[list[dict] | None, int | None]:
        now = time.time()
        cached = getattr(self, "_online_bots_cache", None)
        if not force_refresh and cached is not None and cached[0] > now:
            return cached[1], None

        r = self.api_get("/bot/online", params={"nb": BOT_ONLINE_FETCH_LIMIT})
        if r.status_code != 200:
            return None, r.status_code

        bots = self._online_bots_from_ndjson(r.text)
        self._online_bots_cache = (now + BOT_ONLINE_CACHE_TTL_S, bots)
        return bots, None

    def _seek_candidates(
        self, bots: list[dict], speed: str, rated: bool
    ) -> tuple[list[str], str | None]:
        if not bots:
            return [], "No eligible bots online"

        now = time.time()
        bots = [
            b
            for b in bots
            if not self._bot_on_cooldown(str(b.get("id") or b.get("name") or ""), now)
        ]
        if not bots:
            return [], "All eligible bots are cooling down after declines/timeouts"

        if getattr(self.args, "avoid_repeat_format", False):
            bots = [
                b
                for b in bots
                if not self._bot_played_in_speed(
                    str(b.get("id") or b.get("name") or ""), speed
                )
            ]
            if not bots:
                return [], f"All eligible bots were already played in {speed}"

        if rated:
            bots = [b for b in bots if self._rated_opponent_allowed(b, speed)]
            if not bots:
                return [], "No rated opponents above the rating floor"

        candidates = self._filter_bots_by_elo(bots, speed)
        if not candidates:
            if self.args.elo_seek:
                return [], "No opponents inside Elo seek range"
            candidates = [b.get("id", "") for b in bots if b.get("id")]
            random.shuffle(candidates)

        if not candidates:
            return [], "No eligible bots online"
        return candidates, None

    def preview_seek_once(self, show_config: bool = True) -> bool:
        if show_config:
            print_config_check(self.args)
            print("")

        if not self._resources_allow_new_game():
            print("Seek dry-run: resources busy; no challenge would be sent")
            return False

        limit, inc = self._next_tc()
        tc_label = f"{limit//60}+{inc}"
        speed = self._tc_to_speed(limit, inc)
        rated = self._seek_is_rated()

        bots, status_code = self._online_bots()
        if bots is None:
            print(f"Seek dry-run: /bot/online failed with {status_code}")
            return False

        candidates, reason = self._seek_candidates(bots, speed, rated)
        if reason:
            print(f"Seek dry-run: {reason}")
            return False

        target = candidates[0]
        mode = "rated" if rated else "casual"
        print(
            f"Seek dry-run: would challenge {target} "
            f"({tc_label}, {mode}, {speed}); candidates={len(candidates)}"
        )
        if rated:
            print(f"Seek dry-run: rated floor {self.args.min_rated_opponent_elo}")
        return True

    def _seek_game_once(self):
        if not self._should_seek():
            return

        if not self._resources_allow_new_game():
            print("  Resources busy, deferring seek for 30s...")
            self._schedule_retry(30)
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
        rated = self._seek_is_rated()

        try:
            bots, status_code = self._online_bots()
            if bots is None:
                print(f"  /bot/online failed with {status_code}, retrying...")
                self._schedule_retry()
                return

            candidates, reason = self._seek_candidates(bots, speed, rated)
            if reason:
                print(f"  {reason}, retrying in 30s...")
                self._schedule_retry(30)
                return

            target = candidates[0]
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
                self._pending_challenge_speed = speed
                self._challenge_sent_at = time.time()
                self._challenge_retries = 0
                self._rate_limit_count = 0
                print(
                    f"  Challenged {target} "
                    f"({tc_label}, {'rated' if rated else 'casual'}, "
                    f"candidates={len(candidates)})"
                )
                self._schedule_challenge_timeout()
            elif r.status_code == 429:
                print("  Rate limited, backing off...")
                self._rate_limit_count += 1
                retry_after = _retry_after_seconds(r)
                if self._rate_limit_count >= 3:
                    backoff = max(900, retry_after)
                    print(
                        "  Possible daily cap hit. "
                        f"Waiting {int(backoff) // 60}min before retrying."
                    )
                else:
                    backoff = max(65, retry_after)
                    print(f"  Waiting {int(backoff)}s...")
                self._schedule_retry(backoff)
            else:
                detail = (r.text or "").strip().replace("\n", " ")[:180]
                suffix = f": {detail}" if detail else ""
                print(f"  Challenge to {target} failed ({r.status_code}){suffix}")
                cooldown_s = self._challenge_failure_cooldown_seconds(r)
                if cooldown_s > 300:
                    print(
                        f"  Cooling down {target} for {self._format_duration(cooldown_s)}"
                    )
                self._cooldown_bot(target, duration=cooldown_s)
                self._schedule_retry(2)
        except Exception as e:
            print(f"  Seek error: {e}")
            self._schedule_retry(15)

    def _challenge_failure_cooldown_seconds(self, response: requests.Response) -> int:
        cooldown = 300
        try:
            data = response.json()
        except ValueError:
            return cooldown
        if not isinstance(data, dict):
            return cooldown
        ratelimit = data.get("ratelimit")
        if isinstance(ratelimit, dict):
            try:
                seconds = int(float(ratelimit.get("seconds", 0) or 0))
            except (TypeError, ValueError):
                seconds = 0
            if seconds > 0:
                cooldown = max(cooldown, min(86_400, seconds + 60))
        return cooldown

    def _format_duration(self, seconds: int) -> str:
        seconds = max(0, int(seconds))
        if seconds >= 3600:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h{minutes:02d}m"
        if seconds >= 60:
            return f"{seconds // 60}m{seconds % 60:02d}s"
        return f"{seconds}s"

    def _cooldown_bot(self, bot_id: str | None, duration: int = 600):
        key = self._cooldown_key(bot_id)
        if key:
            if not hasattr(self, "_declined_cooldown"):
                self._declined_cooldown = {}
            self._declined_cooldown[key] = max(
                self._declined_cooldown.get(key, 0), time.time() + duration
            )
            self._save_challenge_cooldowns()

    def _cleanup_cooldowns(self):
        now = time.time()
        self._declined_cooldown = {
            k: v for k, v in self._declined_cooldown.items() if v > now
        }
        self._save_challenge_cooldowns()

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
            self._clear_pending_challenge()
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
        perfs = bot.get("perfs", {})
        if not isinstance(perfs, dict):
            return False
        perf = perfs.get(speed, {})
        if not isinstance(perf, dict):
            return False
        try:
            games = int(perf.get("games", 0) or 0)
        except (TypeError, ValueError):
            games = 0
        return games >= 5 and not perf.get("prov", False)

    def _get_bot_rating(self, bot: dict, speed: str) -> int | None:
        perfs = bot.get("perfs", {})
        if not isinstance(perfs, dict):
            return None
        perf = perfs.get(speed, {})
        if isinstance(perf, dict):
            rating = self._rating_to_int(perf.get("rating"))
            if rating and not perf.get("prov", False):
                return rating
        for s in ("blitz", "rapid", "bullet", "classical"):
            p = perfs.get(s, {})
            rating = (
                self._rating_to_int(p.get("rating")) if isinstance(p, dict) else None
            )
            if rating and not p.get("prov", False):
                return rating
        return None

    def _rating_to_int(self, rating) -> int | None:
        try:
            value = int(rating)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _rated_opponent_allowed(self, bot: dict, speed: str) -> bool:
        floor = getattr(self.args, "min_rated_opponent_elo", 0) or 0
        if floor <= 0:
            return True
        rating = self._get_bot_rating(bot, speed)
        return rating is not None and rating >= floor

    def _filter_bots_by_elo(self, bots: list[dict], speed: str) -> list[str]:
        if getattr(self.args, "seek_highest_rated", False):
            scored: list[tuple[int, str]] = []
            for bot in bots:
                bot_id = bot.get("id", "")
                if not bot_id or not self._bot_plays_speed(bot, speed):
                    continue
                rating = self._get_bot_rating(bot, speed)
                if rating is not None:
                    scored.append((rating, str(bot_id)))
            scored.sort(key=lambda item: (-item[0], item[1].lower()))
            return [bot_id for _, bot_id in scored]

        if not getattr(self.args, "elo_seek", False):
            # Still filter by speed activity even without elo-seek
            active = [b for b in bots if self._bot_plays_speed(b, speed)]
            if not active:
                active = bots
            ids = [str(b.get("id", "")) for b in active if b.get("id")]
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
            return []

        scored.sort(key=lambda x: x[0])
        top = scored[:15]
        random.shuffle(top)
        return [bot_id for _, bot_id in top]

    def _our_rating(self, speed: str) -> int:
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
        return self._cached_ratings.get(
            speed, getattr(self.args, "elo_target", None) or 1500
        )

    def _current_elo_range(self) -> int:
        base = self.args.elo_range if self.args.elo_range else 200
        return base + self._elo_widen_steps * 100

    def _widen_elo_range(self):
        self._elo_widen_steps += 1
        new_range = self._current_elo_range()
        print(f"  Widening Elo range to ±{new_range}")

    def _reset_elo_range(self):
        self._elo_widen_steps = 0

    def _resources_allow_new_game(self) -> bool:
        available_mb = available_memory_mb()
        if available_mb and available_mb < RESOURCE_RESERVE_MB:
            return False
        return system_load_ratio() < DEFER_SEEK_LOAD

    def _should_seek(self) -> bool:
        return (
            self.args.seek
            and not self._draining.is_set()
            and not self._completed_limit_reached()
            and self._reserved_games() < self.args.max_games
        )

    def _user_id(self, user) -> str:
        if isinstance(user, dict):
            return str(user.get("id") or user.get("name") or "")
        if isinstance(user, str):
            return user
        return ""

    def _challenge_reject_reason(self, challenge: dict) -> str | None:
        if not isinstance(challenge, dict):
            return "malformed challenge"
        if self._draining.is_set():
            return "draining"
        ch = challenge.get("challenge", challenge)
        if not isinstance(ch, dict):
            return "malformed challenge"
        challenger_id = self._user_id(ch.get("challenger"))
        if challenger_id == self.bot_id:
            return "own challenge"
        variant_info = ch.get("variant", {})
        variant = (
            variant_info.get("key", "standard")
            if isinstance(variant_info, dict)
            else str(variant_info or "standard")
        )
        if variant != "standard":
            return f"unsupported variant {variant}"
        rated = ch.get("rated", False)
        if rated and not self.args.accept_rated:
            return "rated disabled"
        if not rated and not self.args.accept_casual:
            return "casual disabled"
        speed = ch.get("speed", "")
        if speed not in ACCEPTED_SPEEDS:
            return f"unsupported speed {speed or '?'}"
        if not self._resources_allow_new_game():
            return "resources busy"
        if rated:
            if not self._rated_challenge_allowed(ch, speed):
                return "rated rating floor"
            if not self._rated_challenge_elo_allowed(ch, speed):
                return "rated elo range"
        tc = ch.get("timeControl", {})
        if not isinstance(tc, dict):
            return "unsupported time control"
        if tc.get("type") == "clock":
            try:
                increment = int(tc.get("increment", 0) or 0)
            except (TypeError, ValueError):
                increment = 0
            if speed == "bullet" and not self.args.include_bullet:
                return "bullet disabled"
            if increment == 0 and not self.args.include_zero_increment:
                return "zero increment disabled"
        if self._reserved_games() >= self.args.max_games:
            return "max games reached"
        return None

    def should_accept(self, challenge: dict) -> bool:
        return self._challenge_reject_reason(challenge) is None

    def _challenge_rating(self, challenge: dict, speed: str) -> int | None:
        challenger = challenge.get("challenger", {})
        for candidate, provisional in (
            (
                challenge.get("rating"),
                challenge.get("provisional", challenge.get("prov", False)),
            ),
            (
                challenger.get("rating") if isinstance(challenger, dict) else None,
                (
                    challenger.get("provisional", challenger.get("prov", False))
                    if isinstance(challenger, dict)
                    else False
                ),
            ),
        ):
            rating = self._rating_to_int(candidate)
            if rating and not provisional:
                return rating

        perfs = challenger.get("perfs", {}) if isinstance(challenger, dict) else {}
        if not isinstance(perfs, dict):
            return None
        for key in (speed, "rapid", "blitz", "bullet", "classical"):
            perf = perfs.get(key, {})
            if not isinstance(perf, dict):
                continue
            rating = self._rating_to_int(perf.get("rating"))
            if rating and not perf.get("prov", False):
                return rating
        return None

    def _rated_challenge_allowed(self, challenge: dict, speed: str) -> bool:
        floor = getattr(self.args, "min_rated_opponent_elo", 0) or 0
        if floor <= 0:
            return True
        rating = self._challenge_rating(challenge, speed)
        return rating is not None and rating >= floor

    def _rated_challenge_elo_allowed(self, challenge: dict, speed: str) -> bool:
        if not getattr(self.args, "elo_seek", False):
            return True
        rating = self._challenge_rating(challenge, speed)
        if rating is None:
            return False
        return abs(rating - self._our_rating(speed)) <= self._current_elo_range()

    def _stream_failure_is_transient(self) -> bool:
        status = getattr(self, "_last_game_stream_status", None)
        if status in GAME_STREAM_TRANSIENT_STATUSES:
            return True
        error = getattr(self, "_last_game_stream_error", "")
        return "timed out" in str(error).lower()

    def play_game(self, game_id: str):
        print(f"  [{game_id}] Starting...")
        self._audit(game_id, "game_start")
        engine = None
        finished = False
        inferred_finished = False
        try:
            self._prepare_game_resources(game_id)
            engine = self._acquire_engine(game_id)
            engine.new_game()
            active_retries = 0
            total_attempts = GAME_STREAM_RETRIES + GAME_STREAM_ACTIVE_RETRIES + 1
            for attempt in range(total_attempts):
                try:
                    finished = self._game_loop(game_id, engine)
                except Exception as e:
                    self._last_game_stream_status = None
                    self._last_game_stream_error = str(e)
                    print(f"  [{game_id}] Stream error: {e}")
                    self._audit(game_id, "stream_error", error=str(e))
                if finished:
                    break

                active_status = self._game_active_status(game_id)
                self._audit(
                    game_id,
                    "stream_active_status",
                    active=active_status,
                    attempt=attempt,
                )
                if active_status is False:
                    print(f"  [{game_id}] No longer listed as active on Lichess")
                    finished = True
                    inferred_finished = True
                    break
                if active_status is None and self._stream_failure_is_transient():
                    if active_retries >= GAME_STREAM_ACTIVE_RETRIES:
                        break
                    active_retries += 1
                    print(
                        f"  [{game_id}] Stream transient; active status unknown, "
                        "reconnecting"
                    )
                    self._audit(
                        game_id,
                        "stream_reconnect",
                        reason="transient_unknown_active",
                        active_retries=active_retries,
                        status=getattr(self, "_last_game_stream_status", None),
                        error=getattr(self, "_last_game_stream_error", ""),
                    )
                    time.sleep(GAME_STREAM_RETRY_DELAY_S)
                    continue
                if active_status is True:
                    if active_retries >= GAME_STREAM_ACTIVE_RETRIES:
                        break
                    active_retries += 1
                    print(
                        f"  [{game_id}] Stream dropped; game still active, reconnecting"
                    )
                    self._audit(
                        game_id,
                        "stream_reconnect",
                        reason="still_active",
                        active_retries=active_retries,
                    )
                    time.sleep(GAME_STREAM_RETRY_DELAY_S)
                    continue

                if attempt >= GAME_STREAM_RETRIES:
                    break
                print(f"  [{game_id}] Stream ended early; reconnecting")
                self._audit(
                    game_id,
                    "stream_reconnect",
                    reason="early_end",
                    attempt=attempt + 1,
                )
                time.sleep(GAME_STREAM_RETRY_DELAY_S)
        except Exception as e:
            print(f"  [{game_id}] Error: {e}")
            self._audit(game_id, "game_error", error=str(e))
            traceback.print_exc()
        finally:
            if engine:
                engine.quit()
            with self._submitted_turns_lock:
                self._submitted_turns.pop(game_id, None)
                self._max_seen_ply.pop(game_id, None)
            getattr(self, "_playing_status_cache", {}).pop(game_id, None)
            with self._ponder_disabled_lock:
                self._ponder_disabled_games.discard(game_id)
            self.active_games.pop(game_id, None)
            if finished:
                self._mark_game_completed(game_id)
                self._completed_games += 1
                suffix = " (status inferred)" if inferred_finished else ""
                print(f"  [{game_id}] Finished{suffix}.")
                self._audit(
                    game_id,
                    "game_finished",
                    inferred=inferred_finished,
                    completed_games=self._completed_games,
                )
            else:
                print(
                    f"  [{game_id}] Game stream ended before completion; "
                    "shutting down to avoid stale state"
                )
                self._audit(game_id, "game_unfinished_shutdown")
                self._draining.set()
                self._shutdown.set()
            if finished and self._completed_limit_reached():
                print(f"  Completed {self._completed_games} game(s); shutting down.")
                self._draining.set()
                self._shutdown.set()
            elif self._draining.is_set():
                if not self.active_games:
                    self._shutdown.set()
            elif finished and self._should_seek():
                self._prepare_warm_engine()
                self._schedule_retry(3)

    def _game_loop(self, game_id: str, engine: UCIEngine) -> bool:
        self._last_game_stream_status = None
        self._last_game_stream_error = ""
        with self.api_get(
            f"/bot/game/stream/{game_id}",
            stream=True,
            timeout=(10, GAME_STREAM_TIMEOUT_S),
        ) as r:
            self._last_game_stream_status = r.status_code
            if r.status_code != 200:
                print(f"  [{game_id}] Stream failed: {r.status_code}")
                self._audit(game_id, "stream_failed", status_code=r.status_code)
                return False
            game_info = {}
            my_color = "white"
            initial_fen = "startpos"

            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self._audit(game_id, "stream_frame_ignored", reason="bad_json")
                    continue
                if not isinstance(event, dict):
                    self._audit(game_id, "stream_frame_ignored", reason="not_object")
                    continue

                etype = event.get("type", "")

                if etype == "gameFull":
                    game_info = event
                    my_color = self._get_my_color(event)
                    initial_fen = event.get("initialFen", "startpos")
                    state = event.get("state", {})
                    if not isinstance(state, dict):
                        state = {}
                    status = state.get("status", "started")
                    if status != "started":
                        print(f"  [{game_id}] Game already ended: {status}")
                        self._audit(game_id, "stream_game_full", status=status)
                        return True
                    moves = self._stream_moves(state)
                    self._audit(
                        game_id,
                        "stream_game_full",
                        status=status,
                        color=my_color,
                        **self._audit_context(moves),
                    )
                    self._try_move(game_id, engine, initial_fen, moves, my_color, state)

                elif etype == "gameState":
                    status = event.get("status", "started")
                    if status != "started":
                        moves = self._stream_moves(event)
                        winner = event.get("winner", "")
                        result = f"{status}"
                        if winner:
                            result += f" ({winner} wins)"
                        print(f"  [{game_id}] Game over: {result}, {len(moves)} moves")
                        self._audit(
                            game_id,
                            "stream_game_over",
                            status=status,
                            winner=winner,
                            **self._audit_context(moves),
                        )
                        return True
                    moves = self._stream_moves(event)
                    self._audit(
                        game_id,
                        "stream_game_state",
                        status=status,
                        **self._audit_context(moves),
                    )
                    self._try_move(game_id, engine, initial_fen, moves, my_color, event)

                elif etype == "chatLine":
                    pass
        return False

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

        board = self._build_board(game_id, initial_fen, moves)
        if board is None:
            return

        if not self._remember_stream_ply(game_id, moves):
            print(f"  [{game_id}] Ignoring stale stream state at ply {len(moves)}")
            self._audit(game_id, "turn_stale", **self._audit_context(moves))
            return

        wtime, btime, winc, binc = self._clock_values(state)
        self._audit(
            game_id,
            "turn_seen",
            color=my_color,
            is_my_turn=is_my_turn,
            fen=board.fen(),
            wtime=wtime,
            btime=btime,
            winc=winc,
            binc=binc,
            **self._audit_context(moves),
        )

        if not is_my_turn:
            return

        if self._already_submitted_for_turn(game_id, moves):
            self._audit(game_id, "turn_duplicate", **self._audit_context(moves))
            return

        search_timeout = self._search_timeout_seconds(
            my_color, wtime, btime, winc, binc
        )
        ponder_stop_timeout = self._ponder_stop_timeout_seconds(
            my_color, wtime, btime, winc, binc
        )

        if engine.ponder_move:
            if self._last_move_matches_ponder(
                game_id, initial_fen, moves, engine.ponder_move
            ):
                self._audit(
                    game_id,
                    "ponderhit_start",
                    ponder_move=engine.ponder_move,
                    timeout=round(search_timeout, 3),
                    **self._audit_context(moves),
                )
                try:
                    best, ponder = engine.ponderhit(timeout=search_timeout)
                    self._audit(
                        game_id,
                        "ponderhit_result",
                        best=best,
                        ponder=ponder,
                        **self._audit_context(moves),
                    )
                except (TimeoutError, RuntimeError) as e:
                    print(f"  [{game_id}] Ponderhit failed: {e}; restarting")
                    self._audit(
                        game_id,
                        "ponderhit_failed",
                        error=str(e),
                        **self._audit_context(moves),
                    )
                    self._disable_ponder_for_game(game_id, "ponderhit failed")
                    engine.restart()
                    best, ponder = "0000", None
                if best and best not in ("0000", "(none)"):
                    parsed = self._parse_legal_move(game_id, best, board, "ponder")
                    if parsed is None:
                        print(f"  [{game_id}] Restarting after rejected ponder move")
                        self._audit(
                            game_id,
                            "ponderhit_illegal",
                            best=best,
                            fen=board.fen(),
                            **self._audit_context(moves),
                        )
                        engine.restart()
                    elif self._submit_move(game_id, moves, parsed.uci()):
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
                expected_ponder = engine.ponder_move
                if not engine.stop_pondering(timeout=ponder_stop_timeout):
                    self._disable_ponder_for_game(game_id, "ponder stop timed out")
                    print(f"  [{game_id}] Ponder stop timed out; engine restarted")
                    self._audit(
                        game_id,
                        "ponder_stop_failed",
                        expected=expected_ponder,
                        timeout=round(ponder_stop_timeout, 3),
                        **self._audit_context(moves),
                    )

        if self._should_query_book(board, my_color, wtime, btime):
            book_move = self.book.lookup(board.fen())
            if book_move:
                self._audit(
                    game_id,
                    "book_candidate",
                    move=book_move,
                    fen=board.fen(),
                    **self._audit_context(moves),
                )
                parsed = self._parse_legal_move(game_id, book_move, board, "book")
                if parsed is not None:
                    if self._submit_move(game_id, moves, parsed.uci()):
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

        if not engine.alive():
            print(f"  [{game_id}] Engine died, restarting")
            self._audit(game_id, "engine_dead", **self._audit_context(moves))
            try:
                engine.restart()
            except Exception as e:
                print(f"  [{game_id}] Engine restart failed: {e}")
                self._audit(
                    game_id,
                    "engine_restart_failed",
                    error=str(e),
                    **self._audit_context(moves),
                )
                fallback = self._fallback_move(board)
                if fallback and self._submit_move(game_id, moves, fallback):
                    print(f"  [{game_id}] Fallback legal move: {fallback}")
                return

        if not engine.stop_pondering(timeout=ponder_stop_timeout):
            self._disable_ponder_for_game(game_id, "ponder stop timed out")
            print(f"  [{game_id}] Ponder stop timed out; engine restarted")
            self._audit(
                game_id,
                "ponder_stop_failed",
                timeout=round(ponder_stop_timeout, 3),
                **self._audit_context(moves),
            )
        engine.set_position(initial_fen, moves)

        try:
            search_start = time.time()
            self._audit(
                game_id,
                "engine_search_start",
                timeout=round(search_timeout, 3),
                fen=board.fen(),
                **self._audit_context(moves),
            )
            best, ponder = engine.go(
                wtime=wtime,
                btime=btime,
                winc=winc,
                binc=binc,
                timeout=search_timeout,
            )
            search_ms = int((time.time() - search_start) * 1000)
            self._audit(
                game_id,
                "engine_search_result",
                best=best,
                ponder=ponder,
                elapsed_ms=search_ms,
                **self._audit_context(moves),
            )
            if best == "0000" or best == "(none)":
                fallback = self._fallback_move(board)
                if not fallback:
                    self._audit(game_id, "engine_no_move", **self._audit_context(moves))
                    return
                self._audit(
                    game_id,
                    "fallback_selected",
                    reason="empty_bestmove",
                    move=fallback,
                    **self._audit_context(moves),
                )
                best, ponder = fallback, None
            if not self._is_legal_move(best, board):
                self._audit(
                    game_id,
                    "engine_illegal_move",
                    move=best,
                    fen=board.fen(),
                    **self._audit_context(moves),
                )
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
                    self._audit(
                        game_id, "engine_recovery_failed", **self._audit_context(moves)
                    )
                    return
            parsed = self._parse_legal_move(game_id, best, board, "engine")
            if parsed is not None and self._submit_move(game_id, moves, parsed.uci()):
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
            self._audit(
                game_id,
                "engine_search_error",
                error=str(e),
                **self._audit_context(moves),
            )
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
            if parsed is not None and self._submit_move(game_id, moves, parsed.uci()):
                print(f"  [{game_id}] Recovery: {parsed.uci()}")

    def _build_board(
        self, game_id: str, initial_fen: str, moves: list[str]
    ) -> chess.Board | None:
        try:
            board = (
                chess.Board(initial_fen) if initial_fen != "startpos" else chess.Board()
            )
        except ValueError as e:
            print(f"  [{game_id}] Bad initial FEN from stream: {e}")
            self._audit(game_id, "bad_initial_fen", error=str(e), fen=initial_fen)
            return None

        for ply, move in enumerate(moves, start=1):
            parsed = self._normalize_uci_move(move, board)
            if parsed is None:
                print(f"  [{game_id}] Bad stream move {move} at ply {ply}")
                self._audit(game_id, "bad_stream_move", move=move, ply=ply)
                return None
            board.push(parsed)
        return board

    def _normalize_uci_move(self, move: str, board: chess.Board) -> chess.Move | None:
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
        if not target or target.piece_type != chess.ROOK or target.color != board.turn:
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
                self._audit(
                    game_id,
                    "malformed_move",
                    source=source,
                    move=move,
                    fen=board.fen(),
                )
                return None
            print(
                f"  [{game_id}] Ignoring illegal {source} move {move} "
                f"for FEN: {board.fen()}"
            )
            self._audit(
                game_id,
                "illegal_move",
                source=source,
                move=move,
                fen=board.fen(),
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
        if not self._ponder_allowed_for_game(game_id):
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
            self._audit(
                game_id,
                "ponder_start_rejected",
                best=parsed_best.uci(),
                ponder=ponder,
                reason="illegal",
                **self._audit_context(moves),
            )
            return
        print(f"  [{game_id}] Ponder: {parsed_ponder.uci()}")
        self._audit(
            game_id,
            "ponder_start",
            best=parsed_best.uci(),
            ponder=parsed_ponder.uci(),
            source="engine",
            **self._audit_context(moves),
        )
        try:
            engine.start_pondering(
                initial_fen,
                moves + [parsed_best.uci()],
                parsed_ponder.uci(),
                wtime=wtime,
                btime=btime,
                winc=winc,
                binc=binc,
            )
        except (TimeoutError, RuntimeError) as e:
            self._disable_ponder_for_game(game_id, f"ponder start failed: {e}")
            self._audit(
                game_id,
                "ponder_start_failed",
                best=parsed_best.uci(),
                ponder=parsed_ponder.uci(),
                error=str(e),
                **self._audit_context(moves),
            )
            try:
                engine.restart()
            except Exception as restart_error:
                print(
                    f"  [{game_id}] Engine restart after ponder failure failed: "
                    f"{restart_error}"
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
        if not self._ponder_allowed_for_game(game_id):
            return
        reply = self.book.lookup(board_after.fen())
        parsed_reply = self._normalize_uci_move(reply, board_after) if reply else None
        if parsed_reply is None:
            return
        print(f"  [{game_id}] Book ponder: {parsed_reply.uci()}")
        self._audit(
            game_id,
            "ponder_start",
            best=parsed_best.uci(),
            ponder=parsed_reply.uci(),
            source="book",
            **self._audit_context(moves),
        )
        try:
            engine.start_pondering(
                initial_fen,
                moves + [parsed_best.uci()],
                parsed_reply.uci(),
                wtime=wtime,
                btime=btime,
                winc=winc,
                binc=binc,
            )
        except (TimeoutError, RuntimeError) as e:
            self._disable_ponder_for_game(game_id, f"book ponder start failed: {e}")
            self._audit(
                game_id,
                "ponder_start_failed",
                best=parsed_best.uci(),
                ponder=parsed_reply.uci(),
                source="book",
                error=str(e),
                **self._audit_context(moves),
            )
            try:
                engine.restart()
            except Exception as restart_error:
                print(
                    f"  [{game_id}] Engine restart after book ponder failure "
                    f"failed: {restart_error}"
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
        self._audit(
            game_id,
            "recovery_start",
            illegal_move=illegal_move,
            fen=board.fen(),
            **self._audit_context(moves),
        )
        try:
            engine.restart()
            if not engine.sync_position(initial_fen, moves, timeout=30):
                print(f"  [{game_id}] Engine retry refused: position sync mismatch")
                self._audit(
                    game_id,
                    "recovery_sync_mismatch",
                    **self._audit_context(moves),
                )
                return None, None
            self._audit(game_id, "recovery_sync_ok", **self._audit_context(moves))
            our_time = wtime if my_color == "white" else btime
            movetime = max(100, min(1000, our_time // 20))
            best, ponder = engine.go(
                movetime=movetime,
                timeout=movetime / 1000.0 + ENGINE_STOP_GRACE_S,
            )
            parsed = self._normalize_uci_move(best, board)
            if best not in ("0000", "(none)") and parsed is not None:
                print(f"  [{game_id}] Retry move: {parsed.uci()}")
                self._audit(
                    game_id,
                    "recovery_retry_ok",
                    move=parsed.uci(),
                    ponder=ponder,
                    movetime=movetime,
                    **self._audit_context(moves),
                )
                return parsed.uci(), ponder
            print(f"  [{game_id}] Retry also returned illegal move: {best}")
            self._audit(
                game_id,
                "recovery_retry_illegal",
                move=best,
                movetime=movetime,
                **self._audit_context(moves),
            )
        except (TimeoutError, RuntimeError) as e:
            print(f"  [{game_id}] Engine retry failed: {e}")
            self._audit(
                game_id,
                "recovery_retry_failed",
                error=str(e),
                **self._audit_context(moves),
            )

        fallback = self._fallback_move(board)
        if fallback:
            print(f"  [{game_id}] Fallback legal move: {fallback}")
            self._audit(
                game_id,
                "recovery_fallback",
                move=fallback,
                **self._audit_context(moves),
            )
            return fallback, None
        self._audit(game_id, "recovery_no_fallback", **self._audit_context(moves))
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
            moving_piece = board.piece_at(move.from_square)
            attacker_value = (
                piece_values.get(moving_piece.piece_type, 0) if moving_piece else 0
            )
            total = 0
            if move.promotion:
                total += (
                    piece_values.get(move.promotion, 0) - piece_values[chess.PAWN]
                ) * 8
            if board.is_capture(move):
                victim_value = (
                    piece_values[chess.PAWN] if board.is_en_passant(move) else 0
                )
                victim = board.piece_at(move.to_square)
                if victim:
                    victim_value = piece_values.get(victim.piece_type, 0)
                total += victim_value * 10 - attacker_value
            if board.gives_check(move):
                total += 80

            own_color = board.turn
            board.push(move)
            if board.is_checkmate():
                total += 100000
            elif board.is_stalemate():
                total -= 10000
            else:
                for reply in list(board.legal_moves):
                    board.push(reply)
                    allows_mate = board.is_checkmate()
                    board.pop()
                    if allows_mate:
                        total -= 50000
                        break

                moved_piece = board.piece_at(move.to_square)
                if moved_piece and moved_piece.color == own_color:
                    moved_value = piece_values.get(moved_piece.piece_type, 0)
                    opponent = not own_color
                    attackers = [
                        piece_values.get(piece.piece_type, 0)
                        for square in board.attackers(opponent, move.to_square)
                        if (piece := board.piece_at(square)) is not None
                    ]
                    defenders = [
                        piece_values.get(piece.piece_type, 0)
                        for square in board.attackers(own_color, move.to_square)
                        if (piece := board.piece_at(square)) is not None
                    ]
                    if attackers:
                        cheapest_attacker = min(attackers)
                        if cheapest_attacker < moved_value:
                            total -= moved_value - cheapest_attacker
                        if not defenders:
                            total -= moved_value // 2
            board.pop()
            return total

        return max(legal_moves, key=lambda move: (score(move), move.uci())).uci()

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
        header_options, header_profile = build_engine_options()
        if HYBRID_MCTS_THREADS == 0 and HYBRID_AB_THREADS == 0:
            cap = (
                "auto"
                if HYBRID_AUTO_AB_THREADS_CAP == 0
                else f"AB cap {HYBRID_AUTO_AB_THREADS_CAP}"
            )
            hybrid_split = f"{cap} split + GPU"
        else:
            hybrid_split = (
                f"AB {HYBRID_AB_THREADS}T + MCTS {HYBRID_MCTS_THREADS}T + GPU"
            )
        available_mb = int(header_profile.get("available_mb", 0))
        total_mb = int(header_profile.get("total_mb", 0))
        memory = (
            f"{available_mb}/{total_mb} MB" if available_mb and total_mb else "unknown"
        )
        print("=" * 60)
        print(f"  MetalFish Lichess Bot")
        print(f"  Account:  {self.username}")
        print(
            f"  Engine:   Hybrid ({hybrid_split}"
            f"{' + Ponder' if self.args.ponder else ''})"
        )
        print(
            f"  Workers:  dynamic up to {MAX_SEARCH_WORKERS} search + 1 coordinator "
            f"| CPU: {LOGICAL_CORES} logical"
        )
        print(
            f"  Initial:  {int(header_profile['threads'])} search threads "
            f"| Hash {header_options['Hash']} MB | Free {memory}"
        )
        print(f"  Reserve:  {RESOURCE_RESERVE_MB} MB | Network: BT4-1024x15x32h")
        print(f"  Syzygy:   {SYZYGY_PATH if SYZYGY_PATH else 'disabled'}")
        print(
            f"  Audit:    {'enabled' if LICHESS_AUDIT_ENABLED else 'disabled'} "
            f"| {LICHESS_AUDIT_DIR}"
        )
        print(
            f"  Clock:    Move overhead {BASE_ENGINE_OPTIONS['Move Overhead']} ms "
            f"| hard cap {MAX_SEARCH_TIMEOUT_S:.0f}s"
        )
        print(
            f"  Book:     {self.book.description()} "
            f"(ply < {BOOK_MAX_PLY}, clock >= {BOOK_MIN_CLOCK_MS // 1000}s)"
        )
        print(
            f"  Accepts:  rated={self.args.accept_rated} "
            f"| casual={self.args.accept_casual}"
        )
        print(f"  Seek:     {self._seek_policy_label()} | TC: {tc_mode}")
        print(f"  Incoming rated: {self._rated_policy_label()}")
        print(
            f"  Repeat:   "
            f"{'avoid same bot/speed' if self.args.avoid_repeat_format else 'allowed'}"
        )
        print(f"  Max games: {self.args.max_games}")
        if self.args.quit_after_games:
            print(f"  Quit after: {self.args.quit_after_games} completed game(s)")
        print("=" * 60)

        self._prepare_warm_engine()
        if self.args.engine_self_test and not self._run_engine_self_test():
            self._close_warm_engine()
            sys.exit(1)
        self._start_stdin_watcher()
        print("\nListening... (Ctrl+C to stop, Ctrl+D to drain after current game)\n")

        try:
            self._event_loop()
        finally:
            self._close_warm_engine()
            self.book.close()

    def _event_loop(self):
        if self._should_seek():
            self.seek_game()

        while not self._shutdown.is_set():
            try:
                with self.api_get(
                    "/stream/event",
                    stream=True,
                    timeout=(10, EVENT_STREAM_READ_TIMEOUT_S),
                ) as r:
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
                if not self._shutdown.is_set():
                    time.sleep(EVENT_STREAM_RECONNECT_DELAY_S)
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

    def _handle_game_start(self, game_id: str) -> None:
        if not game_id:
            return
        if self._game_was_completed(game_id):
            print(f"  [{game_id}] Ignoring stale gameStart for completed game")
            return
        if self._seek_timer:
            self._seek_timer.cancel()
            self._seek_timer = None
        if game_id in self.active_games:
            return

        pending_id = self._pending_challenge_id
        pending_target = self._pending_challenge_target
        pending_speed = getattr(self, "_pending_challenge_speed", None)
        if pending_id:
            if self._same_game_id(pending_id, game_id):
                self._mark_seek_opponent_played(pending_target, pending_speed)
                self._cancel_pending_challenge("game started")
            else:
                self._cancel_pending_challenge("game started elsewhere")

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

    def _handle_event(self, event: dict):
        if not isinstance(event, dict):
            return
        etype = event.get("type", "")

        if etype == "challenge":
            ch = event.get("challenge", {})
            if not isinstance(ch, dict):
                return
            challenger_user = ch.get("challenger", {})
            challenger = (
                challenger_user.get("name") or challenger_user.get("id") or "?"
                if isinstance(challenger_user, dict)
                else str(challenger_user or "?")
            )
            challenger_id = self._user_id(challenger_user)
            tc = ch.get("timeControl", {})
            if isinstance(tc, dict) and tc.get("type") == "clock":
                try:
                    limit = int(tc.get("limit", 0) or 0)
                    increment = int(tc.get("increment", 0) or 0)
                except (TypeError, ValueError):
                    limit = 0
                    increment = 0
                tc_str = f"{limit//60}+{increment}"
            elif isinstance(tc, dict):
                tc_str = str(tc.get("type", "?"))
            else:
                tc_str = str(tc or "?")
            rated = "rated" if ch.get("rated") else "casual"

            if challenger_id == self.bot_id:
                return

            challenge_id = ch.get("id")
            reason = self._challenge_reject_reason(event)
            if reason is None:
                if not challenge_id:
                    print(f"  Could not accept challenge from {challenger}: missing id")
                    return
                print(f"  Accepting {rated} from {challenger} ({tc_str})")
                self.accept_challenge(challenge_id)
            else:
                print(
                    f"  Declining from {challenger} ({tc_str}, {rated}): " f"{reason}"
                )
                if challenge_id:
                    self.decline_challenge(challenge_id)

        elif etype == "gameStart":
            game_id = self._event_game_id(event)
            seek_lock = getattr(self, "_seek_lock", None)
            if seek_lock is None:
                self._handle_game_start(game_id)
            else:
                with seek_lock:
                    self._handle_game_start(game_id)

        elif etype == "gameFinish":
            game_id = self._event_game_id(event)
            if game_id:
                self._mark_game_completed(game_id)
            if self._draining.is_set() and not self.active_games:
                self._shutdown.set()

        elif etype in ("challengeDeclined", "challengeCanceled"):
            challenge_id, event_target = self._challenge_event_identity(event)
            event_reason = self._challenge_event_reason(event)
            if not self._pending_challenge_id:
                self._cooldown_bot(event_target, duration=600)
                return
            if challenge_id and challenge_id != self._pending_challenge_id:
                self._cooldown_bot(event_target, duration=600)
                return
            if self._pending_challenge_id and not challenge_id:
                return

            target = self._pending_challenge_target or event_target
            label = "declined" if etype == "challengeDeclined" else "canceled"
            reason_suffix = f": {event_reason}" if event_reason else ""
            print(f"  Challenge to {target} {label}{reason_suffix}")
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
    parser.add_argument("--accept-rated", action="store_true", default=False)
    parser.add_argument("--no-rated", dest="accept_rated", action="store_false")
    parser.add_argument("--accept-casual", action="store_true", default=True)
    parser.add_argument("--no-casual", dest="accept_casual", action="store_false")
    parser.add_argument(
        "--max-games", type=int, default=1, help="Max concurrent games (default: 1)"
    )
    parser.add_argument(
        "--quit-after-games",
        type=int,
        default=0,
        help="Stop the bot after this many completed games (default: run until stopped)",
    )
    parser.add_argument(
        "--seek",
        action="store_true",
        default=False,
        help="Actively challenge online bots",
    )
    parser.add_argument(
        "--seek-dry-run",
        action="store_true",
        default=False,
        help="Preview one seek target and exit without launching the engine or challenging",
    )
    parser.add_argument(
        "--config-check",
        action="store_true",
        default=False,
        help="Print resolved bot policy/resources and exit without engine/API access",
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
        "--engine-self-test",
        dest="engine_self_test",
        action="store_true",
        default=True,
        help="Run startup engine probes before listening to Lichess (default)",
    )
    parser.add_argument(
        "--no-engine-self-test",
        dest="engine_self_test",
        action="store_false",
        help="Skip startup engine probes",
    )
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        default=False,
        help="Run the engine self-test and exit without connecting to Lichess",
    )
    parser.add_argument(
        "--elo-seek",
        action="store_true",
        default=False,
        help="Filter opponents by Elo proximity (starts tight, widens if no match)",
    )
    parser.add_argument(
        "--seek-highest-rated",
        action="store_true",
        default=False,
        help=(
            "Seek the highest-rated eligible online bot first, then descend as "
            "declines/cooldowns remove candidates"
        ),
    )
    parser.add_argument(
        "--avoid-repeat-format",
        action="store_true",
        default=False,
        help="Do not seek the same bot twice in the same speed category",
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
    parser.add_argument(
        "--min-rated-opponent-elo",
        type=int,
        default=MIN_RATED_OPPONENT_ELO,
        help=(
            "Reject rated seeks/challenges below this opponent rating "
            f"(default: {MIN_RATED_OPPONENT_ELO}, 0 disables)"
        ),
    )
    args = parser.parse_args()

    needs_engine = (
        not args.seek_dry_run and not args.config_check
    ) or args.self_test_only
    if needs_engine and not ENGINE.exists():
        print(f"ERROR: Engine not found at {ENGINE}")
        print("Build with: cd build && cmake .. && make -j8")
        sys.exit(1)
    if needs_engine and not WEIGHTS.exists():
        print(f"ERROR: Weights not found at {WEIGHTS}")
        sys.exit(1)

    config_errors = validate_bot_config(args)
    if config_errors:
        for error in config_errors:
            print(f"ERROR: {error}")
        sys.exit(2)

    if args.config_check:
        print_config_check(args)
        sys.exit(0)

    if args.seek_dry_run:
        if requests is None:
            print("ERROR: Python package 'requests' is required for Lichess API mode.")
            print("Install it in this environment or run with the project Python.")
            sys.exit(1)
        api_key = load_api_key()
        bot = LichessBot(api_key, args)
        profile = bot.get_profile()
        bot.bot_id = profile.get("id", "")
        bot.username = profile.get("username", bot.bot_id)
        sys.exit(0 if bot.preview_seek_once() else 1)

    try:
        bot_lock = BotInstanceLock(BOT_LOCK_PATH)
        bot_lock.acquire()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        print(
            "Set METALFISH_ALLOW_CONCURRENT_BOT=1 only for deliberate "
            "throughput experiments."
        )
        sys.exit(2)

    if args.self_test_only:
        try:
            bot = LichessBot("", args)
            sys.exit(0 if bot._run_engine_self_test() else 1)
        finally:
            bot_lock.release()

    if requests is None:
        print("ERROR: Python package 'requests' is required for Lichess API mode.")
        print("Install it in this environment or run with the project Python.")
        bot_lock.release()
        sys.exit(1)

    api_key = load_api_key()
    bot = LichessBot(api_key, args)
    try:
        bot.run()
    finally:
        bot_lock.release()


if __name__ == "__main__":
    main()
