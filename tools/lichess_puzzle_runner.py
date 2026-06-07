#!/usr/bin/env python3
"""Run MetalFish against official Lichess puzzle batches."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import email.utils
import io
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import chess
import chess.pgn

try:
    import requests
except ModuleNotFoundError:
    requests = None

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = ROOT / "build" / "metalfish"
RESULTS_DIR = ROOT / "results" / "lichess_puzzles"
LICHESS_API = "https://lichess.org/api"
DEFAULT_ANE_WEIGHTS = ROOT / "networks" / "t1-512x15x8h-distilled-swa-3395000.pb.gz"
DEFAULT_ANE_MODEL = ROOT / "build" / "coreml" / "compiled" / "t1-512-heads-b8.mlmodelc"
DEFAULT_ANE_ROOT_HINTS = False
DEFAULT_ANE_ONLY_PAWN_ENDGAMES = False
DEFAULT_ANE_ROOT_HINT_WAIT_MS = 0
DEFAULT_ANE_MIN_BUDGET_MS = 0
SETOPTION_ALIASES = {
    "HybridANEWeightsPath": "HybridANEWeights",
}
USER_AGENT = os.environ.get(
    "METALFISH_HTTP_USER_AGENT",
    "MetalFishPuzzleCI/1.0 (+https://github.com/NripeshN/MetalFish)",
)


class LichessRateLimited(RuntimeError):
    def __init__(self, wait_s: float):
        super().__init__(f"Lichess rate limited this run for {wait_s:.0f}s")
        self.wait_s = wait_s


def auto_threads() -> int:
    reserve = max(0, int(os.environ.get("METALFISH_PUZZLE_THREAD_RESERVE", "1")))
    cores = os.cpu_count() or 1
    try:
        if sys.platform == "darwin":
            cores = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.ncpu"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return max(1, cores - reserve)


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
                ["vm_stat"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            pages: dict[str, int] = {}
            for line in out.splitlines():
                if not line.startswith("Pages "):
                    continue
                name, _, value = line.partition(":")
                try:
                    pages[name.removeprefix("Pages ").strip()] = int(
                        value.strip().rstrip(".")
                    )
                except ValueError:
                    continue
            total_pages = (
                pages.get("free", 0)
                + pages.get("speculative", 0)
                + pages.get("inactive", 0) // 2
                + pages.get("purgeable", 0)
            )
            return (total_pages * page_size) // (1024 * 1024)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return 0


def auto_hash_mb() -> int:
    available = available_memory_mb()
    reserve_mb = max(
        1024, int(os.environ.get("METALFISH_PUZZLE_MEMORY_RESERVE_MB", "1536"))
    )
    if available <= 0:
        return 2048
    target = max(512, available - reserve_mb)
    return min(4096, (target // 256) * 256)


def load_token() -> str:
    for name in ("LICHESS_PUZZLES_TOKEN", "Lichess_Puzzles", "LICHESS_API_KEY"):
        token = os.environ.get(name)
        if token:
            return token
    raise RuntimeError("Missing LICHESS_PUZZLES_TOKEN / Lichess_Puzzles secret")


def retry_after_s(response: requests.Response, default_s: float) -> float:
    value = response.headers.get("Retry-After")
    if not value:
        return default_s
    try:
        return max(default_s, float(value))
    except ValueError:
        pass
    try:
        retry_time = email.utils.parsedate_to_datetime(value)
        return max(default_s, retry_time.timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        return default_s


class LichessPuzzleClient:
    def __init__(self, token: str, min_interval_s: float, backoff_s: float):
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
        }
        self.min_interval_s = max(0.0, min_interval_s)
        self.backoff_s = max(60.0, backoff_s)
        self._next_at = 0.0
        self._blocked_until = 0.0
        self._lock = threading.Lock()

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        with self._lock:
            now = time.monotonic()
            wait_s = max(self._next_at - now, self._blocked_until - now)
            if wait_s > 0:
                time.sleep(wait_s)
            response = self.session.request(
                method,
                f"{LICHESS_API}{path}",
                headers=self.headers,
                timeout=kwargs.pop("timeout", 30),
                **kwargs,
            )
            self._next_at = time.monotonic() + self.min_interval_s
            if response.status_code == 429:
                wait_s = retry_after_s(response, self.backoff_s)
                self._blocked_until = time.monotonic() + wait_s
                raise LichessRateLimited(wait_s)
            response.raise_for_status()
            return response

    def fetch_batch(self, angle: str, nb: int) -> list[dict]:
        response = self.request(
            "GET",
            f"/puzzle/batch/{angle}",
            params={"nb": nb},
        )
        data = response.json()
        puzzles = data.get("puzzles", [])
        return puzzles if isinstance(puzzles, list) else []

    def submit_batch(
        self,
        angle: str,
        solutions: list[dict],
        *,
        next_nb: int,
    ) -> list[dict]:
        if not solutions:
            return []
        response = self.request(
            "POST",
            f"/puzzle/batch/{angle}",
            params={"nb": next_nb},
            json={"solutions": solutions},
        )
        data = response.json()
        puzzles = data.get("puzzles", [])
        return puzzles if isinstance(puzzles, list) else []


@dataclass
class SearchAnswer:
    bestmove: str
    nodes: int = 0
    nps: int = 0
    depth: int = 0
    ane_hints: int = 0
    ane_hint_moves: int = 0
    ane_failures: int = 0
    ane_last_hints: str = ""
    hybrid_trace: str = ""
    hybrid_reason: str = ""
    hybrid_selected: str = ""
    hybrid_ab_move: str = ""
    hybrid_mcts_move: str = ""
    hybrid_ane_top: str = ""
    hybrid_ane_agrees_mcts: str = ""
    hybrid_ane_confirmed_mcts: str = ""
    hybrid_ane_top_score: str = ""
    hybrid_ane_score_margin: str = ""
    hybrid_ane_root: str = ""
    hybrid_ab_hints: str = ""
    hybrid_ab_verified_hints: str = ""
    final_summary: str = ""


class UCIEngine:
    def __init__(
        self,
        path: pathlib.Path,
        options: dict[str, str],
        cwd: pathlib.Path | None = None,
    ):
        self.path = path
        self.proc = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            text=True,
            bufsize=1,
        )
        self._stderr: list[str] = []
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self.proc.stderr, self._stderr),
            daemon=True,
        )
        self._stderr_thread.start()
        try:
            self.send("uci")
            self.wait_for("uciok", 120)
            for name, value in options.items():
                self.send(f"setoption name {name} value {value}")
            self.ready()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _read_stderr(stream, tail: list[str]) -> None:
        if stream is None:
            return
        for line in stream:
            tail.append(line.strip())
            if len(tail) > 20:
                tail.pop(0)

    def diagnostic_tail(self) -> str:
        parts = []
        if self.proc.poll() is not None:
            parts.append(f"exit={self.proc.returncode}")
        if self._stderr:
            parts.append("stderr=" + " | ".join(self._stderr[-3:]))
        return " (" + "; ".join(parts) + ")" if parts else ""

    def send(self, command: str) -> None:
        if self.proc.stdin is None or self.proc.poll() is not None:
            raise RuntimeError(f"Engine is not running{self.diagnostic_tail()}")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def wait_for(self, prefix: str, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        assert self.proc.stdout is not None
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"Engine died{self.diagnostic_tail()}")
                time.sleep(0.01)
                continue
            line = line.strip()
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"Timed out waiting for {prefix}{self.diagnostic_tail()}")

    def ready(self) -> None:
        self.send("isready")
        self.wait_for("readyok", 120)

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.ready()

    def search(self, board: chess.Board, movetime_ms: int) -> SearchAnswer:
        self.send(f"position fen {board.fen()}")
        self.send(f"go movetime {movetime_ms}")
        answer = SearchAnswer(bestmove="0000")
        timeout_s = max(10.0, movetime_ms / 1000.0 + 15.0)
        deadline = time.monotonic() + timeout_s
        assert self.proc.stdout is not None
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"Engine died during search{self.diagnostic_tail()}"
                    )
                time.sleep(0.01)
                continue
            line = line.strip()
            if line.startswith("bestmove"):
                parts = line.split()
                answer.bestmove = parts[1] if len(parts) > 1 else "0000"
                return answer
            update_answer_from_info(line, answer)
        self.send("stop")
        line = self.wait_for("bestmove", 5)
        parts = line.split()
        answer.bestmove = parts[1] if len(parts) > 1 else "0000"
        return answer

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=2)
            except Exception:
                pass


def engine_options(args) -> dict[str, str]:
    threads = args.threads or auto_threads()
    hash_mb = args.hash_mb or auto_hash_mb()
    options = {
        "Threads": str(threads),
        "Hash": str(hash_mb),
        "Ponder": "false",
        "MultiPV": "1",
        "MCTSMinibatchSize": "0",
        "MCTSMinimumKLDGainPerNode": "0.00005",
        "MCTSPolicySoftmaxTemp": "1.359",
        "MCTSRootPolicySoftmaxTemp": "1.6",
        "MCTSSmartPruningFactor": "1.33",
        "MCTSCacheHistoryLength": "0",
        "MCTSSolidTreeThreshold": "100",
    }
    if args.mode == "ab":
        options.update({"UseMCTS": "false", "UseHybridSearch": "false"})
    elif args.mode == "mcts":
        options.update(
            {
                "UseMCTS": "true",
                "UseHybridSearch": "false",
                "NNWeights": str(args.weights),
                "MCTSAddDirichletNoise": "false",
                "MCTSParityPreset": "false",
                "PureMCTSCPuctAtRoot": "2.2",
                "MCTSMaxThreads": "0",
                "TransformerLowTimeFallbackMs": "0",
            }
        )
    else:
        options.update(
            {
                "UseMCTS": "false",
                "UseHybridSearch": "true",
                "NNWeights": str(args.weights),
                "MCTSAddDirichletNoise": "false",
                "HybridABRootRejectMCTS": "true",
                "HybridMCTSRootReject": "true",
                "HybridMCTSABRootHints": "true",
                "HybridMCTSABRootHintDelayMs": "0",
                "HybridMCTSABRootHintCount": "8",
                "HybridABCandidateVerifyMs": "120",
                "HybridABCandidateVerifyCount": "5",
                "HybridMCTSUseSharedTT": "false",
                "HybridMCTSMinimumKLDGainPerNode": "0.0",
                "HybridMCTSThreads": "0",
                "HybridABThreads": "0",
                "HybridAutoABThreadsCap": "0",
                "MCTSMaxThreads": "0",
                "TransformerLowTimeFallbackMs": "0",
            }
        )
        if args.hybrid_trace:
            options["HybridTrace"] = "true"
        if args.hybrid_ane_root_probe:
            options.update(
                {
                    "HybridANERootProbe": "true",
                    "HybridANERootHints": (
                        "true" if args.hybrid_ane_root_hints else "false"
                    ),
                    "HybridANEConfirmMCTSOverride": (
                        "true" if args.hybrid_ane_confirm_mcts_override else "false"
                    ),
                    "HybridANEOnlyPawnEndgames": (
                        "true" if args.hybrid_ane_only_pawn_endgames else "false"
                    ),
                    "HybridANEWeights": str(args.hybrid_ane_weights),
                    "HybridANEModelPath": str(args.hybrid_ane_model_path),
                    "HybridANEComputeUnits": args.hybrid_ane_compute_units,
                    "HybridANERootHintCount": str(args.hybrid_ane_root_hint_count),
                    "HybridANERootHintWaitMs": str(args.hybrid_ane_root_hint_wait_ms),
                    "HybridANEMinBudgetMs": str(args.hybrid_ane_min_budget_ms),
                }
            )
    if args.syzygy_path:
        options["SyzygyPath"] = str(args.syzygy_path)
        options["SyzygyProbeDepth"] = "2"
        options["SyzygyProbeLimit"] = "6"
    options.update(parse_setoptions(args.setoption))
    return options


def update_answer_from_info(line: str, answer: SearchAnswer) -> None:
    if not line.startswith("info "):
        return
    if line.startswith("info string Hybrid: AB root hints from ANE"):
        answer.ane_hints += 1
        answer.ane_hint_moves += max(0, len(line.split()) - 8)
        answer.ane_last_hints = line.removeprefix(
            "info string Hybrid: AB root hints from ANE"
        ).strip()
    elif line.startswith("info string Hybrid: ANE root probe failed"):
        answer.ane_failures += 1
    elif line.startswith("info string HybridTrace:"):
        answer.hybrid_trace = line.removeprefix("info string ").strip()
        fields = parse_hybrid_trace_fields(answer.hybrid_trace)
        answer.hybrid_reason = fields.get("reason", "")
        answer.hybrid_selected = fields.get("selected", "")
        answer.hybrid_ab_move = fields.get("ABMove", "")
        answer.hybrid_mcts_move = fields.get("MCTSMove", "")
        answer.hybrid_ane_top = fields.get("ANETop", "")
        answer.hybrid_ane_agrees_mcts = fields.get("ANEAgreesMCTS", "")
        answer.hybrid_ane_confirmed_mcts = fields.get("ANEConfirmedMCTS", "")
        answer.hybrid_ane_top_score = fields.get("ANETopScore", "")
        answer.hybrid_ane_score_margin = fields.get("ANEScoreMargin", "")
        answer.hybrid_ane_root = fields.get("ANERoot", "")
        answer.hybrid_ab_hints = fields.get("ABHints", "")
        answer.hybrid_ab_verified_hints = fields.get("ABVerifiedHints", "")
    elif line.startswith("info string Final:"):
        answer.final_summary = line.removeprefix("info string ").strip()

    parts = line.split()
    for idx, token in enumerate(parts[:-1]):
        try:
            if token == "nodes":
                answer.nodes = int(parts[idx + 1])
            elif token == "nps":
                answer.nps = int(parts[idx + 1])
            elif token == "depth":
                answer.depth = int(parts[idx + 1])
        except ValueError:
            continue


def parse_hybrid_trace_fields(trace: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in trace.split():
        if "=" not in token:
            continue
        name, value = token.split("=", 1)
        if name:
            fields[name] = value
    return fields


def search_trace_fields(answer: SearchAnswer) -> dict[str, str]:
    fields: dict[str, str] = {}
    if answer.ane_last_hints:
        fields["ane_last_hints"] = answer.ane_last_hints
    if answer.hybrid_trace:
        fields["hybrid_trace"] = answer.hybrid_trace
    if answer.hybrid_reason:
        fields["hybrid_reason"] = answer.hybrid_reason
    if answer.hybrid_selected:
        fields["hybrid_selected"] = answer.hybrid_selected
    if answer.hybrid_ab_move:
        fields["hybrid_ab_move"] = answer.hybrid_ab_move
    if answer.hybrid_mcts_move:
        fields["hybrid_mcts_move"] = answer.hybrid_mcts_move
    if answer.hybrid_ane_top:
        fields["hybrid_ane_top"] = answer.hybrid_ane_top
    if answer.hybrid_ane_agrees_mcts:
        fields["hybrid_ane_agrees_mcts"] = answer.hybrid_ane_agrees_mcts
    if answer.hybrid_ane_confirmed_mcts:
        fields["hybrid_ane_confirmed_mcts"] = answer.hybrid_ane_confirmed_mcts
    if answer.hybrid_ane_top_score:
        fields["hybrid_ane_top_score"] = answer.hybrid_ane_top_score
    if answer.hybrid_ane_score_margin:
        fields["hybrid_ane_score_margin"] = answer.hybrid_ane_score_margin
    if answer.hybrid_ane_root:
        fields["hybrid_ane_root"] = answer.hybrid_ane_root
    if answer.hybrid_ab_hints:
        fields["hybrid_ab_hints"] = answer.hybrid_ab_hints
    if answer.hybrid_ab_verified_hints:
        fields["hybrid_ab_verified_hints"] = answer.hybrid_ab_verified_hints
    if answer.final_summary:
        fields["final_summary"] = answer.final_summary
    return fields


def initial_ane_stats(args) -> dict[str, object]:
    requested = bool(getattr(args, "hybrid_ane_root_probe", False))
    return {
        "ane_probe_requested": requested,
        "hybrid_trace_requested": bool(getattr(args, "hybrid_trace", False)),
        "ane_root_hints_requested": bool(getattr(args, "hybrid_ane_root_hints", False)),
        "ane_confirm_mcts_override_requested": bool(
            getattr(args, "hybrid_ane_confirm_mcts_override", False)
        ),
        "ane_only_pawn_endgames": bool(
            getattr(args, "hybrid_ane_only_pawn_endgames", DEFAULT_ANE_ONLY_PAWN_ENDGAMES)
        ),
        "ane_compute_units": (
            str(getattr(args, "hybrid_ane_compute_units", "")) if requested else ""
        ),
        "ane_weights": (
            str(getattr(args, "hybrid_ane_weights", "")) if requested else ""
        ),
        "ane_model_path": (
            str(getattr(args, "hybrid_ane_model_path", "")) if requested else ""
        ),
        "ane_searches": 0,
        "ane_trace_searches": 0,
        "ane_hints": 0,
        "ane_hint_moves": 0,
        "ane_failures": 0,
        "ane_root_nonempty": 0,
        "ane_top_moves": 0,
        "ane_agrees_mcts": 0,
        "ane_confirmed_mcts": 0,
    }


def update_ane_stats(stats: dict[str, object], result: dict) -> None:
    searches = result.get("searches", [])
    if not isinstance(searches, list):
        return
    for search in searches:
        if not isinstance(search, dict):
            continue
        stats["ane_searches"] = int(stats.get("ane_searches", 0)) + 1
        stats["ane_hints"] = int(stats.get("ane_hints", 0)) + int(
            search.get("ane_hints") or 0
        )
        stats["ane_hint_moves"] = int(stats.get("ane_hint_moves", 0)) + int(
            search.get("ane_hint_moves") or 0
        )
        stats["ane_failures"] = int(stats.get("ane_failures", 0)) + int(
            search.get("ane_failures") or 0
        )

        top = str(search.get("hybrid_ane_top", ""))
        root = str(search.get("hybrid_ane_root", ""))
        agrees = str(search.get("hybrid_ane_agrees_mcts", ""))
        confirmed = str(search.get("hybrid_ane_confirmed_mcts", ""))
        has_trace = bool(top or root or agrees or confirmed)
        if has_trace:
            stats["ane_trace_searches"] = int(stats.get("ane_trace_searches", 0)) + 1
        if root and root != "[]":
            stats["ane_root_nonempty"] = int(stats.get("ane_root_nonempty", 0)) + 1
        if top and top not in {"none", "0000"}:
            stats["ane_top_moves"] = int(stats.get("ane_top_moves", 0)) + 1
        if agrees == "1":
            stats["ane_agrees_mcts"] = int(stats.get("ane_agrees_mcts", 0)) + 1
        if confirmed == "1":
            stats["ane_confirmed_mcts"] = int(stats.get("ane_confirmed_mcts", 0)) + 1


def parse_setoptions(items: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--setoption requires NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        name = SETOPTION_ALIASES.get(name, name)
        if not name:
            raise ValueError(f"--setoption has an empty option name: {item!r}")
        options[name] = value.strip()
    return options


def validate_ane_args(args) -> None:
    if not args.hybrid_ane_root_probe:
        return
    if args.mode != "hybrid":
        raise RuntimeError("--hybrid-ane-root-probe requires --mode hybrid")
    if not args.hybrid_ane_weights.exists():
        raise RuntimeError(f"ANE weights not found at {args.hybrid_ane_weights}")
    if not args.hybrid_ane_model_path.exists():
        raise RuntimeError(
            f"ANE Core ML model not found at {args.hybrid_ane_model_path}"
        )


def board_from_api_puzzle(item: dict) -> chess.Board:
    puzzle = item.get("puzzle", {})
    game = item.get("game", {})
    pgn = game.get("pgn")
    initial_ply = int(puzzle.get("initialPly", 0))
    if not isinstance(pgn, str) or not pgn.strip():
        raise ValueError("puzzle is missing PGN")
    parsed = chess.pgn.read_game(io.StringIO(pgn))
    if parsed is None:
        raise ValueError("could not parse puzzle PGN")
    board = parsed.board()
    target_ply = initial_ply + 1
    for ply, move in enumerate(parsed.mainline_moves(), start=1):
        if ply > target_ply:
            break
        board.push(move)
    return board


def board_from_csv_puzzle(item: dict) -> chess.Board:
    puzzle = item.get("puzzle", {})
    fen = puzzle.get("fen")
    moves = puzzle.get("moves", [])
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError("CSV puzzle is missing FEN")
    if not isinstance(moves, list) or len(moves) < 2:
        raise ValueError("CSV puzzle is missing opponent move and solution")
    board = chess.Board(fen)
    opponent_move = normalize_move(str(moves[0]), board)
    if opponent_move is None:
        raise ValueError("CSV puzzle has illegal opponent move")
    board.push(chess.Move.from_uci(opponent_move))
    return board


def board_from_puzzle(item: dict) -> chess.Board:
    if item.get("source") == "lichess_csv":
        return board_from_csv_puzzle(item)
    return board_from_api_puzzle(item)


def csv_puzzle_item(row: dict[str, str]) -> dict:
    moves = row.get("Moves", "").split()
    themes = row.get("Themes", "").split()
    puzzle: dict[str, object] = {
        "id": row.get("PuzzleId", ""),
        "fen": row.get("FEN", ""),
        "moves": moves,
        "solution": moves[1:],
        "themes": themes,
        "gameUrl": row.get("GameUrl", ""),
        "openingTags": row.get("OpeningTags", ""),
    }
    for source_key, target_key in (
        ("Rating", "rating"),
        ("RatingDeviation", "ratingDeviation"),
        ("Popularity", "popularity"),
        ("NbPlays", "nbPlays"),
    ):
        try:
            puzzle[target_key] = int(row.get(source_key, ""))
        except ValueError:
            pass
    return {"source": "lichess_csv", "puzzle": puzzle}


def csv_row_matches(
    row: dict[str, str],
    *,
    min_rating: int,
    max_rating: int,
    min_popularity: int,
    themes: set[str],
) -> bool:
    try:
        rating = int(row.get("Rating", "0"))
    except ValueError:
        return False
    if min_rating and rating < min_rating:
        return False
    if max_rating and rating > max_rating:
        return False
    if min_popularity > -100:
        try:
            popularity = int(row.get("Popularity", "-101"))
        except ValueError:
            return False
        if popularity < min_popularity:
            return False
    if themes:
        row_themes = set(row.get("Themes", "").split())
        if not row_themes.intersection(themes):
            return False
    return True


def parse_theme_filter(value: str) -> set[str]:
    themes: set[str] = set()
    for token in value.replace(",", " ").split():
        token = token.strip()
        if token:
            themes.add(token)
    return themes


def parse_auto_int(value: str, *, option_name: str) -> int:
    if value == "auto":
        return 0
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must be an integer or 'auto'"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{option_name} must be >= 0")
    return parsed


def iter_offline_csv_puzzles(args) -> list[dict]:
    themes = parse_theme_filter(args.themes)
    puzzles: list[dict] = []
    with args.offline_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not csv_row_matches(
                row,
                min_rating=args.min_rating,
                max_rating=args.max_rating,
                min_popularity=args.min_popularity,
                themes=themes,
            ):
                continue
            item = csv_puzzle_item(row)
            try:
                board_from_csv_puzzle(item)
            except ValueError:
                continue
            puzzles.append(item)
            if len(puzzles) >= args.max_puzzles:
                break
    return puzzles


def normalize_move(uci: str, board: chess.Board) -> str | None:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None
    if move in board.legal_moves:
        return move.uci()
    return None


def is_mating_move(board: chess.Board, uci: str | None) -> bool:
    if uci is None:
        return False
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return False
    if move not in board.legal_moves:
        return False
    probe = board.copy(stack=False)
    probe.push(move)
    return probe.is_checkmate()


def solve_puzzle(engine: UCIEngine, item: dict, movetime_ms: int) -> dict:
    puzzle = item.get("puzzle", {})
    puzzle_id = str(puzzle.get("id", ""))
    solution = puzzle.get("solution", [])
    if not isinstance(solution, list) or not solution:
        return {"id": puzzle_id, "solved": False, "error": "missing_solution"}

    started = time.monotonic()
    board = board_from_puzzle(item)
    engine.new_game()
    searches: list[dict] = []

    for idx, expected_raw in enumerate(solution):
        expected = normalize_move(str(expected_raw), board)
        if expected is None:
            return {
                "id": puzzle_id,
                "solved": False,
                "error": "illegal_solution_move",
                "expected": str(expected_raw),
            }

        if idx % 2 == 1:
            board.push(chess.Move.from_uci(expected))
            continue

        answer = engine.search(board, movetime_ms)
        actual = normalize_move(answer.bestmove, board)
        search_record = {
            "ply": idx,
            "expected": expected,
            "actual": actual or answer.bestmove,
            "nodes": answer.nodes,
            "nps": answer.nps,
            "depth": answer.depth,
            "ane_hints": answer.ane_hints,
            "ane_hint_moves": answer.ane_hint_moves,
            "ane_failures": answer.ane_failures,
        }
        search_record.update(search_trace_fields(answer))
        searches.append(search_record)
        mating_alternative = actual != expected and is_mating_move(board, actual)
        if mating_alternative:
            search_record["accepted_mating_alternative"] = True
            return {
                "id": puzzle_id,
                "solved": True,
                "rating": puzzle.get("rating"),
                "themes": puzzle.get("themes", []),
                "searches": searches,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            }
        if actual != expected:
            return {
                "id": puzzle_id,
                "solved": False,
                "rating": puzzle.get("rating"),
                "themes": puzzle.get("themes", []),
                "searches": searches,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            }
        board.push(chess.Move.from_uci(expected))

    return {
        "id": puzzle_id,
        "solved": True,
        "rating": puzzle.get("rating"),
        "themes": puzzle.get("themes", []),
        "searches": searches,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


def tag_repeat_result(result: dict, repeat_idx: int, repeat_count: int) -> dict:
    if repeat_count <= 1:
        return result
    tagged = dict(result)
    puzzle_id = str(tagged.get("id", ""))
    tagged["puzzle_id"] = puzzle_id
    tagged["repeat"] = repeat_idx + 1
    tagged["id"] = f"{puzzle_id}#r{repeat_idx + 1}"
    return tagged


def write_summary(path: pathlib.Path, stats: dict) -> None:
    total = max(1, int(stats.get("puzzles", 0)))
    solved = int(stats.get("solved", 0))
    accuracy = solved / total
    lines = [
        "# Lichess Puzzle Run",
        "",
        f"- Puzzles: {stats.get('puzzles', 0)}",
        f"- Solved: {solved}",
        f"- Accuracy: {accuracy:.2%}",
        f"- Engine mode: {stats.get('mode')}",
        f"- Threads: {stats.get('threads')}",
        f"- Hash: {stats.get('hash_mb')} MB",
        f"- Movetime: {stats.get('movetime_ms')} ms",
        f"- Rated submission: {stats.get('rated')}",
        f"- Duration: {stats.get('duration_s', 0):.1f}s",
    ]
    if int(stats.get("repeat_puzzles", 1)) > 1:
        lines.append(f"- Repeat passes: {stats.get('repeat_puzzles')}")
    if stats.get("source"):
        lines.append(f"- Source: {stats.get('source')}")
    if stats.get("ended"):
        lines.append(f"- Ended: {stats.get('ended')}")
    if stats.get("rate_limit_events"):
        lines.append(f"- Rate-limit events: {stats.get('rate_limit_events')}")
    if stats.get("ane_probe_requested"):
        lines.extend(
            [
                f"- ANE root probe: requested ({stats.get('ane_compute_units')})",
                f"- HybridTrace requested: {stats.get('hybrid_trace_requested')}",
                f"- ANE root hints requested: {stats.get('ane_root_hints_requested')}",
                "- ANE-confirm MCTS override requested: "
                f"{stats.get('ane_confirm_mcts_override_requested')}",
                f"- ANE pawn-only gate: {stats.get('ane_only_pawn_endgames')}",
                f"- ANE searches: {stats.get('ane_searches', 0)}",
                f"- ANE trace fields: {stats.get('ane_trace_searches', 0)}",
                f"- ANE non-empty roots: {stats.get('ane_root_nonempty', 0)}",
                f"- ANE top moves: {stats.get('ane_top_moves', 0)}",
                f"- ANE hint lines: {stats.get('ane_hints', 0)}",
                f"- ANE hint moves: {stats.get('ane_hint_moves', 0)}",
                f"- ANE agrees with MCTS: {stats.get('ane_agrees_mcts', 0)}",
                f"- ANE-confirmed MCTS overrides: {stats.get('ane_confirmed_mcts', 0)}",
                f"- ANE failures: {stats.get('ane_failures', 0)}",
                f"- ANE weights: {stats.get('ane_weights')}",
                f"- ANE model: {stats.get('ane_model_path')}",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def wait_after_rate_limit(
    exc: LichessRateLimited,
    *,
    deadline: float,
    events_seen: int,
    max_events: int,
) -> bool:
    wait_s = max(60.0, exc.wait_s)
    remaining_s = deadline - time.monotonic()
    if events_seen >= max_events or remaining_s <= wait_s + 5.0:
        return False
        print(f"Rate limited; waiting {wait_s:.0f}s before retrying", flush=True)
    time.sleep(wait_s)
    return True


def run(args) -> int:
    if args.offline_csv:
        return run_offline(args)
    if requests is None:
        raise RuntimeError("Python package 'requests' is required")
    token = load_token()
    if not args.engine.exists():
        raise RuntimeError(f"Engine not found at {args.engine}")
    if args.mode in {"mcts", "hybrid"} and not args.weights.exists():
        raise RuntimeError(f"Transformer weights not found at {args.weights}")
    validate_ane_args(args)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    jsonl_path = args.results_dir / f"lichess-puzzles-{stamp}.jsonl"
    summary_path = args.results_dir / f"lichess-puzzles-{stamp}.md"

    options = engine_options(args)
    threads = int(options["Threads"])
    hash_mb = int(options["Hash"])
    client = LichessPuzzleClient(
        token,
        min_interval_s=args.request_interval_s,
        backoff_s=args.rate_limit_backoff_s,
    )
    deadline = time.monotonic() + args.max_minutes * 60.0
    rate_limit_events = 0
    while True:
        try:
            puzzles = client.fetch_batch(args.angle, args.batch_size)
            break
        except LichessRateLimited as exc:
            rate_limit_events += 1
            if wait_after_rate_limit(
                exc,
                deadline=deadline,
                events_seen=rate_limit_events,
                max_events=args.max_rate_limit_waits,
            ):
                continue
            stats = {
                "puzzles": 0,
                "solved": 0,
                "mode": args.mode,
                "threads": threads,
                "hash_mb": hash_mb,
                "movetime_ms": args.movetime_ms,
                "rated": args.rated,
                "duration_s": 0.0,
                "ended": "rate_limited",
                "retry_after_s": exc.wait_s,
                "rate_limit_events": rate_limit_events,
            }
            write_summary(summary_path, stats)
            print(
                f"Rate limited before first batch; retry after {exc.wait_s:.0f}s",
                flush=True,
            )
            return 0

    engine = UCIEngine(args.engine, options, args.engine_cwd)
    solved = 0
    total = 0
    started = time.monotonic()
    ended = "completed"
    ane_stats = initial_ane_stats(args)

    print(
        f"Puzzle run: mode={args.mode}, threads={threads}, hash={hash_mb} MB, "
        f"movetime={args.movetime_ms} ms, batch={args.batch_size}, rated={args.rated}",
        flush=True,
    )
    print(
        f"Resources: logical={os.cpu_count() or 1}, available_memory={available_memory_mb()} MB, "
        f"thread_reserve={os.environ.get('METALFISH_PUZZLE_THREAD_RESERVE', '1')}, "
        f"memory_reserve={os.environ.get('METALFISH_PUZZLE_MEMORY_RESERVE_MB', '1536')} MB",
        flush=True,
    )

    try:
        with jsonl_path.open("w") as out:
            while puzzles and total < args.max_puzzles and time.monotonic() < deadline:
                batch_results: list[dict] = []
                for item in puzzles:
                    if total >= args.max_puzzles or time.monotonic() >= deadline:
                        break
                    try:
                        result = solve_puzzle(engine, item, args.movetime_ms)
                    except Exception as exc:
                        puzzle_id = str(item.get("puzzle", {}).get("id", ""))
                        result = {
                            "id": puzzle_id,
                            "solved": False,
                            "error": str(exc),
                        }
                    total += 1
                    solved += 1 if result.get("solved") else 0
                    update_ane_stats(ane_stats, result)
                    batch_results.append(result)
                    out.write(json.dumps(result, sort_keys=True) + "\n")
                    out.flush()
                    if total % args.progress_interval == 0:
                        print(
                            f"Progress: {solved}/{total} "
                            f"({solved / max(1, total):.1%})",
                            flush=True,
                        )

                solutions = [
                    {
                        "id": result.get("id", ""),
                        "win": bool(result.get("solved")),
                        "rated": bool(args.rated),
                    }
                    for result in batch_results
                    if result.get("id")
                ]
                next_nb = args.batch_size if time.monotonic() < deadline else 0
                while True:
                    try:
                        puzzles = client.submit_batch(
                            args.angle, solutions, next_nb=next_nb
                        )
                        break
                    except LichessRateLimited as exc:
                        rate_limit_events += 1
                        if wait_after_rate_limit(
                            exc,
                            deadline=deadline,
                            events_seen=rate_limit_events,
                            max_events=args.max_rate_limit_waits,
                        ):
                            continue
                        print(
                            f"Rate limited after {total} puzzle(s); "
                            f"retry after {exc.wait_s:.0f}s",
                            flush=True,
                        )
                        ended = "rate_limited"
                        puzzles = []
                        break
    finally:
        engine.close()

    duration_s = time.monotonic() - started
    stats = {
        "puzzles": total,
        "solved": solved,
        "mode": args.mode,
        "threads": threads,
        "hash_mb": hash_mb,
        "movetime_ms": args.movetime_ms,
        "rated": args.rated,
        "duration_s": duration_s,
        "ended": ended,
        "rate_limit_events": rate_limit_events,
    }
    stats.update(ane_stats)
    write_summary(summary_path, stats)
    print(
        f"Finished: solved {solved}/{total} "
        f"({solved / max(1, total):.2%}) in {duration_s:.1f}s",
        flush=True,
    )
    print(f"Results: {jsonl_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)

    accuracy = solved / max(1, total)
    if total == 0:
        return 2
    if accuracy < args.min_accuracy:
        return 1
    return 0


def run_offline(args) -> int:
    if not args.engine.exists():
        raise RuntimeError(f"Engine not found at {args.engine}")
    if not args.offline_csv.exists():
        raise RuntimeError(f"Offline puzzle CSV not found at {args.offline_csv}")
    if args.mode in {"mcts", "hybrid"} and not args.weights.exists():
        raise RuntimeError(f"Transformer weights not found at {args.weights}")
    validate_ane_args(args)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    jsonl_path = args.results_dir / f"lichess-puzzles-offline-{stamp}.jsonl"
    summary_path = args.results_dir / f"lichess-puzzles-offline-{stamp}.md"

    options = engine_options(args)
    threads = int(options["Threads"])
    hash_mb = int(options["Hash"])
    puzzles = iter_offline_csv_puzzles(args)
    if not puzzles:
        raise RuntimeError("No offline puzzles matched the selected filters")

    engine = UCIEngine(args.engine, options, args.engine_cwd)
    solved = 0
    total = 0
    started = time.monotonic()
    deadline = started + args.max_minutes * 60.0
    ended = "completed"
    ane_stats = initial_ane_stats(args)

    print(
        f"Offline puzzle run: mode={args.mode}, threads={threads}, "
        f"hash={hash_mb} MB, movetime={args.movetime_ms} ms, "
        f"puzzles={len(puzzles)}, repeats={args.repeat_puzzles}, "
        f"source={args.offline_csv}",
        flush=True,
    )
    try:
        with jsonl_path.open("w") as out:
            for repeat_idx in range(args.repeat_puzzles):
                for item in puzzles:
                    if time.monotonic() >= deadline:
                        ended = "time_budget"
                        break
                    try:
                        result = solve_puzzle(engine, item, args.movetime_ms)
                    except Exception as exc:
                        result = {
                            "id": str(item.get("puzzle", {}).get("id", "")),
                            "solved": False,
                            "error": str(exc),
                        }
                    result = tag_repeat_result(result, repeat_idx, args.repeat_puzzles)
                    total += 1
                    solved += 1 if result.get("solved") else 0
                    update_ane_stats(ane_stats, result)
                    out.write(json.dumps(result, sort_keys=True) + "\n")
                    out.flush()
                    if total % args.progress_interval == 0:
                        print(
                            f"Progress: {solved}/{total} ({solved / total:.1%})",
                            flush=True,
                        )
                if ended == "time_budget":
                    break
    finally:
        engine.close()

    duration_s = time.monotonic() - started
    stats = {
        "puzzles": total,
        "solved": solved,
        "mode": args.mode,
        "threads": threads,
        "hash_mb": hash_mb,
        "movetime_ms": args.movetime_ms,
        "rated": False,
        "duration_s": duration_s,
        "ended": ended,
        "source": str(args.offline_csv),
        "rate_limit_events": 0,
        "repeat_puzzles": args.repeat_puzzles,
    }
    stats.update(ane_stats)
    write_summary(summary_path, stats)
    print(
        f"Finished: solved {solved}/{total} "
        f"({solved / max(1, total):.2%}) in {duration_s:.1f}s",
        flush=True,
    )
    print(f"Results: {jsonl_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)

    accuracy = solved / max(1, total)
    if total == 0:
        return 2
    if accuracy < args.min_accuracy:
        return 1
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=pathlib.Path, default=ENGINE)
    parser.add_argument("--engine-cwd", type=pathlib.Path, default=None)
    parser.add_argument("--mode", choices=("ab", "mcts", "hybrid"), default="ab")
    parser.add_argument(
        "--weights",
        type=pathlib.Path,
        default=ROOT / "networks" / "BT4-1024x15x32h-swa-6147500.pb",
    )
    parser.add_argument("--syzygy-path", type=pathlib.Path, default=None)
    parser.add_argument(
        "--hybrid-ane-root-probe",
        action="store_true",
        default=False,
        help="Enable the Hybrid ANE/Core ML root hint probe.",
    )
    parser.add_argument(
        "--hybrid-ane-root-hints",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ANE_ROOT_HINTS,
        help="Use ANE root ordering as AB search hints; final ANE evidence remains available without this.",
    )
    parser.add_argument(
        "--hybrid-ane-confirm-mcts-override",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow ANE agreement to confirm an MCTS override in hybrid arbitration.",
    )
    parser.add_argument(
        "--hybrid-ane-only-pawn-endgames",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ANE_ONLY_PAWN_ENDGAMES,
        help="Restrict ANE root probing to pawn-only endgames.",
    )
    parser.add_argument(
        "--hybrid-trace",
        action="store_true",
        default=False,
        help="Enable HybridTrace so puzzle reports include AB/MCTS/ANE arbitration fields.",
    )
    parser.add_argument(
        "--hybrid-ane-weights",
        type=pathlib.Path,
        default=DEFAULT_ANE_WEIGHTS,
        help="Lc0 weights for the ANE/Core ML root hint probe.",
    )
    parser.add_argument(
        "--hybrid-ane-model-path",
        type=pathlib.Path,
        default=DEFAULT_ANE_MODEL,
        help="Compiled .mlmodelc or .mlpackage for the ANE/Core ML root hint probe.",
    )
    parser.add_argument(
        "--hybrid-ane-compute-units",
        choices=("cpu", "cpu-gpu", "cpu-ne", "all"),
        default="cpu-ne",
    )
    parser.add_argument("--hybrid-ane-root-hint-count", type=int, default=10)
    parser.add_argument(
        "--hybrid-ane-root-hint-wait-ms",
        type=int,
        default=DEFAULT_ANE_ROOT_HINT_WAIT_MS,
    )
    parser.add_argument(
        "--hybrid-ane-min-budget-ms",
        type=int,
        default=DEFAULT_ANE_MIN_BUDGET_MS,
    )
    parser.add_argument(
        "--setoption",
        action="append",
        default=[],
        help="Additional UCI option as NAME=VALUE. Can be repeated.",
    )
    parser.add_argument(
        "--threads",
        type=lambda value: parse_auto_int(value, option_name="--threads"),
        default=0,
    )
    parser.add_argument(
        "--hash-mb",
        type=lambda value: parse_auto_int(value, option_name="--hash-mb"),
        default=0,
    )
    parser.add_argument("--movetime-ms", type=int, default=3000)
    parser.add_argument("--max-minutes", type=float, default=335.0)
    parser.add_argument("--max-puzzles", type=int, default=1000000)
    parser.add_argument(
        "--repeat-puzzles",
        type=int,
        default=1,
        help="Offline-only repeat passes for volatility/regression gates.",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--angle", default="mix")
    parser.add_argument("--rated", action="store_true", default=False)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--request-interval-s", type=float, default=2.0)
    parser.add_argument("--rate-limit-backoff-s", type=float, default=65.0)
    parser.add_argument("--max-rate-limit-waits", type=int, default=5)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_DIR)
    parser.add_argument(
        "--offline-csv",
        type=pathlib.Path,
        default=None,
        help="Run a local Lichess puzzle CSV sample instead of the live API.",
    )
    parser.add_argument("--min-rating", type=int, default=0)
    parser.add_argument("--max-rating", type=int, default=0)
    parser.add_argument("--min-popularity", type=int, default=-101)
    parser.add_argument(
        "--themes",
        default="",
        help="Comma/space-separated Lichess puzzle themes; any match is accepted.",
    )
    args = parser.parse_args(argv)
    args.batch_size = max(1, min(50, args.batch_size))
    args.movetime_ms = max(1, args.movetime_ms)
    args.max_minutes = max(0.1, args.max_minutes)
    args.max_puzzles = max(1, args.max_puzzles)
    args.repeat_puzzles = max(1, args.repeat_puzzles)
    args.progress_interval = max(1, args.progress_interval)
    args.max_rate_limit_waits = max(0, args.max_rate_limit_waits)
    args.hybrid_ane_root_hint_count = max(1, min(32, args.hybrid_ane_root_hint_count))
    args.hybrid_ane_root_hint_wait_ms = max(0, args.hybrid_ane_root_hint_wait_ms)
    args.hybrid_ane_min_budget_ms = max(0, args.hybrid_ane_min_budget_ms)
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv or sys.argv[1:]))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
