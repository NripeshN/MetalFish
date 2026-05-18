#!/usr/bin/env python3
"""Run MetalFish against official Lichess puzzle batches."""

from __future__ import annotations

import argparse
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


class UCIEngine:
    def __init__(self, path: pathlib.Path, options: dict[str, str]):
        self.path = path
        self.proc = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
            if not line.startswith("info "):
                continue
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
                "MCTSMaxThreads": "0",
            }
        )
    else:
        options.update(
            {
                "UseMCTS": "false",
                "UseHybridSearch": "true",
                "NNWeights": str(args.weights),
                "MCTSAddDirichletNoise": "false",
                "HybridMCTSRootReject": "true",
                "HybridMCTSABRootHints": "true",
                "HybridMCTSABRootHintDelayMs": "25",
                "HybridMCTSABRootHintCount": "4",
                "HybridMCTSUseSharedTT": "false",
                "HybridMCTSMinimumKLDGainPerNode": "0.0",
                "HybridMCTSThreads": "0",
                "HybridABThreads": "0",
                "HybridAutoABThreadsCap": "0",
                "MCTSMaxThreads": "0",
            }
        )
    if args.syzygy_path:
        options["SyzygyPath"] = str(args.syzygy_path)
        options["SyzygyProbeDepth"] = "2"
        options["SyzygyProbeLimit"] = "6"
    return options


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


def normalize_move(uci: str, board: chess.Board) -> str | None:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None
    if move in board.legal_moves:
        return move.uci()
    return None


def solve_puzzle(engine: UCIEngine, item: dict, movetime_ms: int) -> dict:
    puzzle = item.get("puzzle", {})
    puzzle_id = str(puzzle.get("id", ""))
    solution = puzzle.get("solution", [])
    if not isinstance(solution, list) or not solution:
        return {"id": puzzle_id, "solved": False, "error": "missing_solution"}

    started = time.monotonic()
    board = board_from_api_puzzle(item)
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
        searches.append(
            {
                "ply": idx,
                "expected": expected,
                "actual": actual or answer.bestmove,
                "nodes": answer.nodes,
                "nps": answer.nps,
                "depth": answer.depth,
            }
        )
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
    if stats.get("ended"):
        lines.append(f"- Ended: {stats.get('ended')}")
    if stats.get("rate_limit_events"):
        lines.append(f"- Rate-limit events: {stats.get('rate_limit_events')}")
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
    print(f"Rate limited; waiting {wait_s:.0f}s before retrying")
    time.sleep(wait_s)
    return True


def run(args) -> int:
    if requests is None:
        raise RuntimeError("Python package 'requests' is required")
    token = load_token()
    if not args.engine.exists():
        raise RuntimeError(f"Engine not found at {args.engine}")
    if args.mode in {"mcts", "hybrid"} and not args.weights.exists():
        raise RuntimeError(f"Transformer weights not found at {args.weights}")

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
            print(f"Rate limited before first batch; retry after {exc.wait_s:.0f}s")
            return 0

    engine = UCIEngine(args.engine, options)
    solved = 0
    total = 0
    started = time.monotonic()
    ended = "completed"

    print(
        f"Puzzle run: mode={args.mode}, threads={threads}, hash={hash_mb} MB, "
        f"movetime={args.movetime_ms} ms, batch={args.batch_size}, rated={args.rated}"
    )
    print(
        f"Resources: logical={os.cpu_count() or 1}, available_memory={available_memory_mb()} MB, "
        f"thread_reserve={os.environ.get('METALFISH_PUZZLE_THREAD_RESERVE', '1')}, "
        f"memory_reserve={os.environ.get('METALFISH_PUZZLE_MEMORY_RESERVE_MB', '1536')} MB"
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
                    batch_results.append(result)
                    out.write(json.dumps(result, sort_keys=True) + "\n")
                    out.flush()
                    if total % args.progress_interval == 0:
                        print(
                            f"Progress: {solved}/{total} "
                            f"({solved / max(1, total):.1%})"
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
                            f"retry after {exc.wait_s:.0f}s"
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
    write_summary(summary_path, stats)
    print(
        f"Finished: solved {solved}/{total} "
        f"({solved / max(1, total):.2%}) in {duration_s:.1f}s"
    )
    print(f"Results: {jsonl_path}")
    print(f"Summary: {summary_path}")

    accuracy = solved / max(1, total)
    if total == 0:
        return 2
    if accuracy < args.min_accuracy:
        return 1
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=pathlib.Path, default=ENGINE)
    parser.add_argument("--mode", choices=("ab", "mcts", "hybrid"), default="ab")
    parser.add_argument(
        "--weights",
        type=pathlib.Path,
        default=ROOT / "networks" / "BT4-1024x15x32h-swa-6147500.pb",
    )
    parser.add_argument("--syzygy-path", type=pathlib.Path, default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--hash-mb", type=int, default=0)
    parser.add_argument("--movetime-ms", type=int, default=3000)
    parser.add_argument("--max-minutes", type=float, default=335.0)
    parser.add_argument("--max-puzzles", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--angle", default="mix")
    parser.add_argument("--rated", action="store_true", default=False)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--request-interval-s", type=float, default=2.0)
    parser.add_argument("--rate-limit-backoff-s", type=float, default=65.0)
    parser.add_argument("--max-rate-limit-waits", type=int, default=5)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    args.batch_size = max(1, min(50, args.batch_size))
    args.movetime_ms = max(1, args.movetime_ms)
    args.max_minutes = max(0.1, args.max_minutes)
    args.max_puzzles = max(1, args.max_puzzles)
    args.progress_interval = max(1, args.progress_interval)
    args.max_rate_limit_waits = max(0, args.max_rate_limit_waits)
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
