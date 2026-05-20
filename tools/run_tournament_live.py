#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    import chess
    import chess.pgn
except ImportError:
    print("ERROR: python-chess required. Install: pip install python-chess")
    sys.exit(1)

PROJ = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJ / "tools" / "engines_config.json"
RESULTS_BASE = PROJ / "results"

BUILTIN_OPENING_LINES = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "d2d4 g8f6 c2c4 e7e6 g1f3 d7d5 g2g3",
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5",
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5 c4d5 f6d5",
    "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6 e1g1 f8e7",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7",
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",
]


def detect_default_threads() -> int:
    """Best-effort thread budget for local tournaments."""
    env = os.getenv("METALFISH_THREADS")
    if env:
        try:
            n = int(env)
            if n > 0:
                return n
        except ValueError:
            pass

    if sys.platform == "darwin":
        # Prefer performance cores on Apple Silicon for better chess throughput.
        for key in ("hw.perflevel0.physicalcpu_max", "hw.logicalcpu"):
            try:
                out = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
                n = int(out)
                if n > 0:
                    return n
            except Exception:
                continue

    n = os.cpu_count() or 8
    return max(1, n)


def env_option(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def apply_hybrid_env_options(options: Dict[str, str], force_trace: bool) -> None:
    overrides = {
        "HYBRID_MCTS_THREADS": "HybridMCTSThreads",
        "HYBRID_AB_THREADS": "HybridABThreads",
        "HYBRID_AUTO_AB_THREADS_CAP": "HybridAutoABThreadsCap",
        "HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS": "TransformerLowTimeFallbackMs",
        "HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS": "TransformerMinMoveBudgetMs",
        "HYBRID_MCTS_KLD": "HybridMCTSMinimumKLDGainPerNode",
        "HYBRID_MCTS_ROOT_REJECT": "HybridMCTSRootReject",
        "HYBRID_MCTS_SHARED_TT": "HybridMCTSUseSharedTT",
        "HYBRID_MCTS_AB_ROOT_HINTS": "HybridMCTSABRootHints",
        "HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS": "HybridMCTSABRootHintDelayMs",
        "HYBRID_MCTS_AB_ROOT_HINT_COUNT": "HybridMCTSABRootHintCount",
        "HYBRID_AB_CANDIDATE_VERIFY_MS": "HybridABCandidateVerifyMs",
        "HYBRID_AB_CANDIDATE_VERIFY_COUNT": "HybridABCandidateVerifyCount",
        "HYBRID_AB_POLICY_WEIGHT": "HybridABPolicyWeight",
        "HYBRID_ROOT_PAWN_LEVER_TIEBREAK": "HybridRootPawnLeverTieBreak",
        "HYBRID_TRACE": "HybridTrace",
        "HYBRID_MCTS_MINIBATCH": "MCTSMinibatchSize",
        "HYBRID_MCTS_OUT_OF_ORDER_FACTOR": "MCTSMaxOutOfOrderEvalsFactor",
        "HYBRID_MCTS_MAX_PREFETCH": "MCTSMaxPrefetch",
    }
    for env_name, option_name in overrides.items():
        value = env_option(env_name)
        if value is not None:
            options[option_name] = value
    if "HybridMCTSThreads" in options:
        options["MCTSMaxThreads"] = options["HybridMCTSThreads"]
    if force_trace and "HYBRID_TRACE" not in os.environ:
        options["HybridTrace"] = "true"


def parse_int_option(value: Optional[str], default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def hybrid_low_time_warnings(
    matches: Sequence[Tuple[str, str]],
    engines_cfg: Dict[str, dict],
    movetime_ms: int,
) -> List[str]:
    if movetime_ms <= 0:
        return []

    warnings: List[str] = []
    hybrid_names = {
        name
        for match in matches
        for name in match
        if engines_cfg.get(name, {}).get("options", {}).get("UseHybridSearch") == "true"
    }
    for name in sorted(hybrid_names):
        options = dict(engines_cfg[name].get("options", {}))
        apply_hybrid_env_options(options, force_trace=False)
        fallback_ms = parse_int_option(options.get("TransformerLowTimeFallbackMs"), 0)
        if fallback_ms > 0 and movetime_ms < fallback_ms:
            warnings.append(
                f"{name}: --movetime {movetime_ms}ms is below "
                f"TransformerLowTimeFallbackMs={fallback_ms}ms; this run will "
                "exercise the AB time-safety fallback rather than full Hybrid MCTS."
            )
    return warnings


class UCIEngine:
    def __init__(self, cmd: Sequence[str], name: str, options: Dict[str, str] = None):
        self.name = name
        self.cmd = list(cmd)
        self.options = dict(options or {})
        self.proc: Optional[subprocess.Popen[str]] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self.last_output: List[str] = []
        self.last_error: List[str] = []
        self.last_search_output: List[str] = []
        self._start()

    def _start(self):
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._lines = queue.Queue()
            assert self.proc.stdout is not None
            threading.Thread(
                target=self._read_stdout,
                args=(self.proc.stdout, self._lines),
                daemon=True,
            ).start()
            self.last_error = []
            assert self.proc.stderr is not None
            threading.Thread(
                target=self._read_stderr,
                args=(self.proc.stderr, self.last_error),
                daemon=True,
            ).start()
            self.last_output = []
            self._send("uci")
            self._wait_for("uciok", 30)
            for k, v in self.options.items():
                self._send(f"setoption name {k} value {v}")
            self._send("isready")
            self._wait_for("readyok", 60)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _read_stdout(stream, lines: "queue.Queue[Optional[str]]"):
        try:
            for line in stream:
                lines.put(line.rstrip("\r\n"))
        finally:
            lines.put(None)

    @staticmethod
    def _read_stderr(stream, errors: List[str]):
        for line in stream:
            errors.append(line.rstrip("\r\n"))
            if len(errors) > 20:
                errors.pop(0)

    def _send(self, cmd: str):
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError(f"{self.name}: process is not started")
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"{self.name}: process exited with code {self.proc.returncode}"
            )
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"{self.name}: failed to send '{cmd}': {exc}") from exc

    def _remember_output(self, line: str):
        self.last_output.append(line)
        if len(self.last_output) > 20:
            self.last_output.pop(0)

    def _output_tail(self) -> str:
        parts = []
        if self.last_output:
            parts.append("last output: " + " | ".join(self.last_output[-3:]))
        if self.last_error:
            parts.append("last stderr: " + " | ".join(self.last_error[-3:]))
        return "; " + "; ".join(parts) if parts else ""

    def _wait_for(
        self, prefix: str, timeout: int, collect: Optional[List[str]] = None
    ) -> str:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError(f"{self.name}: process is not started")
        deadline = time.monotonic() + timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name}: process died with code {self.proc.returncode} "
                    f"while waiting for '{prefix}'{self._output_tail()}"
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{self.name}: timeout waiting for '{prefix}'"
                    f"{self._output_tail()}"
                )

            try:
                line = self._lines.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                continue
            if line is None:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{self.name}: process died with code {self.proc.returncode} "
                        f"while waiting for '{prefix}'{self._output_tail()}"
                    )
                continue
            self._remember_output(line)
            if collect is not None:
                collect.append(line)
            if line.startswith(prefix):
                return line

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def restart(self):
        self.close()
        self._start()

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", 30)

    def go(
        self,
        fen: str,
        moves: List[str],
        wtime: int,
        btime: int,
        winc: int = 0,
        binc: int = 0,
        movetime: int = 0,
        nodes: int = 0,
    ) -> str:
        pos_cmd = f"position fen {fen}"
        if moves:
            pos_cmd += " moves " + " ".join(moves)
        self._send(pos_cmd)

        if nodes > 0:
            self._send(f"go nodes {nodes}")
        elif movetime > 0:
            self._send(f"go movetime {movetime}")
        else:
            self._send(f"go wtime {wtime} btime {btime} winc {winc} binc {binc}")

        if nodes > 0:
            timeout = max(30, nodes // 1000 + 30)
        elif movetime > 0:
            timeout = movetime // 1000 + 30
        else:
            timeout = max(wtime, btime) // 1000 + 30
        search_output: List[str] = []
        line = self._wait_for("bestmove", timeout, search_output)
        parts = line.split()
        bestmove = parts[1] if len(parts) > 1 else "0000"
        self._send("isready")
        self._wait_for("readyok", 30, search_output)
        self.last_search_output = search_output
        return bestmove

    def close(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self._send("quit")
                self.proc.wait(timeout=5)
        except Exception:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
        finally:
            for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass
            self.proc = None


def builtin_openings(max_openings: int, seed: int, order: str) -> List[chess.Board]:
    openings: List[chess.Board] = []
    for line in BUILTIN_OPENING_LINES[:max_openings]:
        board = chess.Board()
        try:
            for token in line.split():
                move = chess.Move.from_uci(token)
                if move not in board.legal_moves:
                    raise ValueError(token)
                board.push(move)
        except ValueError:
            continue
        openings.append(board.copy())
    if order == "random":
        random.Random(seed).shuffle(openings)
    return openings


def load_openings(
    book_path: pathlib.Path,
    max_openings: int = 500,
    seed: int = 6147500,
    order: str = "random",
) -> Tuple[List[chess.Board], str]:
    openings = []
    if not book_path.exists():
        return builtin_openings(max_openings, seed, order), "built-in fallback"
    with open(book_path) as f:
        while len(openings) < max_openings:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            openings.append(board.copy())
    if order == "random":
        random.Random(seed).shuffle(openings)
    if openings:
        return openings, str(book_path.relative_to(PROJ))
    return builtin_openings(max_openings, seed, order), "built-in fallback"


@dataclass
class GameResult:
    white: str
    black: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    reason: str
    moves: int
    pgn: str = ""
    search_log: List[dict] = field(default_factory=list)
    engine_error: bool = False


def extract_search_log_lines(lines: Sequence[str]) -> List[str]:
    """Keep compact per-move diagnostics that are useful for hybrid debugging."""
    keep = []
    for line in lines:
        if line.startswith("info string HybridTrace:"):
            keep.append(line)
        elif line.startswith("info string Final:"):
            keep.append(line)
        elif line.startswith("info string Hybrid: AB root hints from MCTS"):
            keep.append(line)
        elif line.startswith("info string Time safety:"):
            keep.append(line)
        elif line.startswith("info string Starting Parallel Hybrid Search"):
            keep.append(line)
        elif line.startswith("info string Hybrid thread split:"):
            keep.append(line)
    return keep


def clock_label(ms: int) -> str:
    return f"{max(0, ms) / 1000.0:.1f}s"


def game_over_reason(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.can_claim_fifty_moves():
        return "50-move rule" if board.is_fifty_moves() else "claimable 50-move rule"
    if board.can_claim_threefold_repetition():
        return (
            "3-fold repetition"
            if board.is_repetition(3)
            else "claimable 3-fold repetition"
        )
    if board.is_insufficient_material():
        return "insufficient material"
    return "adjudication"


def play_game(
    white: UCIEngine,
    black: UCIEngine,
    opening: Optional[chess.Board] = None,
    tc_base_ms: int = 60000,
    tc_inc_ms: int = 100,
    movetime_ms: int = 0,
    nodes: int = 0,
    max_moves: int = 300,
    max_plies: int = 0,
    capture_search_log: bool = False,
    progress_label: str = "",
    progress_plies: int = 10,
) -> GameResult:
    if opening:
        board = opening.copy()
        start_fen = opening.fen()
        move_list = []
    else:
        board = chess.Board()
        start_fen = chess.STARTING_FEN
        move_list = []

    wtime = tc_base_ms
    btime = tc_base_ms
    resign_count = [0, 0]  # [white_resign, black_resign]
    resign_threshold = 1000  # centipawns
    resign_moves_needed = 3

    game_pgn = chess.pgn.Game()
    game_pgn.headers["White"] = white.name
    game_pgn.headers["Black"] = black.name
    game_pgn.headers["Date"] = datetime.date.today().isoformat()
    if opening:
        game_pgn.setup(opening.fen())
    node = game_pgn
    search_log: List[dict] = []
    ply_limit = max_plies if max_plies > 0 else max_moves * 2

    for ply in range(ply_limit):
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            reason = game_over_reason(board)
            return GameResult(
                white.name,
                black.name,
                result,
                reason,
                ply // 2,
                str(game_pgn),
                search_log,
            )

        eng = white if board.turn == chess.WHITE else black
        fen_before = board.fen()
        t0 = time.time()

        try:
            move_str = eng.go(
                start_fen,
                move_list,
                wtime,
                btime,
                tc_inc_ms,
                tc_inc_ms,
                movetime_ms,
                nodes,
            )
        except (TimeoutError, RuntimeError) as exc:
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            reason = f"{eng.name} {type(exc).__name__}: {exc}"
            return GameResult(
                white.name,
                black.name,
                result,
                reason,
                ply // 2,
                str(game_pgn),
                search_log,
                True,
            )

        elapsed_ms = int((time.time() - t0) * 1000)
        if not movetime_ms and nodes <= 0:
            if board.turn == chess.WHITE:
                wtime = max(100, wtime - elapsed_ms + tc_inc_ms)
            else:
                btime = max(100, btime - elapsed_ms + tc_inc_ms)

        try:
            move = chess.Move.from_uci(move_str)
            if move not in board.legal_moves:
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                return GameResult(
                    white.name,
                    black.name,
                    result,
                    "illegal move",
                    ply // 2,
                    str(game_pgn),
                    search_log,
                )
        except Exception:
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            return GameResult(
                white.name,
                black.name,
                result,
                "invalid move string",
                ply // 2,
                str(game_pgn),
                search_log,
            )

        if capture_search_log:
            lines = extract_search_log_lines(eng.last_search_output)
            if lines:
                search_log.append(
                    {
                        "ply": ply + 1,
                        "engine": eng.name,
                        "side": "white" if board.turn == chess.WHITE else "black",
                        "fen": fen_before,
                        "move": move_str,
                        "elapsed_ms": elapsed_ms,
                        "lines": lines,
                    }
                )

        move_list.append(move_str)
        node = node.add_variation(move)
        board.push(move)

        completed_plies = ply + 1
        if progress_plies > 0 and completed_plies % progress_plies == 0:
            label = f"{progress_label} " if progress_label else ""
            print(
                f"    {label}ply {completed_plies:3d}/{ply_limit}: "
                f"{eng.name} {move_str} "
                f"(W {clock_label(wtime)}, B {clock_label(btime)})",
                flush=True,
            )

    completed_plies = len(move_list)
    return GameResult(
        white.name,
        black.name,
        "1/2-1/2",
        "max plies" if max_plies > 0 else "max moves",
        (completed_plies + 1) // 2,
        str(game_pgn),
        search_log,
    )


@dataclass
class MatchResult:
    engine1: str
    engine2: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    games: List[dict] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.wins + self.draws * 0.5

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def pct(self) -> float:
        return self.score / max(1, self.total)

    @property
    def elo_diff(self) -> float:
        p = self.pct
        if p <= 0.0 or p >= 1.0:
            return 400.0 if p >= 1.0 else -400.0
        return -400.0 * math.log10(1.0 / p - 1.0)


def match_result_to_dict(mr: MatchResult) -> dict:
    return {
        "engine1": mr.engine1,
        "engine2": mr.engine2,
        "wins": mr.wins,
        "draws": mr.draws,
        "losses": mr.losses,
        "score": mr.score,
        "total": mr.total,
        "pct": round(mr.pct, 3),
        "elo_diff": round(mr.elo_diff, 1),
        "games": mr.games,
    }


def tournament_tc_label(args: argparse.Namespace, nodes: int, movetime_ms: int) -> str:
    if nodes:
        return f"{nodes} nodes/move"
    if movetime_ms:
        return f"{movetime_ms}ms/move"
    return f"{args.tc_base}+{args.tc_inc}"


def write_results_json(
    results_dir: pathlib.Path,
    timestamp: str,
    args: argparse.Namespace,
    nodes: int,
    movetime_ms: int,
    matches: List[dict],
    opening_source: str,
) -> None:
    payload = {
        "matches": matches,
        "timestamp": timestamp,
        "tc": tournament_tc_label(args, nodes, movetime_ms),
        "games_per_match": args.games,
        "opening_source": opening_source,
        "opening_order": args.opening_order,
        "seed": args.seed,
        "max_plies": args.max_plies,
    }
    tmp_path = results_dir / "results.json.tmp"
    final_path = results_dir / "results.json"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(final_path)


def run_match(
    eng1_name: str,
    eng2_name: str,
    eng1: UCIEngine,
    eng2: UCIEngine,
    num_games: int,
    openings: List[chess.Board],
    tc_base_ms: int = 60000,
    tc_inc_ms: int = 100,
    movetime_ms: int = 0,
    nodes: int = 0,
    max_moves: int = 300,
    max_plies: int = 0,
    capture_search_log: bool = False,
    progress_callback: Optional[Callable[[MatchResult], None]] = None,
    progress_plies: int = 10,
) -> MatchResult:
    result = MatchResult(eng1_name, eng2_name)
    print(f"\n{'='*60}", flush=True)
    print(f"  {eng1_name} vs {eng2_name}  ({num_games} games)", flush=True)
    print(f"{'='*60}", flush=True)

    for g in range(num_games):
        opening = openings[g % len(openings)] if openings else None

        if g % 2 == 0:
            w, b = eng1, eng2
            w_name, b_name = eng1_name, eng2_name
        else:
            w, b = eng2, eng1
            w_name, b_name = eng2_name, eng1_name

        if not w.is_running():
            w.restart()
        if not b.is_running():
            b.restart()

        w.new_game()
        b.new_game()

        gr = play_game(
            w,
            b,
            opening,
            tc_base_ms,
            tc_inc_ms,
            movetime_ms,
            nodes,
            max_moves,
            max_plies,
            capture_search_log=capture_search_log,
            progress_label=f"Game {g + 1}/{num_games}",
            progress_plies=progress_plies,
        )

        if g % 2 == 0:
            if gr.result == "1-0":
                result.wins += 1
            elif gr.result == "0-1":
                result.losses += 1
            else:
                result.draws += 1
        else:
            if gr.result == "1-0":
                result.losses += 1
            elif gr.result == "0-1":
                result.wins += 1
            else:
                result.draws += 1

        result.games.append(
            {
                "game": g + 1,
                "white": gr.white,
                "black": gr.black,
                "result": gr.result,
                "reason": gr.reason,
                "moves": gr.moves,
                "pgn": gr.pgn,
                "engine_error": gr.engine_error,
            }
        )
        if gr.search_log:
            result.games[-1]["search_log"] = gr.search_log

        marker = (
            "+"
            if (g % 2 == 0 and gr.result == "1-0")
            or (g % 2 == 1 and gr.result == "0-1")
            else (
                "-"
                if (g % 2 == 0 and gr.result == "0-1")
                or (g % 2 == 1 and gr.result == "1-0")
                else "="
            )
        )
        score_str = f"W{result.wins}-D{result.draws}-L{result.losses}"
        print(
            f"  Game {g+1:2d}/{num_games}: {marker} {gr.result:7s} "
            f"({gr.reason}, {gr.moves} moves) [{score_str}]",
            flush=True,
        )

        if progress_callback:
            progress_callback(result)

        if gr.engine_error:
            print("    Restarting engines after UCI failure", flush=True)
            w.restart()
            b.restart()

    elo = result.elo_diff
    print(
        f"\n  Result: {eng1_name} {result.wins}W-{result.draws}D-{result.losses}L "
        f"({result.score}/{result.total}) Elo diff: {elo:+.0f}",
        flush=True,
    )
    return result


def create_engine(
    name: str,
    cfg: dict,
    default_threads: int,
    force_hybrid_trace: bool = False,
) -> Optional[UCIEngine]:
    path = PROJ / cfg["path"]
    if not path.exists():
        print(f"  SKIP {name}: binary not found at {path}")
        return None
    cmd = [str(path)] + cfg.get("cmd_args", [])
    options = dict(cfg.get("options", {}))
    if name == "MetalFish-Hybrid":
        apply_hybrid_env_options(options, force_hybrid_trace)
    for thread_option in ("Threads", "MCTSMaxThreads"):
        if thread_option in options:
            t = str(options[thread_option]).strip().lower()
            if t in {"auto", "max", "native"}:
                options[thread_option] = str(default_threads)
    for path_option in ("NNWeights", "SyzygyPath"):
        if path_option in options:
            path_value = pathlib.Path(str(options[path_option]))
            if str(path_value) and not path_value.is_absolute():
                options[path_option] = str(PROJ / path_value)
    try:
        eng = UCIEngine(cmd, name, options)
        return eng
    except Exception as e:
        print(f"  SKIP {name}: failed to start: {e}")
        return None


def run_tournament(args):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    engines_cfg = config["engines"]
    book_cfg = config.get("opening_book", {})
    default_threads = detect_default_threads()

    book_path = PROJ / book_cfg.get("file", "")
    openings, opening_source = load_openings(
        book_path,
        max_openings=200,
        seed=args.seed,
        order=args.opening_order,
    )
    if openings:
        print(f"Loaded {len(openings)} openings from {opening_source}")
    else:
        print("No opening book found, using startpos")

    tc_base_ms = args.tc_base * 1000
    tc_inc_ms = int(args.tc_inc * 1000)
    movetime_ms = args.movetime if args.movetime > 0 else 0
    nodes = args.nodes if args.nodes > 0 else 0

    if args.match:
        matches = [(args.match[0], args.match[1])]
    else:
        matches = [
            # MetalFish engines vs each other
            ("MetalFish-AB", "MetalFish-MCTS"),
            ("MetalFish-AB", "MetalFish-Hybrid"),
            ("MetalFish-MCTS", "MetalFish-Hybrid"),
            # MetalFish-AB vs reference engines
            ("MetalFish-AB", "Stockfish"),
            ("MetalFish-AB", "Berserk"),
            ("MetalFish-AB", "Patricia"),
            # MetalFish-MCTS vs NN baseline
            ("MetalFish-MCTS", "Lc0"),
            ("MetalFish-MCTS", "Patricia"),
            # MetalFish-Hybrid vs reference engines
            ("MetalFish-Hybrid", "Stockfish-L15"),
            ("MetalFish-Hybrid", "Berserk"),
            ("MetalFish-Hybrid", "Patricia"),
            ("MetalFish-Hybrid", "Lc0"),
        ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_BASE / f"tournament_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMetalFish Tournament")
    print(f"{'='*60}")
    print(f"Games per match: {args.games}")
    if nodes:
        print(f"Search limit: {nodes} nodes/move")
    elif movetime_ms:
        print(f"Time control: {movetime_ms}ms/move")
    else:
        print(f"Time control: {args.tc_base}s + {args.tc_inc}s/move")
    print(f"Results: {results_dir}")
    print(f"Matches: {len(matches)}")
    print(f"Default thread budget: {default_threads}")
    print(f"Openings: order={args.opening_order} | seed={args.seed}")
    for warning in hybrid_low_time_warnings(matches, engines_cfg, movetime_ms):
        print(f"Warning: {warning}", flush=True)
    if args.max_plies > 0:
        print(f"Ply cap: {args.max_plies}")
    elif args.max_moves < 100:
        print(
            "Warning: max-moves below 100 can mask long conversion wins; "
            "use it only for quick lifecycle smoke tests."
        )
    print()

    all_results: List[dict] = []
    active_engines: Dict[str, UCIEngine] = {}

    try:
        for e1_name, e2_name in matches:
            if e1_name not in engines_cfg or e2_name not in engines_cfg:
                print(f"\n  SKIP {e1_name} vs {e2_name}: engine not in config")
                continue

            for ename in [e1_name, e2_name]:
                if ename not in active_engines:
                    eng = create_engine(
                        ename,
                        engines_cfg[ename],
                        default_threads,
                        args.save_search_log,
                    )
                    if eng is None:
                        break
                    active_engines[ename] = eng

            if e1_name not in active_engines or e2_name not in active_engines:
                print(f"\n  SKIP {e1_name} vs {e2_name}: engine unavailable")
                continue

            def checkpoint_match(partial: MatchResult) -> None:
                write_results_json(
                    results_dir,
                    timestamp,
                    args,
                    nodes,
                    movetime_ms,
                    [*all_results, match_result_to_dict(partial)],
                    opening_source,
                )

            mr = run_match(
                e1_name,
                e2_name,
                active_engines[e1_name],
                active_engines[e2_name],
                args.games,
                openings,
                tc_base_ms,
                tc_inc_ms,
                movetime_ms,
                nodes,
                args.max_moves,
                args.max_plies,
                args.save_search_log,
                checkpoint_match,
                args.progress_plies,
            )

            match_data = match_result_to_dict(mr)
            all_results.append(match_data)

            write_results_json(
                results_dir,
                timestamp,
                args,
                nodes,
                movetime_ms,
                all_results,
                opening_source,
            )

    finally:
        for eng in active_engines.values():
            eng.close()

    print(f"\n{'='*60}")
    print(f"  TOURNAMENT SUMMARY")
    print(f"{'='*60}")
    print(
        f"\n{'Engine 1':<20s} {'Engine 2':<20s} {'W':>3s} {'D':>3s} {'L':>3s} "
        f"{'Score':>7s} {'Pct':>6s} {'Elo':>6s}"
    )
    print("-" * 75)

    for m in all_results:
        print(
            f"{m['engine1']:<20s} {m['engine2']:<20s} "
            f"{m['wins']:3d} {m['draws']:3d} {m['losses']:3d} "
            f"{m['score']:5.1f}/{m['total']:<2d} "
            f"{m['pct']*100:5.1f}% {m['elo_diff']:+6.0f}"
        )

    print(f"\n  Estimated Elo Ratings (relative to anchors):")
    anchors = {
        n: c["expected_elo"]
        for n, c in engines_cfg.items()
        if c.get("anchor") and c.get("expected_elo")
    }
    engine_elos: Dict[str, List[float]] = {}

    for m in all_results:
        e1, e2 = m["engine1"], m["engine2"]
        diff = m["elo_diff"]
        if e2 in anchors:
            engine_elos.setdefault(e1, []).append(anchors[e2] + diff)
        if e1 in anchors:
            engine_elos.setdefault(e2, []).append(anchors[e1] - diff)

    for name, elos in sorted(engine_elos.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        avg = sum(elos) / len(elos)
        print(f"    {name:<25s} ~{avg:.0f} Elo (from {len(elos)} matchups)")

    print(f"\nResults saved: {results_dir / 'results.json'}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="MetalFish Tournament Runner")
    parser.add_argument(
        "--games", type=int, default=6, help="Games per match (default: 6)"
    )
    parser.add_argument(
        "--tc-base", type=float, default=300, help="Base time in seconds (default: 300)"
    )
    parser.add_argument(
        "--tc-inc", type=float, default=0.1, help="Increment in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--movetime",
        type=int,
        default=0,
        help="Fixed movetime in ms (overrides tc, 0=disabled)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=0,
        help="Fixed node limit per move (overrides tc/movetime, 0=disabled)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=300,
        help="Adjudicate as a draw after this many full moves",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=0,
        help="Adjudicate as a draw after this many searched plies (0=disabled)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 4 games, 10s+0.1s"
    )
    parser.add_argument(
        "--match",
        nargs=2,
        metavar=("E1", "E2"),
        help="Run single match between two engines",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6147500,
        help="Opening shuffle seed (default: 6147500)",
    )
    parser.add_argument(
        "--opening-order",
        choices=("random", "sequential"),
        default="random",
        help="Opening order from book (default: random)",
    )
    parser.add_argument(
        "--save-search-log",
        action="store_true",
        help="Save compact per-move info string diagnostics in results JSON",
    )
    parser.add_argument(
        "--progress-plies",
        type=int,
        default=10,
        help="Print in-game progress every N plies (0 disables, default: 10)",
    )
    parser.add_argument("--resume", type=str, help="Resume from results directory")
    args = parser.parse_args()

    if args.quick:
        args.games = 4
        args.tc_base = 10
        args.tc_inc = 0.1

    return run_tournament(args)


if __name__ == "__main__":
    sys.exit(main())
