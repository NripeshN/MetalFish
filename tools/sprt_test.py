#!/usr/bin/env python3
"""
MetalFish SPRT (Sequential Probability Ratio Test) Engine Testing

Runs games between a baseline and candidate engine configuration,
applying SPRT to determine if the candidate is stronger/weaker/equivalent.

Usage:
    # Test a single option change against baseline
    python3 tools/sprt_test.py --option "MCTSContempt=700" --elo0 0 --elo1 10

    # Test candidate binary against baseline binary
    python3 tools/sprt_test.py --candidate-engine ./build_new/metalfish --elo0 0 --elo1 5

    # Batch test multiple options from a tuning file
    python3 tools/sprt_test.py --batch tools/sprt_tuning_batch.json

    # Quick self-play sanity check (same engine both sides)
    python3 tools/sprt_test.py --self-play --games 50
"""

from __future__ import annotations

import argparse
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
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import chess
    import chess.pgn
except ImportError:
    print("ERROR: python-chess required.  pip install python-chess")
    sys.exit(1)

PROJ = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJ / "results" / "sprt"
OPENINGS_BOOK = PROJ / "reference" / "books" / "8moves_v3.pgn"

BUILTIN_OPENINGS = [
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
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4",
    "g1f3 d7d5 c2c4 d5c4 e2e3 g8f6 f1c4 e7e6",
    "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7 g1f3",
    "e2e4 e7e5 f1c4 g8f6 d2d3 b8c6 g1f3",
]


# ---------------------------------------------------------------------------
# SPRT math
# ---------------------------------------------------------------------------


def elo_to_prob(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def llr_logistic(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    n = wins + draws + losses
    if n == 0:
        return 0.0
    w = wins / n
    d = draws / n
    l_val = losses / n
    s = w + d / 2.0
    if s <= 0.0 or s >= 1.0:
        return 0.0
    s0 = elo_to_prob(elo0)
    s1 = elo_to_prob(elo1)
    if s0 <= 0 or s0 >= 1 or s1 <= 0 or s1 >= 1:
        return 0.0
    return (
        0.5
        * n
        * (
            (s1 - s0) * (2 * s - 1)
            - (s1 * s1 - s0 * s0)
            + (s1 * (1 - s1) - s0 * (1 - s0)) * 0
        )
    )


def llr_trinomial(
    wins: int, draws: int, losses: int, elo0: float, elo1: float
) -> float:
    """Compute LLR using the BayesElo trinomial model (standard for engine testing)."""
    n = wins + draws + losses
    if n < 4:
        return 0.0

    w = (wins + 0.5) / (n + 1.5)
    d = (draws + 0.5) / (n + 1.5)
    lo = (losses + 0.5) / (n + 1.5)

    s = w + d / 2.0
    var = w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + lo * (0 - s) ** 2
    if var <= 0:
        return 0.0

    s0 = elo_to_prob(elo0)
    s1 = elo_to_prob(elo1)

    llr = n * ((s1 - s0) * (2 * s - s0 - s1) / (2 * var))
    return llr


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> Tuple[float, float]:
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)
    return lower, upper


def elo_estimate(wins: int, draws: int, losses: int) -> Tuple[float, float, float]:
    """Returns (elo, elo_lower, elo_upper) at 95% CI."""
    n = wins + draws + losses
    if n == 0:
        return 0.0, 0.0, 0.0
    score = (wins + draws / 2.0) / n
    if score <= 0 or score >= 1:
        elo = 400.0 * math.copysign(1, score - 0.5) * 10
        return elo, elo, elo
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    var = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2) / n
    if var <= 0:
        return elo, elo, elo
    se = math.sqrt(var / n)
    elo_lo = -400.0 * math.log10(max(0.001, 1.0 / max(0.001, score - 1.96 * se) - 1.0))
    elo_hi = -400.0 * math.log10(
        max(0.001, 1.0 / max(0.001, min(0.999, score + 1.96 * se)) - 1.0)
    )
    return elo, elo_lo, elo_hi


@dataclass
class SPRTResult:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    llr: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    elo0: float = 0.0
    elo1: float = 0.0
    status: str = "ongoing"  # "H1" (accept), "H0" (reject), "ongoing", "max_games"
    elo_est: float = 0.0
    elo_ci_lo: float = 0.0
    elo_ci_hi: float = 0.0
    games_played: int = 0
    elapsed_sec: float = 0.0
    candidate_label: str = ""
    options_tested: Dict[str, str] = field(default_factory=dict)

    @property
    def total(self):
        return self.wins + self.draws + self.losses

    def passed(self) -> bool:
        return self.status == "H1"

    def failed(self) -> bool:
        return self.status == "H0"

    def as_dict(self) -> dict:
        return {
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "total_games": self.total,
            "llr": round(self.llr, 4),
            "bounds": [round(self.lower_bound, 4), round(self.upper_bound, 4)],
            "elo0": self.elo0,
            "elo1": self.elo1,
            "status": self.status,
            "elo_estimate": round(self.elo_est, 1),
            "elo_95ci": [round(self.elo_ci_lo, 1), round(self.elo_ci_hi, 1)],
            "elapsed_sec": round(self.elapsed_sec, 1),
            "candidate_label": self.candidate_label,
            "options_tested": self.options_tested,
        }


# ---------------------------------------------------------------------------
# UCI Engine wrapper
# ---------------------------------------------------------------------------


class UCIEngine:
    def __init__(
        self, cmd: str, name: str, options: Dict[str, str], cwd: Optional[str] = None
    ):
        self.name = name
        self.cmd = cmd
        self.options = dict(options)
        self.cwd = cwd
        self.proc: Optional[subprocess.Popen] = None
        self._lines: queue.Queue = queue.Queue()
        self.last_score_cp: Optional[int] = None
        self._start()

    def _start(self):
        self.proc = subprocess.Popen(
            [self.cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.cwd,
        )
        threading.Thread(target=self._reader, daemon=True).start()
        self._send("uci")
        self._wait_for("uciok", 30)
        for k, v in self.options.items():
            self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait_for("readyok", 60)

    def _reader(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._lines.put(line.strip())
        self._lines.put(None)

    def _send(self, cmd: str):
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout: float) -> List[str]:
        lines = []
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"{self.name}: timeout waiting for '{token}'")
            try:
                line = self._lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if self.proc and self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{self.name}: process died (rc={self.proc.returncode})"
                    )
                continue
            if line is None:
                raise RuntimeError(f"{self.name}: EOF before '{token}'")
            lines.append(line)
            if token in line:
                return lines

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", 30)

    def go(
        self, position: str, movetime_ms: int = 0, tc: Optional[Tuple[int, int]] = None
    ) -> str:
        self._send(f"position {position}")
        if tc:
            wtime, btime = tc
            inc = tc[1] if len(tc) > 2 else 0
            self._send(f"go wtime {wtime} btime {btime} winc {inc} binc {inc}")
        else:
            self._send(f"go movetime {movetime_ms}")
        lines = self._wait_for("bestmove", 300)
        self.last_score_cp = parse_last_score_cp(lines)
        for line in reversed(lines):
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else "0000"
        return "0000"

    def go_tc(
        self, position: str, wtime: int, btime: int, winc: int = 0, binc: int = 0
    ) -> str:
        self._send(f"position {position}")
        self._send(f"go wtime {wtime} btime {btime} winc {winc} binc {binc}")
        lines = self._wait_for("bestmove", 600)
        self.last_score_cp = parse_last_score_cp(lines)
        for line in reversed(lines):
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else "0000"
        return "0000"

    def close(self):
        if self.proc:
            try:
                self._send("quit")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None


# ---------------------------------------------------------------------------
# Game playing
# ---------------------------------------------------------------------------


def parse_last_score_cp(lines: Sequence[str]) -> Optional[int]:
    """Return the final UCI score as centipawns from the engine side."""
    for line in reversed(lines):
        parts = line.split()
        if not parts or parts[0] != "info":
            continue
        for i, part in enumerate(parts):
            if part != "score" or i + 2 >= len(parts):
                continue
            kind = parts[i + 1]
            value = parts[i + 2]
            try:
                score = int(value)
            except ValueError:
                continue
            if kind == "cp":
                return score
            if kind == "mate":
                if score == 0:
                    return 0
                sign = 1 if score > 0 else -1
                return sign * max(10000, 32000 - min(abs(score), 31999))
    return None


def load_openings() -> List[str]:
    if OPENINGS_BOOK.exists():
        openings = []
        try:
            with open(OPENINGS_BOOK) as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    moves = []
                    node = game
                    for _ in range(16):
                        node = node.next()
                        if node is None:
                            break
                        moves.append(node.move.uci())
                    if moves:
                        openings.append(" ".join(moves))
                    if len(openings) >= 500:
                        break
        except Exception:
            pass
        if openings:
            return openings
    return BUILTIN_OPENINGS


def play_game(
    white: UCIEngine,
    black: UCIEngine,
    opening_moves: str,
    movetime_ms: int = 0,
    tc_base_ms: int = 0,
    tc_inc_ms: int = 0,
    max_moves: int = 200,
    resign_score: int = 1000,
    resign_count: int = 3,
    draw_move: int = 40,
    draw_count: int = 8,
    draw_score: int = 10,
) -> str:
    """Play a game. Returns '1-0', '0-1', or '1/2-1/2'."""
    board = chess.Board()

    for uci_move in opening_moves.split():
        try:
            board.push(chess.Move.from_uci(uci_move))
        except (chess.InvalidMoveError, chess.IllegalMoveError):
            break

    white_resign = 0
    black_resign = 0
    draw_adj_count = 0
    wtime = tc_base_ms
    btime = tc_base_ms

    for ply in range(max_moves * 2):
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            return result

        moves_uci = " ".join(m.uci() for m in board.move_stack)
        position = f"startpos moves {moves_uci}" if moves_uci else "startpos"

        is_white = board.turn == chess.WHITE
        engine = white if is_white else black

        if tc_base_ms > 0:
            move_str = engine.go_tc(position, wtime, btime, tc_inc_ms, tc_inc_ms)
        else:
            move_str = engine.go(position, movetime_ms=movetime_ms)

        if tc_base_ms > 0:
            move_cost = movetime_ms if movetime_ms else 500
            if is_white:
                wtime = max(100, wtime - move_cost + tc_inc_ms)
            else:
                btime = max(100, btime - move_cost + tc_inc_ms)

        try:
            move = chess.Move.from_uci(move_str)
            if move not in board.legal_moves:
                return "0-1" if is_white else "1-0"
            board.push(move)
        except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
            return "0-1" if is_white else "1-0"

        if engine.last_score_cp is None:
            white_resign = 0
            black_resign = 0
            draw_adj_count = 0
            continue

        white_score = engine.last_score_cp if is_white else -engine.last_score_cp
        if white_score >= resign_score:
            black_resign += 1
            white_resign = 0
        elif white_score <= -resign_score:
            white_resign += 1
            black_resign = 0
        else:
            white_resign = 0
            black_resign = 0

        if white_resign >= resign_count:
            return "0-1"
        if black_resign >= resign_count:
            return "1-0"

        fullmove = max(1, (len(board.move_stack) + 1) // 2)
        if fullmove >= draw_move and abs(white_score) <= draw_score:
            draw_adj_count += 1
        else:
            draw_adj_count = 0
        if draw_adj_count >= draw_count:
            return "1/2-1/2"

    return "1/2-1/2"


def play_game_pair(
    engine1: UCIEngine,
    engine2: UCIEngine,
    opening: str,
    movetime_ms: int = 1000,
    tc_base_ms: int = 0,
    tc_inc_ms: int = 0,
    max_moves: int = 200,
    resign_score: int = 1000,
    resign_count: int = 3,
    draw_move: int = 40,
    draw_count: int = 8,
    draw_score: int = 10,
) -> Tuple[str, str]:
    """Play a pair of games with reversed colors. Returns (result1, result2) from engine1's perspective."""
    engine1.new_game()
    engine2.new_game()
    r1 = play_game(
        engine1,
        engine2,
        opening,
        movetime_ms,
        tc_base_ms,
        tc_inc_ms,
        max_moves=max_moves,
        resign_score=resign_score,
        resign_count=resign_count,
        draw_move=draw_move,
        draw_count=draw_count,
        draw_score=draw_score,
    )

    engine1.new_game()
    engine2.new_game()
    r2 = play_game(
        engine2,
        engine1,
        opening,
        movetime_ms,
        tc_base_ms,
        tc_inc_ms,
        max_moves=max_moves,
        resign_score=resign_score,
        resign_count=resign_count,
        draw_move=draw_move,
        draw_count=draw_count,
        draw_score=draw_score,
    )

    def from_perspective(result: str, played_white: bool) -> str:
        if result == "1-0":
            return "W" if played_white else "L"
        elif result == "0-1":
            return "L" if played_white else "W"
        return "D"

    return from_perspective(r1, True), from_perspective(r2, False)


# ---------------------------------------------------------------------------
# SPRT runner
# ---------------------------------------------------------------------------


def run_sprt(
    baseline_cmd: str,
    candidate_cmd: str,
    baseline_options: Dict[str, str],
    candidate_options: Dict[str, str],
    baseline_cwd: Optional[str] = None,
    candidate_cwd: Optional[str] = None,
    elo0: float = 0.0,
    elo1: float = 10.0,
    alpha: float = 0.05,
    beta: float = 0.05,
    max_games: int = 2000,
    movetime_ms: int = 1000,
    tc_base_ms: int = 0,
    tc_inc_ms: int = 0,
    max_moves: int = 200,
    resign_score: int = 1000,
    resign_count: int = 3,
    draw_move: int = 40,
    draw_count: int = 8,
    draw_score: int = 10,
    label: str = "",
    verbose: bool = True,
) -> SPRTResult:
    lower, upper = sprt_bounds(alpha, beta)
    result = SPRTResult(
        lower_bound=lower,
        upper_bound=upper,
        elo0=elo0,
        elo1=elo1,
        candidate_label=label,
        options_tested={
            k: v
            for k, v in candidate_options.items()
            if k not in baseline_options or baseline_options[k] != v
        },
    )

    openings = load_openings()
    random.shuffle(openings)

    baseline = UCIEngine(baseline_cmd, "baseline", baseline_options, baseline_cwd)
    candidate = UCIEngine(candidate_cmd, "candidate", candidate_options, candidate_cwd)

    start_time = time.time()
    game_idx = 0

    try:
        while result.total < max_games:
            opening = openings[game_idx % len(openings)]
            game_idx += 1

            r1, r2 = play_game_pair(
                candidate,
                baseline,
                opening,
                movetime_ms=movetime_ms,
                tc_base_ms=tc_base_ms,
                tc_inc_ms=tc_inc_ms,
                max_moves=max_moves,
                resign_score=resign_score,
                resign_count=resign_count,
                draw_move=draw_move,
                draw_count=draw_count,
                draw_score=draw_score,
            )

            for r in (r1, r2):
                if r == "W":
                    result.wins += 1
                elif r == "L":
                    result.losses += 1
                else:
                    result.draws += 1

            result.llr = llr_trinomial(
                result.wins, result.draws, result.losses, elo0, elo1
            )
            result.elo_est, result.elo_ci_lo, result.elo_ci_hi = elo_estimate(
                result.wins, result.draws, result.losses
            )
            result.games_played = result.total
            result.elapsed_sec = time.time() - start_time

            if verbose and result.total % 10 == 0:
                pct = result.wins / max(1, result.total) * 100
                print(
                    f"  [{result.total:4d}] W={result.wins} D={result.draws} L={result.losses} "
                    f"| Elo={result.elo_est:+.1f} [{result.elo_ci_lo:+.1f}, {result.elo_ci_hi:+.1f}] "
                    f"| LLR={result.llr:.3f} [{lower:.3f}, {upper:.3f}]"
                )

            if result.llr >= upper:
                result.status = "H1"
                break
            elif result.llr <= lower:
                result.status = "H0"
                break

        if result.status == "ongoing":
            result.status = "max_games"

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        result.status = f"error: {e}"
    finally:
        baseline.close()
        candidate.close()
        result.elapsed_sec = time.time() - start_time

    return result


# ---------------------------------------------------------------------------
# Batch testing
# ---------------------------------------------------------------------------


@dataclass
class BatchTest:
    label: str
    options: Dict[str, str]
    elo0: float = 0.0
    elo1: float = 10.0


def load_batch(path: str) -> List[BatchTest]:
    with open(path) as f:
        data = json.load(f)
    tests = []
    for item in data.get("tests", data if isinstance(data, list) else []):
        tests.append(
            BatchTest(
                label=item["label"],
                options=item["options"],
                elo0=item.get("elo0", 0.0),
                elo1=item.get("elo1", 10.0),
            )
        )
    return tests


def run_batch(
    batch_path: str,
    engine_cmd: str,
    base_options: Dict[str, str],
    engine_cwd: Optional[str] = None,
    max_games: int = 2000,
    movetime_ms: int = 1000,
    tc_base_ms: int = 0,
    tc_inc_ms: int = 0,
    max_moves: int = 200,
    resign_score: int = 1000,
    resign_count: int = 3,
    draw_move: int = 40,
    draw_count: int = 8,
    draw_score: int = 10,
    verbose: bool = True,
) -> List[SPRTResult]:
    tests = load_batch(batch_path)
    results = []

    print(f"\n{'='*60}")
    print(f"  SPRT Batch: {len(tests)} tests from {batch_path}")
    print(f"{'='*60}\n")

    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {test.label}")
        print(f"  Options: {test.options}")
        print(f"  H0: Elo >= {test.elo0}, H1: Elo >= {test.elo1}")
        print()

        candidate_options = dict(base_options)
        candidate_options.update(test.options)

        result = run_sprt(
            baseline_cmd=engine_cmd,
            candidate_cmd=engine_cmd,
            baseline_options=base_options,
            candidate_options=candidate_options,
            baseline_cwd=engine_cwd,
            candidate_cwd=engine_cwd,
            elo0=test.elo0,
            elo1=test.elo1,
            max_games=max_games,
            movetime_ms=movetime_ms,
            tc_base_ms=tc_base_ms,
            tc_inc_ms=tc_inc_ms,
            max_moves=max_moves,
            resign_score=resign_score,
            resign_count=resign_count,
            draw_move=draw_move,
            draw_count=draw_count,
            draw_score=draw_score,
            label=test.label,
            verbose=verbose,
        )

        status_icon = {
            "H1": "PASS",
            "H0": "FAIL",
            "max_games": "INCONCLUSIVE",
        }.get(result.status, "ERROR")
        print(
            f"\n  >> {status_icon}: {test.label} | Elo={result.elo_est:+.1f} "
            f"[{result.elo_ci_lo:+.1f}, {result.elo_ci_hi:+.1f}] "
            f"| {result.total} games in {result.elapsed_sec:.0f}s"
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Default engine configuration
# ---------------------------------------------------------------------------


def detect_threads() -> int:
    env = os.getenv("METALFISH_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if sys.platform == "darwin":
        for key in ("hw.perflevel0.physicalcpu_max", "hw.logicalcpu"):
            try:
                out = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
                return max(1, int(out))
            except Exception:
                continue
    return max(1, os.cpu_count() or 8)


def default_hybrid_options(weights_path: str, threads: int) -> Dict[str, str]:
    return {
        "UseHybridSearch": "true",
        "UseMCTS": "false",
        "NNWeights": weights_path,
        "Threads": str(threads),
        "Hash": "2048",
        "MultiPV": "1",
        "HybridMCTSThreads": "1",
        "HybridABThreads": str(max(1, threads - 1)),
        "HybridAutoABThreadsCap": "0",
        "TransformerLowTimeFallbackMs": "1500",
        "TransformerMinMoveBudgetMs": "400",
        "MCTSMaxThreads": "1",
        "MCTSMinibatchSize": "0",
        "MCTSParityPreset": "false",
        "MCTSAddDirichletNoise": "false",
        "HybridMCTSMinimumKLDGainPerNode": "0.0",
        "HybridABRootRejectMCTS": "true",
        "HybridMCTSRootReject": "false",
        "HybridMCTSUseSharedTT": "false",
        "HybridMCTSABRootHints": "true",
        "HybridMCTSABRootHintDelayMs": "0",
        "HybridMCTSABRootHintCount": "8",
        "HybridABCandidateVerifyMs": "240",
        "HybridABCandidateVerifyCount": "5",
        "HybridABPolicyWeight": "0.0",
        "HybridRootPawnLeverTieBreak": "true",
        "HybridTrace": "false",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MetalFish SPRT Engine Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--engine",
        default=str(PROJ / "build" / "metalfish"),
        help="Path to baseline engine binary",
    )
    parser.add_argument(
        "--candidate-engine",
        default=None,
        help="Path to candidate engine binary (default: same as --engine)",
    )
    parser.add_argument(
        "--weights",
        default=str(PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"),
        help="Path to transformer weights",
    )
    parser.add_argument("--threads", type=int, default=0, help="Thread count (0=auto)")
    parser.add_argument("--hash", type=int, default=2048, help="Hash table size in MB")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "ab", "mcts"],
        default="hybrid",
        help="Engine mode to test",
    )

    # SPRT parameters
    parser.add_argument(
        "--elo0", type=float, default=0.0, help="H0 Elo bound (null hypothesis)"
    )
    parser.add_argument(
        "--elo1", type=float, default=10.0, help="H1 Elo bound (alternative hypothesis)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Type I error rate (false positive)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.05, help="Type II error rate (false negative)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=2000,
        help="Maximum games before declaring inconclusive",
    )

    # Time control
    parser.add_argument(
        "--movetime",
        type=int,
        default=1000,
        help="Fixed time per move in ms (used if --tc not set)",
    )
    parser.add_argument(
        "--tc", default=None, help="Time control as BASE+INC in seconds (e.g. '10+0.1')"
    )

    # Adjudication
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum full moves before adjudicating a draw",
    )
    parser.add_argument(
        "--resign-score",
        type=int,
        default=1000,
        help="Centipawn threshold for resign adjudication",
    )
    parser.add_argument(
        "--resign-count",
        type=int,
        default=3,
        help="Consecutive searches beyond --resign-score before adjudicating a loss",
    )
    parser.add_argument(
        "--draw-move",
        type=int,
        default=40,
        help="Earliest full move for draw adjudication",
    )
    parser.add_argument(
        "--draw-count",
        type=int,
        default=8,
        help="Consecutive searches within --draw-score before adjudicating a draw",
    )
    parser.add_argument(
        "--draw-score",
        type=int,
        default=10,
        help="Centipawn window for draw adjudication",
    )

    # Options to test
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        help="Candidate option override as NAME=VALUE (repeatable)",
    )
    parser.add_argument(
        "--baseline-option",
        action="append",
        default=[],
        help="Baseline option override as NAME=VALUE (repeatable)",
    )

    # Batch mode
    parser.add_argument("--batch", default=None, help="Path to batch test JSON file")

    # Self-play sanity
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Run self-play (same config both sides) as sanity check",
    )
    parser.add_argument(
        "--games", type=int, default=None, help="Fixed game count (overrides SPRT)"
    )

    # Output
    parser.add_argument(
        "--json-out", default=None, help="Write results JSON to this path"
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--label", default="", help="Label for this test run")

    args = parser.parse_args()

    threads = args.threads if args.threads > 0 else detect_threads()
    engine_cmd = args.engine
    candidate_cmd = args.candidate_engine or engine_cmd

    # Parse time control
    tc_base_ms = 0
    tc_inc_ms = 0
    if args.tc:
        parts = args.tc.replace("+", " ").split()
        tc_base_ms = int(float(parts[0]) * 1000)
        tc_inc_ms = int(float(parts[1]) * 1000) if len(parts) > 1 else 0

    # Build base options
    if args.mode == "hybrid":
        base_options = default_hybrid_options(args.weights, threads)
    elif args.mode == "ab":
        base_options = {
            "UseHybridSearch": "false",
            "UseMCTS": "false",
            "Threads": str(threads),
            "Hash": str(args.hash),
            "MultiPV": "1",
        }
    else:  # mcts
        base_options = {
            "UseHybridSearch": "false",
            "UseMCTS": "true",
            "NNWeights": args.weights,
            "Threads": str(threads),
            "Hash": str(args.hash),
            "MCTSMaxThreads": "1",
            "MCTSMinibatchSize": "0",
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "TransformerLowTimeFallbackMs": "0",
        }

    base_options["Hash"] = str(args.hash)

    for opt in args.baseline_option:
        k, v = opt.split("=", 1)
        base_options[k] = v

    # Batch mode
    if args.batch:
        results = run_batch(
            args.batch,
            engine_cmd,
            base_options,
            engine_cwd=str(PROJ),
            max_games=args.max_games,
            movetime_ms=args.movetime,
            tc_base_ms=tc_base_ms,
            tc_inc_ms=tc_inc_ms,
            max_moves=args.max_moves,
            resign_score=args.resign_score,
            resign_count=args.resign_count,
            draw_move=args.draw_move,
            draw_count=args.draw_count,
            draw_score=args.draw_score,
            verbose=not args.quiet,
        )
        out_path = args.json_out or str(RESULTS_DIR / "batch_results.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([r.as_dict() for r in results], f, indent=2)
        print(f"\nBatch results saved to: {out_path}")

        passed = sum(1 for r in results if r.passed())
        failed = sum(1 for r in results if r.failed())
        print(
            f"\nSummary: {passed} passed, {failed} failed, "
            f"{len(results) - passed - failed} inconclusive out of {len(results)} tests"
        )
        return 0 if failed == 0 else 1

    # Single test mode
    candidate_options = dict(base_options)
    for opt in args.option:
        k, v = opt.split("=", 1)
        candidate_options[k] = v

    if args.self_play:
        candidate_options = dict(base_options)

    label = args.label or " ".join(args.option) or "self-play"
    max_games = args.games if args.games else args.max_games

    if not args.quiet:
        print(f"\nMetalFish SPRT Test")
        print(f"{'='*50}")
        print(f"  Engine: {engine_cmd}")
        if candidate_cmd != engine_cmd:
            print(f"  Candidate: {candidate_cmd}")
        print(f"  Mode: {args.mode} | Threads: {threads} | Hash: {args.hash}MB")
        if tc_base_ms:
            print(f"  TC: {tc_base_ms/1000:.1f}+{tc_inc_ms/1000:.1f}s")
        else:
            print(f"  Movetime: {args.movetime}ms")
        print(
            f"  SPRT: H0={args.elo0}, H1={args.elo1}, alpha={args.alpha}, beta={args.beta}"
        )
        print(
            "  Adjudication: "
            f"max-moves={args.max_moves}, "
            f"resign={args.resign_count}x{args.resign_score}cp, "
            f"draw={args.draw_count}x±{args.draw_score}cp after move {args.draw_move}"
        )
        print(f"  Max games: {max_games}")
        if args.option:
            print(f"  Testing: {args.option}")
        print(f"{'='*50}\n")

    result = run_sprt(
        baseline_cmd=engine_cmd,
        candidate_cmd=candidate_cmd,
        baseline_options=base_options,
        candidate_options=candidate_options,
        baseline_cwd=str(PROJ),
        candidate_cwd=str(PROJ),
        elo0=args.elo0,
        elo1=args.elo1,
        alpha=args.alpha,
        beta=args.beta,
        max_games=max_games,
        movetime_ms=args.movetime,
        tc_base_ms=tc_base_ms,
        tc_inc_ms=tc_inc_ms,
        max_moves=args.max_moves,
        resign_score=args.resign_score,
        resign_count=args.resign_count,
        draw_move=args.draw_move,
        draw_count=args.draw_count,
        draw_score=args.draw_score,
        label=label,
        verbose=not args.quiet,
    )

    status_map = {"H1": "PASSED", "H0": "FAILED", "max_games": "INCONCLUSIVE"}
    status_str = status_map.get(result.status, result.status.upper())

    if not args.quiet:
        print(f"\n{'='*50}")
        print(f"  RESULT: {status_str}")
        print(
            f"  W={result.wins} D={result.draws} L={result.losses} ({result.total} games)"
        )
        print(
            f"  Elo: {result.elo_est:+.1f} [{result.elo_ci_lo:+.1f}, {result.elo_ci_hi:+.1f}]"
        )
        print(
            f"  LLR: {result.llr:.4f} [{result.lower_bound:.4f}, {result.upper_bound:.4f}]"
        )
        print(f"  Time: {result.elapsed_sec:.0f}s")
        print(f"{'='*50}")

    out_path = args.json_out
    if not out_path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("=", "-")[:40]
        out_path = str(RESULTS_DIR / f"sprt_{safe_label}_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result.as_dict(), f, indent=2)
    if not args.quiet:
        print(f"  Results: {out_path}")

    return 0 if result.status != "H0" else 1


if __name__ == "__main__":
    sys.exit(main())
