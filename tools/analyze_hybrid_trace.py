#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

try:
    import chess
except ImportError:
    print("ERROR: python-chess required. Install: pip install python-chess")
    sys.exit(1)

PROJ = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_STOCKFISH = PROJ / "reference" / "stockfish" / "src" / "stockfish"
DEFAULT_METALFISH = PROJ / "build" / "metalfish"
DEFAULT_WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"


@dataclass
class TraceDecision:
    game: int
    ply: int
    side: str
    fen: str
    played: str
    reason: str
    selected: str
    ab_move: str
    mcts_move: str
    fields: dict[str, str]


@dataclass
class MoveComparison:
    decision: TraceDecision
    selected_eval: Optional[int] = None
    ab_eval: Optional[int] = None
    mcts_eval: Optional[int] = None

    @property
    def selected_minus_ab(self) -> Optional[int]:
        if self.selected_eval is None or self.ab_eval is None:
            return None
        return self.selected_eval - self.ab_eval

    @property
    def selected_minus_mcts(self) -> Optional[int]:
        if self.selected_eval is None or self.mcts_eval is None:
            return None
        return self.selected_eval - self.mcts_eval

    @property
    def mcts_minus_ab(self) -> Optional[int]:
        if self.ab_eval is None or self.mcts_eval is None:
            return None
        return self.mcts_eval - self.ab_eval


@dataclass
class TraceLogStats:
    search_entries: int = 0
    hybrid_starts: int = 0
    trace_entries: int = 0
    time_safety_fallbacks: int = 0
    time_safety_reasons: dict[str, int] = field(default_factory=dict)
    root_hint_events: int = 0
    root_hint_moves_total: int = 0
    root_hint_sizes: dict[int, int] = field(default_factory=dict)


def latest_results_file() -> pathlib.Path:
    candidates = sorted(
        PROJ.glob("results/tournament_*/results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if any(
            game.get("search_log")
            for match in data.get("matches", [])
            for game in match.get("games", [])
        ):
            return path
    raise FileNotFoundError("no traced tournament results found")


def parse_fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def normalize_time_safety_reason(reason: str) -> str:
    if reason.startswith("estimated move budget "):
        return "estimated move budget"
    if reason.startswith("fixed movetime "):
        return "fixed movetime"
    if reason.startswith("white clock "):
        return "white clock"
    if reason.startswith("black clock "):
        return "black clock"
    return reason


def iter_trace_decisions(results_path: pathlib.Path) -> Iterable[TraceDecision]:
    data = json.loads(results_path.read_text())
    for match in data.get("matches", []):
        for game in match.get("games", []):
            game_no = int(game.get("game", 0))
            for entry in game.get("search_log", []):
                fen = str(entry.get("fen", ""))
                played = str(entry.get("move", ""))
                side = str(entry.get("side", ""))
                ply = int(entry.get("ply", 0))
                last_trace_fields: Optional[dict[str, str]] = None
                for raw in entry.get("lines", []):
                    line = str(raw).removeprefix("info string ")
                    if not line.startswith("HybridTrace:"):
                        continue
                    last_trace_fields = parse_fields(line)
                if last_trace_fields is None:
                    continue
                ab_move = last_trace_fields.get("ABMove", "none")
                mcts_move = last_trace_fields.get("MCTSMove", "none")
                yield TraceDecision(
                    game=game_no,
                    ply=ply,
                    side=side,
                    fen=fen,
                    played=played,
                    reason=last_trace_fields.get("reason", "?"),
                    selected=last_trace_fields.get("selected", "none"),
                    ab_move=ab_move,
                    mcts_move=mcts_move,
                    fields=last_trace_fields,
                )


def collect_trace_log_stats(results_paths: list[pathlib.Path]) -> TraceLogStats:
    stats = TraceLogStats()
    for results_path in results_paths:
        data = json.loads(results_path.read_text())
        for match in data.get("matches", []):
            for game in match.get("games", []):
                for entry in game.get("search_log", []):
                    stats.search_entries += 1
                    has_trace = False
                    for raw in entry.get("lines", []):
                        line = str(raw).removeprefix("info string ")
                        if line.startswith("Starting Parallel Hybrid Search"):
                            stats.hybrid_starts += 1
                        elif line.startswith("Time safety:"):
                            stats.time_safety_fallbacks += 1
                            reason = line.removeprefix("Time safety:").strip()
                            reason = reason.split(";", 1)[0].strip()
                            reason = normalize_time_safety_reason(reason)
                            stats.time_safety_reasons[reason] = (
                                stats.time_safety_reasons.get(reason, 0) + 1
                            )
                        elif line.startswith("Hybrid: AB root hints from MCTS"):
                            stats.root_hint_events += 1
                            hint_count = max(0, len(line.split()) - 6)
                            stats.root_hint_moves_total += hint_count
                            stats.root_hint_sizes[hint_count] = (
                                stats.root_hint_sizes.get(hint_count, 0) + 1
                            )
                        elif line.startswith("HybridTrace:"):
                            has_trace = True
                    if has_trace:
                        stats.trace_entries += 1
    return stats


def legal_uci(fen: str, move: str) -> bool:
    if move in {"", "none", "0000"}:
        return False
    try:
        board = chess.Board(fen)
        return chess.Move.from_uci(move) in board.legal_moves
    except ValueError:
        return False


def comparable_trace_decision(decision: TraceDecision) -> bool:
    selected_legal = legal_uci(decision.fen, decision.selected)
    ab_legal = legal_uci(decision.fen, decision.ab_move)
    mcts_legal = legal_uci(decision.fen, decision.mcts_move)
    if not selected_legal or not (ab_legal or mcts_legal):
        return False
    legal_moves = {decision.selected}
    if ab_legal:
        legal_moves.add(decision.ab_move)
    if mcts_legal:
        legal_moves.add(decision.mcts_move)
    return len(legal_moves) > 1


def field_float(fields: dict[str, str], name: str, default: float = 0.0) -> float:
    try:
        return float(fields.get(name, default))
    except (TypeError, ValueError):
        return default


def field_int(fields: dict[str, str], name: str, default: int = 0) -> int:
    try:
        return int(float(fields.get(name, default)))
    except (TypeError, ValueError):
        return default


def visit_pair(fields: dict[str, str]) -> tuple[int, int, int, int]:
    visits = field_int(fields, "MCTSBestVisits")
    root_visits = field_int(fields, "MCTSRootVisits")
    current_visits = field_int(fields, "MCTSBestCurrentVisits", visits)
    current_root_visits = field_int(fields, "MCTSRootCurrentVisits", root_visits)
    return visits, root_visits, current_visits, current_root_visits


def confidence_pair(fields: dict[str, str]) -> tuple[int, int]:
    visits, root_visits, current_visits, current_root_visits = visit_pair(fields)
    confidence_visits = field_int(fields, "MCTSConfidenceVisits", current_visits)
    confidence_root_visits = field_int(
        fields, "MCTSConfidenceRootVisits", current_root_visits
    )
    return confidence_visits, confidence_root_visits


def bucket(value: float, cuts: list[float], labels: list[str]) -> str:
    for limit, label in zip(cuts, labels):
        if value < limit:
            return label
    return labels[-1]


def interesting_score(decision: TraceDecision) -> tuple[int, float, int, int]:
    fields = decision.fields
    reason_bonus = (
        1000
        if decision.reason.startswith("mcts_")
        or decision.reason == "root_pawn_lever_tiebreak"
        else 0
    )
    share = field_float(fields, "VisitShare")
    gap = field_float(fields, "RootQGap")
    visits = field_int(fields, "MCTSBestVisits")
    delta = field_int(fields, "EvalDelta")
    return (reason_bonus, share + max(0.0, gap), visits, delta)


class StockfishProbe:
    def __init__(
        self,
        path: pathlib.Path,
        threads: int,
        hash_mb: int,
        clear_hash: bool = True,
    ) -> None:
        self.clear_hash = clear_hash
        self.proc = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            if self.proc.stdin is None or self.proc.stdout is None:
                raise RuntimeError("failed to start Stockfish")
            self.send("uci")
            self.wait_for("uciok")
            self.send(f"setoption name Threads value {threads}")
            self.send(f"setoption name Hash value {hash_mb}")
            self.send("isready")
            self.wait_for("readyok")
        except Exception:
            self.close()
            raise

    def send(self, command: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def wait_for(self, prefix: str, timeout: int = 30) -> str:
        assert self.proc.stdout is not None
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith(prefix):
                return line
        raise TimeoutError(prefix)

    def eval_after(self, fen: str, move: str, depth: int) -> int:
        assert self.proc.stdout is not None
        if self.clear_hash:
            self.send("setoption name Clear Hash")
            self.send("isready")
            self.wait_for("readyok")
        self.send(f"position fen {fen} moves {move}")
        self.send(f"go depth {depth}")
        score: Optional[int] = None
        mate: Optional[int] = None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish exited during search")
            line = line.strip()
            if line.startswith("info ") and " score " in line:
                cp_match = re.search(r" score cp (-?\d+)", line)
                mate_match = re.search(r" score mate (-?\d+)", line)
                if cp_match:
                    score = int(cp_match.group(1))
                    mate = None
                elif mate_match:
                    mate = int(mate_match.group(1))
                    score = None
            if line.startswith("bestmove"):
                break
        if mate is not None:
            value = 30000 - mate if mate > 0 else -30000 - mate
        else:
            value = score or 0
        return -value

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.send("quit")
                self.proc.wait(timeout=5)
        except Exception:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
        finally:
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass


@dataclass
class ReplayResult:
    decision: TraceDecision
    bestmove: str
    reason: str
    selected: str
    ab_move: str
    mcts_move: str
    elapsed_ms: int


class MetalFishProbe:
    def __init__(
        self,
        path: pathlib.Path,
        weights: pathlib.Path,
        threads: int,
        hash_mb: int,
        hybrid_mcts_threads: int,
        hybrid_ab_threads: int,
        low_time_fallback_ms: int,
    ) -> None:
        self.proc = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            if self.proc.stdin is None or self.proc.stdout is None:
                raise RuntimeError("failed to start MetalFish")
            self.send("uci")
            self.wait_for("uciok", 60)
            options = {
                "UseMCTS": "false",
                "UseHybridSearch": "true",
                "NNWeights": str(weights),
                "Threads": str(threads),
                "Hash": str(hash_mb),
                "MultiPV": "1",
                "HybridMCTSThreads": str(hybrid_mcts_threads),
                "HybridABThreads": str(hybrid_ab_threads),
                "HybridAutoABThreadsCap": "0",
                "TransformerLowTimeFallbackMs": str(low_time_fallback_ms),
                "TransformerMinMoveBudgetMs": "400",
                "MCTSMaxThreads": str(hybrid_mcts_threads),
                "MCTSMinibatchSize": "0",
                "MCTSParityPreset": "false",
                "MCTSAddDirichletNoise": "false",
                "HybridMCTSMinimumKLDGainPerNode": "0.0",
                "HybridABRootRejectMCTS": "true",
                "HybridMCTSRootReject": "true",
                "HybridMCTSABRootHints": "true",
                "HybridMCTSABRootHintDelayMs": "0",
                "HybridMCTSABRootHintCount": "8",
                "HybridABCandidateVerifyMs": "120",
                "HybridABCandidateVerifyCount": "4",
                "HybridABPolicyWeight": "0.0",
                "HybridRootPawnLeverTieBreak": "true",
                "HybridTrace": "true",
            }
            for name, value in options.items():
                self.send(f"setoption name {name} value {value}")
            self.send("isready")
            self.wait_for("readyok", 120)
        except Exception:
            self.close()
            raise

    def send(self, command: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def wait_for(self, prefix: str, timeout: int = 30) -> str:
        assert self.proc.stdout is not None
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"MetalFish exited with code {self.proc.returncode}"
                    )
                continue
            line = line.strip()
            if line.startswith(prefix):
                return line
        raise TimeoutError(prefix)

    def replay(
        self, decision: TraceDecision, movetime_ms: int, nodes: int
    ) -> ReplayResult:
        assert self.proc.stdout is not None
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok", 120)
        self.send(f"position fen {decision.fen}")
        start = time.time()
        if nodes > 0:
            self.send(f"go nodes {nodes}")
            timeout = max(60.0, nodes / 10.0 + 30.0)
        else:
            self.send(f"go movetime {movetime_ms}")
            timeout = max(60.0, movetime_ms / 1000.0 + 30.0)

        bestmove = "0000"
        trace_fields: dict[str, str] = {}
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"MetalFish exited during search with code {self.proc.returncode}"
                    )
                continue
            line = line.strip()
            if line.startswith("info string HybridTrace:"):
                trace_fields = parse_fields(line.removeprefix("info string ").strip())
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) > 1:
                    bestmove = parts[1]
                break
        else:
            raise TimeoutError("bestmove")

        return ReplayResult(
            decision=decision,
            bestmove=bestmove,
            reason=trace_fields.get("reason", "?"),
            selected=trace_fields.get("selected", bestmove),
            ab_move=trace_fields.get("ABMove", "none"),
            mcts_move=trace_fields.get("MCTSMove", "none"),
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.send("quit")
                self.proc.wait(timeout=5)
        except Exception:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
        finally:
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass


def print_summary(decisions: list[TraceDecision]) -> None:
    reasons: dict[str, int] = {}
    disagreements = 0
    for decision in decisions:
        reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
        if decision.ab_move != decision.mcts_move:
            disagreements += 1

    print(f"Hybrid trace decisions: {len(decisions)}")
    print(f"AB/MCTS disagreements: {disagreements}")
    print("Reasons:")
    for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {reason}: {count}")


def print_trace_log_stats(stats: TraceLogStats) -> None:
    if stats.search_entries == 0:
        return
    coverage = 100.0 * stats.trace_entries / max(1, stats.search_entries)
    print("Trace log coverage:")
    print(f"  search_log entries: {stats.search_entries}")
    print(f"  traced decisions: {stats.trace_entries} ({coverage:.1f}%)")
    print(f"  hybrid starts: {stats.hybrid_starts}")
    print(f"  MCTS-to-AB root hint events: {stats.root_hint_events}")
    if stats.root_hint_events:
        avg_hints = stats.root_hint_moves_total / stats.root_hint_events
        sizes = ", ".join(
            f"{size}:{count}x" for size, count in sorted(stats.root_hint_sizes.items())
        )
        print(f"  MCTS-to-AB root hint avg moves: {avg_hints:.2f}")
        print(f"  MCTS-to-AB root hint sizes: {sizes}")
    print(f"  time-safety AB fallbacks: {stats.time_safety_fallbacks}")
    if stats.time_safety_reasons:
        print("  fallback reasons:")
        for reason, count in sorted(
            stats.time_safety_reasons.items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"    {count}x {reason}")


def print_bucket_summary(comparisons: list[MoveComparison], min_count: int) -> None:
    if not comparisons:
        return

    bucketed: dict[str, dict[str, list[int]]] = {
        "share": {},
        "root_q_gap": {},
        "eval_delta": {},
        "confidence_visits": {},
        "confidence_root": {},
        "abs_ab_score": {},
    }

    def add(kind: str, label: str, diff: int) -> None:
        bucketed[kind].setdefault(label, []).append(diff)

    for comparison in comparisons:
        diff = comparison.mcts_minus_ab
        if diff is None:
            continue
        fields = comparison.decision.fields
        confidence_visits, confidence_root_visits = confidence_pair(fields)
        add(
            "share",
            bucket(
                field_float(fields, "VisitShare"),
                [0.35, 0.45, 0.55, 0.65],
                ["<0.35", "0.35-0.45", "0.45-0.55", "0.55-0.65", ">=0.65"],
            ),
            diff,
        )
        add(
            "root_q_gap",
            bucket(
                field_float(fields, "RootQGap"),
                [0.0, 0.02, 0.08, 0.16],
                ["<0.00", "0.00-0.02", "0.02-0.08", "0.08-0.16", ">=0.16"],
            ),
            diff,
        )
        add(
            "eval_delta",
            bucket(
                field_int(fields, "EvalDelta"),
                [0, 50, 100, 200],
                ["<0", "0-49", "50-99", "100-199", ">=200"],
            ),
            diff,
        )
        add(
            "confidence_visits",
            bucket(
                confidence_visits,
                [20, 40, 80, 140],
                ["<20", "20-39", "40-79", "80-139", ">=140"],
            ),
            diff,
        )
        add(
            "confidence_root",
            bucket(
                confidence_root_visits,
                [50, 100, 180, 300],
                ["<50", "50-99", "100-179", "180-299", ">=300"],
            ),
            diff,
        )
        add(
            "abs_ab_score",
            bucket(
                abs(field_int(fields, "ABScore")),
                [25, 75, 150, 300],
                ["<25", "25-74", "75-149", "150-299", ">=300"],
            ),
            diff,
        )

    print()
    print("Feature bucket summary, centipawns from original side:")
    for kind, groups in bucketed.items():
        print(f"  {kind}:")
        for label, diffs in sorted(groups.items()):
            if len(diffs) < min_count:
                continue
            better = sum(1 for diff in diffs if diff > 0)
            equal = sum(1 for diff in diffs if diff == 0)
            worse = sum(1 for diff in diffs if diff < 0)
            average = sum(diffs) / len(diffs)
            print(
                f"    {label}: n={len(diffs)} mcts_better={better} "
                f"equal={equal} ab_better={worse} avg={average:+.1f}"
            )


def format_cp(value: Optional[int], width: int = 6) -> str:
    if value is None:
        return " " * (width - 2) + "--"
    return f"{value:+{width}d}"


def format_diff(value: Optional[int], width: int = 7) -> str:
    if value is None:
        return " " * (width - 2) + "--"
    return f"{value:+{width}d}"


def summarize_diffs(
    label: str,
    diffs: list[int],
    better_label: str,
    worse_label: str,
    average_label: str,
) -> str:
    better = sum(1 for diff in diffs if diff > 0)
    equal = sum(1 for diff in diffs if diff == 0)
    worse = sum(1 for diff in diffs if diff < 0)
    average = sum(diffs) / len(diffs)
    return (
        f"    {label}: n={len(diffs)} {better_label}={better} "
        f"equal={equal} {worse_label}={worse} {average_label}={average:+.1f}"
    )


def print_comparisons(comparisons: list[MoveComparison], bucket_min: int) -> None:
    if not comparisons:
        print("No decision candidates matched the filters.")
        return

    by_reason: dict[str, dict[str, list[int]]] = {}
    for comparison in comparisons:
        groups = by_reason.setdefault(
            comparison.decision.reason,
            {
                "selected_minus_ab": [],
                "selected_minus_mcts": [],
                "mcts_minus_ab": [],
            },
        )
        if comparison.selected_minus_ab is not None:
            groups["selected_minus_ab"].append(comparison.selected_minus_ab)
        if comparison.selected_minus_mcts is not None:
            groups["selected_minus_mcts"].append(comparison.selected_minus_mcts)
        if comparison.mcts_minus_ab is not None:
            groups["mcts_minus_ab"].append(comparison.mcts_minus_ab)

    print("Stockfish comparison summary, centipawns from original side:")
    for reason, groups in sorted(by_reason.items()):
        print(f"  {reason}:")
        if groups["selected_minus_ab"]:
            print(
                summarize_diffs(
                    "selected_vs_ab",
                    groups["selected_minus_ab"],
                    "selected_better",
                    "ab_better",
                    "avg_selected_minus_ab",
                )
            )
        if groups["selected_minus_mcts"]:
            print(
                summarize_diffs(
                    "selected_vs_mcts",
                    groups["selected_minus_mcts"],
                    "selected_better",
                    "mcts_better",
                    "avg_selected_minus_mcts",
                )
            )
        if groups["mcts_minus_ab"]:
            print(
                summarize_diffs(
                    "mcts_vs_ab",
                    groups["mcts_minus_ab"],
                    "mcts_better",
                    "ab_better",
                    "avg_mcts_minus_ab",
                )
            )
    print()

    header = (
        "game ply side reason selected sfSelected AB sfAB selected-AB "
        "MCTS sfMCTS selected-MCTS MCTS-AB share gap delta "
        "visits/root current/root confidence/root"
    )
    print(header)
    for comparison in comparisons:
        d = comparison.decision
        fields = d.fields
        visits, root_visits, current_visits, current_root_visits = visit_pair(fields)
        confidence_visits, confidence_root_visits = confidence_pair(fields)
        print(
            f"{d.game:>4} {d.ply:>3} {d.side:<5} {d.reason:<34} "
            f"{d.selected:<5} {format_cp(comparison.selected_eval)} "
            f"{d.ab_move:<5} {format_cp(comparison.ab_eval)} "
            f"{format_diff(comparison.selected_minus_ab)} "
            f"{d.mcts_move:<5} {format_cp(comparison.mcts_eval)} "
            f"{format_diff(comparison.selected_minus_mcts)} "
            f"{format_diff(comparison.mcts_minus_ab)} "
            f"{field_float(fields, 'VisitShare'):.3f} "
            f"{field_float(fields, 'RootQGap'):.3f} "
            f"{field_int(fields, 'EvalDelta'):>5} "
            f"{visits}/{root_visits} {current_visits}/{current_root_visits} "
            f"{confidence_visits}/{confidence_root_visits}"
        )
    print_bucket_summary(comparisons, bucket_min)


def print_replay_results(results: list[ReplayResult]) -> None:
    if not results:
        return

    print()
    print("Current MetalFish replay:")
    changed = 0
    for result in results:
        decision = result.decision
        same_move = result.bestmove == decision.selected
        same_reason = result.reason == decision.reason
        if not same_move or not same_reason:
            changed += 1
        marker = "=" if same_move and same_reason else "*"
        print(
            f"  {marker} game={decision.game} ply={decision.ply} side={decision.side} "
            f"orig={decision.selected}/{decision.reason} "
            f"replay={result.bestmove}/{result.reason} "
            f"AB={result.ab_move} MCTS={result.mcts_move} "
            f"elapsed={result.elapsed_ms}ms"
        )
    print(f"Replay deltas: {changed}/{len(results)} changed move or reason")


def evaluate_trace_decision(
    probe: StockfishProbe, decision: TraceDecision, depth: int
) -> MoveComparison:
    evals: dict[str, int] = {}
    for move in (decision.selected, decision.ab_move, decision.mcts_move):
        if move in evals or not legal_uci(decision.fen, move):
            continue
        evals[move] = probe.eval_after(decision.fen, move, depth)
    return MoveComparison(
        decision=decision,
        selected_eval=evals.get(decision.selected),
        ab_eval=evals.get(decision.ab_move),
        mcts_eval=evals.get(decision.mcts_move),
    )


def replay_candidates(
    args: argparse.Namespace, candidates: list[TraceDecision]
) -> None:
    if not args.replay_current:
        return
    if not args.metalfish.exists():
        print(f"ERROR: MetalFish not found at {args.metalfish}", file=sys.stderr)
        raise SystemExit(2)
    if not args.weights.exists():
        print(f"ERROR: weights not found at {args.weights}", file=sys.stderr)
        raise SystemExit(2)

    probe = MetalFishProbe(
        args.metalfish,
        args.weights,
        args.replay_threads,
        args.replay_hash,
        args.replay_hybrid_mcts_threads,
        args.replay_hybrid_ab_threads,
        args.replay_low_time_fallback_ms,
    )
    try:
        replay_results = [
            probe.replay(d, args.replay_movetime, args.replay_nodes) for d in candidates
        ]
    finally:
        probe.close()
    print_replay_results(replay_results)


def selected_quality_failures(
    comparisons: list[MoveComparison], threshold_cp: int
) -> list[str]:
    if threshold_cp <= 0:
        return []

    failures: list[str] = []
    for comparison in comparisons:
        d = comparison.decision
        if (
            comparison.selected_minus_ab is not None
            and comparison.selected_minus_ab < -threshold_cp
        ):
            failures.append(
                f"game={d.game} ply={d.ply} reason={d.reason} "
                f"selected={d.selected} AB={d.ab_move} "
                f"selected_minus_ab={comparison.selected_minus_ab}"
            )
        if (
            comparison.selected_minus_mcts is not None
            and comparison.selected_minus_mcts < -threshold_cp
        ):
            failures.append(
                f"game={d.game} ply={d.ply} reason={d.reason} "
                f"selected={d.selected} MCTS={d.mcts_move} "
                f"selected_minus_mcts={comparison.selected_minus_mcts}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze saved HybridTrace disagreement decisions."
    )
    parser.add_argument(
        "results_json",
        nargs="*",
        type=pathlib.Path,
        help="Tournament results.json file(s) with --save-search-log data.",
    )
    parser.add_argument("--stockfish", type=pathlib.Path, default=DEFAULT_STOCKFISH)
    parser.add_argument("--depth", type=int, default=13)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=256)
    parser.add_argument("--metalfish", type=pathlib.Path, default=DEFAULT_METALFISH)
    parser.add_argument("--weights", type=pathlib.Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--min-share", type=float, default=0.0)
    parser.add_argument("--max-share", type=float, default=10.0)
    parser.add_argument("--min-visits", type=int, default=0)
    parser.add_argument("--min-current-visits", type=int, default=0)
    parser.add_argument("--min-current-root-visits", type=int, default=0)
    parser.add_argument("--min-gap", type=float, default=-10.0)
    parser.add_argument("--max-gap", type=float, default=10.0)
    parser.add_argument("--min-delta", type=int, default=-1_000_000_000)
    parser.add_argument("--max-delta", type=int, default=1_000_000_000)
    parser.add_argument("--bucket-min", type=int, default=3)
    parser.add_argument("--reason", action="append", default=[])
    parser.add_argument("--no-stockfish", action="store_true")
    parser.add_argument("--require-sane-visits", action="store_true")
    parser.add_argument(
        "--fail-selected-worse-than",
        type=int,
        default=0,
        metavar="CP",
        help=(
            "Exit non-zero if the selected move is worse than AB or MCTS by "
            "more than this many Stockfish centipawns (0=disabled)."
        ),
    )
    parser.add_argument(
        "--keep-hash",
        action="store_true",
        help="Do not clear Stockfish hash before each candidate eval.",
    )
    parser.add_argument(
        "--replay-current",
        action="store_true",
        help="Replay filtered candidate FENs with the current MetalFish Hybrid binary.",
    )
    parser.add_argument("--replay-movetime", type=int, default=3000)
    parser.add_argument("--replay-nodes", type=int, default=0)
    parser.add_argument("--replay-threads", type=int, default=8)
    parser.add_argument("--replay-hash", type=int, default=4096)
    parser.add_argument("--replay-hybrid-mcts-threads", type=int, default=0)
    parser.add_argument("--replay-hybrid-ab-threads", type=int, default=0)
    parser.add_argument("--replay-low-time-fallback-ms", type=int, default=3000)
    args = parser.parse_args()

    results_paths = args.results_json or [latest_results_file()]
    decisions = [
        decision
        for results_path in results_paths
        for decision in iter_trace_decisions(results_path)
    ]
    print("Results:")
    for results_path in results_paths:
        print(f"  {results_path}")
    print_trace_log_stats(collect_trace_log_stats(results_paths))
    print_summary(decisions)

    candidates = [
        d
        for d in decisions
        if comparable_trace_decision(d)
        and field_float(d.fields, "VisitShare") >= args.min_share
        and field_float(d.fields, "VisitShare") <= args.max_share
        and field_int(d.fields, "MCTSBestVisits") >= args.min_visits
        and field_int(
            d.fields,
            "MCTSBestCurrentVisits",
            field_int(d.fields, "MCTSBestVisits"),
        )
        >= args.min_current_visits
        and field_int(
            d.fields,
            "MCTSRootCurrentVisits",
            field_int(d.fields, "MCTSRootVisits"),
        )
        >= args.min_current_root_visits
        and field_float(d.fields, "RootQGap") >= args.min_gap
        and field_float(d.fields, "RootQGap") <= args.max_gap
        and field_int(d.fields, "EvalDelta") >= args.min_delta
        and field_int(d.fields, "EvalDelta") <= args.max_delta
        and (
            not args.require_sane_visits
            or field_int(d.fields, "MCTSVisitEvidenceSane") == 1
        )
        and (not args.reason or d.reason in args.reason)
    ]
    candidates.sort(key=interesting_score, reverse=True)
    candidates = candidates[: max(0, args.limit)]

    if args.no_stockfish:
        replay_candidates(args, candidates)
        for d in candidates:
            fields = d.fields
            visits, root_visits, current_visits, current_root_visits = visit_pair(
                fields
            )
            confidence_visits, confidence_root_visits = confidence_pair(fields)
            print(
                f"candidate game={d.game} ply={d.ply} side={d.side} "
                f"reason={d.reason} selected={d.selected} AB={d.ab_move} "
                f"MCTS={d.mcts_move} share={field_float(fields, 'VisitShare'):.3f} "
                f"gap={field_float(fields, 'RootQGap'):.3f} "
                f"delta={field_int(fields, 'EvalDelta')} "
                f"visits={visits}/{root_visits} "
                f"current={current_visits}/{current_root_visits} "
                f"confidence={confidence_visits}/{confidence_root_visits}"
            )
        return 0

    if not args.stockfish.exists():
        print(f"ERROR: Stockfish not found at {args.stockfish}", file=sys.stderr)
        return 2

    probe = StockfishProbe(args.stockfish, args.threads, args.hash, not args.keep_hash)
    comparisons: list[MoveComparison] = []
    try:
        for decision in candidates:
            comparisons.append(evaluate_trace_decision(probe, decision, args.depth))
    finally:
        probe.close()

    replay_candidates(args, candidates)
    print_comparisons(comparisons, args.bucket_min)
    failures = selected_quality_failures(comparisons, args.fail_selected_worse_than)
    if failures:
        print()
        print(
            "Selected-move quality failures "
            f"(threshold {args.fail_selected_worse_than} cp):"
        )
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
