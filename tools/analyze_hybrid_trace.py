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
    ab_eval: Optional[int] = None
    mcts_eval: Optional[int] = None

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
                for raw in entry.get("lines", []):
                    line = str(raw).removeprefix("info string ")
                    if not line.startswith("HybridTrace:"):
                        continue
                    fields = parse_fields(line)
                    ab_move = fields.get("ABMove", "none")
                    mcts_move = fields.get("MCTSMove", "none")
                    yield TraceDecision(
                        game=game_no,
                        ply=ply,
                        side=side,
                        fen=fen,
                        played=played,
                        reason=fields.get("reason", "?"),
                        selected=fields.get("selected", "none"),
                        ab_move=ab_move,
                        mcts_move=mcts_move,
                        fields=fields,
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
    reason_bonus = 1000 if decision.reason.startswith("mcts_") else 0
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


def print_comparisons(comparisons: list[MoveComparison], bucket_min: int) -> None:
    if not comparisons:
        print("No disagreement candidates matched the filters.")
        return

    by_reason: dict[str, list[int]] = {}
    for comparison in comparisons:
        diff = comparison.mcts_minus_ab
        if diff is None:
            continue
        by_reason.setdefault(comparison.decision.reason, []).append(diff)

    print("Stockfish comparison summary, centipawns from original side:")
    for reason, diffs in sorted(by_reason.items()):
        better = sum(1 for diff in diffs if diff > 0)
        equal = sum(1 for diff in diffs if diff == 0)
        worse = sum(1 for diff in diffs if diff < 0)
        average = sum(diffs) / len(diffs)
        print(
            f"  {reason}: n={len(diffs)} mcts_better={better} "
            f"equal={equal} ab_better={worse} avg_mcts_minus_ab={average:+.1f}"
        )
    print()

    header = (
        "game ply side reason selected AB sfAB MCTS sfMCTS "
        "MCTS-AB share gap delta visits/root current/root confidence/root"
    )
    print(header)
    for comparison in comparisons:
        d = comparison.decision
        fields = d.fields
        diff = comparison.mcts_minus_ab
        visits, root_visits, current_visits, current_root_visits = visit_pair(fields)
        confidence_visits, confidence_root_visits = confidence_pair(fields)
        print(
            f"{d.game:>4} {d.ply:>3} {d.side:<5} {d.reason:<34} "
            f"{d.selected:<5} {d.ab_move:<5} {comparison.ab_eval:+5d} "
            f"{d.mcts_move:<5} {comparison.mcts_eval:+6d} {diff:+7d} "
            f"{field_float(fields, 'VisitShare'):.3f} "
            f"{field_float(fields, 'RootQGap'):.3f} "
            f"{field_int(fields, 'EvalDelta'):>5} "
            f"{visits}/{root_visits} {current_visits}/{current_root_visits} "
            f"{confidence_visits}/{confidence_root_visits}"
        )
    print_bucket_summary(comparisons, bucket_min)


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
        "--keep-hash",
        action="store_true",
        help="Do not clear Stockfish hash before each candidate eval.",
    )
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
        if d.ab_move != d.mcts_move
        and legal_uci(d.fen, d.ab_move)
        and legal_uci(d.fen, d.mcts_move)
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
            comparisons.append(
                MoveComparison(
                    decision=decision,
                    ab_eval=probe.eval_after(
                        decision.fen, decision.ab_move, args.depth
                    ),
                    mcts_eval=probe.eval_after(
                        decision.fen, decision.mcts_move, args.depth
                    ),
                )
            )
    finally:
        probe.close()

    print_comparisons(comparisons, args.bucket_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
