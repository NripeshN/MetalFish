#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


PROJ = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_DIR = PROJ / "results" / "lichess_audit"
BOT_ID = "nripesh-metalfish"


@dataclass
class AuditSummary:
    game_id: str
    path: pathlib.Path
    color: str = "?"
    accepted_moves: int = 0
    rejected_moves: int = 0
    stale_rejects: int = 0
    book_moves: int = 0
    engine_searches: int = 0
    ponder_starts: int = 0
    ponderhits: int = 0
    max_engine_ms: int = 0
    max_ponderhit_ms: int = 0
    issue_counts: Counter[str] = field(default_factory=Counter)
    final_status: str = ""
    final_winner: str = ""


@dataclass
class GameSummary:
    audit: AuditSummary
    result: str = "unknown"
    color: str = "?"
    opponent: str = "?"
    opponent_rating: int | None = None
    our_rating: int | None = None
    rating_diff: int | None = None
    speed: str = "?"
    rated: bool | None = None
    plies: int = 0
    opening: str = ""
    url: str = ""


def latest_audit_files(audit_dir: pathlib.Path, limit: int) -> list[pathlib.Path]:
    files = sorted(
        audit_dir.glob("*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[:limit]


def parse_audit(path: pathlib.Path) -> AuditSummary:
    summary = AuditSummary(game_id=path.stem, path=path)
    for raw in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            summary.issue_counts["audit_json_decode"] += 1
            continue

        event = str(row.get("event", ""))
        if event == "stream_game_full":
            summary.color = str(row.get("color") or summary.color)
        elif event == "move_submit":
            if row.get("result") == "accepted":
                summary.accepted_moves += 1
            else:
                summary.rejected_moves += 1
                if row.get("stale"):
                    summary.stale_rejects += 1
                else:
                    summary.issue_counts["move_rejected"] += 1
        elif event == "book_candidate":
            summary.book_moves += 1
        elif event == "engine_search_result":
            summary.engine_searches += 1
            summary.max_engine_ms = max(
                summary.max_engine_ms, int(row.get("elapsed_ms") or 0)
            )
        elif event == "ponder_start":
            summary.ponder_starts += 1
        elif event == "ponderhit_result":
            summary.ponderhits += 1
            summary.max_ponderhit_ms = max(
                summary.max_ponderhit_ms, int(row.get("elapsed_ms") or 0)
            )
        elif event in {
            "ponderhit_failed",
            "ponderhit_illegal",
            "ponder_stop_failed",
            "ponder_start_rejected",
            "engine_search_failed",
            "turn_stale",
        }:
            summary.issue_counts[event] += 1
        elif event == "stream_game_over":
            summary.final_status = str(row.get("status", ""))
            summary.final_winner = str(row.get("winner", ""))

    return summary


def lichess_export(game_id: str, timeout: float = 20.0) -> dict:
    params = urllib.parse.urlencode(
        {
            "pgnInJson": "true",
            "clocks": "true",
            "evals": "false",
            "tags": "true",
        }
    )
    url = f"https://lichess.org/game/export/{game_id}?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "MetalFish-lichess-audit/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def game_summary_from_export(audit: AuditSummary, data: dict) -> GameSummary:
    players = data.get("players", {})
    white_id = players.get("white", {}).get("user", {}).get("id", "")
    black_id = players.get("black", {}).get("user", {}).get("id", "")
    color = "white" if white_id == BOT_ID else "black" if black_id == BOT_ID else "?"
    opponent_side = "black" if color == "white" else "white"
    our_side = color if color in {"white", "black"} else "white"

    status = str(data.get("status", ""))
    winner = str(data.get("winner", ""))
    if status == "draw":
        result = "draw"
    elif winner == color:
        result = "win"
    elif winner in {"white", "black"}:
        result = "loss"
    else:
        result = status or "unknown"

    opponent = players.get(opponent_side, {})
    ours = players.get(our_side, {})
    return GameSummary(
        audit=audit,
        result=result,
        color=color,
        opponent=str(opponent.get("user", {}).get("name", "?")),
        opponent_rating=opponent.get("rating"),
        our_rating=ours.get("rating"),
        rating_diff=ours.get("ratingDiff"),
        speed=str(data.get("speed", "?")),
        rated=data.get("rated"),
        plies=len(str(data.get("moves", "")).split()),
        opening=str(data.get("opening", {}).get("name", "")),
        url=f"https://lichess.org/{audit.game_id}",
    )


def game_summary_from_audit(audit: AuditSummary) -> GameSummary:
    status = audit.final_status or "unknown"
    if status == "draw":
        result = "draw"
    elif audit.final_winner and audit.color in {"white", "black"}:
        result = "win" if audit.final_winner == audit.color else "loss"
    else:
        result = status
    return GameSummary(
        audit=audit,
        result=result,
        color=audit.color,
        url=f"https://lichess.org/{audit.game_id}",
    )


def collect_games(
    audit_files: Iterable[pathlib.Path],
    *,
    fetch: bool,
    pause_s: float,
) -> list[GameSummary]:
    games: list[GameSummary] = []
    for path in audit_files:
        audit = parse_audit(path)
        if not fetch:
            games.append(game_summary_from_audit(audit))
            continue
        try:
            data = lichess_export(audit.game_id)
            games.append(game_summary_from_export(audit, data))
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as e:
            fallback = game_summary_from_audit(audit)
            fallback.audit.issue_counts["lichess_fetch_failed"] += 1
            fallback.opening = str(e)
            games.append(fallback)
        if pause_s > 0:
            time.sleep(pause_s)
    return games


def score(games: Iterable[GameSummary]) -> float:
    total = 0.0
    for game in games:
        if game.result == "win":
            total += 1.0
        elif game.result == "draw":
            total += 0.5
    return total


def format_rating(value: int | None) -> str:
    return "?" if value is None else str(value)


def print_report(games: list[GameSummary]) -> None:
    result_counts = Counter(game.result for game in games)
    speed_counts: dict[str, Counter[str]] = {}
    color_counts: dict[str, Counter[str]] = {}
    for game in games:
        speed_counts.setdefault(game.speed, Counter())[game.result] += 1
        color_counts.setdefault(game.color, Counter())[game.result] += 1

    rating_diffs = [game.rating_diff for game in games if game.rating_diff is not None]
    opponent_ratings = [
        game.opponent_rating for game in games if game.opponent_rating is not None
    ]
    accepted = sum(game.audit.accepted_moves for game in games)
    rejected = sum(game.audit.rejected_moves for game in games)
    stale = sum(game.audit.stale_rejects for game in games)
    issues = Counter()
    for game in games:
        issues.update(game.audit.issue_counts)

    print(f"Games: {len(games)}")
    print(f"Score: {score(games):.1f}/{len(games)} ({dict(result_counts)})")
    if rating_diffs:
        print(f"Rating diff total: {sum(rating_diffs):+d}")
    if opponent_ratings:
        avg = sum(opponent_ratings) / len(opponent_ratings)
        print(
            "Opponent rating: "
            f"avg {avg:.1f}, min {min(opponent_ratings)}, max {max(opponent_ratings)}"
        )
    print(f"By speed: {dict((k, dict(v)) for k, v in sorted(speed_counts.items()))}")
    print(f"By color: {dict((k, dict(v)) for k, v in sorted(color_counts.items()))}")
    print(
        "Moves: "
        f"{accepted} accepted, {rejected} rejected "
        f"({stale} stale after game end)"
    )
    print(
        "Search: "
        f"{sum(g.audit.engine_searches for g in games)} engine searches, "
        f"{sum(g.audit.ponderhits for g in games)} ponderhits, "
        f"{sum(g.audit.ponder_starts for g in games)} ponder starts"
    )
    print(
        "Max elapsed: "
        f"search {max((g.audit.max_engine_ms for g in games), default=0)} ms, "
        f"ponderhit {max((g.audit.max_ponderhit_ms for g in games), default=0)} ms"
    )
    print(f"Issues: {dict(issues)}")
    print()
    print("Recent games:")
    for game in games:
        print(
            f"  {game.audit.game_id} "
            f"{game.result:7} {game.color:5} "
            f"vs {game.opponent} ({format_rating(game.opponent_rating)}) "
            f"{game.speed:5} plies={game.plies:<3} "
            f"rdiff={game.rating_diff if game.rating_diff is not None else '?'} "
            f"rejects={game.audit.rejected_moves} "
            f"{game.url}"
        )


def summaries_to_json(games: list[GameSummary]) -> list[dict]:
    return [
        {
            "id": game.audit.game_id,
            "url": game.url,
            "result": game.result,
            "color": game.color,
            "opponent": game.opponent,
            "opponent_rating": game.opponent_rating,
            "our_rating": game.our_rating,
            "rating_diff": game.rating_diff,
            "speed": game.speed,
            "rated": game.rated,
            "plies": game.plies,
            "opening": game.opening,
            "accepted_moves": game.audit.accepted_moves,
            "rejected_moves": game.audit.rejected_moves,
            "stale_rejects": game.audit.stale_rejects,
            "book_moves": game.audit.book_moves,
            "engine_searches": game.audit.engine_searches,
            "ponder_starts": game.audit.ponder_starts,
            "ponderhits": game.audit.ponderhits,
            "max_engine_ms": game.audit.max_engine_ms,
            "max_ponderhit_ms": game.audit.max_ponderhit_ms,
            "issues": dict(game.audit.issue_counts),
        }
        for game in games
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize recent MetalFish Lichess audit logs."
    )
    parser.add_argument("--audit-dir", type=pathlib.Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--fetch-lichess",
        action="store_true",
        help="Fetch public Lichess exports to include result, opponent, and opening data.",
    )
    parser.add_argument(
        "--fetch-pause",
        type=float,
        default=0.1,
        help="Delay between public Lichess export requests.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    files = latest_audit_files(args.audit_dir, max(0, args.limit))
    games = collect_games(files, fetch=args.fetch_lichess, pause_s=args.fetch_pause)
    if args.json:
        print(json.dumps(summaries_to_json(games), indent=2, sort_keys=True))
    else:
        print_report(games)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
