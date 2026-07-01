#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
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
DEFAULT_SEEK_AUDIT = PROJ / "results" / "lichess_seek_audit.jsonl"
BOT_ID = "nripesh-metalfish"


@dataclass
class AuditSummary:
    game_id: str
    path: pathlib.Path
    color: str = "?"
    max_ply: int = 0
    initial_clock_ms: int | None = None
    increment_ms: int | None = None
    accepted_moves: int = 0
    rejected_moves: int = 0
    stale_rejects: int = 0
    draw_offer_candidates: int = 0
    draw_offer_moves: int = 0
    draw_offers_received: int = 0
    draw_offers_accepted: int = 0
    draw_state_samples: int = 0
    draw_claimable_turns: int = 0
    tablebase_draw_turns: int = 0
    book_moves: int = 0
    engine_searches: int = 0
    ponder_starts: int = 0
    ponderhits: int = 0
    max_engine_ms: int = 0
    max_ponderhit_ms: int = 0
    max_move_submit_ms: int = 0
    stream_errors: int = 0
    stream_failures: int = 0
    stream_reconnects: int = 0
    max_score_cp: int | None = None
    max_score_ply: int = 0
    max_score_move: str = ""
    min_score_cp: int | None = None
    min_score_ply: int = 0
    min_score_move: str = ""
    decisive_score_samples: int = 0
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


@dataclass
class SeekSummary:
    path: pathlib.Path
    events: int = 0
    challenge_sent: int = 0
    challenge_failed: int = 0
    challenge_rate_limited: int = 0
    challenge_timeout: int = 0
    challenge_events: int = 0
    game_started: int = 0
    game_start_aborted: int = 0
    no_candidates: int = 0
    deferred: int = 0
    total_retry_s: float = 0.0
    total_cooldown_s: float = 0.0
    event_counts: Counter[str] = field(default_factory=Counter)
    target_counts: Counter[str] = field(default_factory=Counter)
    speed_counts: Counter[str] = field(default_factory=Counter)
    reason_counts: Counter[str] = field(default_factory=Counter)
    status_counts: Counter[str] = field(default_factory=Counter)


def parse_since(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    return parsed.timestamp()


def latest_audit_files(
    audit_dir: pathlib.Path, limit: int, since_ts: float | None = None
) -> list[pathlib.Path]:
    files = list(audit_dir.glob("*.jsonl"))
    if since_ts is not None:
        files = [path for path in files if path.stat().st_mtime >= since_ts]
    files = sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit]


def parse_audit(path: pathlib.Path) -> AuditSummary:
    summary = AuditSummary(game_id=path.stem, path=path)

    def record_engine_score(row: dict) -> None:
        engine_info = row.get("engine_info")
        if not isinstance(engine_info, dict):
            return
        if engine_info.get("score_mate") is not None:
            try:
                mate = int(engine_info["score_mate"])
            except (TypeError, ValueError):
                return
            score_cp = 32000 if mate > 0 else -32000
        else:
            try:
                score_cp = int(engine_info["score_cp"])
            except (KeyError, TypeError, ValueError):
                return
        try:
            ply = int(row.get("ply") or 0)
        except (TypeError, ValueError):
            ply = 0
        move = str(row.get("best") or "")
        if summary.max_score_cp is None or score_cp > summary.max_score_cp:
            summary.max_score_cp = score_cp
            summary.max_score_ply = ply
            summary.max_score_move = move
        if summary.min_score_cp is None or score_cp < summary.min_score_cp:
            summary.min_score_cp = score_cp
            summary.min_score_ply = ply
            summary.min_score_move = move
        if abs(score_cp) >= 300:
            summary.decisive_score_samples += 1

    for raw in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            summary.issue_counts["audit_json_decode"] += 1
            continue

        event = str(row.get("event", ""))
        try:
            summary.max_ply = max(summary.max_ply, int(row.get("ply") or 0))
        except (TypeError, ValueError):
            pass
        if summary.initial_clock_ms is None:
            clocks = [
                value
                for value in (row.get("wtime"), row.get("btime"))
                if isinstance(value, int) and value > 0
            ]
            if clocks:
                summary.initial_clock_ms = max(clocks)
        if summary.increment_ms is None:
            increments = [
                value
                for value in (row.get("winc"), row.get("binc"))
                if isinstance(value, int) and value >= 0
            ]
            if increments:
                summary.increment_ms = max(increments)

        if event == "stream_game_full":
            summary.color = str(row.get("color") or summary.color)
        elif event == "move_submit":
            if row.get("offering_draw"):
                summary.draw_offer_moves += 1
            summary.max_move_submit_ms = max(
                summary.max_move_submit_ms, int(row.get("elapsed_ms") or 0)
            )
            if row.get("result") == "accepted":
                summary.accepted_moves += 1
            else:
                summary.rejected_moves += 1
                if row.get("stale"):
                    summary.stale_rejects += 1
                else:
                    summary.issue_counts["move_rejected"] += 1
        elif event == "draw_offer_received":
            summary.draw_offers_received += 1
            if row.get("accepted"):
                summary.draw_offers_accepted += 1
        elif event == "book_candidate":
            summary.book_moves += 1
        elif event == "draw_state":
            summary.draw_state_samples += 1
            if row.get("draw_claim_available"):
                summary.draw_claimable_turns += 1
            if row.get("tablebase_wdl") == 0:
                summary.tablebase_draw_turns += 1
        elif event == "draw_offer_candidate":
            summary.draw_offer_candidates += 1
        elif event == "engine_search_result":
            summary.engine_searches += 1
            record_engine_score(row)
            summary.max_engine_ms = max(
                summary.max_engine_ms, int(row.get("elapsed_ms") or 0)
            )
        elif event == "ponder_start":
            summary.ponder_starts += 1
        elif event == "ponderhit_result":
            summary.ponderhits += 1
            record_engine_score(row)
            summary.max_ponderhit_ms = max(
                summary.max_ponderhit_ms, int(row.get("elapsed_ms") or 0)
            )
        elif event == "stream_error":
            summary.stream_errors += 1
        elif event == "stream_failed":
            summary.stream_failures += 1
        elif event == "stream_reconnect":
            summary.stream_reconnects += 1
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
    if status in {
        "draw",
        "stalemate",
        "insufficientMaterialClaim",
        "threefoldRepetition",
        "fivefoldRepetition",
        "fiftyMoveRule",
        "seventyfiveMoveRule",
    }:
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
        speed=infer_speed(audit.initial_clock_ms, audit.increment_ms),
        plies=audit.max_ply,
        url=f"https://lichess.org/{audit.game_id}",
    )


def infer_speed(initial_clock_ms: int | None, increment_ms: int | None) -> str:
    if initial_clock_ms is None:
        return "?"
    estimated_seconds = initial_clock_ms / 1000.0
    if increment_ms is not None:
        estimated_seconds += 40.0 * increment_ms / 1000.0
    if estimated_seconds < 29:
        return "ultraBullet"
    if estimated_seconds < 179:
        return "bullet"
    if estimated_seconds < 479:
        return "blitz"
    if estimated_seconds < 1499:
        return "rapid"
    return "classical"


def _float_field(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_seek_audit(path: pathlib.Path, since_ts: float | None = None) -> SeekSummary:
    summary = SeekSummary(path=path)
    if not path.exists():
        return summary

    for raw in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            summary.event_counts["seek_audit_json_decode"] += 1
            continue
        if since_ts is not None:
            ts = _float_field(row, "ts")
            if ts and ts < since_ts:
                continue

        event = str(row.get("event", ""))
        if not event:
            continue
        summary.events += 1
        summary.event_counts[event] += 1
        if event == "challenge_sent":
            summary.challenge_sent += 1
        elif event == "challenge_failed":
            summary.challenge_failed += 1
        elif event == "challenge_rate_limited":
            summary.challenge_rate_limited += 1
        elif event == "challenge_timeout":
            summary.challenge_timeout += 1
        elif event.startswith("challenge_event"):
            summary.challenge_events += 1
        elif event == "game_started":
            summary.game_started += 1
        elif event == "game_start_aborted":
            summary.game_start_aborted += 1
        elif event == "seek_no_candidates":
            summary.no_candidates += 1
        elif event == "seek_deferred":
            summary.deferred += 1

        target = str(row.get("target") or "")
        if target:
            summary.target_counts[target] += 1
        speed = str(row.get("speed") or "")
        if speed:
            summary.speed_counts[speed] += 1
        reason = str(row.get("reason") or "")
        if reason:
            summary.reason_counts[reason] += 1
        status = str(row.get("status") or "")
        if status:
            summary.status_counts[status] += 1
        summary.total_retry_s += _float_field(row, "retry_s")
        summary.total_cooldown_s += _float_field(row, "cooldown_s")
    return summary


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
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as e:
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
    draw_candidates = sum(g.audit.draw_offer_candidates for g in games)
    draw_moves = sum(g.audit.draw_offer_moves for g in games)
    draw_received = sum(g.audit.draw_offers_received for g in games)
    draw_accepted = sum(g.audit.draw_offers_accepted for g in games)
    draw_samples = sum(g.audit.draw_state_samples for g in games)
    draw_claims = sum(g.audit.draw_claimable_turns for g in games)
    tb_draws = sum(g.audit.tablebase_draw_turns for g in games)
    if draw_candidates or draw_moves or draw_samples or draw_received:
        print(
            "Draw telemetry: "
            f"{draw_samples} samples, {draw_claims} claimable, "
            f"{tb_draws} TB-drawn, {draw_candidates} offer candidates, "
            f"{draw_moves} move requests, "
            f"{draw_received} received, {draw_accepted} accepted"
        )
    stream_errors = sum(g.audit.stream_errors for g in games)
    stream_failures = sum(g.audit.stream_failures for g in games)
    stream_reconnects = sum(g.audit.stream_reconnects for g in games)
    if stream_errors or stream_failures or stream_reconnects:
        print(
            "Stream telemetry: "
            f"{stream_errors} errors, {stream_failures} failed responses, "
            f"{stream_reconnects} reconnects"
        )
    scored_games = [game for game in games if game.audit.max_score_cp is not None]
    if scored_games:
        max_game = max(scored_games, key=lambda game: game.audit.max_score_cp or 0)
        min_game = min(scored_games, key=lambda game: game.audit.min_score_cp or 0)
        decisive_samples = sum(game.audit.decisive_score_samples for game in games)
        print(
            "Score extremes: "
            f"best {max_game.audit.max_score_cp:+d} cp "
            f"({max_game.audit.game_id} ply {max_game.audit.max_score_ply} "
            f"{max_game.audit.max_score_move}), "
            f"worst {min_game.audit.min_score_cp:+d} cp "
            f"({min_game.audit.game_id} ply {min_game.audit.min_score_ply} "
            f"{min_game.audit.min_score_move}), "
            f"{decisive_samples} decisive samples"
        )
    print(
        "Max elapsed: "
        f"search {max((g.audit.max_engine_ms for g in games), default=0)} ms, "
        f"ponderhit {max((g.audit.max_ponderhit_ms for g in games), default=0)} ms, "
        f"submit {max((g.audit.max_move_submit_ms for g in games), default=0)} ms"
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


def print_seek_report(summary: SeekSummary) -> None:
    print()
    print(f"Seek audit: {summary.path}")
    print(f"Seek events: {summary.events}")
    print(
        "Challenges: "
        f"{summary.challenge_sent} sent, "
        f"{summary.challenge_failed} failed, "
        f"{summary.challenge_timeout} timed out, "
        f"{summary.challenge_rate_limited} rate-limited, "
        f"{summary.challenge_events} async events"
    )
    print(
        "Game starts: "
        f"{summary.game_started} started, "
        f"{summary.game_start_aborted} aborted before play"
    )
    print(
        "Seek waits: "
        f"{summary.no_candidates} no-candidate cycles, "
        f"{summary.deferred} resource deferrals, "
        f"{summary.total_retry_s:.0f}s retry delay, "
        f"{summary.total_cooldown_s:.0f}s new cooldowns"
    )
    print(f"By event: {dict(summary.event_counts)}")
    print(f"By speed: {dict(summary.speed_counts)}")
    print(f"Top targets: {dict(summary.target_counts.most_common(10))}")
    print(f"Statuses: {dict(summary.status_counts)}")
    print(f"Top reasons: {dict(summary.reason_counts.most_common(5))}")


def summaries_to_json(games: list[GameSummary]) -> list[dict]:
    return [
        {
            "id": game.audit.game_id,
            "url": game.url,
            "result": game.result,
            "color": game.color,
            "max_ply": game.audit.max_ply,
            "initial_clock_ms": game.audit.initial_clock_ms,
            "increment_ms": game.audit.increment_ms,
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
            "draw_offer_candidates": game.audit.draw_offer_candidates,
            "draw_offer_moves": game.audit.draw_offer_moves,
            "draw_offers_received": game.audit.draw_offers_received,
            "draw_offers_accepted": game.audit.draw_offers_accepted,
            "draw_state_samples": game.audit.draw_state_samples,
            "draw_claimable_turns": game.audit.draw_claimable_turns,
            "tablebase_draw_turns": game.audit.tablebase_draw_turns,
            "book_moves": game.audit.book_moves,
            "engine_searches": game.audit.engine_searches,
            "ponder_starts": game.audit.ponder_starts,
            "ponderhits": game.audit.ponderhits,
            "max_engine_ms": game.audit.max_engine_ms,
            "max_ponderhit_ms": game.audit.max_ponderhit_ms,
            "max_move_submit_ms": game.audit.max_move_submit_ms,
            "stream_errors": game.audit.stream_errors,
            "stream_failures": game.audit.stream_failures,
            "stream_reconnects": game.audit.stream_reconnects,
            "max_score_cp": game.audit.max_score_cp,
            "max_score_ply": game.audit.max_score_ply,
            "max_score_move": game.audit.max_score_move,
            "min_score_cp": game.audit.min_score_cp,
            "min_score_ply": game.audit.min_score_ply,
            "min_score_move": game.audit.min_score_move,
            "decisive_score_samples": game.audit.decisive_score_samples,
            "issues": dict(game.audit.issue_counts),
        }
        for game in games
    ]


def seek_summary_to_json(summary: SeekSummary) -> dict:
    return {
        "path": str(summary.path),
        "events": summary.events,
        "challenge_sent": summary.challenge_sent,
        "challenge_failed": summary.challenge_failed,
        "challenge_rate_limited": summary.challenge_rate_limited,
        "challenge_timeout": summary.challenge_timeout,
        "challenge_events": summary.challenge_events,
        "game_started": summary.game_started,
        "game_start_aborted": summary.game_start_aborted,
        "no_candidates": summary.no_candidates,
        "deferred": summary.deferred,
        "total_retry_s": summary.total_retry_s,
        "total_cooldown_s": summary.total_cooldown_s,
        "event_counts": dict(summary.event_counts),
        "target_counts": dict(summary.target_counts),
        "speed_counts": dict(summary.speed_counts),
        "reason_counts": dict(summary.reason_counts),
        "status_counts": dict(summary.status_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize recent MetalFish Lichess audit logs."
    )
    parser.add_argument("--audit-dir", type=pathlib.Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--since",
        help=(
            "Only include audit files modified at or after this Unix timestamp "
            "or ISO-8601 time."
        ),
    )
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
    parser.add_argument(
        "--seek-audit",
        type=pathlib.Path,
        default=None,
        help=(
            "Also summarize the bot seek audit JSONL file "
            f"(default path: {DEFAULT_SEEK_AUDIT})."
        ),
    )
    parser.add_argument(
        "--seek-only",
        action="store_true",
        help="Only print the seek audit summary.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    try:
        since_ts = parse_since(args.since)
    except ValueError as exc:
        parser.error(f"invalid --since value: {exc}")

    seek_summary = (
        parse_seek_audit(args.seek_audit or DEFAULT_SEEK_AUDIT, since_ts=since_ts)
        if args.seek_audit or args.seek_only
        else None
    )
    files = []
    games: list[GameSummary] = []
    if not args.seek_only:
        files = latest_audit_files(
            args.audit_dir, max(0, args.limit), since_ts=since_ts
        )
        games = collect_games(files, fetch=args.fetch_lichess, pause_s=args.fetch_pause)
    if args.json:
        payload: object
        if seek_summary is not None:
            payload = {
                "games": summaries_to_json(games),
                "seek": seek_summary_to_json(seek_summary),
            }
        else:
            payload = summaries_to_json(games)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if not args.seek_only:
            print_report(games)
        if seek_summary is not None:
            print_seek_report(seek_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
