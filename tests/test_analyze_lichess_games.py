#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import tempfile
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import analyze_lichess_games as analyzer  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def write_audit(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_parse_audit_counts_stale_rejects_and_ponder_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "abc12345.jsonl"
        write_audit(
            path,
            [
                {"event": "game_start"},
                {"event": "stream_game_full", "status": "started", "color": "white"},
                {
                    "event": "turn_seen",
                    "ply": 12,
                    "wtime": 300000,
                    "btime": 299000,
                    "winc": 3000,
                    "binc": 3000,
                },
                {"event": "book_candidate", "move": "e2e4"},
                {
                    "event": "draw_state",
                    "draw_claim_available": True,
                    "tablebase_wdl": 0,
                },
                {"event": "draw_offer_candidate", "reason": "tb_draw_claim"},
                {
                    "event": "move_submit",
                    "move": "e2e4",
                    "result": "accepted",
                    "offering_draw": True,
                    "draw_offer_reason": "tb_draw_claim",
                },
                {"event": "ponder_start", "best": "e2e4", "ponder": "e7e5"},
                {
                    "event": "ponderhit_result",
                    "best": "g1f3",
                    "ponder": "b8c6",
                    "elapsed_ms": 1234,
                },
                {"event": "engine_search_result", "best": "d2d4", "elapsed_ms": 4321},
                {
                    "event": "move_submit",
                    "move": "d2d4",
                    "result": "rejected",
                    "stale": True,
                    "detail": '400: {"error":"Not your turn, or game already over"}',
                },
                {"event": "ponder_start_rejected"},
                {"event": "stream_game_over", "status": "outoftime", "winner": "white"},
            ],
        )

        summary = analyzer.parse_audit(path)
        game = analyzer.game_summary_from_audit(summary)

    expect("game id from filename", summary.game_id == "abc12345")
    expect("color from full event", summary.color == "white")
    expect("offline max ply inferred", summary.max_ply == 12)
    expect("offline speed inferred", game.speed == "blitz")
    expect("offline plies inferred", game.plies == 12)
    expect("accepted moves", summary.accepted_moves == 1)
    expect("rejected moves", summary.rejected_moves == 1)
    expect("stale rejects", summary.stale_rejects == 1)
    expect("draw state samples", summary.draw_state_samples == 1)
    expect("draw claimable turns", summary.draw_claimable_turns == 1)
    expect("tablebase draw turns", summary.tablebase_draw_turns == 1)
    expect("draw offer candidates", summary.draw_offer_candidates == 1)
    expect("draw offer moves", summary.draw_offer_moves == 1)
    expect("book moves", summary.book_moves == 1)
    expect("ponder starts", summary.ponder_starts == 1)
    expect("ponderhits", summary.ponderhits == 1)
    expect("max ponderhit", summary.max_ponderhit_ms == 1234)
    expect("max engine", summary.max_engine_ms == 4321)
    expect("issue counted", summary.issue_counts["ponder_start_rejected"] == 1)
    expect("final status", summary.final_status == "outoftime")
    expect("offline result from audit winner", game.result == "win")


def test_game_summary_from_export_scores_bot_color() -> None:
    audit = analyzer.AuditSummary(game_id="gameid", path=pathlib.Path("gameid.jsonl"))
    data = {
        "rated": True,
        "speed": "blitz",
        "status": "resign",
        "winner": "black",
        "players": {
            "white": {
                "user": {"id": "opponent", "name": "OpponentBot"},
                "rating": 3020,
            },
            "black": {
                "user": {"id": analyzer.BOT_ID, "name": "Nripesh-MetalFish"},
                "rating": 2800,
                "ratingDiff": 11,
            },
        },
        "opening": {"name": "Sicilian Defense"},
        "moves": "e4 c5 Nf3 d6",
    }

    summary = analyzer.game_summary_from_export(audit, data)

    expect("bot color", summary.color == "black")
    expect("win result", summary.result == "win")
    expect("opponent name", summary.opponent == "OpponentBot")
    expect("opponent rating", summary.opponent_rating == 3020)
    expect("rating diff", summary.rating_diff == 11)
    expect("plies", summary.plies == 4)


def test_infer_speed_from_audit_clock() -> None:
    expect("3+2 blitz", analyzer.infer_speed(180000, 2000) == "blitz")
    expect("10+5 rapid", analyzer.infer_speed(600000, 5000) == "rapid")
    expect("30+20 classical", analyzer.infer_speed(1800000, 20000) == "classical")
    expect("missing speed unknown", analyzer.infer_speed(None, None) == "?")


def test_latest_audit_files_can_filter_by_since_time() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        old = root / "old.jsonl"
        new = root / "new.jsonl"
        write_audit(old, [{"event": "stream_game_over", "status": "draw"}])
        write_audit(new, [{"event": "stream_game_over", "status": "draw"}])
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))

        files = analyzer.latest_audit_files(root, 10, since_ts=1500)

    expect(
        "since filter keeps only new audit",
        [path.name for path in files] == ["new.jsonl"],
    )


def test_parse_since_accepts_unix_and_iso_values() -> None:
    expect("unix since parsed", analyzer.parse_since("1234.5") == 1234.5)
    expect(
        "iso since parsed",
        analyzer.parse_since("1970-01-01T00:00:01+00:00") == 1.0,
    )


def test_print_report_includes_stability_summary() -> None:
    audit = analyzer.AuditSummary(
        game_id="gameid",
        path=pathlib.Path("gameid.jsonl"),
        accepted_moves=10,
        rejected_moves=1,
        stale_rejects=1,
        draw_state_samples=3,
        draw_claimable_turns=2,
        tablebase_draw_turns=1,
        draw_offer_candidates=2,
        draw_offer_moves=1,
        engine_searches=3,
        ponderhits=7,
        ponder_starts=9,
        max_engine_ms=5000,
        max_ponderhit_ms=4000,
    )
    game = analyzer.GameSummary(
        audit=audit,
        result="draw",
        color="white",
        opponent="OpponentBot",
        opponent_rating=3000,
        our_rating=2800,
        rating_diff=5,
        speed="blitz",
        rated=True,
        plies=80,
        url="https://lichess.org/gameid",
    )

    output = io.StringIO()
    with redirect_stdout(output):
        analyzer.print_report([game])
    text = output.getvalue()

    expect("score reported", "Score: 0.5/1" in text)
    expect("stale reject reported", "1 rejected (1 stale after game end)" in text)
    expect(
        "draw telemetry reported",
        "Draw telemetry: 3 samples, 2 claimable, 1 TB-drawn, 2 offer candidates, 1 move requests"
        in text,
    )
    expect("search reported", "3 engine searches, 7 ponderhits" in text)
    expect("game line reported", "OpponentBot" in text)


def main() -> int:
    test_parse_audit_counts_stale_rejects_and_ponder_metrics()
    test_game_summary_from_export_scores_bot_color()
    test_infer_speed_from_audit_clock()
    test_latest_audit_files_can_filter_by_since_time()
    test_parse_since_accepts_unix_and_iso_values()
    test_print_report_includes_stability_summary()
    print("Lichess game analyzer tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
