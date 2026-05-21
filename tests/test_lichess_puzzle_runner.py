#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import pathlib
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout

import chess

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.lichess_puzzle_runner as puzzle_runner  # noqa: E402
import tools.compare_puzzle_runs as compare_puzzle_runs  # noqa: E402
from tools.lichess_puzzle_runner import (  # noqa: E402
    LichessRateLimited,
    board_from_csv_puzzle,
    csv_puzzle_item,
    csv_row_matches,
    board_from_api_puzzle,
    normalize_move,
    parse_setoptions,
    parse_theme_filter,
    parse_auto_int,
    wait_after_rate_limit,
)


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_batch_puzzle_position_applies_initial_ply_plus_one() -> None:
    item = {
        "game": {
            "pgn": (
                "d4 d6 c4 Nd7 e3 e5 Nf3 h6 Bd3 Ngf6 Bc2 Be7 O-O c6 "
                "Ne1 Nf8 g3 g5 dxe5 dxe5 Qxd8+ Bxd8 c5 Bh3 Ng2 h5 "
                "Nd2 h4 Nf3 hxg3 fxg3 e4 Nxg5 Bxg2 Kxg2 Ne6 Nxe4 "
                "Nxe4 Bxe4 Nxc5 Bc2 Be7 b4 Ne6 Bb3 Bxb4 Bb2 Rh5 "
                "Bd1 Rd5 a3 Bc5 e4"
            )
        },
        "puzzle": {
            "id": "6ewuj",
            "initialPly": 52,
            "solution": ["d5d2", "g2h1", "d2b2"],
        },
    }

    board = board_from_api_puzzle(item)

    expect("puzzle side to move is black", not board.turn)
    expect("first solution move is legal", normalize_move("d5d2", board) == "d5d2")


def test_csv_puzzle_position_applies_opponent_move() -> None:
    row = {
        "PuzzleId": "00sHx",
        "FEN": "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17",
        "Moves": "e8d7 a2e6 d7d8 f7f8",
        "Rating": "1760",
        "RatingDeviation": "80",
        "Popularity": "83",
        "NbPlays": "72",
        "Themes": "mate mateIn2 middlegame short",
        "GameUrl": "https://lichess.org/yyznGmXs/black#34",
        "OpeningTags": "Italian_Game Italian_Game_Classical_Variation",
    }

    item = csv_puzzle_item(row)
    board = board_from_csv_puzzle(item)

    expect("CSV puzzle applies opponent move", board.turn == chess.WHITE)
    expect("CSV solution starts after opponent move", item["puzzle"]["solution"][0] == "a2e6")
    expect("CSV first solution move is legal", normalize_move("a2e6", board) == "a2e6")


def test_csv_filter_and_setoption_parsing() -> None:
    row = {
        "Rating": "1760",
        "Popularity": "83",
        "Themes": "mate mateIn2 middlegame short",
    }
    expect(
        "CSV row matches tactical filters",
        csv_row_matches(
            row,
            min_rating=1200,
            max_rating=2200,
            min_popularity=70,
            themes=parse_theme_filter("fork,mate"),
        ),
    )
    expect(
        "CSV row rejects missing theme",
        not csv_row_matches(
            row,
            min_rating=1200,
            max_rating=2200,
            min_popularity=70,
            themes=parse_theme_filter("endgame"),
        ),
    )
    expect(
        "setoptions parse NAME=VALUE",
        parse_setoptions(["HybridANERootProbe=true", "Threads=8"]) == {
            "HybridANERootProbe": "true",
            "Threads": "8",
        },
    )
    expect("auto parser maps auto to zero", parse_auto_int("auto", option_name="--threads") == 0)
    expect("auto parser keeps integers", parse_auto_int("8", option_name="--threads") == 8)


def test_compare_puzzle_runs_detects_regression() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        baseline = root / "baseline.jsonl"
        candidate = root / "candidate.jsonl"
        baseline.write_text(
            "\n".join(
                json.dumps({"id": puzzle_id, "solved": True})
                for puzzle_id in ("a", "b", "c")
            )
            + "\n"
        )
        candidate.write_text(
            "\n".join(
                [
                    json.dumps({"id": "a", "solved": True}),
                    json.dumps({"id": "b", "solved": False}),
                    json.dumps({"id": "c", "solved": False}),
                ]
            )
            + "\n"
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            ok = compare_puzzle_runs.run(
                compare_puzzle_runs.parse_args(
                    [
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--max-solved-drop",
                        "2",
                        "--max-accuracy-drop",
                        "1.0",
                    ]
                )
            )
            bad = compare_puzzle_runs.run(
                compare_puzzle_runs.parse_args(
                    [
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--max-solved-drop",
                        "1",
                        "--max-accuracy-drop",
                        "1.0",
                    ]
                )
            )
    expect("puzzle compare allows configured drop", ok == 0)
    expect("puzzle compare fails excessive drop", bad == 1)


def test_rate_limit_wait_respects_budget() -> None:
    old_monotonic = puzzle_runner.time.monotonic
    old_sleep = puzzle_runner.time.sleep
    sleeps: list[float] = []

    try:
        puzzle_runner.time.monotonic = lambda: 100.0
        puzzle_runner.time.sleep = lambda seconds: sleeps.append(seconds)

        with redirect_stdout(io.StringIO()):
            waited = wait_after_rate_limit(
                LichessRateLimited(65.0),
                deadline=300.0,
                events_seen=1,
                max_events=5,
            )
        expect("rate limit waits when budget remains", waited)
        expect("rate limit slept requested interval", sleeps == [65.0])

        sleeps.clear()
        expect(
            "rate limit stops when event budget is exhausted",
            not wait_after_rate_limit(
                LichessRateLimited(65.0),
                deadline=300.0,
                events_seen=5,
                max_events=5,
            ),
        )
        expect("exhausted rate limit did not sleep", sleeps == [])
    finally:
        puzzle_runner.time.monotonic = old_monotonic
        puzzle_runner.time.sleep = old_sleep


def main() -> int:
    test_batch_puzzle_position_applies_initial_ply_plus_one()
    test_csv_puzzle_position_applies_opponent_move()
    test_csv_filter_and_setoption_parsing()
    test_compare_puzzle_runs_detects_regression()
    test_rate_limit_wait_respects_budget()
    print("Lichess puzzle runner tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
