#!/usr/bin/env python3
from __future__ import annotations

import io
import pathlib
import sys
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.lichess_puzzle_runner as puzzle_runner  # noqa: E402
from tools.lichess_puzzle_runner import (  # noqa: E402
    LichessRateLimited,
    board_from_api_puzzle,
    normalize_move,
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
    test_rate_limit_wait_respects_budget()
    print("Lichess puzzle runner tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
