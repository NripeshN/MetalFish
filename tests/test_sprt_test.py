#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

import chess

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import sprt_test  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


class FakeEngine:
    def __init__(self, scores: list[int]):
        self.scores = list(scores)
        self.last_score_cp: int | None = None

    def go(self, position: str, movetime_ms: int = 0) -> str:
        del movetime_ms
        return self._move(position)

    def go_tc(self, position: str, wtime: int, btime: int, winc: int = 0, binc: int = 0) -> str:
        del wtime, btime, winc, binc
        return self._move(position)

    def _move(self, position: str) -> str:
        moves = position.removeprefix("startpos moves ").strip()
        board = chess.Board()
        if moves and moves != "startpos":
            for move in moves.split():
                board.push(chess.Move.from_uci(move))
        self.last_score_cp = self.scores.pop(0) if self.scores else 0
        return next(iter(board.legal_moves)).uci()


def test_parse_last_score_cp() -> None:
    expect(
        "cp score parsed",
        sprt_test.parse_last_score_cp(["info depth 1 score cp -42 nodes 1", "bestmove e2e4"])
        == -42,
    )
    expect(
        "mate score converted",
        sprt_test.parse_last_score_cp(["info depth 1 score mate 3 nodes 1"]) == 31997,
    )
    expect(
        "latest score wins",
        sprt_test.parse_last_score_cp(
            [
                "info depth 1 score cp 10 nodes 1",
                "info depth 2 score cp 25 nodes 2",
                "bestmove e2e4",
            ]
        )
        == 25,
    )


def test_play_game_resign_adjudication() -> None:
    white = FakeEngine([1500, 1500])
    black = FakeEngine([-1500, -1500])
    result = sprt_test.play_game(
        white,
        black,
        "",
        movetime_ms=1,
        resign_score=1000,
        resign_count=3,
        draw_move=40,
    )
    expect("black resigns after repeated losing scores", result == "1-0")


def test_play_game_draw_adjudication() -> None:
    white = FakeEngine([0, 0, 0])
    black = FakeEngine([0, 0, 0])
    result = sprt_test.play_game(
        white,
        black,
        "",
        movetime_ms=1,
        resign_score=1000,
        resign_count=3,
        draw_move=1,
        draw_count=2,
        draw_score=10,
    )
    expect("stable equal scores adjudicate draw", result == "1/2-1/2")


def main() -> int:
    test_parse_last_score_cp()
    test_play_game_resign_adjudication()
    test_play_game_draw_adjudication()
    print("test_sprt_test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
