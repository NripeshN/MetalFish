#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

import chess
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import lc0_coreml_bk_policy_probe as probe  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_vertical_mirror_keeps_promotion_suffix() -> None:
    expect("white move", probe.mirror_uci_vertical("h5f6") == "h4f3")
    expect("promotion", probe.mirror_uci_vertical("a7a8q") == "a2a1q")


def test_policy_key_for_black_move_is_mirrored() -> None:
    board = chess.Board("8/8/8/8/8/8/p7/K6k b - - 0 1")
    move = chess.Move.from_uci("a2a1q")
    expect("black key", probe.policy_key_for_move(board, move) == "a7a8q")


def test_legal_policy_ranking() -> None:
    board = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")
    index = {"a1a2": 0, "a1b1": 1}
    policy = np.array([[1.0, 2.0]], dtype=np.float32)
    ranked = probe.legal_policy_ranking(board, policy, index)
    expect("top move", ranked[0]["move"] == "a1b1")
    expect("rank assigned", ranked[0]["rank"] == 1)


def test_best_expected_rank() -> None:
    ranked = [{"move": "a1b1", "rank": 1}, {"move": "a1a2", "rank": 2}]
    expect("rank found", probe.best_expected_rank(ranked, {"a1a2"}) == 2)
    expect("rank missing", probe.best_expected_rank(ranked, {"b1b2"}) is None)


def test_parse_args_defaults() -> None:
    args = probe.parse_args(["net.pb.gz"])
    expect("compute unit", args.compute_unit == "cpu-ne")
    expect("precision", args.precision == "fp16")
    expect("top", args.top == 5)
    expect("positions", args.positions == "all")


def main() -> int:
    test_vertical_mirror_keeps_promotion_suffix()
    test_policy_key_for_black_move_is_mirrored()
    test_legal_policy_ranking()
    test_best_expected_rank()
    test_parse_args_defaults()
    print("Lc0 Core ML BK policy probe tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
