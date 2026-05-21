#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

import chess
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import lc0_coreml_root_value_probe as probe  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_wdl_value() -> None:
    value = probe.wdl_value(np.array([0.7, 0.2, 0.1], dtype=np.float32))
    expect("wdl value", abs(value - 0.6) < 1e-6)


def test_child_score_is_root_perspective() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    score = probe.score_child_for_root(
        board, move, np.array([0.65, 0.2, 0.15], dtype=np.float32)
    )
    expect("root perspective negates child", abs(score + 0.5) < 1e-6)


def test_rank_bucket() -> None:
    expect("rank inside", probe.rank_bucket(3, 3) == 1)
    expect("rank outside", probe.rank_bucket(4, 3) == 0)
    expect("rank missing", probe.rank_bucket(None, 3) == 0)


def test_parse_args() -> None:
    args = probe.parse_args(
        [
            "networks/t1-512x15x8h-distilled-swa-3395000.pb.gz",
            "--compute-unit",
            "cpu-ne",
            "--batch-size",
            "8",
            "--positions",
            "BK.07",
            "--top",
            "10",
        ]
    )
    expect("weights", args.weights.endswith(".pb.gz"))
    expect("compute unit", args.compute_unit == "cpu-ne")
    expect("batch", args.batch_size == 8)
    expect("positions", args.positions == "BK.07")
    expect("top", args.top == 10)


def main() -> int:
    test_wdl_value()
    test_child_score_is_root_perspective()
    test_rank_bucket()
    test_parse_args()
    print("Lc0 Core ML root value probe tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
