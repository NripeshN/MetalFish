#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import lc0_coreml_position_parity as parity  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_startpos_encoder_shape_and_aux_planes() -> None:
    planes = parity.encode_fen_classical_112(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    expect("shape", planes.shape == (1, 64, 112))
    expect("white pawns", float(planes[0, parity.square_index("a2"), 0]) == 1.0)
    expect("black king", float(planes[0, parity.square_index("e8"), 11]) == 1.0)
    expect("start history empty", float(planes[0, parity.square_index("a2"), 13]) == 0.0)
    expect("our qs castling", np.all(planes[0, :, 104] == 1.0))
    expect("our ks castling", np.all(planes[0, :, 105] == 1.0))
    expect("their qs castling", np.all(planes[0, :, 106] == 1.0))
    expect("their ks castling", np.all(planes[0, :, 107] == 1.0))
    expect("side plane", np.all(planes[0, :, 108] == 0.0))
    expect("ones plane", np.all(planes[0, :, 111] == 1.0))


def test_black_to_move_is_oriented_to_side_to_move() -> None:
    planes = parity.encode_fen_classical_112("8/8/8/8/8/8/p7/K6k b - - 12 1")
    expect("black pawn becomes our pawn", float(planes[0, parity.square_index("a7"), 0]) == 1.0)
    expect("white king becomes their king", float(planes[0, parity.square_index("a8"), 11]) == 1.0)
    expect("black king becomes our king", float(planes[0, parity.square_index("h8"), 5]) == 1.0)
    expect("side plane black", np.all(planes[0, :, 108] == 1.0))
    expect("rule50", np.all(planes[0, :, 109] == 12.0))


def test_policy_moves_loaded() -> None:
    moves = parity.load_policy_moves()
    expect("move count", len(moves) == 1858)
    expect("first move", moves[0] == "a1b1")


def test_parse_args_defaults() -> None:
    args = parity.parse_args(["networks/t1-256x10-distilled-swa-2432500.pb.gz"])
    expect("candidate cpu-ne", args.candidate_compute_unit == "cpu-ne")
    expect("candidate fp16", args.candidate_precision == "fp16")
    expect("value head fp32 default", args.candidate_value_head_fp32)
    expect("policy head fp32 default", not args.candidate_policy_head_fp32)


def main() -> int:
    test_startpos_encoder_shape_and_aux_planes()
    test_black_to_move_is_oriented_to_side_to_move()
    test_policy_moves_loaded()
    test_parse_args_defaults()
    print("Lc0 Core ML position parity tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
