#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import lc0_coreml_value_export as lc0_export

DEFAULT_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - - 0 1",
    "r1bq1rk1/ppp2ppp/2nbpn2/3p4/3P4/2PBPN2/PP3PPP/RNBQ1RK1 b - - 0 7",
]

PIECE_PLANES = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

START_BOARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def square_index(square: str) -> int:
    return ord(square[0]) - ord("a") + 8 * (int(square[1]) - 1)


def mirror_vertical_index(index: int) -> int:
    return index ^ 56


def parse_fen(fen: str) -> dict[str, Any]:
    parts = fen.split()
    if len(parts) < 4:
        raise ValueError(f"FEN must contain at least four fields: {fen}")
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    return {
        "board": parts[0],
        "turn": parts[1],
        "castling": parts[2],
        "ep": parts[3],
        "halfmove": halfmove,
    }


def is_start_position(fen_info: dict[str, Any]) -> bool:
    return (
        fen_info["board"] == START_BOARD
        and fen_info["turn"] == "w"
        and fen_info["castling"] == "KQkq"
        and fen_info["ep"] == "-"
        and fen_info["halfmove"] == 0
    )


def board_planes(fen_info: dict[str, Any]) -> np.ndarray:
    planes = np.zeros((13, 64), dtype=np.float32)
    rank = 7
    file = 0
    for char in fen_info["board"]:
        if char == "/":
            if file != 8:
                raise ValueError("invalid FEN row width")
            rank -= 1
            file = 0
            continue
        if char.isdigit():
            file += int(char)
            continue
        if char not in PIECE_PLANES:
            raise ValueError(f"invalid FEN piece {char}")
        sq = file + 8 * rank
        plane = PIECE_PLANES[char]
        if fen_info["turn"] == "b":
            sq = mirror_vertical_index(sq)
            plane = plane - 6 if plane >= 6 else plane + 6
        planes[plane, sq] = 1.0
        file += 1
    if rank != 0 or file != 8:
        raise ValueError("invalid FEN board")
    return planes


def add_synthetic_ep_pawn(planes: np.ndarray, fen_info: dict[str, Any]) -> None:
    ep = fen_info["ep"]
    if ep == "-":
        return
    ep_idx = square_index(ep)
    if fen_info["turn"] == "b":
        ep_idx = mirror_vertical_index(ep_idx)
    if ep_idx < 8:
        for sq in range(ep_idx + 8, ep_idx + 24, 8):
            planes[0, sq] = 1.0
    elif ep_idx >= 56:
        for sq in range(ep_idx - 24, ep_idx - 8, 8):
            planes[6, sq] = 1.0


def encode_fen_classical_112(fen: str) -> np.ndarray:
    fen_info = parse_fen(fen)
    current = board_planes(fen_info)
    planes = np.zeros((112, 64), dtype=np.float32)
    history_limit = 1 if is_start_position(fen_info) else 8
    for i in range(history_limit):
        board = current.copy()
        if i > 0:
            add_synthetic_ep_pawn(board, fen_info)
        planes[i * 13 : i * 13 + 13, :] = board

    aux = 104
    castling = fen_info["castling"]
    us_black = fen_info["turn"] == "b"
    our_queenside = "q" if us_black else "Q"
    our_kingside = "k" if us_black else "K"
    their_queenside = "Q" if us_black else "q"
    their_kingside = "K" if us_black else "k"
    planes[aux + 0, :] = 1.0 if our_queenside in castling else 0.0
    planes[aux + 1, :] = 1.0 if our_kingside in castling else 0.0
    planes[aux + 2, :] = 1.0 if their_queenside in castling else 0.0
    planes[aux + 3, :] = 1.0 if their_kingside in castling else 0.0
    planes[aux + 4, :] = 1.0 if us_black else 0.0
    planes[aux + 5, :] = float(fen_info["halfmove"])
    planes[aux + 6, :] = 0.0
    planes[aux + 7, :] = 1.0
    return np.transpose(planes, (1, 0))[None, :, :]


def load_policy_moves() -> list[str]:
    text = (ROOT / "src" / "nn" / "policy_map.cpp").read_text(encoding="utf-8")
    match = re.search(
        r"const char \*kMoveStrings\[kPolicyOutputs\] = \{(.*?)\};",
        text,
        re.S,
    )
    if not match:
        raise RuntimeError("could not locate policy move strings")
    moves = re.findall(r'"([^"]+)"', match.group(1))
    if len(moves) != 1858:
        raise RuntimeError(f"expected 1858 policy moves, found {len(moves)}")
    return moves


def build_model(args: argparse.Namespace, output_stage: str, candidate: bool) -> Any:
    np_mod, ct, mb, types = lc0_export.load_coremltools()
    net = lc0_export.load_weights_file(Path(args.weights))
    unit_name = args.candidate_compute_unit if candidate else "cpu"
    precision = args.candidate_precision if candidate else "fp32"
    unit = lc0_export.compute_unit_from_name(ct, unit_name)
    precision_args = argparse.Namespace(
        precision=precision,
        value_head_fp32=candidate and args.candidate_value_head_fp32,
        policy_head_fp32=candidate and args.candidate_policy_head_fp32,
    )
    compute_precision = lc0_export.compute_precision_for_args(precision_args, ct)
    return lc0_export.build_value_model(
        np_mod,
        ct,
        mb,
        types,
        net,
        unit,
        compute_precision,
        output_stage,
        None,
        args.batch_size,
    )


def predict(
    model: Any, planes: np.ndarray, warmup: int, iterations: int
) -> tuple[np.ndarray, float]:
    outputs, latency = predict_outputs(model, planes, warmup, iterations)
    return np.asarray(next(iter(outputs.values())), dtype=np.float32), latency


def predict_outputs(
    model: Any, planes: np.ndarray, warmup: int, iterations: int
) -> tuple[dict[str, Any], float]:
    for _ in range(warmup):
        model.predict({"x": planes})
    latencies: list[float] = []
    output = None
    for _ in range(iterations):
        start = time.perf_counter()
        output = model.predict({"x": planes})
        latencies.append((time.perf_counter() - start) * 1000.0)
    if not output:
        raise RuntimeError("model returned no outputs")
    return output, statistics.median(latencies)


def split_heads(outputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    wdl = None
    policy = None
    for value in outputs.values():
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1] == 3:
            wdl = array
        elif array.shape[-1] == 1858:
            policy = array
    if wdl is None or policy is None:
        shapes = [list(np.asarray(value).shape) for value in outputs.values()]
        raise RuntimeError(
            f"combined model did not return WDL and policy outputs: {shapes}"
        )
    return wdl, policy


def top_policy(
    policy: np.ndarray, moves: list[str], count: int
) -> list[dict[str, Any]]:
    flat = policy.reshape(-1, policy.shape[-1])[0]
    top = np.argsort(-flat)[:count]
    return [
        {
            "index": int(index),
            "move": moves[int(index)],
            "logit": float(flat[int(index)]),
        }
        for index in top
    ]


def compare_arrays(ref: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = np.abs(ref - candidate)
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "rms": float(np.sqrt(np.mean(delta * delta))),
    }


def run_metal_probe(args: argparse.Namespace, fen: str) -> dict[str, Any]:
    if not args.metal_probe:
        return {}
    command = [
        str(Path(args.metal_probe)),
        "--weights",
        str(Path(args.weights)),
        "--fen",
        fen,
        "--top",
        str(args.top),
        "--full-policy",
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--batch-size",
        str(args.batch_size),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(
            f"failed to run Metal probe {args.metal_probe}: {exc}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise RuntimeError(
            f"Metal probe failed with exit code {exc.returncode}: {stderr or exc.stdout.strip()}"
        ) from exc

    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("Metal probe did not emit JSON")


def run(args: argparse.Namespace) -> dict[str, Any]:
    moves = load_policy_moves()
    fens = args.fen or DEFAULT_FENS
    if args.combined_model:
        models = {
            "ref_heads": build_model(args, "heads", False),
            "candidate_heads": build_model(args, "heads", True),
        }
    else:
        models = {
            "ref_wdl": build_model(args, "wdl", False),
            "ref_policy": build_model(args, "policy", False),
            "candidate_wdl": build_model(args, "wdl", True),
            "candidate_policy": build_model(args, "policy", True),
        }
    positions = []
    for fen in fens:
        planes = encode_fen_classical_112(fen)
        if args.batch_size > 1:
            planes = np.repeat(planes, args.batch_size, axis=0)
        if args.combined_model:
            ref_outputs, ref_heads_ms = predict_outputs(
                models["ref_heads"], planes, args.warmup, args.iterations
            )
            cand_outputs, cand_heads_ms = predict_outputs(
                models["candidate_heads"], planes, args.warmup, args.iterations
            )
            ref_wdl, ref_policy = split_heads(ref_outputs)
            cand_wdl, cand_policy = split_heads(cand_outputs)
            ref_wdl_ms = ref_policy_ms = ref_heads_ms
            cand_wdl_ms = cand_policy_ms = cand_heads_ms
        else:
            ref_wdl, ref_wdl_ms = predict(
                models["ref_wdl"], planes, args.warmup, args.iterations
            )
            cand_wdl, cand_wdl_ms = predict(
                models["candidate_wdl"], planes, args.warmup, args.iterations
            )
            ref_policy, ref_policy_ms = predict(
                models["ref_policy"], planes, args.warmup, args.iterations
            )
            cand_policy, cand_policy_ms = predict(
                models["candidate_policy"], planes, args.warmup, args.iterations
            )
        metal = run_metal_probe(args, fen)
        metal_result = None
        if metal:
            metal_wdl = np.asarray(metal["wdl"], dtype=np.float32).reshape(1, 3)
            metal_policy = np.asarray(metal["policy"], dtype=np.float32).reshape(
                1, 1858
            )
            ref_wdl_first = ref_wdl.reshape(-1, 3)[:1]
            cand_wdl_first = cand_wdl.reshape(-1, 3)[:1]
            ref_policy_first = ref_policy.reshape(-1, 1858)[:1]
            cand_policy_first = cand_policy.reshape(-1, 1858)[:1]
            metal_result = {
                "backend": metal["backend"],
                "network_info": metal["network_info"],
                "format": metal["format"],
                "transform": metal["transform"],
                "value": metal["value"],
                "wdl": metal["wdl"],
                "moves_left": metal["moves_left"],
                "latency": metal.get("latency", {}),
                "policy_top": metal["policy_top"],
                "reference_wdl_delta": compare_arrays(metal_wdl, ref_wdl_first),
                "candidate_wdl_delta": compare_arrays(metal_wdl, cand_wdl_first),
                "reference_policy_delta": compare_arrays(
                    metal_policy, ref_policy_first
                ),
                "candidate_policy_delta": compare_arrays(
                    metal_policy, cand_policy_first
                ),
            }
        positions.append(
            {
                "fen": fen,
                "wdl": {
                    "reference": ref_wdl.reshape(-1).tolist(),
                    "candidate": cand_wdl.reshape(-1).tolist(),
                    "delta": compare_arrays(ref_wdl, cand_wdl),
                    "reference_ms": ref_wdl_ms,
                    "candidate_ms": cand_wdl_ms,
                },
                "policy": {
                    "delta": compare_arrays(ref_policy, cand_policy),
                    "reference_ms": ref_policy_ms,
                    "candidate_ms": cand_policy_ms,
                    "reference_top": top_policy(ref_policy, moves, args.top),
                    "candidate_top": top_policy(cand_policy, moves, args.top),
                },
                "metal": metal_result,
            }
        )
    return {
        "weights": str(Path(args.weights).resolve()),
        "reference": {"compute_unit": "cpu", "precision": "fp32"},
        "candidate": {
            "compute_unit": args.candidate_compute_unit,
            "precision": args.candidate_precision,
            "value_head_fp32": args.candidate_value_head_fp32,
            "policy_head_fp32": args.candidate_policy_head_fp32,
        },
        "combined_model": args.combined_model,
        "batch_size": args.batch_size,
        "positions": positions,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Core ML T1 CPU/fp32 and ANE candidate outputs on FENs."
    )
    parser.add_argument("weights", help="Lc0 T1 .pb or .pb.gz weights")
    parser.add_argument("--fen", action="append", default=[])
    parser.add_argument(
        "--candidate-compute-unit",
        choices=["all", "cpu", "cpu-gpu", "cpu-ne"],
        default="cpu-ne",
    )
    parser.add_argument(
        "--candidate-precision", choices=["fp16", "fp32"], default="fp16"
    )
    parser.add_argument(
        "--candidate-value-head-fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--candidate-policy-head-fp32",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--metal-probe",
        default="",
        help="Optional path to build/metalfish_nn_probe for Metal backend parity",
    )
    parser.add_argument(
        "--combined-model",
        action="store_true",
        help="Use one Core ML model that returns WDL and policy together",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Fixed Core ML/Metal batch size"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def print_human(result: dict[str, Any]) -> None:
    print("MetalFish Core ML T1 position parity")
    print(f"  Weights: {result['weights']}")
    print(f"  Candidate: {result['candidate']}")
    if result.get("combined_model"):
        print("  Mode: combined WDL+policy model")
    print(f"  Batch size: {result['batch_size']}")
    for idx, position in enumerate(result["positions"], 1):
        print(f"\nPosition {idx}: {position['fen']}")
        wdl = position["wdl"]
        policy = position["policy"]
        print(
            "  WDL: "
            f"max={wdl['delta']['max_abs']:.7f} mean={wdl['delta']['mean_abs']:.7f} "
            f"ref={wdl['reference_ms']:.3f}ms cand={wdl['candidate_ms']:.3f}ms "
            f"cand_pps={1000.0 * result['batch_size'] / wdl['candidate_ms']:.1f}"
        )
        print(
            "  Policy: "
            f"max={policy['delta']['max_abs']:.7f} mean={policy['delta']['mean_abs']:.7f} "
            f"ref={policy['reference_ms']:.3f}ms cand={policy['candidate_ms']:.3f}ms "
            f"cand_pps={1000.0 * result['batch_size'] / policy['candidate_ms']:.1f}"
        )
        ref_moves = ", ".join(item["move"] for item in policy["reference_top"][:5])
        cand_moves = ", ".join(item["move"] for item in policy["candidate_top"][:5])
        print(f"  Ref top:  {ref_moves}")
        print(f"  Cand top: {cand_moves}")
        if position.get("metal"):
            metal = position["metal"]
            print(
                "  Metal vs ref:  "
                f"WDL max={metal['reference_wdl_delta']['max_abs']:.7f} "
                f"policy max={metal['reference_policy_delta']['max_abs']:.7f}"
            )
            print(
                "  Metal vs cand: "
                f"WDL max={metal['candidate_wdl_delta']['max_abs']:.7f} "
                f"policy max={metal['candidate_policy_delta']['max_abs']:.7f}"
            )
            if metal.get("latency"):
                print(
                    "  Metal latency: "
                    f"{metal['latency']['median_ms']:.3f}ms "
                    f"pps={metal['latency']['median_positions_per_second']:.1f}"
                )
            metal_moves = ", ".join(item["move"] for item in metal["policy_top"][:5])
            print(f"  Metal top: {metal_moves}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
