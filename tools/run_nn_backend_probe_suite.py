#!/usr/bin/env python3
"""Run the NN backend probe across a fixed parity-position suite."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import subprocess
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProbePosition:
    name: str
    fen: str
    moves: str = ""


DEFAULT_POSITIONS: list[ProbePosition] = [
    ProbePosition(
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    ProbePosition(
        "bk07",
        "1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - - 0 1",
    ),
    ProbePosition(
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2QK2R w KQkq - 0 1",
    ),
    ProbePosition(
        "white-promotion",
        "6bk/P7/8/8/8/8/8/K7 w - - 0 1",
    ),
    ProbePosition(
        "black-promotion",
        "k7/8/8/8/8/8/6p1/KB6 b - - 0 1",
    ),
    ProbePosition(
        "history-repetition",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "g1f3 g8f6 f3g1 f6g8",
    ),
    ProbePosition(
        "canonical-black-to-move",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "e2e4",
    ),
    ProbePosition(
        "castling-history",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 e1g1",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--coreml-model")
    parser.add_argument("--coreml-compute-units", default="cpu-ne")
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--cuda-graph-execution")
    parser.add_argument("--cuda-stable-execution-batch-size", type=int)
    parser.add_argument("--cuda-deterministic-attention-softmax")
    parser.add_argument("--cuda-full-buffer-clear")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--full-policy", action="store_true")
    parser.add_argument("--backend-label")
    parser.add_argument(
        "--require-network-info-substring",
        action="append",
        default=[],
        help="Require each probe network_info field to contain this substring.",
    )
    parser.add_argument("--require-wdl", dest="require_wdl", action="store_true")
    parser.add_argument("--no-require-wdl", dest="require_wdl", action="store_false")
    parser.set_defaults(require_wdl=None)
    parser.add_argument(
        "--require-moves-left",
        dest="require_moves_left",
        action="store_true",
    )
    parser.add_argument(
        "--no-require-moves-left",
        dest="require_moves_left",
        action="store_false",
    )
    parser.set_defaults(require_moves_left=None)
    parser.add_argument("--expected-policy-count", type=int)
    parser.add_argument(
        "--position",
        action="append",
        default=[],
        metavar="NAME=FEN",
        help="Override or extend the suite with a named FEN.",
    )
    parser.add_argument(
        "--line",
        action="append",
        default=[],
        metavar="NAME=FEN|UCI_MOVES",
        help="Override the suite with a named FEN plus UCI move history.",
    )
    return parser.parse_args()


def parse_positions(values: list[str], lines: list[str]) -> list[ProbePosition]:
    if not values and not lines:
        return DEFAULT_POSITIONS

    positions: list[ProbePosition] = []
    for value in values:
        name, separator, fen = value.partition("=")
        if not separator or not name or not fen:
            raise RuntimeError(f"invalid --position value: {value!r}")
        positions.append(ProbePosition(name, fen))
    for value in lines:
        name, separator, payload = value.partition("=")
        fen, move_separator, moves = payload.partition("|")
        if not separator or not move_separator or not name or not fen:
            raise RuntimeError(f"invalid --line value: {value!r}")
        positions.append(ProbePosition(name, fen, moves.strip()))
    return positions


def probe_command(args: argparse.Namespace, position: ProbePosition) -> list[str]:
    command = [
        args.probe,
        "--weights",
        args.weights,
        "--backend",
        args.backend,
        "--fen",
        position.fen,
        "--top",
        str(args.top),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]
    if position.moves:
        command.extend(["--moves", position.moves])
    if args.coreml_model:
        command.extend(["--coreml-model", args.coreml_model])
        command.extend(["--coreml-compute-units", args.coreml_compute_units])
    if args.cuda_device is not None:
        command.extend(["--cuda-device", str(args.cuda_device)])
    if args.cuda_graph_execution is not None:
        command.extend(["--cuda-graph-execution", args.cuda_graph_execution])
    if args.cuda_stable_execution_batch_size is not None:
        command.extend(
            [
                "--cuda-stable-execution-batch-size",
                str(args.cuda_stable_execution_batch_size),
            ]
        )
    if args.cuda_deterministic_attention_softmax is not None:
        command.extend(
            [
                "--cuda-deterministic-attention-softmax",
                args.cuda_deterministic_attention_softmax,
            ]
        )
    if args.cuda_full_buffer_clear is not None:
        command.extend(["--cuda-full-buffer-clear", args.cuda_full_buffer_clear])
    if args.full_policy:
        command.append("--full-policy")
    return command


def subprocess_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def load_probe_jsons(text: str) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            probes.append(payload)
    return probes


def validate_probe(
    args: argparse.Namespace, position: ProbePosition, output: str
) -> None:
    probes = load_probe_jsons(output)
    if not probes:
        raise RuntimeError(f"{position.name}: probe did not emit JSON output")

    probe = probes[-1]
    if args.backend_label:
        network_info = str(probe.get("network_info", ""))
        if args.backend_label not in network_info:
            raise RuntimeError(
                f"{position.name}: expected backend label "
                f"{args.backend_label!r} in network_info={network_info!r}"
            )
    network_info = str(probe.get("network_info", ""))
    for required in args.require_network_info_substring:
        if required not in network_info:
            raise RuntimeError(
                f"{position.name}: expected network_info substring "
                f"{required!r} in network_info={network_info!r}"
            )
    if args.require_wdl is not None and bool(probe.get("has_wdl")) != args.require_wdl:
        expected = "present" if args.require_wdl else "absent"
        actual = "present" if probe.get("has_wdl") else "absent"
        raise RuntimeError(f"{position.name}: expected WDL {expected}, got {actual}")
    if (
        args.require_moves_left is not None
        and bool(probe.get("has_moves_left")) != args.require_moves_left
    ):
        expected = "present" if args.require_moves_left else "absent"
        actual = "present" if probe.get("has_moves_left") else "absent"
        raise RuntimeError(
            f"{position.name}: expected moves-left {expected}, got {actual}"
        )
    if args.expected_policy_count is not None:
        policy = probe.get("policy")
        if not isinstance(policy, list):
            raise RuntimeError(f"{position.name}: probe did not emit full policy")
        if len(policy) != args.expected_policy_count:
            raise RuntimeError(
                f"{position.name}: expected policy length "
                f"{args.expected_policy_count}, got {len(policy)}"
            )


def main() -> int:
    args = parse_args()
    positions = parse_positions(args.position, args.line)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for index, position in enumerate(positions, start=1):
            handle.write(
                f"info string probe-suite {index}/{len(positions)} name={position.name}\n"
            )
            handle.flush()
            try:
                result = subprocess.run(
                    probe_command(args, position),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                handle.write(subprocess_output_text(exc.stdout))
                handle.write(subprocess_output_text(exc.stderr))
                raise RuntimeError(
                    f"{position.name}: probe timed out after {args.timeout}s"
                )

            handle.write(result.stdout)
            handle.write(result.stderr)
            if result.returncode != 0:
                raise RuntimeError(
                    f"{position.name}: probe failed with exit code {result.returncode}"
                )
            validate_probe(args, position, result.stdout + result.stderr)

    print(f"NN backend probe suite: PASS probes={len(positions)} log={output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"NN backend probe suite: FAIL: {exc}")
        raise SystemExit(1)
