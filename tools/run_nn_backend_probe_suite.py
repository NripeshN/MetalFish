#!/usr/bin/env python3
"""Run the NN backend probe across a fixed parity-position suite."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_POSITIONS: list[tuple[str, str]] = [
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "bk07",
        "1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - - 0 1",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2QK2R w KQkq - 0 1",
    ),
    (
        "white-promotion",
        "6bk/P7/8/8/8/8/8/K7 w - - 0 1",
    ),
    (
        "black-promotion",
        "k7/8/8/8/8/8/6p1/KB6 b - - 0 1",
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
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--full-policy", action="store_true")
    parser.add_argument(
        "--position",
        action="append",
        default=[],
        metavar="NAME=FEN",
        help="Override or extend the suite with a named FEN.",
    )
    return parser.parse_args()


def parse_positions(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_POSITIONS

    positions: list[tuple[str, str]] = []
    for value in values:
        name, separator, fen = value.partition("=")
        if not separator or not name or not fen:
            raise RuntimeError(f"invalid --position value: {value!r}")
        positions.append((name, fen))
    return positions


def probe_command(args: argparse.Namespace, fen: str) -> list[str]:
    command = [
        args.probe,
        "--weights",
        args.weights,
        "--backend",
        args.backend,
        "--fen",
        fen,
        "--top",
        str(args.top),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]
    if args.coreml_model:
        command.extend(["--coreml-model", args.coreml_model])
        command.extend(["--coreml-compute-units", args.coreml_compute_units])
    if args.full_policy:
        command.append("--full-policy")
    return command


def subprocess_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def main() -> int:
    args = parse_args()
    positions = parse_positions(args.position)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for index, (name, fen) in enumerate(positions, start=1):
            handle.write(
                f"info string probe-suite {index}/{len(positions)} name={name}\n"
            )
            handle.flush()
            try:
                result = subprocess.run(
                    probe_command(args, fen),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                handle.write(subprocess_output_text(exc.stdout))
                handle.write(subprocess_output_text(exc.stderr))
                raise RuntimeError(f"{name}: probe timed out after {args.timeout}s")

            handle.write(result.stdout)
            handle.write(result.stderr)
            if result.returncode != 0:
                raise RuntimeError(
                    f"{name}: probe failed with exit code {result.returncode}"
                )

    print(f"NN backend probe suite: PASS probes={len(positions)} log={output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"NN backend probe suite: FAIL: {exc}")
        raise SystemExit(1)
