#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import apple_mpsgraph_transformer_bench as bench  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_render_source_substitutes_shape() -> None:
    args = bench.parse_args(
        [
            "--batch",
            "2",
            "--tokens",
            "64",
            "--channels",
            "128",
            "--heads",
            "8",
            "--ffn-mult",
            "4",
            "--layers",
            "3",
            "--warmup",
            "3",
            "--iterations",
            "5",
        ]
    )

    source = bench.render_source(args)

    expect("batch substituted", "constexpr NSUInteger batch = 2;" in source)
    expect("channels substituted", "constexpr NSUInteger channels = 128;" in source)
    expect("heads substituted", "constexpr NSUInteger heads = 8;" in source)
    expect("ffn substituted", "constexpr NSUInteger ffnChannels = 512;" in source)
    expect("layers substituted", "constexpr NSUInteger layers = 3;" in source)


def test_render_source_rejects_non_64_tokens() -> None:
    args = bench.parse_args(["--tokens", "32"])

    try:
        bench.render_source(args)
    except ValueError as exc:
        expect("mentions token limit", "tokens 64" in str(exc))
    else:
        raise AssertionError("non-64 token shape should fail")


def test_render_source_rejects_bad_heads() -> None:
    args = bench.parse_args(["--channels", "130", "--heads", "8"])

    try:
        bench.render_source(args)
    except ValueError as exc:
        expect("mentions divisibility", "divisible" in str(exc))
    else:
        raise AssertionError("non-divisible channels should fail")


def test_render_source_rejects_zero_layers() -> None:
    args = bench.parse_args(["--layers", "0"])

    try:
        bench.render_source(args)
    except ValueError as exc:
        expect("mentions layer floor", "at least 1" in str(exc))
    else:
        raise AssertionError("zero layers should fail")


def main() -> int:
    test_render_source_substitutes_shape()
    test_render_source_rejects_non_64_tokens()
    test_render_source_rejects_bad_heads()
    test_render_source_rejects_zero_layers()
    print("Apple MPSGraph transformer bench tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
