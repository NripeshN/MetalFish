#!/usr/bin/env python3
"""Compare structured UCI smoke search results from two backend gates."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def load_result(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if payload.get("schema") != "metalfish.uci_smoke_result":
        raise ValueError(f"{path}: not a metalfish.uci_smoke_result payload")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    if not payload.get("bestmove"):
        raise ValueError(f"{path}: missing bestmove")
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True, type=pathlib.Path)
    parser.add_argument("--actual", required=True, type=pathlib.Path)
    parser.add_argument("--expected-label", default="expected")
    parser.add_argument("--actual-label", default="actual")
    parser.add_argument(
        "--no-require-same-bestmove",
        action="store_true",
        help="Only validate payload shape and matching position/go metadata.",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        help="Optional path for a machine-readable comparison summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    expected = load_result(args.expected)
    actual = load_result(args.actual)

    failures: list[str] = []
    for key in ("position", "go"):
        if expected.get(key) != actual.get(key):
            failures.append(
                f"{key} differs: {args.expected_label}={expected.get(key)!r} "
                f"{args.actual_label}={actual.get(key)!r}"
            )

    require_same_bestmove = not args.no_require_same_bestmove
    if require_same_bestmove and expected.get("bestmove") != actual.get("bestmove"):
        failures.append(
            "bestmove differs: "
            f"{args.expected_label}={expected.get('bestmove')} "
            f"{args.actual_label}={actual.get('bestmove')}"
        )

    summary = {
        "schema": "metalfish.uci_search_result_comparison",
        "schema_version": 1,
        "expected_label": args.expected_label,
        "actual_label": args.actual_label,
        "expected": {
            "path": str(args.expected),
            "bestmove": expected.get("bestmove"),
            "position": expected.get("position"),
            "go": expected.get("go"),
            "elapsed_sec": expected.get("elapsed_sec"),
            "transcript_sha256": expected.get("transcript_sha256"),
        },
        "actual": {
            "path": str(args.actual),
            "bestmove": actual.get("bestmove"),
            "position": actual.get("position"),
            "go": actual.get("go"),
            "elapsed_sec": actual.get("elapsed_sec"),
            "transcript_sha256": actual.get("transcript_sha256"),
        },
        "require_same_bestmove": require_same_bestmove,
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        f"{args.actual_label} bestmove={actual.get('bestmove')} "
        f"{args.expected_label} bestmove={expected.get('bestmove')}"
    )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
