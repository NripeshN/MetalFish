#!/usr/bin/env python3
"""Compare structured UCI smoke search results from two backend gates."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

try:
    from tools.uci_smoke import extract_final_metrics, extract_last_search_info
except ModuleNotFoundError:
    from uci_smoke import extract_final_metrics, extract_last_search_info


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
    transcript_tail = payload.get("transcript_tail") or []
    if isinstance(transcript_tail, list):
        if "search_info" not in payload:
            search_info = extract_last_search_info(transcript_tail)
            if search_info:
                payload["search_info"] = search_info
        if "final_metrics" not in payload:
            final_metrics = extract_final_metrics(transcript_tail)
            if final_metrics:
                payload["final_metrics"] = final_metrics
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
        "--require-same-pv-head",
        action="store_true",
        help="Require both payloads to expose a PV and have the same first move.",
    )
    parser.add_argument(
        "--require-same-pv-prefix",
        type=int,
        default=0,
        help="Require the first N PV moves to match when both payloads expose PVs.",
    )
    parser.add_argument(
        "--max-score-cp-delta",
        type=int,
        default=-1,
        help="Maximum allowed centipawn score delta for parsed cp scores.",
    )
    parser.add_argument(
        "--require-final-metric",
        action="append",
        default=[],
        metavar="NAME",
        help="Require a final metric key to exist in both payloads; may be repeated.",
    )
    parser.add_argument(
        "--require-positive-final-metric",
        action="append",
        default=[],
        metavar="NAME",
        help="Require a final metric key to exist and be positive in both payloads.",
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

    expected_info = expected.get("search_info") or {}
    actual_info = actual.get("search_info") or {}
    expected_pv = expected_info.get("pv") or []
    actual_pv = actual_info.get("pv") or []
    if args.require_same_pv_head or args.require_same_pv_prefix > 0:
        prefix_len = max(1 if args.require_same_pv_head else 0, args.require_same_pv_prefix)
        if len(expected_pv) < prefix_len or len(actual_pv) < prefix_len:
            failures.append(
                f"PV prefix of {prefix_len} move(s) missing: "
                f"{args.expected_label}={expected_pv!r} "
                f"{args.actual_label}={actual_pv!r}"
            )
        elif expected_pv[:prefix_len] != actual_pv[:prefix_len]:
            failures.append(
                f"PV prefix differs: {args.expected_label}={expected_pv[:prefix_len]!r} "
                f"{args.actual_label}={actual_pv[:prefix_len]!r}"
            )

    if args.max_score_cp_delta >= 0:
        expected_score = expected_info.get("score") or {}
        actual_score = actual_info.get("score") or {}
        if expected_score.get("type") != "cp" or actual_score.get("type") != "cp":
            failures.append(
                "cp score missing for comparison: "
                f"{args.expected_label}={expected_score!r} "
                f"{args.actual_label}={actual_score!r}"
            )
        else:
            score_delta = abs(
                int(expected_score["value"]) - int(actual_score["value"])
            )
            if score_delta > args.max_score_cp_delta:
                failures.append(
                    f"cp score delta {score_delta} exceeds {args.max_score_cp_delta}: "
                    f"{args.expected_label}={expected_score['value']} "
                    f"{args.actual_label}={actual_score['value']}"
                )

    expected_metrics = expected.get("final_metrics") or {}
    actual_metrics = actual.get("final_metrics") or {}
    for metric in args.require_final_metric:
        if metric not in expected_metrics or metric not in actual_metrics:
            failures.append(
                f"final metric {metric!r} missing: "
                f"{args.expected_label}={expected_metrics.get(metric)!r} "
                f"{args.actual_label}={actual_metrics.get(metric)!r}"
            )
    for metric in args.require_positive_final_metric:
        expected_value = expected_metrics.get(metric)
        actual_value = actual_metrics.get(metric)
        if not isinstance(expected_value, (int, float)) or expected_value <= 0:
            failures.append(
                f"{args.expected_label} final metric {metric!r} is not positive: "
                f"{expected_value!r}"
            )
        if not isinstance(actual_value, (int, float)) or actual_value <= 0:
            failures.append(
                f"{args.actual_label} final metric {metric!r} is not positive: "
                f"{actual_value!r}"
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
            "search_info": expected_info,
            "final_metrics": expected_metrics,
            "transcript_sha256": expected.get("transcript_sha256"),
        },
        "actual": {
            "path": str(args.actual),
            "bestmove": actual.get("bestmove"),
            "position": actual.get("position"),
            "go": actual.get("go"),
            "elapsed_sec": actual.get("elapsed_sec"),
            "search_info": actual_info,
            "final_metrics": actual_metrics,
            "transcript_sha256": actual.get("transcript_sha256"),
        },
        "require_same_bestmove": require_same_bestmove,
        "require_same_pv_head": args.require_same_pv_head,
        "require_same_pv_prefix": args.require_same_pv_prefix,
        "max_score_cp_delta": args.max_score_cp_delta,
        "require_final_metric": args.require_final_metric,
        "require_positive_final_metric": args.require_positive_final_metric,
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
