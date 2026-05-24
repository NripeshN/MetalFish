#!/usr/bin/env python3
"""Compare two NN backend probe JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-log", required=True)
    parser.add_argument("--actual-log", required=True)
    parser.add_argument("--expected-label")
    parser.add_argument("--actual-label")
    parser.add_argument("--summary-out")
    parser.add_argument("--top-count", type=int, default=3)
    parser.add_argument("--value-tolerance", type=float, default=5e-3)
    parser.add_argument("--wdl-tolerance", type=float, default=5e-3)
    parser.add_argument("--moves-left-tolerance", type=float, default=2e-1)
    parser.add_argument("--policy-logit-tolerance", type=float, default=2e-2)
    parser.add_argument("--policy-max-tolerance", type=float, default=2e-2)
    parser.add_argument("--policy-mean-tolerance", type=float, default=2e-3)
    parser.add_argument("--require-full-policy", action="store_true")
    return parser.parse_args()


def load_probe_json(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    last_error: Exception | None = None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(data, dict):
            return data
    if last_error is not None:
        raise RuntimeError(f"{path}: malformed JSON: {last_error}")
    raise RuntimeError(f"{path}: no JSON probe object found")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    require(isinstance(value, (int, float)), f"{key} is not numeric")
    value = float(value)
    require(math.isfinite(value), f"{key} is not finite")
    return value


def finite_list(data: dict[str, Any], key: str) -> list[float]:
    value = data.get(key)
    require(isinstance(value, list), f"{key} is not a list")
    out: list[float] = []
    for index, entry in enumerate(value):
        require(
            isinstance(entry, (int, float)),
            f"{key}[{index}] is not numeric",
        )
        number = float(entry)
        require(math.isfinite(number), f"{key}[{index}] is not finite")
        out.append(number)
    return out


def max_abs_delta(lhs: list[float], rhs: list[float]) -> float:
    require(len(lhs) == len(rhs), f"length mismatch: {len(lhs)} != {len(rhs)}")
    return max((abs(a - b) for a, b in zip(lhs, rhs)), default=0.0)


def mean_abs_delta(lhs: list[float], rhs: list[float]) -> float:
    require(len(lhs) == len(rhs), f"length mismatch: {len(lhs)} != {len(rhs)}")
    if not lhs:
        return 0.0
    return sum(abs(a - b) for a, b in zip(lhs, rhs)) / len(lhs)


def top_policy(data: dict[str, Any], count: int) -> list[dict[str, Any]]:
    value = data.get("policy_top")
    require(isinstance(value, list), "policy_top is not a list")
    require(len(value) >= count, f"policy_top has fewer than {count} entries")
    out = []
    for index, entry in enumerate(value[:count]):
        require(isinstance(entry, dict), f"policy_top[{index}] is not an object")
        require("move" in entry, f"policy_top[{index}] is missing move")
        require("logit" in entry, f"policy_top[{index}] is missing logit")
        out.append(entry)
    return out


def backend_check(data: dict[str, Any], label: str | None, role: str) -> None:
    if not label:
        return
    network_info = str(data.get("network_info", ""))
    require(
        label in network_info,
        f"{role} backend mismatch: expected {label!r}, got {network_info!r}",
    )


def main() -> int:
    args = parse_args()
    expected = load_probe_json(args.expected_log)
    actual = load_probe_json(args.actual_log)
    backend_check(expected, args.expected_label, "expected")
    backend_check(actual, args.actual_label, "actual")

    require(
        expected.get("fen") == actual.get("fen"),
        f"FEN mismatch: {expected.get('fen')!r} != {actual.get('fen')!r}",
    )
    require(
        expected.get("transform") == actual.get("transform"),
        "policy transform mismatch",
    )
    require(
        expected.get("has_wdl") and actual.get("has_wdl"),
        "both probes must include WDL",
    )
    require(
        expected.get("has_moves_left") and actual.get("has_moves_left"),
        "both probes must include moves-left",
    )

    value_delta = abs(finite_number(expected, "value") -
                      finite_number(actual, "value"))
    require(
        value_delta <= args.value_tolerance,
        f"value delta {value_delta:.9g} exceeds {args.value_tolerance}",
    )

    wdl_delta = max_abs_delta(finite_list(expected, "wdl"),
                              finite_list(actual, "wdl"))
    require(
        wdl_delta <= args.wdl_tolerance,
        f"WDL delta {wdl_delta:.9g} exceeds {args.wdl_tolerance}",
    )

    moves_left_delta = abs(
        finite_number(expected, "moves_left") - finite_number(actual, "moves_left")
    )
    require(
        moves_left_delta <= args.moves_left_tolerance,
        f"moves-left delta {moves_left_delta:.9g} exceeds "
        f"{args.moves_left_tolerance}",
    )

    expected_top = top_policy(expected, args.top_count)
    actual_top = top_policy(actual, args.top_count)
    max_top_logit_delta = 0.0
    for index, (left, right) in enumerate(zip(expected_top, actual_top)):
        require(
            left["move"] == right["move"],
            f"top policy move {index} mismatch: {left['move']} != {right['move']}",
        )
        logit_delta = abs(float(left["logit"]) - float(right["logit"]))
        max_top_logit_delta = max(max_top_logit_delta, logit_delta)
        require(
            logit_delta <= args.policy_logit_tolerance,
            f"top policy logit {index} delta {logit_delta:.9g} exceeds "
            f"{args.policy_logit_tolerance}",
        )

    policy_max_delta: float | None = None
    policy_mean_delta: float | None = None
    if "policy" in expected or "policy" in actual or args.require_full_policy:
        require("policy" in expected, "expected probe is missing full policy")
        require("policy" in actual, "actual probe is missing full policy")
        expected_policy = finite_list(expected, "policy")
        actual_policy = finite_list(actual, "policy")
        policy_max_delta = max_abs_delta(expected_policy, actual_policy)
        policy_mean_delta = mean_abs_delta(expected_policy, actual_policy)
        require(
            policy_max_delta <= args.policy_max_tolerance,
            f"full policy max delta {policy_max_delta:.9g} exceeds "
            f"{args.policy_max_tolerance}",
        )
        require(
            policy_mean_delta <= args.policy_mean_tolerance,
            f"full policy mean delta {policy_mean_delta:.9g} exceeds "
            f"{args.policy_mean_tolerance}",
        )

    summary = {
        "expected_backend": expected.get("backend"),
        "actual_backend": actual.get("backend"),
        "fen": expected.get("fen"),
        "value_delta": value_delta,
        "wdl_delta": wdl_delta,
        "moves_left_delta": moves_left_delta,
        "top_count": args.top_count,
        "max_top_logit_delta": max_top_logit_delta,
        "policy_max_delta": policy_max_delta,
        "policy_mean_delta": policy_mean_delta,
    }
    if args.summary_out:
        output = Path(args.summary_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    policy_text = ""
    if policy_max_delta is not None and policy_mean_delta is not None:
        policy_text = (
            f" policy_max_delta={policy_max_delta:.6g}"
            f" policy_mean_delta={policy_mean_delta:.6g}"
        )
    print(
        "NN backend output compare: PASS "
        f"value_delta={value_delta:.6g} wdl_delta={wdl_delta:.6g} "
        f"moves_left_delta={moves_left_delta:.6g} "
        f"top_logit_delta={max_top_logit_delta:.6g}{policy_text}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"NN backend output compare: FAIL: {exc}")
        raise SystemExit(1)
