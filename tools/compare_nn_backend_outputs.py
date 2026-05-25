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
    parser.add_argument(
        "--require-wdl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require and compare WDL heads. Disable for scalar-value legacy nets.",
    )
    parser.add_argument(
        "--require-moves-left",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require and compare moves-left heads. Disable for legacy nets.",
    )
    parser.add_argument("--require-full-policy", action="store_true")
    parser.add_argument(
        "--all-probes",
        action="store_true",
        help="Compare every JSON probe object in both logs, in order.",
    )
    return parser.parse_args()


def load_probe_jsons(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    last_error: Exception | None = None
    probes: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(data, dict):
            probes.append(data)
    if probes:
        return probes
    if last_error is not None:
        raise RuntimeError(f"{path}: malformed JSON: {last_error}")
    raise RuntimeError(f"{path}: no JSON probe object found")


def load_probe_json(path: str | Path) -> dict[str, Any]:
    probes = load_probe_jsons(path)
    return probes[-1]


def load_selected_probes(path: str | Path, *, all_probes: bool) -> list[dict[str, Any]]:
    probes = load_probe_jsons(path)
    if all_probes:
        return probes
    return [probes[-1]]


def probe_label(data: dict[str, Any], index: int) -> str:
    fen = str(data.get("fen", "<missing-fen>"))
    return f"probe {index + 1} ({fen})"


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


def compare_probe(
    expected: dict[str, Any],
    actual: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
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
    value_delta = abs(finite_number(expected, "value") - finite_number(actual, "value"))
    require(
        value_delta <= args.value_tolerance,
        f"value delta {value_delta:.9g} exceeds {args.value_tolerance}",
    )

    expected_has_wdl = bool(expected.get("has_wdl"))
    actual_has_wdl = bool(actual.get("has_wdl"))
    if args.require_wdl:
        require(expected_has_wdl and actual_has_wdl, "both probes must include WDL")
    else:
        require(
            expected_has_wdl == actual_has_wdl,
            "WDL head presence mismatch",
        )

    wdl_delta: float | None = None
    if expected_has_wdl and actual_has_wdl:
        wdl_delta = max_abs_delta(
            finite_list(expected, "wdl"), finite_list(actual, "wdl")
        )
        require(
            wdl_delta <= args.wdl_tolerance,
            f"WDL delta {wdl_delta:.9g} exceeds {args.wdl_tolerance}",
        )

    expected_has_moves_left = bool(expected.get("has_moves_left"))
    actual_has_moves_left = bool(actual.get("has_moves_left"))
    if args.require_moves_left:
        require(
            expected_has_moves_left and actual_has_moves_left,
            "both probes must include moves-left",
        )
    else:
        require(
            expected_has_moves_left == actual_has_moves_left,
            "moves-left head presence mismatch",
        )

    moves_left_delta: float | None = None
    if expected_has_moves_left and actual_has_moves_left:
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

    return {
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


def max_optional(values: list[float | None]) -> float | None:
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return max(finite_values)


def aggregate_summary(
    probes: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "probe_count": len(probes),
        "probes": probes,
        "max_value_delta": max((probe["value_delta"] for probe in probes), default=0.0),
        "max_wdl_delta": max_optional([probe["wdl_delta"] for probe in probes]),
        "max_moves_left_delta": max_optional(
            [probe["moves_left_delta"] for probe in probes]
        ),
        "max_top_logit_delta": max(
            (probe["max_top_logit_delta"] for probe in probes),
            default=0.0,
        ),
        "top_count": args.top_count,
        "policy_max_delta": max_optional(
            [probe["policy_max_delta"] for probe in probes]
        ),
        "policy_mean_delta": max_optional(
            [probe["policy_mean_delta"] for probe in probes]
        ),
    }
    if len(probes) == 1:
        summary.update(probes[0])
    return summary


def delta_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def main() -> int:
    args = parse_args()
    expected_probes = load_selected_probes(
        args.expected_log,
        all_probes=args.all_probes,
    )
    actual_probes = load_selected_probes(
        args.actual_log,
        all_probes=args.all_probes,
    )
    require(
        len(expected_probes) == len(actual_probes),
        f"probe count mismatch: {len(expected_probes)} != {len(actual_probes)}",
    )

    probe_summaries: list[dict[str, Any]] = []
    for index, (expected, actual) in enumerate(zip(expected_probes, actual_probes)):
        try:
            probe_summaries.append(compare_probe(expected, actual, args))
        except RuntimeError as exc:
            raise RuntimeError(f"{probe_label(expected, index)}: {exc}") from exc

    summary = aggregate_summary(probe_summaries, args)
    if args.summary_out:
        output = Path(args.summary_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    policy_text = ""
    if (
        summary["policy_max_delta"] is not None
        and summary["policy_mean_delta"] is not None
    ):
        policy_text = (
            f" policy_max_delta={summary['policy_max_delta']:.6g}"
            f" policy_mean_delta={summary['policy_mean_delta']:.6g}"
        )
    probe_text = ""
    if len(probe_summaries) > 1:
        probe_text = f"probes={len(probe_summaries)} "
    print(
        "NN backend output compare: PASS "
        f"{probe_text}"
        f"value_delta={summary['max_value_delta']:.6g} "
        f"wdl_delta={delta_text(summary['max_wdl_delta'])} "
        f"moves_left_delta={delta_text(summary['max_moves_left_delta'])} "
        f"top_logit_delta={summary['max_top_logit_delta']:.6g}{policy_text}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"NN backend output compare: FAIL: {exc}")
        raise SystemExit(1)
