#!/usr/bin/env python3
"""Validate NN backend parity/probe artifacts and write a compact manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-label", required=True)
    parser.add_argument("--parity-report", required=True)
    parser.add_argument("--probe-log", required=True)
    parser.add_argument("--comparison-log")
    parser.add_argument("--manifest-out")
    parser.add_argument("--min-policy-top", type=int, default=1)
    parser.add_argument("--require-batch-benchmark", action="store_true")
    parser.add_argument("--allow-missing-wdl", action="store_true")
    parser.add_argument("--allow-missing-moves-left", action="store_true")
    return parser.parse_args()


def read_text(path: str | Path) -> str:
    file_path = Path(path)
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing artifact: {file_path}") from exc


def find_prefixed_line(lines: str | Iterable[str], prefix: str) -> str | None:
    if isinstance(lines, str):
        line_iter = lines.splitlines()
    else:
        line_iter = lines
    for line in line_iter:
        if line.startswith(prefix):
            return line
    return None


def load_probe_json(path: str | Path) -> dict[str, Any]:
    text = read_text(path)
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
        raise RuntimeError(f"probe log has malformed JSON: {last_error}")
    raise RuntimeError("probe log does not contain a JSON object")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def extract_executor(network_info: str) -> str | None:
    match = re.search(r"executor=([^,()]+(?:\([^)]*\))?)", network_info)
    if not match:
        return None
    return match.group(1).strip()


def main() -> int:
    args = parse_args()
    backend_label = args.backend_label

    parity_text = read_text(args.parity_report)
    probe_text = read_text(args.probe_log)
    backend_line = find_prefixed_line(parity_text, "- Backend:")
    require(backend_line is not None, "parity report is missing backend line")
    require(
        backend_label in backend_line,
        f"parity backend mismatch: expected {backend_label!r}, got {backend_line!r}",
    )

    batch_line = None
    warmup_line = None
    comparison_backend_line = None
    comparison_backend_after_line = None
    comparison_executor_before = None
    comparison_executor_after = None
    comparison_profile_enabled = False
    comparison_backend_found = False
    require(
        bool(args.comparison_log) or not args.require_batch_benchmark,
        "--require-batch-benchmark requires --comparison-log",
    )
    if args.comparison_log:
        comparison_text = read_text(args.comparison_log)
        stripped_comparison_lines = [
            line.strip() for line in comparison_text.splitlines()
        ]
        comparison_backend_found = backend_label in comparison_text
        require(
            comparison_backend_found,
            f"comparison log does not mention backend {backend_label!r}",
        )
        comparison_backend_line = find_prefixed_line(
            stripped_comparison_lines, "backend:"
        )
        comparison_backend_after_line = find_prefixed_line(
            stripped_comparison_lines, "backend_after:"
        )
        comparison_executor_before = extract_executor(comparison_backend_line or "")
        comparison_executor_after = extract_executor(comparison_backend_after_line or "")
        comparison_profile_enabled = "CUDA profile report=" in comparison_text
        warmup_line = find_prefixed_line(
            stripped_comparison_lines, "benchmark_warmups:"
        )
        batch_line = find_prefixed_line(stripped_comparison_lines, "batches:")
        if args.require_batch_benchmark:
            require(batch_line is not None, "comparison log is missing batch benchmark")
            require(
                warmup_line is not None,
                "comparison log is missing batch benchmark warmup count",
            )

    probe = load_probe_json(args.probe_log)
    network_info = str(probe.get("network_info", ""))
    require(
        backend_label in network_info,
        f"probe backend mismatch: expected {backend_label!r}, got {network_info!r}",
    )
    if not args.allow_missing_wdl:
        require(bool(probe.get("has_wdl")), "probe output did not decode WDL")
    if not args.allow_missing_moves_left:
        require(
            bool(probe.get("has_moves_left")),
            "probe output did not decode moves-left",
        )
    policy_top = probe.get("policy_top", [])
    require(
        isinstance(policy_top, list) and len(policy_top) >= args.min_policy_top,
        f"probe policy_top has fewer than {args.min_policy_top} entries",
    )

    manifest = {
        "backend_label": backend_label,
        "parity_report": str(args.parity_report),
        "probe_log": str(args.probe_log),
        "comparison_log": str(args.comparison_log) if args.comparison_log else None,
        "parity_backend_line": backend_line,
        "comparison_backend_found": comparison_backend_found,
        "comparison_backend_line": comparison_backend_line,
        "comparison_backend_after_line": comparison_backend_after_line,
        "comparison_executor_before": comparison_executor_before,
        "comparison_executor_after": comparison_executor_after,
        "comparison_profile_enabled": comparison_profile_enabled,
        "benchmark_warmup_line": warmup_line,
        "batch_line": batch_line,
        "probe": {
            "backend": probe.get("backend"),
            "network_info": network_info,
            "executor": extract_executor(network_info),
            "profile_enabled": "CUDA profile report=" in probe_text,
            "format": probe.get("format"),
            "has_wdl": bool(probe.get("has_wdl")),
            "wdl": probe.get("wdl"),
            "has_moves_left": bool(probe.get("has_moves_left")),
            "moves_left": probe.get("moves_left"),
            "policy_top": policy_top[: args.min_policy_top],
        },
    }

    if args.manifest_out:
        output = Path(args.manifest_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        "NN artifact check: PASS "
        f"backend={backend_label} wdl={manifest['probe']['has_wdl']} "
        f"moves_left={manifest['probe']['has_moves_left']} "
        f"policy_top={len(policy_top)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"NN artifact check: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
