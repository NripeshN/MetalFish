#!/usr/bin/env python3
"""Extract semantic CUDA runtime facts from gate artifacts."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

from tools.cuda_runtime_search_contract import search_summary_paths

_BOOL_RE = {
    "1": True,
    "true": True,
    "yes": True,
    "on": True,
    "0": False,
    "false": False,
    "no": False,
    "off": False,
}


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _value(text: str, key: str) -> str | None:
    match = re.search(rf"{re.escape(key)}=([^,\s)]+)", text)
    return match.group(1) if match else None


def _bool_value(text: str, key: str) -> bool | None:
    value = _value(text, key)
    if value is None:
        return None
    return _BOOL_RE.get(value.strip().lower())


def _int_value(text: str, key: str) -> int | None:
    value = _value(text, key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_cuda_backend_line(line: object) -> dict[str, Any]:
    text = str(line or "")
    executor = ""
    executor_index = text.find("executor=")
    if executor_index >= 0:
        executor = text[executor_index + len("executor=") :].strip()
        while executor.endswith(")") and executor.count("(") < executor.count(")"):
            executor = executor[:-1]
    executor_kind = executor.split("(", 1)[0] if executor else ""
    return {
        "raw": text,
        "backend_is_cuda": "CUDA transformer backend" in text,
        "cuda_graph_effective": _bool_value(text, "cuda_graph_effective"),
        "cuda_stable_execution_batch_effective": _int_value(
            text, "cuda_stable_execution_batch_effective"
        ),
        "cuda_deterministic_attention_softmax": _bool_value(
            text, "cuda_deterministic_attention_softmax"
        ),
        "cuda_full_buffer_clear_effective": _bool_value(
            text, "cuda_full_buffer_clear_effective"
        ),
        "executor": executor,
        "executor_kind": executor_kind,
        "executor_resolved": executor_kind.startswith("resolved+"),
        "executor_graph_replay": executor_kind == "resolved+graph-replay",
        "graph_captures": _int_value(executor, "captures"),
        "graph_replays": _int_value(executor, "replays"),
        "graph_caches": _int_value(executor, "caches"),
        "graph_primed": _int_value(executor, "primed"),
    }


def _benchmark_facts(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"present": False}
    actual = summary.get("actual") or {}
    expected = summary.get("expected") or {}
    best_common = summary.get("best_common_actual") or {}
    actual_best = actual.get("best_batch") or {}
    cuda_backend_after = parse_cuda_backend_line(actual.get("backend_after_line"))
    stable_batch = cuda_backend_after.get("cuda_stable_execution_batch_effective")
    stable_common = None
    if stable_batch is not None:
        for row in summary.get("common_batches") or []:
            if row.get("batch_size") == stable_batch:
                stable_common = row
                break
    return {
        "present": True,
        "cuda_backend": parse_cuda_backend_line(actual.get("backend_line")),
        "cuda_backend_after": cuda_backend_after,
        "metal_backend_label": expected.get("label"),
        "common_batch_count": summary.get("common_batch_count"),
        "best_common_batch": best_common.get("batch_size"),
        "best_common_cuda_eval_ms": _float_or_none(best_common.get("actual_eval_ms")),
        "best_common_metal_eval_ms": _float_or_none(
            best_common.get("expected_eval_ms")
        ),
        "best_common_speedup_vs_metal": _float_or_none(
            best_common.get("actual_speedup_vs_expected")
        ),
        "worst_eval_ms_ratio": _float_or_none(summary.get("worst_eval_ms_ratio")),
        "stable_batch": stable_batch,
        "stable_batch_cuda_eval_ms": _float_or_none(
            (stable_common or {}).get("actual_eval_ms")
        ),
        "stable_batch_metal_eval_ms": _float_or_none(
            (stable_common or {}).get("expected_eval_ms")
        ),
        "stable_batch_eval_ms_ratio": _float_or_none(
            (stable_common or {}).get("eval_ms_ratio")
        ),
        "stable_batch_speedup_vs_metal": _float_or_none(
            (stable_common or {}).get("actual_speedup_vs_expected")
        ),
        "fastest_cuda_batch": actual_best.get("batch_size"),
        "fastest_cuda_eval_ms": _float_or_none(actual_best.get("eval_ms")),
    }


def _search_facts(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"present": False}
    actual = summary.get("actual") or {}
    expected = summary.get("expected") or {}
    return {
        "present": True,
        "status": summary.get("status"),
        "same_bestmove_required": bool(summary.get("require_same_bestmove")),
        "cuda_bestmove": actual.get("bestmove"),
        "metal_bestmove": expected.get("bestmove"),
        "cuda_nodes": (actual.get("search_info") or {}).get("nodes"),
        "metal_nodes": (expected.get("search_info") or {}).get("nodes"),
    }


def collect_observed_runtime_facts(
    artifact_dir: pathlib.Path | str,
    *,
    runtime_kind: str,
) -> dict[str, Any]:
    root = pathlib.Path(artifact_dir)
    if runtime_kind == "linux-cuda":
        benchmark = root / "metal-cuda-nn-benchmark-summary.json"
    elif runtime_kind == "windows-cuda":
        logs = root / "logs"
        benchmark = logs / "metal-windows-cuda-nn-benchmark-summary.json"
    else:
        raise ValueError(f"unsupported CUDA runtime kind: {runtime_kind}")
    searches = search_summary_paths(root, runtime_kind=runtime_kind)

    return {
        "schema_version": 1,
        "runtime_kind": runtime_kind,
        "benchmark_compare": _benchmark_facts(_read_json(benchmark)),
        "search_compare": {
            name: _search_facts(_read_json(path)) for name, path in searches.items()
        },
    }
