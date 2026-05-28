#!/usr/bin/env python3
"""Validate Linux and Windows CUDA runtime gate manifests."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


RUNTIME_KINDS = {
    "linux-cuda": {
        "schema": "metalfish.cuda_gpu_runtime_gate",
        "status_fields": {
            "remote_status": "Linux remote_status",
            "bt4_compare_status": "BT4 compare status",
            "legacy_compare_status": "legacy compare status",
            "final_compare_status": "final compare status",
        },
    },
    "windows-cuda": {
        "schema": "metalfish.windows_cuda_runtime_gate",
        "status_fields": {
            "runtime_status": "Windows runtime_status",
            "bt4_compare_status": "BT4 compare status",
            "legacy_compare_status": "legacy compare status",
            "final_compare_status": "final compare status",
        },
    },
}

REQUIRED_RELEASE_ARTIFACTS = {
    "linux-cuda": {
        "linux-cuda-package-manifest.json",
        "cuda-gpu-package-nn-comparison.log",
        "cuda-gpu-package-nn-probe-suite.log",
        "cuda-gpu-package-legacy-nn-probe-suite.log",
        "cuda-gpu-package-nn-isolation-bt4-legacy.log",
        "cuda-gpu-package-nn-isolation-legacy-bt4.log",
        "metal-cuda-nn-probe-suite-summary.json",
        "metal-cuda-legacy-nn-probe-suite-summary.json",
        "metal-cuda-nn-benchmark-summary.json",
        "metal-cuda-nn-benchmark-compare.log",
        "metal-cuda-mcts-bk07-search-summary.json",
        "metal-cuda-mcts-kiwipete-search-summary.json",
        "metal-cuda-hybrid-bk07-search-summary.json",
        "metal-cuda-hybrid-kiwipete-search-summary.json",
        "cuda-gpu-uci-bk07-smoke.log",
        "cuda-gpu-uci-bk07-search.json",
        "cuda-gpu-uci-kiwipete-smoke.log",
        "cuda-gpu-uci-kiwipete-search.json",
        "cuda-gpu-uci-hybrid-search.json",
        "cuda-gpu-uci-hybrid-kiwipete-search.json",
        "cuda-gpu-uci-hybrid-clock-safety-smoke.log",
    },
    "windows-cuda": {
        "logs/windows-cuda-runtime-manifest.json",
        "logs/cuda-nn-comparison.stdout.log",
        "logs/cuda-probe-suite.stdout.log",
        "logs/cuda-legacy-probe-suite.stdout.log",
        "logs/cuda-isolation-bt4-legacy.stdout.log",
        "logs/cuda-isolation-legacy-bt4.stdout.log",
        "logs/metal-windows-cuda-nn-probe-suite-summary.json",
        "logs/metal-windows-cuda-legacy-nn-probe-suite-summary.json",
        "logs/metal-windows-cuda-nn-benchmark-summary.json",
        "logs/metal-windows-cuda-nn-benchmark-compare.log",
        "logs/metal-windows-cuda-mcts-bk07-search-summary.json",
        "logs/metal-windows-cuda-mcts-kiwipete-search-summary.json",
        "logs/metal-windows-cuda-hybrid-bk07-search-summary.json",
        "logs/metal-windows-cuda-hybrid-kiwipete-search-summary.json",
        "logs/cuda-bk07-mcts.stdout.log",
        "logs/cuda-bk07-mcts-search.json",
        "logs/cuda-kiwipete-mcts.stdout.log",
        "logs/cuda-kiwipete-mcts-search.json",
        "logs/hybrid-cuda-search.json",
        "logs/hybrid-cuda-kiwipete-search.json",
        "logs/hybrid-cuda-clock-safety.stdout.log",
    },
}


def require_zero_status(status: object, *, label: str) -> None:
    if str(status) != "0":
        raise ValueError(f"{label} must be 0, got {status!r}")


def is_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def require_file_record(record: object, *, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record is missing")
    if int(record.get("size_bytes") or 0) <= 0:
        raise ValueError(f"{label} record has invalid size")
    if not str(record.get("sha256") or ""):
        raise ValueError(f"{label} record is missing sha256")


def validate_metal_compare_inputs(
    inputs: object,
    *,
    require_benchmark_compare: bool = False,
    require_search_compare: bool = False,
) -> None:
    if not isinstance(inputs, dict):
        raise ValueError("runtime manifest is missing inputs object")
    if not is_truthy(inputs.get("require_metal_compare")):
        raise ValueError("runtime manifest did not require Metal comparison")
    require_file_record(
        inputs.get("metal_probe_suite_log"), label="Metal BT4 probe suite"
    )
    require_file_record(
        inputs.get("metal_legacy_probe_suite_log"), label="Metal legacy probe suite"
    )
    if require_benchmark_compare:
        if not is_truthy(inputs.get("require_metal_benchmark_compare")):
            raise ValueError(
                "runtime manifest did not require Metal benchmark comparison"
            )
        require_file_record(
            inputs.get("metal_comparison_log"), label="Metal benchmark comparison"
        )
    if require_search_compare:
        if not is_truthy(inputs.get("require_metal_search_compare")):
            raise ValueError("runtime manifest did not require Metal search comparison")
        require_file_record(
            inputs.get("metal_mcts_bk07_search_json"),
            label="Metal MCTS BK.07 search JSON",
        )
        require_file_record(
            inputs.get("metal_mcts_kiwipete_search_json"),
            label="Metal MCTS kiwipete search JSON",
        )
        require_file_record(
            inputs.get("metal_hybrid_bk07_search_json"),
            label="Metal Hybrid BK.07 search JSON",
        )
        require_file_record(
            inputs.get("metal_hybrid_kiwipete_search_json"),
            label="Metal Hybrid kiwipete search JSON",
        )


def validate_release_artifacts(data: dict, *, runtime_kind: str) -> None:
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("runtime manifest is missing artifacts object")
    package_records = [
        name
        for name in artifacts
        if runtime_kind == "linux-cuda"
        and name.startswith("metalfish-linux-x86_64-cuda")
        and name.endswith(".tar.gz")
    ]
    if runtime_kind == "linux-cuda" and not package_records:
        raise ValueError("runtime manifest is missing Linux CUDA package artifact")
    for name in sorted(REQUIRED_RELEASE_ARTIFACTS[runtime_kind]):
        require_file_record(artifacts.get(name), label=f"artifact {name}")


def validate_runtime_manifest(
    manifest: pathlib.Path,
    *,
    runtime_kind: str,
    require_metal_compare: bool = False,
    require_metal_benchmark_compare: bool = False,
    require_metal_search_compare: bool = False,
    require_release_evidence: bool = False,
    expected_head_sha: str | None = None,
) -> dict:
    if runtime_kind not in RUNTIME_KINDS:
        raise ValueError(f"unsupported CUDA runtime kind: {runtime_kind}")
    spec = RUNTIME_KINDS[runtime_kind]
    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if data.get("schema") != spec["schema"]:
        raise ValueError(
            f"runtime manifest {manifest} has unexpected schema: "
            f"{data.get('schema')!r}"
        )
    git = data.get("git") or {}
    if expected_head_sha and git.get("head_sha") != expected_head_sha:
        raise ValueError(
            f"runtime manifest {manifest} has unexpected git head: "
            f"{git.get('head_sha')!r}; expected {expected_head_sha}"
        )
    status = data.get("status")
    if not isinstance(status, dict):
        raise ValueError(f"runtime manifest {manifest} is missing status object")
    for field, label in spec["status_fields"].items():
        require_zero_status(status.get(field), label=label)
    if require_metal_benchmark_compare:
        require_zero_status(
            status.get("benchmark_compare_status"), label="benchmark compare status"
        )
    if require_metal_search_compare:
        require_zero_status(
            status.get("search_compare_status"), label="search compare status"
        )
    inputs = data.get("inputs") or {}
    if require_metal_compare:
        validate_metal_compare_inputs(
            inputs,
            require_benchmark_compare=require_metal_benchmark_compare,
            require_search_compare=require_metal_search_compare,
        )
    if require_release_evidence:
        validate_release_artifacts(data, runtime_kind=runtime_kind)
    return {
        "schema": data["schema"],
        "kind": runtime_kind,
        "head_sha": git.get("head_sha"),
        "status": status,
        "gcp": data.get("gcp") or {},
        "inputs": inputs,
        "artifacts": data.get("artifacts") or {},
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--runtime-kind", choices=tuple(RUNTIME_KINDS.keys()), required=True
    )
    parser.add_argument("--require-metal-compare", action="store_true")
    parser.add_argument("--require-metal-benchmark-compare", action="store_true")
    parser.add_argument("--require-metal-search-compare", action="store_true")
    parser.add_argument("--require-release-evidence", action="store_true")
    parser.add_argument("--expected-head-sha", default="")
    parser.add_argument("--json-output", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = pathlib.Path(args.manifest).expanduser().resolve()
    summary = validate_runtime_manifest(
        manifest,
        runtime_kind=args.runtime_kind,
        require_metal_compare=args.require_metal_compare,
        require_metal_benchmark_compare=args.require_metal_benchmark_compare,
        require_metal_search_compare=args.require_metal_search_compare,
        require_release_evidence=args.require_release_evidence,
        expected_head_sha=args.expected_head_sha or None,
    )
    if args.json_output:
        path = pathlib.Path(args.json_output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "CUDA runtime manifest check: PASS "
        f"kind={summary['kind']} schema={summary['schema']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
