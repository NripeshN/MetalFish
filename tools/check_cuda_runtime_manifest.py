#!/usr/bin/env python3
"""Validate Linux and Windows CUDA runtime gate manifests."""
from __future__ import annotations

import argparse
import hashlib
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
        "linux-cuda-package-check.json",
        "cuda-gpu-package-nn-probe-suite.log",
        "cuda-gpu-package-legacy-nn-probe-suite.log",
        "cuda-gpu-package-nn-isolation-bt4-legacy.log",
        "cuda-gpu-package-nn-isolation-legacy-bt4.log",
        "cuda-gpu-package-uci-ponder-mcts-smoke.log",
        "cuda-gpu-package-uci-ponder-mcts.json",
        "cuda-gpu-package-uci-bk07-smoke.log",
        "cuda-gpu-package-uci-bk07-search.json",
        "cuda-gpu-package-uci-kiwipete-smoke.log",
        "cuda-gpu-package-uci-kiwipete-search.json",
        "cuda-gpu-package-uci-hybrid-smoke.log",
        "cuda-gpu-package-uci-hybrid-search.json",
        "cuda-gpu-package-uci-hybrid-kiwipete-smoke.log",
        "cuda-gpu-package-uci-hybrid-kiwipete-search.json",
        "metal-cuda-nn-probe-suite-summary.json",
        "metal-cuda-legacy-nn-probe-suite-summary.json",
        "metal-cuda-nn-benchmark-summary.json",
        "metal-cuda-nn-benchmark-compare.log",
        "metal-cuda-mcts-bk07-search-summary.json",
        "metal-cuda-mcts-kiwipete-search-summary.json",
        "metal-cuda-hybrid-bk07-search-summary.json",
        "metal-cuda-hybrid-kiwipete-search-summary.json",
        "cuda-gpu-uci-bk07-smoke.log",
        "cuda-gpu-uci-timed-mcts-smoke.log",
        "cuda-gpu-uci-timed-mcts-search.json",
        "cuda-gpu-uci-ponder-mcts-smoke.log",
        "cuda-gpu-uci-ponder-mcts.json",
        "cuda-gpu-uci-bk07-search.json",
        "cuda-gpu-uci-kiwipete-smoke.log",
        "cuda-gpu-uci-kiwipete-search.json",
        "cuda-gpu-uci-hybrid-search.json",
        "cuda-gpu-uci-hybrid-kiwipete-search.json",
        "cuda-gpu-uci-hybrid-clock-start-smoke.log",
        "cuda-gpu-uci-hybrid-clock-safety-smoke.log",
    },
    "windows-cuda": {
        "logs/windows-cuda-runtime-manifest.json",
        "logs/cuda-nn-comparison.stdout.log",
        "logs/cuda-probe-suite.stdout.log",
        "logs/cuda-legacy-probe-suite.stdout.log",
        "logs/cuda-isolation-bt4-legacy.stdout.log",
        "logs/cuda-isolation-legacy-bt4.stdout.log",
        "logs/cuda-timed-mcts.stdout.log",
        "logs/cuda-timed-mcts.stderr.log",
        "logs/cuda-timed-mcts-search.json",
        "logs/cuda-ponder-mcts.stdout.log",
        "logs/cuda-ponder-mcts.stderr.log",
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
        "logs/hybrid-cuda-clock-start.stdout.log",
        "logs/hybrid-cuda-clock-start.stderr.log",
        "logs/hybrid-cuda-clock-safety.stdout.log",
        "logs/hybrid-cuda-clock-safety.stderr.log",
    },
}


def require_zero_status(status: object, *, label: str) -> None:
    if str(status) != "0":
        raise ValueError(f"{label} must be 0, got {status!r}")


def is_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def is_falsey_or_empty(value: object) -> bool:
    return str(value or "").strip().lower() in {"", "0", "false", "no", "off"}


def is_explicitly_disabled(value: object) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "off"}


def require_file_record(record: object, *, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record is missing")
    if int(record.get("size_bytes") or 0) <= 0:
        raise ValueError(f"{label} record has invalid size")
    if not str(record.get("sha256") or ""):
        raise ValueError(f"{label} record is missing sha256")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_record_path(
    *,
    key: str,
    record: dict,
    artifact_root: pathlib.Path,
) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    root = artifact_root.resolve()
    record_path = str(record.get("path") or "")
    if record_path:
        path = pathlib.Path(record_path)
        candidate = path if path.is_absolute() else artifact_root / path
        try:
            candidate.resolve().relative_to(root)
            candidates.append(candidate)
        except ValueError:
            pass
    candidates.append(artifact_root / key)

    checked: list[str] = []
    for candidate in candidates:
        if str(candidate) in checked:
            continue
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise ValueError(f"artifact {key} file is missing; checked: {', '.join(checked)}")


def validate_artifact_files(
    artifacts: object,
    *,
    artifact_root: pathlib.Path,
) -> None:
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("runtime manifest is missing artifacts object")
    for key, record in sorted(artifacts.items()):
        require_file_record(record, label=f"artifact {key}")
        path = resolve_record_path(key=key, record=record, artifact_root=artifact_root)
        expected_size = int(record["size_bytes"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"artifact {key} size mismatch: {actual_size} != {expected_size}"
            )
        expected_sha = str(record["sha256"])
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"artifact {key} sha256 mismatch: {actual_sha} != {expected_sha}"
            )


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


def validate_release_runtime_policy(
    runtime: object,
    *,
    stable_batch_size: str,
) -> None:
    if not isinstance(runtime, dict):
        raise ValueError("runtime manifest is missing runtime object")
    stable_batch = str(runtime.get("cuda_stable_execution_batch_size") or "")
    if stable_batch != stable_batch_size:
        raise ValueError(
            "release runtime policy requires CUDA stable batch "
            f"{stable_batch_size}, got {stable_batch!r}"
        )
    for field in ("cuda_graph", "cuda_graph_execution"):
        if not is_truthy(runtime.get(field)):
            raise ValueError(f"release runtime policy requires graph enabled: {field}")
    for field in ("cuda_deterministic_attention_softmax", "cuda_full_buffer_clear"):
        if not is_truthy(runtime.get(field)):
            raise ValueError(f"release runtime policy requires enabled: {field}")
    if not is_falsey_or_empty(runtime.get("cuda_profile")):
        raise ValueError("release runtime policy requires CUDA profiling disabled")
    if str(runtime.get("cublas_workspace_config") or "").strip():
        raise ValueError(
            "release runtime policy requires empty CUBLAS_WORKSPACE_CONFIG"
        )


def validate_release_compare_policy(
    inputs: object,
    *,
    max_cuda_metal_eval_ms_ratio: str,
) -> None:
    if not isinstance(inputs, dict):
        raise ValueError("runtime manifest is missing inputs object")
    value = str(inputs.get("max_cuda_metal_eval_ms_ratio") or "").strip()
    if not value:
        raise ValueError(
            "release comparison policy requires max_cuda_metal_eval_ms_ratio"
        )
    try:
        actual = float(value)
        maximum = float(max_cuda_metal_eval_ms_ratio)
    except ValueError as exc:
        raise ValueError(
            "release comparison policy requires numeric "
            "max_cuda_metal_eval_ms_ratio"
        ) from exc
    if actual <= 0.0:
        raise ValueError(
            "release comparison policy requires positive "
            "max_cuda_metal_eval_ms_ratio"
        )
    if actual > maximum:
        raise ValueError(
            "release comparison policy rejects relaxed Metal comparison ratio: "
            f"{actual:g} > {maximum:g}"
        )


def validate_observed_runtime_facts(
    observed: object,
    *,
    runtime_kind: str,
    stable_batch_size: str,
    max_cuda_metal_eval_ms_ratio: str,
) -> None:
    if not isinstance(observed, dict):
        raise ValueError("runtime manifest is missing observed_runtime object")
    if observed.get("runtime_kind") != runtime_kind:
        raise ValueError(
            "observed runtime kind mismatch: "
            f"{observed.get('runtime_kind')!r} != {runtime_kind!r}"
        )
    benchmark = observed.get("benchmark_compare")
    if not isinstance(benchmark, dict) or not benchmark.get("present"):
        raise ValueError("observed runtime is missing benchmark comparison facts")
    backend_after = benchmark.get("cuda_backend_after")
    if not isinstance(backend_after, dict):
        raise ValueError("observed runtime is missing CUDA backend_after facts")
    if backend_after.get("backend_is_cuda") is not True:
        raise ValueError("observed runtime backend_after is not CUDA")
    if backend_after.get("cuda_graph_effective") is not True:
        raise ValueError("observed runtime did not prove cuda_graph_effective=true")
    if backend_after.get("executor_graph_replay") is not True:
        raise ValueError("observed runtime did not prove graph replay executor")
    if int(backend_after.get("graph_replays") or 0) <= 0:
        raise ValueError("observed runtime did not prove graph replay count")
    expected_batch = int(stable_batch_size)
    if backend_after.get("cuda_stable_execution_batch_effective") != expected_batch:
        raise ValueError(
            "observed runtime stable batch mismatch: "
            f"{backend_after.get('cuda_stable_execution_batch_effective')!r} "
            f"!= {expected_batch}"
        )
    if backend_after.get("cuda_deterministic_attention_softmax") is not True:
        raise ValueError(
            "observed runtime did not prove deterministic attention softmax"
        )
    if backend_after.get("cuda_full_buffer_clear_effective") is not True:
        raise ValueError("observed runtime did not prove full buffer clear")
    if benchmark.get("stable_batch") != expected_batch:
        raise ValueError(
            "observed runtime stable benchmark batch mismatch: "
            f"{benchmark.get('stable_batch')!r} != {expected_batch}"
        )
    ratio = benchmark.get("stable_batch_eval_ms_ratio")
    if ratio is None:
        raise ValueError("observed runtime is missing stable batch eval-ms ratio")
    try:
        actual_ratio = float(ratio)
        maximum_ratio = float(max_cuda_metal_eval_ms_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "observed runtime stable batch eval-ms ratio is not numeric"
        ) from exc
    if actual_ratio > maximum_ratio:
        raise ValueError(
            "observed runtime stable batch exceeds CUDA-vs-Metal eval-ms ratio: "
            f"{actual_ratio:g} > {maximum_ratio:g}"
        )
    searches = observed.get("search_compare")
    if not isinstance(searches, dict):
        raise ValueError("observed runtime is missing search comparison facts")
    for name in ("mcts_bk07", "mcts_kiwipete", "hybrid_bk07", "hybrid_kiwipete"):
        search = searches.get(name)
        if not isinstance(search, dict) or not search.get("present"):
            raise ValueError(f"observed runtime is missing {name} search facts")
        if search.get("status") != "passed":
            raise ValueError(f"observed runtime {name} search did not pass")
        if search.get("same_bestmove_required") is not True:
            raise ValueError(f"observed runtime {name} did not require same bestmove")
        if search.get("cuda_bestmove") != search.get("metal_bestmove"):
            raise ValueError(
                f"observed runtime {name} bestmove mismatch: "
                f"{search.get('cuda_bestmove')!r} != {search.get('metal_bestmove')!r}"
            )


def validate_runtime_manifest(
    manifest: pathlib.Path,
    *,
    runtime_kind: str,
    require_metal_compare: bool = False,
    require_metal_benchmark_compare: bool = False,
    require_metal_search_compare: bool = False,
    require_release_evidence: bool = False,
    require_release_policy: bool = False,
    require_observed_runtime: bool = False,
    release_stable_batch_size: str = "16",
    release_max_cuda_metal_eval_ms_ratio: str = "1.0",
    require_artifact_files: bool = False,
    artifact_root: pathlib.Path | None = None,
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
    runtime = data.get("runtime") or {}
    if require_release_policy:
        validate_release_runtime_policy(
            runtime,
            stable_batch_size=release_stable_batch_size,
        )
        validate_release_compare_policy(
            inputs,
            max_cuda_metal_eval_ms_ratio=release_max_cuda_metal_eval_ms_ratio,
        )
    if require_observed_runtime:
        validate_observed_runtime_facts(
            data.get("observed_runtime"),
            runtime_kind=runtime_kind,
            stable_batch_size=release_stable_batch_size,
            max_cuda_metal_eval_ms_ratio=release_max_cuda_metal_eval_ms_ratio,
        )
    if require_artifact_files:
        validate_artifact_files(
            data.get("artifacts"),
            artifact_root=(artifact_root or manifest.parent).resolve(),
        )
    return {
        "schema": data["schema"],
        "kind": runtime_kind,
        "head_sha": git.get("head_sha"),
        "status": status,
        "gcp": data.get("gcp") or {},
        "inputs": inputs,
        "runtime": runtime,
        "observed_runtime": data.get("observed_runtime") or {},
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
    parser.add_argument("--require-release-policy", action="store_true")
    parser.add_argument("--require-observed-runtime", action="store_true")
    parser.add_argument("--release-stable-batch-size", default="16")
    parser.add_argument("--release-max-cuda-metal-eval-ms-ratio", default="1.0")
    parser.add_argument("--require-artifact-files", action="store_true")
    parser.add_argument(
        "--artifact-root",
        default="",
        help="Directory used to resolve artifact manifest paths; defaults to manifest parent.",
    )
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
        require_release_policy=args.require_release_policy,
        require_observed_runtime=args.require_observed_runtime,
        release_stable_batch_size=args.release_stable_batch_size,
        release_max_cuda_metal_eval_ms_ratio=(
            args.release_max_cuda_metal_eval_ms_ratio
        ),
        require_artifact_files=args.require_artifact_files,
        artifact_root=(
            pathlib.Path(args.artifact_root).expanduser().resolve()
            if args.artifact_root
            else None
        ),
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
