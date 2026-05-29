#!/usr/bin/env python3
"""Write Linux and Windows CUDA runtime gate manifests from gate environment."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
from collections.abc import Mapping
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cuda_runtime_observed import collect_observed_runtime_facts  # noqa: E402
from tools.cuda_runtime_search_contract import SEARCH_COMPARISONS  # noqa: E402


SCHEMAS = {
    "linux-cuda": "metalfish.cuda_gpu_runtime_gate",
    "windows-cuda": "metalfish.windows_cuda_runtime_gate",
}


def file_record(path: str | pathlib.Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = pathlib.Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def artifact_records(
    artifact_dir: pathlib.Path,
    *,
    manifest_path: pathlib.Path,
    recursive: bool,
) -> dict[str, dict[str, Any]]:
    if not artifact_dir.is_dir():
        return {}
    candidates = (
        sorted(artifact_dir.rglob("*")) if recursive else sorted(artifact_dir.iterdir())
    )
    artifacts = {}
    manifest_resolved = manifest_path.resolve()
    for candidate in candidates:
        if not candidate.is_file() or candidate.resolve() == manifest_resolved:
            continue
        record = file_record(candidate)
        if record is None:
            continue
        key = (
            str(candidate.relative_to(artifact_dir))
            if recursive
            else candidate.name
        )
        artifacts[key] = record
    return artifacts


def env_value(env: Mapping[str, str], name: str) -> str:
    try:
        return env[name]
    except KeyError as exc:
        raise KeyError(f"missing required environment variable {name}") from exc


def metal_input_records(env: Mapping[str, str]) -> dict[str, Any]:
    records = {
        "require_metal_compare": env_value(env, "GATE_REQUIRE_METAL_COMPARE"),
        "require_metal_benchmark_compare": env_value(
            env, "GATE_REQUIRE_METAL_BENCHMARK_COMPARE"
        ),
        "require_metal_search_compare": env_value(
            env, "GATE_REQUIRE_METAL_SEARCH_COMPARE"
        ),
        "max_cuda_metal_eval_ms_ratio": env_value(
            env, "GATE_MAX_CUDA_METAL_EVAL_MS_RATIO"
        ),
        "metal_comparison_log": file_record(env_value(env, "GATE_METAL_COMPARISON_LOG")),
        "metal_probe_suite_log": file_record(
            env_value(env, "GATE_METAL_PROBE_SUITE_LOG")
        ),
        "metal_legacy_probe_suite_log": file_record(
            env_value(env, "GATE_METAL_LEGACY_PROBE_SUITE_LOG")
        ),
    }
    for spec in SEARCH_COMPARISONS:
        records[spec.metal_input_key] = file_record(env_value(env, spec.gate_env_var))
    return records


def common_runtime(env: Mapping[str, str]) -> dict[str, str]:
    return {
        "cuda_stable_execution_batch_size": env_value(
            env, "GATE_CUDA_STABLE_BATCH_SIZE"
        ),
        "cuda_graph": env_value(env, "GATE_CUDA_GRAPH"),
        "cuda_graph_execution": env_value(env, "GATE_CUDA_GRAPH_EXECUTION"),
        "cuda_deterministic_attention_softmax": env_value(
            env, "GATE_CUDA_DETERMINISTIC_ATTENTION_SOFTMAX"
        ),
        "cuda_full_buffer_clear": env_value(env, "GATE_CUDA_FULL_BUFFER_CLEAR"),
        "cuda_profile": env_value(env, "GATE_CUDA_PROFILE"),
        "cuda_profile_limit": env_value(env, "GATE_CUDA_PROFILE_LIMIT"),
        "cublas_workspace_config": env_value(env, "GATE_CUBLAS_WORKSPACE_CONFIG"),
    }


def linux_manifest(
    *,
    manifest_path: pathlib.Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    artifact_dir = pathlib.Path(env_value(env, "GATE_ARTIFACT_DIR"))
    runtime = common_runtime(env)
    runtime.update(
        {
            "uci_go": env_value(env, "GATE_CUDA_UCI_GO"),
            "mcts_ponder_uci_go": env_value(env, "GATE_CUDA_MCTS_PONDER_GO"),
            "mcts_ponder_settle_sec": env_value(
                env, "GATE_CUDA_MCTS_PONDER_SETTLE_SEC"
            ),
        }
    )
    return {
        "schema_version": 1,
        "schema": SCHEMAS["linux-cuda"],
        "created_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "git": {
            "head_sha": env_value(env, "GIT_HEAD_SHA"),
            "archive": file_record(env_value(env, "GATE_ARCHIVE")),
        },
        "gcp": {
            "project": env_value(env, "GATE_PROJECT"),
            "instance": env_value(env, "GATE_INSTANCE"),
            "zone": env_value(env, "GATE_ZONE"),
            "machine": env_value(env, "GATE_MACHINE"),
            "accelerator": env_value(env, "GATE_ACCELERATOR"),
            "image_project": env_value(env, "GATE_IMAGE_PROJECT"),
            "image_family": env_value(env, "GATE_IMAGE_FAMILY"),
            "boot_disk_size": env_value(env, "GATE_BOOT_DISK_SIZE"),
            "delete_on_exit": env_value(env, "GATE_DELETE_ON_EXIT") == "1",
            "gcs_prefix": env_value(env, "GATE_GCS_PREFIX"),
        },
        "inputs": metal_input_records(env),
        "runtime": runtime,
        "status": {
            "remote_status": env_value(env, "REMOTE_STATUS_FOR_MANIFEST"),
            "bt4_compare_status": env_value(env, "BT4_COMPARE_STATUS_FOR_MANIFEST"),
            "legacy_compare_status": env_value(
                env, "LEGACY_COMPARE_STATUS_FOR_MANIFEST"
            ),
            "benchmark_compare_status": env_value(
                env, "BENCHMARK_COMPARE_STATUS_FOR_MANIFEST"
            ),
            "search_compare_status": env_value(
                env, "SEARCH_COMPARE_STATUS_FOR_MANIFEST"
            ),
            "final_compare_status": env_value(
                env, "FINAL_COMPARE_STATUS_FOR_MANIFEST"
            ),
        },
        "observed_runtime": collect_observed_runtime_facts(
            artifact_dir, runtime_kind="linux-cuda"
        ),
        "artifacts": artifact_records(
            artifact_dir, manifest_path=manifest_path, recursive=False
        ),
    }


def windows_manifest(
    *,
    manifest_path: pathlib.Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    artifact_dir = pathlib.Path(env_value(env, "GATE_ARTIFACT_DIR"))
    inputs = metal_input_records(env)
    inputs.update(
        {
            "windows_cuda_compile_run_id": env_value(
                env, "GATE_WINDOWS_CUDA_COMPILE_RUN_ID"
            ),
            "package": {
                "name": env_value(env, "GATE_PACKAGE_BASENAME"),
                "record": file_record(env_value(env, "GATE_PACKAGE_ZIP")),
            },
        }
    )
    runtime = common_runtime(env)
    runtime.update(
        {
            "uci_go": env_value(env, "GATE_UCI_GO"),
            "mcts_timed_uci_go": env_value(env, "GATE_MCTS_TIMED_UCI_GO"),
            "mcts_ponder_uci_go": env_value(env, "GATE_MCTS_PONDER_UCI_GO"),
            "mcts_ponder_settle_ms": env_value(env, "GATE_MCTS_PONDER_SETTLE_MS"),
            "hybrid_uci_go": env_value(env, "GATE_HYBRID_UCI_GO"),
            "hybrid_parity_uci_go": env_value(env, "GATE_HYBRID_PARITY_UCI_GO"),
        }
    )
    return {
        "schema": SCHEMAS["windows-cuda"],
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "git": {
            "head_sha": env_value(env, "GIT_HEAD_SHA"),
        },
        "gcp": {
            "project": env_value(env, "GATE_PROJECT"),
            "instance": env_value(env, "GATE_INSTANCE"),
            "zone": env_value(env, "GATE_ZONE"),
            "machine": env_value(env, "GATE_MACHINE"),
            "machine_candidates": env_value(env, "GATE_MACHINES"),
            "accelerator": env_value(env, "GATE_ACCELERATOR"),
            "image_project": env_value(env, "GATE_IMAGE_PROJECT"),
            "image_family": env_value(env, "GATE_IMAGE_FAMILY"),
            "boot_disk_size": env_value(env, "GATE_BOOT_DISK_SIZE"),
            "boot_disk_type": env_value(env, "GATE_BOOT_DISK_TYPE"),
            "delete_on_exit": env_value(env, "GATE_DELETE_ON_EXIT") == "1",
            "gcs_prefix": env_value(env, "GATE_GCS_PREFIX"),
        },
        "inputs": inputs,
        "runtime": runtime,
        "status": {
            "runtime_status": env_value(env, "RUNTIME_STATUS_FOR_MANIFEST"),
            "bt4_compare_status": env_value(env, "BT4_COMPARE_STATUS_FOR_MANIFEST"),
            "legacy_compare_status": env_value(
                env, "LEGACY_COMPARE_STATUS_FOR_MANIFEST"
            ),
            "benchmark_compare_status": env_value(
                env, "BENCHMARK_COMPARE_STATUS_FOR_MANIFEST"
            ),
            "search_compare_status": env_value(
                env, "SEARCH_COMPARE_STATUS_FOR_MANIFEST"
            ),
            "final_compare_status": env_value(
                env, "FINAL_COMPARE_STATUS_FOR_MANIFEST"
            ),
        },
        "observed_runtime": collect_observed_runtime_facts(
            artifact_dir, runtime_kind="windows-cuda"
        ),
        "artifacts": artifact_records(
            artifact_dir, manifest_path=manifest_path, recursive=True
        ),
    }


def write_manifest_from_env(
    *,
    runtime_kind: str,
    manifest_path: pathlib.Path,
    env: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if runtime_kind == "linux-cuda":
        manifest = linux_manifest(manifest_path=manifest_path, env=env)
    elif runtime_kind == "windows-cuda":
        manifest = windows_manifest(manifest_path=manifest_path, env=env)
    else:
        raise ValueError(f"unsupported CUDA runtime kind: {runtime_kind}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-kind", choices=sorted(SCHEMAS), required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    write_manifest_from_env(runtime_kind=args.runtime_kind, manifest_path=args.manifest)
    print(f"Wrote CUDA runtime manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
