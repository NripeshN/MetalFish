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


def require_zero_status(status: object, *, label: str) -> None:
    if str(status) != "0":
        raise ValueError(f"{label} must be 0, got {status!r}")


def validate_runtime_manifest(manifest: pathlib.Path, *, runtime_kind: str) -> dict:
    if runtime_kind not in RUNTIME_KINDS:
        raise ValueError(f"unsupported CUDA runtime kind: {runtime_kind}")
    spec = RUNTIME_KINDS[runtime_kind]
    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if data.get("schema") != spec["schema"]:
        raise ValueError(
            f"runtime manifest {manifest} has unexpected schema: "
            f"{data.get('schema')!r}"
        )
    status = data.get("status")
    if not isinstance(status, dict):
        raise ValueError(f"runtime manifest {manifest} is missing status object")
    for field, label in spec["status_fields"].items():
        require_zero_status(status.get(field), label=label)
    return {
        "schema": data["schema"],
        "kind": runtime_kind,
        "status": status,
        "gcp": data.get("gcp") or {},
        "inputs": data.get("inputs") or {},
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--runtime-kind", choices=tuple(RUNTIME_KINDS.keys()), required=True
    )
    parser.add_argument("--json-output", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = pathlib.Path(args.manifest).expanduser().resolve()
    summary = validate_runtime_manifest(manifest, runtime_kind=args.runtime_kind)
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
