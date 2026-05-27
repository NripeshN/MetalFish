#!/usr/bin/env python3
"""Dispatch same-commit CUDA runtime gate workflows."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.PIPE, text=True).strip()


def run_json(cmd: list[str]) -> object:
    text = run_text(cmd)
    return json.loads(text) if text else None


def default_repo() -> str:
    data = run_json(["gh", "repo", "view", "--json", "nameWithOwner"])
    if not isinstance(data, dict):
        raise RuntimeError("gh repo view returned invalid JSON")
    return str(data["nameWithOwner"])


def git_value(args: list[str]) -> str:
    return run_text(["git", *args])


def find_successful_run(
    *,
    repo: str,
    workflow: str,
    branch: str,
    expected_sha: str,
    limit: int,
) -> dict:
    data = run_json(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            workflow,
            "--branch",
            branch,
            "--limit",
            str(limit),
            "--json",
            "conclusion,createdAt,databaseId,headSha,status,url,workflowName",
        ]
    )
    if not isinstance(data, list):
        raise RuntimeError(f"gh run list returned invalid JSON for {workflow}")
    matches = [
        item
        for item in data
        if item.get("headSha") == expected_sha
        and item.get("status") == "completed"
        and item.get("conclusion") == "success"
    ]
    if not matches:
        raise RuntimeError(
            f"no successful {workflow} run found on {branch} at {expected_sha}"
        )
    return matches[0]


def bool_input(value: bool) -> str:
    return "true" if value else "false"


def append_field(cmd: list[str], key: str, value: str | int | bool) -> None:
    cmd.extend(["-f", f"{key}={value}"])


def require_dispatchable_workflow(*, repo: str, workflow_file: str) -> None:
    proc = subprocess.run(
        ["gh", "workflow", "view", workflow_file, "--repo", repo],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode == 0:
        return
    detail = (proc.stderr or proc.stdout).strip()
    raise RuntimeError(
        f"workflow {workflow_file!r} is not dispatchable in {repo}. "
        "GitHub workflow_dispatch can only target workflows present on the "
        "repository default branch. If this workflow exists only on a PR branch, "
        "use tools/run_cuda_runtime_gates_direct.py, or the lower-level "
        "tools/fetch_cuda_gpu_gate_inputs.py plus tools/run_gcp_cuda_gpu_gate.sh "
        "and tools/fetch_windows_cuda_runtime_inputs.py plus "
        "tools/run_gcp_windows_cuda_runtime_gate.sh, until the workflow file is "
        f"merged. gh said: {detail}"
    )


def dispatch_workflow(*, repo: str, workflow_file: str, ref: str, fields: dict) -> None:
    cmd = ["gh", "workflow", "run", workflow_file, "--repo", repo, "--ref", ref]
    for key, value in fields.items():
        if value is None or value == "":
            continue
        append_field(cmd, key, value)
    subprocess.run(cmd, check=True)


def print_run(label: str, run: dict) -> None:
    print(
        f"{label}: run_id={run['databaseId']} "
        f"workflow={run.get('workflowName')} sha={run.get('headSha')} "
        f"url={run.get('url')}"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="", help="GitHub repo, for example owner/name")
    parser.add_argument(
        "--ref",
        default="",
        help="Branch or tag to dispatch. Defaults to the current branch.",
    )
    parser.add_argument(
        "--expected-sha",
        default="",
        help="Required source SHA. Defaults to the current checkout HEAD.",
    )
    parser.add_argument(
        "--target",
        choices=("linux", "windows", "both"),
        default="both",
        help="Runtime gate workflow to dispatch.",
    )
    parser.add_argument("--metal-ci-run-id", default="")
    parser.add_argument("--windows-cuda-run-id", default="")
    parser.add_argument(
        "--require-metal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require same-commit Metal probe artifacts; use --no-require-metal "
            "for diagnostics."
        ),
    )
    parser.add_argument("--stable-batch-size", default="16")
    parser.add_argument("--linux-machine", default="g2-standard-8")
    parser.add_argument("--linux-zones", default="")
    parser.add_argument("--linux-uci-go", default="nodes 8")
    parser.add_argument("--windows-machine", default="g2-standard-8 g2-standard-4")
    parser.add_argument("--windows-zones", default="")
    parser.add_argument("--windows-uci-go", default="nodes 1")
    parser.add_argument("--lookup-limit", type=int, default=30)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and print dispatches without starting workflows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo or default_repo()
    ref = args.ref or git_value(["rev-parse", "--abbrev-ref", "HEAD"])
    expected_sha = args.expected_sha or git_value(["rev-parse", "HEAD"])

    needs_linux = args.target in {"linux", "both"}
    needs_windows = args.target in {"windows", "both"}
    metal_run = None
    windows_run = None

    metal_ci_run_id = args.metal_ci_run_id
    if args.require_metal and not metal_ci_run_id:
        metal_run = find_successful_run(
            repo=repo,
            workflow="MetalFish CI",
            branch=ref,
            expected_sha=expected_sha,
            limit=args.lookup_limit,
        )
        metal_ci_run_id = str(metal_run["databaseId"])
    elif metal_ci_run_id:
        metal_run = {"databaseId": metal_ci_run_id, "headSha": expected_sha}

    windows_cuda_run_id = args.windows_cuda_run_id
    if needs_windows and not windows_cuda_run_id:
        windows_run = find_successful_run(
            repo=repo,
            workflow="Windows CUDA Compile Gate",
            branch=ref,
            expected_sha=expected_sha,
            limit=args.lookup_limit,
        )
        windows_cuda_run_id = str(windows_run["databaseId"])
    elif windows_cuda_run_id:
        windows_run = {"databaseId": windows_cuda_run_id, "headSha": expected_sha}

    if args.require_metal and not metal_ci_run_id:
        raise RuntimeError("MetalFish CI run id is required with --require-metal")
    if not args.dry_run:
        if needs_linux:
            require_dispatchable_workflow(repo=repo, workflow_file="cuda-gpu-gate.yml")
        if needs_windows:
            require_dispatchable_workflow(
                repo=repo, workflow_file="windows-cuda-runtime-gate.yml"
            )

    print(f"repo={repo}")
    print(f"ref={ref}")
    print(f"expected_sha={expected_sha}")
    if metal_run:
        print_run("MetalFish CI", metal_run)
    if windows_run:
        print_run("Windows CUDA Compile Gate", windows_run)

    if needs_linux:
        fields = {
            "metal_ci_run_id": metal_ci_run_id,
            "require_metal_compare": bool_input(args.require_metal),
            "machine": args.linux_machine,
            "zones": args.linux_zones,
            "uci_go": args.linux_uci_go,
            "stable_batch_size": args.stable_batch_size,
        }
        print(f"Dispatch CUDA GPU Gate with {fields}")
        if not args.dry_run:
            dispatch_workflow(
                repo=repo,
                workflow_file="cuda-gpu-gate.yml",
                ref=ref,
                fields=fields,
            )

    if needs_windows:
        fields = {
            "windows_cuda_run_id": windows_cuda_run_id,
            "metal_ci_run_id": metal_ci_run_id,
            "require_metal_compare": bool_input(args.require_metal),
            "machine": args.windows_machine,
            "zones": args.windows_zones,
            "uci_go": args.windows_uci_go,
            "stable_batch_size": args.stable_batch_size,
        }
        print(f"Dispatch Windows CUDA Runtime Gate with {fields}")
        if not args.dry_run:
            dispatch_workflow(
                repo=repo,
                workflow_file="windows-cuda-runtime-gate.yml",
                ref=ref,
                fields=fields,
            )

    if not args.dry_run:
        time.sleep(3)
        print(
            "Dispatched requested workflow(s). "
            "Use `gh run list --branch {}`.".format(ref)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
