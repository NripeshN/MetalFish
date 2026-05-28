#!/usr/bin/env python3
"""Dispatch same-commit CUDA release artifact promotion."""
from __future__ import annotations

import argparse
import sys
import time

try:
    import dispatch_cuda_runtime_gates as dispatch
except ModuleNotFoundError:
    from tools import dispatch_cuda_runtime_gates as dispatch


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="", help="GitHub repo, for example owner/name")
    parser.add_argument(
        "--ref",
        default="",
        help="Branch or tag to dispatch cuda-release.yml on.",
    )
    parser.add_argument(
        "--gate-ref",
        default="",
        help="Branch that contains the successful CUDA runtime gate runs.",
    )
    parser.add_argument(
        "--expected-sha",
        default="",
        help="Required source SHA. Defaults to the current checkout HEAD.",
    )
    parser.add_argument("--linux-cuda-run-id", default="")
    parser.add_argument("--windows-cuda-runtime-run-id", default="")
    parser.add_argument("--tag-name", default="")
    parser.add_argument("--lookup-limit", type=int, default=30)
    parser.add_argument(
        "--attach-to-release",
        action="store_true",
        help="Attach validated CUDA packages to the GitHub release named by tag.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and print dispatch without starting the workflow.",
    )
    return parser.parse_args(argv)


def run_successful_gate(
    *,
    repo: str,
    workflow: str,
    gate_ref: str,
    expected_sha: str,
    run_id: str,
    lookup_limit: int,
) -> dict:
    if run_id:
        return {"databaseId": run_id, "headSha": expected_sha, "workflowName": workflow}
    return dispatch.find_successful_run(
        repo=repo,
        workflow=workflow,
        branch=gate_ref,
        expected_sha=expected_sha,
        limit=lookup_limit,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo or dispatch.default_repo()
    ref = args.ref or dispatch.git_value(["rev-parse", "--abbrev-ref", "HEAD"])
    gate_ref = args.gate_ref or ref
    expected_sha = args.expected_sha or dispatch.git_value(["rev-parse", "HEAD"])

    linux_run = run_successful_gate(
        repo=repo,
        workflow="CUDA GPU Gate",
        gate_ref=gate_ref,
        expected_sha=expected_sha,
        run_id=args.linux_cuda_run_id,
        lookup_limit=args.lookup_limit,
    )
    windows_run = run_successful_gate(
        repo=repo,
        workflow="Windows CUDA Runtime Gate",
        gate_ref=gate_ref,
        expected_sha=expected_sha,
        run_id=args.windows_cuda_runtime_run_id,
        lookup_limit=args.lookup_limit,
    )

    if args.attach_to_release and not args.tag_name:
        raise RuntimeError("--tag-name is required with --attach-to-release")

    if not args.dry_run:
        dispatch.require_dispatchable_workflow(
            repo=repo, workflow_file="cuda-release.yml"
        )

    fields = {
        "linux_cuda_run_id": linux_run["databaseId"],
        "windows_cuda_runtime_run_id": windows_run["databaseId"],
        "expected_sha": expected_sha,
        "tag_name": args.tag_name,
        "attach_to_release": dispatch.bool_input(args.attach_to_release),
    }

    print(f"repo={repo}")
    print(f"ref={ref}")
    print(f"gate_ref={gate_ref}")
    print(f"expected_sha={expected_sha}")
    dispatch.print_run("CUDA GPU Gate", linux_run)
    dispatch.print_run("Windows CUDA Runtime Gate", windows_run)
    print(f"Dispatch CUDA Release Artifacts with {fields}")

    if not args.dry_run:
        dispatch.dispatch_workflow(
            repo=repo,
            workflow_file="cuda-release.yml",
            ref=ref,
            fields=fields,
        )
        time.sleep(3)
        print(
            "Dispatched CUDA release promotion. "
            "Use `gh run list --workflow cuda-release.yml --branch {}`.".format(ref)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
