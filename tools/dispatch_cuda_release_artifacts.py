#!/usr/bin/env python3
"""Dispatch same-commit CUDA release artifact promotion."""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

try:
    import dispatch_cuda_runtime_gates as dispatch
    import fetch_cuda_release_artifacts as cuda_release
except ModuleNotFoundError:
    from tools import dispatch_cuda_runtime_gates as dispatch
    from tools import fetch_cuda_release_artifacts as cuda_release


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
    parser.add_argument(
        "--direct-runtime-root",
        default="",
        help=(
            "Promote packages locally from a direct runtime root, for example "
            "results/cuda_runtime_direct/<sha>, instead of dispatching the "
            "GitHub release workflow."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="results/cuda_release_artifacts",
        help="Output directory used with --direct-runtime-root.",
    )
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


def attach_local_release_artifacts(
    *, repo: str, tag_name: str, out_dir: pathlib.Path, dry_run: bool
) -> None:
    package_dir = out_dir / "packages"
    files = [
        *sorted(package_dir.glob("*")),
        out_dir / "cuda-release-artifacts-manifest.json",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError("release artifacts missing: " + ", ".join(missing))
    cmd = [
        "gh",
        "release",
        "upload",
        tag_name,
        *[str(path) for path in files],
        "--repo",
        repo,
        "--clobber",
    ]
    if dry_run:
        print("+ " + " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def promote_direct_runtime_root(args: argparse.Namespace) -> int:
    if args.linux_cuda_run_id or args.windows_cuda_runtime_run_id:
        raise RuntimeError("--direct-runtime-root is mutually exclusive with run IDs")
    if args.attach_to_release and not args.tag_name:
        raise RuntimeError("--tag-name is required with --attach-to-release")

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    promote_args = [
        "--direct-runtime-root",
        args.direct_runtime_root,
        "--out-dir",
        str(out_dir),
    ]
    if args.expected_sha:
        promote_args.extend(["--expected-sha", args.expected_sha])
    if args.tag_name:
        promote_args.extend(["--tag-name", args.tag_name])

    print("Local direct CUDA release promotion")
    print(f"direct_runtime_root={args.direct_runtime_root}")
    print(f"out_dir={out_dir}")
    if args.expected_sha:
        print(f"expected_sha={args.expected_sha}")
    else:
        print("expected_sha=<from direct runtime manifest>")
    if args.dry_run:
        print("dry_run=true")
        print("+ python3 tools/fetch_cuda_release_artifacts.py " + " ".join(promote_args))
        return 0

    result = cuda_release.main(promote_args)
    if result != 0:
        return result

    manifest_path = out_dir / "cuda-release-artifacts-manifest.json"
    print("Validated CUDA release manifest:")
    print(f"  {manifest_path}")
    print("Release packages:")
    for package in sorted((out_dir / "packages").glob("*")):
        print(f"  {package}")

    if args.attach_to_release:
        repo = args.repo or dispatch.default_repo()
        attach_local_release_artifacts(
            repo=repo, tag_name=args.tag_name, out_dir=out_dir, dry_run=False
        )
        print(f"Attached CUDA packages to GitHub release {args.tag_name}")
    return 0


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
        data = dispatch.run_json(
            [
                "gh",
                "run",
                "view",
                str(run_id),
                "--repo",
                repo,
                "--json",
                "conclusion,createdAt,databaseId,headSha,status,url,workflowName",
            ]
        )
        if not isinstance(data, dict):
            raise RuntimeError(f"gh run view returned invalid JSON for {workflow}")
        if data.get("workflowName") != workflow:
            raise RuntimeError(
                f"run {run_id} is {data.get('workflowName')!r}, "
                f"expected {workflow!r}"
            )
        if data.get("headSha") != expected_sha:
            raise RuntimeError(
                f"run {run_id} is for {data.get('headSha')!r}, "
                f"expected {expected_sha!r}"
            )
        if data.get("status") != "completed" or data.get("conclusion") != "success":
            raise RuntimeError(
                f"run {run_id} is not successful: "
                f"status={data.get('status')!r} "
                f"conclusion={data.get('conclusion')!r}"
            )
        return data
    return dispatch.find_successful_run(
        repo=repo,
        workflow=workflow,
        branch=gate_ref,
        expected_sha=expected_sha,
        limit=lookup_limit,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.direct_runtime_root:
        return promote_direct_runtime_root(args)

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
