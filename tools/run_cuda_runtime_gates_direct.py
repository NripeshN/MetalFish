#!/usr/bin/env python3
"""Run same-commit CUDA runtime gates directly from a clean worktree."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time

import dispatch_cuda_runtime_gates as dispatch


ROOT = pathlib.Path(__file__).resolve().parent.parent


def run_text(cmd: list[str], *, cwd: pathlib.Path = ROOT) -> str:
    return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.PIPE, text=True).strip()


def run_command(
    cmd: list[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
    dry_run: bool,
) -> None:
    print(f"+ cd {cwd} && {shlex.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def git_value(args: list[str], *, cwd: pathlib.Path = ROOT) -> str:
    return run_text(["git", *args], cwd=cwd)


def default_repo() -> str:
    data = dispatch.run_json(["gh", "repo", "view", "--json", "nameWithOwner"])
    if not isinstance(data, dict):
        raise RuntimeError("gh repo view returned invalid JSON")
    return str(data["nameWithOwner"])


def bool_env(value: bool) -> str:
    return "1" if value else "0"


def compact_instance_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in name.lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"metalfish-{cleaned}"
    return cleaned[:63].rstrip("-")


def remove_tree(path: pathlib.Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def write_manifest(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def shell_source_command(env_file: pathlib.Path, tool: str) -> list[str]:
    return ["bash", "-lc", f". {shlex.quote(str(env_file))}; {shlex.quote(tool)}"]


def direct_command(tool: str) -> list[str]:
    return ["bash", "-lc", shlex.quote(tool)]


def add_common_gcp_env(env: dict[str, str], args: argparse.Namespace) -> None:
    if args.gcp_project:
        env["METALFISH_GCP_PROJECT"] = args.gcp_project
    env["METALFISH_GCP_COLLECT_ARTIFACTS"] = bool_env(args.collect_artifacts)
    env["METALFISH_GCP_DELETE_ON_EXIT"] = "0" if args.keep_vms else "1"


def resolve_runs(args: argparse.Namespace, repo: str, ref: str, expected_sha: str) -> dict:
    needs_windows = args.target in {"windows", "both"}
    metal_run = None
    windows_run = None
    metal_ci_run_id = args.metal_ci_run_id
    windows_cuda_run_id = args.windows_cuda_run_id

    if args.require_metal and not metal_ci_run_id:
        metal_run = dispatch.find_successful_run(
            repo=repo,
            workflow="MetalFish CI",
            branch=ref,
            expected_sha=expected_sha,
            limit=args.lookup_limit,
        )
        metal_ci_run_id = str(metal_run["databaseId"])
    elif metal_ci_run_id:
        metal_run = {"databaseId": metal_ci_run_id, "headSha": expected_sha}

    if needs_windows and not windows_cuda_run_id:
        windows_run = dispatch.find_successful_run(
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
    if needs_windows and not windows_cuda_run_id:
        raise RuntimeError("Windows CUDA Compile Gate run id is required for Windows")

    return {
        "metal_ci_run_id": metal_ci_run_id,
        "windows_cuda_run_id": windows_cuda_run_id,
        "metal_run": metal_run,
        "windows_run": windows_run,
    }


def fetch_linux_inputs(
    *,
    args: argparse.Namespace,
    repo: str,
    expected_sha: str,
    metal_ci_run_id: str,
    worktree: pathlib.Path,
    input_dir: pathlib.Path,
    dry_run: bool,
) -> pathlib.Path:
    if not dry_run:
        remove_tree(input_dir)
    cmd = [
        sys.executable,
        "tools/fetch_cuda_gpu_gate_inputs.py",
        "--repo",
        repo,
        "--metal-ci-run-id",
        metal_ci_run_id,
        "--expected-sha",
        expected_sha,
        "--out-dir",
        str(input_dir),
    ]
    run_command(cmd, cwd=worktree, dry_run=dry_run)
    return input_dir / "cuda-gpu-gate-env.sh"


def fetch_windows_inputs(
    *,
    args: argparse.Namespace,
    repo: str,
    expected_sha: str,
    windows_cuda_run_id: str,
    metal_ci_run_id: str,
    worktree: pathlib.Path,
    input_dir: pathlib.Path,
    dry_run: bool,
) -> pathlib.Path:
    if not dry_run:
        remove_tree(input_dir)
    cmd = [
        sys.executable,
        "tools/fetch_windows_cuda_runtime_inputs.py",
        "--repo",
        repo,
        "--windows-cuda-run-id",
        windows_cuda_run_id,
        "--expected-sha",
        expected_sha,
        "--out-dir",
        str(input_dir),
    ]
    if args.require_metal:
        cmd.extend(["--metal-ci-run-id", metal_ci_run_id, "--require-metal"])
    run_command(cmd, cwd=worktree, dry_run=dry_run)
    return input_dir / "runtime-gate-env.sh"


def run_network_downloads(
    *, args: argparse.Namespace, worktree: pathlib.Path, dry_run: bool
) -> None:
    if args.skip_network_download:
        return
    run_command(
        [sys.executable, "tools/download_engine_networks.py", "--include-legacy"],
        cwd=worktree,
        dry_run=dry_run,
    )


def run_linux_gate(
    *,
    args: argparse.Namespace,
    worktree: pathlib.Path,
    artifact_root: pathlib.Path,
    linux_env_file: pathlib.Path | None,
    linux_instance: str,
    dry_run: bool,
) -> None:
    env = os.environ.copy()
    add_common_gcp_env(env, args)
    env["METALFISH_GCP_INSTANCE"] = linux_instance
    env["METALFISH_GCP_ARTIFACT_DIR"] = str(artifact_root / "linux")
    env["METALFISH_GCP_MACHINE"] = args.linux_machine
    env["METALFISH_CUDA_UCI_GO"] = args.linux_uci_go
    env["METALFISH_CUDA_STABLE_EXECUTION_BATCH_SIZE"] = args.stable_batch_size
    if args.linux_zones:
        env["METALFISH_GCP_ZONES"] = args.linux_zones
    if linux_env_file is not None:
        cmd = shell_source_command(linux_env_file, "tools/run_gcp_cuda_gpu_gate.sh")
    else:
        cmd = direct_command("tools/run_gcp_cuda_gpu_gate.sh")
    run_command(cmd, cwd=worktree, env=env, dry_run=dry_run)


def run_windows_gate(
    *,
    args: argparse.Namespace,
    worktree: pathlib.Path,
    artifact_root: pathlib.Path,
    windows_env_file: pathlib.Path,
    windows_instance: str,
    dry_run: bool,
) -> None:
    env = os.environ.copy()
    add_common_gcp_env(env, args)
    env["METALFISH_GCP_INSTANCE"] = windows_instance
    env["METALFISH_GCP_ARTIFACT_DIR"] = str(artifact_root / "windows")
    env["METALFISH_GCP_MACHINES"] = args.windows_machines
    env["METALFISH_WINDOWS_CUDA_UCI_GO"] = args.windows_uci_go
    env["METALFISH_WINDOWS_CUDA_STABLE_EXECUTION_BATCH_SIZE"] = args.stable_batch_size
    if args.windows_zones:
        env["METALFISH_GCP_ZONES"] = args.windows_zones
    run_command(
        shell_source_command(windows_env_file, "tools/run_gcp_windows_cuda_runtime_gate.sh"),
        cwd=worktree,
        env=env,
        dry_run=dry_run,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="", help="GitHub repo, for example owner/name")
    parser.add_argument("--ref", default="", help="Branch to inspect for CI runs")
    parser.add_argument("--expected-sha", default="", help="Source SHA to validate")
    parser.add_argument(
        "--target",
        choices=("linux", "windows", "both"),
        default="both",
        help="Runtime gate target to run.",
    )
    parser.add_argument("--metal-ci-run-id", default="")
    parser.add_argument("--windows-cuda-run-id", default="")
    parser.add_argument(
        "--require-metal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require same-commit Metal probe artifacts for CUDA comparison.",
    )
    parser.add_argument("--lookup-limit", type=int, default=30)
    parser.add_argument("--stable-batch-size", default="16")
    parser.add_argument("--gcp-project", default=os.environ.get("METALFISH_GCP_PROJECT", "metalfish"))
    parser.add_argument("--instance-prefix", default="metalfish-cuda-direct")
    parser.add_argument("--linux-instance", default="")
    parser.add_argument("--windows-instance", default="")
    parser.add_argument("--linux-machine", default="g2-standard-8")
    parser.add_argument("--linux-zones", default="")
    parser.add_argument("--linux-uci-go", default="nodes 8")
    parser.add_argument("--windows-machines", default="g2-standard-8 g2-standard-4")
    parser.add_argument("--windows-zones", default="")
    parser.add_argument("--windows-uci-go", default="nodes 1")
    parser.add_argument(
        "--worktree-dir",
        default="",
        help="Detached worktree path. Defaults to /tmp/metalfish-cuda-runtime-<sha>.",
    )
    parser.add_argument(
        "--artifact-root",
        default="",
        help="Artifact/input directory. Defaults to results/cuda_runtime_direct/<sha>.",
    )
    parser.add_argument(
        "--cleanup-worktree",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove the detached worktree after a successful run.",
    )
    parser.add_argument(
        "--collect-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect remote GCP logs and packages into the artifact root.",
    )
    parser.add_argument(
        "--keep-vms",
        action="store_true",
        help="Leave GCP VMs running for interactive debugging.",
    )
    parser.add_argument(
        "--skip-network-download",
        action="store_true",
        help="Skip local NNUE/BT4/legacy weight downloads before Windows runtime.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve CI inputs and print commands without creating worktrees or VMs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo or default_repo()
    ref = args.ref or git_value(["rev-parse", "--abbrev-ref", "HEAD"])
    expected_sha = args.expected_sha or git_value(["rev-parse", "HEAD"])
    short_sha = expected_sha[:8]
    needs_linux = args.target in {"linux", "both"}
    needs_windows = args.target in {"windows", "both"}
    runs = resolve_runs(args, repo, ref, expected_sha)
    artifact_root = (
        pathlib.Path(args.artifact_root).expanduser().resolve()
        if args.artifact_root
        else (ROOT / "results" / "cuda_runtime_direct" / short_sha).resolve()
    )
    worktree = (
        pathlib.Path(args.worktree_dir).expanduser().resolve()
        if args.worktree_dir
        else pathlib.Path(f"/tmp/metalfish-cuda-runtime-{short_sha}")
    )
    linux_instance = compact_instance_name(
        args.linux_instance or f"{args.instance_prefix}-linux-{short_sha}"
    )
    windows_instance = compact_instance_name(
        args.windows_instance or f"{args.instance_prefix}-win-{short_sha}"
    )
    manifest_path = artifact_root / "direct-runtime-gates-manifest.json"
    created_worktree = False
    success = False

    print(f"repo={repo}")
    print(f"ref={ref}")
    print(f"expected_sha={expected_sha}")
    print(f"worktree={worktree}")
    print(f"artifact_root={artifact_root}")
    if runs["metal_run"]:
        dispatch.print_run("MetalFish CI", runs["metal_run"])
    if runs["windows_run"]:
        dispatch.print_run("Windows CUDA Compile Gate", runs["windows_run"])

    if not args.dry_run:
        if worktree.exists():
            raise FileExistsError(
                f"worktree path already exists: {worktree}. "
                "Use --worktree-dir for a different path or remove the stale tree."
            )
        run_command(
            ["git", "worktree", "add", "--detach", str(worktree), expected_sha],
            cwd=ROOT,
            dry_run=False,
        )
        created_worktree = True
        artifact_root.mkdir(parents=True, exist_ok=True)
    else:
        print("+ dry-run: skip git worktree add and GCP execution", flush=True)

    try:
        linux_env_file = None
        if needs_linux and args.require_metal:
            linux_env_file = fetch_linux_inputs(
                args=args,
                repo=repo,
                expected_sha=expected_sha,
                metal_ci_run_id=str(runs["metal_ci_run_id"]),
                worktree=worktree,
                input_dir=artifact_root / "cuda_gpu_gate_inputs",
                dry_run=args.dry_run,
            )

        windows_env_file = None
        if needs_windows:
            windows_env_file = fetch_windows_inputs(
                args=args,
                repo=repo,
                expected_sha=expected_sha,
                windows_cuda_run_id=str(runs["windows_cuda_run_id"]),
                metal_ci_run_id=str(runs["metal_ci_run_id"] or ""),
                worktree=worktree,
                input_dir=artifact_root / "windows_cuda_runtime_inputs",
                dry_run=args.dry_run,
            )
            run_network_downloads(args=args, worktree=worktree, dry_run=args.dry_run)

        if needs_linux:
            run_linux_gate(
                args=args,
                worktree=worktree,
                artifact_root=artifact_root,
                linux_env_file=linux_env_file,
                linux_instance=linux_instance,
                dry_run=args.dry_run,
            )

        if needs_windows:
            assert windows_env_file is not None
            run_windows_gate(
                args=args,
                worktree=worktree,
                artifact_root=artifact_root,
                windows_env_file=windows_env_file,
                windows_instance=windows_instance,
                dry_run=args.dry_run,
            )

        manifest = {
            "schema": "metalfish.cuda_runtime_gates_direct",
            "schema_version": 1,
            "created_at_unix": int(time.time()),
            "repo": repo,
            "ref": ref,
            "expected_sha": expected_sha,
            "target": args.target,
            "require_metal": args.require_metal,
            "metal_ci_run_id": runs["metal_ci_run_id"],
            "windows_cuda_run_id": runs["windows_cuda_run_id"],
            "worktree": str(worktree),
            "artifact_root": str(artifact_root),
            "linux_instance": linux_instance if needs_linux else None,
            "windows_instance": windows_instance if needs_windows else None,
            "stable_batch_size": args.stable_batch_size,
            "linux_machine": args.linux_machine if needs_linux else None,
            "windows_machines": args.windows_machines if needs_windows else None,
        }
        if not args.dry_run:
            write_manifest(manifest_path, manifest)
            print(f"Wrote {manifest_path}")
        else:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        success = True
    finally:
        if (
            success
            and created_worktree
            and args.cleanup_worktree
            and not args.dry_run
        ):
            run_command(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=ROOT,
                dry_run=False,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
