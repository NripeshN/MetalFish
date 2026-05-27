#!/usr/bin/env python3
"""Fetch and validate same-commit CUDA release packages."""
from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import zipfile

try:
    from tools.check_cuda_package_artifacts import (
        validate_linux_cuda_package,
        validate_windows_cuda_package,
    )
except ModuleNotFoundError:
    from check_cuda_package_artifacts import (
        validate_linux_cuda_package,
        validate_windows_cuda_package,
    )

ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclasses.dataclass(frozen=True)
class RunInfo:
    run_id: str
    workflow_name: str
    status: str
    conclusion: str
    head_sha: str
    url: str


@dataclasses.dataclass(frozen=True)
class ArtifactInfo:
    artifact_id: int
    name: str
    size_in_bytes: int
    archive_download_url: str


def run_text(cmd: list[str], *, stdout=None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=stdout is None,
        stdout=stdout if stdout is not None else subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if stdout is not None:
        return ""
    return proc.stdout.strip()


def run_json(cmd: list[str]) -> dict:
    text = run_text(cmd)
    return json.loads(text) if text else {}


def default_repo() -> str:
    data = run_json(["gh", "repo", "view", "--json", "nameWithOwner"])
    return str(data["nameWithOwner"])


def git_head() -> str:
    return run_text(["git", "rev-parse", "HEAD"])


def read_run(repo: str, run_id: str) -> RunInfo:
    data = run_json(
        [
            "gh",
            "run",
            "view",
            run_id,
            "--repo",
            repo,
            "--json",
            "conclusion,headSha,status,url,workflowName",
        ]
    )
    return RunInfo(
        run_id=run_id,
        workflow_name=str(data["workflowName"]),
        status=str(data["status"]),
        conclusion=str(data["conclusion"]),
        head_sha=str(data["headSha"]),
        url=str(data["url"]),
    )


def require_run(
    run: RunInfo, *, expected_workflow: str, expected_sha: str | None
) -> None:
    if run.workflow_name != expected_workflow:
        raise ValueError(
            f"run {run.run_id} must be from {expected_workflow!r}, "
            f"got {run.workflow_name!r}: {run.url}"
        )
    if run.status != "completed" or run.conclusion != "success":
        raise ValueError(
            f"run {run.run_id} is not successful: "
            f"status={run.status} conclusion={run.conclusion}: {run.url}"
        )
    if expected_sha and run.head_sha != expected_sha:
        raise ValueError(
            f"run {run.run_id} is for {run.head_sha}, expected {expected_sha}: "
            f"{run.url}"
        )


def list_artifacts(repo: str, run_id: str) -> list[ArtifactInfo]:
    data = run_json(
        [
            "gh",
            "api",
            f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
        ]
    )
    return [
        ArtifactInfo(
            artifact_id=int(item["id"]),
            name=str(item["name"]),
            size_in_bytes=int(item["size_in_bytes"]),
            archive_download_url=str(item["archive_download_url"]),
        )
        for item in data.get("artifacts", [])
        if not item.get("expired", False)
    ]


def select_artifact(artifacts: list[ArtifactInfo], *, pattern: str) -> ArtifactInfo:
    matches = [artifact for artifact in artifacts if fnmatch.fnmatch(artifact.name, pattern)]
    if not matches:
        available = ", ".join(artifact.name for artifact in artifacts) or "<none>"
        raise ValueError(f"artifact {pattern!r} not found; available: {available}")
    if len(matches) > 1:
        raise ValueError(
            f"artifact {pattern!r} matched multiple artifacts: "
            + ", ".join(artifact.name for artifact in matches)
        )
    return matches[0]


def download_artifact(repo: str, artifact: ArtifactInfo, archive_path: pathlib.Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as handle:
        run_text(
            [
                "gh",
                "api",
                f"repos/{repo}/actions/artifacts/{artifact.artifact_id}/zip",
            ],
            stdout=handle,
        )


def safe_extract_zip(archive_path: pathlib.Path, dest: pathlib.Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (dest / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"zip member escapes extraction root: {member.filename}")
        archive.extractall(dest)


def find_one(root: pathlib.Path, pattern: str, label: str) -> pathlib.Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"{label} not found under {root}")
    unique = {path.resolve() for path in matches}
    if len(unique) != 1:
        raise RuntimeError(
            f"{label} matched multiple files under {root}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: pathlib.Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_zero_status(status: object, *, label: str) -> None:
    if str(status) != "0":
        raise ValueError(f"{label} must be 0 for release promotion, got {status!r}")


def validate_runtime_manifest(path: pathlib.Path, *, schema: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema") != schema:
        raise ValueError(
            f"runtime manifest {path} has unexpected schema: {data.get('schema')!r}"
        )
    status = data.get("status") or {}
    if schema == "metalfish.cuda_gpu_runtime_gate":
        require_zero_status(status.get("remote_status"), label="Linux remote_status")
    else:
        require_zero_status(status.get("runtime_status"), label="Windows runtime_status")
    require_zero_status(status.get("bt4_compare_status"), label="BT4 compare status")
    require_zero_status(status.get("legacy_compare_status"), label="legacy compare status")
    require_zero_status(status.get("final_compare_status"), label="final compare status")
    return {
        "schema": data["schema"],
        "status": status,
        "gcp": data.get("gcp") or {},
        "inputs": data.get("inputs") or {},
    }


def release_package_name(path: pathlib.Path, *, tag_name: str, platform: str) -> str:
    if not tag_name:
        return path.name
    suffix = ".tar.gz" if path.name.endswith(".tar.gz") else path.suffix
    return f"metalfish-{tag_name}-{platform}{suffix}"


def copy_release_package(
    source: pathlib.Path,
    packages_dir: pathlib.Path,
    *,
    tag_name: str,
    platform: str,
) -> pathlib.Path:
    packages_dir.mkdir(parents=True, exist_ok=True)
    target = packages_dir / release_package_name(
        source, tag_name=tag_name, platform=platform
    )
    shutil.copy2(source, target)
    return target


def fetch_gate_artifact(
    *,
    repo: str,
    run_id: str,
    workflow: str,
    expected_sha: str | None,
    artifact_pattern: str,
    out_dir: pathlib.Path,
) -> tuple[RunInfo, ArtifactInfo, pathlib.Path]:
    run = read_run(repo, run_id)
    require_run(run, expected_workflow=workflow, expected_sha=expected_sha)
    artifact = select_artifact(list_artifacts(repo, run_id), pattern=artifact_pattern)
    archive = out_dir / "downloads" / f"{artifact.name}.zip"
    extract_dir = out_dir / "extracted" / artifact.name
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    download_artifact(repo, artifact, archive)
    safe_extract_zip(archive, extract_dir)
    return run, artifact, extract_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="", help="GitHub repo, for example owner/name")
    parser.add_argument("--linux-cuda-run-id", required=True)
    parser.add_argument("--windows-cuda-runtime-run-id", required=True)
    parser.add_argument("--out-dir", default="results/cuda_release_artifacts")
    parser.add_argument(
        "--expected-sha",
        default="",
        help="Expected commit SHA. Defaults to the current checkout HEAD.",
    )
    parser.add_argument(
        "--no-expected-sha",
        action="store_true",
        help="Skip same-commit provenance checks.",
    )
    parser.add_argument(
        "--tag-name",
        default="",
        help="Optional release tag used to rename promoted packages.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo or default_repo()
    expected_sha = None if args.no_expected_sha else (args.expected_sha or git_head())
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    linux_run, linux_artifact, linux_dir = fetch_gate_artifact(
        repo=repo,
        run_id=args.linux_cuda_run_id,
        workflow="CUDA GPU Gate",
        expected_sha=expected_sha,
        artifact_pattern="cuda-gpu-gate-*",
        out_dir=out_dir,
    )
    windows_run, windows_artifact, windows_dir = fetch_gate_artifact(
        repo=repo,
        run_id=args.windows_cuda_runtime_run_id,
        workflow="Windows CUDA Runtime Gate",
        expected_sha=expected_sha,
        artifact_pattern="windows-cuda-runtime-*",
        out_dir=out_dir,
    )

    linux_package = find_one(
        linux_dir, "metalfish*linux-x86_64-cuda.tar.gz", "Linux CUDA package"
    )
    linux_runtime_manifest = find_one(
        linux_dir, "cuda-gpu-runtime-manifest.json", "Linux CUDA runtime manifest"
    )
    windows_package = find_one(
        windows_dir,
        "metalfish*windows-x86_64-msvc-cuda.zip",
        "Windows CUDA package",
    )
    windows_runtime_manifest = find_one(
        windows_dir,
        "windows-cuda-runtime-gate-manifest.json",
        "Windows CUDA runtime manifest",
    )

    linux_package_manifest = validate_linux_cuda_package(linux_package)
    windows_package_manifest = validate_windows_cuda_package(windows_package)
    linux_runtime = validate_runtime_manifest(
        linux_runtime_manifest, schema="metalfish.cuda_gpu_runtime_gate"
    )
    windows_runtime = validate_runtime_manifest(
        windows_runtime_manifest, schema="metalfish.windows_cuda_runtime_gate"
    )

    packages_dir = out_dir / "packages"
    release_linux_package = copy_release_package(
        linux_package,
        packages_dir,
        tag_name=args.tag_name,
        platform="linux-x86_64-cuda",
    )
    release_windows_package = copy_release_package(
        windows_package,
        packages_dir,
        tag_name=args.tag_name,
        platform="windows-x86_64-msvc-cuda",
    )
    manifest = {
        "schema": "metalfish.cuda_release_artifacts",
        "schema_version": 1,
        "repo": repo,
        "expected_sha": expected_sha,
        "tag_name": args.tag_name,
        "runs": {
            "linux_cuda": dataclasses.asdict(linux_run),
            "windows_cuda_runtime": dataclasses.asdict(windows_run),
        },
        "source_artifacts": {
            "linux_cuda": dataclasses.asdict(linux_artifact),
            "windows_cuda_runtime": dataclasses.asdict(windows_artifact),
        },
        "packages": {
            "linux_cuda": {
                "source": file_record(linux_package),
                "release": file_record(release_linux_package),
                "manifest": linux_package_manifest,
                "runtime": linux_runtime,
            },
            "windows_cuda": {
                "source": file_record(windows_package),
                "release": file_record(release_windows_package),
                "manifest": windows_package_manifest,
                "runtime": windows_runtime,
            },
        },
    }
    manifest_path = out_dir / "cuda-release-artifacts-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Promoted Linux CUDA package: {release_linux_package}")
    print(f"Promoted Windows CUDA package: {release_windows_package}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
