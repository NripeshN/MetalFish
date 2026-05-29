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
import time
import zipfile

try:
    from tools.check_cuda_runtime_manifest import validate_runtime_manifest
    from tools.check_cuda_package_artifacts import (
        validate_linux_cuda_package,
        validate_windows_cuda_package,
    )
    from tools.github_cli import gh_cmd
except ModuleNotFoundError:
    from check_cuda_runtime_manifest import validate_runtime_manifest
    from check_cuda_package_artifacts import (
        validate_linux_cuda_package,
        validate_windows_cuda_package,
    )
    from github_cli import gh_cmd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOWNLOAD_RETRIES = 3


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
    data = run_json(gh_cmd("repo", "view", "--json", "nameWithOwner"))
    return str(data["nameWithOwner"])


def git_head() -> str:
    return run_text(["git", "rev-parse", "HEAD"])


def read_run(repo: str, run_id: str) -> RunInfo:
    data = run_json(
        gh_cmd(
            "run",
            "view",
            run_id,
            "--repo",
            repo,
            "--json",
            "conclusion,headSha,status,url,workflowName",
        )
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
        gh_cmd(
            "api",
            f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
        )
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


def complete_zip(path: pathlib.Path, *, expected_size: int) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if expected_size > 0 and path.stat().st_size != expected_size:
            return False
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def download_artifact(repo: str, artifact: ArtifactInfo, archive_path: pathlib.Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if complete_zip(archive_path, expected_size=artifact.size_in_bytes):
        return
    if archive_path.exists():
        archive_path.unlink()
    tmp_path = archive_path.with_name(f"{archive_path.name}.part")
    last_error = ""
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        if tmp_path.exists():
            tmp_path.unlink()
        with tmp_path.open("wb") as handle:
            proc = subprocess.run(
                gh_cmd(
                    "api",
                    f"repos/{repo}/actions/artifacts/{artifact.artifact_id}/zip",
                ),
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.PIPE,
            )
        if proc.returncode == 0 and complete_zip(
            tmp_path, expected_size=artifact.size_in_bytes
        ):
            tmp_path.replace(archive_path)
            return
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        last_error = (
            f"exit={proc.returncode} stderr={stderr.strip() or '<empty>'} "
            f"size={tmp_path.stat().st_size if tmp_path.exists() else 0} "
            f"expected={artifact.size_in_bytes}"
        )
        if tmp_path.exists():
            tmp_path.unlink()
        if attempt < DOWNLOAD_RETRIES:
            print(
                f"Retrying artifact download for {artifact.name} "
                f"({attempt}/{DOWNLOAD_RETRIES} failed): {last_error}",
                file=sys.stderr,
            )
            time.sleep(1.0)
    raise RuntimeError(f"failed to download artifact {artifact.name}: {last_error}")


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


def require_matching_file_record(actual: dict, expected: object, *, label: str) -> None:
    if not isinstance(expected, dict):
        raise ValueError(f"{label} expected record is missing")
    for field in ("size_bytes", "sha256"):
        if actual.get(field) != expected.get(field):
            raise ValueError(
                f"{label} record mismatch for {field}: "
                f"{actual.get(field)!r} != {expected.get(field)!r}"
            )


def require_linux_runtime_package_record(runtime: dict, package: pathlib.Path) -> None:
    artifacts = runtime.get("artifacts") or {}
    matches = [
        record
        for name, record in artifacts.items()
        if name.startswith("metalfish-linux-x86_64-cuda") and name.endswith(".tar.gz")
    ]
    if len(matches) != 1:
        raise ValueError(
            "Linux runtime manifest must contain exactly one Linux CUDA package "
            f"artifact, got {len(matches)}"
        )
    require_matching_file_record(
        file_record(package), matches[0], label="Linux runtime package"
    )


def require_windows_runtime_package_record(runtime: dict, package: pathlib.Path) -> None:
    package_input = (runtime.get("inputs") or {}).get("package") or {}
    require_matching_file_record(
        file_record(package),
        package_input.get("record"),
        label="Windows runtime package",
    )


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


def read_direct_runtime_manifest(root: pathlib.Path) -> dict:
    manifest = root / "direct-runtime-gates-manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"direct runtime manifest not found: {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8-sig"))


def validate_direct_runtime_manifest(
    manifest: dict, *, expected_sha: str | None = None
) -> None:
    if manifest.get("schema") != "metalfish.cuda_runtime_gates_direct":
        raise ValueError(
            "direct runtime manifest has unexpected schema: "
            f"{manifest.get('schema')!r}"
        )
    target = manifest.get("target")
    if target not in {"both", "linux", "windows"}:
        raise ValueError(
            "direct runtime manifest has unexpected target "
            f"{target!r}"
        )
    completed_targets = set()
    if target == "both":
        completed_targets.update({"linux", "windows"})
    elif target in {"linux", "windows"}:
        completed_targets.add(target)
    completed_targets.update(
        target
        for target in manifest.get("completed_targets") or []
        if target in {"linux", "windows"}
    )
    if not {"linux", "windows"}.issubset(completed_targets):
        raise ValueError(
            "direct runtime manifest did not complete both Linux and Windows gates"
        )
    if manifest.get("require_metal") is not True:
        raise ValueError("direct runtime manifest did not require Metal comparison")
    if expected_sha and manifest.get("expected_sha") != expected_sha:
        raise ValueError(
            "direct runtime manifest has unexpected expected_sha: "
            f"{manifest.get('expected_sha')!r}; expected {expected_sha}"
        )


def fetch_direct_runtime_artifacts(
    root: pathlib.Path,
) -> tuple[dict, dict, pathlib.Path, pathlib.Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"direct runtime root not found: {root}")
    linux_dir = root / "linux"
    windows_dir = root / "windows"
    windows_inputs_dir = root / "windows_cuda_runtime_inputs"
    linux_package = find_one(
        linux_dir, "metalfish*linux-x86_64-cuda.tar.gz", "Linux CUDA package"
    )
    linux_runtime_manifest = find_one(
        linux_dir, "cuda-gpu-runtime-manifest.json", "Linux CUDA runtime manifest"
    )
    windows_package = find_one(
        windows_inputs_dir,
        "metalfish*windows-x86_64-msvc-cuda.zip",
        "Windows CUDA package",
    )
    windows_runtime_manifest = find_one(
        windows_dir,
        "windows-cuda-runtime-gate-manifest.json",
        "Windows CUDA runtime manifest",
    )
    return (
        {
            "mode": "direct-runtime-root",
            "root": str(root),
            "runtime_manifest": str(linux_runtime_manifest),
        },
        {
            "mode": "direct-runtime-root",
            "root": str(root),
            "runtime_manifest": str(windows_runtime_manifest),
        },
        linux_package,
        windows_package,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="", help="GitHub repo, for example owner/name")
    parser.add_argument("--linux-cuda-run-id", default="")
    parser.add_argument("--windows-cuda-runtime-run-id", default="")
    parser.add_argument(
        "--direct-runtime-root",
        default="",
        help=(
            "Promote packages from a direct runtime gate root, for example "
            "results/cuda_runtime_direct/<sha>. This is mutually exclusive "
            "with GitHub run IDs."
        ),
    )
    parser.add_argument("--out-dir", default="results/cuda_release_artifacts")
    parser.add_argument(
        "--expected-sha",
        default="",
        help="Expected commit SHA. Defaults to the current checkout HEAD.",
    )
    parser.add_argument(
        "--tag-name",
        default="",
        help="Optional release tag used to rename promoted packages.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    direct_root = (
        pathlib.Path(args.direct_runtime_root).expanduser().resolve()
        if args.direct_runtime_root
        else None
    )
    if direct_root and (args.linux_cuda_run_id or args.windows_cuda_runtime_run_id):
        raise ValueError("--direct-runtime-root is mutually exclusive with run IDs")
    if not direct_root and not (args.linux_cuda_run_id and args.windows_cuda_runtime_run_id):
        raise ValueError(
            "provide --direct-runtime-root or both --linux-cuda-run-id and "
            "--windows-cuda-runtime-run-id"
        )
    direct_manifest = read_direct_runtime_manifest(direct_root) if direct_root else {}
    repo = args.repo or str(direct_manifest.get("repo") or "") or default_repo()
    direct_expected_sha = str(direct_manifest.get("expected_sha") or "")
    expected_sha = args.expected_sha or direct_expected_sha or git_head()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    if direct_root:
        if out_dir == direct_root:
            raise ValueError("--out-dir must not equal --direct-runtime-root")
        try:
            direct_root.relative_to(out_dir)
        except ValueError:
            pass
        else:
            raise ValueError("--out-dir must not contain --direct-runtime-root")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if direct_root:
        validate_direct_runtime_manifest(direct_manifest, expected_sha=expected_sha)

    if direct_root:
        linux_run, windows_run, linux_package, windows_package = (
            fetch_direct_runtime_artifacts(direct_root)
        )
        linux_artifact = {
            "mode": "direct-runtime-root",
            "name": linux_package.name,
            "record": file_record(linux_package),
        }
        windows_artifact = {
            "mode": "direct-runtime-root",
            "name": windows_package.name,
            "record": file_record(windows_package),
        }
        linux_runtime_manifest = pathlib.Path(linux_run["runtime_manifest"])
        windows_runtime_manifest = pathlib.Path(windows_run["runtime_manifest"])
    else:
        linux_run_info, linux_artifact_info, linux_dir = fetch_gate_artifact(
            repo=repo,
            run_id=args.linux_cuda_run_id,
            workflow="CUDA GPU Gate",
            expected_sha=expected_sha,
            artifact_pattern="cuda-gpu-gate-*",
            out_dir=out_dir,
        )
        windows_run_info, windows_artifact_info, windows_dir = fetch_gate_artifact(
            repo=repo,
            run_id=args.windows_cuda_runtime_run_id,
            workflow="Windows CUDA Runtime Gate",
            expected_sha=expected_sha,
            artifact_pattern="windows-cuda-runtime-*",
            out_dir=out_dir,
        )
        linux_run = dataclasses.asdict(linux_run_info)
        windows_run = dataclasses.asdict(windows_run_info)
        linux_artifact = dataclasses.asdict(linux_artifact_info)
        windows_artifact = dataclasses.asdict(windows_artifact_info)
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

    linux_package_manifest = validate_linux_cuda_package(
        linux_package,
        expected_source_commit=expected_sha,
    )
    windows_package_manifest = validate_windows_cuda_package(
        windows_package,
        expected_source_commit=expected_sha,
    )
    linux_runtime = validate_runtime_manifest(
        linux_runtime_manifest,
        runtime_kind="linux-cuda",
        require_metal_compare=True,
        require_metal_benchmark_compare=True,
        require_metal_search_compare=True,
        require_release_evidence=True,
        require_release_policy=True,
        require_observed_runtime=True,
        require_artifact_files=True,
        artifact_root=linux_runtime_manifest.parent,
        expected_head_sha=expected_sha,
    )
    windows_runtime = validate_runtime_manifest(
        windows_runtime_manifest,
        runtime_kind="windows-cuda",
        require_metal_compare=True,
        require_metal_benchmark_compare=True,
        require_metal_search_compare=True,
        require_release_evidence=True,
        require_release_policy=True,
        require_observed_runtime=True,
        require_artifact_files=True,
        artifact_root=windows_runtime_manifest.parent,
        expected_head_sha=expected_sha,
    )
    require_linux_runtime_package_record(linux_runtime, linux_package)
    require_windows_runtime_package_record(windows_runtime, windows_package)

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
            "linux_cuda": linux_run,
            "windows_cuda_runtime": windows_run,
        },
        "source_artifacts": {
            "linux_cuda": linux_artifact,
            "windows_cuda_runtime": windows_artifact,
        },
        "direct_runtime": direct_manifest if direct_root else None,
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
