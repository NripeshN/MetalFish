#!/usr/bin/env python3
"""Fetch inputs for the direct Windows CUDA runtime gate."""
from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import time
import zipfile

try:
    from tools.check_cuda_package_artifacts import validate_windows_cuda_package
except ModuleNotFoundError:
    from check_cuda_package_artifacts import validate_windows_cuda_package


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
    return proc.stdout


def run_json(cmd: list[str]) -> dict:
    return json.loads(run_text(cmd))


def default_repo() -> str:
    data = run_json(["gh", "repo", "view", "--json", "nameWithOwner"])
    return str(data["nameWithOwner"])


def git_head() -> str:
    return run_text(["git", "rev-parse", "HEAD"]).strip()


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


def require_run_provenance(
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


def select_artifact(
    artifacts: list[ArtifactInfo], *, name: str | None = None, pattern: str | None = None
) -> ArtifactInfo:
    if bool(name) == bool(pattern):
        raise ValueError("select exactly one of name or pattern")
    if name:
        matches = [artifact for artifact in artifacts if artifact.name == name]
        label = name
    else:
        assert pattern is not None
        matches = [
            artifact for artifact in artifacts if fnmatch.fnmatch(artifact.name, pattern)
        ]
        label = pattern
    if not matches:
        available = ", ".join(artifact.name for artifact in artifacts) or "<none>"
        raise ValueError(f"artifact {label!r} not found; available: {available}")
    if len(matches) > 1:
        raise ValueError(
            f"artifact {label!r} matched multiple artifacts: "
            + ", ".join(artifact.name for artifact in matches)
        )
    return matches[0]


def complete_zip(path: pathlib.Path, *, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if expected_size > 0 and path.stat().st_size != expected_size:
        return False
    return zipfile.is_zipfile(path)


def download_artifact(repo: str, artifact: ArtifactInfo, archive_path: pathlib.Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if complete_zip(archive_path, expected_size=artifact.size_in_bytes):
        return
    tmp_path = archive_path.with_name(f"{archive_path.name}.part")
    last_error = ""
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        if tmp_path.exists():
            tmp_path.unlink()
        with tmp_path.open("wb") as handle:
            proc = subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{repo}/actions/artifacts/{artifact.artifact_id}/zip",
                ],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.PIPE,
            )
        if proc.returncode == 0 and complete_zip(
            tmp_path, expected_size=artifact.size_in_bytes
        ):
            tmp_path.replace(archive_path)
            return
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            last_error = f"gh api exited {proc.returncode}: {stderr}"
        else:
            last_error = (
                f"downloaded {tmp_path.stat().st_size} bytes, expected "
                f"{artifact.size_in_bytes} byte ZIP"
            )
        if tmp_path.exists():
            tmp_path.unlink()
        if attempt < DOWNLOAD_RETRIES:
            time.sleep(min(2 * attempt, 5))
    raise RuntimeError(
        f"failed to download artifact {artifact.name!r} after "
        f"{DOWNLOAD_RETRIES} attempts: {last_error}"
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
    if len(matches) > 1:
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


def validate_package_manifest(
    package: pathlib.Path, *, expected_source_commit: str | None = None
) -> dict:
    return validate_windows_cuda_package(
        package,
        expected_source_commit=expected_source_commit,
    )


def shell_exports(values: dict[str, str]) -> str:
    return "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in sorted(values.items())
    )


def write_manifest(path: pathlib.Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Windows CUDA package artifact and optional Metal probe "
            "suite logs needed by tools/run_gcp_windows_cuda_runtime_gate.sh."
        )
    )
    parser.add_argument("--repo", default="", help="GitHub repo, for example owner/name")
    parser.add_argument("--windows-cuda-run-id", required=True)
    parser.add_argument("--metal-ci-run-id", default="")
    parser.add_argument("--out-dir", default="/tmp/metalfish-win-cuda-runtime-inputs")
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
        "--require-metal",
        action="store_true",
        help="Fail unless --metal-ci-run-id is supplied and Metal logs are fetched.",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print shell exports for the direct runtime gate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo or default_repo()
    expected_sha = None if args.no_expected_sha else (args.expected_sha or git_head())
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    downloads_dir = out_dir / "downloads"
    windows_dir = out_dir / "windows"
    metal_dir = out_dir / "metal"

    if args.require_metal and not args.metal_ci_run_id:
        raise ValueError("--metal-ci-run-id is required with --require-metal")

    windows_run = read_run(repo, args.windows_cuda_run_id)
    require_run_provenance(
        windows_run,
        expected_workflow="Windows CUDA Compile Gate",
        expected_sha=expected_sha,
    )
    windows_artifact = select_artifact(
        list_artifacts(repo, windows_run.run_id), pattern="windows-cuda-compile-*"
    )
    windows_archive = downloads_dir / f"{windows_artifact.name}.zip"
    download_artifact(repo, windows_artifact, windows_archive)
    safe_extract_zip(windows_archive, windows_dir)
    package = find_one(
        windows_dir,
        "metalfish*windows-x86_64-msvc-cuda.zip",
        "Windows CUDA package zip",
    )
    package_manifest = validate_package_manifest(
        package,
        expected_source_commit=expected_sha,
    )

    metal_run = None
    metal_artifact = None
    metal_archive = None
    metal_probe_log = None
    metal_legacy_probe_log = None
    metal_comparison_log = None
    metal_mcts_search_json = None
    metal_hybrid_search_json = None
    if args.metal_ci_run_id:
        metal_run = read_run(repo, args.metal_ci_run_id)
        require_run_provenance(
            metal_run, expected_workflow="MetalFish CI", expected_sha=expected_sha
        )
        metal_artifact = select_artifact(
            list_artifacts(repo, metal_run.run_id), name="metalfish-macos-arm64"
        )
        metal_archive = downloads_dir / f"{metal_artifact.name}.zip"
        download_artifact(repo, metal_artifact, metal_archive)
        safe_extract_zip(metal_archive, metal_dir)
        metal_probe_log = find_one(
            metal_dir, "metal-nn-probe-suite.log", "Metal BT4 probe suite log"
        )
        metal_legacy_probe_log = find_one(
            metal_dir,
            "metal-legacy-nn-probe-suite.log",
            "Metal legacy probe suite log",
        )
        metal_comparison_log = find_one(
            metal_dir, "metal-nn-comparison.log", "Metal NN comparison log"
        )
        metal_mcts_search_json = find_one(
            metal_dir, "metal-mcts-bk07-search.json", "Metal MCTS search JSON"
        )
        metal_hybrid_search_json = find_one(
            metal_dir,
            "metal-hybrid-startpos-search.json",
            "Metal Hybrid search JSON",
        )

    env = {
        "METALFISH_WINDOWS_CUDA_PACKAGE": str(package),
        "METALFISH_WINDOWS_CUDA_COMPILE_RUN_ID": windows_run.run_id,
    }
    if metal_probe_log and metal_legacy_probe_log:
        env.update(
            {
                "METALFISH_REQUIRE_METAL_BENCHMARK_COMPARE": "1",
                "METALFISH_REQUIRE_METAL_SEARCH_COMPARE": "1",
                "METALFISH_METAL_COMPARISON_LOG": str(metal_comparison_log),
                "METALFISH_METAL_PROBE_SUITE_LOG": str(metal_probe_log),
                "METALFISH_METAL_LEGACY_PROBE_SUITE_LOG": str(metal_legacy_probe_log),
                "METALFISH_METAL_MCTS_BK07_SEARCH_JSON": str(
                    metal_mcts_search_json
                ),
                "METALFISH_METAL_HYBRID_STARTPOS_SEARCH_JSON": str(
                    metal_hybrid_search_json
                ),
                "METALFISH_REQUIRE_METAL_COMPARE": "1",
            }
        )

    env_path = out_dir / "runtime-gate-env.sh"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(shell_exports(env) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "repo": repo,
        "expected_sha": expected_sha,
        "windows_cuda": {
            "run": dataclasses.asdict(windows_run),
            "artifact": dataclasses.asdict(windows_artifact),
            "archive": str(windows_archive),
            "package": str(package),
            "package_sha256": sha256_file(package),
            "package_manifest": package_manifest,
        },
        "metal": None,
        "env_file": str(env_path),
    }
    if metal_run and metal_artifact and metal_archive and metal_probe_log:
        manifest["metal"] = {
            "run": dataclasses.asdict(metal_run),
            "artifact": dataclasses.asdict(metal_artifact),
            "archive": str(metal_archive),
            "comparison_log": str(metal_comparison_log),
            "probe_suite_log": str(metal_probe_log),
            "legacy_probe_suite_log": str(metal_legacy_probe_log),
            "mcts_bk07_search_json": str(metal_mcts_search_json),
            "hybrid_startpos_search_json": str(metal_hybrid_search_json),
        }
    manifest_path = out_dir / "runtime-gate-inputs-manifest.json"
    write_manifest(manifest_path, manifest)

    print(f"Windows CUDA package: {package}")
    print(
        "Windows CUDA package manifest: "
        f"{package_manifest['schema']} {package_manifest['kind']} "
        f"files={package_manifest['file_count']}"
    )
    if metal_probe_log and metal_legacy_probe_log:
        print(f"Metal NN comparison: {metal_comparison_log}")
        print(f"Metal BT4 probe suite: {metal_probe_log}")
        print(f"Metal legacy probe suite: {metal_legacy_probe_log}")
    print(f"Environment file: {env_path}")
    print(f"Manifest: {manifest_path}")
    if args.print_env:
        print()
        print(shell_exports(env))
        print("tools/run_gcp_windows_cuda_runtime_gate.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
