#!/usr/bin/env python3
"""Fetch Metal comparison inputs for the direct Linux CUDA GPU gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import shlex
import subprocess
import sys
import time
import zipfile


ROOT = pathlib.Path(__file__).resolve().parent.parent
DOWNLOAD_RETRIES = 3


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(
        cmd,
        cwd=ROOT,
        stderr=subprocess.PIPE,
        text=True,
    ).strip()


def run_json(cmd: list[str]) -> dict:
    return json.loads(run_text(cmd))


def complete_zip(path: pathlib.Path, *, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if expected_size > 0 and path.stat().st_size != expected_size:
        return False
    return zipfile.is_zipfile(path)


def run_to_file(cmd: list[str], path: pathlib.Path, *, expected_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if complete_zip(path, expected_size=expected_size):
        return
    tmp_path = path.with_name(f"{path.name}.part")
    last_error = ""
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        if tmp_path.exists():
            tmp_path.unlink()
        with tmp_path.open("wb") as handle:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.PIPE,
            )
        if proc.returncode == 0 and complete_zip(tmp_path, expected_size=expected_size):
            tmp_path.replace(path)
            return
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            last_error = f"gh api exited {proc.returncode}: {stderr}"
        else:
            last_error = (
                f"downloaded {tmp_path.stat().st_size} bytes, expected "
                f"{expected_size} byte ZIP"
            )
        if tmp_path.exists():
            tmp_path.unlink()
        if attempt < DOWNLOAD_RETRIES:
            time.sleep(min(2 * attempt, 5))
    raise RuntimeError(
        f"failed to download {path.name} after {DOWNLOAD_RETRIES} attempts: "
        f"{last_error}"
    )


def default_repo() -> str:
    return str(run_json(["gh", "repo", "view", "--json", "nameWithOwner"])["nameWithOwner"])


def git_head() -> str:
    return run_text(["git", "rev-parse", "HEAD"])


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_metal_run(repo: str, run_id: str, expected_sha: str | None) -> dict:
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
    workflow_name = str(data["workflowName"])
    status = str(data["status"])
    conclusion = str(data["conclusion"])
    head_sha = str(data["headSha"])
    url = str(data["url"])
    if workflow_name != "MetalFish CI":
        raise ValueError(
            f"run {run_id} must be from 'MetalFish CI', got {workflow_name!r}: {url}"
        )
    if status != "completed" or conclusion != "success":
        raise ValueError(
            f"run {run_id} is not successful: "
            f"status={status} conclusion={conclusion}: {url}"
        )
    if expected_sha and head_sha != expected_sha:
        raise ValueError(f"run {run_id} is for {head_sha}, expected {expected_sha}: {url}")
    return data


def artifact_for_name(repo: str, run_id: str, name: str) -> tuple[int, int]:
    data = run_json(
        [
            "gh",
            "api",
            f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
        ]
    )
    matches = [
        item
        for item in data.get("artifacts", [])
        if item.get("name") == name and not item.get("expired", False)
    ]
    if not matches:
        available = ", ".join(
            str(item.get("name")) for item in data.get("artifacts", [])
        )
        raise ValueError(f"artifact {name!r} not found; available: {available or '<none>'}")
    if len(matches) > 1:
        raise ValueError(f"artifact {name!r} matched multiple artifacts")
    return int(matches[0]["id"]), int(matches[0]["size_in_bytes"])


def safe_extract_zip(archive_path: pathlib.Path, dest: pathlib.Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (dest / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"zip member escapes extraction root: {member.filename}")
        archive.extractall(dest)


def shell_exports(values: dict[str, str]) -> str:
    return "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in sorted(values.items())
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="", help="GitHub repo, for example owner/name")
    parser.add_argument("--metal-ci-run-id", required=True)
    parser.add_argument("--out-dir", default="/tmp/metalfish-cuda-gpu-gate-inputs")
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
        "--print-env",
        action="store_true",
        help="Print shell exports for tools/run_gcp_cuda_gpu_gate.sh.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo or default_repo()
    expected_sha = None if args.no_expected_sha else (args.expected_sha or git_head())
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    downloads_dir = out_dir / "downloads"
    metal_dir = out_dir / "metal"
    if metal_dir.exists():
        shutil.rmtree(metal_dir)
    metal_dir.mkdir(parents=True, exist_ok=True)

    metal_run = require_metal_run(repo, args.metal_ci_run_id, expected_sha)
    metal_artifact_id, metal_artifact_size = artifact_for_name(
        repo, args.metal_ci_run_id, "metalfish-macos-arm64"
    )
    metal_archive = downloads_dir / "metalfish-macos-arm64.zip"
    run_to_file(
        [
            "gh",
            "api",
            f"repos/{repo}/actions/artifacts/{metal_artifact_id}/zip",
        ],
        metal_archive,
        expected_size=metal_artifact_size,
    )
    safe_extract_zip(metal_archive, metal_dir)

    bt4_log = metal_dir / "build" / "metal-nn-probe-suite.log"
    legacy_log = metal_dir / "build" / "metal-legacy-nn-probe-suite.log"
    comparison_log = metal_dir / "build" / "metal-nn-comparison.log"
    mcts_search_json = metal_dir / "build" / "metal-mcts-bk07-search.json"
    mcts_kiwipete_search_json = (
        metal_dir / "build" / "metal-mcts-kiwipete-search.json"
    )
    hybrid_search_json = metal_dir / "build" / "metal-hybrid-startpos-search.json"
    for path in (
        bt4_log,
        legacy_log,
        comparison_log,
        mcts_search_json,
        mcts_kiwipete_search_json,
        hybrid_search_json,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    env = {
        "METALFISH_REQUIRE_METAL_COMPARE": "1",
        "METALFISH_REQUIRE_METAL_BENCHMARK_COMPARE": "1",
        "METALFISH_REQUIRE_METAL_SEARCH_COMPARE": "1",
        "METALFISH_METAL_COMPARISON_LOG": str(comparison_log),
        "METALFISH_METAL_PROBE_SUITE_LOG": str(bt4_log),
        "METALFISH_METAL_LEGACY_PROBE_SUITE_LOG": str(legacy_log),
        "METALFISH_METAL_MCTS_BK07_SEARCH_JSON": str(mcts_search_json),
        "METALFISH_METAL_MCTS_KIWIPETE_SEARCH_JSON": str(
            mcts_kiwipete_search_json
        ),
        "METALFISH_METAL_HYBRID_STARTPOS_SEARCH_JSON": str(hybrid_search_json),
    }
    env_path = out_dir / "cuda-gpu-gate-env.sh"
    env_path.write_text(shell_exports(env) + "\n", encoding="utf-8")

    manifest = {
        "schema": "metalfish.cuda_gpu_gate_inputs",
        "schema_version": 1,
        "repo": repo,
        "expected_sha": expected_sha,
        "metal_run": metal_run,
        "env": env,
        "files": {
            "metal_probe_suite_log": {
                "path": str(bt4_log),
                "size_bytes": bt4_log.stat().st_size,
                "sha256": sha256_file(bt4_log),
            },
            "metal_legacy_probe_suite_log": {
                "path": str(legacy_log),
                "size_bytes": legacy_log.stat().st_size,
                "sha256": sha256_file(legacy_log),
            },
            "metal_comparison_log": {
                "path": str(comparison_log),
                "size_bytes": comparison_log.stat().st_size,
                "sha256": sha256_file(comparison_log),
            },
            "metal_mcts_bk07_search_json": {
                "path": str(mcts_search_json),
                "size_bytes": mcts_search_json.stat().st_size,
                "sha256": sha256_file(mcts_search_json),
            },
            "metal_mcts_kiwipete_search_json": {
                "path": str(mcts_kiwipete_search_json),
                "size_bytes": mcts_kiwipete_search_json.stat().st_size,
                "sha256": sha256_file(mcts_kiwipete_search_json),
            },
            "metal_hybrid_startpos_search_json": {
                "path": str(hybrid_search_json),
                "size_bytes": hybrid_search_json.stat().st_size,
                "sha256": sha256_file(hybrid_search_json),
            },
        },
    }
    manifest_path = out_dir / "cuda-gpu-gate-inputs-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.print_env:
        print(env_path.read_text(), end="")
    else:
        print(f"Wrote {env_path}")
        print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
