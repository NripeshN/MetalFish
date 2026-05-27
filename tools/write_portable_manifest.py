#!/usr/bin/env python3
"""Write a small manifest for portable MetalFish artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_value(args: list[str], fallback: str = "unknown") -> str:
    try:
        return (
            subprocess.check_output(
                ["git", *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            or fallback
        )
    except Exception:
        return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output")
    parser.add_argument("--package-name")
    parser.add_argument("--package-kind", default="portable")
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--notes", action="append", default=[])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_or_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    sha = os.environ.get("METALFISH_SOURCE_COMMIT") or git_value(
        ["rev-parse", "HEAD"], os.environ.get("GITHUB_SHA", "unknown")
    )
    branch = (
        os.environ.get("METALFISH_SOURCE_BRANCH")
        or os.environ.get("GITHUB_HEAD_REF")
        or os.environ.get("GITHUB_REF_NAME")
        or git_value(["rev-parse", "--abbrev-ref", "HEAD"])
    )
    build_type = os.environ.get("BUILD_TYPE", "Release")
    runner = os.environ.get("RUNNER_OS", platform.system() or "unknown")

    notes = list(args.notes)
    if not notes:
        notes = ["No additional package notes were provided."]

    lines = [
        "# MetalFish Portable Artifact",
        "",
        f"- Platform: {args.platform}",
        f"- Backend support: {args.backend}",
        f"- Binary: `{args.binary}`",
        f"- Build type: {build_type}",
        f"- Source branch: `{branch}`",
        f"- Source commit: `{sha}`",
        f"- Runner OS: {runner}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "## UCI Smoke",
            "",
            "Start with a CPU search smoke:",
            "",
            "```text",
            "uci",
            "isready",
            "position startpos",
            "go depth 3",
            "```",
            "",
            "Use `NNBackend=stub` only for construction diagnostics. It is not a "
            "strength backend.",
            "",
        ]
    )

    output.write_text("\n".join(lines))

    if args.json_output:
        json_output = Path(args.json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        package_root = output.parent
        files = []
        for file_arg in args.file:
            file_path = Path(file_arg)
            if not file_path.exists():
                raise FileNotFoundError(file_arg)
            stat = file_path.stat()
            relative_path = relative_to_or_name(file_path, package_root)
            files.append(
                {
                    "name": relative_path,
                    "path": relative_path,
                    "size_bytes": stat.st_size,
                    "sha256": sha256_file(file_path),
                    "executable": os.access(file_path, os.X_OK),
                }
            )

        manifest = {
            "schema": "metalfish.portable_artifact",
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "package": {
                "name": args.package_name or output.parent.name,
                "kind": args.package_kind,
                "platform": args.platform,
                "backend": args.backend,
                "binary": args.binary,
                "build_type": build_type,
                "source_branch": branch,
                "source_commit": sha,
                "runner_os": runner,
            },
            "notes": notes,
            "files": files,
        }
        json_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
