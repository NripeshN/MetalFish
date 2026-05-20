#!/usr/bin/env python3
"""Write a small manifest for portable MetalFish artifacts."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
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
    parser.add_argument("--notes", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    sha = os.environ.get("GITHUB_SHA") or git_value(["rev-parse", "HEAD"])
    branch = os.environ.get("GITHUB_REF_NAME") or git_value(
        ["rev-parse", "--abbrev-ref", "HEAD"]
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
