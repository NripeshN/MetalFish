#!/usr/bin/env python3
"""Resolve the GitHub CLI consistently for local and CI helper scripts."""

from __future__ import annotations

import os
import pathlib
import shutil


def gh_executable() -> str:
    configured = os.environ.get("METALFISH_GH") or os.environ.get("GH")
    candidates = [
        configured,
        shutil.which("gh"),
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        "/usr/bin/gh",
        pathlib.Path(os.environ.get("ProgramFiles", "")) / "GitHub CLI" / "gh.exe",
        pathlib.Path(os.environ.get("ProgramFiles(x86)", "")) / "GitHub CLI" / "gh.exe",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = pathlib.Path(candidate)
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        "GitHub CLI 'gh' was not found. Install gh, add it to PATH, or set "
        "METALFISH_GH to the executable path."
    )


def gh_cmd(*args: str) -> list[str]:
    return [gh_executable(), *args]
