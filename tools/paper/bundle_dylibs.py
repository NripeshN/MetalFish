#!/usr/bin/env python3
"""Bundle the Homebrew dynamic libraries a Mach-O binary depends on next to it, so that a later
`brew upgrade` cannot break a frozen measurement binary (docs/paper_v2/03_codebase_changes.md C19).

For each dependency under a bundled prefix (default /opt/homebrew) the library is copied into
<lib_dir> (default <binary dir>/lib), its own dependencies are rewritten recursively to
@loader_path/<name>, its install name to @rpath/<name>, and the binary's load command to
@executable_path/lib/<name>. Everything touched is re-signed ad hoc (required on Apple Silicon).
Prints the final `otool -L` of the binary; exits 1 if any bundled-prefix path remains.

Usage:
    python3 tools/paper/bundle_dylibs.py build/metalfish [build/metalfish_nn_probe ...] [--lib-dir build/lib]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def deps(path: Path) -> list[str]:
    out = subprocess.run(
        ["otool", "-L", str(path)], capture_output=True, text=True, check=True
    ).stdout
    libs = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        libs.append(line.split(" (")[0])
    return libs


def run(*cmd: str) -> None:
    subprocess.run(list(cmd), check=True, capture_output=True, text=True)


def bundle(binary: Path, lib_dir: Path, prefix: str, done: dict) -> None:
    lib_dir.mkdir(parents=True, exist_ok=True)
    for dep in deps(binary):
        if not dep.startswith(prefix):
            continue
        name = os.path.basename(dep)
        target = lib_dir / name
        if name not in done:
            src = Path(os.path.realpath(dep))
            shutil.copy2(src, target)
            os.chmod(target, 0o755)
            done[name] = target
            run("install_name_tool", "-id", f"@rpath/{name}", str(target))
            bundle(target, lib_dir, prefix, done)  # its own dependencies first
            run("codesign", "--force", "-s", "-", str(target))
        rel = (
            f"@loader_path/{name}"
            if binary.parent == lib_dir
            else f"@executable_path/lib/{name}"
        )
        run("install_name_tool", "-change", dep, rel, str(binary))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("binaries", nargs="+")
    ap.add_argument(
        "--lib-dir", default=None, help="default: <dir of first binary>/lib"
    )
    ap.add_argument(
        "--prefix", default="/opt/homebrew", help="dependency prefix to bundle"
    )
    args = ap.parse_args()
    binaries = [Path(b).resolve() for b in args.binaries]
    lib_dir = (
        Path(args.lib_dir).resolve() if args.lib_dir else binaries[0].parent / "lib"
    )
    done: dict = {}
    bad = 0
    for b in binaries:
        bundle(b, lib_dir, args.prefix, done)
        run("codesign", "--force", "-s", "-", str(b))
        remaining = [d for d in deps(b) if d.startswith(args.prefix)]
        print(
            f"{b.name}: {len([d for d in deps(b) if '@executable_path' in d])} bundled, {len(remaining)} remaining under {args.prefix}"
        )
        for d in deps(b):
            print("   ", d)
        bad += len(remaining)
    for name, path in done.items():
        remaining = [d for d in deps(path) if d.startswith(args.prefix)]
        if remaining:
            print(f"{name}: still references {remaining}")
            bad += len(remaining)
    print(f"{len(done)} libraries in {lib_dir}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
