#!/usr/bin/env python3
"""Validate packaged Linux and Windows CUDA release artifacts."""
from __future__ import annotations

import argparse
import fnmatch
import json
import pathlib
import sys
import tarfile
import zipfile


REQUIRED_LINUX_CUDA_FILES = {
    "metalfish",
    "metalfish_nn_probe",
    "test_nn_comparison",
    "PORTABLE_ARTIFACT.md",
}
REQUIRED_WINDOWS_CUDA_FILES = {
    "metalfish.exe",
    "metalfish_nn_probe.exe",
    "test_nn_comparison.exe",
    "PORTABLE_ARTIFACT.md",
}
REQUIRED_WINDOWS_CUDA_DLL_PATTERNS = (
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
)


def normalize_archive_name(name: str) -> str:
    return pathlib.PurePosixPath(name).as_posix().lstrip("./")


def validate_portable_manifest(manifest: dict, *, package_kind: str) -> dict:
    if manifest.get("schema") != "metalfish.portable_artifact":
        raise ValueError(
            "package manifest has unexpected schema: "
            f"{manifest.get('schema')!r}"
        )
    package = manifest.get("package") or {}
    if package.get("kind") != package_kind:
        raise ValueError(
            "package manifest has unexpected kind: " f"{package.get('kind')!r}"
        )
    return package


def manifest_file_names(manifest: dict) -> set[str]:
    return {
        str(item.get("name", ""))
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }


def validate_linux_cuda_package(package: pathlib.Path) -> dict:
    with tarfile.open(package, "r:gz") as archive:
        names = {
            normalize_archive_name(member.name)
            for member in archive.getmembers()
            if member.isfile()
        }
        try:
            manifest_member = next(
                member
                for member in archive.getmembers()
                if normalize_archive_name(member.name) == "linux-cuda-package-manifest.json"
            )
        except StopIteration as exc:
            raise ValueError("Linux CUDA package is missing JSON manifest") from exc
        manifest_file = archive.extractfile(manifest_member)
        if manifest_file is None:
            raise ValueError("Linux CUDA package manifest could not be read")
        manifest = json.loads(manifest_file.read().decode("utf-8-sig"))

    missing = sorted(REQUIRED_LINUX_CUDA_FILES - names)
    if missing:
        raise ValueError("Linux CUDA package is missing entries: " + ", ".join(missing))
    package_info = validate_portable_manifest(manifest, package_kind="linux-cuda")
    missing_manifest = sorted(REQUIRED_LINUX_CUDA_FILES - manifest_file_names(manifest))
    if missing_manifest:
        raise ValueError(
            "Linux CUDA package manifest is missing file entries: "
            + ", ".join(missing_manifest)
        )
    return {
        "schema": manifest["schema"],
        "kind": package_info["kind"],
        "package_name": package_info.get("name"),
        "file_count": len(manifest.get("files", [])),
        "archive_entry_count": len(names),
    }


def validate_windows_cuda_package(package: pathlib.Path) -> dict:
    with zipfile.ZipFile(package) as archive:
        names = {
            normalize_archive_name(info.filename)
            for info in archive.infolist()
            if not info.is_dir()
        }
        try:
            raw_manifest = archive.read("windows-cuda-package-manifest.json")
        except KeyError as exc:
            raise ValueError("Windows CUDA package is missing JSON manifest") from exc
    manifest = json.loads(raw_manifest.decode("utf-8-sig"))
    missing = sorted(REQUIRED_WINDOWS_CUDA_FILES - names)
    if missing:
        raise ValueError("Windows CUDA package is missing entries: " + ", ".join(missing))
    package_info = validate_portable_manifest(manifest, package_kind="windows-cuda")
    manifest_files = manifest_file_names(manifest)
    missing_manifest = sorted(REQUIRED_WINDOWS_CUDA_FILES - manifest_files)
    if missing_manifest:
        raise ValueError(
            "Windows CUDA package manifest is missing file entries: "
            + ", ".join(missing_manifest)
        )
    missing_manifest_dlls = [
        pattern
        for pattern in REQUIRED_WINDOWS_CUDA_DLL_PATTERNS
        if not any(fnmatch.fnmatch(name, pattern) for name in manifest_files)
    ]
    if missing_manifest_dlls:
        raise ValueError(
            "Windows CUDA package manifest is missing CUDA DLL entries: "
            + ", ".join(missing_manifest_dlls)
        )
    missing_zip_dlls = [
        pattern
        for pattern in REQUIRED_WINDOWS_CUDA_DLL_PATTERNS
        if not any(fnmatch.fnmatch(name, pattern) for name in names)
    ]
    if missing_zip_dlls:
        raise ValueError(
            "Windows CUDA package zip is missing CUDA DLL entries: "
            + ", ".join(missing_zip_dlls)
        )
    return {
        "schema": manifest["schema"],
        "kind": package_info["kind"],
        "package_name": package_info.get("name"),
        "file_count": len(manifest.get("files", [])),
        "archive_entry_count": len(names),
    }


def validate_package(package: pathlib.Path, *, package_kind: str) -> dict:
    if package_kind == "linux-cuda":
        return validate_linux_cuda_package(package)
    if package_kind == "windows-cuda":
        return validate_windows_cuda_package(package)
    raise ValueError(f"unsupported CUDA package kind: {package_kind}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-kind", choices=("linux-cuda", "windows-cuda"), required=True)
    parser.add_argument("--json-output", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    package = pathlib.Path(args.package).expanduser().resolve()
    summary = validate_package(package, package_kind=args.package_kind)
    if args.json_output:
        path = pathlib.Path(args.json_output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "CUDA package artifact check: PASS "
        f"kind={summary['kind']} files={summary['file_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
