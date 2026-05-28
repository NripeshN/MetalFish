#!/usr/bin/env python3
"""Validate packaged Linux and Windows CUDA release artifacts."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
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
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
}
REQUIRED_WINDOWS_CUDA_FILES = {
    "metalfish.exe",
    "metalfish_nn_probe.exe",
    "test_nn_comparison.exe",
    "PORTABLE_ARTIFACT.md",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
}
REQUIRED_WINDOWS_CUDA_DLL_PATTERNS = (
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
)
REQUIRED_WINDOWS_VCPKG_DLL_PATTERNS = (
    "abseil_dll.dll",
    "libprotobuf*.dll",
    "z*.dll",
)


def normalize_archive_name(name: str) -> str:
    return pathlib.PurePosixPath(name).as_posix().lstrip("./")


def validate_portable_manifest(
    manifest: dict, *, package_kind: str, expected_source_commit: str | None = None
) -> dict:
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
    if expected_source_commit and package.get("source_commit") != expected_source_commit:
        raise ValueError(
            "package manifest has unexpected source commit: "
            f"{package.get('source_commit')!r}; expected {expected_source_commit}"
        )
    return package


def manifest_file_names(manifest: dict) -> set[str]:
    return {
        str(item.get("name", ""))
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }


def manifest_file_records(manifest: dict) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for item in manifest.get("files", []):
        if not isinstance(item, dict):
            continue
        name = normalize_archive_name(str(item.get("path") or item.get("name") or ""))
        if not name:
            raise ValueError("package manifest contains a file record without a name")
        if name in records:
            raise ValueError(f"package manifest contains duplicate file record: {name}")
        records[name] = item
    return records


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def require_manifest_hashes(
    manifest: dict,
    archive_records: dict[str, dict],
    *,
    executable_required: set[str] = frozenset(),
) -> None:
    for name, record in manifest_file_records(manifest).items():
        archive_record = archive_records.get(name)
        if archive_record is None:
            raise ValueError(f"package manifest references missing archive entry: {name}")
        expected_size = int(record.get("size_bytes") or -1)
        if expected_size != archive_record["size_bytes"]:
            raise ValueError(
                f"package manifest size mismatch for {name}: "
                f"{expected_size} != {archive_record['size_bytes']}"
            )
        expected_sha = str(record.get("sha256") or "")
        if expected_sha != archive_record["sha256"]:
            raise ValueError(
                f"package manifest sha256 mismatch for {name}: "
                f"{expected_sha!r} != {archive_record['sha256']!r}"
            )
        if name in executable_required:
            if record.get("executable") is not True:
                raise ValueError(f"package manifest does not mark {name} executable")
            if archive_record.get("executable") is not True:
                raise ValueError(f"Linux CUDA archive entry is not executable: {name}")


def require_manifest_file_set(
    manifest: dict, archive_records: dict[str, dict], *, manifest_name: str
) -> None:
    expected = manifest_file_names(manifest) | {manifest_name}
    actual = set(archive_records)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(
            "CUDA package archive is missing manifest-covered entries: "
            + ", ".join(missing)
        )
    extra = sorted(actual - expected)
    if extra:
        raise ValueError(
            "CUDA package archive contains unmanifested entries: "
            + ", ".join(extra)
        )


def read_tar_records(archive: tarfile.TarFile) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        name = normalize_archive_name(member.name)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"Linux CUDA package entry could not be read: {name}")
        with extracted:
            digest = sha256_stream(extracted)
        records[name] = {
            "size_bytes": member.size,
            "sha256": digest,
            "executable": bool(member.mode & 0o111),
        }
    return records


def read_zip_records(archive: zipfile.ZipFile) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = normalize_archive_name(info.filename)
        with archive.open(info, "r") as handle:
            digest = sha256_stream(handle)
        records[name] = {
            "size_bytes": info.file_size,
            "sha256": digest,
        }
    return records


def validate_linux_cuda_package(
    package: pathlib.Path, *, expected_source_commit: str | None = None
) -> dict:
    with tarfile.open(package, "r:gz") as archive:
        archive_records = read_tar_records(archive)
        names = set(archive_records)
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
    package_info = validate_portable_manifest(
        manifest,
        package_kind="linux-cuda",
        expected_source_commit=expected_source_commit,
    )
    require_manifest_file_set(
        manifest,
        archive_records,
        manifest_name="linux-cuda-package-manifest.json",
    )
    missing_manifest = sorted(REQUIRED_LINUX_CUDA_FILES - manifest_file_names(manifest))
    if missing_manifest:
        raise ValueError(
            "Linux CUDA package manifest is missing file entries: "
            + ", ".join(missing_manifest)
        )
    require_manifest_hashes(
        manifest,
        archive_records,
        executable_required={"metalfish", "metalfish_nn_probe", "test_nn_comparison"},
    )
    return {
        "schema": manifest["schema"],
        "kind": package_info["kind"],
        "package_name": package_info.get("name"),
        "source_commit": package_info.get("source_commit"),
        "file_count": len(manifest.get("files", [])),
        "archive_entry_count": len(names),
    }


def validate_windows_cuda_package(
    package: pathlib.Path, *, expected_source_commit: str | None = None
) -> dict:
    with zipfile.ZipFile(package) as archive:
        archive_records = read_zip_records(archive)
        names = set(archive_records)
        try:
            raw_manifest = archive.read("windows-cuda-package-manifest.json")
        except KeyError as exc:
            raise ValueError("Windows CUDA package is missing JSON manifest") from exc
    manifest = json.loads(raw_manifest.decode("utf-8-sig"))
    missing = sorted(REQUIRED_WINDOWS_CUDA_FILES - names)
    if missing:
        raise ValueError("Windows CUDA package is missing entries: " + ", ".join(missing))
    package_info = validate_portable_manifest(
        manifest,
        package_kind="windows-cuda",
        expected_source_commit=expected_source_commit,
    )
    require_manifest_file_set(
        manifest,
        archive_records,
        manifest_name="windows-cuda-package-manifest.json",
    )
    manifest_files = manifest_file_names(manifest)
    missing_manifest = sorted(REQUIRED_WINDOWS_CUDA_FILES - manifest_files)
    if missing_manifest:
        raise ValueError(
            "Windows CUDA package manifest is missing file entries: "
            + ", ".join(missing_manifest)
        )
    required_dll_patterns = (
        REQUIRED_WINDOWS_CUDA_DLL_PATTERNS + REQUIRED_WINDOWS_VCPKG_DLL_PATTERNS
    )
    missing_manifest_dlls = [
        pattern
        for pattern in required_dll_patterns
        if not any(fnmatch.fnmatch(name, pattern) for name in manifest_files)
    ]
    if missing_manifest_dlls:
        raise ValueError(
            "Windows CUDA package manifest is missing runtime DLL entries: "
            + ", ".join(missing_manifest_dlls)
        )
    missing_zip_dlls = [
        pattern
        for pattern in required_dll_patterns
        if not any(fnmatch.fnmatch(name, pattern) for name in names)
    ]
    if missing_zip_dlls:
        raise ValueError(
            "Windows CUDA package zip is missing runtime DLL entries: "
            + ", ".join(missing_zip_dlls)
        )
    require_manifest_hashes(manifest, archive_records)
    return {
        "schema": manifest["schema"],
        "kind": package_info["kind"],
        "package_name": package_info.get("name"),
        "source_commit": package_info.get("source_commit"),
        "file_count": len(manifest.get("files", [])),
        "archive_entry_count": len(names),
    }


def validate_package(
    package: pathlib.Path,
    *,
    package_kind: str,
    expected_source_commit: str | None = None,
) -> dict:
    if package_kind == "linux-cuda":
        return validate_linux_cuda_package(
            package,
            expected_source_commit=expected_source_commit,
        )
    if package_kind == "windows-cuda":
        return validate_windows_cuda_package(
            package,
            expected_source_commit=expected_source_commit,
        )
    raise ValueError(f"unsupported CUDA package kind: {package_kind}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-kind", choices=("linux-cuda", "windows-cuda"), required=True)
    parser.add_argument("--expected-source-commit", default="")
    parser.add_argument("--json-output", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    package = pathlib.Path(args.package).expanduser().resolve()
    summary = validate_package(
        package,
        package_kind=args.package_kind,
        expected_source_commit=args.expected_source_commit or None,
    )
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
