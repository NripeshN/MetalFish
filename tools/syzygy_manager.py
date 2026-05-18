#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "https://tablebase.lichess.ovh/tables/standard"
SETS = {
    "3-4-5": ("3-4-5-wdl", "3-4-5-dtz"),
    "6-wdl": ("6-wdl",),
    "6": ("6-wdl", "6-dtz"),
}
MARKERS = {
    "3-4-5": ".metalfish_syzygy_3-4-5.ok",
    "6-wdl": ".metalfish_syzygy_6-wdl.ok",
    "6": ".metalfish_syzygy_6.ok",
}
GIB = 1024**3


def open_url(url, timeout: float = 120.0):
    last_error = None
    for _ in range(3):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = exc
            time.sleep(1)
    raise last_error


def fetch_text(url: str, timeout: float = 120.0) -> str:
    with open_url(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def manifest_for_set(base_url: str, set_name: str) -> list[tuple[str, str, str]]:
    wanted_dirs = SETS[set_name]
    hashes = {}
    for line in fetch_text(f"{base_url}/sha256").splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            hashes[parts[1]] = parts[0].lower()

    files = []
    for url in fetch_text(f"{base_url}/download.txt").splitlines():
        url = url.strip()
        if not url:
            continue
        if not any(f"/{directory}/" in url for directory in wanted_dirs):
            continue
        name = pathlib.PurePosixPath(url).name
        expected = hashes.get(name)
        if not expected:
            raise RuntimeError(f"missing sha256 entry for {name}")
        if base_url != BASE_URL:
            suffix = url.split("/tables/standard/", 1)[1]
            url = f"{base_url.rstrip('/')}/{suffix}"
        files.append((name, url, expected))
    return files


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: pathlib.Path, expected_sha256: str) -> bool:
    return path.exists() and sha256_file(path).lower() == expected_sha256.lower()


def format_gib(size_bytes: int) -> str:
    return f"{size_bytes / GIB:.2f} GiB"


def remote_file_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with open_url(request, timeout=30) as response:
        size = response.headers.get("Content-Length")
        if not size:
            raise RuntimeError(f"missing Content-Length for {url}")
        return int(size)


def remote_sizes(files: list[tuple[str, str, str]], jobs: int) -> dict[str, int]:
    sizes: dict[str, int] = {}
    workers = max(1, min(jobs, len(files)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(remote_file_size, url): name for name, url, _ in files
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            sizes[name] = future.result()
    return sizes


def disk_free_bytes(dest: pathlib.Path) -> int:
    path = dest if dest.exists() else dest.parent
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def preflight_report(
    dest: pathlib.Path,
    files: list[tuple[str, str, str]],
    sizes: dict[str, int],
    reserve_bytes: int,
) -> tuple[bool, dict[str, int]]:
    total = sum(sizes[name] for name, _, _ in files)
    already_valid = 0
    already_valid_bytes = 0
    download_bytes = 0
    invalid_existing = 0

    for name, _, expected in files:
        path = dest / name
        size = sizes[name]
        if is_valid(path, expected):
            already_valid += 1
            already_valid_bytes += size
        else:
            download_bytes += size
            if path.exists():
                invalid_existing += 1

    free = disk_free_bytes(dest)
    ok = free - download_bytes >= reserve_bytes
    return ok, {
        "files": len(files),
        "total_bytes": total,
        "already_valid": already_valid,
        "already_valid_bytes": already_valid_bytes,
        "download_bytes": download_bytes,
        "invalid_existing": invalid_existing,
        "free_bytes": free,
        "reserve_bytes": reserve_bytes,
    }


def print_preflight(set_name: str, dest: pathlib.Path, stats: dict[str, int]) -> None:
    print(f"Syzygy {set_name} preflight -> {dest}")
    print(f"  files: {stats['files']} " f"({format_gib(stats['total_bytes'])} total)")
    print(
        f"  already valid: {stats['already_valid']} "
        f"({format_gib(stats['already_valid_bytes'])})"
    )
    print(f"  download needed: {format_gib(stats['download_bytes'])}")
    if stats["invalid_existing"]:
        print(f"  invalid existing files to repair: {stats['invalid_existing']}")
    print(f"  free space: {format_gib(stats['free_bytes'])}")
    print(f"  reserve: {format_gib(stats['reserve_bytes'])}")


def download_file(url: str, dest: pathlib.Path, expected_sha256: str) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    with open_url(url, timeout=180) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    actual = sha256_file(tmp)
    if actual.lower() != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 mismatch for {dest.name}: expected {expected_sha256}, got {actual}"
        )
    tmp.replace(dest)


def validate_probes(dest: pathlib.Path, set_name: str) -> None:
    try:
        import chess
        import chess.syzygy
    except ModuleNotFoundError as exc:
        raise RuntimeError("python-chess is required for probe validation") from exc

    with chess.syzygy.open_tablebase(str(dest)) as tablebase:
        if set_name == "3-4-5":
            probes = [
                "7k/8/8/8/8/8/QRR5/K7 w - - 0 1",
                "7k/8/8/8/8/8/6R1/K7 w - - 0 1",
                "8/8/8/8/8/8/4K3/7k w - - 0 1",
            ]
            for fen in probes:
                board = chess.Board(fen)
                tablebase.probe_wdl(board)
                tablebase.probe_dtz(board)
            return

        board = chess.Board("7k/6pp/8/8/8/8/1PP5/K7 w - - 0 1")
        tablebase.probe_wdl(board)
        if set_name == "6":
            tablebase.probe_dtz(board)
        for fen in (
            "7k/8/8/8/8/8/QRR5/K7 w - - 0 1",
            "7k/8/8/8/8/8/6R1/K7 w - - 0 1",
        ):
            board = chess.Board(fen)
            tablebase.probe_wdl(board)
            tablebase.probe_dtz(board)


def write_marker(
    dest: pathlib.Path, set_name: str, files: list[tuple[str, str, str]]
) -> None:
    marker = dest / MARKERS[set_name]
    payload = {
        "set": set_name,
        "source": BASE_URL,
        "files": len(files),
        "validated_at": int(time.time()),
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command_download(args: argparse.Namespace) -> int:
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    files = manifest_for_set(args.base_url.rstrip("/"), args.set)
    reserve_bytes = int(args.reserve_gib * GIB)
    if not args.skip_preflight:
        sizes = remote_sizes(files, args.jobs)
        ok, stats = preflight_report(dest, files, sizes, reserve_bytes)
        print_preflight(args.set, dest, stats)
        if not ok:
            print(
                "ERROR: not enough free space after reserve; "
                "choose a smaller set or free disk space",
                file=sys.stderr,
            )
            return 1

    print(f"Syzygy {args.set}: {len(files)} files -> {dest}", flush=True)

    repaired = 0
    skipped = 0
    for index, (name, url, expected) in enumerate(files, start=1):
        path = dest / name
        if is_valid(path, expected):
            skipped += 1
            if args.verbose:
                print(f"  [{index}/{len(files)}] OK {name}")
            continue
        repaired += 1
        print(f"  [{index}/{len(files)}] downloading {name}", flush=True)
        download_file(url, path, expected)

    validate_probes(dest, args.set)
    write_marker(dest, args.set, files)
    print(
        f"Syzygy {args.set}: OK ({skipped} already valid, {repaired} downloaded)",
        flush=True,
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    dest = args.dest.resolve()
    files = manifest_for_set(args.base_url.rstrip("/"), args.set)
    missing = []
    bad = []
    for name, _, expected in files:
        path = dest / name
        if not path.exists():
            missing.append(name)
        elif not is_valid(path, expected):
            bad.append(name)

    if missing or bad:
        print(f"Syzygy {args.set}: invalid")
        if missing:
            print(f"  missing: {len(missing)}")
            for name in missing[:20]:
                print(f"    {name}")
        if bad:
            print(f"  bad checksum: {len(bad)}")
            for name in bad[:20]:
                print(f"    {name}")
        return 1

    validate_probes(dest, args.set)
    write_marker(dest, args.set, files)
    print(f"Syzygy {args.set}: OK ({len(files)} files)")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    dest = args.dest.resolve()
    files = manifest_for_set(args.base_url.rstrip("/"), args.set)
    sizes = remote_sizes(files, args.jobs)
    ok, stats = preflight_report(dest, files, sizes, int(args.reserve_gib * GIB))
    print_preflight(args.set, dest, stats)
    if ok:
        print("  result: OK")
        return 0
    print("  result: insufficient free space")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate Syzygy files")
    parser.add_argument("command", choices=("download", "preflight", "validate"))
    parser.add_argument("--dest", type=pathlib.Path, default=pathlib.Path("syzygy"))
    parser.add_argument("--set", choices=sorted(SETS), default="3-4-5")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        if args.command == "download":
            return command_download(args)
        if args.command == "preflight":
            return command_preflight(args)
        return command_validate(args)
    except (OSError, urllib.error.URLError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
