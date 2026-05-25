#!/usr/bin/env python3
"""Download the neural-network files needed by MetalFish builds."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

NNUE_URLS = {
    "nn-c288c895ea92.nnue": "https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue",
    "nn-37f18f62d772.nnue": "https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue",
}

BT4_FILENAME = "BT4-1024x15x32h-swa-6147500.pb"
BT4_GZ_FILENAME = f"{BT4_FILENAME}.gz"
BT4_URL = "https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1024x15x32h-swa-6147500.pb.gz"

LEGACY_42850_FILENAME = "legacy-42850.pb.gz"
LEGACY_42850_URL = "https://storage.lczero.org/files/networks/00af53b081e80147172e6f281c01daf5ca19ada173321438914c730370aa4267"


def download(url: str, dest: Path, retries: int, force: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"Using cached {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url} -> {dest} (attempt {attempt}/{retries})")
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "MetalFish-CI/0.1 (+https://github.com/NripeshN/MetalFish)"
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with tmp.open("wb") as out:
                    shutil.copyfileobj(response, out)
            if tmp.stat().st_size <= 0:
                raise RuntimeError(f"Downloaded empty file: {url}")
            tmp.replace(dest)
            return
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt != retries:
                time.sleep(min(10, 2 * attempt))

    raise RuntimeError(f"Failed to download {url}: {last_error}")


def decompress_gzip(src: Path, dest: Path, force: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"Using cached {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"Decompressing {src} -> {dest}")
    with gzip.open(src, "rb") as gz:
        with tmp.open("wb") as out:
            shutil.copyfileobj(gz, out)
    if tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Decompressed empty file: {dest}")
    tmp.replace(dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="networks", help="output directory")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="overwrite files")
    parser.add_argument("--nnue-only", action="store_true")
    parser.add_argument("--bt4-only", action="store_true")
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="download the legacy 42850 classical convolution Lc0 net only",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="include the legacy 42850 classical convolution Lc0 net",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    only_flags = [args.nnue_only, args.bt4_only, args.legacy_only]
    if sum(bool(flag) for flag in only_flags) > 1:
        print(
            "--nnue-only, --bt4-only, and --legacy-only are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    want_nnue = not (args.bt4_only or args.legacy_only)
    want_bt4 = not (args.nnue_only or args.legacy_only)
    want_legacy = args.legacy_only or args.include_legacy

    if want_nnue:
        for filename, url in NNUE_URLS.items():
            download(url, dest / filename, args.retries, args.force)

    if want_bt4:
        url = os.environ.get("METALFISH_BT4_WEIGHTS_URL", BT4_URL)
        gz_path = dest / BT4_GZ_FILENAME
        pb_path = dest / BT4_FILENAME
        download(url, gz_path, args.retries, args.force)
        decompress_gzip(gz_path, pb_path, args.force)

    if want_legacy:
        url = os.environ.get("METALFISH_LEGACY_WEIGHTS_URL", LEGACY_42850_URL)
        download(url, dest / LEGACY_42850_FILENAME, args.retries, args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
