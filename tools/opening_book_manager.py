#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.error
import urllib.request

import chess
import chess.polyglot

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "books"

BOOKS = {
    "gm2001": "https://github.com/gmcheems-org/free-opening-books/raw/main/books/bin/gm2001.bin",
    "komodo": "https://github.com/gmcheems-org/free-opening-books/raw/main/books/bin/komodo.bin",
    "rodent": "https://github.com/gmcheems-org/free-opening-books/raw/main/books/bin/rodent.bin",
}


def open_url(url: str, timeout: float = 120.0):
    last_error = None
    for _ in range(3):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = exc
            time.sleep(1)
    raise last_error


def validate_polyglot(path: pathlib.Path) -> None:
    size = path.stat().st_size
    if size <= 0 or size % 16 != 0:
        raise ValueError(f"{path} is not a valid Polyglot book")
    with chess.polyglot.open_reader(str(path)) as reader:
        list(reader.find_all(chess.Board()))


def download_file(url: str, dest: pathlib.Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    with open_url(url) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    validate_polyglot(tmp)
    tmp.replace(dest)


def selected_books(name: str) -> dict[str, str]:
    if name == "all":
        return BOOKS
    if name not in BOOKS:
        raise ValueError(f"unknown book {name}")
    return {name: BOOKS[name]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download local Polyglot books")
    parser.add_argument("command", choices=("download", "validate", "list"))
    parser.add_argument("--book", choices=sorted([*BOOKS, "all"]), default="all")
    parser.add_argument("--dest", type=pathlib.Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.command == "list":
        for name, url in BOOKS.items():
            print(f"{name}: {url}")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    books = selected_books(args.book)

    try:
        for name, url in books.items():
            dest = args.dest / pathlib.PurePosixPath(url).name
            if args.command == "download":
                if dest.exists() and dest.stat().st_size > 0 and not args.force:
                    validate_polyglot(dest)
                    print(f"{name}: already valid -> {dest}")
                    continue
                print(f"{name}: downloading -> {dest}")
                download_file(url, dest)
                print(f"{name}: OK")
            else:
                validate_polyglot(dest)
                print(f"{name}: OK -> {dest}")
        return 0
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
