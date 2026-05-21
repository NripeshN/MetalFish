#!/usr/bin/env python3
"""Filter a Lichess puzzle CSV stream into a bounded deterministic sample."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.lichess_puzzle_runner import (  # noqa: E402
    csv_row_matches,
    parse_theme_filter,
)


FIELDS = [
    "PuzzleId",
    "FEN",
    "Moves",
    "Rating",
    "RatingDeviation",
    "Popularity",
    "NbPlays",
    "Themes",
    "GameUrl",
    "OpeningTags",
]


def run(args: argparse.Namespace) -> int:
    themes = parse_theme_filter(args.themes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    reader = csv.DictReader(sys.stdin)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in reader:
            if not csv_row_matches(
                row,
                min_rating=args.min_rating,
                max_rating=args.max_rating,
                min_popularity=args.min_popularity,
                themes=themes,
            ):
                continue
            writer.writerow({field: row.get(field, "") for field in FIELDS})
            count += 1
            if count >= args.max_puzzles:
                break
    print(f"Wrote {count} puzzle rows to {args.out}")
    if count < args.min_puzzles:
        print(
            f"ERROR: only wrote {count} puzzle rows, below --min-puzzles "
            f"{args.min_puzzles}",
            file=sys.stderr,
        )
        return 1
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--max-puzzles", type=int, default=500)
    parser.add_argument("--min-puzzles", type=int, default=1)
    parser.add_argument("--min-rating", type=int, default=0)
    parser.add_argument("--max-rating", type=int, default=0)
    parser.add_argument("--min-popularity", type=int, default=-101)
    parser.add_argument("--themes", default="")
    args = parser.parse_args(argv)
    args.max_puzzles = max(1, args.max_puzzles)
    args.min_puzzles = max(0, args.min_puzzles)
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
