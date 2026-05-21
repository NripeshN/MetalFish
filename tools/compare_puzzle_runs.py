#!/usr/bin/env python3
"""Compare two offline/online puzzle-run JSONL reports."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def load_results(path: pathlib.Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            puzzle_id = str(item.get("id", ""))
            if puzzle_id:
                results[puzzle_id] = item
    if not results:
        raise ValueError(f"{path}: no puzzle results found")
    return results


def solved_count(results: dict[str, dict]) -> int:
    return sum(1 for item in results.values() if item.get("solved"))


def error_count(results: dict[str, dict]) -> int:
    return sum(1 for item in results.values() if item.get("error"))


def run(args: argparse.Namespace) -> int:
    baseline = load_results(args.baseline)
    candidate = load_results(args.candidate)
    common_ids = sorted(set(baseline).intersection(candidate))
    if len(common_ids) < args.min_common:
        print(
            f"ERROR: only {len(common_ids)} common puzzle ids, "
            f"below --min-common {args.min_common}",
            file=sys.stderr,
        )
        return 1

    base_common = {puzzle_id: baseline[puzzle_id] for puzzle_id in common_ids}
    cand_common = {puzzle_id: candidate[puzzle_id] for puzzle_id in common_ids}
    base_solved = solved_count(base_common)
    cand_solved = solved_count(cand_common)
    base_acc = base_solved / len(common_ids)
    cand_acc = cand_solved / len(common_ids)
    solved_drop = base_solved - cand_solved
    accuracy_drop = base_acc - cand_acc

    print(
        f"Puzzle compare: baseline {base_solved}/{len(common_ids)} "
        f"({base_acc:.2%}), candidate {cand_solved}/{len(common_ids)} "
        f"({cand_acc:.2%})"
    )
    print(
        f"Delta: solved_drop={solved_drop}, accuracy_drop={accuracy_drop:.2%}, "
        f"errors baseline={error_count(base_common)}, "
        f"candidate={error_count(cand_common)}"
    )

    failures: list[str] = []
    if cand_solved < args.min_candidate_solved:
        failures.append(
            f"candidate solved {cand_solved}, below {args.min_candidate_solved}"
        )
    if solved_drop > args.max_solved_drop:
        failures.append(
            f"candidate lost {solved_drop} solved puzzles, "
            f"above {args.max_solved_drop}"
        )
    if accuracy_drop > args.max_accuracy_drop:
        failures.append(
            f"candidate accuracy dropped {accuracy_drop:.2%}, "
            f"above {args.max_accuracy_drop:.2%}"
        )
    if error_count(cand_common) > args.max_candidate_errors:
        failures.append(
            f"candidate had {error_count(cand_common)} errors, "
            f"above {args.max_candidate_errors}"
        )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--min-common", type=int, default=1)
    parser.add_argument("--min-candidate-solved", type=int, default=0)
    parser.add_argument("--max-solved-drop", type=int, default=10)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.03)
    parser.add_argument("--max-candidate-errors", type=int, default=0)
    args = parser.parse_args(argv)
    args.min_common = max(1, args.min_common)
    args.max_solved_drop = max(0, args.max_solved_drop)
    args.max_accuracy_drop = max(0.0, args.max_accuracy_drop)
    args.max_candidate_errors = max(0, args.max_candidate_errors)
    args.min_candidate_solved = max(0, args.min_candidate_solved)
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv or sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
