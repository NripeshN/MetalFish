#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple


def load_report(path: pathlib.Path) -> Dict[str, object]:
    with open(path) as f:
        data = json.load(f)
    if "engines" not in data or not isinstance(data["engines"], dict):
        raise ValueError(f"{path}: not a bk_parity JSON report")
    return data


def choose_engine(report: Dict[str, object], requested: str) -> str:
    engines = report["engines"]
    assert isinstance(engines, dict)
    if requested:
        if requested not in engines:
            available = ", ".join(sorted(engines))
            raise ValueError(f"engine '{requested}' not in report; available: {available}")
        return requested
    if len(engines) != 1:
        available = ", ".join(sorted(engines))
        raise ValueError(f"report has multiple engines; use --engine ({available})")
    return next(iter(engines))


def position_map(report: Dict[str, object], engine: str) -> Dict[str, Dict[str, object]]:
    engines = report["engines"]
    assert isinstance(engines, dict)
    engine_data = engines[engine]
    assert isinstance(engine_data, dict)
    positions = engine_data.get("positions", [])
    if not isinstance(positions, list):
        raise ValueError(f"{engine}: positions must be a list")
    return {str(item["id"]): item for item in positions if isinstance(item, dict)}


def score(report: Dict[str, object], engine: str) -> Tuple[int, int]:
    engines = report["engines"]
    assert isinstance(engines, dict)
    engine_data = engines[engine]
    assert isinstance(engine_data, dict)
    return int(engine_data.get("score", 0)), int(engine_data.get("total", 0))


def failures(positions: Dict[str, Dict[str, object]]) -> List[str]:
    return [
        f"{pid}:{item.get('bestmove', '')}"
        for pid, item in sorted(positions.items())
        if not bool(item.get("pass", False))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare bk_parity JSON reports")
    parser.add_argument("reports", nargs="+", type=pathlib.Path)
    parser.add_argument("--engine", default="", help="Engine key to compare")
    args = parser.parse_args()

    loaded = []
    try:
        for path in args.reports:
            report = load_report(path)
            engine = choose_engine(report, args.engine)
            positions = position_map(report, engine)
            loaded.append((path, report, engine, positions))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for path, report, engine, positions in loaded:
        passed, total = score(report, engine)
        failed = failures(positions)
        print(f"{path}: {engine} {passed}/{total}")
        if failed:
            print("  failures: " + ", ".join(failed))
        else:
            print("  failures: none")

    if len(loaded) < 2:
        return 0

    base_path, _, _, base_positions = loaded[0]
    for path, _, _, positions in loaded[1:]:
        print(f"\nDelta vs {base_path} -> {path}:")
        all_ids = sorted(set(base_positions) | set(positions))
        changed = False
        for pid in all_ids:
            before = base_positions.get(pid, {})
            after = positions.get(pid, {})
            before_pass = bool(before.get("pass", False))
            after_pass = bool(after.get("pass", False))
            before_move = before.get("bestmove", "")
            after_move = after.get("bestmove", "")
            if before_pass == after_pass and before_move == after_move:
                continue
            changed = True
            marker = "changed"
            if not before_pass and after_pass:
                marker = "fixed"
            elif before_pass and not after_pass:
                marker = "regressed"
            print(f"  {pid}: {marker} {before_move} -> {after_move}")
        if not changed:
            print("  no per-position move/pass changes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
