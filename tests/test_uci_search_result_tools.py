#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import compare_uci_search_results as comparer  # noqa: E402
from tools import uci_smoke  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def write_result(
    path: pathlib.Path,
    *,
    score: int = 96,
    pv_head: str = "a3b4",
    mcts_move: str = "a3d6",
    options: list[str] | None = None,
) -> None:
    output = [
        "info string Starting Parallel Hybrid Search (MCTS + AB)...",
        (
            f"info depth 2 score cp {score} nodes 149 time 659 nps 226 "
            f"pv {pv_head} string hybrid-final"
        ),
        (
            "info string Final: MCTSPlayouts=1 MCTSEvals=1 "
            f"ABDepth=2 ABMove=a3b4 MCTSMove={mcts_move}"
        ),
        f"bestmove {pv_head}",
    ]
    uci_smoke.write_json_result(
        path,
        engine=pathlib.Path("metalfish"),
        position="startpos",
        go="nodes 50",
        options=options or ["UseHybridSearch=true", "MCTSMinibatchSize=1"],
        bestmove=pv_head,
        output=output,
        elapsed_sec=0.1,
        returncode=0,
    )


def run_comparer(args: list[str]) -> int:
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return comparer.main(args)


def remove_parsed_search_fields(path: pathlib.Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("search_info", None)
    payload.pop("final_metrics", None)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_uci_smoke_extracts_search_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result_path = pathlib.Path(tmp) / "result.json"
        write_result(result_path)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        expect("search score", payload["search_info"]["score"]["value"] == 96)
        expect("search pv", payload["search_info"]["pv"][0] == "a3b4")
        expect("final metric", payload["final_metrics"]["MCTSPlayouts"] == 1)
        expect("string final metric", payload["final_metrics"]["ABMove"] == "a3b4")


def test_compare_uci_search_result_enforces_shape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "expected.json"
        actual = root / "actual.json"
        write_result(expected, score=96, pv_head="a3b4")
        write_result(actual, score=99, pv_head="a3b4")
        remove_parsed_search_fields(actual)
        expect(
            "comparison passes",
            run_comparer(
                [
                    "--expected",
                    str(expected),
                    "--actual",
                    str(actual),
                    "--require-same-pv-head",
                    "--max-score-cp-delta",
                    "5",
                    "--require-positive-final-metric",
                    "MCTSPlayouts",
                    "--require-final-metric",
                    "ABMove",
                    "--require-same-final-metric",
                    "MCTSMove",
                    "--require-same-setoption",
                    "MCTSMinibatchSize",
                ]
            )
            == 0,
        )

        write_result(actual, score=120, pv_head="h5f6")
        expect(
            "comparison rejects",
            run_comparer(
                [
                    "--expected",
                    str(expected),
                    "--actual",
                    str(actual),
                    "--require-same-pv-head",
                    "--max-score-cp-delta",
                    "5",
                ]
            )
            == 1,
        )

        write_result(actual, score=99, pv_head="a3b4", mcts_move="h5f6")
        expect(
            "comparison rejects final metric drift",
            run_comparer(
                [
                    "--expected",
                    str(expected),
                    "--actual",
                    str(actual),
                    "--require-same-final-metric",
                    "MCTSMove",
                ]
            )
            == 1,
        )

        write_result(
            actual,
            score=99,
            pv_head="a3b4",
            options=["UseHybridSearch=true", "MCTSMinibatchSize=2"],
        )
        expect(
            "comparison rejects setoption drift",
            run_comparer(
                [
                    "--expected",
                    str(expected),
                    "--actual",
                    str(actual),
                    "--require-same-setoption",
                    "MCTSMinibatchSize",
                ]
            )
            == 1,
        )


def main() -> int:
    test_uci_smoke_extracts_search_fields()
    test_compare_uci_search_result_enforces_shape()
    print("UCI search result tool tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
