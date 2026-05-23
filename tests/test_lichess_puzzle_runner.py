#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import pathlib
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout

import chess

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.compare_puzzle_runs as compare_puzzle_runs  # noqa: E402
import tools.filter_lichess_puzzle_csv as filter_puzzle_csv  # noqa: E402
import tools.lichess_puzzle_runner as puzzle_runner  # noqa: E402
from tools.lichess_puzzle_runner import (  # noqa: E402
    LichessRateLimited,
    SearchAnswer,
    board_from_api_puzzle,
    board_from_csv_puzzle,
    csv_puzzle_item,
    csv_row_matches,
    normalize_move,
    parse_auto_int,
    parse_setoptions,
    parse_theme_filter,
    tag_repeat_result,
    update_answer_from_info,
    wait_after_rate_limit,
)


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_batch_puzzle_position_applies_initial_ply_plus_one() -> None:
    item = {
        "game": {
            "pgn": (
                "d4 d6 c4 Nd7 e3 e5 Nf3 h6 Bd3 Ngf6 Bc2 Be7 O-O c6 "
                "Ne1 Nf8 g3 g5 dxe5 dxe5 Qxd8+ Bxd8 c5 Bh3 Ng2 h5 "
                "Nd2 h4 Nf3 hxg3 fxg3 e4 Nxg5 Bxg2 Kxg2 Ne6 Nxe4 "
                "Nxe4 Bxe4 Nxc5 Bc2 Be7 b4 Ne6 Bb3 Bxb4 Bb2 Rh5 "
                "Bd1 Rd5 a3 Bc5 e4"
            )
        },
        "puzzle": {
            "id": "6ewuj",
            "initialPly": 52,
            "solution": ["d5d2", "g2h1", "d2b2"],
        },
    }

    board = board_from_api_puzzle(item)

    expect("puzzle side to move is black", not board.turn)
    expect("first solution move is legal", normalize_move("d5d2", board) == "d5d2")


def test_csv_puzzle_position_applies_opponent_move() -> None:
    row = {
        "PuzzleId": "00sHx",
        "FEN": "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17",
        "Moves": "e8d7 a2e6 d7d8 f7f8",
        "Rating": "1760",
        "RatingDeviation": "80",
        "Popularity": "83",
        "NbPlays": "72",
        "Themes": "mate mateIn2 middlegame short",
        "GameUrl": "https://lichess.org/yyznGmXs/black#34",
        "OpeningTags": "Italian_Game Italian_Game_Classical_Variation",
    }

    item = csv_puzzle_item(row)
    board = board_from_csv_puzzle(item)

    expect("CSV puzzle applies opponent move", board.turn == chess.WHITE)
    expect(
        "CSV solution starts after opponent move",
        item["puzzle"]["solution"][0] == "a2e6",
    )
    expect("CSV first solution move is legal", normalize_move("a2e6", board) == "a2e6")


def test_csv_filter_and_setoption_parsing() -> None:
    row = {
        "Rating": "1760",
        "Popularity": "83",
        "Themes": "mate mateIn2 middlegame short",
    }
    expect(
        "CSV row matches tactical filters",
        csv_row_matches(
            row,
            min_rating=1200,
            max_rating=2200,
            min_popularity=70,
            themes=parse_theme_filter("fork,mate"),
        ),
    )
    expect(
        "CSV row rejects missing theme",
        not csv_row_matches(
            row,
            min_rating=1200,
            max_rating=2200,
            min_popularity=70,
            themes=parse_theme_filter("endgame"),
        ),
    )
    expect(
        "setoptions parse NAME=VALUE",
        parse_setoptions(["HybridANERootProbe=true", "Threads=8"])
        == {
            "HybridANERootProbe": "true",
            "Threads": "8",
        },
    )
    expect(
        "setoptions normalize ANE weights alias",
        parse_setoptions(["HybridANEWeightsPath=networks/t1.pb.gz"])
        == {
            "HybridANEWeights": "networks/t1.pb.gz",
        },
    )
    expect(
        "auto parser maps auto to zero",
        parse_auto_int("auto", option_name="--threads") == 0,
    )
    expect(
        "auto parser keeps integers", parse_auto_int("8", option_name="--threads") == 8
    )


def test_hybrid_ane_flags_set_uci_options() -> None:
    args = puzzle_runner.parse_args(
        [
            "--mode",
            "hybrid",
            "--hybrid-ane-root-probe",
            "--hybrid-ane-root-hints",
            "--hybrid-trace",
            "--hybrid-ane-weights",
            "networks/t1.pb.gz",
            "--hybrid-ane-model-path",
            "build/coreml/t1.mlmodelc",
            "--hybrid-ane-compute-units",
            "cpu-ne",
            "--hybrid-ane-root-hint-count",
            "12",
            "--hybrid-ane-root-hint-wait-ms",
            "50",
            "--hybrid-ane-min-budget-ms",
            "500",
        ]
    )

    options = puzzle_runner.engine_options(args)

    expect("ANE probe enabled", options["HybridANERootProbe"] == "true")
    expect("ANE root hints enabled", options["HybridANERootHints"] == "true")
    expect("HybridTrace enabled", options["HybridTrace"] == "true")
    expect("ANE weights option", options["HybridANEWeights"] == "networks/t1.pb.gz")
    expect(
        "ANE model option", options["HybridANEModelPath"] == "build/coreml/t1.mlmodelc"
    )
    expect("ANE compute units", options["HybridANEComputeUnits"] == "cpu-ne")
    expect("ANE hint count", options["HybridANERootHintCount"] == "12")
    expect("ANE wait", options["HybridANERootHintWaitMs"] == "50")
    expect("ANE min budget", options["HybridANEMinBudgetMs"] == "500")
    expect(
        "ANE benchmark keeps transformer active",
        options["TransformerLowTimeFallbackMs"] == "0",
    )


def test_hybrid_mode_keeps_transformer_active() -> None:
    args = puzzle_runner.parse_args(["--mode", "hybrid"])
    options = puzzle_runner.engine_options(args)

    expect(
        "hybrid benchmark disables low-time fallback",
        options["TransformerLowTimeFallbackMs"] == "0",
    )


def test_hybrid_ane_default_wait_uses_benchmarked_profile() -> None:
    args = puzzle_runner.parse_args(["--mode", "hybrid", "--hybrid-ane-root-probe"])
    options = puzzle_runner.engine_options(args)

    expect("ANE benchmark default wait", options["HybridANERootHintWaitMs"] == "0")
    expect("ANE root hints default off", options["HybridANERootHints"] == "false")
    expect("HybridTrace default off", "HybridTrace" not in options)
    expect(
        "ANE benchmark default min budget", options["HybridANEMinBudgetMs"] == "1000"
    )
    expect(
        "ANE benchmark disables low-time fallback",
        options["TransformerLowTimeFallbackMs"] == "0",
    )


def test_hybrid_ane_low_time_fallback_can_be_overridden() -> None:
    args = puzzle_runner.parse_args(
        [
            "--mode",
            "hybrid",
            "--hybrid-ane-root-probe",
            "--setoption",
            "TransformerLowTimeFallbackMs=3000",
        ]
    )
    options = puzzle_runner.engine_options(args)

    expect(
        "explicit fallback override wins",
        options["TransformerLowTimeFallbackMs"] == "3000",
    )


def test_hybrid_ane_stats_distinguish_configured_from_active() -> None:
    args = puzzle_runner.parse_args(
        [
            "--mode",
            "hybrid",
            "--hybrid-ane-root-probe",
            "--hybrid-ane-root-hints",
            "--hybrid-ane-weights",
            "networks/t1.pb.gz",
            "--hybrid-ane-model-path",
            "build/coreml/t1.mlmodelc",
            "--hybrid-ane-compute-units",
            "cpu-ne",
        ]
    )
    stats = puzzle_runner.initial_ane_stats(args)
    expect("ANE requested", stats["ane_probe_requested"] is True)
    expect("HybridTrace starts off", stats["hybrid_trace_requested"] is False)
    expect("ANE hints requested", stats["ane_root_hints_requested"] is True)
    expect("ANE starts inactive", stats["ane_root_nonempty"] == 0)

    puzzle_runner.update_ane_stats(
        stats,
        {
            "searches": [
                {
                    "ane_hints": 1,
                    "ane_hint_moves": 3,
                    "ane_failures": 0,
                    "hybrid_ane_top": "e2e4",
                    "hybrid_ane_root": "[e2e4:0.52,d2d4:0.22]",
                    "hybrid_ane_agrees_mcts": "1",
                    "hybrid_ane_confirmed_mcts": "0",
                },
                {
                    "ane_hints": 0,
                    "ane_hint_moves": 0,
                    "ane_failures": 1,
                    "hybrid_ane_top": "none",
                    "hybrid_ane_root": "[]",
                    "hybrid_ane_agrees_mcts": "0",
                    "hybrid_ane_confirmed_mcts": "1",
                },
            ]
        },
    )

    expect("ANE searches counted", stats["ane_searches"] == 2)
    expect("ANE trace searches counted", stats["ane_trace_searches"] == 2)
    expect("ANE nonempty roots counted", stats["ane_root_nonempty"] == 1)
    expect("ANE top moves counted", stats["ane_top_moves"] == 1)
    expect("ANE hints counted", stats["ane_hints"] == 1)
    expect("ANE hint moves counted", stats["ane_hint_moves"] == 3)
    expect("ANE failures counted", stats["ane_failures"] == 1)
    expect("ANE agrees counted", stats["ane_agrees_mcts"] == 1)
    expect("ANE confirmations counted", stats["ane_confirmed_mcts"] == 1)


def test_solver_accepts_alternate_mating_move() -> None:
    class ScriptedEngine:
        def __init__(self, moves: list[str]) -> None:
            self.moves = moves
            self.index = 0

        def new_game(self) -> None:
            self.index = 0

        def search(self, board: chess.Board, movetime_ms: int) -> SearchAnswer:
            del board, movetime_ms
            move = self.moves[self.index]
            self.index += 1
            return SearchAnswer(bestmove=move)

    row = {
        "PuzzleId": "02yab",
        "FEN": "8/8/8/8/8/2R1bk1p/5Np1/6K1 w - - 6 62",
        "Moves": "c3c2 f3g3 c2c8 h3h2",
        "Rating": "2201",
        "RatingDeviation": "79",
        "Popularity": "87",
        "NbPlays": "488",
        "Themes": "advancedPawn endgame master mate mateIn2 short",
        "GameUrl": "https://lichess.org/9wLtXMET#123",
        "OpeningTags": "",
    }

    result = puzzle_runner.solve_puzzle(
        ScriptedEngine(["f3g3", "e3f2"]), csv_puzzle_item(row), 1000
    )

    expect("alternate checkmate solves puzzle", result["solved"])
    expect("alternate mate recorded", result["searches"][-1]["actual"] == "e3f2")
    expect(
        "alternate mate marker recorded",
        result["searches"][-1]["accepted_mating_alternative"],
    )


def test_search_info_parser_tracks_ane_hints() -> None:
    answer = SearchAnswer(bestmove="0000")
    update_answer_from_info(
        "info depth 7 nodes 123 nps 456 string not-a-real-format",
        answer,
    )
    update_answer_from_info(
        "info string Hybrid: AB root hints from ANE e2e4 d2d4 g1f3",
        answer,
    )
    update_answer_from_info(
        "info string Hybrid: ANE root probe failed: synthetic",
        answer,
    )
    update_answer_from_info(
        "info string HybridTrace: reason=ane_confirmed_mcts selected=c3c4 "
        "ABMove=h4h5 MCTSMove=c3c4 ANETop=c3c4 ANEAgreesMCTS=1 "
        "ANEConfirmedMCTS=1 ANETopScore=0.421 ANEScoreMargin=0.248 "
        "ANERoot=[c3c4:v=0.421,h4h5:v=0.173]",
        answer,
    )
    update_answer_from_info(
        "info string Final: MCTSPlayouts=10 ABMove=h4h5 MCTSMove=c3c4",
        answer,
    )
    expect("nodes parsed", answer.nodes == 123)
    expect("nps parsed", answer.nps == 456)
    expect("depth parsed", answer.depth == 7)
    expect("ANE hint event counted", answer.ane_hints == 1)
    expect("ANE hint moves counted", answer.ane_hint_moves == 3)
    expect("ANE failure counted", answer.ane_failures == 1)
    expect("ANE hint move list retained", answer.ane_last_hints == "e2e4 d2d4 g1f3")
    expect("hybrid reason parsed", answer.hybrid_reason == "ane_confirmed_mcts")
    expect("hybrid selected parsed", answer.hybrid_selected == "c3c4")
    expect("hybrid AB parsed", answer.hybrid_ab_move == "h4h5")
    expect("hybrid MCTS parsed", answer.hybrid_mcts_move == "c3c4")
    expect("hybrid ANE top parsed", answer.hybrid_ane_top == "c3c4")
    expect("hybrid ANE agreement parsed", answer.hybrid_ane_agrees_mcts == "1")
    expect("hybrid ANE confirmed parsed", answer.hybrid_ane_confirmed_mcts == "1")
    expect("hybrid ANE top score parsed", answer.hybrid_ane_top_score == "0.421")
    expect("hybrid ANE margin parsed", answer.hybrid_ane_score_margin == "0.248")
    expect(
        "hybrid ANE root parsed",
        answer.hybrid_ane_root == "[c3c4:v=0.421,h4h5:v=0.173]",
    )
    expect("final summary retained", answer.final_summary.startswith("Final:"))

    trace_fields = puzzle_runner.search_trace_fields(answer)
    expect(
        "trace fields include hints", trace_fields["ane_last_hints"] == "e2e4 d2d4 g1f3"
    )
    expect(
        "trace fields include reason",
        trace_fields["hybrid_reason"] == "ane_confirmed_mcts",
    )
    expect("trace fields include ANE top", trace_fields["hybrid_ane_top"] == "c3c4")
    expect(
        "trace fields include ANE margin",
        trace_fields["hybrid_ane_score_margin"] == "0.248",
    )


def test_repeat_result_ids_are_comparable() -> None:
    base = {"id": "abc", "solved": True}
    single = tag_repeat_result(base, 0, 1)
    repeated = tag_repeat_result(base, 1, 3)

    expect("single repeat leaves result unchanged", single == base)
    expect("repeat result keeps original puzzle id", repeated["puzzle_id"] == "abc")
    expect("repeat result stores pass number", repeated["repeat"] == 2)
    expect("repeat result id is unique", repeated["id"] == "abc#r2")


def test_compare_puzzle_runs_detects_regression() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        baseline = root / "baseline.jsonl"
        candidate = root / "candidate.jsonl"
        baseline.write_text(
            "\n".join(
                json.dumps({"id": puzzle_id, "solved": True})
                for puzzle_id in ("a", "b", "c")
            )
            + "\n"
        )
        candidate.write_text(
            "\n".join(
                [
                    json.dumps({"id": "a", "solved": True}),
                    json.dumps({"id": "b", "solved": False}),
                    json.dumps({"id": "c", "solved": False}),
                ]
            )
            + "\n"
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            ok = compare_puzzle_runs.run(
                compare_puzzle_runs.parse_args(
                    [
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--max-solved-drop",
                        "2",
                        "--max-accuracy-drop",
                        "1.0",
                    ]
                )
            )
            bad = compare_puzzle_runs.run(
                compare_puzzle_runs.parse_args(
                    [
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--max-solved-drop",
                        "1",
                        "--max-accuracy-drop",
                        "1.0",
                    ]
                )
            )
    expect("puzzle compare allows configured drop", ok == 0)
    expect("puzzle compare fails excessive drop", bad == 1)


def test_compare_puzzle_runs_summarizes_ane_trace() -> None:
    results = {
        "p1": {
            "id": "p1",
            "solved": False,
            "searches": [
                {
                    "ply": 0,
                    "expected": "c3c4",
                    "actual": "h4h5",
                    "ane_last_hints": "c3c4 h4h5",
                    "hybrid_ane_top": "c3c4",
                    "hybrid_ane_score_margin": "0.100",
                    "hybrid_mcts_move": "c3c4",
                    "hybrid_ab_move": "h4h5",
                    "hybrid_selected": "h4h5",
                    "hybrid_reason": "ab_default",
                }
            ],
        },
        "p2": {
            "id": "p2",
            "solved": True,
            "searches": [
                {
                    "ply": 0,
                    "expected": "e2e4",
                    "actual": "e2e4",
                    "ane_last_hints": "e2e4 d2d4",
                    "hybrid_ane_top": "e2e4",
                    "hybrid_ane_score_margin": "0.250",
                    "hybrid_ane_confirmed_mcts": "1",
                    "hybrid_mcts_move": "e2e4",
                    "hybrid_ab_move": "e2e4",
                    "hybrid_selected": "e2e4",
                    "hybrid_reason": "ane_confirmed_mcts",
                }
            ],
        },
    }

    summary = compare_puzzle_runs.ane_trace_summary(results)

    expect("trace search count", summary["searches"] == 2)
    expect("ANE hints counted", summary["ane_hint_searches"] == 2)
    expect("ANE/MCTS agreement counted", summary["ane_mcts_agree"] == 2)
    expect("ANE/MCTS selected counted", summary["ane_mcts_selected"] == 1)
    expect("AB blocked counted", summary["ane_mcts_blocked_by_ab"] == 1)
    expect("ANE-confirmed MCTS counted", summary["ane_confirmed_mcts"] == 1)
    expect("ANE margin min", summary["ane_score_margin_min"] == 0.1)
    expect("ANE margin median", summary["ane_score_margin_median"] == 0.25)
    expect("ANE margin max", summary["ane_score_margin_max"] == 0.25)
    expect("unsolved blocked counted", summary["unsolved_blocked"] == 1)


def test_compare_puzzle_runs_matches_repeat_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        baseline = root / "baseline.jsonl"
        candidate = root / "candidate.jsonl"
        baseline.write_text(json.dumps({"id": "abc", "solved": True}) + "\n")
        candidate.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "abc#r1",
                            "puzzle_id": "abc",
                            "repeat": 1,
                            "solved": True,
                        }
                    ),
                    json.dumps(
                        {
                            "id": "abc#r2",
                            "puzzle_id": "abc",
                            "repeat": 2,
                            "solved": False,
                        }
                    ),
                ]
            )
            + "\n"
        )

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            no_match = compare_puzzle_runs.run(
                compare_puzzle_runs.parse_args(
                    [
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--min-common",
                        "1",
                    ]
                )
            )
            matched = compare_puzzle_runs.run(
                compare_puzzle_runs.parse_args(
                    [
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--min-common",
                        "2",
                        "--max-solved-drop",
                        "1",
                        "--max-accuracy-drop",
                        "1.0",
                        "--match-repeat-ids",
                    ]
                )
            )

    expect("repeat ids do not match by default", no_match == 1)
    expect("repeat ids match with flag", matched == 0)


def test_compare_puzzle_runs_preserves_repeated_baseline_results() -> None:
    baseline = [
        {"id": "abc#r1", "puzzle_id": "abc", "repeat": 1, "solved": False},
        {"id": "abc#r2", "puzzle_id": "abc", "repeat": 2, "solved": True},
    ]
    candidate = [
        {"id": "abc#r1", "puzzle_id": "abc", "repeat": 1, "solved": True},
        {"id": "abc#r2", "puzzle_id": "abc", "repeat": 2, "solved": True},
    ]

    pairs = compare_puzzle_runs.pair_results(baseline, candidate, match_repeat_ids=True)
    baseline_common = [base for _, base, _ in pairs]
    candidate_common = [cand for _, _, cand in pairs]

    expect("exact repeat pairs retained", len(pairs) == 2)
    expect(
        "baseline repeated score preserved",
        compare_puzzle_runs.solved_count(baseline_common) == 1,
    )
    expect(
        "candidate repeated score preserved",
        compare_puzzle_runs.solved_count(candidate_common) == 2,
    )


def test_filter_puzzle_csv_can_skip_and_exclude_ids() -> None:
    csv_text = (
        "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\n"
        "a,8/8/8/8/8/8/8/K6k w - - 0 1,a1a2,2500,80,90,1,advantage,,\n"
        "b,8/8/8/8/8/8/8/K6k w - - 0 1,a1a2,2501,80,90,1,advantage,,\n"
        "c,8/8/8/8/8/8/8/K6k w - - 0 1,a1a2,2502,80,90,1,advantage,,\n"
        "d,8/8/8/8/8/8/8/K6k w - - 0 1,a1a2,2503,80,90,1,advantage,,\n"
    )
    old_stdin = filter_puzzle_csv.sys.stdin
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        exclude = root / "exclude.csv"
        out = root / "out.csv"
        exclude.write_text(
            "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\n"
            "b,,,,,,,,,\n"
        )
        try:
            filter_puzzle_csv.sys.stdin = io.StringIO(csv_text)
            with redirect_stdout(io.StringIO()):
                rc = filter_puzzle_csv.run(
                    filter_puzzle_csv.parse_args(
                        [
                            "--out",
                            str(out),
                            "--max-puzzles",
                            "2",
                            "--min-puzzles",
                            "2",
                            "--min-rating",
                            "2400",
                            "--min-popularity",
                            "80",
                            "--skip-matches",
                            "1",
                            "--exclude-ids-csv",
                            str(exclude),
                        ]
                    )
                )
        finally:
            filter_puzzle_csv.sys.stdin = old_stdin

        with out.open(newline="") as f:
            ids = [row["PuzzleId"] for row in csv.DictReader(f)]
    expect("filter exits successfully", rc == 0)
    expect("filter skipped a and excluded b", ids == ["c", "d"])


def test_rate_limit_wait_respects_budget() -> None:
    old_monotonic = puzzle_runner.time.monotonic
    old_sleep = puzzle_runner.time.sleep
    sleeps: list[float] = []

    try:
        puzzle_runner.time.monotonic = lambda: 100.0
        puzzle_runner.time.sleep = lambda seconds: sleeps.append(seconds)

        with redirect_stdout(io.StringIO()):
            waited = wait_after_rate_limit(
                LichessRateLimited(65.0),
                deadline=300.0,
                events_seen=1,
                max_events=5,
            )
        expect("rate limit waits when budget remains", waited)
        expect("rate limit slept requested interval", sleeps == [65.0])

        sleeps.clear()
        expect(
            "rate limit stops when event budget is exhausted",
            not wait_after_rate_limit(
                LichessRateLimited(65.0),
                deadline=300.0,
                events_seen=5,
                max_events=5,
            ),
        )
        expect("exhausted rate limit did not sleep", sleeps == [])
    finally:
        puzzle_runner.time.monotonic = old_monotonic
        puzzle_runner.time.sleep = old_sleep


def main() -> int:
    test_batch_puzzle_position_applies_initial_ply_plus_one()
    test_csv_puzzle_position_applies_opponent_move()
    test_csv_filter_and_setoption_parsing()
    test_hybrid_ane_flags_set_uci_options()
    test_hybrid_ane_default_wait_uses_benchmarked_profile()
    test_hybrid_ane_low_time_fallback_can_be_overridden()
    test_search_info_parser_tracks_ane_hints()
    test_repeat_result_ids_are_comparable()
    test_compare_puzzle_runs_detects_regression()
    test_compare_puzzle_runs_summarizes_ane_trace()
    test_compare_puzzle_runs_matches_repeat_ids()
    test_compare_puzzle_runs_preserves_repeated_baseline_results()
    test_filter_puzzle_csv_can_skip_and_exclude_ids()
    test_rate_limit_wait_respects_budget()
    print("Lichess puzzle runner tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
