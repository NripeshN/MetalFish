#!/usr/bin/env python3
"""Regression checks for benchmark UCI option drift."""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "tests"))

import paper_benchmarks  # noqa: E402
import bk_parity  # noqa: E402

sys.path.insert(0, str(PROJ / "tools"))
import analyze_hybrid_trace  # noqa: E402
import run_tournament_live  # noqa: E402


def assert_options_include(
    label: str, actual: dict[str, str], expected: dict[str, str]
) -> None:
    mismatches = [
        f"{key}: actual={actual.get(key)!r} expected={value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if mismatches:
        raise AssertionError(f"{label} option drift:\n" + "\n".join(mismatches))


def assert_file_contains(path: pathlib.Path, required: list[str]) -> None:
    text = path.read_text()
    missing = [token for token in required if token not in text]
    if missing:
        raise AssertionError(
            f"{path.relative_to(PROJ)} missing tokens:\n" + "\n".join(missing)
        )


def with_clean_hybrid_env(callback):
    env_names = paper_benchmarks.HYBRID_ENV_OPTION_OVERRIDES.keys()
    old_values = {key: os.environ.get(key) for key in env_names}
    try:
        for key in env_names:
            os.environ.pop(key, None)
        return callback()
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def assert_paper_hybrid_env_overrides() -> None:
    overrides = {
        "HYBRID_MCTS_THREADS": "1",
        "HYBRID_AB_THREADS": "7",
        "HYBRID_AB_POLICY_WEIGHT": "0.02",
        "HYBRID_AB_ROOT_REJECT_MCTS": "false",
        "HYBRID_MCTS_ROOT_REJECT": "false",
        "HYBRID_MCTS_AB_ROOT_HINTS": "true",
        "HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS": "50",
        "HYBRID_AB_CANDIDATE_VERIFY_MS": "120",
        "HYBRID_MCTS_OUT_OF_ORDER_FACTOR": "1.5",
        "HYBRID_ROOT_PAWN_LEVER_TIEBREAK": "false",
    }

    def detect_with_overrides() -> dict[str, str]:
        os.environ.update(overrides)
        return paper_benchmarks.detect_engines(threads=8, hash_mb=4096)[
            "metalfish-hybrid"
        ].uci_options

    options = with_clean_hybrid_env(detect_with_overrides)
    assert_options_include(
        "paper hybrid env overrides",
        options,
        {
            "HybridMCTSThreads": "1",
            "HybridABThreads": "7",
            "MCTSMaxThreads": "1",
            "HybridABPolicyWeight": "0.02",
            "HybridABRootRejectMCTS": "false",
            "HybridMCTSRootReject": "false",
            "HybridMCTSABRootHints": "true",
            "HybridMCTSABRootHintDelayMs": "50",
            "HybridABCandidateVerifyMs": "120",
            "MCTSMaxOutOfOrderEvalsFactor": "1.5",
            "HybridRootPawnLeverTieBreak": "false",
        },
    )


def assert_hybrid_trace_final_decision_only() -> None:
    data = {
        "matches": [
            {
                "games": [
                    {
                        "game": 1,
                        "search_log": [
                            {
                                "ply": 12,
                                "side": "black",
                                "fen": "8/8/8/8/8/8/8/8 b - - 0 1",
                                "move": "b7b5",
                                "lines": [
                                    "info string HybridTrace: reason=ab_default "
                                    "selected=a7a6 ABMove=a7a6 MCTSMove=b7b5",
                                    "info string HybridTrace: reason=root_pawn_lever_tiebreak "
                                    "selected=b7b5 ABMove=a7a6 MCTSMove=b7b5",
                                ],
                            }
                        ],
                    }
                ]
            }
        ]
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "results.json"
        path.write_text(json.dumps(data))
        decisions = list(analyze_hybrid_trace.iter_trace_decisions(path))
    if len(decisions) != 1:
        raise AssertionError(f"expected one final trace decision, got {len(decisions)}")
    decision = decisions[0]
    if decision.reason != "root_pawn_lever_tiebreak" or decision.selected != "b7b5":
        raise AssertionError(
            "trace analyzer did not keep the final HybridTrace decision"
        )


def assert_tournament_draw_reason_precision() -> None:
    claimable = paper_benchmarks.chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        claimable.push(paper_benchmarks.chess.Move.from_uci(move))
    if run_tournament_live.game_over_reason(claimable) != "claimable 3-fold repetition":
        raise AssertionError("claimable repetition reason was not precise")

    repeated = claimable.copy()
    repeated.push(paper_benchmarks.chess.Move.from_uci("f6g8"))
    if run_tournament_live.game_over_reason(repeated) != "3-fold repetition":
        raise AssertionError("actual repetition reason was not precise")


def assert_transformer_low_time_warnings_cover_mcts() -> None:
    engines_cfg = {
        "MetalFish-MCTS": {
            "options": {
                "UseMCTS": "true",
                "UseHybridSearch": "false",
                "TransformerLowTimeFallbackMs": "3000",
            }
        },
        "MetalFish-AB": {"options": {"UseMCTS": "false", "UseHybridSearch": "false"}},
    }
    warnings = run_tournament_live.hybrid_low_time_warnings(
        [("MetalFish-MCTS", "MetalFish-AB")], engines_cfg, movetime_ms=1000
    )
    if not warnings or "MetalFish-MCTS" not in warnings[0]:
        raise AssertionError("low-time warning did not cover pure MCTS fallback")


def assert_bk_hybrid_fallback_warning() -> None:
    args = type(
        "Args",
        (),
        {
            "engine": "hybrid",
            "nodes": 0,
            "movetime": 2000,
            "hybrid_low_time_fallback_ms": 3000,
        },
    )()
    warning = bk_parity.hybrid_time_safety_fallback_warning(args)
    if "not full Hybrid MCTS" not in warning:
        raise AssertionError("BK parity did not warn for Hybrid fallback benchmarks")
    args.hybrid_low_time_fallback_ms = 0
    if bk_parity.hybrid_time_safety_fallback_warning(args):
        raise AssertionError("BK parity warned after fallback was disabled")


def assert_tactical_fail_under_guard() -> None:
    selected = ["metalfish-hybrid"]
    if paper_benchmarks.parse_tactical_fail_under("", selected) != {}:
        raise AssertionError("empty tactical fail-under should parse to no floors")
    if paper_benchmarks.parse_tactical_fail_under("21", selected) != {
        "metalfish-hybrid": 21
    }:
        raise AssertionError(
            "bare tactical fail-under should apply to selected engines"
        )
    if paper_benchmarks.parse_tactical_fail_under("hybrid=21,mcts=12", None) != {
        "metalfish-hybrid": 21,
        "metalfish-mcts": 12,
    }:
        raise AssertionError("engine-specific tactical floors parsed incorrectly")

    sample = {
        "engines": {
            "metalfish-hybrid": {
                "score": 21,
                "total": 24,
                "completed": 24,
                "complete": True,
            },
            "metalfish-mcts": {
                "score": 12,
                "total": 24,
                "completed": 23,
                "complete": False,
            },
        }
    }
    if paper_benchmarks.enforce_tactical_fail_under(sample, {"metalfish-hybrid": 21}):
        raise AssertionError("hybrid tactical floor should pass at the threshold")
    failures = paper_benchmarks.enforce_tactical_fail_under(
        sample, {"metalfish-hybrid": 22, "metalfish-mcts": 12}
    )
    expected = [
        "metalfish-hybrid=21/24 (floor 22)",
        "metalfish-mcts=incomplete 23/24 (floor 12)",
    ]
    if failures != expected:
        raise AssertionError(f"unexpected tactical floor failures: {failures!r}")


def detect_paper_engines_clean() -> dict[str, paper_benchmarks.EngineConfig]:
    return with_clean_hybrid_env(
        lambda: paper_benchmarks.detect_engines(threads=8, hash_mb=4096)
    )


def main() -> int:
    assert_tactical_fail_under_guard()
    paper = detect_paper_engines_clean()

    assert_options_include(
        "paper MCTS strength resources",
        paper["metalfish-mcts"].uci_options,
        {
            "Threads": "8",
            "Hash": "4096",
            "MCTSMaxThreads": "1",
            "MCTSParallelSearch": "false",
            "UseHybridSearch": "false",
            "UseMCTS": "true",
            "MultiPV": "1",
            "TransformerLowTimeFallbackMs": "0",
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "MCTSMinimumKLDGainPerNode": "0.00005",
        },
    )
    scaled_mcts = paper_benchmarks.config_with_thread_count(paper["metalfish-mcts"], 8)
    assert_options_include(
        "paper MCTS scaled strength resources",
        scaled_mcts.uci_options,
        {
            "Threads": "8",
            "MCTSMaxThreads": str(paper_benchmarks.pure_mcts_strength_threads(8)),
            "MCTSParallelSearch": "false",
            "UseHybridSearch": "false",
            "UseMCTS": "true",
            "TransformerLowTimeFallbackMs": "0",
        },
    )

    assert_options_include(
        "paper Hybrid fair resources",
        paper["metalfish-hybrid"].uci_options,
        {
            "Threads": "8",
            "Hash": "4096",
            "HybridMCTSThreads": "1",
            "HybridABThreads": "7",
            "HybridAutoABThreadsCap": "0",
            "MCTSMaxThreads": "1",
            "UseMCTS": "false",
            "UseHybridSearch": "true",
            "MultiPV": "1",
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "TransformerLowTimeFallbackMs": "3000",
            "TransformerMinMoveBudgetMs": "400",
            "HybridMCTSMinimumKLDGainPerNode": "0.0",
            "HybridABRootRejectMCTS": "true",
            "HybridMCTSRootReject": "true",
            "HybridMCTSUseSharedTT": "false",
            "HybridMCTSABRootHints": "true",
            "HybridMCTSABRootHintDelayMs": "25",
            "HybridMCTSABRootHintCount": "8",
            "HybridABCandidateVerifyMs": "120",
            "HybridABCandidateVerifyCount": "4",
            "HybridABPolicyWeight": "0.0",
            "HybridRootPawnLeverTieBreak": "true",
            "HybridTrace": "false",
        },
    )
    assert_options_include(
        "paper Stockfish fair resources",
        paper["stockfish"].uci_options,
        {"Threads": "8", "Hash": "4096", "Skill Level": "20"},
    )
    assert_options_include(
        "paper Lc0 fair resources",
        paper["lc0"].uci_options,
        {"Threads": "8", "Temperature": "0"},
    )
    assert_paper_hybrid_env_overrides()
    assert_hybrid_trace_final_decision_only()
    assert_tournament_draw_reason_precision()
    assert_transformer_low_time_warnings_cover_mcts()
    assert_bk_hybrid_fallback_warning()

    assert_file_contains(
        PROJ / "src/uci/engine.cpp",
        [
            'options.add("MCTSPolicySoftmaxTemp"',
            'options.add("MCTSPolicyTemperature"',
            'options.add("MCTSMaxOutOfOrderFactor"',
            'options.add("MCTSMaxOutOfOrderEvalsFactor"',
            'options.add("HybridMCTSABRootHints"',
            'options.add("HybridMCTSABRootHintDelayMs"',
            'options.add("HybridABCandidateVerifyMs"',
            'options.add("HybridABRootRejectMCTS"',
            'options.add("HybridRootPawnLeverTieBreak"',
            'options.add("HybridANERootProbe"',
            'options.add("HybridANEConfirmMCTSOverride"',
            'options.add("HybridANEModelPath"',
            'options.add("HybridANEMinBudgetMs"',
        ],
    )
    assert_file_contains(
        PROJ / "src/uci/uci.cpp",
        [
            "get_float_option_alias(",
            '"MCTSPolicyTemperature"',
            '"MCTSPolicySoftmaxTemp"',
            '"MCTSMaxOutOfOrderEvalsFactor"',
            '"MCTSMaxOutOfOrderFactor"',
            "stable_hybrid_split",
            "can_preload_hybrid",
            '"HybridMCTSThreads"',
            '"HybridABThreads"',
            '"HybridMCTSABRootHints"',
            '"HybridMCTSABRootHintDelayMs"',
            '"HybridABCandidateVerifyMs"',
            '"HybridABRootRejectMCTS"',
            '"HybridRootPawnLeverTieBreak"',
            '"HybridANERootProbe"',
            '"HybridANEConfirmMCTSOverride"',
            '"HybridANEModelPath"',
            '"HybridANEMinBudgetMs"',
        ],
    )

    engines_config = json.loads((PROJ / "tools/engines_config.json").read_text())
    engine_options = engines_config["engines"]
    assert_options_include(
        "tools MetalFish-AB",
        engine_options["MetalFish-AB"]["options"],
        {
            "UseMCTS": "false",
            "UseHybridSearch": "false",
            "MultiPV": "1",
            "Hash": "4096",
        },
    )
    assert_options_include(
        "tools MetalFish-MCTS",
        engine_options["MetalFish-MCTS"]["options"],
        {
            "UseHybridSearch": "false",
            "UseMCTS": "true",
            "MultiPV": "1",
            "Hash": "4096",
            "MCTSMaxThreads": "1",
            "MCTSParallelSearch": "false",
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "MCTSMinimumKLDGainPerNode": "0.00005",
            "TransformerLowTimeFallbackMs": "0",
        },
    )
    assert_options_include(
        "tools MetalFish-Hybrid",
        engine_options["MetalFish-Hybrid"]["options"],
        {
            "UseMCTS": "false",
            "UseHybridSearch": "true",
            "MultiPV": "1",
            "Hash": "4096",
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "TransformerLowTimeFallbackMs": "3000",
            "TransformerMinMoveBudgetMs": "400",
            "HybridMCTSMinimumKLDGainPerNode": "0.0",
            "HybridABRootRejectMCTS": "true",
            "HybridMCTSRootReject": "true",
            "HybridMCTSUseSharedTT": "false",
            "HybridMCTSABRootHints": "true",
            "HybridMCTSABRootHintDelayMs": "25",
            "HybridMCTSABRootHintCount": "8",
            "HybridABCandidateVerifyMs": "120",
            "HybridABCandidateVerifyCount": "4",
            "HybridABPolicyWeight": "0.0",
            "HybridRootPawnLeverTieBreak": "true",
            "HybridAutoABThreadsCap": "0",
            "HybridTrace": "false",
        },
    )

    tournament_tokens = [
        "option.UseMCTS=false",
        "option.UseHybridSearch=false",
        "option.MultiPV=1",
        "option.MCTSParityPreset=false",
        "option.MCTSAddDirichletNoise=false",
        "option.MCTSMinimumKLDGainPerNode=0.00005",
        "option.TransformerLowTimeFallbackMs=0",
        "option.MCTSParallelSearch=$MCTS_PARALLEL_SEARCH",
        "option.HybridMCTSMinimumKLDGainPerNode=$HYBRID_MCTS_KLD",
        "option.HybridABRootRejectMCTS=$HYBRID_AB_ROOT_REJECT_MCTS",
        "option.HybridMCTSRootReject=$HYBRID_MCTS_ROOT_REJECT",
        "option.HybridMCTSUseSharedTT=$HYBRID_MCTS_SHARED_TT",
        "option.HybridMCTSABRootHints=$HYBRID_MCTS_AB_ROOT_HINTS",
        "option.HybridMCTSABRootHintDelayMs=$HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS",
        "option.HybridMCTSABRootHintCount=$HYBRID_MCTS_AB_ROOT_HINT_COUNT",
        "option.HybridABCandidateVerifyMs=$HYBRID_AB_CANDIDATE_VERIFY_MS",
        "option.HybridABCandidateVerifyCount=$HYBRID_AB_CANDIDATE_VERIFY_COUNT",
        "option.HybridABPolicyWeight=$HYBRID_AB_POLICY_WEIGHT",
        "option.HybridRootPawnLeverTieBreak=$HYBRID_ROOT_PAWN_LEVER_TIEBREAK",
        "option.HybridAutoABThreadsCap=$HYBRID_AUTO_AB_THREADS_CAP",
        "option.MCTSMinibatchSize=$HYBRID_MCTS_MINIBATCH",
        "option.TransformerLowTimeFallbackMs=$HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS",
        "option.TransformerMinMoveBudgetMs=$HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS",
        "option.HybridTrace=$HYBRID_TRACE",
    ]
    assert_file_contains(
        PROJ / "tools/run_cutechess_tournament.sh",
        [
            'ENGINE_RESTART="${ENGINE_RESTART:-on}"',
            'MCTS_THREADS="${MCTS_THREADS:-${METALFISH_PURE_MCTS_THREADS:-1}}"',
            'MCTS_PARALLEL_SEARCH="${MCTS_PARALLEL_SEARCH:-false}"',
            'HYBRID_MCTS_KLD="${HYBRID_MCTS_KLD:-0.0}"',
            'HYBRID_AB_ROOT_REJECT_MCTS="${HYBRID_AB_ROOT_REJECT_MCTS:-true}"',
            'HYBRID_MCTS_ROOT_REJECT="${HYBRID_MCTS_ROOT_REJECT:-true}"',
            'HYBRID_MCTS_SHARED_TT="${HYBRID_MCTS_SHARED_TT:-false}"',
            'HYBRID_MCTS_AB_ROOT_HINTS="${HYBRID_MCTS_AB_ROOT_HINTS:-true}"',
            'HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS="${HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS:-25}"',
            'HYBRID_MCTS_AB_ROOT_HINT_COUNT="${HYBRID_MCTS_AB_ROOT_HINT_COUNT:-8}"',
            'HYBRID_AB_POLICY_WEIGHT="${HYBRID_AB_POLICY_WEIGHT:-0.0}"',
            'HYBRID_ROOT_PAWN_LEVER_TIEBREAK="${HYBRID_ROOT_PAWN_LEVER_TIEBREAK:-true}"',
            'HYBRID_TRACE="${HYBRID_TRACE:-false}"',
            'HYBRID_MCTS_MINIBATCH="${HYBRID_MCTS_MINIBATCH:-0}"',
            'HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS="${HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS:-3000}"',
            'HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS="${HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS:-400}"',
            'restart="$ENGINE_RESTART"',
            *tournament_tokens,
        ],
    )
    assert_file_contains(
        PROJ / "tools/lichess_bot.py",
        [
            'env_int("METALFISH_HYBRID_AUTO_AB_THREADS_CAP", 0)',
            'env_int("METALFISH_TRANSFORMER_LOW_TIME_FALLBACK_MS", 3000)',
            'env_int("METALFISH_TRANSFORMER_MIN_MOVE_BUDGET_MS", 400)',
            'env_float("METALFISH_HYBRID_MCTS_KLD", 0.0)',
            '"METALFISH_HYBRID_AB_ROOT_REJECT_MCTS", True',
            '"METALFISH_HYBRID_MCTS_ROOT_REJECT", True',
            '"METALFISH_HYBRID_MCTS_SHARED_TT", False',
            '"METALFISH_HYBRID_MCTS_AB_ROOT_HINTS", True',
            'env_int("METALFISH_HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS", 25)',
            'env_int("METALFISH_HYBRID_MCTS_AB_ROOT_HINT_COUNT", 8)',
            'env_float("METALFISH_HYBRID_AB_POLICY_WEIGHT", 0.0)',
            '"METALFISH_HYBRID_ROOT_PAWN_LEVER_TIEBREAK", True',
            'env_bool_string("METALFISH_HYBRID_TRACE", False)',
            'env_int("METALFISH_HYBRID_MCTS_MINIBATCH", 0)',
            'env_int("METALFISH_HASH_MB", 0)',
            'env_int("METALFISH_RESOURCE_RESERVE_MB", 2048)',
            'os.environ.get("METALFISH_SYZYGY_PATH", "").strip()',
            "def syzygy_path_is_safe(",
            "def resolve_syzygy_path(",
            "def available_memory_mb(",
            "def build_engine_options(",
            "def configure(self, options:",
            "def _run_engine_self_test(",
            "--self-test-only",
            "--quit-after-games",
            '"HybridAutoABThreadsCap": str(HYBRID_AUTO_AB_THREADS_CAP)',
            '"TransformerLowTimeFallbackMs": str(TRANSFORMER_LOW_TIME_FALLBACK_MS)',
            '"TransformerMinMoveBudgetMs": str(TRANSFORMER_MIN_MOVE_BUDGET_MS)',
            '"HybridMCTSMinimumKLDGainPerNode": str(HYBRID_MCTS_KLD)',
            '"HybridABRootRejectMCTS": HYBRID_AB_ROOT_REJECT_MCTS',
            '"HybridMCTSRootReject": HYBRID_MCTS_ROOT_REJECT',
            '"HybridMCTSUseSharedTT": HYBRID_MCTS_SHARED_TT',
            '"HybridMCTSABRootHints": HYBRID_MCTS_AB_ROOT_HINTS',
            '"HybridMCTSABRootHintDelayMs": str(HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS)',
            '"HybridMCTSABRootHintCount": str(HYBRID_MCTS_AB_ROOT_HINT_COUNT)',
            '"HybridABCandidateVerifyMs": str(HYBRID_AB_CANDIDATE_VERIFY_MS)',
            '"HybridABCandidateVerifyCount": str(HYBRID_AB_CANDIDATE_VERIFY_COUNT)',
            '"HybridABPolicyWeight": str(HYBRID_AB_POLICY_WEIGHT)',
            '"HybridRootPawnLeverTieBreak": HYBRID_ROOT_PAWN_LEVER_TIEBREAK',
            '"HybridTrace": HYBRID_TRACE',
            '"MCTSMinibatchSize": str(HYBRID_MCTS_MINIBATCH)',
            "if SYZYGY_PATH:",
            "dynamic up to {MAX_SEARCH_WORKERS}",
            "Syzygy:   {SYZYGY_PATH if SYZYGY_PATH else 'disabled'}",
        ],
    )
    assert_file_contains(
        PROJ / "tests/paper_benchmarks.py",
        [
            "HYBRID_ENV_OPTION_OVERRIDES",
            "def apply_hybrid_env_options",
            "def parse_tactical_fail_under",
            "def enforce_tactical_fail_under",
            "--tactical-fail-under",
            '"HYBRID_MCTS_AB_ROOT_HINTS"',
            '"HYBRID_MCTS_OUT_OF_ORDER_FACTOR"',
            "def benchmark_warmup_ms",
            '"HYBRID_AB_ROOT_REJECT_MCTS"',
            '"TransformerLowTimeFallbackMs"',
            "eng.warmup(benchmark_warmup_ms(cfg, movetime_ms))",
            "eng.warmup(benchmark_warmup_ms(cfg_copy, movetime_ms))",
            "--tactical-repeat",
            "--positions",
            "def select_bk_positions",
            "repeat_count",
            "move_counts",
            "position_ids",
        ],
    )
    assert_file_contains(
        PROJ / "tools/syzygy_manager.py",
        [
            "BASE_URL =",
            '"3-4-5":',
            "def manifest_for_set(",
            "def validate_probes(",
            "def command_download(",
            "def command_validate(",
        ],
    )
    assert_file_contains(
        PROJ / "tests/bk_parity.py",
        [
            "def warmup_movetime_ms",
            "HYBRID_LOW_TIME_FALLBACK_MS",
            "warmup_movetime_ms(movetime_ms, hybrid=True)",
            "def hybrid_time_safety_fallback_warning",
            "hybrid_time_safety_fallback_active",
            "not full Hybrid MCTS",
            "METALFISH_MCTS_ROOT_TRACE_MOVES",
            "--root-trace-moves",
            "--mcts-minibatch-size",
            "--mcts-kld",
            "--mcts-policy-temperature",
            "--mcts-cpuct-at-root",
            "--mcts-parallel-search",
            "--hybrid-ab-root-reject-mcts",
            "--hybrid-mcts-minibatch-size",
            "--hybrid-ane-root-probe",
            'sess.setoption("TransformerLowTimeFallbackMs", "0")',
            'sess.setoption("MCTSMaxThreads", str(mcts_threads))',
            '"MCTSParallelSearch", "true" if mcts_parallel_search else "false"',
            'sess.setoption("MCTSMinimumKLDGainPerNode", str(mcts_kld))',
            'sess.setoption("MCTSMinibatchSize", str(minibatch_size))',
            'sess.setoption("MCTSPolicyTemperature", str(mcts_policy_temperature))',
            'sess.setoption("MCTSCPuctAtRoot", str(mcts_cpuct_at_root))',
            '"HybridABRootRejectMCTS"',
        ],
    )
    assert_file_contains(
        PROJ / "tools/run_tournament_live.py",
        [
            "def apply_hybrid_env_options",
            "BUILTIN_OPENING_LINES",
            "def builtin_openings",
            "built-in fallback",
            '"HYBRID_MCTS_THREADS": "HybridMCTSThreads"',
            '"HYBRID_AB_THREADS": "HybridABThreads"',
            '"HYBRID_AUTO_AB_THREADS_CAP": "HybridAutoABThreadsCap"',
            '"HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS": "TransformerLowTimeFallbackMs"',
            '"HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS": "TransformerMinMoveBudgetMs"',
            '"HYBRID_MCTS_KLD": "HybridMCTSMinimumKLDGainPerNode"',
            '"HYBRID_AB_ROOT_REJECT_MCTS": "HybridABRootRejectMCTS"',
            '"HYBRID_MCTS_ROOT_REJECT": "HybridMCTSRootReject"',
            '"HYBRID_MCTS_SHARED_TT": "HybridMCTSUseSharedTT"',
            '"HYBRID_MCTS_AB_ROOT_HINTS": "HybridMCTSABRootHints"',
            '"HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS": "HybridMCTSABRootHintDelayMs"',
            '"HYBRID_MCTS_AB_ROOT_HINT_COUNT": "HybridMCTSABRootHintCount"',
            '"HYBRID_AB_CANDIDATE_VERIFY_MS": "HybridABCandidateVerifyMs"',
            '"HYBRID_AB_CANDIDATE_VERIFY_COUNT": "HybridABCandidateVerifyCount"',
            '"HYBRID_AB_POLICY_WEIGHT": "HybridABPolicyWeight"',
            '"HYBRID_ROOT_PAWN_LEVER_TIEBREAK": "HybridRootPawnLeverTieBreak"',
            '"HYBRID_TRACE": "HybridTrace"',
            '"HYBRID_MCTS_MINIBATCH": "MCTSMinibatchSize"',
            '"HYBRID_MCTS_OUT_OF_ORDER_FACTOR": "MCTSMaxOutOfOrderEvalsFactor"',
            '"HYBRID_MCTS_MAX_PREFETCH": "MCTSMaxPrefetch"',
            "def hybrid_low_time_warnings",
            "TransformerLowTimeFallbackMs=",
            "AB time-safety fallback rather than transformer search",
            'help="Games per match (default: 6)"',
            "max-moves below 100 can mask long conversion wins",
            '"--max-plies"',
            '"max_plies": args.max_plies',
            "Ply cap:",
            '"--progress-plies"',
            "Print in-game progress every N plies",
            'options["HybridTrace"] = "true"',
            'line.startswith("info string Time safety:")',
            'line.startswith("info string Hybrid: AB root hints from MCTS")',
            'line.startswith("info string Starting Parallel Hybrid Search")',
            'line.startswith("info string Hybrid thread split:")',
            '"--max-moves"',
            "args.save_search_log",
            "def match_result_to_dict",
            "def write_results_json",
            "progress_callback",
            "results.json.tmp",
        ],
    )
    assert_file_contains(
        PROJ / "src/hybrid/hybrid_search.cpp",
        [
            "ABInMCTSRank=",
            "ABInMCTSVisits=",
            "ABInMCTSQ=",
            "ABInMCTSPolicy=",
            "MCTSInABRank=",
            "MCTSInABScore=",
            "MCTSInABAvg=",
            "MCTSInABEffort=",
        ],
    )
    assert_file_contains(
        PROJ / ".github/workflows/ci.yml",
        [
            "python3 tools/download_engine_networks.py",
            "python3 tests/test_benchmark_configs.py",
            "python3 tools/uci_smoke.py",
            "Run Hybrid clock safety smoke",
            "Move\\ Overhead=500",
            'go "wtime 1000 btime 1000 winc 3000 binc 3000"',
            'go "wtime 800 btime 800 winc 3000 binc 3000"',
            '--expect-output "Starting Parallel Hybrid Search"',
            '--expect-output "Time safety: estimated move budget"',
            '--reject-output "Time safety:"',
            "Run MCTS low-node tactical smoke",
            'go "movetime 1000"',
            '--expect-output "Starting Multi-Threaded MCTS Search"',
            "MCTSParallelSearch=false",
            "TransformerLowTimeFallbackMs=0",
            'go "nodes 50"',
            "--expect-bestmove h5f6",
            "Run Apple accelerator tool tests",
            "tools/lc0_coreml_root_value_probe.py",
            "tests/test_lc0_coreml_root_value_probe.py",
            "tests/test_lc0_coreml_value_export.py",
            "Run ANE option/config smoke",
            "--hybrid-ane-root-probe",
            "--hybrid-ane-compute-units cpu-ne",
            "--hybrid-ane-min-budget-ms 1000",
        ],
    )
    assert_file_contains(
        PROJ / ".github/workflows/lichess-puzzles.yml",
        ["python3 tools/download_engine_networks.py"],
    )
    assert_file_contains(
        PROJ / ".github/workflows/hybrid-regression.yml",
        [
            "Hybrid Regression Benchmarks",
            "Checkout main baseline",
            "Repeated hybrid regression benchmark",
            "Download large offline puzzle sample",
            "Large offline puzzle regression",
            "tests/hybrid_regression_compare.py",
            "tools/filter_lichess_puzzle_csv.py",
            "tools/lichess_puzzle_runner.py",
            "tools/compare_puzzle_runs.py",
            "candidate_setoptions",
            "--candidate-setoption",
            "puzzle_count",
            "puzzle_movetime",
            "puzzle_run_max_minutes",
            '--max-minutes "$PUZZLE_RUN_MAX_MINUTES"',
            '--repeat "$BENCH_REPEAT"',
            "--min-candidate-bk-score 20",
            "--max-bk-mean-drop 0.67",
            "--max-perf-regression 0.40",
            "--max-solved-drop 15",
        ],
    )
    assert_file_contains(
        PROJ / "tools/uci_smoke.py",
        [
            "queue.Queue",
            "threading.Thread",
            "Timed out waiting for engine response",
            "Engine exited with status",
            "--expect-bestmove",
            "--expect-output",
            "--reject-output",
        ],
    )
    assert_file_contains(
        PROJ / "tools/download_engine_networks.py",
        [
            "BT4-1024x15x32h-swa-6147500.pb.gz",
            "METALFISH_BT4_WEIGHTS_URL",
            "Decompressing",
            "--nnue-only",
            "--bt4-only",
        ],
    )
    assert_file_contains(
        PROJ / "src/mcts/search.cpp",
        [
            'EnvFlagEnabled("METALFISH_MCTS_ROOT_TRACE")',
            'EnvInt("METALFISH_MCTS_ROOT_TRACE_MOVES", 8, 1, 64)',
            '<< ":n=" << entry.visits',
            '<< ":cn=" << entry.current_visits',
            '<< ":q=" << entry.q',
            '<< ":p=" << entry.policy',
        ],
    )
    assert_file_contains(
        PROJ / "tools/analyze_hybrid_trace.py",
        [
            "HybridTrace:",
            "AB/MCTS disagreements",
            "Trace log coverage",
            "time-safety AB fallbacks",
            "fallback reasons",
            "Feature bucket summary",
            "MCTS-to-AB root hint events",
            "MCTS-to-AB root hint avg moves",
            "MCTS-to-AB root hint sizes",
            "MCTSBestCurrentVisits",
            "MCTSRootCurrentVisits",
            "MCTSConfidenceVisits",
            "MCTSConfidenceRootVisits",
            "comparable_trace_decision",
            "selected_minus_ab",
            "selected_minus_mcts",
            "avg_selected_minus_ab",
            "avg_selected_minus_mcts",
            "--fail-selected-worse-than",
            "selected_quality_failures",
            "Selected-move quality failures",
            "avg_mcts_minus_ab",
            "StockfishProbe",
            "legal_uci",
            "MetalFishProbe",
            "--replay-current",
            "Current MetalFish replay",
            "Replay deltas",
            "replay_hybrid_mcts_threads",
        ],
    )

    print("Benchmark config parity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
