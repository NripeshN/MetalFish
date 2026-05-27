#!/usr/bin/env python3
"""Regression checks for benchmark UCI option drift."""
from __future__ import annotations

import json
import os
import pathlib
import platform
import sys
import tempfile

PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "tests"))

import bk_parity  # noqa: E402
import paper_benchmarks  # noqa: E402

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


def assert_file_not_contains(path: pathlib.Path, forbidden: list[str]) -> None:
    text = path.read_text()
    present = [token for token in forbidden if token in text]
    if present:
        raise AssertionError(
            f"{path.relative_to(PROJ)} contains forbidden tokens:\n"
            + "\n".join(present)
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
    expected_pure_mcts_threads = "1" if platform.system() == "Darwin" else "8"

    assert_options_include(
        "paper MCTS strength resources",
        paper["metalfish-mcts"].uci_options,
        {
            "Threads": "8",
            "Hash": "4096",
            "MCTSMaxThreads": expected_pure_mcts_threads,
            "MCTSParallelSearch": "false",
            "UseHybridSearch": "false",
            "UseMCTS": "true",
            "MultiPV": "1",
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
            'options.add("MCTSCudaAutoMinibatchSize"',
            'options.add("NNBackendRequireAccelerator"',
            'options.add("NNCudaDevice"',
            'options.add("NNCudaGraphExecution"',
            'options.add("NNCudaStableExecutionBatchSize"',
            'options.add("NNCudaDeterministicAttentionSoftmax"',
            'options.add("NNCudaFullBufferClear"',
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
            "backend_can_select_cuda",
            "cuda_auto_mcts_minibatch_size",
            "log_mcts_runtime_config",
            "MCTS runtime",
            "Hybrid MCTS runtime",
            "resolve_nn_backend",
            "METALFISH_CUDA_STABLE_EXECUTION_BATCH_SIZE",
            '"MCTSCudaAutoMinibatchSize"',
            '"NNBackendRequireAccelerator"',
            '"NNCudaDevice"',
            '"NNCudaGraphExecution"',
            '"NNCudaStableExecutionBatchSize"',
            '"NNCudaDeterministicAttentionSoftmax"',
            '"NNCudaFullBufferClear"',
            "config.cuda_device",
            "config.cuda_graph_execution",
            "config.cuda_stable_execution_batch_size",
            "config.cuda_deterministic_attention_softmax",
            "config.cuda_full_buffer_clear",
            '"accelerator"',
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
            "--mcts-parallel-search",
            "--hybrid-ab-root-reject-mcts",
            "--hybrid-mcts-minibatch-size",
            "--hybrid-ane-root-probe",
            'sess.setoption("TransformerLowTimeFallbackMs", "0")',
            'sess.setoption("MCTSMaxThreads", str(mcts_threads))',
            '"MCTSParallelSearch", "true" if mcts_parallel_search else "false"',
            'sess.setoption("MCTSMinimumKLDGainPerNode", str(mcts_kld))',
            'sess.setoption("MCTSMinibatchSize", str(minibatch_size))',
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
            "config_.mcts_config.GetBackendConfig()",
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
            "test_nn_comparison",
            "metalfish_nn_probe",
            "python3 tests/test_benchmark_configs.py",
            "python3 tools/uci_smoke.py",
            "python3 tests/test_nn_backend_artifacts.py",
            "Run Hybrid clock safety smoke",
            "Move\\ Overhead=500",
            'go "wtime 1000 btime 1000 winc 3000 binc 3000"',
            'go "wtime 800 btime 800 winc 3000 binc 3000"',
            '--expect-output "Starting Parallel Hybrid Search"',
            '--expect-output "Time safety: estimated move budget"',
            '--reject-output "Time safety:"',
            "Run MCTS low-node tactical smoke",
            "MCTSMaxThreads=1",
            "MCTSParallelSearch=true",
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
            "tools/check_nn_backend_artifacts.py",
            "tools/compare_nn_backend_outputs.py",
            "tools/run_nn_backend_probe_suite.py",
            "Run Metal NN parity report",
            "METALFISH_NN_PARITY_REPORT=build/metal-nn-parity-report.md",
            "METALFISH_NN_BENCH_WARMUP_ITERS=3",
            "metal-nn-comparison.log",
            "metal-nn-probe.log",
            "metal-nn-probe-suite.log",
            "metal-legacy-nn-probe-suite.log",
            "metal-nn-isolation-bt4-legacy.log",
            "metal-nn-isolation-legacy-bt4.log",
            "metal-nn-artifact-manifest.json",
            "check_nn_backend_artifacts.py",
            "--backend metal",
            "--isolation-weights",
            "--full-policy",
            "Metal (MPSGraph) backend",
            "Run ANE option/config smoke",
            "--hybrid-ane-root-probe",
            "--hybrid-ane-compute-units cpu-ne",
            "--hybrid-ane-min-budget-ms 1000",
        ],
    )
    assert_file_contains(
        PROJ / ".github/workflows/portable-ci.yml",
        [
            "push:\n    branches: [main, dev]\n    tags:",
            "tools/write_portable_manifest.py",
            "import_msvc_dev_env.ps1",
            "actions/cache@v5",
            ".vcpkg-bincache",
            "VCPKG_DEFAULT_BINARY_CACHE",
            "VCPKG_BINARY_SOURCES",
            "VCPKG_ROOT",
            "vcpkg-x64-windows-protobuf-zlib-abseil",
            "--output build-linux/PORTABLE_ARTIFACT.md",
            "--output build-windows/PORTABLE_ARTIFACT.md",
            "--output build-windows-msvc/PORTABLE_ARTIFACT.md",
            "--json-output portable-linux/linux-cpu-package-manifest.json",
            "--json-output portable-windows/windows-mingw-cpu-package-manifest.json",
            "windows-msvc-cpu-package-manifest.json",
            "--package-kind",
            "linux-cpu",
            "windows-mingw-cpu",
            "windows-msvc-cpu",
            "build-linux/PORTABLE_ARTIFACT.md",
            "cp build-windows/PORTABLE_ARTIFACT.md",
            "linux-cpu-package-manifest.json",
            "windows-mingw-cpu-package-manifest.json",
            "cmake --build build-windows --target metalfish metalfish_tests metalfish_nn_probe",
            "Windows MSVC CPU",
            "metalfish-windows-x86_64-msvc-cpu",
            "MSVC is the required host toolchain for future Windows CUDA builds.",
            "portable CPU transformer fallback",
            "CPU transformer backend is for correctness/fallback only",
            "metalfish_nn_probe",
            "./build-windows/metalfish_nn_probe.exe",
            "--metadata-only",
            "--construct-backend",
            "Start-Process",
            "WaitForExit(180000)",
            "build-windows/cpu-bt4-eval-smoke.json",
            "cpu-bt4-eval-smoke.json",
            "CPU BT4 eval smoke did not decode WDL",
            "tools/check_cuda_package_artifacts.py",
            "tools/check_cuda_runtime_manifest.py",
            "tools/fetch_cuda_release_artifacts.py",
            '"has_wdl":true',
            '"has_moves_left":true',
            "metalfish.portable_artifact",
            "BT4-1024x15x32h-swa-6147500.pb",
            "Smoke Linux artifact contents",
            "Smoke Windows MinGW artifact contents",
            "Smoke Windows MSVC artifact contents",
            "tar -xzf",
            "bsdtar -xf",
            "Expand-Archive",
            "portable-linux-check/metalfish",
            "portable-windows-check/metalfish.exe",
            "portable-windows-msvc-check",
            "installed\\x64-windows\\bin",
            "- Platform: Linux x86_64",
            "- Platform: Windows x86_64 MinGW",
            "- Platform: Windows x86_64 MSVC",
            "Packaged MinGW runtime DLL missing",
            "Packaged MSVC runtime DLL missing",
            "Packaged runtime DLL is empty",
            "compression-level: 0",
        ],
    )
    assert_file_not_contains(
        PROJ / ".github/workflows/portable-ci.yml",
        [
            "ilammy/msvc-dev-cmd",
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
            "id: puzzle-sample",
            "steps.puzzle-sample.outputs.available == 'true'",
            "Skipping large puzzle regression because the external Lichess puzzle database was unavailable",
            "results/hybrid_regression/puzzles/skipped.md",
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
        PROJ / ".github/workflows/cuda-gpu-gate.yml",
        [
            "CUDA GPU Gate",
            "CUDA L4 Runtime Gate",
            "workflow_dispatch",
            "metal_ci_run_id",
            "require_metal_compare",
            "cublas_workspace_config",
            "Validate Metal comparison inputs",
            "Fetch Metal comparison inputs",
            "tools/fetch_cuda_gpu_gate_inputs.py",
            "--metal-ci-run-id",
            "--expected-sha",
            "github.sha",
            "cuda_gpu_gate_inputs",
            "cuda-gpu-gate-inputs-manifest.json",
            "cuda-gpu-gate-env.sh",
            "python3 -m json.tool",
            "actions: read",
            "google-github-actions/auth",
            "GCP_CREDENTIALS_JSON",
            ". \"${GITHUB_WORKSPACE}/results/cuda_gpu_gate_inputs/cuda-gpu-gate-env.sh\"",
            "METALFISH_REQUIRE_METAL_COMPARE",
            "CUBLAS_WORKSPACE_CONFIG",
            "tools/run_gcp_cuda_gpu_gate.sh",
            "cuda-gpu-gate-${{ github.run_id }}",
        ],
    )
    assert_file_contains(
        PROJ / ".github/workflows/cuda-release.yml",
        [
            "CUDA Release Artifacts",
            "linux_cuda_run_id",
            "windows_cuda_runtime_run_id",
            "expected_sha",
            "tag_name",
            "attach_to_release",
            "tools/fetch_cuda_release_artifacts.py",
            "--linux-cuda-run-id",
            "--windows-cuda-runtime-run-id",
            "--expected-sha",
            "cuda-release-artifacts-manifest.json",
            "cuda-release-artifacts-${{ github.run_id }}",
            "softprops/action-gh-release",
            "fail_on_unmatched_files: true",
            "cuda-release/packages/*",
        ],
    )
    assert_file_contains(
        PROJ / "tools/fetch_cuda_release_artifacts.py",
        [
            "Fetch and validate same-commit CUDA release packages",
            "CUDA GPU Gate",
            "Windows CUDA Runtime Gate",
            "cuda-gpu-gate-*",
            "windows-cuda-runtime-*",
            "--direct-runtime-root",
            "direct-runtime-gates-manifest.json",
            "direct-runtime-root",
            "metalfish*linux-x86_64-cuda.tar.gz",
            "metalfish*windows-x86_64-msvc-cuda.zip",
            "cuda-gpu-runtime-manifest.json",
            "windows-cuda-runtime-gate-manifest.json",
            "metalfish.cuda_release_artifacts",
            "check_cuda_package_artifacts",
            "check_cuda_runtime_manifest",
            "validate_linux_cuda_package",
            "validate_windows_cuda_package",
            "expected_source_commit=expected_sha",
            "runtime_kind=\"linux-cuda\"",
            "runtime_kind=\"windows-cuda\"",
            "require_metal_compare=True",
            "expected_head_sha=expected_sha",
        ],
    )
    assert_file_contains(
        PROJ / "tools/check_cuda_runtime_manifest.py",
        [
            "Validate Linux and Windows CUDA runtime gate manifests",
            "metalfish.cuda_gpu_runtime_gate",
            "metalfish.windows_cuda_runtime_gate",
            "linux-cuda",
            "windows-cuda",
            "remote_status",
            "runtime_status",
            "require_metal_compare",
            "--require-metal-compare",
            "--expected-head-sha",
            "head_sha",
            "Metal BT4 probe suite",
            "Metal legacy probe suite",
            "bt4_compare_status",
            "legacy_compare_status",
            "final_compare_status",
            "CUDA runtime manifest check: PASS",
        ],
    )
    assert_file_contains(
        PROJ / "tools/check_cuda_package_artifacts.py",
        [
            "Validate packaged Linux and Windows CUDA release artifacts",
            "linux-cuda-package-manifest.json",
            "windows-cuda-package-manifest.json",
            "metalfish.portable_artifact",
            "linux-cuda",
            "windows-cuda",
            "cudart64_*.dll",
            "cublas64_*.dll",
            "cublasLt64_*.dll",
            "validate_linux_cuda_package",
            "validate_windows_cuda_package",
            "--expected-source-commit",
            "source_commit",
            "CUDA package artifact check: PASS",
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
        PROJ / "tools/write_portable_manifest.py",
        [
            "MetalFish Portable Artifact",
            "--platform",
            "--backend",
            "--binary",
            "--output",
            "--json-output",
            "--package-kind",
            "metalfish.portable_artifact",
            "sha256_file",
            "NNBackend=stub",
        ],
    )
    assert_file_contains(
        PROJ / "tools/download_engine_networks.py",
        [
            "BT4-1024x15x32h-swa-6147500.pb.gz",
            "METALFISH_BT4_WEIGHTS_URL",
            "legacy-42850.pb.gz",
            "METALFISH_LEGACY_WEIGHTS_URL",
            "METALFISH_NNUE_BIG_URL",
            "METALFISH_NNUE_SMALL_URL",
            "Decompressing",
            "validate_gzip_weights",
            "WEIGHTS_PROTO_MAGIC_PREFIX",
            "re-downloading",
            "--nnue-only",
            "--bt4-only",
            "--legacy-only",
        ],
    )
    assert_file_contains(
        PROJ / "tools/check_nn_backend_artifacts.py",
        [
            "Validate NN backend parity/probe artifacts",
            "--backend-label",
            "--parity-report",
            "--probe-log",
            "--comparison-log",
            "--manifest-out",
            "--require-batch-benchmark",
            "comparison_executor_before",
            "comparison_executor_after",
            "comparison_profile_enabled",
            "benchmark_warmup_line",
            "benchmark_graph_reuse_line",
            "probe",
            "executor",
            "profile_enabled",
            "NN artifact check: PASS",
            "probe output did not decode WDL",
            "probe output did not decode moves-left",
        ],
    )
    assert_file_contains(
        PROJ / "tests/test_nn_comparison.cpp",
        [
            "METALFISH_NN_BENCH_WARMUP_ITERS",
            "METALFISH_NN_BENCH_GRAPH_REUSE_PROBE",
            "benchmark_warmups:",
            "graph_reuse_probe:",
        ],
    )
    assert_file_contains(
        PROJ / "tools/compare_nn_backend_outputs.py",
        [
            "Compare two NN backend probe JSON artifacts",
            "--expected-log",
            "--actual-log",
            "--require-full-policy",
            "--require-wdl",
            "--require-moves-left",
            "--all-probes",
            "NN backend output compare: PASS",
            "top policy move",
            "full policy max delta",
        ],
    )
    assert_file_contains(
        PROJ / "tools/run_nn_backend_probe_suite.py",
        [
            "Run the NN backend probe across a fixed parity-position suite",
            "import json",
            "DEFAULT_POSITIONS",
            "bk07",
            "kiwipete",
            "white-promotion",
            "black-promotion",
            "history-repetition",
            "canonical-black-to-move",
            "castling-history",
            "--line",
            "--full-policy",
            "--cuda-device",
            "--cuda-graph-execution",
            "--cuda-stable-execution-batch-size",
            "--cuda-deterministic-attention-softmax",
            "--cuda-full-buffer-clear",
            "--backend-label",
            "--require-network-info-substring",
            "--require-wdl",
            "--no-require-wdl",
            "--require-moves-left",
            "--no-require-moves-left",
            "--expected-policy-count",
            "load_probe_jsons",
            "validate_probe",
            "NN backend probe suite: PASS",
        ],
    )
    assert_file_contains(
        PROJ / "tools/nn_metal_probe.cpp",
        [
            "BackendConfigFromOptions",
            "ParseBool",
            "--cuda-device",
            "--cuda-graph-execution",
            "--cuda-stable-execution-batch-size",
            "--cuda-deterministic-attention-softmax",
            "--cuda-full-buffer-clear",
            "config.cuda_device",
            "config.cuda_graph_execution",
            "config.cuda_stable_execution_batch_size",
            "--isolation-weights",
            "CreateProbeInstance",
            "EvaluateInstance",
            "RequireIsolationStable",
            "backend isolation output drift",
            '\\"isolation\\":true',
        ],
    )
    assert_file_contains(
        PROJ / "cloudbuild/cuda-entrypoint.yaml",
        [
            "test_nn_comparison",
            "python3 tools/download_engine_networks.py --bt4-only",
            "python3 tools/download_engine_networks.py --legacy-only",
            "metalfish_nn_probe",
            "--backend cuda",
            "--metadata-only",
            "metalfish-cuda-bt4-metadata.json",
            "metalfish-cuda-legacy-metadata.json",
            '"metadata_only":true',
            '"backend":"cuda"',
            '"format":"attention_body=yes',
            '"format":"attention_body=no',
            '"policy_head":"',
            '"value_head":"',
            '"execution_plan":"109 resolved execution steps',
            '"cuda_schedule_fully_supported":true',
            '"cuda_output_mapping_ok":true',
        ],
    )
    assert_file_contains(
        PROJ / "tools/run_cuda_gpu_gate.sh",
        [
            "METALFISH_CUDA_SUMMARY",
            "METALFISH_NN_PARITY_REPORT",
            "MetalFish CUDA GPU Gate Summary",
            "cuda-gpu-nn-comparison.log",
            "cuda-gpu-nn-probe-suite.log",
            "cuda-gpu-legacy-nn-probe-suite.log",
            "cuda-gpu-nn-isolation-bt4-legacy.log",
            "cuda-gpu-nn-isolation-legacy-bt4.log",
            "cuda-gpu-nn-artifact-manifest.json",
            "cuda-gpu-parity-report.md",
            "cuda-gpu-uci-auto-smoke.log",
            "cuda-gpu-uci-accelerator-smoke.log",
            "cuda-gpu-uci-smoke.log",
            "cuda-gpu-uci-bk07-smoke.log",
            "cuda-gpu-uci-hybrid-clock-start-smoke.log",
            "cuda-gpu-uci-hybrid-clock-safety-smoke.log",
            "cuda-gpu-uci-hybrid-auto-smoke.log",
            "BK07_FEN",
            "--expect-bestmove h5f6",
            "--go \"nodes 50\"",
            "--go \"wtime 1000 btime 1000 winc 3000 binc 3000\"",
            "--go \"wtime 800 btime 800 winc 3000 binc 3000\"",
            "--reject-output \"Time safety:\"",
            "--expect-output \"Time safety: estimated move budget\"",
            "check_nn_backend_artifacts.py",
            "run_nn_backend_probe_suite.py",
            "METALFISH_CUDA_PROFILE=0",
            "METALFISH_NN_BENCH_WARMUP_ITERS",
            "METALFISH_NN_BENCH_GRAPH_REUSE_PROBE",
            "METALFISH_CUDA_GRAPH_STATUS_DETAIL",
            "CUDA_STABLE_BATCH_SIZE",
            "CUDA_GRAPH_REPLAY_REQUIRE_ARGS",
            "assert_cuda_isolation_probe_log",
            "METALFISH_CUDA_STABLE_EXECUTION_BATCH_SIZE must be a positive integer",
            "NNBackendRequireAccelerator=true",
            "--setoption NNBackend=accelerator",
            "NNCudaGraphExecution=true",
            "--setoption NNCudaDevice=-1",
            "--setoption NNCudaGraphExecution=true",
            '--setoption NNCudaStableExecutionBatchSize="${CUDA_STABLE_BATCH_SIZE}"',
            "--setoption NNCudaDeterministicAttentionSoftmax=true",
            "--setoption NNCudaFullBufferClear=true",
            "--setoption MCTSMinibatchSize=0",
            "MCTS runtime: backend=accelerator",
            "MCTS runtime: backend=cuda",
            "Hybrid MCTS runtime: backend=accelerator",
            "Hybrid MCTS runtime: backend=cuda",
            "CUDA_RUNTIME_REQUIRE_ARGS",
            "UCI_CUDA_RUNTIME_EXPECT_ARGS",
            "UCI_CUDA_MCTS_WARMUP_EXPECT_ARGS",
            "MCTS backend warmup actual=",
            "metalfish-linux-x86_64-cuda",
            "Linux x86_64 CUDA",
            "cuda-gpu-package-nn-comparison.log",
            "cuda-gpu-package-parity-report.md",
            "cuda-gpu-package-probe.log",
            "cuda-gpu-package-nn-probe-suite.log",
            "cuda-gpu-package-legacy-nn-probe-suite.log",
            "cuda-gpu-package-nn-isolation-bt4-legacy.log",
            "cuda-gpu-package-nn-isolation-legacy-bt4.log",
            "cuda-gpu-package-uci-smoke.log",
            "linux-cuda-package-manifest.json",
            "cp \"${BUILD_DIR}/test_nn_comparison\"",
            "test -x \"${CUDA_PACKAGE_CHECK_DIR}/test_nn_comparison\"",
            "test -s \"${CUDA_PACKAGE_CHECK_DIR}/linux-cuda-package-manifest.json\"",
            "--json-output \"${CUDA_PACKAGE_DIR}/linux-cuda-package-manifest.json\"",
            "--package-kind \"linux-cuda\"",
            '"schema": "metalfish.portable_artifact"',
            "The package includes test_nn_comparison",
            "MCTS evaluator batch parity",
            "SINGLE_REUSE_STRESS_MAX:",
            "REUSE_STRESS_MAX:",
            "batches:",
            "PORTABLE_ARTIFACT.md",
            "tools/write_portable_manifest.py",
            "cuda_device_config=-1",
            "cuda_graph_effective=true",
            "cuda_stable_execution_batch_effective=${CUDA_STABLE_BATCH_SIZE}",
            "cuda_deterministic_attention_softmax=true",
            "cuda_full_buffer_clear_effective=true",
            "--cuda-device -1",
            "--cuda-graph-execution true",
            '--cuda-stable-execution-batch-size "${CUDA_STABLE_BATCH_SIZE}"',
            "--cuda-deterministic-attention-softmax true",
            "--cuda-full-buffer-clear true",
            "--isolation-weights",
            '--backend-label "CUDA transformer backend"',
            '--require-network-info-substring "executor=resolved+graph-replay"',
            "--require-wdl",
            "--require-moves-left",
            "--no-require-wdl",
            "--no-require-moves-left",
            "--expected-policy-count 1858",
            "--full-policy",
            "Parity Report",
            "Batch Timings",
            "UCI Smokes",
            "- accelerator:",
        ],
    )
    assert_file_contains(
        PROJ / "tools/run_windows_cuda_compile_gate.ps1",
        [
            'Require-Command "python"',
            "import_msvc_dev_env.ps1",
            "Running CUDA-linked MCTS module smoke",
            "metalfish_tests.exe mcts",
            "Downloading NNUE files for CUDA-linked AB UCI smoke",
            "tools\\download_engine_networks.py",
            "--nnue-only",
            "Downloading BT4 weights for CUDA-linked metadata probe",
            "--bt4-only",
            "Running CUDA-linked BT4 metadata probe",
            "Downloading legacy 42850 weights for CUDA-linked metadata probe",
            "--legacy-only",
            "Running CUDA-linked legacy metadata probe",
            "metalfish_nn_probe.exe --metadata-only --backend cuda",
            "metalfish_nn_probe.exe",
            "test_nn_comparison.exe",
            "windows-cuda-packaged-bt4-metadata-probe.json",
            "windows-cuda-packaged-legacy-metadata-probe.json",
            "windows-cuda-compile-artifact-manifest.json",
            "windows-cuda-package-manifest.json",
            '"metadata_only":true',
            '"backend":"cuda"',
            '"format":"attention_body=no',
            '"execution_plan":"',
            '"cuda_schedule_fully_supported":true',
            '"cuda_output_mapping_ok":true',
            "Read-ProbeJson",
            "Assert-CudaMetadataProbe",
            "packaged_bt4_matches_build_tree",
            "packaged_legacy_matches_build_tree",
            "test_nn_comparison_exe",
            "gpu_runtime_smoke",
            "not_run_in_github_windows_ci",
            "runtime_gate",
            "EvalFile=$NnueBigPath",
            "EvalFileSmall=$NnueSmallPath",
            "Running CUDA-linked AB UCI smoke",
            "UseMCTS=false",
            "UseHybridSearch=false",
            "depth 1",
            "metalfish-windows-x86_64-msvc-cuda",
            "write_portable_manifest.py",
            "--json-output",
            "--package-kind",
            "windows-cuda",
            "metalfish.portable_artifact",
            "cudart64_*.dll",
            "cublas64_*.dll",
            "cublasLt64_*.dll",
            "Running packaged Windows CUDA AB self-smoke",
            "Packaged JSON manifest missing file entry",
            "Packaged JSON manifest missing CUDA runtime DLL entry",
            "Packaged runtime DLLs",
            "Artifact manifest",
            "windows_cuda_package_manifest_json",
            "The package includes test_nn_comparison.exe",
            "Running packaged Windows CUDA BT4 metadata probe",
            "Running packaged Windows CUDA legacy metadata probe",
            "Packaged Windows CUDA BT4 metadata probe missing expected output",
            "Packaged Windows CUDA legacy metadata probe missing expected output",
            "$PackageName.zip includes test_nn_comparison.exe",
            "- Smoke tests: $SmokeText",
        ],
    )
    assert_file_contains(
        PROJ / "tools/nn_metal_probe.cpp",
        [
            "cuda_schedule_fully_supported",
            "CreateCudaExecutionSchedule",
            "cuda_output_mapping_ok",
            "CreateCudaOutputMapping",
        ],
    )
    assert_file_contains(
        PROJ / ".github/workflows/windows-cuda-compile.yml",
        [
            "branches: [main, dev]",
            "metalfish*windows-x86_64-msvc-cuda.zip",
            "build-windows-cuda/**/*.json",
            "tools/download_engine_networks.py",
            "tools/import_msvc_dev_env.ps1",
            "tools/install_windows_cuda_toolkit.ps1",
            "tools/nn_metal_probe.cpp",
            "tools/write_portable_manifest.py",
            "actions/cache@v5",
            ".vcpkg-bincache",
            "VCPKG_DEFAULT_BINARY_CACHE",
            "VCPKG_BINARY_SOURCES",
            "VCPKG_ROOT",
            "vcpkg-x64-windows-protobuf-zlib-abseil",
            "tools/uci_smoke.py",
            "src/hybrid/**",
            "src/mcts/**",
            "src/nn/cpu/**",
            "src/nn/input_plane_packing.h",
            "src/nn/policy_map.*",
            "src/nn/proto/net.proto",
            "src/nn/tables/**",
            "src/nn/weights_file.h",
            "src/uci/**",
            "tests/nn_input_fixture.*",
            "tests/requirements.txt",
            "tests/test_mcts_module.cpp",
        ],
    )
    assert_file_not_contains(
        PROJ / ".github/workflows/windows-cuda-compile.yml",
        [
            "Jimver/cuda-toolkit",
            "ilammy/msvc-dev-cmd",
        ],
    )
    assert_file_contains(
        PROJ / "tools/install_windows_cuda_toolkit.ps1",
        [
            "developer.download.nvidia.com/compute/cuda",
            "network_installers/cuda_${Version}_windows_network.exe",
            "nvcc_$MajorMinor",
            "cudart_$MajorMinor",
            "cublas_$MajorMinor",
            "cublas_dev_$MajorMinor",
            "thrust_$MajorMinor",
            "visual_studio_integration_$MajorMinor",
            "CUDA_PATH",
            "GITHUB_PATH",
            "Start-Process",
            "nvcc.exe",
            "cublas*.lib",
        ],
    )
    assert_file_contains(
        PROJ / "tools/import_msvc_dev_env.ps1",
        [
            "vswhere.exe",
            "VsDevCmd.bat",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "MSVC developer environment imported",
        ],
    )
    assert_file_contains(
        PROJ / ".github/workflows/windows-cuda-runtime-gate.yml",
        [
            "Windows CUDA Runtime Gate",
            "windows_cuda_run_id",
            "metal_ci_run_id",
            "require_metal_compare",
            "stable_batch_size",
            "Validate Metal comparison inputs",
            "Fetch Windows CUDA runtime inputs",
            "tools/fetch_windows_cuda_runtime_inputs.py",
            "--windows-cuda-run-id",
            "--metal-ci-run-id",
            "--require-metal",
            "--expected-sha",
            "github.sha",
            "windows_cuda_runtime_inputs",
            "runtime-gate-inputs-manifest.json",
            "runtime-gate-env.sh",
            "python3 -m json.tool",
            "METALFISH_REQUIRE_METAL_COMPARE",
            "download_engine_networks.py --legacy-only",
            "g2-standard-8 g2-standard-4",
            "tools/run_gcp_windows_cuda_runtime_gate.sh",
            "windows-cuda-runtime-${{ github.run_id }}",
            "METALFISH_GCP_INSTANCE",
            "METALFISH_GCP_MACHINES",
            ". \"${GITHUB_WORKSPACE}/results/windows_cuda_runtime_inputs/runtime-gate-env.sh\"",
            "METALFISH_WINDOWS_CUDA_STABLE_EXECUTION_BATCH_SIZE",
            "METALFISH_WINDOWS_CUDA_COMPILE_RUN_ID",
        ],
    )
    assert_file_contains(
        PROJ / "tools/fetch_windows_cuda_runtime_inputs.py",
        [
            "Windows CUDA Compile Gate",
            "MetalFish CI",
            "actions/runs/{run_id}/artifacts?per_page=100",
            "actions/artifacts/{artifact.artifact_id}/zip",
            "windows-cuda-compile-*",
            "metalfish-macos-arm64",
            "metalfish*windows-x86_64-msvc-cuda.zip",
            "metal-nn-probe-suite.log",
            "metal-legacy-nn-probe-suite.log",
            "METALFISH_WINDOWS_CUDA_PACKAGE",
            "METALFISH_WINDOWS_CUDA_COMPILE_RUN_ID",
            "METALFISH_METAL_PROBE_SUITE_LOG",
            "METALFISH_METAL_LEGACY_PROBE_SUITE_LOG",
            "METALFISH_REQUIRE_METAL_COMPARE",
            "runtime-gate-env.sh",
            "runtime-gate-inputs-manifest.json",
            "validate_package_manifest",
            "check_cuda_package_artifacts",
            "validate_windows_cuda_package",
            "expected_source_commit=expected_sha",
            "package_manifest",
            "tools/run_gcp_windows_cuda_runtime_gate.sh",
        ],
    )
    assert_file_contains(
        PROJ / "tools/fetch_cuda_gpu_gate_inputs.py",
        [
            "MetalFish CI",
            "gh",
            "actions/runs/{run_id}/artifacts?per_page=100",
            "actions/artifacts/{metal_artifact_id}/zip",
            "safe_extract_zip",
            "metalfish-macos-arm64",
            "metal-nn-probe-suite.log",
            "metal-legacy-nn-probe-suite.log",
            "METALFISH_METAL_PROBE_SUITE_LOG",
            "METALFISH_METAL_LEGACY_PROBE_SUITE_LOG",
            "METALFISH_REQUIRE_METAL_COMPARE",
            "cuda-gpu-gate-env.sh",
            "cuda-gpu-gate-inputs-manifest.json",
            "metalfish.cuda_gpu_gate_inputs",
        ],
    )
    assert_file_contains(
        PROJ / "tools/dispatch_cuda_runtime_gates.py",
        [
            "CUDA GPU Gate",
            "Windows CUDA Runtime Gate",
            "cuda-gpu-gate.yml",
            "windows-cuda-runtime-gate.yml",
            "gh",
            "workflow",
            "run",
            "MetalFish CI",
            "Windows CUDA Compile Gate",
            "find_successful_run",
            "require_dispatchable_workflow",
            "gh",
            "workflow",
            "view",
            "workflow_dispatch can only target workflows present on the",
            "repository default branch",
            "tools/run_cuda_runtime_gates_direct.py",
            "tools/fetch_cuda_gpu_gate_inputs.py",
            "tools/run_gcp_cuda_gpu_gate.sh",
            "tools/fetch_windows_cuda_runtime_inputs.py",
            "tools/run_gcp_windows_cuda_runtime_gate.sh",
            "headSha",
            "conclusion",
            "status",
            "--target",
            "--metal-ci-run-id",
            "--windows-cuda-run-id",
            "--require-metal",
            "--no-require-metal",
            "--dry-run",
            "stable_batch_size",
            "g2-standard-8 g2-standard-4",
        ],
    )
    assert_file_contains(
        PROJ / "tools/run_cuda_runtime_gates_direct.py",
        [
            "Run same-commit CUDA runtime gates directly from a clean worktree",
            "git",
            "worktree",
            "add",
            "--detach",
            "tools/fetch_cuda_gpu_gate_inputs.py",
            "tools/run_gcp_cuda_gpu_gate.sh",
            "tools/fetch_windows_cuda_runtime_inputs.py",
            "tools/run_gcp_windows_cuda_runtime_gate.sh",
            "tools/download_engine_networks.py",
            "--include-legacy",
            "MetalFish CI",
            "Windows CUDA Compile Gate",
            "METALFISH_GCP_ARTIFACT_DIR",
            "METALFISH_CUDA_STABLE_EXECUTION_BATCH_SIZE",
            "METALFISH_WINDOWS_CUDA_STABLE_EXECUTION_BATCH_SIZE",
            "metalfish.cuda_runtime_gates_direct",
            "results",
            "cuda_runtime_direct",
            "--cleanup-worktree",
            "--keep-vms",
            "--dry-run",
            "g2-standard-8 g2-standard-4",
        ],
    )
    assert_file_contains(
        PROJ / "tools/run_gcp_windows_cuda_runtime_gate.sh",
        [
            "nvidia-l4-vws",
            "METALFISH_GCP_MACHINES",
            "g2-standard-8 g2-standard-4",
            "windows-cloud",
            "windows-2022",
            "windows-startup-script-ps1",
            "metalfish-windows-user",
            "metalfish-ssh-pubkey",
            "administrators_authorized_keys",
            "OpenSSH.Server",
            "SSH_READY",
            "METALFISH_WINDOWS_UCI_TRACE",
            "METALFISH_WINDOWS_CUDA_GRAPH",
            "METALFISH_WINDOWS_CUDA_PROFILE",
            "METALFISH_WINDOWS_CUDA_STABLE_EXECUTION_BATCH_SIZE",
            "METALFISH_WINDOWS_CUDA_STABLE_EXECUTION_BATCH_SIZE must be a positive integer",
            "METALFISH_WINDOWS_CUDA_COMPILE_RUN_ID",
            "METALFISH_METAL_PROBE_SUITE_LOG",
            "METALFISH_METAL_LEGACY_PROBE_SUITE_LOG",
            "METALFISH_REQUIRE_METAL_COMPARE",
            "write_runtime_manifest",
            "windows-cuda-runtime-gate-manifest.json",
            "metalfish.windows_cuda_runtime_gate",
            "BT4_COMPARE_STATUS",
            "LEGACY_COMPARE_STATUS",
            "FINAL_COMPARE_STATUS_FOR_MANIFEST",
            "GATE_PACKAGE_ZIP",
            "GATE_WINDOWS_CUDA_COMPILE_RUN_ID",
            "upload_collected_artifacts",
            "METALFISH_LEGACY_NN_WEIGHTS",
            "METALFISH_WINDOWS_CUDA_UCI_TIMEOUT:-420",
            "METALFISH_WINDOWS_CUDA_PROBE_TIMEOUT:-420",
            "METALFISH_WINDOWS_CUDA_COMPARISON_TIMEOUT:-900",
            "METALFISH_WINDOWS_CUDA_HYBRID_UCI_GO:-movetime 8000",
            "metalfish_nn_probe.exe",
            "test_nn_comparison.exe",
            "windows-cuda-package-manifest.json",
            "metalfish.portable_artifact",
            "Packaged JSON manifest missing file entry",
            "Packaged JSON manifest missing CUDA runtime DLL entry",
            "legacy-42850.pb.gz",
            '$ProbeArgs = "--weights "',
            'Invoke-ProbeSmoke -Name "cuda-probe"',
            'Invoke-ProbeSmoke -Name "cuda-legacy-probe"',
            "Invoke-ProbeSuiteSmoke",
            "from tools.run_nn_backend_probe_suite import DEFAULT_POSITIONS",
            "probe-positions.json",
            '"positions": [position.__dict__ for position in DEFAULT_POSITIONS]',
            "positionsDoc.positions",
            "System.Collections.Generic.List[object]",
            "ConvertFrom-Json",
            "cuda-probe-suite",
            "cuda-legacy-probe-suite",
            "cuda-isolation-bt4-legacy",
            "cuda-isolation-legacy-bt4",
            "Invoke-ComparisonSmoke",
            "cuda-nn-comparison",
            "METALFISH_NN_BATCH_REUSE_STRESS",
            "SINGLE_REUSE_STRESS_MAX:",
            "REUSE_STRESS_MAX:",
            "cuda-nn-comparison-parity-report.md",
            "cuda-bk07-mcts",
            "bestmove h5f6",
            "cuda_bk07_mcts",
            "expected_bestmove",
            "--isolation-weights",
            "--full-policy",
            "1858",
            "resolved+graph-replay",
            '"isolation":true',
            "isolation_probes",
            '"has_wdl":false',
            '"has_moves_left":false',
            "Assert-PositiveMetric",
            "[string[]]\\$PositiveMetrics = @()",
            "\\$stdoutTask = \\$proc.StandardOutput.ReadLineAsync()",
            "\\$stdoutTask.Wait([int]\\$remainingMs)",
            "\\$err = \\$proc.StandardError.ReadToEnd()",
            "failed while driving UCI commands",
            "timed out waiting for bestmove",
            "cuda-auto-mcts",
            "hybrid-auto",
            "hybrid-cuda-clock-start",
            "hybrid-cuda-clock-safety",
            "[string[]]\\$RejectedText = @()",
            "contained rejected output",
            "hybrid-cuda-ane-disabled",
            "NNBackend value auto",
            "NNBackendRequireAccelerator value true",
            "MCTSMinibatchSize value 0",
            "CudaNetworkInfoRequiredText",
            "CudaMctsWarmupRequiredText",
            "MCTS backend warmup actual=",
            "Assert-CudaNetworkInfo",
            "cuda_device_config=-1",
            "cuda_graph_effective=true",
            "cuda_stable_execution_batch_effective=${CUDA_STABLE_BATCH_SIZE}",
            "cuda_deterministic_attention_softmax=true",
            "cuda_full_buffer_clear_effective=true",
            "NNCudaGraphExecution value true",
            "NNCudaStableExecutionBatchSize value ${CUDA_STABLE_BATCH_SIZE}",
            "$CudaProbeOptions",
            "--cuda-device -1",
            "--cuda-graph-execution true",
            "--cuda-stable-execution-batch-size ${CUDA_STABLE_BATCH_SIZE}",
            "--cuda-deterministic-attention-softmax true",
            "--cuda-full-buffer-clear true",
            "cuda_stable_execution_batch_size = ${CUDA_STABLE_BATCH_SIZE}",
            "HybridANERootProbe value true",
            "HybridANERootHints value true",
            "HybridANEModelPath value \\$DummyCoreMl",
            "TransformerLowTimeFallbackMs value 0",
            "MCTS runtime: backend=accelerator",
            "MCTS runtime: backend=cuda",
            "Hybrid MCTS runtime: backend=accelerator",
            "Hybrid MCTS runtime: backend=cuda",
            "ANE root probe disabled",
            "Final: MCTSPlayouts=",
            'PositiveMetrics @("MCTSPlayouts", "MCTSEvals", "ABDepth")',
            "windows-cuda-runtime-manifest.json",
            "schema_version",
            "Get-FileHash",
            "Get-GpuInfo",
            "file_count = @(\\$PackageJsonManifest.files).Count",
            "compile_run_id",
            "Read-ProbeJson",
            "Read-SmokeText",
            "legacy_probe",
            "probe_suites",
            "cuda_auto_mcts",
            "cuda_accelerator_mcts",
            "hybrid_auto",
            "hybrid_cuda_clock_start",
            "hybrid_cuda_clock_safety",
            "hybrid_cuda_ane_disabled",
            "ane_disabled",
            "compare_collected_probe_suite",
            "compare_collected_legacy_probe_suite",
            "compare_nn_backend_outputs.py",
            "metal-windows-cuda-nn-probe-suite-summary.json",
            "metal-windows-cuda-legacy-nn-probe-suite-summary.json",
            "METALFISH_GCP_COLLECT_ARTIFACTS=1",
            "--all-probes",
            "Find-Executor",
            "Find-FinalMetrics",
            "ConvertTo-Json -Depth 12",
            "policy_top",
            "backend_selected",
            "cuda-accelerator-mcts",
            "cuda-probe",
            "cuda-legacy-probe",
            "install_gpu_driver.ps1",
            "vc_redist.x64.exe",
            "C:/metalfish/metalfish-windows-cuda.zip",
            "CUDA transformer backend",
            "Starting Parallel Hybrid Search",
            "windows-cuda-runtime-summary.md",
            "collect_remote_artifacts",
        ],
    )
    assert_file_not_contains(
        PROJ / "tools/run_gcp_windows_cuda_runtime_gate.sh",
        [
            "Invoke-ProbeSmoke `",
        ],
    )
    assert_file_contains(
        PROJ / "tools/run_gcp_cuda_gpu_gate.sh",
        [
            "METALFISH_GCP_COLLECT_ARTIFACTS",
            "METALFISH_GCP_ARTIFACT_DIR",
            "METALFISH_GCP_GCS_PREFIX",
            "METALFISH_METAL_PROBE_SUITE_LOG",
            "METALFISH_NN_BENCH_WARMUP_ITERS",
            "METALFISH_NN_BENCH_GRAPH_REUSE_PROBE",
            "METALFISH_CUDA_GRAPH_STATUS_DETAIL",
            "collect_remote_artifacts",
            "compare_collected_probe_suite",
            "cuda-gpu-summary.md",
            "cuda-gpu-tests.log",
            "cuda-gpu-nn-comparison.log",
            "cuda-gpu-nn-probe-suite.log",
            "cuda-gpu-legacy-nn-probe-suite.log",
            "cuda-gpu-nn-isolation-bt4-legacy.log",
            "cuda-gpu-nn-isolation-legacy-bt4.log",
            "cuda-gpu-nn-artifact-manifest.json",
            "cuda-gpu-parity-report.md",
            "cuda-gpu-uci-accelerator-smoke.log",
            "cuda-gpu-uci-bk07-smoke.log",
            "cuda-gpu-uci-hybrid-clock-start-smoke.log",
            "cuda-gpu-uci-hybrid-clock-safety-smoke.log",
            "cuda-gpu-uci-hybrid-auto-smoke.log",
            "cuda-gpu-package-nn-comparison.log",
            "cuda-gpu-package-parity-report.md",
            "cuda-gpu-package-probe.log",
            "cuda-gpu-package-nn-probe-suite.log",
            "cuda-gpu-package-legacy-nn-probe-suite.log",
            "cuda-gpu-package-nn-isolation-bt4-legacy.log",
            "cuda-gpu-package-nn-isolation-legacy-bt4.log",
            "cuda-gpu-package-uci-smoke.log",
            "metalfish-linux-x86_64-cuda.tar.gz",
            "PORTABLE_ARTIFACT.md",
            "linux-cuda-package-manifest.json",
            "METALFISH_METAL_LEGACY_PROBE_SUITE_LOG",
            "METALFISH_REQUIRE_METAL_COMPARE",
            "require_metal_compare",
            "write_runtime_manifest",
            "cuda-gpu-runtime-manifest.json",
            "upload_collected_artifacts",
            "metal-cuda-nn-probe-suite-summary.json",
            "metal-cuda-legacy-nn-probe-suite-summary.json",
            "metal-cuda-nn-probe-suite-compare.log",
            "metal-cuda-legacy-nn-probe-suite-compare.log",
            "--all-probes",
            "REMOTE_STATUS",
            "BT4_COMPARE_STATUS",
            "LEGACY_COMPARE_STATUS",
        ],
    )
    assert_file_contains(
        PROJ / "src/nn/cuda/cuda_executor.cpp",
        [
            "kMaxCudaGraphExecutionCacheEntries",
            "SameCudaGraphResourceState",
            "PruneGraphResourceState",
            "GraphStatusName",
        ],
    )
    assert_file_contains(
        PROJ / "src/mcts/search.cpp",
        [
            'EnvFlagEnabled("METALFISH_MCTS_ROOT_TRACE")',
            'EnvInt("METALFISH_MCTS_ROOT_TRACE_MOVES", 8, 1, 64)',
            "ShouldReplayWarmCudaGraph",
            "params_.minibatch_size > 1",
            "std::clamp(params_.minibatch_size, 1, 256)",
            "replay_warm_cuda_graph ? 3 : 1",
            "MCTS backend warmup actual=",
            "backend_->GetNetworkInfo()",
            '<< ":n=" << entry.visits',
            '<< ":cn=" << entry.current_visits',
            '<< ":q=" << entry.q',
            '<< ":p=" << entry.policy',
        ],
    )
    assert_file_contains(
        PROJ / "src/hybrid/hybrid_search.cpp",
        [
            "#ifdef USE_COREML",
            "ANE root probe disabled: Core ML backend was not compiled",
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
