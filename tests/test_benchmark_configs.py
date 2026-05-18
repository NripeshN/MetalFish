#!/usr/bin/env python3
"""Regression checks for benchmark UCI option drift."""
from __future__ import annotations

import json
import os
import pathlib
import platform
import sys

PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "tests"))

import paper_benchmarks  # noqa: E402


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
        "HYBRID_AB_POLICY_WEIGHT": "0.02",
        "HYBRID_MCTS_ROOT_REJECT": "false",
        "HYBRID_MCTS_AB_ROOT_HINTS": "true",
        "HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS": "50",
        "HYBRID_MCTS_OUT_OF_ORDER_FACTOR": "1.5",
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
            "HybridABPolicyWeight": "0.02",
            "HybridMCTSRootReject": "false",
            "HybridMCTSABRootHints": "true",
            "HybridMCTSABRootHintDelayMs": "50",
            "MCTSMaxOutOfOrderEvalsFactor": "1.5",
        },
    )


def detect_paper_engines_clean() -> dict[str, paper_benchmarks.EngineConfig]:
    return with_clean_hybrid_env(
        lambda: paper_benchmarks.detect_engines(threads=8, hash_mb=4096)
    )


def main() -> int:
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

    assert_options_include(
        "paper Hybrid fair resources",
        paper["metalfish-hybrid"].uci_options,
        {
            "Threads": "8",
            "Hash": "4096",
            "HybridMCTSThreads": "2",
            "HybridABThreads": "6",
            "HybridAutoABThreadsCap": "0",
            "MCTSMaxThreads": "2",
            "UseMCTS": "false",
            "UseHybridSearch": "true",
            "MultiPV": "1",
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "TransformerLowTimeFallbackMs": "3000",
            "TransformerMinMoveBudgetMs": "400",
            "HybridMCTSMinimumKLDGainPerNode": "0.0",
            "HybridMCTSRootReject": "true",
            "HybridMCTSUseSharedTT": "false",
            "HybridMCTSABRootHints": "true",
            "HybridMCTSABRootHintDelayMs": "25",
            "HybridMCTSABRootHintCount": "4",
            "HybridABPolicyWeight": "0.0",
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

    assert_file_contains(
        PROJ / "src/uci/engine.cpp",
        [
            'options.add("MCTSPolicySoftmaxTemp"',
            'options.add("MCTSPolicyTemperature"',
            'options.add("MCTSMaxOutOfOrderFactor"',
            'options.add("MCTSMaxOutOfOrderEvalsFactor"',
            'options.add("HybridMCTSABRootHints"',
            'options.add("HybridMCTSABRootHintDelayMs"',
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
            "MCTSMaxThreads": "auto",
            "MCTSParallelSearch": "false",
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "MCTSMinimumKLDGainPerNode": "0.00005",
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
            "HybridMCTSRootReject": "true",
            "HybridMCTSUseSharedTT": "false",
            "HybridMCTSABRootHints": "true",
            "HybridMCTSABRootHintDelayMs": "25",
            "HybridMCTSABRootHintCount": "4",
            "HybridABPolicyWeight": "0.0",
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
        "option.HybridMCTSMinimumKLDGainPerNode=$HYBRID_MCTS_KLD",
        "option.HybridMCTSRootReject=$HYBRID_MCTS_ROOT_REJECT",
        "option.HybridMCTSUseSharedTT=$HYBRID_MCTS_SHARED_TT",
        "option.HybridMCTSABRootHints=$HYBRID_MCTS_AB_ROOT_HINTS",
        "option.HybridMCTSABRootHintDelayMs=$HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS",
        "option.HybridMCTSABRootHintCount=$HYBRID_MCTS_AB_ROOT_HINT_COUNT",
        "option.HybridABPolicyWeight=$HYBRID_AB_POLICY_WEIGHT",
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
            'HYBRID_MCTS_KLD="${HYBRID_MCTS_KLD:-0.0}"',
            'HYBRID_MCTS_ROOT_REJECT="${HYBRID_MCTS_ROOT_REJECT:-true}"',
            'HYBRID_MCTS_SHARED_TT="${HYBRID_MCTS_SHARED_TT:-false}"',
            'HYBRID_MCTS_AB_ROOT_HINTS="${HYBRID_MCTS_AB_ROOT_HINTS:-true}"',
            'HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS="${HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS:-25}"',
            'HYBRID_MCTS_AB_ROOT_HINT_COUNT="${HYBRID_MCTS_AB_ROOT_HINT_COUNT:-4}"',
            'HYBRID_AB_POLICY_WEIGHT="${HYBRID_AB_POLICY_WEIGHT:-0.0}"',
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
            '"METALFISH_HYBRID_MCTS_ROOT_REJECT", True',
            '"METALFISH_HYBRID_MCTS_SHARED_TT", False',
            '"METALFISH_HYBRID_MCTS_AB_ROOT_HINTS", True',
            'env_int("METALFISH_HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS", 25)',
            'env_int("METALFISH_HYBRID_MCTS_AB_ROOT_HINT_COUNT", 4)',
            'env_float("METALFISH_HYBRID_AB_POLICY_WEIGHT", 0.0)',
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
            '"HybridMCTSRootReject": HYBRID_MCTS_ROOT_REJECT',
            '"HybridMCTSUseSharedTT": HYBRID_MCTS_SHARED_TT',
            '"HybridMCTSABRootHints": HYBRID_MCTS_AB_ROOT_HINTS',
            '"HybridMCTSABRootHintDelayMs": str(HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS)',
            '"HybridMCTSABRootHintCount": str(HYBRID_MCTS_AB_ROOT_HINT_COUNT)',
            '"HybridABPolicyWeight": str(HYBRID_AB_POLICY_WEIGHT)',
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
            '"HYBRID_MCTS_AB_ROOT_HINTS"',
            '"HYBRID_MCTS_OUT_OF_ORDER_FACTOR"',
            "def benchmark_warmup_ms",
            '"TransformerLowTimeFallbackMs"',
            "eng.warmup(benchmark_warmup_ms(cfg, movetime_ms))",
            "eng.warmup(benchmark_warmup_ms(cfg_copy, movetime_ms))",
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
        ],
    )
    assert_file_contains(
        PROJ / "tools/run_tournament_live.py",
        [
            "def apply_hybrid_env_options",
            '"HYBRID_MCTS_THREADS": "HybridMCTSThreads"',
            '"HYBRID_AB_THREADS": "HybridABThreads"',
            '"HYBRID_AUTO_AB_THREADS_CAP": "HybridAutoABThreadsCap"',
            '"HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS": "TransformerLowTimeFallbackMs"',
            '"HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS": "TransformerMinMoveBudgetMs"',
            '"HYBRID_MCTS_KLD": "HybridMCTSMinimumKLDGainPerNode"',
            '"HYBRID_MCTS_ROOT_REJECT": "HybridMCTSRootReject"',
            '"HYBRID_MCTS_SHARED_TT": "HybridMCTSUseSharedTT"',
            '"HYBRID_MCTS_AB_ROOT_HINTS": "HybridMCTSABRootHints"',
            '"HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS": "HybridMCTSABRootHintDelayMs"',
            '"HYBRID_MCTS_AB_ROOT_HINT_COUNT": "HybridMCTSABRootHintCount"',
            '"HYBRID_AB_POLICY_WEIGHT": "HybridABPolicyWeight"',
            '"HYBRID_TRACE": "HybridTrace"',
            '"HYBRID_MCTS_MINIBATCH": "MCTSMinibatchSize"',
            '"HYBRID_MCTS_OUT_OF_ORDER_FACTOR": "MCTSMaxOutOfOrderEvalsFactor"',
            '"HYBRID_MCTS_MAX_PREFETCH": "MCTSMaxPrefetch"',
            'help="Games per match (default: 6)"',
            "max-moves below 100 can mask long conversion wins",
            'options["HybridTrace"] = "true"',
            'line.startswith("info string Time safety:")',
            'line.startswith("info string Hybrid: AB root hints from MCTS")',
            'line.startswith("info string Starting Parallel Hybrid Search")',
            'line.startswith("info string Hybrid thread split:")',
            '"--max-moves"',
            "args.save_search_log",
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
            "avg_mcts_minus_ab",
            "StockfishProbe",
            "legal_uci",
        ],
    )

    print("Benchmark config parity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
