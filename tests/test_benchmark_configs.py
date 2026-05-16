#!/usr/bin/env python3
"""Regression checks for benchmark UCI option drift."""
from __future__ import annotations

import json
import pathlib
import sys

PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "tests"))

import bk_parity  # noqa: E402
import paper_benchmarks  # noqa: E402


class CaptureSession:
    def __init__(self) -> None:
        self.options: dict[str, str] = {}

    def setoption(self, name: str, value: str) -> None:
        self.options[name] = value

    def send(self, cmd: str) -> None:
        if cmd != "isready":
            raise AssertionError(f"unexpected command: {cmd}")

    def wait_for(self, prefix: str, timeout: int = 120) -> str:
        if prefix != "readyok":
            raise AssertionError(f"unexpected wait prefix: {prefix}")
        return "readyok"


def capture_parity_mcts() -> dict[str, str]:
    session = CaptureSession()
    bk_parity.setup_metalfish(
        session,
        paper_benchmarks.WEIGHTS,
        threads=1,
        deterministic=False,
        multipv=1,
    )
    return session.options


def capture_parity_hybrid() -> dict[str, str]:
    session = CaptureSession()
    bk_parity.setup_metalfish_hybrid(
        session,
        paper_benchmarks.WEIGHTS,
        threads=1,
        deterministic=False,
        trace=False,
        mcts_threads=2,
        ab_threads=1,
        multipv=1,
        hybrid_mcts_kld=0.0,
        hybrid_root_reject=True,
        hybrid_shared_tt=False,
        ab_policy_weight=0.0,
    )
    return session.options


def assert_options_equal(
    label: str, left: dict[str, str], right: dict[str, str], keys: list[str]
) -> None:
    mismatches = [
        f"{key}: parity={left.get(key)!r} paper={right.get(key)!r}"
        for key in keys
        if left.get(key) != right.get(key)
    ]
    if mismatches:
        raise AssertionError(f"{label} option drift:\n" + "\n".join(mismatches))


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


def main() -> int:
    paper = paper_benchmarks.detect_engines(threads=1)

    mcts_keys = [
        "UseHybridSearch",
        "UseMCTS",
        "NNWeights",
        "Threads",
        "MultiPV",
        "MCTSMaxThreads",
        "MCTSMinibatchSize",
        "MCTSParityPreset",
        "MCTSAddDirichletNoise",
        "MCTSMinimumKLDGainPerNode",
    ]
    assert_options_equal(
        "MCTS",
        capture_parity_mcts(),
        paper["metalfish-mcts"].uci_options,
        mcts_keys,
    )

    hybrid_keys = [
        "UseMCTS",
        "UseHybridSearch",
        "NNWeights",
        "Threads",
        "MultiPV",
        "HybridMCTSThreads",
        "HybridABThreads",
        "TransformerLowTimeFallbackMs",
        "TransformerMinMoveBudgetMs",
        "MCTSMaxThreads",
        "MCTSMinibatchSize",
        "MCTSParityPreset",
        "MCTSAddDirichletNoise",
        "HybridMCTSMinimumKLDGainPerNode",
        "HybridMCTSRootReject",
        "HybridMCTSUseSharedTT",
        "HybridABPolicyWeight",
        "HybridTrace",
    ]
    assert_options_equal(
        "Hybrid",
        capture_parity_hybrid(),
        paper["metalfish-hybrid"].uci_options,
        hybrid_keys,
    )

    assert_file_contains(
        PROJ / "src/uci/engine.cpp",
        [
            'options.add("MCTSPolicySoftmaxTemp"',
            'options.add("MCTSPolicyTemperature"',
            'options.add("MCTSMaxOutOfOrderFactor"',
            'options.add("MCTSMaxOutOfOrderEvalsFactor"',
        ],
    )
    assert_file_contains(
        PROJ / "src/uci/uci.cpp",
        [
            'get_float_option_alias(engine, "MCTSPolicyTemperature"',
            '"MCTSPolicySoftmaxTemp"',
            'get_float_option_alias(engine, "MCTSMaxOutOfOrderEvalsFactor"',
            '"MCTSMaxOutOfOrderFactor"',
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
        },
    )
    assert_options_include(
        "tools MetalFish-MCTS",
        engine_options["MetalFish-MCTS"]["options"],
        {
            "UseHybridSearch": "false",
            "UseMCTS": "true",
            "MultiPV": "1",
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
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "TransformerLowTimeFallbackMs": "5000",
            "TransformerMinMoveBudgetMs": "1200",
            "HybridMCTSMinimumKLDGainPerNode": "0.0",
            "HybridMCTSRootReject": "true",
            "HybridMCTSUseSharedTT": "false",
            "HybridABPolicyWeight": "0.0",
            "HybridAutoABThreadsCap": "2",
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
            'HYBRID_AB_POLICY_WEIGHT="${HYBRID_AB_POLICY_WEIGHT:-0.0}"',
            'HYBRID_TRACE="${HYBRID_TRACE:-false}"',
            'HYBRID_MCTS_MINIBATCH="${HYBRID_MCTS_MINIBATCH:-0}"',
            'HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS="${HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS:-5000}"',
            'HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS="${HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS:-1200}"',
            'restart="$ENGINE_RESTART"',
            *tournament_tokens,
        ],
    )
    assert_file_contains(
        PROJ / "tools/run_distributed_tournament.sh",
        [
            'ENGINE_RESTART="${ENGINE_RESTART:-on}"',
            'HYBRID_MCTS_KLD="${HYBRID_MCTS_KLD:-0.0}"',
            'HYBRID_MCTS_ROOT_REJECT="${HYBRID_MCTS_ROOT_REJECT:-true}"',
            'HYBRID_MCTS_SHARED_TT="${HYBRID_MCTS_SHARED_TT:-false}"',
            'HYBRID_AB_POLICY_WEIGHT="${HYBRID_AB_POLICY_WEIGHT:-0.0}"',
            'HYBRID_TRACE="${HYBRID_TRACE:-false}"',
            'HYBRID_MCTS_MINIBATCH="${HYBRID_MCTS_MINIBATCH:-0}"',
            'HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS="${HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS:-5000}"',
            'HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS="${HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS:-1200}"',
            "restart=$ENGINE_RESTART",
            *tournament_tokens,
        ],
    )
    assert_file_contains(
        PROJ / "tools/lichess_bot.py",
        [
            'env_int("METALFISH_HYBRID_AUTO_AB_THREADS_CAP", 2)',
            'env_int("METALFISH_TRANSFORMER_LOW_TIME_FALLBACK_MS", 5000)',
            'env_int("METALFISH_TRANSFORMER_MIN_MOVE_BUDGET_MS", 1200)',
            'env_float("METALFISH_HYBRID_MCTS_KLD", 0.0)',
            '"METALFISH_HYBRID_MCTS_ROOT_REJECT", True',
            '"METALFISH_HYBRID_MCTS_SHARED_TT", False',
            'env_float("METALFISH_HYBRID_AB_POLICY_WEIGHT", 0.0)',
            'env_bool_string("METALFISH_HYBRID_TRACE", False)',
            'env_int("METALFISH_HYBRID_MCTS_MINIBATCH", 0)',
            '"HybridAutoABThreadsCap": str(HYBRID_AUTO_AB_THREADS_CAP)',
            '"TransformerLowTimeFallbackMs": str(TRANSFORMER_LOW_TIME_FALLBACK_MS)',
            '"TransformerMinMoveBudgetMs": str(TRANSFORMER_MIN_MOVE_BUDGET_MS)',
            '"HybridMCTSMinimumKLDGainPerNode": str(HYBRID_MCTS_KLD)',
            '"HybridMCTSRootReject": HYBRID_MCTS_ROOT_REJECT',
            '"HybridMCTSUseSharedTT": HYBRID_MCTS_SHARED_TT',
            '"HybridABPolicyWeight": str(HYBRID_AB_POLICY_WEIGHT)',
            '"HybridTrace": HYBRID_TRACE',
            '"MCTSMinibatchSize": str(HYBRID_MCTS_MINIBATCH)',
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
            '"HYBRID_AB_POLICY_WEIGHT": "HybridABPolicyWeight"',
            '"HYBRID_TRACE": "HybridTrace"',
            '"HYBRID_MCTS_MINIBATCH": "MCTSMinibatchSize"',
            'options["HybridTrace"] = "true"',
            'line.startswith("info string Time safety:")',
            'line.startswith("info string Starting Parallel Hybrid Search")',
            'line.startswith("info string Hybrid thread split:")',
            '"--max-moves"',
            "args.save_search_log",
        ],
    )

    print("Benchmark config parity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
