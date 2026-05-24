#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import chess

PROJ = pathlib.Path(__file__).resolve().parent.parent
METALFISH = PROJ / "build" / "metalfish"
LC0 = PROJ / "reference" / "lc0" / "build" / "release" / "lc0"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"
HYBRID_LOW_TIME_FALLBACK_MS = 3000

BK_POSITIONS: List[Tuple[str, List[str], str]] = [
    ("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - -", ["Qd1+"], "BK.01"),
    ("3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - -", ["d5"], "BK.02"),
    ("2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - -", ["f5"], "BK.03"),
    ("rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R w KQkq -", ["e6"], "BK.04"),
    (
        "r1b2rk1/2q1b1pp/p2ppn2/1p6/3QP3/1BN1B3/PPP3PP/R4RK1 w - -",
        ["Nd5", "a4"],
        "BK.05",
    ),
    ("2r3k1/pppR1pp1/4p3/4P1P1/5P2/1P4K1/P1P5/8 w - -", ["g6"], "BK.06"),
    (
        "1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - -",
        ["Nf6"],
        "BK.07",
    ),
    ("4b3/p3kp2/6p1/3pP2p/2pP1P2/4K1P1/P3N2P/8 w - -", ["f5"], "BK.08"),
    ("2kr1bnr/pbpq4/2n1pp2/3p3p/3P1P1B/2N2N1Q/PPP3PP/2KR1B1R w - -", ["f5"], "BK.09"),
    ("3rr1k1/pp3pp1/1qn2np1/8/3p4/PP1R1P2/2P1NQPP/R1B3K1 b - -", ["Ne5"], "BK.10"),
    ("2r1nrk1/p2q1ppp/bp1p4/n1pPp3/P1P1P3/2PBB1N1/4QPPP/R4RK1 w - -", ["f4"], "BK.11"),
    ("r3r1k1/ppqb1ppp/8/4p1NQ/8/2P5/PP3PPP/R3R1K1 b - -", ["Bf5"], "BK.12"),
    ("r2q1rk1/4bppp/p2p4/2pP4/3pP3/3Q4/PP1B1PPP/R3R1K1 w - -", ["b4"], "BK.13"),
    (
        "rnb2r1k/pp2p2p/2pp2p1/q2P1p2/8/1Pb2NP1/PB2PPBP/R2Q1RK1 w - -",
        ["Qd2", "Qe1"],
        "BK.14",
    ),
    ("2r3k1/1p2q1pp/2b1pr2/p1pp4/6Q1/1P1PP1R1/P1PN2PP/5RK1 w - -", ["Qxg7+"], "BK.15"),
    ("r1bqkb1r/4npp1/p1p4p/1p1pP1B1/8/1B6/PPPN1PPP/R2Q1RK1 w kq -", ["Ne4"], "BK.16"),
    (
        "r2q1rk1/1ppnbppp/p2p1nb1/3Pp3/2P1P1P1/2N2N1P/PPB1QP2/R1B2RK1 b - -",
        ["h5"],
        "BK.17",
    ),
    (
        "r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - -",
        ["Nb3"],
        "BK.18",
    ),
    ("3rr3/2pq2pk/p2p1pnp/8/2QBPP2/1P6/P5PP/4RRK1 b - -", ["Rxe4"], "BK.19"),
    ("r4k2/pb2bp1r/1p1qp2p/3pNp2/3P1P2/2N3P1/PPP1Q2P/2KRR3 w - -", ["g4"], "BK.20"),
    ("3rn2k/ppb2rpp/2ppqp2/5N2/2P1P3/1P5Q/PB3PPP/3RR1K1 w - -", ["Nh6"], "BK.21"),
    (
        "2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - -",
        ["Bxe4"],
        "BK.22",
    ),
    ("r1bqk2r/pp2bppp/2p5/3pP3/P2Q1P2/2N1B3/1PP3PP/R4RK1 b kq -", ["f6"], "BK.23"),
    ("r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1 w - -", ["f4"], "BK.24"),
]


def san_to_uci(fen: str, san: str) -> str:
    board = chess.Board(fen)
    return board.parse_san(san).uci()


def warmup_movetime_ms(movetime_ms: int, *, hybrid: bool = False) -> int:
    warmup_ms = min(3000, movetime_ms)
    if hybrid and movetime_ms >= HYBRID_LOW_TIME_FALLBACK_MS:
        warmup_ms = max(warmup_ms, HYBRID_LOW_TIME_FALLBACK_MS)
    return warmup_ms


@dataclass
class SearchResult:
    bestmove: str
    nodes: int
    nps: int
    nn_evals: int
    elapsed: float
    root_summary: str = ""
    hybrid_trace: str = ""
    final_summary: str = ""
    ab_updates: Dict[str, int] = field(default_factory=dict)


@dataclass
class AggregateStats:
    runs: int = 0
    passes: int = 0
    nodes_total: int = 0
    move_counts: Dict[str, int] = field(default_factory=dict)


class UCISession:
    def __init__(
        self, cmd: Sequence[str], name: str, env: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.proc = subprocess.Popen(
            list(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            self.send("uci")
            self.wait_for("uciok", 120)
            self.send("isready")
            self.wait_for("readyok", 120)
        except Exception:
            self.close()
            raise

    def send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_for(self, prefix: str, timeout: int) -> str:
        deadline = time.time() + timeout
        assert self.proc.stdout is not None
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{self.name}: process died (exit={self.proc.returncode})"
                    )
                continue
            line = line.strip()
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"{self.name}: timeout waiting for {prefix}")

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def warmup(self, mode: str, movetime_ms: int, nodes: int) -> None:
        self.send("ucinewgame")
        self.send("position startpos")
        if mode == "nodes":
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go movetime {movetime_ms}")
        self.wait_for("bestmove", 120)
        self.send("isready")
        self.wait_for("readyok", 120)

    def search(
        self,
        fen: str,
        mode: str,
        movetime_ms: int,
        nodes: int,
        collect_ab_updates: bool = False,
    ) -> SearchResult:
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok", 120)
        self.send(f"position fen {fen}")
        if mode == "nodes":
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go movetime {movetime_ms}")

        bestmove = "0000"
        info_nodes = 0
        info_nps = 0
        info_nn_evals = 0
        root_summary = ""
        hybrid_trace = ""
        final_summary = ""
        ab_updates: Dict[str, int] = {}
        t0 = time.time()
        timeout = max(movetime_ms / 1000.0, nodes / 10.0 if nodes > 0 else 0) + 60
        assert self.proc.stdout is not None
        while True:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{self.name}: search timeout after {timeout:.0f}s")
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{self.name}: process died during search (exit={self.proc.returncode})"
                    )
                continue
            line = line.strip()
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) > 1:
                    bestmove = parts[1]
                break
            if line.startswith("info "):
                if " root " in line:
                    root_summary = line.split(" root ", 1)[1]
                if collect_ab_updates and line.endswith(" string hybrid"):
                    move = parse_info_pv_first_move(line)
                    if move:
                        ab_updates[move] = ab_updates.get(move, 0) + 1
                if line.startswith("info string HybridTrace:"):
                    hybrid_trace = line.removeprefix("info string ").strip()
                if line.startswith("info string Final:"):
                    final_summary = line.removeprefix("info string ").strip()
                    evals = parse_keyed_int(final_summary, "MCTSEvals")
                    if evals is not None:
                        info_nn_evals = evals
                parts = line.split()
                for i, tok in enumerate(parts):
                    if tok == "nodes" and i + 1 < len(parts):
                        try:
                            info_nodes = int(parts[i + 1])
                        except ValueError:
                            pass
                    if tok == "nps" and i + 1 < len(parts):
                        try:
                            info_nps = int(parts[i + 1])
                        except ValueError:
                            pass
                    if tok == "nn_evals" and i + 1 < len(parts):
                        try:
                            info_nn_evals = int(parts[i + 1])
                        except ValueError:
                            pass
        # Drain post-search stats (MetalFish prints final counters asynchronously).
        self.send("isready")
        drain_deadline = time.time() + 30
        while True:
            if time.time() > drain_deadline:
                break
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    break
                continue
            line = line.strip()
            if line.startswith("readyok"):
                break
            if line.startswith("info string HybridTrace:"):
                hybrid_trace = line.removeprefix("info string ").strip()
            if line.startswith("info string Final:"):
                final_summary = line.removeprefix("info string ").strip()
                evals = parse_keyed_int(final_summary, "MCTSEvals")
                if evals is not None:
                    info_nn_evals = evals
            if "info string   Nodes:" in line:
                try:
                    info_nodes = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
            if "info string   NPS:" in line:
                try:
                    info_nps = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
            if "info string   NN evals:" in line:
                try:
                    info_nn_evals = int(line.split(":")[-1].strip())
                except ValueError:
                    pass

        return SearchResult(
            bestmove=bestmove,
            nodes=info_nodes,
            nps=info_nps,
            nn_evals=info_nn_evals,
            elapsed=time.time() - t0,
            root_summary=root_summary,
            hybrid_trace=hybrid_trace,
            final_summary=final_summary,
            ab_updates=ab_updates,
        )

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.send("quit")
                self.proc.wait(timeout=5)
        except Exception:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=5)
        finally:
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass


def setup_metalfish(
    sess: UCISession,
    weights: pathlib.Path,
    threads: int,
    deterministic: bool,
    multipv: int,
    minibatch_size: int,
    mcts_kld: float,
    mcts_parallel_search: bool,
    mcts_policy_temperature: Optional[float],
    mcts_cpuct_at_root: Optional[float],
    mcts_fpu_reduction: Optional[float],
    mcts_fpu_reduction_at_root: Optional[float],
    mcts_fpu_value: Optional[float],
    mcts_fpu_value_at_root: Optional[float],
    mcts_fpu_absolute: Optional[bool],
    mcts_fpu_absolute_at_root: Optional[bool],
    mcts_cache_history_length: Optional[int],
    mcts_nn_cache_size: Optional[int],
) -> None:
    mcts_threads = max(1, threads) if mcts_parallel_search else 1
    sess.setoption("UseHybridSearch", "false")
    sess.setoption("UseMCTS", "true")
    sess.setoption("NNWeights", str(weights))
    sess.setoption("TransformerLowTimeFallbackMs", "0")
    sess.setoption("Threads", str(threads))
    sess.setoption("MultiPV", str(multipv))
    sess.setoption("MCTSMaxThreads", str(mcts_threads))
    sess.setoption("MCTSParallelSearch", "true" if mcts_parallel_search else "false")
    sess.setoption("MCTSMinibatchSize", str(minibatch_size))
    sess.setoption("MCTSParityPreset", "true" if deterministic else "false")
    sess.setoption("MCTSAddDirichletNoise", "false")
    sess.setoption("MCTSMinimumKLDGainPerNode", str(mcts_kld))
    if mcts_policy_temperature is not None:
        sess.setoption("MCTSPolicyTemperature", str(mcts_policy_temperature))
        sess.setoption("MCTSPolicySoftmaxTemp", str(mcts_policy_temperature))
    if mcts_cpuct_at_root is not None:
        sess.setoption("MCTSCPuctAtRoot", str(mcts_cpuct_at_root))
    if mcts_fpu_reduction is not None:
        sess.setoption("MCTSFpuReduction", str(mcts_fpu_reduction))
    if mcts_fpu_reduction_at_root is not None:
        sess.setoption("MCTSFpuReductionAtRoot", str(mcts_fpu_reduction_at_root))
    if mcts_fpu_value is not None:
        sess.setoption("MCTSFpuValue", str(mcts_fpu_value))
    if mcts_fpu_value_at_root is not None:
        sess.setoption("MCTSFpuValueAtRoot", str(mcts_fpu_value_at_root))
    if mcts_fpu_absolute is not None:
        sess.setoption("MCTSFpuAbsolute", "true" if mcts_fpu_absolute else "false")
    if mcts_fpu_absolute_at_root is not None:
        sess.setoption(
            "MCTSFpuAbsoluteAtRoot",
            "true" if mcts_fpu_absolute_at_root else "false",
        )
    if mcts_cache_history_length is not None:
        sess.setoption("MCTSCacheHistoryLength", str(mcts_cache_history_length))
    if mcts_nn_cache_size is not None:
        sess.setoption("MCTSNNCacheSize", str(mcts_nn_cache_size))
    sess.send("isready")
    sess.wait_for("readyok", 120)


def setup_metalfish_ab(sess: UCISession, threads: int, multipv: int) -> None:
    sess.setoption("UseMCTS", "false")
    sess.setoption("UseHybridSearch", "false")
    sess.setoption("Threads", str(max(1, threads)))
    sess.setoption("MultiPV", str(multipv))
    sess.send("isready")
    sess.wait_for("readyok", 120)


def setup_metalfish_hybrid(
    sess: UCISession,
    weights: pathlib.Path,
    threads: int,
    deterministic: bool,
    trace: bool,
    mcts_threads: int,
    ab_threads: int,
    multipv: int,
    hybrid_mcts_kld: float,
    hybrid_ab_root_reject_mcts: bool,
    hybrid_root_reject: bool,
    hybrid_shared_tt: bool,
    hybrid_root_hints: bool,
    ab_policy_weight: float,
    root_hint_delay_ms: int,
    root_hint_count: int,
    ab_candidate_verify_ms: int,
    ab_candidate_verify_count: int,
    mcts_minibatch: int,
    low_time_fallback_ms: int,
    root_pawn_lever_tiebreak: bool,
    ane_root_probe: bool,
    ane_root_hints: bool,
    ane_weights: pathlib.Path,
    ane_model_path: pathlib.Path,
    ane_compute_units: str,
    ane_root_hint_count: int,
    ane_root_hint_wait_ms: int,
    ane_min_budget_ms: int,
    mcts_policy_temperature: Optional[float],
    mcts_cpuct_at_root: Optional[float],
    mcts_fpu_reduction: Optional[float],
    mcts_fpu_reduction_at_root: Optional[float],
    mcts_fpu_value: Optional[float],
    mcts_fpu_value_at_root: Optional[float],
    mcts_fpu_absolute: Optional[bool],
    mcts_fpu_absolute_at_root: Optional[bool],
    mcts_cache_history_length: Optional[int],
    mcts_nn_cache_size: Optional[int],
) -> None:
    total_threads = max(3, threads, mcts_threads + ab_threads)
    sess.setoption("UseMCTS", "false")
    sess.setoption("UseHybridSearch", "true")
    sess.setoption("NNWeights", str(weights))
    sess.setoption("Threads", str(total_threads))
    sess.setoption("MultiPV", str(multipv))
    sess.setoption("HybridMCTSThreads", str(mcts_threads))
    sess.setoption("HybridABThreads", str(ab_threads))
    sess.setoption("HybridAutoABThreadsCap", "0")
    sess.setoption("TransformerLowTimeFallbackMs", str(low_time_fallback_ms))
    sess.setoption("TransformerMinMoveBudgetMs", "400")
    sess.setoption("MCTSMaxThreads", str(mcts_threads))
    sess.setoption("MCTSMinibatchSize", str(mcts_minibatch))
    sess.setoption("MCTSParityPreset", "true" if deterministic else "false")
    sess.setoption("MCTSAddDirichletNoise", "false")
    if mcts_policy_temperature is not None:
        sess.setoption("MCTSPolicyTemperature", str(mcts_policy_temperature))
        sess.setoption("MCTSPolicySoftmaxTemp", str(mcts_policy_temperature))
    if mcts_cpuct_at_root is not None:
        sess.setoption("MCTSCPuctAtRoot", str(mcts_cpuct_at_root))
    if mcts_fpu_reduction is not None:
        sess.setoption("MCTSFpuReduction", str(mcts_fpu_reduction))
    if mcts_fpu_reduction_at_root is not None:
        sess.setoption("MCTSFpuReductionAtRoot", str(mcts_fpu_reduction_at_root))
    if mcts_fpu_value is not None:
        sess.setoption("MCTSFpuValue", str(mcts_fpu_value))
    if mcts_fpu_value_at_root is not None:
        sess.setoption("MCTSFpuValueAtRoot", str(mcts_fpu_value_at_root))
    if mcts_fpu_absolute is not None:
        sess.setoption("MCTSFpuAbsolute", "true" if mcts_fpu_absolute else "false")
    if mcts_fpu_absolute_at_root is not None:
        sess.setoption(
            "MCTSFpuAbsoluteAtRoot",
            "true" if mcts_fpu_absolute_at_root else "false",
        )
    if mcts_cache_history_length is not None:
        sess.setoption("MCTSCacheHistoryLength", str(mcts_cache_history_length))
    if mcts_nn_cache_size is not None:
        sess.setoption("MCTSNNCacheSize", str(mcts_nn_cache_size))
    sess.setoption("HybridMCTSMinimumKLDGainPerNode", str(hybrid_mcts_kld))
    sess.setoption(
        "HybridABRootRejectMCTS",
        "true" if hybrid_ab_root_reject_mcts else "false",
    )
    sess.setoption("HybridMCTSRootReject", "true" if hybrid_root_reject else "false")
    sess.setoption("HybridMCTSUseSharedTT", "true" if hybrid_shared_tt else "false")
    sess.setoption("HybridMCTSABRootHints", "true" if hybrid_root_hints else "false")
    sess.setoption("HybridMCTSABRootHintDelayMs", str(root_hint_delay_ms))
    sess.setoption("HybridMCTSABRootHintCount", str(root_hint_count))
    sess.setoption("HybridABCandidateVerifyMs", str(ab_candidate_verify_ms))
    sess.setoption("HybridABCandidateVerifyCount", str(ab_candidate_verify_count))
    sess.setoption(
        "HybridRootPawnLeverTieBreak",
        "true" if root_pawn_lever_tiebreak else "false",
    )
    sess.setoption("HybridANERootProbe", "true" if ane_root_probe else "false")
    sess.setoption("HybridANERootHints", "true" if ane_root_hints else "false")
    sess.setoption("HybridANEWeights", str(ane_weights))
    sess.setoption("HybridANEModelPath", str(ane_model_path))
    sess.setoption("HybridANEComputeUnits", ane_compute_units)
    sess.setoption("HybridANERootHintCount", str(ane_root_hint_count))
    sess.setoption("HybridANERootHintWaitMs", str(ane_root_hint_wait_ms))
    sess.setoption("HybridANEMinBudgetMs", str(ane_min_budget_ms))
    sess.setoption("HybridABPolicyWeight", str(ab_policy_weight))
    sess.setoption("HybridTrace", "true" if trace else "false")
    sess.send("isready")
    sess.wait_for("readyok", 120)


def setup_lc0(sess: UCISession, threads: int) -> None:
    sess.setoption("Threads", "1")
    sess.setoption("Temperature", "0")
    sess.send("isready")
    sess.wait_for("readyok", 120)


def run_engine(
    name: str,
    sess: UCISession,
    mode: str,
    movetime_ms: int,
    nodes: int,
    positions: Sequence[Tuple[str, List[str], str]],
    quiet: bool,
    collect_ab_updates: bool,
) -> Dict[str, SearchResult]:
    results: Dict[str, SearchResult] = {}
    passed = 0
    if not quiet:
        print(f"\n{name}:", flush=True)
    for fen, expected_sans, bk_id in positions:
        expected_uci = set()
        for san in expected_sans:
            try:
                expected_uci.add(san_to_uci(fen, san))
            except Exception:
                expected_uci.add(san.lower().replace("+", "").replace("#", ""))

        out = sess.search(
            fen, mode, movetime_ms, nodes, collect_ab_updates=collect_ab_updates
        )
        results[bk_id] = out
        ok = out.bestmove in expected_uci
        passed += int(ok)
        if not quiet:
            status = "PASS" if ok else "FAIL"
            print(
                f"  {bk_id}: {status:4s} {out.bestmove:8s}"
                f" expected={expected_sans} nodes={out.nodes} nps={out.nps}"
                f" nn_evals={out.nn_evals} t={out.elapsed:.2f}s",
                flush=True,
            )
            if out.root_summary:
                print(f"    root {out.root_summary}", flush=True)
            if out.hybrid_trace:
                print(f"    {out.hybrid_trace}", flush=True)
            if out.final_summary:
                print(f"    {out.final_summary}", flush=True)
            if out.ab_updates:
                print(f"    ABUpdates {format_move_counts(out.ab_updates)}", flush=True)

    if quiet:
        print(f"{name}: {passed}/{len(positions)}", flush=True)
    else:
        print(f"  Score: {passed}/{len(positions)}", flush=True)
    return results


def select_positions(selection: str) -> List[Tuple[str, List[str], str]]:
    if not selection:
        return BK_POSITIONS

    wanted = set()
    for raw_token in selection.replace(",", " ").split():
        token = raw_token.strip().upper()
        if not token:
            continue
        if token.isdigit():
            token = f"BK.{int(token):02d}"
        elif token.startswith("BK.") and token[3:].isdigit():
            token = f"BK.{int(token[3:]):02d}"
        elif token.startswith("BK") and not token.startswith("BK."):
            token = f"BK.{int(token[2:]):02d}"
        wanted.add(token)

    selected = [entry for entry in BK_POSITIONS if entry[2].upper() in wanted]
    found = {entry[2].upper() for entry in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Unknown BK position(s): {', '.join(missing)}")
    return selected


def hybrid_time_safety_fallback_active(args: argparse.Namespace) -> bool:
    want_hybrid = args.engine in ("hybrid", "metalfish-hybrid", "all")
    return (
        want_hybrid
        and args.nodes <= 0
        and args.hybrid_low_time_fallback_ms > 0
        and args.movetime < args.hybrid_low_time_fallback_ms
    )


def hybrid_time_safety_fallback_warning(args: argparse.Namespace) -> str:
    if not hybrid_time_safety_fallback_active(args):
        return ""
    return (
        "WARNING: Hybrid movetime "
        f"{args.movetime}ms is below TransformerLowTimeFallbackMs="
        f"{args.hybrid_low_time_fallback_ms}ms; this run measures the AB "
        "time-safety fallback, not full Hybrid MCTS. Use "
        "--hybrid-low-time-fallback-ms 0 or a larger --movetime for true "
        "Hybrid benchmarking."
    )


def parse_keyed_int(text: str, key: str) -> Optional[int]:
    prefix = key + "="
    for token in text.split():
        if token.startswith(prefix):
            try:
                return int(token[len(prefix) :])
            except ValueError:
                return None
    return None


def parse_info_pv_first_move(line: str) -> str:
    parts = line.split()
    try:
        pv_index = parts.index("pv")
    except ValueError:
        return ""
    if pv_index + 1 >= len(parts):
        return ""
    return parts[pv_index + 1]


def format_move_counts(counts: Dict[str, int]) -> str:
    return (
        "["
        + ", ".join(
            f"{move}:{count}"
            for move, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        )
        + "]"
    )


def update_aggregate(
    aggregate: Dict[str, Dict[str, AggregateStats]],
    positions: Sequence[Tuple[str, List[str], str]],
    all_results: Dict[str, Dict[str, SearchResult]],
) -> None:
    expected_by_id: Dict[str, set[str]] = {}
    for fen, expected_sans, bk_id in positions:
        expected_uci = set()
        for san in expected_sans:
            try:
                expected_uci.add(san_to_uci(fen, san))
            except Exception:
                expected_uci.add(san.lower().replace("+", "").replace("#", ""))
        expected_by_id[bk_id] = expected_uci

    for engine_name, results in all_results.items():
        engine_stats = aggregate.setdefault(engine_name, {})
        for _, _, bk_id in positions:
            out = results[bk_id]
            stats = engine_stats.setdefault(bk_id, AggregateStats())
            stats.runs += 1
            stats.passes += int(out.bestmove in expected_by_id[bk_id])
            stats.nodes_total += out.nodes
            stats.move_counts[out.bestmove] = stats.move_counts.get(out.bestmove, 0) + 1


def print_repeat_summary(
    aggregate: Dict[str, Dict[str, AggregateStats]],
    positions: Sequence[Tuple[str, List[str], str]],
) -> None:
    print("\nRepeat summary:", flush=True)
    for engine_name, engine_stats in aggregate.items():
        total_passes = sum(stats.passes for stats in engine_stats.values())
        total_runs = sum(stats.runs for stats in engine_stats.values())
        print(f"\n{engine_name}: {total_passes}/{total_runs}", flush=True)
        for _, _, bk_id in positions:
            stats = engine_stats.get(bk_id)
            if not stats:
                continue
            avg_nodes = stats.nodes_total // stats.runs if stats.runs else 0
            moves = ", ".join(
                f"{move}:{count}"
                for move, count in sorted(
                    stats.move_counts.items(), key=lambda item: (-item[1], item[0])
                )
            )
            print(
                f"  {bk_id}: {stats.passes}/{stats.runs}"
                f" avg_nodes={avg_nodes} moves=[{moves}]",
                flush=True,
            )


def expected_uci_moves(fen: str, expected_sans: Sequence[str]) -> List[str]:
    moves: List[str] = []
    for san in expected_sans:
        try:
            moves.append(san_to_uci(fen, san))
        except Exception:
            moves.append(san.lower().replace("+", "").replace("#", ""))
    return moves


def search_result_payload(
    fen: str, expected_sans: Sequence[str], bk_id: str, out: SearchResult
) -> Dict[str, object]:
    expected_uci = expected_uci_moves(fen, expected_sans)
    return {
        "id": bk_id,
        "fen": fen,
        "expected_san": list(expected_sans),
        "expected_uci": expected_uci,
        "bestmove": out.bestmove,
        "pass": out.bestmove in set(expected_uci),
        "nodes": out.nodes,
        "nps": out.nps,
        "nn_evals": out.nn_evals,
        "elapsed_sec": round(out.elapsed, 3),
        "root_summary": out.root_summary,
        "hybrid_trace": out.hybrid_trace,
        "final_summary": out.final_summary,
        "ab_updates": dict(sorted(out.ab_updates.items())),
    }


def json_output_path(path: pathlib.Path, repeat: int, run_index: int) -> pathlib.Path:
    if repeat <= 1:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}.run{run_index + 1:02d}{path.suffix}")
    return path / f"bk_parity.run{run_index + 1:02d}.json"


def write_json_report(
    path: pathlib.Path,
    args: argparse.Namespace,
    positions: Sequence[Tuple[str, List[str], str]],
    all_results: Dict[str, Dict[str, SearchResult]],
    run_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {bk_id: (fen, expected_sans) for fen, expected_sans, bk_id in positions}
    report: Dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_index": run_index,
        "mode": "nodes" if args.nodes > 0 else "movetime",
        "movetime_ms": args.movetime,
        "nodes": args.nodes,
        "threads": args.threads,
        "deterministic": args.deterministic,
        "hybrid_time_safety_fallback_active": hybrid_time_safety_fallback_active(
            args
        ),
        "positions": [bk_id for _, _, bk_id in positions],
        "options": {
            "engine": args.engine,
            "multipv": args.multipv,
            "root_trace": args.root_trace,
            "hybrid_trace": args.hybrid_trace,
            "ab_trace": args.ab_trace,
            "hybrid_mcts_threads": args.hybrid_mcts_threads,
            "hybrid_ab_threads": args.hybrid_ab_threads,
            "hybrid_mcts_kld": args.hybrid_mcts_kld,
            "hybrid_root_reject": args.hybrid_root_reject,
            "hybrid_shared_tt": args.hybrid_shared_tt,
            "hybrid_root_hints": args.hybrid_root_hints,
            "hybrid_ab_policy_weight": args.hybrid_ab_policy_weight,
            "hybrid_root_hint_delay_ms": args.hybrid_root_hint_delay_ms,
            "hybrid_root_hint_count": args.hybrid_root_hint_count,
            "hybrid_ab_candidate_verify_ms": args.hybrid_ab_candidate_verify_ms,
            "hybrid_ab_candidate_verify_count": args.hybrid_ab_candidate_verify_count,
            "hybrid_mcts_minibatch": args.hybrid_mcts_minibatch_size,
            "hybrid_low_time_fallback_ms": args.hybrid_low_time_fallback_ms,
            "hybrid_root_pawn_lever_tiebreak": args.hybrid_root_pawn_lever_tiebreak,
            "hybrid_ane_root_probe": args.hybrid_ane_root_probe,
            "hybrid_ane_root_hints": args.hybrid_ane_root_hints,
            "hybrid_ane_weights": str(args.hybrid_ane_weights),
            "hybrid_ane_model_path": str(args.hybrid_ane_model_path),
            "hybrid_ane_compute_units": args.hybrid_ane_compute_units,
            "hybrid_ane_root_hint_count": args.hybrid_ane_root_hint_count,
            "hybrid_ane_root_hint_wait_ms": args.hybrid_ane_root_hint_wait_ms,
            "hybrid_ane_min_budget_ms": args.hybrid_ane_min_budget_ms,
            "mcts_policy_temperature": args.mcts_policy_temperature,
            "mcts_cpuct_at_root": args.mcts_cpuct_at_root,
            "mcts_fpu_reduction": args.mcts_fpu_reduction,
            "mcts_fpu_reduction_at_root": args.mcts_fpu_reduction_at_root,
            "mcts_fpu_value": args.mcts_fpu_value,
            "mcts_fpu_value_at_root": args.mcts_fpu_value_at_root,
            "mcts_fpu_absolute": args.mcts_fpu_absolute,
            "mcts_fpu_absolute_at_root": args.mcts_fpu_absolute_at_root,
            "mcts_cache_history_length": args.mcts_cache_history_length,
            "mcts_nn_cache_size": args.mcts_nn_cache_size,
            "mcts_parallel_search": args.mcts_parallel_search,
        },
        "engines": {},
    }

    engines = report["engines"]
    assert isinstance(engines, dict)
    for engine_name, results in all_results.items():
        payloads = []
        for _, _, bk_id in positions:
            fen, expected_sans = by_id[bk_id]
            payloads.append(
                search_result_payload(fen, expected_sans, bk_id, results[bk_id])
            )
        engines[engine_name] = {
            "score": sum(1 for item in payloads if item["pass"]),
            "total": len(payloads),
            "positions": payloads,
        }

    with open(path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


def engine_score(
    positions: Sequence[Tuple[str, List[str], str]],
    results: Dict[str, SearchResult],
) -> Tuple[int, int]:
    passed = 0
    for fen, expected_sans, bk_id in positions:
        expected = set(expected_uci_moves(fen, expected_sans))
        passed += int(results[bk_id].bestmove in expected)
    return passed, len(positions)


def run_once(
    args: argparse.Namespace,
    aggregate: Optional[Dict[str, Dict[str, AggregateStats]]] = None,
    run_index: int = 0,
) -> int:
    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}")
        return 2
    if args.hybrid_ane_root_probe:
        if not args.hybrid_ane_weights.exists():
            print(f"ERROR: ANE weights not found: {args.hybrid_ane_weights}")
            return 2
        if not args.hybrid_ane_model_path.exists():
            print(f"ERROR: ANE model not found: {args.hybrid_ane_model_path}")
            return 2

    mode = "nodes" if args.nodes > 0 else "movetime"
    threads = 1 if args.deterministic else args.threads
    nodes = args.nodes if args.nodes > 0 else 800
    movetime_ms = args.movetime
    positions = select_positions(args.positions)
    want_ab = args.engine in ("ab", "metalfish-ab", "all")
    want_mcts = args.engine in ("metalfish", "metalfish-mcts", "both", "all")
    want_hybrid = args.engine in ("hybrid", "metalfish-hybrid", "all")
    want_lc0 = args.engine in ("lc0", "both", "all")

    sessions: Dict[str, UCISession] = {}
    try:
        if want_ab:
            if not args.metalfish.exists():
                print(f"ERROR: metalfish not found: {args.metalfish}")
                return 2
            s = UCISession([str(args.metalfish)], "metalfish-ab")
            setup_metalfish_ab(s, threads, args.multipv)
            s.warmup(mode, warmup_movetime_ms(movetime_ms), min(200, nodes))
            sessions["metalfish-ab"] = s

        if want_mcts:
            if not args.metalfish.exists():
                print(f"ERROR: metalfish not found: {args.metalfish}")
                return 2
            env = os.environ.copy()
            if args.root_trace:
                env["METALFISH_MCTS_ROOT_TRACE"] = "1"
                env["METALFISH_MCTS_ROOT_TRACE_MOVES"] = str(args.root_trace_moves)
            s = UCISession([str(args.metalfish)], "metalfish-mcts", env=env)
            setup_metalfish(
                s,
                args.weights,
                threads,
                args.deterministic,
                args.multipv,
                args.mcts_minibatch_size,
                args.mcts_kld,
                args.mcts_parallel_search,
                args.mcts_policy_temperature,
                args.mcts_cpuct_at_root,
                args.mcts_fpu_reduction,
                args.mcts_fpu_reduction_at_root,
                args.mcts_fpu_value,
                args.mcts_fpu_value_at_root,
                args.mcts_fpu_absolute,
                args.mcts_fpu_absolute_at_root,
                args.mcts_cache_history_length,
                args.mcts_nn_cache_size,
            )
            s.warmup(mode, warmup_movetime_ms(movetime_ms), min(200, nodes))
            sessions["metalfish-mcts"] = s

        if want_hybrid:
            if not args.metalfish.exists():
                print(f"ERROR: metalfish not found: {args.metalfish}")
                return 2
            env = os.environ.copy()
            if args.root_trace:
                env["METALFISH_MCTS_ROOT_TRACE"] = "1"
                env["METALFISH_MCTS_ROOT_TRACE_MOVES"] = str(args.root_trace_moves)
            s = UCISession([str(args.metalfish)], "metalfish-hybrid", env=env)
            setup_metalfish_hybrid(
                s,
                args.weights,
                threads,
                args.deterministic,
                args.hybrid_trace,
                args.hybrid_mcts_threads,
                args.hybrid_ab_threads,
                args.multipv,
                args.hybrid_mcts_kld,
                args.hybrid_ab_root_reject_mcts,
                args.hybrid_root_reject,
                args.hybrid_shared_tt,
                args.hybrid_root_hints,
                args.hybrid_ab_policy_weight,
                args.hybrid_root_hint_delay_ms,
                args.hybrid_root_hint_count,
                args.hybrid_ab_candidate_verify_ms,
                args.hybrid_ab_candidate_verify_count,
                args.hybrid_mcts_minibatch_size,
                args.hybrid_low_time_fallback_ms,
                args.hybrid_root_pawn_lever_tiebreak,
                args.hybrid_ane_root_probe,
                args.hybrid_ane_root_hints,
                args.hybrid_ane_weights,
                args.hybrid_ane_model_path,
                args.hybrid_ane_compute_units,
                args.hybrid_ane_root_hint_count,
                args.hybrid_ane_root_hint_wait_ms,
                args.hybrid_ane_min_budget_ms,
                args.mcts_policy_temperature,
                args.mcts_cpuct_at_root,
                args.mcts_fpu_reduction,
                args.mcts_fpu_reduction_at_root,
                args.mcts_fpu_value,
                args.mcts_fpu_value_at_root,
                args.mcts_fpu_absolute,
                args.mcts_fpu_absolute_at_root,
                args.mcts_cache_history_length,
                args.mcts_nn_cache_size,
            )
            s.warmup(
                mode,
                warmup_movetime_ms(movetime_ms, hybrid=True),
                min(200, nodes),
            )
            sessions["metalfish-hybrid"] = s

        if want_lc0:
            if not args.lc0.exists():
                print(f"ERROR: lc0 not found: {args.lc0}")
                return 2
            s = UCISession(
                [
                    str(args.lc0),
                    f"--weights={args.weights}",
                    f"--backend={args.backend}",
                ],
                "lc0",
            )
            setup_lc0(s, threads)
            s.warmup(mode, warmup_movetime_ms(movetime_ms), min(200, nodes))
            sessions["lc0"] = s

        all_results: Dict[str, Dict[str, SearchResult]] = {}
        for name, sess in sessions.items():
            all_results[name] = run_engine(
                name,
                sess,
                mode,
                movetime_ms,
                nodes,
                positions,
                args.quiet,
                args.ab_trace,
            )

        if aggregate is not None:
            update_aggregate(aggregate, positions, all_results)

        names = list(all_results)
        if len(names) > 1 and not args.quiet:
            print("\nAgreement:")
            for left_index, left in enumerate(names):
                for right in names[left_index + 1 :]:
                    agree = 0
                    print(f"  {left} vs {right}:")
                    for _, _, bk_id in positions:
                        lmove = all_results[left][bk_id].bestmove
                        rmove = all_results[right][bk_id].bestmove
                        ok = lmove == rmove
                        agree += int(ok)
                        print(
                            f"    {bk_id}: {'MATCH' if ok else 'DIFF '} "
                            f"{lmove} vs {rmove}"
                        )
                    print(f"    Bestmove agreement: {agree}/{len(positions)}")
        if args.json_out:
            path = json_output_path(args.json_out, args.repeat, run_index)
            write_json_report(path, args, positions, all_results, run_index)
            if not args.quiet:
                print(f"\nSaved JSON report: {path}", flush=True)
        if args.fail_under is not None:
            below_threshold = []
            for name, results in all_results.items():
                passed, total = engine_score(positions, results)
                if passed < args.fail_under:
                    below_threshold.append(f"{name}={passed}/{total}")
            if below_threshold:
                print(
                    "ERROR: score below --fail-under "
                    f"{args.fail_under}: {', '.join(below_threshold)}",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
        return 0
    finally:
        for sess in sessions.values():
            sess.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="BK parity harness")
    parser.add_argument(
        "--engine",
        choices=[
            "ab",
            "metalfish-ab",
            "metalfish",
            "metalfish-mcts",
            "hybrid",
            "metalfish-hybrid",
            "lc0",
            "both",
            "all",
        ],
        default="both",
    )
    parser.add_argument("--movetime", type=int, default=10_000)
    parser.add_argument("--nodes", type=int, default=0, help="If >0, uses go nodes")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--repeat-summary",
        action="store_true",
        help="Print per-engine move counts and pass rates across repeats",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print one score line per engine and suppress per-position details",
    )
    parser.add_argument(
        "--positions",
        default="",
        help="Comma or space separated BK IDs to run, for example BK.09,11",
    )
    parser.add_argument(
        "--root-trace",
        action="store_true",
        help="Print MetalFish MCTS root visit summaries when available",
    )
    parser.add_argument("--root-trace-moves", type=int, default=8)
    parser.add_argument(
        "--hybrid-trace",
        action="store_true",
        help="Print MetalFish Hybrid final arbitration traces",
    )
    parser.add_argument(
        "--ab-trace",
        action="store_true",
        help="Print AB primary-move update counts during Hybrid searches",
    )
    parser.add_argument("--hybrid-mcts-threads", type=int, default=0)
    parser.add_argument("--hybrid-ab-threads", type=int, default=0)
    parser.add_argument("--hybrid-mcts-kld", type=float, default=0.0)
    parser.add_argument(
        "--hybrid-ab-root-reject-mcts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hybrid-root-reject",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hybrid-shared-tt", action="store_true")
    parser.add_argument(
        "--hybrid-root-hints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hybrid-ab-policy-weight", type=float, default=0.0)
    parser.add_argument("--hybrid-root-hint-delay-ms", type=int, default=25)
    parser.add_argument("--hybrid-root-hint-count", type=int, default=8)
    parser.add_argument("--hybrid-ab-candidate-verify-ms", type=int, default=120)
    parser.add_argument("--hybrid-ab-candidate-verify-count", type=int, default=4)
    parser.add_argument("--mcts-minibatch-size", type=int, default=0)
    parser.add_argument("--mcts-kld", type=float, default=0.00005)
    parser.add_argument(
        "--mcts-policy-temperature",
        type=float,
        default=None,
        help="Override MCTSPolicyTemperature/MCTSPolicySoftmaxTemp for allocation probes",
    )
    parser.add_argument(
        "--mcts-cpuct-at-root",
        type=float,
        default=None,
        help="Override MCTSCPuctAtRoot for root allocation probes",
    )
    parser.add_argument(
        "--mcts-fpu-reduction",
        type=float,
        default=None,
        help="Override MCTSFpuReduction for FPU allocation probes",
    )
    parser.add_argument(
        "--mcts-fpu-reduction-at-root",
        type=float,
        default=None,
        help="Override MCTSFpuReductionAtRoot for root FPU probes",
    )
    parser.add_argument(
        "--mcts-fpu-value",
        type=float,
        default=None,
        help="Override MCTSFpuValue for absolute-FPU probes",
    )
    parser.add_argument(
        "--mcts-fpu-value-at-root",
        type=float,
        default=None,
        help="Override MCTSFpuValueAtRoot for root absolute-FPU probes",
    )
    parser.add_argument(
        "--mcts-fpu-absolute",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override MCTSFpuAbsolute for FPU probes",
    )
    parser.add_argument(
        "--mcts-fpu-absolute-at-root",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override MCTSFpuAbsoluteAtRoot for root FPU probes",
    )
    parser.add_argument(
        "--mcts-cache-history-length",
        type=int,
        default=None,
        help="Override MCTSCacheHistoryLength for cache-key breadth probes",
    )
    parser.add_argument(
        "--mcts-nn-cache-size",
        type=int,
        default=None,
        help="Override MCTSNNCacheSize for cache pressure probes",
    )
    parser.add_argument(
        "--mcts-parallel-search",
        action="store_true",
        help="Use all requested threads for pure MCTS throughput experiments",
    )
    parser.add_argument(
        "--hybrid-mcts-minibatch",
        "--hybrid-mcts-minibatch-size",
        dest="hybrid_mcts_minibatch_size",
        type=int,
        default=0,
    )
    parser.add_argument("--hybrid-low-time-fallback-ms", type=int, default=3000)
    parser.add_argument(
        "--hybrid-root-pawn-lever-tiebreak",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hybrid-ane-root-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hybrid-ane-root-hints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hybrid-ane-weights",
        type=pathlib.Path,
        default=PROJ / "networks" / "t1-512x15x8h-distilled-swa-3395000.pb.gz",
    )
    parser.add_argument(
        "--hybrid-ane-model-path",
        type=pathlib.Path,
        default=PROJ / "build" / "coreml" / "t1-512-heads-b8.mlpackage",
    )
    parser.add_argument("--hybrid-ane-compute-units", default="cpu-ne")
    parser.add_argument("--hybrid-ane-root-hint-count", type=int, default=10)
    parser.add_argument("--hybrid-ane-root-hint-wait-ms", type=int, default=250)
    parser.add_argument("--hybrid-ane-min-budget-ms", type=int, default=1000)
    parser.add_argument("--multipv", type=int, default=1)
    parser.add_argument("--backend", default="metal")
    parser.add_argument("--weights", type=pathlib.Path, default=WEIGHTS)
    parser.add_argument("--metalfish", type=pathlib.Path, default=METALFISH)
    parser.add_argument("--lc0", type=pathlib.Path, default=LC0)
    parser.add_argument(
        "--fail-under",
        type=int,
        default=None,
        help="Exit non-zero if any selected engine scores below this many positions",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        default=None,
        help="Write a machine-readable per-position report",
    )
    args = parser.parse_args()

    warning = hybrid_time_safety_fallback_warning(args)
    if warning:
        print(warning, file=sys.stderr, flush=True)

    aggregate: Optional[Dict[str, Dict[str, AggregateStats]]] = (
        {} if args.repeat_summary or args.repeat > 1 else None
    )
    rc = 0
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n=== Run {i + 1}/{args.repeat} ===", flush=True)
        rc = run_once(args, aggregate, i)
        if rc != 0:
            return rc
    if aggregate:
        print_repeat_summary(aggregate, select_positions(args.positions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
