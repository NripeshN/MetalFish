#!/usr/bin/env python3
"""Paper benchmark suite -- collects all data needed for the MetalFish paper."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import chess

PROJ = pathlib.Path(__file__).resolve().parent.parent
METALFISH = PROJ / "build" / "metalfish"
LC0 = PROJ / "reference" / "lc0" / "build" / "release" / "lc0"
STOCKFISH = PROJ / "reference" / "stockfish" / "src" / "stockfish"
PATRICIA = PROJ / "reference" / "Patricia" / "engine" / "patricia"
BERSERK = PROJ / "reference" / "berserk" / "src" / "berserk"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"
RESULTS_DIR = PROJ / "results"


def _run_int(cmd: Sequence[str]) -> Optional[int]:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    try:
        return int(out.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def detect_default_threads() -> int:
    """Return a fair default worker budget for one engine process.

    On Apple Silicon, the performance-core count is the cleanest full-strength
    default: it gives CPU engines all high-performance cores without forcing
    them onto efficiency cores that can distort NPS and time-to-depth. On other
    systems, fall back to the online logical CPU count.
    """
    if platform.system() == "Darwin":
        perf = _run_int(["sysctl", "-n", "hw.perflevel0.physicalcpu_max"])
        if perf and perf > 0:
            return perf
        logical = _run_int(["sysctl", "-n", "hw.logicalcpu"])
        if logical and logical > 0:
            return logical
    return max(1, os.cpu_count() or 1)


def detect_memory_mib() -> Optional[int]:
    if platform.system() == "Darwin":
        mem_bytes = _run_int(["sysctl", "-n", "hw.memsize"])
        if mem_bytes and mem_bytes > 0:
            return mem_bytes // (1024 * 1024)
    return None


def default_hash_mb() -> int:
    mem_mib = detect_memory_mib()
    if not mem_mib:
        return 1024
    # Benchmarks run engines sequentially. Use enough TT for serious searches
    # while leaving memory for transformer weights, OS cache, and tooling.
    return min(4096, max(512, mem_mib // 8))


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


def select_bk_positions(selection: str) -> List[Tuple[str, List[str], str]]:
    if not selection or not selection.strip():
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


SCALING_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 6 7",
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 9",
]

TOURNAMENT_OPENINGS: List[Tuple[str, str]] = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    (
        "sicilian_najdorf",
        "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    ),
    ("ruy_lopez", "r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 6"),
    (
        "queens_gambit",
        "rnbq1rk1/ppp1bppp/4pn2/3p2B1/2PP4/2N1P3/PP3PPP/R2QKBNR w KQ - 1 6",
    ),
    (
        "kings_indian",
        "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 2 6",
    ),
    ("english", "rnbqkb1r/ppp2ppp/1n6/4p3/8/2N3P1/PP1PPPBP/R1BQK1NR w KQkq - 2 6"),
    ("caro_kann", "rn1qkbnr/pp2pppp/2p3b1/8/3P4/6N1/PPP2PPP/R1BQKBNR w KQkq - 3 6"),
    ("french", "rnbqk2r/pppnbppp/4p3/3pP1B1/3P4/2N5/PPP2PPP/R2QKBNR w KQkq - 1 6"),
]


def san_to_uci(fen: str, san: str) -> str:
    return chess.Board(fen).parse_san(san).uci()


@dataclass
class SearchResult:
    bestmove: str = "0000"
    nodes: int = 0
    nps: int = 0
    nn_evals: int = 0
    depth: int = 0
    elapsed: float = 0.0


class UCIEngine:
    def __init__(self, cmd: Sequence[str], name: str):
        self.name = name
        self.proc = subprocess.Popen(
            list(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
        assert self.proc.stdin
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_for(self, prefix: str, timeout: int = 120) -> str:
        deadline = time.time() + timeout
        assert self.proc.stdout
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

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok", 120)

    def search(self, fen: str, movetime_ms: int = 10000) -> SearchResult:
        # Benchmark FENs are independent positions, not a single game. Reset
        # TT/history between searches so one puzzle cannot train or poison the
        # next one through UCI game state.
        self.new_game()
        self.send(f"position fen {fen}")
        self.send(f"go movetime {movetime_ms}")
        r = SearchResult()
        t0 = time.time()
        timeout = movetime_ms / 1000.0 + 30  # movetime + 30s grace
        assert self.proc.stdout
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
                    r.bestmove = parts[1]
                break
            if line.startswith("info "):
                parts = line.split()
                for i, tok in enumerate(parts):
                    if tok == "nodes" and i + 1 < len(parts):
                        try:
                            r.nodes = int(parts[i + 1])
                        except ValueError:
                            pass
                    if tok == "nps" and i + 1 < len(parts):
                        try:
                            r.nps = int(parts[i + 1])
                        except ValueError:
                            pass
                    if tok == "nn_evals" and i + 1 < len(parts):
                        try:
                            r.nn_evals = int(parts[i + 1])
                        except ValueError:
                            pass
                    if tok == "depth" and i + 1 < len(parts):
                        try:
                            r.depth = int(parts[i + 1])
                        except ValueError:
                            pass
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
            if "NPS:" in line:
                try:
                    r.nps = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
            if "Nodes:" in line:
                try:
                    r.nodes = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
            if "NN evals:" in line:
                try:
                    r.nn_evals = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
        r.elapsed = time.time() - t0
        return r

    def warmup(self, movetime_ms: int = 3000) -> None:
        self.send("position startpos")
        self.send(f"go movetime {movetime_ms}")
        self.wait_for("bestmove", 120)
        self.send("isready")
        self.wait_for("readyok", 120)

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                try:
                    self.send("quit")
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
        finally:
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass


@dataclass
class EngineConfig:
    name: str
    path: pathlib.Path
    cmd_args: List[str] = field(default_factory=list)
    uci_options: Dict[str, str] = field(default_factory=dict)
    available: bool = False


HYBRID_ENV_OPTION_OVERRIDES = {
    "HYBRID_MCTS_THREADS": "HybridMCTSThreads",
    "HYBRID_AB_THREADS": "HybridABThreads",
    "HYBRID_AUTO_AB_THREADS_CAP": "HybridAutoABThreadsCap",
    "HYBRID_TRANSFORMER_LOW_TIME_FALLBACK_MS": "TransformerLowTimeFallbackMs",
    "HYBRID_TRANSFORMER_MIN_MOVE_BUDGET_MS": "TransformerMinMoveBudgetMs",
    "HYBRID_MCTS_KLD": "HybridMCTSMinimumKLDGainPerNode",
    "HYBRID_AB_ROOT_REJECT_MCTS": "HybridABRootRejectMCTS",
    "HYBRID_MCTS_ROOT_REJECT": "HybridMCTSRootReject",
    "HYBRID_MCTS_SHARED_TT": "HybridMCTSUseSharedTT",
    "HYBRID_MCTS_AB_ROOT_HINTS": "HybridMCTSABRootHints",
    "HYBRID_MCTS_AB_ROOT_HINT_DELAY_MS": "HybridMCTSABRootHintDelayMs",
    "HYBRID_MCTS_AB_ROOT_HINT_COUNT": "HybridMCTSABRootHintCount",
    "HYBRID_AB_CANDIDATE_VERIFY_MS": "HybridABCandidateVerifyMs",
    "HYBRID_AB_CANDIDATE_VERIFY_COUNT": "HybridABCandidateVerifyCount",
    "HYBRID_AB_POLICY_WEIGHT": "HybridABPolicyWeight",
    "HYBRID_ROOT_PAWN_LEVER_TIEBREAK": "HybridRootPawnLeverTieBreak",
    "HYBRID_TRACE": "HybridTrace",
    "HYBRID_MCTS_MINIBATCH": "MCTSMinibatchSize",
    "HYBRID_MCTS_OUT_OF_ORDER_FACTOR": "MCTSMaxOutOfOrderEvalsFactor",
    "HYBRID_MCTS_MAX_PREFETCH": "MCTSMaxPrefetch",
}


def env_option(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def apply_hybrid_env_options(options: Dict[str, str]) -> None:
    for env_name, option_name in HYBRID_ENV_OPTION_OVERRIDES.items():
        value = env_option(env_name)
        if value is not None:
            options[option_name] = value
    if "HybridMCTSThreads" in options:
        options["MCTSMaxThreads"] = options["HybridMCTSThreads"]


def hybrid_split_for_threads(threads: int) -> Tuple[int, int]:
    if threads <= 3:
        return 1, max(1, threads - 1)
    # Keep the transformer side present while giving AB the CPU budget it needs
    # for tactical verification. On an 8-performance-core M2 Max this is 1 + 7.
    mcts_threads = 1
    return mcts_threads, max(1, threads - mcts_threads)


def pure_mcts_strength_threads(threads: int) -> int:
    override = env_option("METALFISH_PURE_MCTS_THREADS")
    if override is not None:
        try:
            return max(1, min(threads, int(override)))
        except ValueError:
            pass
    if platform.system() == "Darwin":
        return 1
    return max(1, threads)


def detect_engines(threads: int, hash_mb: int) -> Dict[str, EngineConfig]:
    engines = {}
    mf = METALFISH
    w = str(WEIGHTS)
    threads = max(1, threads)
    hash_mb = max(16, hash_mb)
    pure_mcts_threads = pure_mcts_strength_threads(threads)
    hybrid_threads = max(3, threads)
    hybrid_mcts_threads, hybrid_ab_threads = hybrid_split_for_threads(hybrid_threads)

    engines["metalfish-ab"] = EngineConfig(
        name="MetalFish-AB",
        path=mf,
        uci_options={
            "UseMCTS": "false",
            "UseHybridSearch": "false",
            "Threads": str(threads),
            "Hash": str(hash_mb),
            "MultiPV": "1",
        },
    )

    engines["metalfish-mcts"] = EngineConfig(
        name="MetalFish-MCTS",
        path=mf,
        uci_options={
            "UseHybridSearch": "false",
            "UseMCTS": "true",
            "NNWeights": w,
            "TransformerLowTimeFallbackMs": "0",
            "Threads": str(threads),
            "Hash": str(hash_mb),
            "MultiPV": "1",
            "MCTSMaxThreads": str(pure_mcts_threads),
            "MCTSParallelSearch": "false",
            "MCTSMinibatchSize": "0",
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "PureMCTSSmartPruningFactor": "0.5",
            "PureMCTSCPuctAtRoot": "2.4",
            "MCTSMinimumKLDGainPerNode": "0.00005",
            "TransformerLowTimeFallbackMs": "0",
        },
    )

    engines["metalfish-hybrid"] = EngineConfig(
        name="MetalFish-Hybrid",
        path=mf,
        uci_options={
            "UseMCTS": "false",
            "UseHybridSearch": "true",
            "NNWeights": w,
            "Threads": str(hybrid_threads),
            "Hash": str(hash_mb),
            "MultiPV": "1",
            "HybridMCTSThreads": str(hybrid_mcts_threads),
            "HybridABThreads": str(hybrid_ab_threads),
            "HybridAutoABThreadsCap": "0",
            "TransformerLowTimeFallbackMs": "3000",
            "TransformerMinMoveBudgetMs": "400",
            "MCTSMaxThreads": str(hybrid_mcts_threads),
            "MCTSMinibatchSize": "0",
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "HybridMCTSMinimumKLDGainPerNode": "0.0",
            "HybridABRootRejectMCTS": "true",
            "HybridMCTSRootReject": "true",
            "HybridMCTSUseSharedTT": "false",
            "HybridMCTSABRootHints": "true",
            "HybridMCTSABRootHintDelayMs": "0",
            "HybridMCTSABRootHintCount": "8",
            "HybridABCandidateVerifyMs": "240",
            "HybridABCandidateVerifyCount": "5",
            "HybridABPolicyWeight": "0.0",
            "HybridRootPawnLeverTieBreak": "true",
            "HybridTrace": "false",
        },
    )
    apply_hybrid_env_options(engines["metalfish-hybrid"].uci_options)

    engines["lc0"] = EngineConfig(
        name="Lc0",
        path=LC0,
        cmd_args=[f"--weights={w}", "--backend=metal"],
        uci_options={"Threads": str(threads), "Temperature": "0"},
    )

    engines["stockfish"] = EngineConfig(
        name="Stockfish",
        path=STOCKFISH,
        uci_options={
            "Threads": str(threads),
            "Hash": str(hash_mb),
            "Skill Level": "20",
        },
    )

    engines["patricia"] = EngineConfig(
        name="Patricia",
        path=PATRICIA,
        uci_options={"Threads": str(threads), "Hash": str(hash_mb)},
    )

    engines["berserk"] = EngineConfig(
        name="Berserk",
        path=BERSERK,
        uci_options={"Threads": str(threads), "Hash": str(hash_mb)},
    )

    for eid, cfg in engines.items():
        cfg.available = cfg.path.exists()

    return engines


def start_engine(cfg: EngineConfig) -> UCIEngine:
    cmd = [str(cfg.path)] + cfg.cmd_args
    eng = UCIEngine(cmd, cfg.name)
    for k, v in cfg.uci_options.items():
        eng.setoption(k, v)
    eng.send("isready")
    eng.wait_for("readyok")
    return eng


def config_with_thread_count(cfg: EngineConfig, threads: int) -> EngineConfig:
    options = {**cfg.uci_options, "Threads": str(max(1, threads))}
    if options.get("UseMCTS") == "true":
        options["MCTSMaxThreads"] = str(pure_mcts_strength_threads(max(1, threads)))
        options["MCTSParallelSearch"] = "false"
    if options.get("UseHybridSearch") == "true":
        hybrid_threads = max(3, threads)
        mcts_threads, ab_threads = hybrid_split_for_threads(hybrid_threads)
        options["Threads"] = str(hybrid_threads)
        options["HybridMCTSThreads"] = str(mcts_threads)
        options["HybridABThreads"] = str(ab_threads)
        options["HybridAutoABThreadsCap"] = "0"
        options["MCTSMaxThreads"] = str(mcts_threads)
    return EngineConfig(
        name=cfg.name,
        path=cfg.path,
        cmd_args=list(cfg.cmd_args),
        uci_options=options,
        available=True,
    )


ENGINE_ID_ALIASES = {
    "ab": "metalfish-ab",
    "mcts": "metalfish-mcts",
    "hybrid": "metalfish-hybrid",
    "metalfish": "metalfish-hybrid",
}


def normalize_engine_id(raw: str) -> str:
    token = raw.strip()
    return ENGINE_ID_ALIASES.get(token, token)


def parse_engine_ids(raw: str) -> Optional[List[str]]:
    if not raw:
        return None
    ids = []
    for item in raw.replace(",", " ").split():
        eid = normalize_engine_id(item)
        if eid not in ids:
            ids.append(eid)
    return ids or None


def parse_tactical_fail_under(
    raw: str, selected_engine_ids: Optional[List[str]]
) -> Dict[str, int]:
    if not raw:
        return {}

    tokens = [token for token in raw.replace(",", " ").split() if token]
    if not tokens:
        return {}

    def parse_score(token: str) -> int:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid tactical fail-under score: {token!r}") from exc
        if not 0 <= value <= len(BK_POSITIONS):
            raise ValueError(
                f"tactical fail-under score must be between 0 and {len(BK_POSITIONS)}"
            )
        return value

    if all("=" not in token for token in tokens):
        if len(tokens) != 1:
            raise ValueError(
                "use a single score or engine=score pairs for --tactical-fail-under"
            )
        score = parse_score(tokens[0])
        if selected_engine_ids:
            return {eid: score for eid in selected_engine_ids}
        return {"*": score}

    thresholds: Dict[str, int] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(
                "mixing bare scores and engine=score pairs is not supported"
            )
        engine_id, score_text = token.split("=", 1)
        engine_id = normalize_engine_id(engine_id)
        if not engine_id:
            raise ValueError(f"missing engine id in fail-under token: {token!r}")
        thresholds[engine_id] = parse_score(score_text)
    return thresholds


def parse_thread_counts(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.replace(",", " ").split():
        token = item.strip().lower()
        if not token:
            continue
        if token == "auto":
            value = detect_default_threads()
        else:
            value = int(token)
        if value < 1:
            raise ValueError("thread counts must be positive")
        values.append(value)
    return values or [detect_default_threads()]


def parse_hash_mb(raw: str) -> int:
    token = raw.strip().lower()
    if token == "auto":
        return default_hash_mb()
    value = int(token)
    if value < 16:
        raise ValueError("hash must be at least 16 MB")
    return value


def resource_policy(threads: int, hash_mb: int) -> dict:
    return {
        "thread_policy": (
            "one shared worker budget per engine; Darwin defaults to Apple "
            "Silicon performance cores; pure MetalFish MCTS uses its "
            "strength-first Apple worker cap"
        ),
        "threads": threads,
        "hash_mb": hash_mb,
        "metalfish_mcts_threads": pure_mcts_strength_threads(threads),
        "detected_default_threads": detect_default_threads(),
        "detected_memory_mib": detect_memory_mib(),
        "hybrid_split": {
            "mcts_threads": hybrid_split_for_threads(max(3, threads))[0],
            "ab_threads": hybrid_split_for_threads(max(3, threads))[1],
        },
        "notes": [
            "Reference engines are no longer pinned to one thread.",
            "MetalFish pure MCTS keeps the engine-recommended MCTSMaxThreads cap.",
            "Lc0 receives the requested Threads value for its own backend.",
            "Hybrid receives the same total worker budget split between MCTS and AB.",
        ],
    }


def benchmark_warmup_ms(cfg: EngineConfig, movetime_ms: int) -> int:
    warmup_ms = min(3000, movetime_ms)
    if cfg.uci_options.get("UseHybridSearch") == "true":
        try:
            fallback_ms = int(cfg.uci_options.get("TransformerLowTimeFallbackMs", "0"))
        except ValueError:
            fallback_ms = 0
        if fallback_ms > 0 and movetime_ms >= fallback_ms:
            warmup_ms = max(warmup_ms, fallback_ms)
    return warmup_ms


def run_tactical(
    engines: Dict[str, EngineConfig],
    movetime_ms: int = 10000,
    engine_ids: Optional[List[str]] = None,
    repeat_count: int = 1,
    positions: Optional[Sequence[Tuple[str, List[str], str]]] = None,
) -> dict:
    tactical_positions = list(positions or BK_POSITIONS)
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Tactical Accuracy (Bratko-Kopec)")
    print("=" * 60)

    if engine_ids is None:
        engine_ids = [
            "metalfish-ab",
            "metalfish-mcts",
            "metalfish-hybrid",
            "lc0",
            "stockfish",
        ]

    results = {
        "experiment": "tactical",
        "movetime_ms": movetime_ms,
        "repeat_count": repeat_count,
        "position_count": len(tactical_positions),
        "position_ids": [bk_id for _, _, bk_id in tactical_positions],
        "position_expected": {
            bk_id: expected_sans for _, expected_sans, bk_id in tactical_positions
        },
        "timestamp": datetime.datetime.now().isoformat(),
        "engines": {},
        "agreement": {},
    }

    all_bestmoves: Dict[str, Dict[str, str]] = {}

    for eid in engine_ids:
        cfg = engines.get(eid)
        if not cfg or not cfg.available:
            print(f"\n  SKIP {eid}: not available")
            continue

        print(f"\n--- {cfg.name} ---")
        eng = None
        try:
            eng = start_engine(cfg)
            try:
                eng.warmup(benchmark_warmup_ms(cfg, movetime_ms))
            except TimeoutError:
                print(f"  WARNING: warmup timeout, continuing...")

            score = 0
            completed_runs = 0
            positions = []
            bestmoves = {}
            error_message = None

            for fen, expected_sans, bk_id in tactical_positions:
                expected_uci = set()
                for san in expected_sans:
                    try:
                        expected_uci.add(san_to_uci(fen, san))
                    except Exception:
                        expected_uci.add(san.lower().replace("+", "").replace("#", ""))

                runs = []
                move_counts: Dict[str, int] = {}
                passes = 0
                for run_idx in range(repeat_count):
                    try:
                        r = eng.search(fen, movetime_ms)
                    except (RuntimeError, TimeoutError) as e:
                        error_message = str(e)
                        print(f"  {bk_id}: ERROR {e}")
                        print(
                            f"  Engine crashed/timed out — reporting partial results for {cfg.name}"
                        )
                        break

                    ok = r.bestmove in expected_uci
                    passes += int(ok)
                    score += int(ok)
                    completed_runs += 1
                    move_counts[r.bestmove] = move_counts.get(r.bestmove, 0) + 1
                    runs.append((r, ok))

                if not runs:
                    break

                modal_move = sorted(
                    move_counts.items(), key=lambda item: (-item[1], item[0])
                )[0][0]
                bestmoves[bk_id] = modal_move
                first_result = runs[0][0]
                nodes_total = sum(item[0].nodes for item in runs)
                nps_total = sum(item[0].nps for item in runs)
                nn_evals_total = sum(item[0].nn_evals for item in runs)
                elapsed_total = sum(item[0].elapsed for item in runs)

                pos_data = {
                    "id": bk_id,
                    "fen": fen,
                    "expected": expected_sans,
                    "bestmove": modal_move,
                    "pass": passes == len(runs),
                    "passes": passes,
                    "runs": len(runs),
                    "pass_rate": round(passes / max(1, len(runs)), 3),
                    "move_counts": move_counts,
                    "nodes": nodes_total // max(1, len(runs)),
                    "nps": nps_total // max(1, len(runs)),
                    "nn_evals": nn_evals_total // max(1, len(runs)),
                    "depth": max(item[0].depth for item in runs),
                    "time_s": round(elapsed_total / max(1, len(runs)), 2),
                    "first_bestmove": first_result.bestmove,
                }
                positions.append(pos_data)
                if repeat_count == 1:
                    status = "PASS" if passes else "FAIL"
                    print(
                        f"  {bk_id}: {status:4s} {modal_move:8s} exp={expected_sans}"
                        f" n={pos_data['nodes']} nps={pos_data['nps']}"
                        f" t={pos_data['time_s']:.1f}s"
                    )
                else:
                    moves = ",".join(
                        f"{move}:{count}"
                        for move, count in sorted(
                            move_counts.items(), key=lambda item: (-item[1], item[0])
                        )
                    )
                    print(
                        f"  {bk_id}: {passes}/{len(runs)} modal={modal_move:8s}"
                        f" exp={expected_sans} avg_n={pos_data['nodes']}"
                        f" avg_nps={pos_data['nps']} moves=[{moves}]"
                    )

                if error_message:
                    break

            total = len(tactical_positions) * repeat_count
            completed = len(positions)
            suffix = (
                ""
                if completed_runs == total
                else f", completed {completed_runs}/{total}"
            )
            print(f"  Score: {score}/{total} ({100*score/total:.1f}%){suffix}")

            avg_nps = sum(p["nps"] for p in positions) // max(1, len(positions))
            results["engines"][eid] = {
                "name": cfg.name,
                "score": score,
                "total": total,
                "completed": completed_runs,
                "completed_positions": completed,
                "complete": completed_runs == total,
                "error": error_message,
                "solve_rate": round(score / total, 3),
                "avg_nps": avg_nps,
                "positions": positions,
            }
            all_bestmoves[eid] = bestmoves
        except (RuntimeError, TimeoutError) as e:
            print(f"  ERROR starting {cfg.name}: {e}")
        finally:
            if eng:
                eng.close()

    eids = list(all_bestmoves.keys())
    for i in range(len(eids)):
        for j in range(i + 1, len(eids)):
            a, b = eids[i], eids[j]
            agree = sum(
                1
                for bk_id in [p[2] for p in tactical_positions]
                if all_bestmoves[a].get(bk_id) == all_bestmoves[b].get(bk_id)
            )
            key = f"{a}_vs_{b}"
            results["agreement"][key] = {
                "agree": agree,
                "total": len(tactical_positions),
                "rate": round(agree / len(tactical_positions), 3),
            }
            print(f"  Agreement {a} vs {b}: {agree}/{len(tactical_positions)}")

    return results


def enforce_tactical_fail_under(results: dict, thresholds: Dict[str, int]) -> List[str]:
    if not thresholds:
        return []

    engines = results.get("engines", {})
    if "*" in thresholds:
        items = [(eid, thresholds["*"]) for eid in engines]
    else:
        items = list(thresholds.items())

    failures = []
    for eid, threshold in items:
        data = engines.get(eid)
        if not data:
            failures.append(f"{eid}=missing (floor {threshold})")
            continue
        if not data.get("complete", False):
            completed = data.get("completed", 0)
            total = data.get("total", len(BK_POSITIONS))
            failures.append(f"{eid}=incomplete {completed}/{total} (floor {threshold})")
            continue
        score = int(data.get("score", 0))
        if score < threshold:
            total = data.get("total", len(BK_POSITIONS))
            failures.append(f"{eid}={score}/{total} (floor {threshold})")
    return failures


def run_nps(
    engines: Dict[str, EngineConfig], thread_counts: List[int], movetime_ms: int = 10000
) -> dict:
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: NPS Throughput (MetalFish MCTS vs Lc0)")
    print("=" * 60)

    test_fens = [
        ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        (
            "sicilian",
            "r1b1kbnr/pp3ppp/2n1p3/2ppP3/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 5",
        ),
        (
            "middlegame",
            "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 6 7",
        ),
        ("endgame", "8/5pk1/5p1p/2R5/pp6/1P4PP/P4PK1/2r5 w - - 0 36"),
    ]

    results = {
        "experiment": "nps_throughput",
        "movetime_ms": movetime_ms,
        "timestamp": datetime.datetime.now().isoformat(),
        "thread_counts": thread_counts,
        "engines": {},
    }

    for eid in ["metalfish-mcts", "lc0"]:
        cfg = engines.get(eid)
        if not cfg or not cfg.available:
            print(f"\n  SKIP {eid}: not available")
            continue

        results["engines"][eid] = {"name": cfg.name, "by_threads": {}}

        for tc in thread_counts:
            print(f"\n--- {cfg.name} @ {tc} threads ---")
            cfg_copy = config_with_thread_count(cfg, tc)
            eng = start_engine(cfg_copy)
            try:
                try:
                    eng.warmup(benchmark_warmup_ms(cfg_copy, movetime_ms))
                except TimeoutError:
                    print("  WARNING: warmup timeout")

                pos_results = []
                for label, fen in test_fens:
                    r = eng.search(fen, movetime_ms)
                    pos_results.append(
                        {
                            "position": label,
                            "fen": fen,
                            "nodes": r.nodes,
                            "nps": r.nps,
                            "nn_evals": r.nn_evals,
                            "depth": r.depth,
                            "time_s": round(r.elapsed, 2),
                        }
                    )
                    print(f"  {label}: nps={r.nps} nodes={r.nodes} depth={r.depth}")

                avg_nps = sum(p["nps"] for p in pos_results) // max(1, len(pos_results))
                results["engines"][eid]["by_threads"][str(tc)] = {
                    "threads": tc,
                    "avg_nps": avg_nps,
                    "positions": pos_results,
                }
            finally:
                eng.close()

    mf_data = results["engines"].get("metalfish-mcts", {}).get("by_threads", {})
    lc0_data = results["engines"].get("lc0", {}).get("by_threads", {})
    if mf_data and lc0_data:
        print("\n--- Speedup Ratios ---")
        for tc_str in mf_data:
            mf_nps = mf_data[tc_str]["avg_nps"]
            lc0_nps = lc0_data.get(tc_str, {}).get("avg_nps", 1)
            ratio = mf_nps / max(1, lc0_nps)
            print(f"  {tc_str}T: MetalFish {mf_nps} vs Lc0 {lc0_nps} = {ratio:.2f}x")

    return results


def run_scaling(
    engines: Dict[str, EngineConfig], thread_counts: List[int], movetime_ms: int = 10000
) -> dict:
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Thread Scaling")
    print("=" * 60)

    results = {
        "experiment": "thread_scaling",
        "movetime_ms": movetime_ms,
        "timestamp": datetime.datetime.now().isoformat(),
        "thread_counts": thread_counts,
        "engines": {},
    }

    for eid in ["metalfish-ab", "metalfish-mcts", "metalfish-hybrid"]:
        cfg = engines.get(eid)
        if not cfg or not cfg.available:
            print(f"\n  SKIP {eid}: not available")
            continue

        results["engines"][eid] = {"name": cfg.name, "by_threads": {}}
        baseline_nps = None

        for tc in thread_counts:
            print(f"\n--- {cfg.name} @ {tc} threads ---")
            cfg_copy = config_with_thread_count(cfg, tc)
            eng = start_engine(cfg_copy)
            try:
                try:
                    eng.warmup(benchmark_warmup_ms(cfg_copy, movetime_ms))
                except TimeoutError:
                    print("  WARNING: warmup timeout")

                total_nps = 0
                total_nodes = 0
                total_depth = 0
                n_pos = len(SCALING_POSITIONS)

                for i, fen in enumerate(SCALING_POSITIONS):
                    r = eng.search(fen, movetime_ms)
                    total_nps += r.nps
                    total_nodes += r.nodes
                    total_depth += r.depth
                    print(
                        f"  pos {i+1}/{n_pos}: nps={r.nps} nodes={r.nodes} depth={r.depth}"
                    )

                avg_nps = total_nps // n_pos
                avg_depth = total_depth / n_pos
                if baseline_nps is None:
                    baseline_nps = max(1, avg_nps)
                scaling = avg_nps / baseline_nps

                results["engines"][eid]["by_threads"][str(tc)] = {
                    "threads": tc,
                    "avg_nps": avg_nps,
                    "total_nodes": total_nodes,
                    "avg_depth": round(avg_depth, 1),
                    "scaling_factor": round(scaling, 2),
                }
                print(
                    f"  AVG: nps={avg_nps} depth={avg_depth:.1f} scaling={scaling:.2f}x"
                )
            finally:
                eng.close()

    return results


def play_game(
    white: UCIEngine,
    black: UCIEngine,
    fen: str,
    movetime_ms: int = 5000,
    max_plies: int = 200,
) -> dict:
    """Play one game and return result plus enough data to debug it."""
    white.new_game()
    black.new_game()
    board = chess.Board(fen)
    moves: List[str] = []
    for _ in range(max_plies):
        if board.is_game_over():
            return {
                "result": board.result(),
                "termination": (
                    board.outcome().termination.name if board.outcome() else "GAME_OVER"
                ),
                "ply_count": len(moves),
                "final_fen": board.fen(),
                "moves": moves,
            }
        if board.can_claim_threefold_repetition():
            return {
                "result": "1/2-1/2",
                "termination": "THREEFOLD_REPETITION_CLAIM",
                "ply_count": len(moves),
                "final_fen": board.fen(),
                "moves": moves,
            }
        if board.can_claim_fifty_moves():
            return {
                "result": "1/2-1/2",
                "termination": "FIFTY_MOVES_CLAIM",
                "ply_count": len(moves),
                "final_fen": board.fen(),
                "moves": moves,
            }
        eng = white if board.turn == chess.WHITE else black
        eng.send(f"position fen {board.fen()}")
        eng.send(f"go movetime {movetime_ms}")
        try:
            line = eng.wait_for("bestmove", movetime_ms // 1000 + 30)
        except TimeoutError:
            return {
                "result": "0-1" if board.turn == chess.WHITE else "1-0",
                "termination": "TIMEOUT",
                "ply_count": len(moves),
                "final_fen": board.fen(),
                "moves": moves,
            }
        move_str = line.split()[1] if len(line.split()) > 1 else "0000"
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                moves.append(move_str)
                board.push(move)
            else:
                return {
                    "result": "0-1" if board.turn == chess.WHITE else "1-0",
                    "termination": "ILLEGAL_MOVE",
                    "ply_count": len(moves),
                    "final_fen": board.fen(),
                    "moves": moves,
                    "illegal_move": move_str,
                }
        except Exception:
            return {
                "result": "0-1" if board.turn == chess.WHITE else "1-0",
                "termination": "INVALID_MOVE",
                "ply_count": len(moves),
                "final_fen": board.fen(),
                "moves": moves,
                "invalid_move": move_str,
            }
    return {
        "result": "1/2-1/2",
        "termination": "MAX_MOVES",
        "ply_count": len(moves),
        "final_fen": board.fen(),
        "moves": moves,
    }


def run_tournament(
    engines: Dict[str, EngineConfig],
    games_per_match: int = 10,
    movetime_ms: int = 5000,
    max_game_plies: int = 200,
) -> dict:
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Elo Tournament")
    print("=" * 60)

    matches = [
        ("metalfish-ab", "patricia"),
        ("metalfish-ab", "berserk"),
        ("metalfish-mcts", "lc0"),
        ("metalfish-mcts", "patricia"),
        ("metalfish-hybrid", "patricia"),
        ("metalfish-hybrid", "berserk"),
        ("metalfish-ab", "metalfish-hybrid"),
        ("metalfish-ab", "metalfish-mcts"),
    ]

    results = {
        "experiment": "tournament",
        "movetime_ms": movetime_ms,
        "games_per_match": games_per_match,
        "max_game_plies": max_game_plies,
        "timestamp": datetime.datetime.now().isoformat(),
        "matches": [],
    }

    for e1_id, e2_id in matches:
        c1, c2 = engines.get(e1_id), engines.get(e2_id)
        if not c1 or not c1.available or not c2 or not c2.available:
            print(f"\n  SKIP {e1_id} vs {e2_id}: engine not available")
            continue

        print(f"\n--- {c1.name} vs {c2.name} ({games_per_match} games) ---")
        wins, draws, losses = 0, 0, 0
        games = []

        eng1 = start_engine(c1)
        eng2 = start_engine(c2)
        try:
            for g in range(games_per_match):
                opening_name, opening_fen = TOURNAMENT_OPENINGS[
                    g % len(TOURNAMENT_OPENINGS)
                ]
                if g % 2 == 0:
                    game_data = play_game(
                        eng1, eng2, opening_fen, movetime_ms, max_game_plies
                    )
                    result = game_data["result"]
                    white_name, black_name = c1.name, c2.name
                    if result == "1-0":
                        wins += 1
                    elif result == "0-1":
                        losses += 1
                    else:
                        draws += 1
                else:
                    game_data = play_game(
                        eng2, eng1, opening_fen, movetime_ms, max_game_plies
                    )
                    result = game_data["result"]
                    white_name, black_name = c2.name, c1.name
                    if result == "1-0":
                        losses += 1
                    elif result == "0-1":
                        wins += 1
                    else:
                        draws += 1
                games.append(
                    {
                        "game": g + 1,
                        "opening": opening_name,
                        "white": white_name,
                        "black": black_name,
                        **game_data,
                    }
                )
                print(
                    f"  Game {g+1} ({opening_name}): {result} "
                    f"{game_data['termination']} {game_data['ply_count']} plies "
                    f"(W{wins}-D{draws}-L{losses})"
                )
        finally:
            eng1.close()
            eng2.close()

        total = wins + draws + losses
        score = wins + draws * 0.5
        pct = score / max(1, total)
        match_data = {
            "engine1": e1_id,
            "engine2": e2_id,
            "name1": c1.name,
            "name2": c2.name,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": score,
            "total": total,
            "pct": round(pct, 3),
            "games": games,
        }
        results["matches"].append(match_data)
        print(f"  Result: {c1.name} {wins}W-{draws}D-{losses}L ({pct:.1%})")

    return results


def generate_summary(all_results: Dict[str, dict]) -> str:
    first_result = next(iter(all_results.values()), {})
    resources = first_result.get("resource_policy", {})
    hybrid_split = resources.get("hybrid_split", {})
    lines = [
        "# MetalFish Paper Benchmark Results",
        "",
        f"Generated: {datetime.datetime.now().isoformat()}",
        "",
        "## Resource Policy",
        "",
        f"- Threads: {resources.get('threads', 'unknown')}",
        f"- Hash: {resources.get('hash_mb', 'unknown')} MB",
        f"- MetalFish pure MCTS workers: "
        f"{resources.get('metalfish_mcts_threads', 'unknown')}",
        f"- Hybrid split: MCTS={hybrid_split.get('mcts_threads', 'unknown')}, "
        f"AB={hybrid_split.get('ab_threads', 'unknown')}",
        "- CPU/reference engines use the same thread and hash budget where supported.",
        "",
    ]

    tactical = all_results.get("tactical")
    if tactical:
        repeat_count = max(1, int(tactical.get("repeat_count", 1)))
        position_ids = tactical.get("position_ids") or [p[2] for p in BK_POSITIONS]
        position_expected = tactical.get("position_expected") or {
            bk_id: expected for _, expected, bk_id in BK_POSITIONS
        }
        position_count = int(tactical.get("position_count", len(position_ids)))
        total_runs = position_count * repeat_count
        solved_header = "Solved" if repeat_count == 1 else "Solved Runs"
        completed_header = "Completed" if repeat_count == 1 else "Completed Runs"
        lines += [
            f"## Table 1: Tactical Accuracy (Bratko-Kopec, {position_count} positions)",
            "",
        ]
        if repeat_count > 1:
            lines += [f"Repeats per position: {repeat_count}", ""]
        lines += [
            f"| Engine | {solved_header} | {completed_header} | Rate (%) | Avg NPS | Status |",
            "|--------|--------|-----------|----------|---------|--------|",
        ]
        for eid, data in tactical.get("engines", {}).items():
            status = "complete" if data.get("complete", True) else "partial"
            lines.append(
                f"| {data['name']} | {data['score']}/{data.get('total', total_runs)} | "
                f"{data.get('completed', len(data.get('positions', [])))}/"
                f"{data.get('total', total_runs)} | "
                f"{data['solve_rate']*100:.1f} | {data['avg_nps']:,} | "
                f"{status} |"
            )
            if data.get("error"):
                lines.append(f"<!-- {data['name']} partial error: {data['error']} -->")
        lines += [
            "",
            "### Per-Position Breakdown",
            "",
            "| Position | "
            + " | ".join(d["name"] for d in tactical.get("engines", {}).values())
            + " |",
            "|----------|"
            + "|".join("-------" for _ in tactical.get("engines", {}))
            + "|",
        ]
        for i, bk_id in enumerate(position_ids):
            exp = position_expected.get(bk_id, [])
            row = f"| {bk_id} ({','.join(exp)}) |"
            for eid, data in tactical.get("engines", {}).items():
                p = data["positions"][i] if i < len(data["positions"]) else {}
                if repeat_count == 1:
                    status = "PASS" if p.get("pass") else "FAIL"
                    row += f" {status} ({p.get('bestmove','?')}) |"
                else:
                    row += (
                        f" {p.get('passes', 0)}/{p.get('runs', 0)}"
                        f" ({p.get('bestmove','?')}) |"
                    )
            lines.append(row)
        lines += ["", "### Agreement Matrix", ""]
        for key, data in tactical.get("agreement", {}).items():
            lines.append(
                f"- {key}: {data['agree']}/{data['total']} ({data['rate']*100:.1f}%)"
            )
        lines.append("")

    nps = all_results.get("nps_throughput")
    if nps:
        lines += [
            "## Table 2: NPS Throughput (MetalFish MCTS vs Lc0)",
            "",
            "| Engine | Threads | Avg NPS | Speedup |",
            "|--------|---------|---------|---------|",
        ]
        mf = nps.get("engines", {}).get("metalfish-mcts", {}).get("by_threads", {})
        lc = nps.get("engines", {}).get("lc0", {}).get("by_threads", {})
        for tc_str in sorted(mf.keys(), key=int):
            mf_nps = mf[tc_str]["avg_nps"]
            lc_nps = lc.get(tc_str, {}).get("avg_nps", 0)
            ratio = mf_nps / max(1, lc_nps) if lc_nps else 0
            lines.append(f"| MetalFish-MCTS | {tc_str} | {mf_nps:,} | -- |")
            if lc_nps:
                lines.append(f"| Lc0 | {tc_str} | {lc_nps:,} | {ratio:.2f}x |")
        lines.append("")

    scaling = all_results.get("thread_scaling")
    if scaling:
        lines += [
            "## Table 3: Thread Scaling",
            "",
            "| Engine | 1T NPS | 2T NPS | 4T NPS | 8T NPS | 8T Scaling |",
            "|--------|--------|--------|--------|--------|------------|",
        ]
        for eid, data in scaling.get("engines", {}).items():
            bt = data.get("by_threads", {})
            nps_vals = [
                str(bt.get(str(t), {}).get("avg_nps", "N/A")) for t in [1, 2, 4, 8]
            ]
            sc8 = bt.get("8", {}).get("scaling_factor", "N/A")
            lines.append(f"| {data['name']} | {' | '.join(nps_vals)} | {sc8}x |")
        lines.append("")

    tournament = all_results.get("tournament")
    if tournament:
        lines += [
            "## Table 4: Tournament Results",
            "",
            "| Match | W | D | L | Score | Pct |",
            "|-------|---|---|---|-------|-----|",
        ]
        for m in tournament.get("matches", []):
            lines.append(
                f"| {m['name1']} vs {m['name2']} | "
                f"{m['wins']} | {m['draws']} | {m['losses']} | "
                f"{m['score']}/{m['total']} | {m['pct']*100:.1f}% |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper benchmark suite")
    parser.add_argument(
        "--tactical", action="store_true", help="Run BK tactical accuracy"
    )
    parser.add_argument(
        "--nps", action="store_true", help="Run NPS throughput comparison"
    )
    parser.add_argument("--scaling", action="store_true", help="Run thread scaling")
    parser.add_argument("--tournament", action="store_true", help="Run Elo tournament")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument(
        "--threads",
        type=str,
        default="auto",
        help="Thread counts (comma-sep; use auto for Apple performance cores)",
    )
    parser.add_argument(
        "--hash",
        type=str,
        default="auto",
        help="Hash/TT size in MB for engines that support it, or auto",
    )
    parser.add_argument(
        "--movetime", type=int, default=10000, help="Movetime per position (ms)"
    )
    parser.add_argument(
        "--tactical-repeat",
        type=int,
        default=1,
        help="Repeat each tactical position this many times and aggregate runs",
    )
    parser.add_argument(
        "--games", type=int, default=10, help="Games per tournament match"
    )
    parser.add_argument(
        "--tc-movetime", type=int, default=5000, help="Movetime for tournament games"
    )
    parser.add_argument(
        "--max-game-plies",
        type=int,
        default=200,
        help="Maximum plies per tournament game before adjudicating a draw",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="",
        help="Optional comma/space-separated engine IDs to benchmark",
    )
    parser.add_argument(
        "--tactical-fail-under",
        type=str,
        default="",
        help=(
            "Fail tactical runs below a floor. Use SCORE for selected engines "
            "or engine=SCORE pairs, e.g. hybrid=21,mcts=12."
        ),
    )
    parser.add_argument(
        "--positions",
        type=str,
        default="",
        help="Optional comma/space-separated BK IDs for tactical runs, e.g. BK.03,BK.07",
    )
    args = parser.parse_args()

    if args.all:
        args.tactical = args.nps = args.scaling = args.tournament = True
    if not any([args.tactical, args.nps, args.scaling, args.tournament]):
        args.tactical = args.nps = args.scaling = True

    thread_counts = parse_thread_counts(args.threads)
    hash_mb = parse_hash_mb(args.hash)
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results: Dict[str, dict] = {}
    exit_code = 0

    primary_threads = thread_counts[0] if len(thread_counts) == 1 else thread_counts[-1]
    resources = resource_policy(primary_threads, hash_mb)
    engines = detect_engines(threads=primary_threads, hash_mb=hash_mb)
    selected_engine_ids = parse_engine_ids(args.engines)
    try:
        tactical_thresholds = parse_tactical_fail_under(
            args.tactical_fail_under, selected_engine_ids
        )
        tactical_positions = select_bk_positions(args.positions)
    except ValueError as exc:
        parser.error(str(exc))
    if selected_engine_ids is not None:
        selected = set(selected_engine_ids)
        engines = {eid: cfg for eid, cfg in engines.items() if eid in selected}
    print("Detected engines:")
    for eid, cfg in engines.items():
        status = "OK" if cfg.available else "MISSING"
        print(f"  {eid}: {status} ({cfg.path}) options={cfg.uci_options}")
    print(
        f"Resource policy: threads={primary_threads}, hash={hash_mb} MB, "
        f"pure-mcts={resources['metalfish_mcts_threads']}, "
        f"hybrid={resources['hybrid_split']['mcts_threads']} MCTS + "
        f"{resources['hybrid_split']['ab_threads']} AB"
    )

    if args.tactical:
        r = run_tactical(
            engines,
            args.movetime,
            selected_engine_ids,
            max(1, args.tactical_repeat),
            tactical_positions,
        )
        r["resource_policy"] = resources
        all_results["tactical"] = r
        with open(RESULTS_DIR / "paper_tactical.json", "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nSaved: results/paper_tactical.json")
        floor_failures = enforce_tactical_fail_under(r, tactical_thresholds)
        if floor_failures:
            print(
                "ERROR: tactical fail-under guard tripped: "
                + "; ".join(floor_failures),
                file=sys.stderr,
            )
            exit_code = 1

    if args.nps:
        r = run_nps(engines, thread_counts, args.movetime)
        r["resource_policy"] = resources
        all_results["nps_throughput"] = r
        with open(RESULTS_DIR / "paper_nps.json", "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nSaved: results/paper_nps.json")

    if args.scaling:
        r = run_scaling(engines, thread_counts, args.movetime)
        r["resource_policy"] = resources
        all_results["thread_scaling"] = r
        with open(RESULTS_DIR / "paper_scaling.json", "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nSaved: results/paper_scaling.json")

    if args.tournament:
        r = run_tournament(engines, args.games, args.tc_movetime, args.max_game_plies)
        r["resource_policy"] = resources
        all_results["tournament"] = r
        with open(RESULTS_DIR / "paper_tournament.json", "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nSaved: results/paper_tournament.json")

    if all_results:
        summary = generate_summary(all_results)
        summary_path = RESULTS_DIR / "paper_summary.md"
        summary_path.write_text(summary)
        print(f"\nSaved: {summary_path}")
        print("\n" + summary)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
