#!/usr/bin/env python3
"""
MetalFish Comprehensive Elo Tournament

Runs a tournament between multiple chess engines to determine Elo ratings:
- MetalFish-AB (Alpha-Beta search with 'go' command) - Best for tactical positions, ~1.5M NPS
- MetalFish-Hybrid (Hybrid MCTS search with 'mcts' command) - General play
- MetalFish-MCTS (Multi-threaded MCTS with 'mctsmt' command) - Strategic positions, ~700K NPS (4 threads)
- Stockfish at various skill levels (0-20)
- Patricia (aggressive engine, ~3500 Elo)
- Lc0 (Leela Chess Zero - neural network engine)

Uses cutechess-cli for tournament management and calculates Elo ratings.

Usage:
    # Full tournament
    python elo_tournament.py [--games N] [--time TC] [--concurrency N]

    # CI mode - run single match (for GitHub Actions matrix)
    python elo_tournament.py --ci-match --engine1 "MetalFish-AB" --engine2 "Stockfish-L10"

    # CI mode - aggregate results from matrix jobs
    python elo_tournament.py --ci-aggregate --results-dir ./results
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ANSI colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    DIM = "\033[2m"


@dataclass
class EngineConfig:
    """Configuration for a chess engine."""

    name: str
    cmd: str
    proto: str = "uci"
    options: Dict[str, str] = field(default_factory=dict)
    expected_elo: Optional[int] = None  # For reference

    def to_cutechess_args(self) -> List[str]:
        """Convert to cutechess-cli engine arguments."""
        args = [
            "-engine",
            f"cmd={self.cmd}",
            f"name={self.name}",
            f"proto={self.proto}",
        ]
        for key, value in self.options.items():
            args.append(f"option.{key}={value}")
        return args


@dataclass
class GameResult:
    """Result of a single game."""

    white: str
    black: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    reason: str = ""
    moves: int = 0


@dataclass
class MatchResult:
    """Results between two engines."""

    engine1: str
    engine2: str
    wins1: int = 0
    wins2: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.wins1 + self.wins2 + self.draws

    @property
    def score1(self) -> float:
        return self.wins1 + self.draws * 0.5

    @property
    def score2(self) -> float:
        return self.wins2 + self.draws * 0.5


class EloCalculator:
    """Calculate Elo ratings from tournament results."""

    def __init__(self, anchor_engine: str = None, anchor_elo: int = 3000):
        self.anchor_engine = anchor_engine
        self.anchor_elo = anchor_elo
        self.results: Dict[Tuple[str, str], MatchResult] = {}

    def add_result(self, engine1: str, engine2: str, result: str):
        """Add a game result."""
        key = tuple(sorted([engine1, engine2]))
        if key not in self.results:
            self.results[key] = MatchResult(key[0], key[1])

        match = self.results[key]
        if result == "1-0":
            if engine1 == match.engine1:
                match.wins1 += 1
            else:
                match.wins2 += 1
        elif result == "0-1":
            if engine1 == match.engine1:
                match.wins2 += 1
            else:
                match.wins1 += 1
        else:
            match.draws += 1

    def calculate_elo_diff(self, score: float, total: int) -> Optional[float]:
        """Calculate Elo difference from score percentage."""
        if total == 0:
            return None
        pct = score / total
        if pct <= 0 or pct >= 1:
            return None
        return -400 * math.log10(1 / pct - 1)

    def calculate_ratings(self) -> Dict[str, float]:
        """Calculate Elo ratings for all engines using iterative method."""
        # Collect all engines
        engines = set()
        for key in self.results:
            engines.add(key[0])
            engines.add(key[1])

        if not engines:
            return {}

        # Initialize ratings
        ratings = {e: 2000.0 for e in engines}
        if self.anchor_engine and self.anchor_engine in ratings:
            ratings[self.anchor_engine] = float(self.anchor_elo)

        # Iterative rating calculation
        for iteration in range(100):
            max_change = 0
            for engine in engines:
                if engine == self.anchor_engine:
                    continue

                total_expected = 0
                total_actual = 0
                total_games = 0

                for key, match in self.results.items():
                    if engine not in key:
                        continue

                    opponent = key[0] if key[1] == engine else key[1]

                    # Get scores
                    if engine == match.engine1:
                        actual = match.score1
                    else:
                        actual = match.score2

                    # Expected score based on current ratings
                    rating_diff = ratings[opponent] - ratings[engine]
                    expected = match.total / (1 + 10 ** (rating_diff / 400))

                    total_expected += expected
                    total_actual += actual
                    total_games += match.total

                if total_games > 0:
                    # Adjust rating
                    k = 32  # K-factor
                    adjustment = (
                        k * (total_actual - total_expected) / (total_games / 10)
                    )
                    ratings[engine] += adjustment
                    max_change = max(max_change, abs(adjustment))

            if max_change < 0.1:
                break

        return ratings

    def get_confidence_interval(
        self, wins: int, draws: int, losses: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for Elo difference using Wilson score."""
        n = wins + draws + losses
        if n == 0:
            return (0, 0)

        score = (wins + draws * 0.5) / n

        # Wilson score interval
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

        denominator = 1 + z * z / n
        centre = (score + z * z / (2 * n)) / denominator
        spread = (
            z * math.sqrt((score * (1 - score) + z * z / (4 * n)) / n) / denominator
        )

        low = max(0, centre - spread)
        high = min(1, centre + spread)

        # Convert to Elo
        elo_low = -400 * math.log10(1 / low - 1) if 0 < low < 1 else -float("inf")
        elo_high = -400 * math.log10(1 / high - 1) if 0 < high < 1 else float("inf")

        return (elo_low, elo_high)


class Tournament:
    """Manages a chess tournament using cutechess-cli."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.cutechess = (
            base_dir / "reference" / "cutechess" / "build" / "cutechess-cli"
        )
        self.engines: List[EngineConfig] = []
        self.results: List[GameResult] = []
        self.elo_calc = EloCalculator()

    def add_engine(self, config: EngineConfig):
        """Add an engine to the tournament."""
        self.engines.append(config)

    def setup_default_engines(self, stockfish_levels: List[int] = None):
        """Setup default engine configurations."""

        metalfish_path = self.base_dir / "build" / "metalfish"

        # MetalFish with standard Alpha-Beta search (best for tactical positions, ~1.5M NPS)
        self.add_engine(
            EngineConfig(
                name="MetalFish-AB",
                cmd=str(metalfish_path),
                options={"Threads": "1", "Hash": "128"},
                expected_elo=None,  # To be determined
            )
        )

        # MetalFish with Hybrid MCTS search (general play, combining strengths)
        hybrid_wrapper = self.base_dir / "tools" / "metalfish_hybrid_wrapper.sh"
        self._create_hybrid_wrapper(metalfish_path, hybrid_wrapper)
        self.add_engine(
            EngineConfig(
                name="MetalFish-Hybrid",
                cmd=str(hybrid_wrapper),
                options={},
                expected_elo=None,
            )
        )

        # MetalFish with Multi-threaded MCTS (strategic positions, ~700K NPS with 4 threads)
        mctsmt_wrapper = self.base_dir / "tools" / "metalfish_mctsmt_wrapper.sh"
        self._create_mctsmt_wrapper(metalfish_path, mctsmt_wrapper)
        self.add_engine(
            EngineConfig(
                name="MetalFish-MCTS",
                cmd=str(mctsmt_wrapper),
                options={},
                expected_elo=None,
            )
        )

        # Patricia - known aggressive engine (~3500 Elo)
        patricia_path = self.base_dir / "reference" / "Patricia" / "engine" / "patricia"
        if patricia_path.exists():
            self.add_engine(
                EngineConfig(
                    name="Patricia",
                    cmd=str(patricia_path),
                    options={"Threads": "1", "Hash": "128"},
                    expected_elo=3500,
                )
            )
            # Set Patricia as anchor for Elo calculation
            self.elo_calc.anchor_engine = "Patricia"
            self.elo_calc.anchor_elo = 3500

        # Lc0 (Leela Chess Zero) - neural network engine
        lc0_path = self.base_dir / "reference" / "lc0" / "build" / "release" / "lc0"
        if lc0_path.exists():
            self.add_engine(
                EngineConfig(
                    name="Lc0",
                    cmd=str(lc0_path),
                    options={"Threads": "1"},
                    expected_elo=3600,  # Depends on network
                )
            )

        # Stockfish at various skill levels
        stockfish_path = self.base_dir / "reference" / "stockfish" / "src" / "stockfish"
        if stockfish_path.exists():
            if stockfish_levels is None:
                # Default: test against multiple levels
                stockfish_levels = [1, 5, 10, 15, 20]

            # Approximate Elo for each skill level (based on Stockfish docs)
            skill_elo_map = {
                0: 1350,
                1: 1500,
                2: 1600,
                3: 1700,
                4: 1800,
                5: 1900,
                6: 2000,
                7: 2100,
                8: 2200,
                9: 2300,
                10: 2400,
                11: 2500,
                12: 2600,
                13: 2700,
                14: 2800,
                15: 2900,
                16: 3000,
                17: 3100,
                18: 3200,
                19: 3300,
                20: 3600,  # Full strength
            }

            for level in stockfish_levels:
                name = f"Stockfish-L{level}" if level < 20 else "Stockfish-Full"
                options = {"Threads": "1", "Hash": "128"}
                if level < 20:
                    options["Skill Level"] = str(level)

                self.add_engine(
                    EngineConfig(
                        name=name,
                        cmd=str(stockfish_path),
                        options=options,
                        expected_elo=skill_elo_map.get(level, 3000),
                    )
                )

    def _create_hybrid_wrapper(self, metalfish_path: Path, wrapper_path: Path):
        """Create a wrapper script that uses 'mcts' command (hybrid search) instead of 'go'."""
        wrapper_content = f"""#!/bin/bash
# MetalFish Hybrid wrapper - intercepts 'go' and runs 'mcts' (hybrid search) instead

ENGINE="{metalfish_path}"

# Read UCI commands and transform 'go' to 'mcts'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'mcts' for hybrid search
        echo "mcts ${{line#go}}"
    else
        echo "$line"
    fi
done | "$ENGINE"
"""
        with open(wrapper_path, "w") as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)

    def _create_mctsmt_wrapper(self, metalfish_path: Path, wrapper_path: Path):
        """Create a wrapper script that uses 'mctsmt' command (multi-threaded MCTS) instead of 'go'."""
        wrapper_content = f"""#!/bin/bash
# MetalFish MCTS-MT wrapper - intercepts 'go' and runs 'mctsmt' (multi-threaded MCTS) instead

ENGINE="{metalfish_path}"

# Read UCI commands and transform 'go' to 'mctsmt threads=4'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'mctsmt threads=4' for multi-threaded MCTS
        echo "mctsmt threads=4 ${{line#go}}"
    else
        echo "$line"
    fi
done | "$ENGINE"
"""
        with open(wrapper_path, "w") as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)

    def _create_mcts_wrapper(self, metalfish_path: Path, wrapper_path: Path):
        """Create a wrapper script that uses 'mcts' command instead of 'go'."""
        wrapper_content = f"""#!/bin/bash
# MetalFish MCTS wrapper - intercepts 'go' and runs 'mcts' instead

ENGINE="{metalfish_path}"

# Read UCI commands and transform 'go' to 'mcts'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'mcts'
        echo "mcts ${{line#go}}"
    else
        echo "$line"
    fi
done | "$ENGINE"
"""
        with open(wrapper_path, "w") as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)

    def run_match(
        self,
        engine1: EngineConfig,
        engine2: EngineConfig,
        games: int,
        time_control: str,
        concurrency: int = 1,
    ) -> List[GameResult]:
        """Run a match between two engines."""

        pgn_file = tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False)
        pgn_file.close()

        cmd = [str(self.cutechess)]
        cmd.extend(engine1.to_cutechess_args())
        cmd.extend(engine2.to_cutechess_args())
        cmd.extend(
            [
                "-each",
                f"tc={time_control}",
                "-rounds",
                str(games // 2),  # Each round is 2 games (alternating colors)
                "-games",
                "2",  # 2 games per round (one as white, one as black)
                "-pgnout",
                pgn_file.name,
                "-concurrency",
                str(concurrency),
                "-recover",
                "-repeat",
            ]
        )

        print(
            f"  Running: {engine1.name} vs {engine2.name} ({games} games)...",
            end=" ",
            flush=True,
        )

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            print(f"{Colors.RED}TIMEOUT{Colors.RESET}")
            return []

        # Parse results from output
        game_results = []

        # Find ALL score lines and use the LAST one (final result)
        score_matches = re.findall(
            rf"Score of {re.escape(engine1.name)} vs {re.escape(engine2.name)}: (\d+) - (\d+) - (\d+)",
            output,
        )

        if score_matches:
            # Use the last match (final score)
            last_match = score_matches[-1]
            wins1 = int(last_match[0])
            wins2 = int(last_match[1])
            draws = int(last_match[2])

            # Create game results
            for _ in range(wins1):
                game_results.append(GameResult(engine1.name, engine2.name, "1-0"))
            for _ in range(wins2):
                game_results.append(GameResult(engine1.name, engine2.name, "0-1"))
            for _ in range(draws):
                game_results.append(GameResult(engine1.name, engine2.name, "1/2-1/2"))

            total = wins1 + wins2 + draws
            score_pct = (wins1 + draws * 0.5) / total * 100 if total > 0 else 50
            print(
                f"{Colors.GREEN}Done{Colors.RESET} ({wins1}-{wins2}-{draws}, {score_pct:.1f}%)"
            )
        else:
            print(f"{Colors.YELLOW}No results parsed{Colors.RESET}")
            # Try to parse individual game results from PGN
            try:
                with open(pgn_file.name, "r") as f:
                    pgn_content = f.read()

                # Count results in PGN
                wins1 = pgn_content.count('[Result "1-0"]')
                wins2 = pgn_content.count('[Result "0-1"]')
                draws = pgn_content.count('[Result "1/2-1/2"]')

                for _ in range(wins1):
                    game_results.append(GameResult(engine1.name, engine2.name, "1-0"))
                for _ in range(wins2):
                    game_results.append(GameResult(engine1.name, engine2.name, "0-1"))
                for _ in range(draws):
                    game_results.append(
                        GameResult(engine1.name, engine2.name, "1/2-1/2")
                    )
            except:
                pass

        # Clean up
        try:
            os.unlink(pgn_file.name)
        except:
            pass

        return game_results

    def run_round_robin(
        self, games_per_pair: int, time_control: str, concurrency: int = 1
    ) -> Dict[str, float]:
        """Run a round-robin tournament between all engines."""

        print(f"\n{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}  MetalFish Elo Tournament{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        print(f"\n  Engines: {len(self.engines)}")
        print(f"  Games per pair: {games_per_pair}")
        print(f"  Time control: {time_control}")
        print(
            f"  Total games: {len(self.engines) * (len(self.engines) - 1) * games_per_pair // 2}"
        )
        print(f"\n{Colors.DIM}  Starting tournament...{Colors.RESET}\n")

        # Run matches between all pairs
        for i, engine1 in enumerate(self.engines):
            for engine2 in self.engines[i + 1 :]:
                results = self.run_match(
                    engine1, engine2, games_per_pair, time_control, concurrency
                )

                # Record results
                for result in results:
                    self.results.append(result)
                    self.elo_calc.add_result(result.white, result.black, result.result)

        # Calculate and return Elo ratings
        return self.elo_calc.calculate_ratings()

    def print_results(self, ratings: Dict[str, float]):
        """Print tournament results and Elo ratings."""

        print(f"\n{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}  TOURNAMENT RESULTS{Colors.RESET}")
        print(f"{'═' * 70}")

        # Sort engines by rating
        sorted_engines = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  {'Rank':<6} {'Engine':<25} {'Elo':>8} {'Expected':>10}")
        print(f"  {'-'*6} {'-'*25} {'-'*8} {'-'*10}")

        for rank, (engine, elo) in enumerate(sorted_engines, 1):
            # Find expected Elo
            expected = "N/A"
            for e in self.engines:
                if e.name == engine and e.expected_elo:
                    expected = str(e.expected_elo)
                    break

            # Color based on performance vs expected
            color = Colors.RESET
            if expected != "N/A":
                diff = elo - int(expected)
                if diff > 50:
                    color = Colors.GREEN
                elif diff < -50:
                    color = Colors.RED

            print(
                f"  {rank:<6} {color}{engine:<25}{Colors.RESET} {elo:>8.0f} {expected:>10}"
            )

        # Print head-to-head results
        print(f"\n{Colors.BOLD}  HEAD-TO-HEAD RESULTS{Colors.RESET}")
        print(f"  {'-'*60}")

        for key, match in sorted(self.elo_calc.results.items()):
            score1 = match.score1
            score2 = match.score2
            elo_diff = self.elo_calc.calculate_elo_diff(score1, match.total)

            elo_str = f"{elo_diff:+.0f}" if elo_diff else "N/A"

            print(f"  {match.engine1:<20} vs {match.engine2:<20}")
            print(
                f"    Score: {match.wins1}-{match.draws}-{match.wins2} "
                f"({score1:.1f}/{match.total}) Elo diff: {elo_str}"
            )

        print(f"\n{'═' * 70}\n")

        # Save results to JSON
        self._save_results(ratings)

    def _save_results(self, ratings: Dict[str, float]):
        """Save tournament results to JSON file."""
        results_file = self.base_dir / "tools" / "tournament_results.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "ratings": ratings,
            "matches": [
                {
                    "engine1": m.engine1,
                    "engine2": m.engine2,
                    "wins1": m.wins1,
                    "wins2": m.wins2,
                    "draws": m.draws,
                }
                for m in self.elo_calc.results.values()
            ],
            "engines": [
                {"name": e.name, "expected_elo": e.expected_elo} for e in self.engines
            ],
        }

        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  Results saved to: {results_file}")


# =============================================================================
# CI Mode Functions
# =============================================================================


def get_engine_configs(
    base_dir: Path, stockfish_levels: List[int] = None
) -> Dict[str, EngineConfig]:
    """Get all available engine configurations for CI mode."""
    configs = {}

    metalfish_path = base_dir / "build" / "metalfish"

    # MetalFish with standard Alpha-Beta search (best for tactical positions, ~1.5M NPS)
    configs["MetalFish-AB"] = EngineConfig(
        name="MetalFish-AB",
        cmd=str(metalfish_path),
        options={"Threads": "1", "Hash": "128"},
        expected_elo=None,
    )

    # MetalFish with Hybrid MCTS search (general play)
    hybrid_wrapper = base_dir / "tools" / "metalfish_hybrid_wrapper.sh"
    configs["MetalFish-Hybrid"] = EngineConfig(
        name="MetalFish-Hybrid", cmd=str(hybrid_wrapper), options={}, expected_elo=None
    )

    # MetalFish with Multi-threaded MCTS (strategic positions, ~700K NPS with 4 threads)
    mctsmt_wrapper = base_dir / "tools" / "metalfish_mctsmt_wrapper.sh"
    configs["MetalFish-MCTS"] = EngineConfig(
        name="MetalFish-MCTS", cmd=str(mctsmt_wrapper), options={}, expected_elo=None
    )

    # Patricia - aggressive engine (~3500 Elo)
    patricia_path = base_dir / "reference" / "Patricia" / "engine" / "patricia"
    if patricia_path.exists():
        configs["Patricia"] = EngineConfig(
            name="Patricia",
            cmd=str(patricia_path),
            options={"Threads": "1", "Hash": "128"},
            expected_elo=3500,
        )

    # Lc0 (Leela Chess Zero) - neural network engine
    lc0_path = base_dir / "reference" / "lc0" / "build" / "release" / "lc0"
    if lc0_path.exists():
        configs["Lc0"] = EngineConfig(
            name="Lc0",
            cmd=str(lc0_path),
            options={"Threads": "1"},
            expected_elo=3600,
        )

    # Stockfish at various levels
    stockfish_path = base_dir / "reference" / "stockfish" / "src" / "stockfish"
    if stockfish_path.exists():
        if stockfish_levels is None:
            stockfish_levels = [1, 5, 10, 15, 20]

        skill_elo_map = {
            0: 1350,
            1: 1500,
            2: 1600,
            3: 1700,
            4: 1800,
            5: 1900,
            6: 2000,
            7: 2100,
            8: 2200,
            9: 2300,
            10: 2400,
            11: 2500,
            12: 2600,
            13: 2700,
            14: 2800,
            15: 2900,
            16: 3000,
            17: 3100,
            18: 3200,
            19: 3300,
            20: 3600,
        }

        for level in stockfish_levels:
            name = f"Stockfish-L{level}" if level < 20 else "Stockfish-Full"
            options = {"Threads": "1", "Hash": "128"}
            if level < 20:
                options["Skill Level"] = str(level)

            configs[name] = EngineConfig(
                name=name,
                cmd=str(stockfish_path),
                options=options,
                expected_elo=skill_elo_map.get(level, 3000),
            )

    return configs


def list_engine_pairs(engines: List[str]) -> List[Tuple[str, str]]:
    """Generate all unique pairs of engines for matrix jobs."""
    pairs = []
    for i, e1 in enumerate(engines):
        for e2 in engines[i + 1 :]:
            pairs.append((e1, e2))
    return pairs


def run_ci_match(
    base_dir: Path,
    engine1_name: str,
    engine2_name: str,
    games: int,
    time_control: str,
    output_file: Path,
) -> Dict[str, Any]:
    """Run a single match between two engines and output JSON results."""

    configs = get_engine_configs(base_dir)

    if engine1_name not in configs:
        raise ValueError(f"Unknown engine: {engine1_name}")
    if engine2_name not in configs:
        raise ValueError(f"Unknown engine: {engine2_name}")

    engine1 = configs[engine1_name]
    engine2 = configs[engine2_name]

    metalfish_path = base_dir / "build" / "metalfish"

    # Create wrapper scripts if needed
    if "MetalFish-Hybrid" in [engine1_name, engine2_name]:
        hybrid_wrapper = base_dir / "tools" / "metalfish_hybrid_wrapper.sh"
        _create_hybrid_wrapper_file(metalfish_path, hybrid_wrapper)

    if "MetalFish-MCTS" in [engine1_name, engine2_name]:
        mctsmt_wrapper = base_dir / "tools" / "metalfish_mctsmt_wrapper.sh"
        _create_mctsmt_wrapper_file(metalfish_path, mctsmt_wrapper)

    cutechess = base_dir / "reference" / "cutechess" / "build" / "cutechess-cli"

    if not cutechess.exists():
        raise FileNotFoundError(f"cutechess-cli not found at {cutechess}")

    # Verify engine binaries exist
    print(f"Verifying engine binaries...")
    print(f"  Engine 1: {engine1.cmd}")
    if not Path(engine1.cmd).exists():
        raise FileNotFoundError(f"Engine binary not found: {engine1.cmd}")
    print(f"    ✓ Found")

    print(f"  Engine 2: {engine2.cmd}")
    if not Path(engine2.cmd).exists():
        raise FileNotFoundError(f"Engine binary not found: {engine2.cmd}")
    print(f"    ✓ Found")

    # Run match
    pgn_file = tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False)
    pgn_file.close()

    cmd = [str(cutechess)]
    cmd.extend(engine1.to_cutechess_args())
    cmd.extend(engine2.to_cutechess_args())
    cmd.extend(
        [
            "-each",
            f"tc={time_control}",
            "-rounds",
            str(games // 2),
            "-games",
            "2",
            "-pgnout",
            pgn_file.name,
            "-concurrency",
            "1",
            "-recover",
            "-repeat",
        ]
    )

    print(f"Running: {engine1_name} vs {engine2_name} ({games} games)...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        output = result.stdout + result.stderr

        # Always print cutechess output for debugging
        print(f"\n=== cutechess-cli output ===")
        print(output[:5000] if len(output) > 5000 else output)  # Limit output length
        print(f"=== end cutechess-cli output ===\n")

        if result.returncode != 0:
            print(
                f"Warning: cutechess-cli returned non-zero exit code: {result.returncode}"
            )

    except subprocess.TimeoutExpired:
        print(f"ERROR: Match timed out after 7200 seconds")
        return {
            "engine1": engine1_name,
            "engine2": engine2_name,
            "wins1": 0,
            "wins2": 0,
            "draws": 0,
            "error": "timeout",
        }

    # Parse results
    wins1, wins2, draws = 0, 0, 0

    # Find ALL score lines and use the LAST one (final result)
    score_matches = re.findall(
        rf"Score of {re.escape(engine1_name)} vs {re.escape(engine2_name)}: (\d+) - (\d+) - (\d+)",
        output,
    )

    if score_matches:
        # Use the last match (final score)
        last_match = score_matches[-1]
        wins1 = int(last_match[0])
        wins2 = int(last_match[1])
        draws = int(last_match[2])
    else:
        print(f"Warning: Could not parse score from cutechess output")
        # Try PGN parsing
        try:
            with open(pgn_file.name, "r") as f:
                pgn_content = f.read()
            print(f"PGN file contents ({len(pgn_content)} bytes):")
            print(pgn_content[:2000] if len(pgn_content) > 2000 else pgn_content)

            wins1 = pgn_content.count('[Result "1-0"]')
            wins2 = pgn_content.count('[Result "0-1"]')
            draws = pgn_content.count('[Result "1/2-1/2"]')
        except Exception as e:
            print(f"Error reading PGN file: {e}")
        except:
            pass

    # Cleanup
    try:
        os.unlink(pgn_file.name)
    except:
        pass

    match_result = {
        "engine1": engine1_name,
        "engine2": engine2_name,
        "wins1": wins1,
        "wins2": wins2,
        "draws": draws,
        "total": wins1 + wins2 + draws,
        "timestamp": datetime.now().isoformat(),
    }

    # Save to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(match_result, f, indent=2)

    print(f"Match complete: {wins1}-{wins2}-{draws}")
    print(f"Results saved to: {output_file}")

    return match_result


def _create_hybrid_wrapper_file(metalfish_path: Path, wrapper_path: Path):
    """Create a wrapper script that uses 'mcts' command (hybrid search) instead of 'go'."""
    wrapper_content = f"""#!/bin/bash
# MetalFish Hybrid wrapper - intercepts 'go' and runs 'mcts' (hybrid search) instead

ENGINE="{metalfish_path}"

# Read UCI commands and transform 'go' to 'mcts'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'mcts' for hybrid search
        echo "mcts ${{line#go}}"
    else
        echo "$line"
    fi
done | "$ENGINE"
"""
    with open(wrapper_path, "w") as f:
        f.write(wrapper_content)
    os.chmod(wrapper_path, 0o755)


def _create_mctsmt_wrapper_file(metalfish_path: Path, wrapper_path: Path):
    """Create a wrapper script that uses 'mctsmt' command (multi-threaded MCTS) instead of 'go'."""
    wrapper_content = f"""#!/bin/bash
# MetalFish MCTS-MT wrapper - intercepts 'go' and runs 'mctsmt' (multi-threaded MCTS) instead

ENGINE="{metalfish_path}"

# Read UCI commands and transform 'go' to 'mctsmt threads=4'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'mctsmt threads=4' for multi-threaded MCTS
        echo "mctsmt threads=4 ${{line#go}}"
    else
        echo "$line"
    fi
done | "$ENGINE"
"""
    with open(wrapper_path, "w") as f:
        f.write(wrapper_content)
    os.chmod(wrapper_path, 0o755)


def _create_mcts_wrapper_file(metalfish_path: Path, wrapper_path: Path):
    """Create a wrapper script that uses 'mcts' command instead of 'go'."""
    wrapper_content = f"""#!/bin/bash
# MetalFish MCTS wrapper - intercepts 'go' and runs 'mcts' instead

ENGINE="{metalfish_path}"

# Read UCI commands and transform 'go' to 'mcts'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'mcts'
        echo "mcts ${{line#go}}"
    else
        echo "$line"
    fi
done | "$ENGINE"
"""
    with open(wrapper_path, "w") as f:
        f.write(wrapper_content)
    os.chmod(wrapper_path, 0o755)


def aggregate_ci_results(results_dir: Path, output_file: Path = None) -> Dict[str, Any]:
    """Aggregate results from multiple CI match jobs."""

    # Find all result JSON files
    result_files = list(results_dir.glob("*.json"))

    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    # Aggregate all matches
    all_matches = []
    engines = set()

    for result_file in result_files:
        with open(result_file, "r") as f:
            match = json.load(f)

        if "error" not in match:
            all_matches.append(match)
            engines.add(match["engine1"])
            engines.add(match["engine2"])

    # Calculate Elo ratings
    elo_calc = EloCalculator(anchor_engine="Patricia", anchor_elo=3500)

    for match in all_matches:
        # Add individual game results
        for _ in range(match["wins1"]):
            elo_calc.add_result(match["engine1"], match["engine2"], "1-0")
        for _ in range(match["wins2"]):
            elo_calc.add_result(match["engine1"], match["engine2"], "0-1")
        for _ in range(match["draws"]):
            elo_calc.add_result(match["engine1"], match["engine2"], "1/2-1/2")

    ratings = elo_calc.calculate_ratings()

    # Sort by rating
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    # Build summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_games": sum(m["total"] for m in all_matches),
        "total_matches": len(all_matches),
        "ratings": {name: round(elo, 0) for name, elo in sorted_ratings},
        "rankings": [
            {"rank": i + 1, "engine": name, "elo": round(elo, 0)}
            for i, (name, elo) in enumerate(sorted_ratings)
        ],
        "matches": all_matches,
    }

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {output_file}")

    return summary


def generate_pr_comment(summary: Dict[str, Any]) -> str:
    """Generate a markdown comment for PR."""

    lines = [
        "## 🏆 MetalFish Elo Tournament Results",
        "",
        f"**Total Games:** {summary['total_games']} | **Matches:** {summary['total_matches']}",
        "",
        "### Engine Rankings",
        "",
        "| Rank | Engine | Elo |",
        "|------|--------|-----|",
    ]

    for r in summary["rankings"]:
        medal = ""
        if r["rank"] == 1:
            medal = "🥇 "
        elif r["rank"] == 2:
            medal = "🥈 "
        elif r["rank"] == 3:
            medal = "🥉 "

        lines.append(f"| {r['rank']} | {medal}{r['engine']} | {r['elo']:.0f} |")

    lines.extend(
        [
            "",
            "### Match Results",
            "",
            "<details>",
            "<summary>Click to expand match details</summary>",
            "",
            "| Match | Result | Score |",
            "|-------|--------|-------|",
        ]
    )

    for match in summary["matches"]:
        total = match["total"]
        score_pct = (
            (match["wins1"] + match["draws"] * 0.5) / total * 100 if total > 0 else 50
        )
        lines.append(
            f"| {match['engine1']} vs {match['engine2']} | "
            f"{match['wins1']}-{match['draws']}-{match['wins2']} | "
            f"{score_pct:.1f}% |"
        )

    lines.extend(["", "</details>", "", f"*Generated at {summary['timestamp']}*"])

    return "\n".join(lines)


def print_ci_engines(base_dir: Path, stockfish_levels: List[int] = None):
    """Print available engines as JSON for CI matrix generation."""
    configs = get_engine_configs(base_dir, stockfish_levels)
    engines = list(configs.keys())
    pairs = list_engine_pairs(engines)

    output = {
        "engines": engines,
        "pairs": [{"engine1": p[0], "engine2": p[1]} for p in pairs],
        "matrix": [f"{p[0]}__vs__{p[1]}" for p in pairs],
    }

    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run a chess engine Elo tournament")

    # Standard tournament arguments
    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=20,
        help="Number of games per engine pair (default: 20)",
    )
    parser.add_argument(
        "--time",
        "-t",
        type=str,
        default="10+0.1",
        help="Time control (default: 10+0.1 = 10 sec + 0.1 sec increment)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=1,
        help="Number of concurrent games (default: 1)",
    )
    parser.add_argument(
        "--stockfish-levels",
        "-s",
        type=str,
        default="1,5,10,15,20",
        help="Comma-separated Stockfish skill levels to test (default: 1,5,10,15,20)",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Quick test with fewer games and faster time control",
    )
    parser.add_argument(
        "--metalfish-only",
        "-m",
        action="store_true",
        help="Only test MetalFish variants (AB vs Hybrid vs MCTS)",
    )

    # CI mode arguments
    parser.add_argument(
        "--ci-match",
        action="store_true",
        help="CI mode: run a single match between two engines",
    )
    parser.add_argument("--engine1", type=str, help="First engine name for CI match")
    parser.add_argument("--engine2", type=str, help="Second engine name for CI match")
    parser.add_argument(
        "--output", "-o", type=str, help="Output file for CI match results (JSON)"
    )
    parser.add_argument(
        "--ci-aggregate",
        action="store_true",
        help="CI mode: aggregate results from multiple match jobs",
    )
    parser.add_argument(
        "--results-dir", type=str, help="Directory containing match result JSON files"
    )
    parser.add_argument(
        "--ci-list-engines",
        action="store_true",
        help="CI mode: list available engines and pairs as JSON",
    )
    parser.add_argument(
        "--ci-comment",
        action="store_true",
        help="Output PR comment markdown instead of JSON",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Override base directory for engine paths (default: auto-detect from script location)",
    )

    args = parser.parse_args()

    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path(__file__).parent.parent

    print(f"Base directory: {base_dir.absolute()}", file=sys.stderr)

    # Parse Stockfish levels
    stockfish_levels = [int(x) for x in args.stockfish_levels.split(",")]

    # CI mode: list engines
    if args.ci_list_engines:
        print_ci_engines(base_dir, stockfish_levels)
        return

    # CI mode: run single match
    if args.ci_match:
        if not args.engine1 or not args.engine2:
            print(
                "Error: --engine1 and --engine2 required for --ci-match",
                file=sys.stderr,
            )
            sys.exit(1)

        output_file = (
            Path(args.output)
            if args.output
            else Path(f"results/{args.engine1}_vs_{args.engine2}.json")
        )

        if args.quick:
            args.games = 10
            args.time = "5+0.05"

        try:
            run_ci_match(
                base_dir, args.engine1, args.engine2, args.games, args.time, output_file
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # CI mode: aggregate results
    if args.ci_aggregate:
        if not args.results_dir:
            print("Error: --results-dir required for --ci-aggregate", file=sys.stderr)
            sys.exit(1)

        results_dir = Path(args.results_dir)
        output_file = Path(args.output) if args.output else results_dir / "summary.json"

        try:
            summary = aggregate_ci_results(results_dir, output_file)

            if args.ci_comment:
                comment = generate_pr_comment(summary)
                print(comment)
            else:
                print(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Standard tournament mode
    # Quick mode
    if args.quick:
        args.games = 10
        args.time = "5+0.05"

    # Setup tournament
    tournament = Tournament(base_dir)

    # Verify cutechess exists
    if not tournament.cutechess.exists():
        print(
            f"{Colors.RED}Error: cutechess-cli not found at {tournament.cutechess}{Colors.RESET}"
        )
        sys.exit(1)

    if args.metalfish_only:
        # Test all MetalFish variants against each other
        metalfish_path = base_dir / "build" / "metalfish"

        # Alpha-Beta (standard 'go' command)
        tournament.add_engine(
            EngineConfig(
                name="MetalFish-AB",
                cmd=str(metalfish_path),
                options={"Threads": "1", "Hash": "128"},
            )
        )

        # Hybrid MCTS ('mcts' command)
        hybrid_wrapper = base_dir / "tools" / "metalfish_hybrid_wrapper.sh"
        tournament._create_hybrid_wrapper(metalfish_path, hybrid_wrapper)
        tournament.add_engine(
            EngineConfig(name="MetalFish-Hybrid", cmd=str(hybrid_wrapper), options={})
        )

        # Multi-threaded MCTS ('mctsmt' command)
        mctsmt_wrapper = base_dir / "tools" / "metalfish_mctsmt_wrapper.sh"
        tournament._create_mctsmt_wrapper(metalfish_path, mctsmt_wrapper)
        tournament.add_engine(
            EngineConfig(name="MetalFish-MCTS", cmd=str(mctsmt_wrapper), options={})
        )
    else:
        tournament.setup_default_engines(stockfish_levels)

    # Run tournament
    ratings = tournament.run_round_robin(
        games_per_pair=args.games, time_control=args.time, concurrency=args.concurrency
    )

    # Print results
    tournament.print_results(ratings)


if __name__ == "__main__":
    main()
