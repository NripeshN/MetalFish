#!/usr/bin/env python3
"""
MetalFish Comprehensive Elo Tournament

Runs a tournament between multiple chess engines to determine Elo ratings:
- MetalFish-AB (Alpha-Beta search with 'go' command) - Full Stockfish search with NNUE
- MetalFish-MCTS (GPU MCTS with 'mctsmt' command) - Pure GPU-accelerated MCTS  
- MetalFish-Hybrid (Parallel MCTS+AB with 'parallel_hybrid' command) - Best of both worlds
- Stockfish at various skill levels (0-20)
- Patricia (aggressive engine, ~3500 Elo)
- Berserk (strong NNUE engine, ~3550 Elo)
- Lc0 (Leela Chess Zero - neural network engine)

Engine configurations are loaded from engines_config.json.

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


# Default engines configuration (used if engines_config.json is not found)
DEFAULT_ENGINES_CONFIG = {
    "engines": {
        "MetalFish-AB": {
            "description": "MetalFish with Alpha-Beta search (full Stockfish with NNUE)",
            "expected_elo": None,
            "options": {"Threads": "1", "Hash": "128"}
        },
        "MetalFish-MCTS": {
            "description": "MetalFish with Pure GPU MCTS",
            "expected_elo": None,
            "options": {}
        },
        "MetalFish-Hybrid": {
            "description": "MetalFish with Parallel MCTS+AB hybrid search",
            "expected_elo": None,
            "options": {}
        },
        "Patricia": {
            "description": "Aggressive chess engine by Adam Kulju",
            "expected_elo": 3460,
            "options": {"Threads": "1", "Hash": "128"},
            "path": "reference/Patricia/engine/patricia",
            "anchor": True,
            "anchor_elo": 3460
        },
        "Berserk": {
            "description": "Strong NNUE engine by Jay Honnold",
            "expected_elo": 3617,
            "options": {"Threads": "1", "Hash": "128"},
            "path": "reference/berserk/src/berserk",
            "anchor": True,
            "anchor_elo": 3617
        },
        "Lc0": {
            "description": "Leela Chess Zero - neural network engine",
            "expected_elo": 3600,
            "options": {"Threads": "1"},
            "path": "reference/lc0/build/release/lc0",
            "network_path": "reference/lc0/build/release/network.pb.gz"
        }
    },
    "stockfish": {
        "path": "reference/stockfish/src/stockfish",
        "default_levels": [1, 5, 10, 15, 20],
        "options": {"Threads": "1", "Hash": "128"},
        "skill_elo_map": {
            "0": 1350, "1": 1500, "2": 1600, "3": 1700, "4": 1800,
            "5": 1900, "6": 2000, "7": 2100, "8": 2200, "9": 2300,
            "10": 2400, "11": 2500, "12": 2600, "13": 2700, "14": 2800,
            "15": 2900, "16": 3000, "17": 3100, "18": 3200, "19": 3300,
            "20": 3600
        }
    },
    "tournament_defaults": {
        "games_per_pair": 20,
        "time_control": "10+0.1",
        "concurrency": 1
    }
}


def load_engines_config(base_dir: Path) -> Dict[str, Any]:
    """Load engines configuration from JSON file or use defaults."""
    config_path = base_dir / "tools" / "engines_config.json"
    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"{Colors.DIM}  Loaded engine config from: {config_path}{Colors.RESET}", file=sys.stderr)
            return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"{Colors.YELLOW}  Warning: Could not load {config_path}: {e}{Colors.RESET}", file=sys.stderr)
            print(f"{Colors.DIM}  Using default configuration{Colors.RESET}", file=sys.stderr)
    
    return DEFAULT_ENGINES_CONFIG


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
    pgn: str = ""  # Full PGN of the game


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
        """Setup default engine configurations from engines_config.json."""
        
        # Load configuration
        config = load_engines_config(self.base_dir)
        engines_config = config.get("engines", {})
        stockfish_config = config.get("stockfish", {})

        metalfish_path = self.base_dir / "build" / "metalfish"

        # MetalFish with standard Alpha-Beta search (full Stockfish with NNUE)
        ab_config = engines_config.get("MetalFish-AB", {})
        self.add_engine(
            EngineConfig(
                name="MetalFish-AB",
                cmd=str(metalfish_path),
                options=ab_config.get("options", {"Threads": "1", "Hash": "128"}),
                expected_elo=ab_config.get("expected_elo"),
            )
        )

        # MetalFish with Pure GPU MCTS (uses 'mctsmt' command)
        mcts_wrapper = self.base_dir / "tools" / "metalfish_mcts_wrapper.sh"
        self._create_mcts_wrapper(metalfish_path, mcts_wrapper)
        mcts_config = engines_config.get("MetalFish-MCTS", {})
        self.add_engine(
            EngineConfig(
                name="MetalFish-MCTS",
                cmd=str(mcts_wrapper),
                options=mcts_config.get("options", {}),
                expected_elo=mcts_config.get("expected_elo"),
            )
        )

        # MetalFish with Parallel Hybrid search (MCTS + AB in parallel)
        hybrid_wrapper = self.base_dir / "tools" / "metalfish_hybrid_wrapper.sh"
        self._create_hybrid_wrapper(metalfish_path, hybrid_wrapper)
        hybrid_config = engines_config.get("MetalFish-Hybrid", {})
        self.add_engine(
            EngineConfig(
                name="MetalFish-Hybrid",
                cmd=str(hybrid_wrapper),
                options=hybrid_config.get("options", {}),
                expected_elo=hybrid_config.get("expected_elo"),
            )
        )

        # Add external engines from config
        anchor_set = False
        for engine_name, engine_cfg in engines_config.items():
            # Skip MetalFish variants (already added above)
            if engine_name.startswith("MetalFish"):
                continue
            
            # Get engine path from config or use default
            engine_path_str = engine_cfg.get("path", "")
            if not engine_path_str:
                continue
                
            engine_path = self.base_dir / engine_path_str
            
            # Special handling for Lc0 (needs network file)
            if engine_name == "Lc0":
                network_path_str = engine_cfg.get("network_path", "")
                if network_path_str:
                    network_path = self.base_dir / network_path_str
                    if not (engine_path.exists() and network_path.exists()):
                        continue
                    options = engine_cfg.get("options", {}).copy()
                    options["WeightsFile"] = str(network_path)
                    self.add_engine(
                        EngineConfig(
                            name=engine_name,
                            cmd=str(engine_path),
                            options=options,
                            expected_elo=engine_cfg.get("expected_elo"),
                        )
                    )
                continue
            
            # Standard engine
            if engine_path.exists():
                self.add_engine(
                    EngineConfig(
                        name=engine_name,
                        cmd=str(engine_path),
                        options=engine_cfg.get("options", {}),
                        expected_elo=engine_cfg.get("expected_elo"),
                    )
                )
                # Set anchor if configured and not already set
                if engine_cfg.get("anchor") and not anchor_set:
                    self.elo_calc.anchor_engine = engine_name
                    self.elo_calc.anchor_elo = engine_cfg.get("anchor_elo", 3000)
                    anchor_set = True

        # Stockfish at various skill levels
        stockfish_path_str = stockfish_config.get("path", "reference/stockfish/src/stockfish")
        stockfish_path = self.base_dir / stockfish_path_str
        
        if stockfish_path.exists():
            if stockfish_levels is None:
                stockfish_levels = stockfish_config.get("default_levels", [1, 5, 10, 15, 20])

            # Get Elo map from config
            skill_elo_map = {int(k): v for k, v in stockfish_config.get("skill_elo_map", {}).items()}
            default_options = stockfish_config.get("options", {"Threads": "1", "Hash": "128"})

            for level in stockfish_levels:
                name = f"Stockfish-L{level}" if level < 20 else "Stockfish-Full"
                options = default_options.copy()
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
        """Create a wrapper script that uses 'parallel_hybrid' command for parallel MCTS+AB."""
        wrapper_content = f"""#!/bin/bash
# MetalFish Hybrid wrapper - intercepts 'go' and runs 'parallel_hybrid' (parallel MCTS+AB) instead

ENGINE="{metalfish_path}"

# Read UCI commands and transform 'go' to 'parallel_hybrid'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'parallel_hybrid' for parallel hybrid search
        echo "parallel_hybrid ${{line#go}}"
    else
        echo "$line"
    fi
done | "$ENGINE"
"""
        with open(wrapper_path, "w") as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)

    def _create_mcts_wrapper(self, metalfish_path: Path, wrapper_path: Path):
        """Create a wrapper script that uses 'mctsmt' command for pure GPU MCTS."""
        wrapper_content = f"""#!/bin/bash
# MetalFish MCTS wrapper - intercepts 'go' and runs 'mctsmt' (GPU MCTS) instead

ENGINE="{metalfish_path}"

# Read UCI commands and transform 'go' to 'mctsmt threads=4'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'mctsmt threads=4' for multi-threaded GPU MCTS
        echo "mctsmt threads=4 ${{line#go}}"
    else
        echo "$line"
    fi
done | "$ENGINE"
"""
        with open(wrapper_path, "w") as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)

    def _create_mctsmt_wrapper(self, metalfish_path: Path, wrapper_path: Path):
        """Alias for _create_mcts_wrapper for backwards compatibility."""
        self._create_mcts_wrapper(metalfish_path, wrapper_path)

    def _parse_pgn_games(self, pgn_content: str) -> List[Dict[str, Any]]:
        """Parse individual games from PGN content."""
        games = []
        current_game = {"headers": {}, "moves": "", "raw": ""}
        in_moves = False
        current_raw = []
        
        for line in pgn_content.split('\n'):
            current_raw.append(line)
            line = line.strip()
            
            if line.startswith('['):
                # Header line
                in_moves = False
                match = re.match(r'\[(\w+)\s+"([^"]*)"\]', line)
                if match:
                    current_game["headers"][match.group(1)] = match.group(2)
            elif line and not line.startswith('['):
                # Move line
                in_moves = True
                current_game["moves"] += line + " "
            elif not line and in_moves and current_game["moves"]:
                # Empty line after moves - game complete
                current_game["raw"] = '\n'.join(current_raw)
                games.append(current_game)
                current_game = {"headers": {}, "moves": "", "raw": ""}
                current_raw = []
                in_moves = False
        
        # Don't forget the last game
        if current_game["moves"]:
            current_game["raw"] = '\n'.join(current_raw)
            games.append(current_game)
        
        return games

    def _format_game_output(self, game_num: int, total_games: int, 
                           white: str, black: str, result: str, 
                           reason: str, pgn: str, moves: str) -> str:
        """Format a single game's output with pretty printing."""
        lines = []
        
        # Game header
        lines.append(f"\n{Colors.BOLD}{'─' * 70}{Colors.RESET}")
        lines.append(f"{Colors.BOLD}  📋 Game {game_num}/{total_games}: {white} vs {black}{Colors.RESET}")
        lines.append(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
        
        # Result with color coding
        result_color = Colors.RESET
        result_icon = "🤝"
        if result == "1-0":
            result_color = Colors.GREEN
            result_icon = "👑"
        elif result == "0-1":
            result_color = Colors.RED
            result_icon = "👑"
        
        lines.append(f"  {result_icon} {Colors.BOLD}Result:{Colors.RESET} {result_color}{result}{Colors.RESET}")
        if reason:
            lines.append(f"  📝 {Colors.DIM}Reason: {reason}{Colors.RESET}")
        
        # Move count
        move_list = moves.strip().split()
        # Count actual moves (filter out move numbers and results)
        actual_moves = [m for m in move_list if not m.endswith('.') and m not in ['1-0', '0-1', '1/2-1/2', '*']]
        num_moves = len(actual_moves) // 2 + len(actual_moves) % 2
        lines.append(f"  🎯 {Colors.DIM}Moves: {num_moves}{Colors.RESET}")
        
        # PGN section
        lines.append(f"\n  {Colors.CYAN}┌{'─' * 66}┐{Colors.RESET}")
        lines.append(f"  {Colors.CYAN}│{Colors.RESET} {Colors.BOLD}PGN:{Colors.RESET}{' ' * 60}{Colors.CYAN}│{Colors.RESET}")
        lines.append(f"  {Colors.CYAN}├{'─' * 66}┤{Colors.RESET}")
        
        # Format PGN with word wrap
        for pgn_line in pgn.strip().split('\n'):
            if pgn_line.startswith('['):
                # Header line - dim
                formatted = f"  {Colors.CYAN}│{Colors.RESET} {Colors.DIM}{pgn_line[:64]:<64}{Colors.RESET} {Colors.CYAN}│{Colors.RESET}"
            else:
                # Move line - wrap at 64 chars
                words = pgn_line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 64:
                        current_line += (" " if current_line else "") + word
                    else:
                        if current_line:
                            lines.append(f"  {Colors.CYAN}│{Colors.RESET} {current_line:<64} {Colors.CYAN}│{Colors.RESET}")
                        current_line = word
                if current_line:
                    formatted = f"  {Colors.CYAN}│{Colors.RESET} {current_line:<64} {Colors.CYAN}│{Colors.RESET}"
                else:
                    continue
            lines.append(formatted)
        
        lines.append(f"  {Colors.CYAN}└{'─' * 66}┘{Colors.RESET}")
        
        return '\n'.join(lines)

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

        print(f"\n{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}  🎮 MATCH: {engine1.name} vs {engine2.name}{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"  📊 Games: {games} | ⏱️  Time Control: {time_control}")
        print(f"{Colors.DIM}  Running...{Colors.RESET}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            print(f"  {Colors.RED}⚠️  TIMEOUT{Colors.RESET}")
            return []

        # Parse results from output
        game_results = []

        # Read and parse PGN file for individual games
        pgn_content = ""
        try:
            with open(pgn_file.name, "r") as f:
                pgn_content = f.read()
        except:
            pass

        # Parse individual games from PGN
        parsed_games = self._parse_pgn_games(pgn_content)
        
        # Track running score
        wins1, wins2, draws = 0, 0, 0

        # Print each game with its PGN
        for i, game in enumerate(parsed_games, 1):
            headers = game["headers"]
            white = headers.get("White", engine1.name)
            black = headers.get("Black", engine2.name)
            result = headers.get("Result", "*")
            reason = headers.get("Termination", "")
            moves = game["moves"]
            raw_pgn = game["raw"]
            
            # Update score
            if result == "1-0":
                if white == engine1.name:
                    wins1 += 1
                else:
                    wins2 += 1
            elif result == "0-1":
                if black == engine1.name:
                    wins1 += 1
                else:
                    wins2 += 1
            else:
                draws += 1
            
            # Print formatted game output
            print(self._format_game_output(
                i, games, white, black, result, reason, raw_pgn, moves
            ))
            
            # Print running score
            total = wins1 + wins2 + draws
            score_pct = (wins1 + draws * 0.5) / total * 100 if total > 0 else 50
            print(f"\n  {Colors.MAGENTA}📈 Running Score ({engine1.name} vs {engine2.name}):{Colors.RESET}")
            print(f"     {Colors.BOLD}{wins1} - {draws} - {wins2}{Colors.RESET}  [{score_pct:.1f}%]")
            
            # Create GameResult
            game_results.append(GameResult(
                white=white,
                black=black,
                result=result,
                reason=reason,
                moves=len(moves.split()) // 3,  # Rough move count
                pgn=raw_pgn
            ))

        # If no games parsed from PGN, fall back to score parsing
        if not game_results:
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
            else:
                # Try to parse from PGN counts
                wins1 = pgn_content.count('[Result "1-0"]')
                wins2 = pgn_content.count('[Result "0-1"]')
                draws = pgn_content.count('[Result "1/2-1/2"]')

                for _ in range(wins1):
                    game_results.append(GameResult(engine1.name, engine2.name, "1-0"))
                for _ in range(wins2):
                    game_results.append(GameResult(engine1.name, engine2.name, "0-1"))
                for _ in range(draws):
                    game_results.append(GameResult(engine1.name, engine2.name, "1/2-1/2"))

        # Print final match summary
        total = wins1 + wins2 + draws
        score_pct = (wins1 + draws * 0.5) / total * 100 if total > 0 else 50
        
        print(f"\n{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}  ✅ MATCH COMPLETE: {engine1.name} vs {engine2.name}{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"  {Colors.GREEN}Final Score: {wins1} - {draws} - {wins2}  [{score_pct:.1f}%]{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}\n")

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
    """Get all available engine configurations for CI mode from engines_config.json."""
    configs = {}
    
    # Load configuration
    config = load_engines_config(base_dir)
    engines_config = config.get("engines", {})
    stockfish_config = config.get("stockfish", {})

    metalfish_path = base_dir / "build" / "metalfish"

    # MetalFish with standard Alpha-Beta search (full Stockfish with NNUE)
    ab_config = engines_config.get("MetalFish-AB", {})
    configs["MetalFish-AB"] = EngineConfig(
        name="MetalFish-AB",
        cmd=str(metalfish_path),
        options=ab_config.get("options", {"Threads": "1", "Hash": "128"}),
        expected_elo=ab_config.get("expected_elo"),
    )

    # MetalFish with Pure GPU MCTS (uses 'mctsmt' command)
    mcts_wrapper = base_dir / "tools" / "metalfish_mcts_wrapper.sh"
    mcts_config = engines_config.get("MetalFish-MCTS", {})
    configs["MetalFish-MCTS"] = EngineConfig(
        name="MetalFish-MCTS",
        cmd=str(mcts_wrapper),
        options=mcts_config.get("options", {}),
        expected_elo=mcts_config.get("expected_elo"),
    )

    # MetalFish with Parallel Hybrid search (MCTS + AB in parallel)
    hybrid_wrapper = base_dir / "tools" / "metalfish_hybrid_wrapper.sh"
    hybrid_config = engines_config.get("MetalFish-Hybrid", {})
    configs["MetalFish-Hybrid"] = EngineConfig(
        name="MetalFish-Hybrid",
        cmd=str(hybrid_wrapper),
        options=hybrid_config.get("options", {}),
        expected_elo=hybrid_config.get("expected_elo"),
    )

    # Add external engines from config
    for engine_name, engine_cfg in engines_config.items():
        # Skip MetalFish variants (already added above)
        if engine_name.startswith("MetalFish"):
            continue
        
        # Get engine path from config
        engine_path_str = engine_cfg.get("path", "")
        if not engine_path_str:
            continue
            
        engine_path = base_dir / engine_path_str
        
        # Special handling for Lc0 (needs network file)
        if engine_name == "Lc0":
            network_path_str = engine_cfg.get("network_path", "")
            if network_path_str:
                network_path = base_dir / network_path_str
                if engine_path.exists() and network_path.exists():
                    options = engine_cfg.get("options", {}).copy()
                    options["WeightsFile"] = str(network_path)
                    configs[engine_name] = EngineConfig(
                        name=engine_name,
                        cmd=str(engine_path),
                        options=options,
                        expected_elo=engine_cfg.get("expected_elo"),
                    )
            continue
        
        # Standard engine
        if engine_path.exists():
            configs[engine_name] = EngineConfig(
                name=engine_name,
                cmd=str(engine_path),
                options=engine_cfg.get("options", {}),
                expected_elo=engine_cfg.get("expected_elo"),
            )

    # Stockfish at various levels
    stockfish_path_str = stockfish_config.get("path", "reference/stockfish/src/stockfish")
    stockfish_path = base_dir / stockfish_path_str
    
    if stockfish_path.exists():
        if stockfish_levels is None:
            stockfish_levels = stockfish_config.get("default_levels", [1, 5, 10, 15, 20])

        # Get Elo map from config
        skill_elo_map = {int(k): v for k, v in stockfish_config.get("skill_elo_map", {}).items()}
        default_options = stockfish_config.get("options", {"Threads": "1", "Hash": "128"})

        for level in stockfish_levels:
            name = f"Stockfish-L{level}" if level < 20 else "Stockfish-Full"
            options = default_options.copy()
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


def _parse_pgn_games_standalone(pgn_content: str) -> List[Dict[str, Any]]:
    """Parse individual games from PGN content (standalone function for CI mode)."""
    games = []
    current_game = {"headers": {}, "moves": "", "raw": ""}
    in_moves = False
    current_raw = []
    
    for line in pgn_content.split('\n'):
        current_raw.append(line)
        line_stripped = line.strip()
        
        if line_stripped.startswith('['):
            # Header line
            in_moves = False
            match = re.match(r'\[(\w+)\s+"([^"]*)"\]', line_stripped)
            if match:
                current_game["headers"][match.group(1)] = match.group(2)
        elif line_stripped and not line_stripped.startswith('['):
            # Move line
            in_moves = True
            current_game["moves"] += line_stripped + " "
        elif not line_stripped and in_moves and current_game["moves"]:
            # Empty line after moves - game complete
            current_game["raw"] = '\n'.join(current_raw)
            games.append(current_game)
            current_game = {"headers": {}, "moves": "", "raw": ""}
            current_raw = []
            in_moves = False
    
    # Don't forget the last game
    if current_game["moves"]:
        current_game["raw"] = '\n'.join(current_raw)
        games.append(current_game)
    
    return games


def _format_ci_game_output(game_num: int, total_games: int, 
                           white: str, black: str, result: str, 
                           reason: str, pgn: str, moves: str) -> str:
    """Format a single game's output with pretty printing (CI mode)."""
    lines = []
    
    # Game header
    lines.append(f"\n{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    lines.append(f"{Colors.BOLD}  📋 Game {game_num}/{total_games}: {white} vs {black}{Colors.RESET}")
    lines.append(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    
    # Result with color coding
    result_color = Colors.RESET
    result_icon = "🤝"
    if result == "1-0":
        result_color = Colors.GREEN
        result_icon = "👑"
    elif result == "0-1":
        result_color = Colors.RED
        result_icon = "👑"
    
    lines.append(f"  {result_icon} {Colors.BOLD}Result:{Colors.RESET} {result_color}{result}{Colors.RESET}")
    if reason:
        lines.append(f"  📝 {Colors.DIM}Reason: {reason}{Colors.RESET}")
    
    # Move count
    move_list = moves.strip().split()
    actual_moves = [m for m in move_list if not m.endswith('.') and m not in ['1-0', '0-1', '1/2-1/2', '*']]
    num_moves = len(actual_moves) // 2 + len(actual_moves) % 2
    lines.append(f"  🎯 {Colors.DIM}Moves: {num_moves}{Colors.RESET}")
    
    # PGN section
    lines.append(f"\n  {Colors.CYAN}┌{'─' * 66}┐{Colors.RESET}")
    lines.append(f"  {Colors.CYAN}│{Colors.RESET} {Colors.BOLD}PGN:{Colors.RESET}{' ' * 60}{Colors.CYAN}│{Colors.RESET}")
    lines.append(f"  {Colors.CYAN}├{'─' * 66}┤{Colors.RESET}")
    
    # Format PGN with word wrap
    for pgn_line in pgn.strip().split('\n'):
        if pgn_line.startswith('['):
            # Header line - dim
            formatted = f"  {Colors.CYAN}│{Colors.RESET} {Colors.DIM}{pgn_line[:64]:<64}{Colors.RESET} {Colors.CYAN}│{Colors.RESET}"
            lines.append(formatted)
        elif pgn_line.strip():
            # Move line - wrap at 64 chars
            words = pgn_line.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= 64:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(f"  {Colors.CYAN}│{Colors.RESET} {current_line:<64} {Colors.CYAN}│{Colors.RESET}")
                    current_line = word
            if current_line:
                lines.append(f"  {Colors.CYAN}│{Colors.RESET} {current_line:<64} {Colors.CYAN}│{Colors.RESET}")
    
    lines.append(f"  {Colors.CYAN}└{'─' * 66}┘{Colors.RESET}")
    
    return '\n'.join(lines)


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
        mcts_wrapper = base_dir / "tools" / "metalfish_mcts_wrapper.sh"
        _create_mcts_wrapper_file(metalfish_path, mcts_wrapper)

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

    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  🎮 CI MATCH: {engine1_name} vs {engine2_name}{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"  📊 Games: {games} | ⏱️  Time Control: {time_control}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        output = result.stdout + result.stderr

        if result.returncode != 0:
            print(
                f"  {Colors.YELLOW}⚠️  cutechess-cli returned non-zero exit code: {result.returncode}{Colors.RESET}"
            )

    except subprocess.TimeoutExpired:
        print(f"  {Colors.RED}⚠️  ERROR: Match timed out after 7200 seconds{Colors.RESET}")
        return {
            "engine1": engine1_name,
            "engine2": engine2_name,
            "wins1": 0,
            "wins2": 0,
            "draws": 0,
            "error": "timeout",
        }

    # Read PGN file for individual games
    pgn_content = ""
    try:
        with open(pgn_file.name, "r") as f:
            pgn_content = f.read()
    except Exception as e:
        print(f"  {Colors.YELLOW}⚠️  Error reading PGN file: {e}{Colors.RESET}")

    # Parse individual games from PGN
    parsed_games = _parse_pgn_games_standalone(pgn_content)
    
    # Track running score
    wins1, wins2, draws = 0, 0, 0

    # Print each game with its PGN
    for i, game in enumerate(parsed_games, 1):
        headers = game["headers"]
        white = headers.get("White", engine1_name)
        black = headers.get("Black", engine2_name)
        game_result = headers.get("Result", "*")
        reason = headers.get("Termination", "")
        moves = game["moves"]
        raw_pgn = game["raw"]
        
        # Update score
        if game_result == "1-0":
            if white == engine1_name:
                wins1 += 1
            else:
                wins2 += 1
        elif game_result == "0-1":
            if black == engine1_name:
                wins1 += 1
            else:
                wins2 += 1
        else:
            draws += 1
        
        # Print formatted game output
        print(_format_ci_game_output(
            i, games, white, black, game_result, reason, raw_pgn, moves
        ))
        
        # Print running score
        total = wins1 + wins2 + draws
        score_pct = (wins1 + draws * 0.5) / total * 100 if total > 0 else 50
        print(f"\n  {Colors.MAGENTA}📈 Running Score ({engine1_name} vs {engine2_name}):{Colors.RESET}")
        print(f"     {Colors.BOLD}{wins1} - {draws} - {wins2}{Colors.RESET}  [{score_pct:.1f}%]")

    # If no games parsed from PGN, fall back to score parsing from output
    if not parsed_games:
        print(f"\n  {Colors.YELLOW}⚠️  No games parsed from PGN, using cutechess output{Colors.RESET}")
        
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
            print(f"  {Colors.YELLOW}⚠️  Could not parse score from cutechess output{Colors.RESET}")
            # Try PGN counts
            wins1 = pgn_content.count('[Result "1-0"]')
            wins2 = pgn_content.count('[Result "0-1"]')
            draws = pgn_content.count('[Result "1/2-1/2"]')

    # Cleanup
    try:
        os.unlink(pgn_file.name)
    except:
        pass

    # Print final match summary
    total = wins1 + wins2 + draws
    score_pct = (wins1 + draws * 0.5) / total * 100 if total > 0 else 50
    
    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  ✅ MATCH COMPLETE: {engine1_name} vs {engine2_name}{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"  {Colors.GREEN}Final Score: {wins1} - {draws} - {wins2}  [{score_pct:.1f}%]{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}\n")

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

    print(f"  📁 Results saved to: {output_file}")

    return match_result


def _create_hybrid_wrapper_file(metalfish_path: Path, wrapper_path: Path):
    """Create a wrapper script that uses 'parallel_hybrid' command for parallel MCTS+AB."""
    wrapper_content = f"""#!/bin/bash
# MetalFish Hybrid wrapper - intercepts 'go' and runs 'parallel_hybrid' (parallel MCTS+AB) instead

ENGINE="{metalfish_path}"

# Read UCI commands and transform 'go' to 'parallel_hybrid'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'parallel_hybrid' for parallel hybrid search
        echo "parallel_hybrid ${{line#go}}"
    else
        echo "$line"
    fi
done | "$ENGINE"
"""
    with open(wrapper_path, "w") as f:
        f.write(wrapper_content)
    os.chmod(wrapper_path, 0o755)


def _create_mcts_wrapper_file(metalfish_path: Path, wrapper_path: Path):
    """Create a wrapper script that uses 'mctsmt' command for pure GPU MCTS."""
    wrapper_content = f"""#!/bin/bash
# MetalFish MCTS wrapper - intercepts 'go' and runs 'mctsmt' (GPU MCTS) instead

ENGINE="{metalfish_path}"

# Read UCI commands and transform 'go' to 'mctsmt threads=4'
while IFS= read -r line; do
    if [[ "$line" == go* ]]; then
        # Replace 'go' with 'mctsmt threads=4' for multi-threaded GPU MCTS
        echo "mctsmt threads=4 ${{line#go}}"
    else
        echo "$line"
    fi
done | "$ENGINE"
"""
    with open(wrapper_path, "w") as f:
        f.write(wrapper_content)
    os.chmod(wrapper_path, 0o755)


def _create_mctsmt_wrapper_file(metalfish_path: Path, wrapper_path: Path):
    """Alias for _create_mcts_wrapper_file for backwards compatibility."""
    _create_mcts_wrapper_file(metalfish_path, wrapper_path)


def aggregate_ci_results(results_dir: Path, output_file: Path = None) -> Dict[str, Any]:
    """Aggregate results from multiple CI match jobs."""

    # Find all result JSON files (exclude summary.json)
    result_files = [f for f in results_dir.glob("*.json") if f.name != "summary.json"]

    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    # Aggregate all matches
    all_matches = []
    engines = set()

    for result_file in result_files:
        try:
            with open(result_file, "r") as f:
                match = json.load(f)

            # Skip files that don't look like match results
            if "engine1" not in match or "engine2" not in match:
                continue
                
            if "error" not in match:
                all_matches.append(match)
                engines.add(match["engine1"])
                engines.add(match["engine2"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping invalid result file {result_file}: {e}", file=sys.stderr)
            continue

    # Calculate Elo ratings - use Patricia as anchor with updated Elo
    elo_calc = EloCalculator(anchor_engine="Patricia", anchor_elo=3460)

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
