#!/usr/bin/env python3
"""Pentanomial match statistics from a fastchess/cutechess PGN.

Game pairs (same opening, colours swapped) are the unit of observation. For each pair the
score of the reference engine (``--engine``) is 0, 0.5, 1, 1.5 or 2 points; per-game score
x_i = pair_points / 2. With s = mean(x_i) and var = population variance of x_i:

    SE(s)      = sqrt(var / n_pairs)
    Elo        = 400 * log10(s / (1 - s))
    Elo CI     = [Elo(s - 1.96 SE), Elo(s + 1.96 SE)]
    sigma_game = sqrt(2 * var)                       (per-game standard deviation)
    nElo       = (800 / ln 10) * (s - 0.5) / sigma_game
    LOS        = Phi((s - 0.5) / SE)

These reproduce fastchess's ``-report penta=true`` printout (validated on 2026-10-12:
Elo +/- 38.22 and nElo +/- 76.13 for Ptnml [0, 10, 20, 10, 0]). The trinomial (per-game)
variance is reported alongside for comparison.

Usage:
    python3 tools/paper/penta_stats.py games.pgn --engine MF-Hybrid-T8 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import chess.pgn

ELO_SCALE = 400.0 / math.log(10.0)  # 173.7
NELO_SCALE = 800.0 / math.log(10.0)  # 347.4


def elo_from_score(s: float) -> float:
    if s <= 0.0:
        return -math.inf
    if s >= 1.0:
        return math.inf
    return 400.0 * math.log10(s / (1.0 - s))


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def read_games(path: str) -> List[Dict]:
    games = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            h = game.headers
            games.append(
                {
                    "round": h.get("Round", "?"),
                    "white": h.get("White", "?"),
                    "black": h.get("Black", "?"),
                    "result": h.get("Result", "*"),
                    "termination": h.get("Termination", ""),
                    "fen": h.get("FEN", ""),
                    "plycount": h.get("PlyCount", ""),
                    "start_time": h.get("GameStartTime", ""),
                    "end_time": h.get("GameEndTime", ""),
                    "comment_tail": (game.end().comment or "")[:120],
                }
            )
    return games


def score_for(engine: str, g: Dict) -> Optional[float]:
    r = g["result"]
    if r == "1-0":
        return 1.0 if g["white"] == engine else 0.0
    if r == "0-1":
        return 1.0 if g["black"] == engine else 0.0
    if r == "1/2-1/2":
        return 0.5
    return None


def classify_termination(g: Dict) -> str:
    t = (g["termination"] or "").lower()
    c = (g["comment_tail"] or "").lower()
    if "time" in t or "loses on time" in c or "time forfeit" in c:
        return "time_forfeit"
    if "illegal" in t or "illegal" in c:
        return "illegal_move"
    if "adjudication" in t or "adjudicat" in c:
        return "adjudication"
    if "disconnect" in t or "disconnect" in c or "crash" in c or "stall" in c:
        return "disconnect"
    if "abandoned" in t or "unterminated" in t:
        return "abandoned"
    if "maxmoves" in c or "max moves" in c or "move limit" in c:
        return "max_moves"
    return "normal"


def compute(games: List[Dict], engine: str) -> Dict:
    # A pair is the two games of one fastchess round. fastchess restarts round numbers at 1 in
    # every invocation, and a cell is played in several invocations (blocks, resumes), so the
    # round number alone collides; the opening FEN disambiguates (blocks draw disjoint openings).
    by_round: Dict[tuple, List[Dict]] = defaultdict(list)
    for g in games:
        by_round[(g["round"], g["fen"])].append(g)

    pair_scores: List[float] = []
    penta = [0, 0, 0, 0, 0]
    wl_pairs = 0
    dd_pairs = 0
    unpaired = 0
    per_game = Counter()
    terminations = Counter()
    for rnd, gs in by_round.items():
        for g in gs:
            terminations[classify_termination(g)] += 1
            sc = score_for(engine, g)
            if sc is None:
                per_game["unfinished"] += 1
            elif sc == 1.0:
                per_game["win"] += 1
            elif sc == 0.0:
                per_game["loss"] += 1
            else:
                per_game["draw"] += 1
        if len(gs) != 2:
            unpaired += len(gs)
            continue
        scores = [score_for(engine, g) for g in gs]
        if any(s is None for s in scores):
            unpaired += 2
            continue
        colours_swapped = gs[0]["white"] != gs[1]["white"]
        if not colours_swapped:
            unpaired += 2
            continue
        total = scores[0] + scores[1]  # 0 .. 2
        pair_scores.append(total / 2.0)
        penta[int(round(total * 2))] += 1
        if total == 1.0:
            if 0.5 in scores:
                dd_pairs += 1
            else:
                wl_pairs += 1

    n = len(pair_scores)
    out: Dict = {
        "engine": engine,
        "games": sum(per_game.values()),
        "pairs": n,
        "unpaired_games": unpaired,
        "wins": per_game["win"],
        "draws": per_game["draw"],
        "losses": per_game["loss"],
        "unfinished": per_game["unfinished"],
        "ptnml_0_2": penta,
        "wl_pairs": wl_pairs,
        "dd_pairs": dd_pairs,
        "terminations": dict(terminations),
    }
    if n == 0:
        return out
    s = sum(pair_scores) / n
    var = sum((x - s) ** 2 for x in pair_scores) / n
    se = math.sqrt(var / n) if var > 0 else 0.0
    sigma_game = math.sqrt(2.0 * var)
    elo = elo_from_score(s)
    lo = elo_from_score(max(0.0, s - 1.96 * se))
    hi = elo_from_score(min(1.0, s + 1.96 * se))
    games_total = out["wins"] + out["draws"] + out["losses"]
    draw_ratio = out["draws"] / games_total if games_total else 0.0
    # Trinomial comparison (per-game, ignoring pairing)
    if games_total:
        s_tri = (out["wins"] + 0.5 * out["draws"]) / games_total
        var_tri = (
            out["wins"] * (1 - s_tri) ** 2
            + out["draws"] * (0.5 - s_tri) ** 2
            + out["losses"] * (0 - s_tri) ** 2
        ) / games_total
        se_tri = math.sqrt(var_tri / games_total) if var_tri > 0 else 0.0
        elo_tri_half = (
            (
                elo_from_score(min(1, s_tri + 1.96 * se_tri))
                - elo_from_score(max(0, s_tri - 1.96 * se_tri))
            )
            / 2
            if 0 < s_tri < 1
            else math.inf
        )
    else:
        var_tri = se_tri = elo_tri_half = 0.0
    out.update(
        {
            "score": s,
            "score_se": se,
            "pair_variance": var,
            "sigma_game": sigma_game,
            "elo": elo,
            "elo_ci95": [lo, hi],
            "elo_pm": (
                (hi - lo) / 2 if math.isfinite(hi) and math.isfinite(lo) else math.inf
            ),
            "nelo": (
                (NELO_SCALE * (s - 0.5) / sigma_game) if sigma_game > 0 else math.inf
            ),
            "nelo_pm": (
                (NELO_SCALE * 1.96 * se / sigma_game) if sigma_game > 0 else math.inf
            ),
            "los": (
                phi((s - 0.5) / se)
                if se > 0
                else (1.0 if s > 0.5 else 0.5 if s == 0.5 else 0.0)
            ),
            "draw_ratio": draw_ratio,
            "draw_ratio_pairs": (penta[2] / n) if n else 0.0,
            "pairs_ratio": (
                ((penta[3] + penta[4]) / (penta[0] + penta[1]))
                if (penta[0] + penta[1])
                else math.inf
            ),
            "wl_dd_ratio": (wl_pairs / dd_pairs) if dd_pairs else math.inf,
            "trinomial": {"variance_per_game": var_tri, "elo_pm": elo_tri_half},
            "variance_ratio_penta_over_tri": (
                ((2 * var) / var_tri) if var_tri > 0 else None
            ),
        }
    )
    return out


def fmt(stats: Dict) -> str:
    if stats.get("pairs", 0) == 0:
        return f"{stats['engine']}: no complete game pairs found ({stats.get('games', 0)} games)"
    return (
        f"Results of {stats['engine']} ({stats['games']} games, {stats['pairs']} pairs):\n"
        f"Elo: {stats['elo']:+.2f} +/- {stats['elo_pm']:.2f}, nElo: {stats['nelo']:+.2f} +/- {stats['nelo_pm']:.2f}\n"
        f"LOS: {100*stats['los']:.2f} %, DrawRatio(pairs): {100*stats['draw_ratio_pairs']:.2f} %, DrawRatio(games): {100*stats['draw_ratio']:.2f} %, PairsRatio: {stats['pairs_ratio']:.2f}\n"
        f"Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}, Points: {stats['score']*stats['games']:.1f} ({100*stats['score']:.2f} %)\n"
        f"Ptnml(0-2): {stats['ptnml_0_2']}, WL/DD Ratio: {stats['wl_dd_ratio']:.2f}\n"
        f"Trinomial Elo +/- {stats['trinomial']['elo_pm']:.2f} (pentanomial/trinomial variance ratio "
        f"{(stats['variance_ratio_penta_over_tri'] or float('nan')):.2f}); terminations: {stats['terminations']}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pgn")
    ap.add_argument(
        "--engine", required=True, help="engine name whose perspective is reported"
    )
    ap.add_argument("--json", help="write the statistics as JSON to this path")
    args = ap.parse_args(argv)
    games = read_games(args.pgn)
    stats = compute(games, args.engine)
    print(fmt(stats))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(stats, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
