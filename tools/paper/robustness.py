#!/usr/bin/env python3
"""Robustness and timing analyses for the final reading of the paper cells (protocol amendment 12
(b), (c), (g), (h), and the review items on time per move, adjudication and the arbiter's flags).
Pure functions over the game lists of tools/paper/penta_stats.py, host-load samples and trace
records; tools/paper/make_figures.py writes the tables. No games are run.

- pair_records / elo_of_scores: pair scores in play order and their pentanomial Elo
- load_regression: ordinary least squares of the pair score on the fraction of the pair's logged
  minutes above the load threshold; the intercept is the estimate at zero outside load
- forfeit_variants: pairs containing a time forfeit removed; forfeits scored as draws
- without_first_openings: a cell without the pairs of its first k openings (RQ1a: 92)
- cochran_q: heterogeneity of the block estimates; chunk_bootstrap: resamples the runner's chunks
- linear_combination: a sum of cells' Elo estimates whose variance includes the covariance of
  their pair scores on shared openings (derived RQ2c = RQ2b + RQ2a)
- move_stats: time, depth and nodes per second of each arm's moves from the PGN move comments
- adjudications: adjudicated games by type and by the arm they favoured
- flag_counts: how often each 0/1 trace flag held on the disagreement records
"""

from __future__ import annotations

import bisect
import datetime as dt
import math
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import penta_stats


def elo_of_scores(scores: Sequence[float]) -> Dict:
    """Pentanomial Elo of pair scores in [0, 1] (as penta_stats.compute)."""
    n = len(scores)
    if not n:
        return {"pairs": 0, "elo": None, "elo_pm": None, "score": None, "se": None}
    s = sum(scores) / n
    var = sum((x - s) ** 2 for x in scores) / n
    se = math.sqrt(var / n) if var > 0 else 0.0
    out = {"pairs": n, "score": s, "se": se, "elo": None, "elo_pm": None}
    if 0.0 < s < 1.0:
        out["elo"] = penta_stats.elo_from_score(s)
        lo = penta_stats.elo_from_score(max(1e-9, s - 1.96 * se))
        hi = penta_stats.elo_from_score(min(1 - 1e-9, s + 1.96 * se))
        out["elo_pm"] = (hi - lo) / 2
    return out


def pair_records(games: List[Dict], engine: str) -> List[Dict]:
    """Scored pairs in play order: {i, j, score (0..1 for `engine`), fen, round}."""
    pairs, _ = penta_stats.make_pairs(games)
    out = []
    for i, j in sorted(pairs):
        a, b = penta_stats.score_for(engine, games[i]), penta_stats.score_for(
            engine, games[j]
        )
        if a is None or b is None:
            continue
        out.append(
            {
                "i": i,
                "j": j,
                "score": (a + b) / 2.0,
                "fen": games[i]["fen"],
                "round": games[i]["round"],
            }
        )
    return out


def _interval(g: Dict):
    try:
        fmt = "%Y-%m-%dT%H:%M:%S %z"
        return dt.datetime.strptime(g["start_time"], fmt), dt.datetime.strptime(
            g["end_time"], fmt
        )
    except (KeyError, ValueError, TypeError):
        return None


def pair_load_fraction(
    games: List[Dict],
    pair: Dict,
    samples: List[Tuple],
    threshold: float,
    times: Optional[List] = None,
) -> Optional[float]:
    """Fraction of the logged minutes overlapping either game of the pair whose outside CPU is
    above `threshold` (a sample at t covers (t - 60 s, t]); None when no minute was logged.
    `times` is the samples' time column, passed in by callers that loop over pairs."""
    if times is None:
        times = [t for t, *_ in samples]
    idx = set()
    for k in (pair["i"], pair["j"]):
        iv = _interval(games[k])
        if not iv:
            continue
        lo = bisect.bisect_left(times, iv[0])
        hi = bisect.bisect_right(times, iv[1] + dt.timedelta(seconds=60))
        idx.update(range(lo, hi))
    if not idx:
        return None
    return sum(1 for m in idx if samples[m][1] > threshold) / len(idx)


def ols(x: Sequence[float], y: Sequence[float]) -> Optional[Dict]:
    """y = a + b x by ordinary least squares, with classical standard errors."""
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx <= 0:
        return {"a": my, "b": None, "se_a": None, "se_b": None, "n": n}
    b = sum((u - mx) * (v - my) for u, v in zip(x, y)) / sxx
    a = my - b * mx
    rss = sum((v - a - b * u) ** 2 for u, v in zip(x, y))
    s2 = rss / (n - 2)
    return {
        "a": a,
        "b": b,
        "se_a": math.sqrt(s2 * (1.0 / n + mx * mx / sxx)),
        "se_b": math.sqrt(s2 / sxx),
        "n": n,
    }


def load_regression(
    games: List[Dict], pairs: List[Dict], samples: List[Tuple], threshold: float
) -> Dict:
    """Pair score on the pair's high-load minute fraction; the intercept is the estimate at zero
    outside load (Elo with a 95% interval). Pairs without a logged minute are left out.
    """
    xs, ys = [], []
    times = [t for t, *_ in samples]
    for p in pairs:
        f = pair_load_fraction(games, p, samples, threshold, times)
        if f is not None:
            xs.append(f)
            ys.append(p["score"])
    out: Dict = {
        "pairs_with_samples": len(xs),
        "pairs_without": len(pairs) - len(xs),
        "pairs_exposed": sum(1 for v in xs if v > 0),
    }
    fit = ols(xs, ys)
    if not fit or fit.get("se_a") is None:
        out["fit"] = fit
        return out
    a, se = fit["a"], fit["se_a"]
    out["fit"] = fit
    if 0.0 < a < 1.0:
        out["elo_at_zero_load"] = penta_stats.elo_from_score(a)
        lo = penta_stats.elo_from_score(max(1e-9, a - 1.96 * se))
        hi = penta_stats.elo_from_score(min(1 - 1e-9, a + 1.96 * se))
        out["elo_at_zero_load_pm"] = (hi - lo) / 2
    return out


def forfeit_variants(games: List[Dict], engine: str) -> Dict[str, Dict]:
    """The cell's statistics with forfeit pairs removed and with forfeits scored as draws."""
    forfeit = {
        k
        for k, g in enumerate(games)
        if penta_stats.classify_termination(g) == "time_forfeit"
    }
    pairs, _ = penta_stats.make_pairs(games)
    drop = set()
    for i, j in pairs:
        if i in forfeit or j in forfeit:
            drop.update((i, j))
    removed = [g for k, g in enumerate(games) if k not in drop]
    as_draws = [
        (
            dict(g, result="1/2-1/2", termination="", comment_tail="")
            if k in forfeit
            else g
        )
        for k, g in enumerate(games)
    ]
    return {
        "forfeit_games": len(forfeit),
        "removed": penta_stats.compute(removed, engine),
        "as_draws": penta_stats.compute(as_draws, engine),
    }


def without_first_openings(games: List[Dict], k: int) -> List[Dict]:
    """The games whose opening is not among the first `k` distinct openings in play order."""
    first: List[str] = []
    for g in games:
        if g["fen"] not in first:
            first.append(g["fen"])
            if len(first) == k:
                break
    skip = set(first)
    return [g for g in games if g["fen"] not in skip]


def _gammaincc(s: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(s, x) (series below s + 1, continued fraction above)."""
    if x <= 0:
        return 1.0
    lg = math.lgamma(s)
    if x < s + 1.0:
        term = total = 1.0 / s
        a = s
        for _ in range(1000):
            a += 1.0
            term *= x / a
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + s * math.log(x) - lg))
    b = x + 1.0 - s
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + s * math.log(x) - lg) * h


def chi2_sf(q: float, df: int) -> float:
    return _gammaincc(df / 2.0, q / 2.0)


def cochran_q(estimates: Sequence[float], ses: Sequence[float]) -> Optional[Dict]:
    """Cochran's Q for k estimates with standard errors: sum of w (e - pooled)^2, w = 1/se^2,
    referred to chi-square with k - 1 degrees of freedom."""
    pts = [(e, s) for e, s in zip(estimates, ses) if e is not None and s and s > 0]
    if len(pts) < 2:
        return None
    w = [1.0 / (s * s) for _, s in pts]
    pooled = sum(wi * e for wi, (e, _) in zip(w, pts)) / sum(w)
    q = sum(wi * (e - pooled) ** 2 for wi, (e, _) in zip(w, pts))
    df = len(pts) - 1
    return {"q": q, "df": df, "p": chi2_sf(q, df), "pooled": pooled}


def chunk_of_pairs(pairs: List[Dict], chunk_starts: Sequence[int]) -> List[int]:
    """Chunk index of each pair, from the PGN game index at which each runner invocation began
    (manifest history `games_before`); without starts, a chunk begins where rounds restart.
    """
    if chunk_starts:
        starts = sorted(set(chunk_starts))
        return [max(0, bisect.bisect_right(starts, p["i"]) - 1) for p in pairs]
    out, chunk, prev = [], 0, None
    for p in pairs:
        try:
            r = int(str(p["round"]).split(".")[0])
        except ValueError:
            r = None
        if prev is not None and r is not None and r <= prev:
            chunk += 1
        out.append(chunk)
        prev = r
    return out


def chunk_bootstrap(
    pairs: List[Dict], chunks: List[int], n_boot: int = 10000, seed: int = 1
) -> Optional[Dict]:
    """Percentile interval of the Elo when the runner's chunks (contiguous sessions of play) are
    resampled with replacement as clusters."""
    groups: Dict[int, List[float]] = {}
    for p, c in zip(pairs, chunks):
        groups.setdefault(c, []).append(p["score"])
    units = [(sum(v), len(v)) for v in groups.values()]
    if len(units) < 2:
        return None
    rng = random.Random(seed)
    reps = []
    for _ in range(n_boot):
        drawn = rng.choices(units, k=len(units))
        tot = sum(u[0] for u in drawn)
        n = sum(u[1] for u in drawn)
        s = min(1 - 1e-9, max(1e-9, tot / n))
        reps.append(penta_stats.elo_from_score(s))
    reps.sort()
    return {
        "chunks": len(units),
        "ci95": [reps[int(0.025 * n_boot)], reps[min(n_boot - 1, int(0.975 * n_boot))]],
    }


def linear_combination(cells: List[Tuple[List[Dict], float]]) -> Optional[Dict]:
    """Sum of coef * Elo over cells, with a delta-method variance that includes the covariance of
    the cells' pair scores on the openings they share: cov(mean_a, mean_b) = n_shared * c_ab /
    (n_a * n_b), c_ab the covariance of the paired pair scores."""
    parts = []
    for pairs, coef in cells:
        st = elo_of_scores([p["score"] for p in pairs])
        if st["elo"] is None:
            return None
        s = st["score"]
        deriv = 400.0 / (math.log(10.0) * s * (1.0 - s))
        by_fen: Dict[str, List[float]] = {}
        for p in pairs:
            by_fen.setdefault(p["fen"], []).append(p["score"])
        parts.append(
            {
                "coef": coef,
                "elo": st["elo"],
                "var": (st["se"] ** 2),
                "n": st["pairs"],
                "d": deriv,
                "by_fen": {f: sum(v) / len(v) for f, v in by_fen.items()},
            }
        )
    est = sum(p["coef"] * p["elo"] for p in parts)
    var = sum((p["coef"] * p["d"]) ** 2 * p["var"] for p in parts)
    shared_report = []
    for a in range(len(parts)):
        for b in range(a + 1, len(parts)):
            pa, pb = parts[a], parts[b]
            common = sorted(set(pa["by_fen"]) & set(pb["by_fen"]))
            if len(common) < 2:
                shared_report.append(0)
                continue
            xs = [pa["by_fen"][f] for f in common]
            ys = [pb["by_fen"][f] for f in common]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            c = sum((u - mx) * (v - my) for u, v in zip(xs, ys)) / len(xs)
            cov_means = len(common) * c / (pa["n"] * pb["n"])
            var += 2 * pa["coef"] * pb["coef"] * pa["d"] * pb["d"] * cov_means
            shared_report.append(len(common))
    se = math.sqrt(var) if var > 0 else 0.0
    return {
        "elo": est,
        "elo_pm": 1.96 * se,
        "shared_openings": shared_report,
        "components": [(p["coef"], p["elo"]) for p in parts],
    }


def load_coverage(
    games: List[Dict], pairs: List[Dict], samples: List[Tuple], blocks: List[int]
) -> Dict[int, Tuple[int, int]]:
    """Per block: (pairs with at least one logged minute, pairs)."""
    out: Dict[int, List[int]] = {}
    times = [t for t, *_ in samples]
    for p, b in zip(pairs, blocks):
        f = pair_load_fraction(games, p, samples, -1.0, times)
        rec = out.setdefault(b, [0, 0])
        rec[1] += 1
        if f is not None:
            rec[0] += 1
    return {b: (v[0], v[1]) for b, v in sorted(out.items())}


COMMENT_RE = re.compile(
    r"(?P<eval>[+-]?\d+\.\d+|[+-]?M\d+)/(?P<depth>\d+)\s+(?P<time>\d+(?:\.\d+)?)s"
    r"(?:,\s*n=(?P<nodes>\d+))?(?:,\s*nps=(?P<nps>\d+))?"
)


def parse_move_comment(comment: str) -> Optional[Dict]:
    m = COMMENT_RE.search(comment or "")
    if not m:
        return None
    return {
        "depth": int(m.group("depth")),
        "time": float(m.group("time")),
        "nodes": int(m.group("nodes")) if m.group("nodes") else None,
        "nps": int(m.group("nps")) if m.group("nps") else None,
    }


def iter_engine_moves(pgn_path: str, with_fen: bool = False) -> Iterable[Dict]:
    """One dict per engine move of every game: {game, engine, index (the engine's own move
    count in the game, from 1), time, depth, nodes, nps[, fen before the move]}."""
    import chess.pgn

    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        k = -1
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            k += 1
            h = game.headers
            names = {True: h.get("White", "?"), False: h.get("Black", "?")}
            board = game.board()
            count = {True: 0, False: 0}
            for node in game.mainline():
                side = board.turn
                info = parse_move_comment(node.comment)
                if info is not None:
                    count[side] += 1
                    rec = dict(info, game=k, engine=names[side], index=count[side])
                    if with_fen:
                        rec["fen"] = board.fen()
                    yield rec
                board.push(node.move)


def quantile(sorted_vals: Sequence[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(len(sorted_vals) - 1, lo + 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


INDEX_BINS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 60), (61, 10**9))


def bin_label(b: int) -> str:
    lo, hi = INDEX_BINS[b]
    return f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}"


def move_stats(moves: Iterable[Dict], classify=None) -> Dict[Tuple[str, str], Dict]:
    """Per (engine, class) time/depth/nps summaries and the mean time per move in INDEX_BINS of
    the engine's own move count. `classify(move) -> str` splits an engine's moves (for example
    transformer-enabled against time-safety moves of the hybrid); default class "all".
    """
    acc: Dict[Tuple[str, str], Dict[str, list]] = {}
    for m in moves:
        cls = classify(m) if classify else "all"
        if cls is None:
            continue
        keys = [(m["engine"], "all")]
        if cls != "all":
            keys.append((m["engine"], cls))
        for key in keys:
            a = acc.setdefault(key, {"time": [], "depth": [], "nps": [], "bins": {}})
            a["time"].append(m["time"])
            a["depth"].append(m["depth"])
            if m.get("nps"):
                a["nps"].append(m["nps"])
            b = next(
                i for i, (lo, hi) in enumerate(INDEX_BINS) if lo <= m["index"] <= hi
            )
            a["bins"].setdefault(b, []).append(m["time"])
    out = {}
    for key, a in acc.items():
        t, d, n = sorted(a["time"]), sorted(a["depth"]), sorted(a["nps"])
        out[key] = {
            "moves": len(t),
            "time_mean": sum(t) / len(t),
            "time_p50": quantile(t, 0.5),
            "time_p90": quantile(t, 0.9),
            "time_p99": quantile(t, 0.99),
            "time_max": t[-1],
            "depth_p50": quantile(d, 0.5),
            "nps_p50": quantile(n, 0.5),
            "time_by_index": {
                bin_label(b): sum(v) / len(v) for b, v in sorted(a["bins"].items())
            },
        }
    return out


def adjudications(games: List[Dict], arms: Sequence[str]) -> Dict:
    """Adjudicated games: draws, and wins by each arm."""
    out = {"draws": 0, "wins": {a: 0 for a in arms}}
    for g in games:
        if penta_stats.classify_termination(g) != "adjudication":
            continue
        if g["result"] == "1/2-1/2":
            out["draws"] += 1
        elif g["result"] == "1-0":
            out["wins"][g["white"]] = out["wins"].get(g["white"], 0) + 1
        elif g["result"] == "0-1":
            out["wins"][g["black"]] = out["wins"].get(g["black"], 0) + 1
    return out


def flag_counts(records: Iterable[Dict]) -> Dict:
    """On the traced disagreements (alpha-beta and MCTS moves differ, not a time-safety move):
    how often each 0/1 trace field held, plus the hint fields' presence and agreement.
    """
    counts: Dict[str, int] = {}
    nonflag = set()
    n = hints = first_is_mcts = first_is_ab = 0
    for r in records:
        if r.get("reason") == "ab_fallback_time_safety":
            continue
        if (
            not (r.get("ab_move") and r.get("mcts_move"))
            or r["ab_move"] == r["mcts_move"]
        ):
            continue
        n += 1
        for k, v in r.items():
            if k in nonflag or isinstance(v, (list, dict, str)) or v is None:
                continue
            if isinstance(v, bool):
                v = int(v)
            if v in (0, 1):
                counts[k] = counts.get(k, 0) + v
            else:
                nonflag.add(k)
        h = r.get("ab_hints") or []
        if h:
            hints += 1
            first = h[0].get("move")
            first_is_mcts += first == r["mcts_move"]
            first_is_ab += first == r["ab_move"]
    flags = {k: c for k, c in counts.items() if k not in nonflag}
    return {
        "disagreements": n,
        "flags": dict(sorted(flags.items(), key=lambda kv: -kv[1])),
        "with_hints": hints,
        "first_hint_is_mcts_move": first_is_mcts,
        "first_hint_is_ab_move": first_is_ab,
    }
