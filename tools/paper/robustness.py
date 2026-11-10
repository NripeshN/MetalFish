#!/usr/bin/env python3
"""Robustness and timing analyses for the final reading of the paper cells (protocol amendment 12
(b), (c), (g), (h), and the review items on time per move, adjudication and the arbiter's flags).
Pure functions over the game lists of tools/paper/penta_stats.py, host-load samples and trace
records; tools/paper/make_figures.py writes the tables. No games are run.

- pair_records / elo_of_scores: pair scores in play order and their pentanomial Elo
- load_regression: ordinary least squares of the pair score on the fraction of the pair's logged
  minutes above the load threshold; the intercept is the estimate at zero outside load
- forfeit_variants: pairs containing a time forfeit removed; forfeits scored as draws
- first_overruns / strict_referee: the moves that overran their clock, and the cell scored by the
  registered referee rule (no time margin) for games played with fastchess's timemargin
  (runner cells, amendment 15 (j))
- without_first_openings: a cell without the pairs of its first k openings (RQ1a: 92)
- cochran_q: heterogeneity of the block estimates; chunk_bootstrap: resamples the runner's chunks
- block_t_interval / block_paired_contrast: the mean of a runner cell's block estimates, or of
  the block-by-block differences of compared cells, with a Student t interval over the blocks
  (amendment 17)
- linear_combination: a sum of cells' Elo estimates whose variance includes the covariance of
  their pair scores on shared openings (derived RQ2c = RQ2b + RQ2a)
- move_stats: time, depth and nodes per second of each arm's moves from the PGN move comments
- adjudications: adjudicated games by type and by the arm they favoured
- flag_counts: how often each 0/1 trace flag held on the disagreement records
- playout_counter_shares: how many transformer-enabled moves ended with the MCTS playout
  counter at the agreement rule's thresholds (amendment 17)
- evals_by_position / eval_rate_by_block: the hybrid's MCTS evaluation rate by block, a measure
  of GPU health (amendment 12 (c)), from the traces joined to the PGN move times by position
- load_builds / split_by_build: the games grouped by the macOS build each pair started on
  (amendment 13; build dates in docs/paper_v2/os_builds.json)
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


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) (continued fraction, modified Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 1000):
        for num in (
            m * (b - m) * x / ((a + 2 * m - 1.0) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1.0)),
        ):
            d = 1.0 + num * d
            d = 1.0 / (tiny if abs(d) < tiny else d)
            c = 1.0 + num / c
            c = tiny if abs(c) < tiny else c
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return front * h / a


def t_two_sided_tail(t: float, df: int) -> float:
    """P(|T| > t) for Student's t with df degrees of freedom."""
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def t_quantile(df: int, two_sided: float = 0.05) -> float:
    """The t with P(|T| > t) = two_sided (0.05: the multiplier of a 95% interval), by bisection."""
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_two_sided_tail(mid, df) > two_sided:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def block_t_interval(estimates: Sequence[Optional[float]]) -> Optional[Dict]:
    """Unweighted mean of k block estimates with a 95% Student t interval on k - 1 degrees of
    freedom (amendment 17 (b)). Each block of a runner cell ran on its own runner, so the blocks
    are the units and the interval includes the variation between runners. None unless there are
    at least two blocks and every block has a finite estimate."""
    k = len(estimates)
    if k < 2 or any(e is None or not math.isfinite(e) for e in estimates):
        return None
    mean = sum(estimates) / k
    sd = math.sqrt(sum((e - mean) ** 2 for e in estimates) / (k - 1))
    return {
        "mean": mean,
        "pm": t_quantile(k - 1) * sd / math.sqrt(k),
        "sd": sd,
        "blocks": k,
        "df": k - 1,
    }


def block_paired_contrast(
    cells: List[Tuple[Sequence[Optional[float]], float]],
) -> Optional[Dict]:
    """Sum of coef * (block estimate) formed block by block over cells whose blocks played the
    same openings and were dispatched together, with the t interval of block_t_interval over the
    k differences (amendment 17 (c)). None unless the cells have the same number of blocks, each
    with a finite estimate."""
    if not cells or len({len(ests) for ests, _ in cells}) != 1:
        return None
    if any(e is None or not math.isfinite(e) for ests, _ in cells for e in ests):
        return None
    k = len(cells[0][0])
    return block_t_interval(
        [sum(coef * ests[i] for ests, coef in cells) for i in range(k)]
    )


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
)
# fastchess writes the optional fields after the time in its own order (with timeleft=true and
# seldepth=true: "tl=..., n=..., sd=..., nps=..."), so each is looked up by name
NODES_RE = re.compile(r",\s*n=(?P<nodes>\d+)")
NPS_RE = re.compile(r",\s*nps=(?P<nps>\d+)")
TL_RE = re.compile(r",\s*tl=(?P<tl>-?\d+(?:\.\d+)?)s")


def parse_move_comment(comment: str) -> Optional[Dict]:
    m = COMMENT_RE.search(comment or "")
    if not m:
        return None
    rest = comment[m.end() :]
    n, nps, tl = NODES_RE.search(rest), NPS_RE.search(rest), TL_RE.search(rest)
    return {
        "depth": int(m.group("depth")),
        "time": float(m.group("time")),
        "nodes": int(n.group("nodes")) if n else None,
        "nps": int(nps.group("nps")) if nps else None,
        "timeleft": float(tl.group("tl")) if tl else None,
    }


def parse_tc(tc: str) -> Tuple[int, int]:
    """Base time and increment in ms of a fastchess `tc=` value such as "30+0.3"."""
    base, _, inc = tc.partition("+")
    return round(float(base) * 1000), round(float(inc or 0) * 1000)


def first_overruns(pgn_path: str, tc: str) -> Dict[int, Tuple[str, int]]:
    """For each game (its index in the PGN) in which a move took longer than the clock its side
    had: the engine that made the first such move, and that move's index among the engine's own
    moves. fastchess 1.8.2 starts each clock at the base time plus one increment and records,
    after each move, the time it measured and the clock left (in whole ms); a move overruns when
    its time exceeds the clock left after its side's previous move. With no time margin fastchess
    ends the game there, lost by that engine; with a margin (amendment 15 (j)) the overrun is
    forgiven, the clock set to 0 plus the increment, and play goes on, the engines having been
    sent the same clocks up to that move either way.
    """
    base, inc = parse_tc(tc)
    clock: Dict[Tuple[int, str], int] = {}
    out: Dict[int, Tuple[str, int]] = {}
    for m in iter_engine_moves(pgn_path):
        key = (m["game"], m["engine"])
        have = clock.get(key, base + inc)
        used = round(m["time"] * 1000)
        if used > have and m["game"] not in out:
            out[m["game"]] = (m["engine"], m["index"])
        if m.get("timeleft") is not None:
            clock[key] = round(m["timeleft"] * 1000)
        else:
            clock[key] = max(0, have - used) + inc
    return out


def strict_referee(
    games: List[Dict], pgn_path: str, tc: str, engine: str
) -> Dict[str, object]:
    """The cell under the registered referee rule (no time margin) for games played with a
    margin: a game with an overrun is lost by the engine whose move overran first (as fastchess
    with timemargin 0 would have ended it); `games` are penta_stats.read_games of `pgn_path`.
    """
    over = first_overruns(pgn_path, tc)
    rescored = []
    for k, g in enumerate(games):
        if k in over:
            loser = over[k][0]
            rescored.append(
                dict(
                    g,
                    result="0-1" if g["white"] == loser else "1-0",
                    termination="time forfeit",
                    comment_tail="",
                )
            )
        else:
            rescored.append(g)
    by: Dict[str, int] = {}
    for loser, _ in over.values():
        by[loser] = by.get(loser, 0) + 1
    return {
        "games_with_overrun": len(over),
        "overruns_by": by,
        "results_changed": sum(
            1 for k in over if rescored[k]["result"] != games[k]["result"]
        ),
        "stats": penta_stats.compute(rescored, engine),
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
    how often each 0/1 trace field held, plus the hint fields' presence and agreement. Over every
    traced move that is not a time-safety move: how many had root hints and how many ran the
    verification sub-search (the engine fills ab_verified_hints only when it ran; RQ6a).
    """
    counts: Dict[str, int] = {}
    nonflag = set()
    n = hints = first_is_mcts = first_is_ab = 0
    traced = traced_hints = traced_verified = 0
    for r in records:
        if r.get("reason") == "ab_fallback_time_safety":
            continue
        traced += 1
        traced_hints += bool(r.get("ab_hints"))
        traced_verified += bool(r.get("ab_verified_hints"))
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
        "traced": traced,
        "traced_with_hints": traced_hints,
        "traced_with_verification": traced_verified,
    }


def playout_counter_shares(
    records: Iterable[Dict], thresholds: Sequence[int] = (100, 150)
) -> Dict:
    """Over the traced transformer-enabled moves (every trace record that is not a time-safety
    move): how many ended with the MCTS search's playout counter at or above each threshold. The
    agreement rule reads that counter (hybrid_search.cpp: Stats().total_nodes) against 100, or 150
    in positions the classifier rates as highly complex; the trace records it at the end of the
    move as mcts_playouts, but not the position's class (amendment 17 (e))."""
    enabled = counted = 0
    reached = {t: 0 for t in thresholds}
    for r in records:
        if r.get("reason") == "ab_fallback_time_safety":
            continue
        enabled += 1
        n = r.get("mcts_playouts")
        if not isinstance(n, (int, float)) or isinstance(n, bool):
            continue
        counted += 1
        for t in thresholds:
            reached[t] += n >= t
    return {"enabled": enabled, "with_counter": counted, "reached": reached}


def evals_by_position(records: Iterable[Dict]) -> Dict[str, Optional[int]]:
    """FEN -> network evaluations of the transformer-enabled trace record at that position (not a
    time-safety move, mcts_evals > 0). None marks a position with more than one record of any
    kind, which cannot be matched to a single move."""
    out: Dict[str, Optional[int]] = {}
    seen = set()
    for r in records:
        fen = r.get("fen")
        if not fen:
            continue
        if fen in seen:
            out[fen] = None
            continue
        seen.add(fen)
        ev = r.get("mcts_evals")
        if (
            r.get("reason") != "ab_fallback_time_safety"
            and isinstance(ev, (int, float))
            and ev > 0
        ):
            out[fen] = int(ev)
    return out


def class_switches(seq: Iterable[Tuple[int, str]]) -> Dict:
    """Switches between transformer-enabled and time-safety moves within each game, from the
    hybrid's moves in game order as (game, class)."""
    last: Dict[int, str] = {}
    to_enabled = to_safety = 0
    for game, cls in seq:
        prev = last.get(game)
        if prev is not None and prev != cls:
            if cls == "transformer-enabled":
                to_enabled += 1
            else:
                to_safety += 1
        last[game] = cls
    return {"games": len(last), "to_enabled": to_enabled, "to_safety": to_safety}


def eval_rate_by_block(
    moves: Iterable[Tuple[int, int, float]], blocks: Sequence[Tuple[int, int, int]]
) -> Dict[int, Dict]:
    """The hybrid's MCTS evaluation rate by block (amendment 12 (c)) from (PGN game index, network
    evaluations, move time in s) of its matched transformer-enabled moves and the blocks' (block,
    first game, end) ranges: per block the moves, the median and 10th percentile of evaluations
    per second of move time, and total evaluations over total move time."""
    acc: Dict[int, Dict] = {}
    for k, ev, t in moves:
        if not t or t <= 0:
            continue
        b = next((b for b, first, end in blocks if first <= k < end), None)
        if b is None:
            continue
        a = acc.setdefault(b, {"rates": [], "evals": 0, "time": 0.0})
        a["rates"].append(ev / t)
        a["evals"] += ev
        a["time"] += t
    out = {}
    for b, a in sorted(acc.items()):
        r = sorted(a["rates"])
        out[b] = {
            "moves": len(r),
            "rate_p50": quantile(r, 0.5),
            "rate_p10": quantile(r, 0.1),
            "rate_total": a["evals"] / a["time"],
        }
    return out


def load_builds(path: str) -> List[Tuple[dt.datetime, str]]:
    """[(install time, build)] from docs/paper_v2/os_builds.json, in time order."""
    import json

    with open(path) as fh:
        data = json.load(fh)
    out = []
    for b in data["builds"]:
        t = dt.datetime.strptime(b["installed_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc
        )
        out.append((t, b["build"]))
    return sorted(out)


def split_by_build(
    games: List[Dict], builds: List[Tuple[dt.datetime, str]]
) -> Dict[str, List[Dict]]:
    """The games grouped by the macOS build that was installed when each pair's first game
    started (unpaired games by their own start), keeping each pair's games adjacent and in play
    order, so penta_stats.compute pairs them again."""
    times = [t for t, _ in builds]

    def build_at(g: Dict) -> str:
        iv = _interval(g)
        if not iv:
            return "unknown"
        k = bisect.bisect_right(times, iv[0]) - 1
        return builds[k][1] if k >= 0 else "before the first recorded build"

    pairs, singles = penta_stats.make_pairs(games)
    units = [(min(i, j), [i, j]) for i, j in pairs] + [(k, [k]) for k in singles]
    out: Dict[str, List[Dict]] = {}
    for first, idx in sorted(units):
        label = build_at(games[first])
        out.setdefault(label, []).extend(games[k] for k in sorted(idx))
    return out
