#!/usr/bin/env python3
"""Parse MetalFish hybrid telemetry (``info string HybridTrace: ...`` and the time-safety
fallback lines) from a fastchess engine log, a cutechess log, a results.json produced by
tools/run_tournament_live.py, or a raw UCI transcript, into one JSON record per move.

Record fields (missing fields are omitted): source, engine, reason, selected, ab_move,
mcts_move, ab_score, ab_depth, mcts_q, mcts_cp, mcts_playouts, mcts_evals,
mcts_best_visits, mcts_root_visits, gpu_active, fallback_budget_ms, thread_split
(dict), plus every other ``Key=value`` scalar in the trace (lower_snake_case) and the
bracketed lists ``mcts_top``/``ab_root``/``ab_hints`` as parsed lists. Records from an engine
log also carry ``game`` (the game's ordinal for that engine in the log; a new game starts when the
``position`` command's base position changes or its move list stops extending the previous one)
and ``opening`` (the base position: the book FEN, or ``startpos``).

Usage:
    python3 tools/paper/parse_hybrid_trace.py engines.log --out moves.jsonl
    python3 tools/paper/parse_hybrid_trace.py results.json --summary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional

TRACE_RE = re.compile(r"HybridTrace:\s*(.*)$")
SPLIT_RE = re.compile(
    r"Hybrid thread split:\s*MCTS=(\d+)\s+AB=(\d+)\s*\(search workers=(\d+),\s*\+(\d+) coordinator,\s*total=(\d+)\)"
)
FALLBACK_RE = re.compile(
    r"Time safety:\s*(.*?using Alpha-Beta without transformer MCTS)"
)
BUDGET_RE = re.compile(r"estimated move budget (\d+)ms")
TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=(\[[^\]]*\]|\S+)")
# fastchess engine log line (v1.8): "[Engine] [HH:MM:SS.ffffff] <  thread-id>  NAME ---> payload"
# ("<---" marks GUI->engine traffic, "--->" engine->GUI). cutechess logs are "N <NAME(k): payload".
FASTCHESS_LINE_RE = re.compile(
    r"^\[Engine\]\s+\[[^\]]+\]\s+<[^>]*>\s+(?P<engine>.+?)\s+(?:--->|<---)\s+(?P<payload>.*)$"
)
CUTECHESS_LINE_RE = re.compile(
    r"^\d+\s+[<>](?P<engine>[^()]+)\(\d+\):\s*(?P<payload>.*)$"
)


def to_snake(key: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def parse_scalar(v: str):
    if v in ("none", "None", ""):
        return None
    try:
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        if re.fullmatch(r"-?\d+\.\d*(e-?\d+)?", v):
            return float(v)
    except ValueError:
        pass
    return v


def parse_list(v: str) -> List[Dict]:
    inner = v.strip()[1:-1].strip()
    if not inner:
        return []
    items = []
    for item in inner.split(","):
        parts = item.split(":")
        rec: Dict = {"move": parts[0]}
        for p in parts[1:]:
            if "=" in p:
                k, val = p.split("=", 1)
                rec[k] = parse_scalar(val)
        items.append(rec)
    return items


def parse_trace_payload(payload: str) -> Dict:
    rec: Dict = {}
    for key, val in TOKEN_RE.findall(payload):
        k = to_snake(key)
        if val.startswith("["):
            rec[k] = parse_list(val)
        else:
            rec[k] = parse_scalar(val)
    # normalised aliases
    alias = {
        "ab_move": "ab_move",
        "mcts_move": "mcts_move",
        "abscore": "ab_score",
        "abdepth": "ab_depth",
        "mctsq": "mcts_q",
        "mctscp": "mcts_cp",
        "mctsplayouts": "mcts_playouts",
        "mctsevals": "mcts_evals",
        "mctsbestvisits": "mcts_best_visits",
        "mctsrootvisits": "mcts_root_visits",
    }
    for k in list(rec.keys()):
        flat = k.replace("_", "")
        if flat in alias and alias[flat] not in rec:
            rec[alias[flat]] = rec[k]
    rec["gpu_active"] = bool(rec.get("mcts_evals", 0)) and rec.get("mcts_evals", 0) > 0
    return rec


def iter_lines(path: str) -> Iterable[Dict]:
    """Yield dicts {engine, text} for every candidate line in the input file."""
    if path.endswith(".json"):
        data = json.load(open(path))
        for m in data.get("matches", []):
            for g in m.get("games", []):
                for entry in g.get("search_log", []):
                    for line in entry.get("lines", []):
                        yield {
                            "engine": entry.get("engine"),
                            "text": line,
                            "ply": entry.get("ply"),
                            "game": g.get("game"),
                            "fen": entry.get("fen"),
                        }
        return
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if (
                "HybridTrace:" not in line
                and "Hybrid thread split" not in line
                and "Time safety" not in line
                and "position " not in line
            ):
                continue
            m = FASTCHESS_LINE_RE.match(line) or CUTECHESS_LINE_RE.match(line)
            if m:
                yield {"engine": m.group("engine").strip(), "text": m.group("payload")}
            else:
                yield {"engine": None, "text": line}


def split_position(cmd: str):
    """(base, moves) of a UCI ``position`` command; base is ``startpos`` or the FEN."""
    toks = cmd.split()
    end = toks.index("moves") if "moves" in toks else len(toks)
    base = " ".join(toks[1:end])
    if base.startswith("fen "):
        base = base[4:]
    return base, toks[end + 1 :]


def parse(path: str) -> List[Dict]:
    return list(iter_records(path))


def iter_records(path: str) -> Iterable[Dict]:
    """Yield the records of `path` one at a time (see the module docstring)."""
    current_split: Dict[Optional[str], Dict] = {}
    last_position: Dict[Optional[str], str] = {}
    game_state: Dict[Optional[str], tuple] = {}
    game_no: Dict[Optional[str], int] = {}

    def tag_game(rec: Dict, item: Dict, eng: Optional[str]) -> None:
        if item.get("ply") is not None:
            rec["ply"] = item["ply"]
            rec["game"] = item.get("game")
        elif eng in game_state:
            rec["game"] = game_no[eng]
            rec["opening"] = game_state[eng][0]

    for item in iter_lines(path):
        text = item["text"]
        eng = item.get("engine")
        if text.startswith("position "):
            last_position[eng] = text
            base, moves = split_position(text)
            prev = game_state.get(eng)
            if prev is None or prev[0] != base or moves[: len(prev[1])] != prev[1]:
                game_no[eng] = game_no.get(eng, 0) + 1
            game_state[eng] = (base, moves)
            continue
        sm = SPLIT_RE.search(text)
        if sm:
            current_split[eng] = {
                "mcts": int(sm.group(1)),
                "ab": int(sm.group(2)),
                "workers": int(sm.group(3)),
                "coordinator": int(sm.group(4)),
                "total": int(sm.group(5)),
            }
            continue
        fm = FALLBACK_RE.search(text)
        if fm:
            bm = BUDGET_RE.search(text)
            rec = {
                "source": path,
                "engine": eng,
                "reason": "ab_fallback_time_safety",
                "gpu_active": False,
                "fallback_budget_ms": int(bm.group(1)) if bm else None,
                "fallback_text": fm.group(1)[:160],
            }
            tag_game(rec, item, eng)
            if item.get("fen"):
                rec["fen"] = item["fen"]
            elif eng in last_position:
                rec["fen"] = fen_from_position_command(last_position[eng])
            yield rec
            continue
        tm = TRACE_RE.search(text)
        if tm:
            rec = parse_trace_payload(tm.group(1))
            rec["source"] = path
            rec["engine"] = eng
            if eng in current_split:
                rec["thread_split"] = current_split[eng]
            tag_game(rec, item, eng)
            if item.get("fen"):
                rec["fen"] = item["fen"]
            elif eng in last_position:
                rec["fen"] = fen_from_position_command(last_position[eng])
            yield rec


def fen_from_position_command(cmd: str) -> Optional[str]:
    """Resolve a UCI ``position [startpos|fen ...] [moves ...]`` command to a FEN."""
    try:
        import chess  # local import keeps the parser usable without python-chess
    except ImportError:
        return None
    toks = cmd.split()
    try:
        if len(toks) >= 2 and toks[1] == "startpos":
            board = chess.Board()
            mv_idx = 2
        elif len(toks) >= 2 and toks[1] == "fen":
            end = toks.index("moves") if "moves" in toks else len(toks)
            board = chess.Board(" ".join(toks[2:end]))
            mv_idx = end
        else:
            return None
        if mv_idx < len(toks) and toks[mv_idx] == "moves":
            for m in toks[mv_idx + 1 :]:
                board.push_uci(m)
        return board.fen()
    except Exception:  # noqa: BLE001
        return None


def summarise(records: List[Dict]) -> Dict:
    reasons = Counter(r.get("reason") for r in records)
    hybrid = [r for r in records if r.get("reason") != "ab_fallback_time_safety"]
    evals = [
        r["mcts_evals"] for r in hybrid if isinstance(r.get("mcts_evals"), (int, float))
    ]
    playouts = [
        r["mcts_playouts"]
        for r in hybrid
        if isinstance(r.get("mcts_playouts"), (int, float))
    ]
    disagree = [
        r
        for r in hybrid
        if r.get("ab_move") and r.get("mcts_move") and r["ab_move"] != r["mcts_move"]
    ]
    chose_mcts = [r for r in disagree if r.get("selected") == r.get("mcts_move")]

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    return {
        "moves_total": len(records),
        "moves_hybrid": len(hybrid),
        "moves_fallback": reasons.get("ab_fallback_time_safety", 0),
        "gpu_active_fraction": (len(hybrid) / len(records)) if records else None,
        "reasons": dict(reasons),
        "mcts_evals_median": med(evals),
        "mcts_playouts_median": med(playouts),
        "disagreements": len(disagree),
        "disagreement_rate": (len(disagree) / len(hybrid)) if hybrid else None,
        "mcts_move_selected_on_disagreement": len(chose_mcts),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "input", help="fastchess/cutechess engine log, results.json, or UCI transcript"
    )
    ap.add_argument("--out", help="write JSONL records here")
    ap.add_argument("--summary", action="store_true", help="print a summary")
    args = ap.parse_args(argv)
    if args.out and not args.summary:  # stream: engine logs run to gigabytes
        with open(args.out, "w") as fh:
            for r in iter_records(args.input):
                fh.write(json.dumps(r) + "\n")
        return 0
    records = parse(args.input)
    if args.out:
        with open(args.out, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
    print(json.dumps(summarise(records), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
