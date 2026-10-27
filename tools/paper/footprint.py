#!/usr/bin/env python3
"""M4 (protocol section 6): memory footprint of each engine configuration on the frozen binary.

Each configuration (AB(8), Hybrid(8) and MCTS(1) by default) is started with the options the cells
use: its engine definition in docs/paper_v2/experiments.json plus Threads, Hash and Move Overhead
as tools/paper/run_cell.py sets them. After `isready` the physical footprint is read with
`vmmap -summary` ("Physical footprint" and its peak). The engine then searches a fixed position for
--movetime ms while `ps` samples its resident set, and the footprint is read again. Writes JSON and
exits 1 if any reading failed. Run only with the machine otherwise idle (no cell running).

Usage:
    python3 tools/paper/footprint.py --out results/paper_v2/micro/footprint_frozen.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_cell  # noqa: E402

# "Kiwipete", a standard middlegame test position with many legal moves
FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
CONFIGS = {"AB(8)": ("AB", 8), "Hybrid(8)": ("Hybrid", 8), "MCTS(1)": ("MCTS1", None)}
UNITS = {"B": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


def parse_size(text: str) -> Optional[int]:
    m = re.match(r"\s*([\d.]+)\s*([BKMG])", text)
    return int(float(m.group(1)) * UNITS[m.group(2)]) if m else None


def vmmap_footprint(pid: int) -> Dict[str, Optional[int]]:
    out = subprocess.run(
        ["vmmap", "-summary", str(pid)], capture_output=True, text=True, timeout=120
    ).stdout
    cur = re.search(r"^Physical footprint:\s+(\S+)", out, re.M)
    peak = re.search(r"^Physical footprint \(peak\):\s+(\S+)", out, re.M)
    return {
        "bytes": parse_size(cur.group(1)) if cur else None,
        "peak_bytes": parse_size(peak.group(1)) if peak else None,
    }


def rss_bytes(pid: int) -> Optional[int]:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return int(out) * 1024 if out.isdigit() else None


def configure(name: str, threads: Optional[int]) -> Tuple[List[str], Dict[str, str]]:
    """The command and UCI options of an engine definition, set up as the cells set it up."""
    manifest = json.loads(run_cell.MANIFEST_PATH.read_text())
    spec = run_cell.resolve_engine(manifest["engines"], name)
    cmd = run_cell.subst(spec["cmd"])
    if not pathlib.Path(cmd).is_absolute():
        cmd = str(run_cell.REPO / cmd)
    opts = {k: run_cell.subst(str(v)) for k, v in spec.get("options", {}).items()}
    if threads is not None:
        opts["Threads"] = str(threads)
    opts.setdefault("Hash", str(manifest["defaults"]["hash_mb"]))
    opts.setdefault("Move Overhead", str(manifest["defaults"]["move_overhead_ms"]))
    return [cmd] + [run_cell.subst(a) for a in spec.get("args", [])], opts


class Engine:
    def __init__(self, cmd: List[str]):
        self.p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.lines: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.lines.put(line.strip())

    def send(self, line: str):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def wait_for(self, prefix: str, timeout: float) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                break
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"no '{prefix}' within {timeout:.0f} s")

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.p.kill()
            self.p.wait()
        for f in (self.p.stdin, self.p.stdout):
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass


def measure(cmd: List[str], opts: Dict[str, str], movetime_ms: int) -> Dict:
    eng = Engine(cmd)
    try:
        eng.send("uci")
        eng.wait_for("uciok", 60)
        for k, v in opts.items():
            eng.send(f"setoption name {k} value {v}")
        eng.send("isready")
        eng.wait_for("readyok", 600)
        time.sleep(2)  # let the allocations settle
        ready = vmmap_footprint(eng.p.pid)
        eng.send(f"position fen {FEN}")
        eng.send(f"go movetime {movetime_ms}")
        rss: List[int] = []
        deadline = time.time() + movetime_ms / 1000 + 120
        while True:
            r = rss_bytes(eng.p.pid)
            if r:
                rss.append(r)
            try:
                if eng.wait_for("bestmove", 0.5):
                    break
            except TimeoutError:
                if time.time() > deadline:
                    raise
        searched = vmmap_footprint(eng.p.pid)
        return {
            "after_isready": ready,
            "after_search": searched,
            "max_rss_bytes": max(rss) if rss else None,
            "rss_samples": len(rss),
        }
    finally:
        eng.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--movetime", type=int, default=20000, help="search time in ms")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args(argv)
    res: Dict = {
        "position": FEN,
        "movetime_ms": args.movetime,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "configs": {},
    }
    failed = 0
    for label in args.configs.split(","):
        name, threads = CONFIGS[label]
        cmd, opts = configure(name, threads)
        entry: Dict = {"engine": name, "cmd": cmd, "options": opts}
        try:
            entry["sha256"] = hashlib.sha256(
                pathlib.Path(cmd[0]).read_bytes()
            ).hexdigest()
            entry.update(measure(cmd, opts, args.movetime))
        except Exception as exc:  # noqa: BLE001
            entry["error"] = repr(exc)
        ok = entry.get("after_isready", {}).get("bytes") and entry.get("max_rss_bytes")
        failed += not ok
        res["configs"][label] = entry
        print(label, json.dumps({k: v for k, v in entry.items() if k != "options"}))
    res["hardware"] = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
    ).stdout.strip()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    if failed:
        print(
            f"{failed} configuration(s) without a reading; see {args.out}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
