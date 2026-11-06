"""GPU errors in an engine's stderr (protocol amendment 15 (i)).

When Metal reports a failed command buffer, the work it held may not have run; the runners'
paravirtualised GPU did this to the oracle's Lc0 judge (a hang, then every later submission
ignored, then an abort). These are the lines that report it, as Metal and Metal Performance
Shaders print them on the engine's stderr (fastchess keeps a game engine's stderr in engines.log).
"""

from __future__ import annotations

import pathlib
import subprocess

MARKS = (
    "command buffer exited with error",
    "kIOGPUCommandBuffer",
    "MTLCommandBufferErrorDomain",
    "failed assertion",
)


def is_gpu_error(line: str) -> bool:
    return any(m in line for m in MARKS)


def count_in(path: pathlib.Path) -> int:
    """GPU error lines in a text file, or in a zstd-compressed one (.zst); 0 if it is absent."""
    path = pathlib.Path(path)
    if not path.exists():
        return 0
    if path.suffix != ".zst":
        with open(path, errors="replace") as fh:
            return sum(1 for line in fh if is_gpu_error(line))
    proc = subprocess.Popen(
        ["zstd", "-dc", str(path)], stdout=subprocess.PIPE, text=True, errors="replace"
    )
    n = sum(1 for line in proc.stdout if is_gpu_error(line))
    if proc.wait():
        raise RuntimeError(f"zstd could not read {path}")
    return n
