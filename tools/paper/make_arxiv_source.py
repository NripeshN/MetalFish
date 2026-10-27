#!/usr/bin/env python3
"""Assemble and test-compile the arXiv source package of the manuscript
(docs/paper_v2/paper/ARXIV_CHECKLIST.md, "Source upload").

Run after a normal build (docs/paper_v2/paper/build.sh), which leaves main.bbl next to main.tex and
the generated figures and tables in docs/paper_v2/figures/ and docs/paper_v2/tables/. Copies into
<out>/src/: main.tex with its \\graphicspath and \\input paths rewritten to figures/ and tables/,
PRIMEarxiv.sty, main.bbl (arXiv does not run BibTeX reliably), and exactly the figures and tables
that main.tex references. Then compiles a scratch copy with pdflatex only, as arXiv does, and fails
on a LaTeX error, a missing file, or an undefined reference or citation; on success it packs
<out>/arxiv_source.tar.gz.

--final hard-codes \\finaltrue in the packaged main.tex, so any \\pending{} left is a compile error
(the arXiv-ready check). Without it the package is a dry run that shows pending content in red.

Usage:
    python3 tools/paper/make_arxiv_source.py [--final] [--out docs/paper_v2/paper/arxiv]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
PAPER = REPO / "docs" / "paper_v2" / "paper"
FIGURES = REPO / "docs" / "paper_v2" / "figures"
TABLES = REPO / "docs" / "paper_v2" / "tables"

GRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
TABLE_INPUT = re.compile(r"\\input\{\.\./tables/([^}]+)\}")
LOG_PROBLEMS = re.compile(
    r"(LaTeX Warning: (Reference|Citation) .* undefined"
    r"|There were undefined (references|citations)"
    r"|! LaTeX Error: File .* not found"
    r"|Package natbib Warning: Citation .* undefined)"
)


def package(final: bool, src: pathlib.Path) -> list[str]:
    """Write the flat source tree to `src`; returns the list of packaged paths."""
    tex = (PAPER / "main.tex").read_text()
    figures = sorted(set(GRAPHICS.findall(tex)))
    tables = sorted(set(TABLE_INPUT.findall(tex)))
    missing = [f"figures/{f}" for f in figures if not (FIGURES / f).exists()]
    missing += [f"tables/{t}" for t in tables if not (TABLES / t).exists()]
    for name in ("PRIMEarxiv.sty", "main.bbl"):
        if not (PAPER / name).exists():
            missing.append(name)
    if missing:
        raise SystemExit(
            "missing (run docs/paper_v2/paper/build.sh first): " + ", ".join(missing)
        )
    if tex.count("\\graphicspath{{../figures/}}") != 1:
        raise SystemExit("main.tex: expected one \\graphicspath{{../figures/}}")
    tex = tex.replace("\\graphicspath{{../figures/}}", "\\graphicspath{{figures/}}")
    tex = tex.replace("\\input{../tables/", "\\input{tables/")
    if final:
        if len(re.findall(r"^\\finalfalse$", tex, flags=re.M)) != 1:
            raise SystemExit("main.tex: expected one \\finalfalse line")
        tex = re.sub(r"^\\finalfalse$", r"\\finaltrue", tex, flags=re.M)
    if src.exists():
        shutil.rmtree(src)
    (src / "figures").mkdir(parents=True)
    (src / "tables").mkdir()
    (src / "main.tex").write_text(tex)
    shutil.copy2(PAPER / "PRIMEarxiv.sty", src / "PRIMEarxiv.sty")
    shutil.copy2(PAPER / "main.bbl", src / "main.bbl")
    for f in figures:
        shutil.copy2(FIGURES / f, src / "figures" / f)
    for t in tables:
        shutil.copy2(TABLES / t, src / "tables" / t)
    return (
        ["main.tex", "main.bbl", "PRIMEarxiv.sty"]
        + [f"figures/{f}" for f in figures]
        + [f"tables/{t}" for t in tables]
    )


def test_compile(src: pathlib.Path) -> tuple[int, list[str]]:
    """pdflatex only (no BibTeX), rerun until the cross-references settle; returns (pages, problems)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp) / "src"
        shutil.copytree(src, work)
        log = ""
        for _ in range(3):
            run = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=work,
                capture_output=True,
                text=True,
                errors="replace",
            )
            log = (work / "main.log").read_text(errors="replace")
            if run.returncode != 0:
                tail = [line for line in log.splitlines() if line.startswith("!")][:5]
                return 0, ["pdflatex failed"] + tail
            if "Rerun to get cross-references right" not in log:
                break
        problems = sorted({m.group(0) for m in LOG_PROBLEMS.finditer(log)})
        pages = re.search(r"Output written on main\.pdf \((\d+) pages?", log)
        return (int(pages.group(1)) if pages else 0), problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--final", action="store_true", help="\\pending{} becomes a compile error"
    )
    ap.add_argument("--out", default=str(PAPER / "arxiv"))
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    src = out / "src"
    files = package(args.final, src)
    pages, problems = test_compile(src)
    if problems:
        print("arXiv source does not compile cleanly:")
        for p in problems:
            print("  " + p)
        return 1
    tarball = out / "arxiv_source.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for rel in files:
            tar.add(src / rel, arcname=rel)
    size = tarball.stat().st_size / 1024
    mode = "final" if args.final else "dry run (pending content shown in red)"
    print(f"{tarball}: {len(files)} files, {size:.0f} KiB, {pages} pages, {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
