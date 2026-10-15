#!/usr/bin/env python3
"""Build the paper's BibTeX file from the verified reference map docs/paper_v2/05_related_work.md.

Each table row (Key | Reference | Identifier | Use | Status) becomes one entry with the row's
key. Rows with a DOI are fetched from Crossref (application/x-bibtex); rows with only an arXiv
identifier from the arXiv API. Struck-through keys (~~key~~) and rows whose status says
"do not cite" or "not located" are skipped. Rows with neither identifier (web pages, software,
notes) must be present in refs_manual.bib next to the output; manual entries also override
fetched ones (e.g. to give a conference version precedence over the arXiv record). The script
exits 1 if any citable row has no entry, so the bibliography can never silently lose a source.

Usage:
    python3 tools/paper/make_bib.py [--map docs/paper_v2/05_related_work.md]
        [--manual docs/paper_v2/paper/refs_manual.bib] [--out docs/paper_v2/paper/refs.bib]
        [--offline]   # reuse the cached fetches in docs/paper_v2/paper/.bibcache/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s|]+)")
ARXIV_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})(?:v\d+)?")
KEY_RE = re.compile(r"^@\w+\{([^,]+),", re.M)
UA = "metalfish-paper-tools/1.0 (mailto:nripesh14@gmail.com)"


def parse_map(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] in ("Key", "---"):
            continue
        key, ref, ident, use, status = cells
        skip = key.startswith("~~") or re.search(
            r"do not cite|not located", status, re.I
        )
        rows.append(
            {
                "key": key.strip("~"),
                "ref": ref,
                "ident": ident,
                "use": use,
                "status": status,
                "skip": bool(skip),
                "doi": (
                    (DOI_RE.search(ident) or [None, None])[1]
                    if DOI_RE.search(ident)
                    else None
                ),
                "arxiv": (
                    (ARXIV_RE.search(ident) or [None, None])[1]
                    if ARXIV_RE.search(ident)
                    else None
                ),
            }
        )
    return rows


def fetch(url: str, accept: str | None = None) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


PROTECT = {
    "Apple",
    "Silicon",
    "Monte",
    "Carlo",
    "Monte-Carlo",
    "Elo",
    "Lc0",
    "Stockfish",
    "AlphaZero",
    "Kalah",
    "Breakthrough",
    "Go",
    "Metal",
    "Chess",
}


def protect_title(text: str) -> str:
    """Brace tokens that BibTeX would otherwise lowercase in titles: acronyms and names."""

    def fix(m):
        inner = m.group(1)
        out = []
        for tok in inner.split(" "):
            core = tok.strip("(),.:;")
            if (
                core
                and ("{" not in tok)
                and (sum(1 for ch in core if ch.isupper()) >= 2 or core in PROTECT)
            ):
                tok = tok.replace(core, "{" + core + "}", 1)
            out.append(tok)
        return "title = {" + " ".join(out) + "}"

    return re.sub(r"title\s*=\s*\{([^{}]*)\}", fix, text, count=1)


def crossref_bibtex(doi: str, key: str) -> str:
    doi = doi.rstrip(".,;)")
    raw = fetch(
        f"https://api.crossref.org/works/{urllib.request.quote(doi)}/transform/application/x-bibtex"
    ).strip()
    raw = re.sub(r"^@(\w+)\{[^,]+,", lambda m: f"@{m.group(1)}{{{key},", raw, count=1)
    # Crossref emits month macros that unsrtnat/BibTeX do not define (june, july, sept)
    raw = re.sub(
        r"month\s*=\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*",
        lambda m: "month = " + m.group(1).lower(),
        raw,
        flags=re.I,
    )
    return protect_title(raw)


def arxiv_bibtex(arxiv_id: str, key: str) -> str:
    xml = fetch(f"https://export.arxiv.org/api/query?id_list={arxiv_id}")
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    entry = ET.fromstring(xml).find("a:entry", ns)
    if entry is None:
        raise RuntimeError(f"arXiv {arxiv_id}: no entry")
    title = " ".join(entry.findtext("a:title", default="", namespaces=ns).split())
    authors = " and ".join(
        a.findtext("a:name", default="", namespaces=ns)
        for a in entry.findall("a:author", ns)
    )
    year = entry.findtext("a:published", default="", namespaces=ns)[:4]
    cat = entry.find("x:primary_category", ns)
    primary = cat.get("term") if cat is not None else ""
    return protect_title(
        f"@misc{{{key},\n  title = {{{title}}},\n  author = {{{authors}}},\n  year = {{{year}}},\n"
        f"  eprint = {{{arxiv_id}}},\n  archivePrefix = {{arXiv}},\n  primaryClass = {{{primary}}},\n"
        f"  url = {{https://arxiv.org/abs/{arxiv_id}}}\n}}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--map", default=str(ROOT / "docs/paper_v2/05_related_work.md"))
    ap.add_argument(
        "--manual", default=str(ROOT / "docs/paper_v2/paper/refs_manual.bib")
    )
    ap.add_argument("--out", default=str(ROOT / "docs/paper_v2/paper/refs.bib"))
    ap.add_argument("--offline", action="store_true", help="use cached fetches only")
    args = ap.parse_args()

    rows = parse_map(Path(args.map))
    manual_text = (
        Path(args.manual).read_text(encoding="utf-8")
        if Path(args.manual).exists()
        else ""
    )
    manual_keys = set(KEY_RE.findall(manual_text))
    cache_dir = Path(args.out).parent / ".bibcache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries, missing, sources = [], [], {}
    for row in rows:
        key = row["key"]
        if row["skip"]:
            sources[key] = "skipped"
            continue
        if key in manual_keys:
            sources[key] = "manual"
            continue
        cache = cache_dir / f"{key}.bib"
        if cache.exists():
            entries.append(cache.read_text(encoding="utf-8"))
            sources[key] = "cache"
            continue
        if args.offline:
            missing.append(key)
            continue
        try:
            if row["doi"]:
                text = crossref_bibtex(row["doi"], key)
                sources[key] = f"crossref {row['doi']}"
            elif row["arxiv"]:
                text = arxiv_bibtex(row["arxiv"], key)
                sources[key] = f"arxiv {row['arxiv']}"
            else:
                missing.append(key)
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"fetch failed for {key}: {exc}", file=sys.stderr)
            missing.append(key)
            continue
        cache.write_text(text + "\n", encoding="utf-8")
        entries.append(text + "\n")

    header = (
        "% Generated by tools/paper/make_bib.py from docs/paper_v2/05_related_work.md.\n"
        "% Fetched entries (Crossref / arXiv) first, then the hand-maintained entries of refs_manual.bib.\n"
        "% Do not edit by hand: change 05_related_work.md or refs_manual.bib and regenerate.\n\n"
    )
    Path(args.out).write_text(
        header + "\n".join(entries) + "\n" + manual_text, encoding="utf-8"
    )
    if (
        not args.offline
    ):  # an offline build only knows "cache"; keep the record of the real sources
        (Path(args.out).parent / "refs_sources.json").write_text(
            json.dumps(sources, indent=1, sort_keys=True) + "\n"
        )
    n_fetched = sum(1 for v in sources.values() if v not in ("manual", "skipped"))
    print(
        f"{len(rows)} rows: {n_fetched} fetched/cached, {sum(1 for v in sources.values() if v == 'manual')} manual, "
        f"{sum(1 for v in sources.values() if v == 'skipped')} skipped, {len(missing)} MISSING"
    )
    for key in missing:
        row = next(r for r in rows if r["key"] == key)
        print(f"  MISSING {key}: {row['ident'][:70]} | {row['ref'][:80]}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
