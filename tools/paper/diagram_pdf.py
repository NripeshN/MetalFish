#!/usr/bin/env python3
"""Export diagram-design HTML diagrams to vector PDF for the paper's LaTeX build.

For each input file the installed diagram-design skill's `scripts/export_svg.py --system-fonts`
extracts the standalone diagram SVG (CSS carried in, IDs namespaced, no network); the SVG is then
printed to a one-page PDF by headless Chromium (Playwright) at exactly its viewBox size, 96 px per
inch, so a figure drawn on the paper profile's 624-px canvas is 6.5 in wide and is included at
`\\includegraphics[width=\\linewidth]` without rescaling. Text stays text (font subsets embedded) and
paths stay vector.

Runs in the diagram tooling environment, not the measurement one:
    ~/.venvs/diagram-design/bin/python tools/paper/diagram_pdf.py docs/paper_v2/diagrams/*.html \\
        --out docs/paper_v2/figures
The skill directory is taken from $DIAGRAM_DESIGN_SKILL_DIR or the newest installed plugin version.
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import re
import subprocess
import sys
import tempfile


def skill_dir() -> pathlib.Path:
    env = os.environ.get("DIAGRAM_DESIGN_SKILL_DIR")
    if env:
        return pathlib.Path(env)
    pattern = os.path.expanduser(
        "~/.claude/plugins/cache/diagram-design/diagram-design/*/skills/diagram-design"
    )
    found = sorted(
        glob.glob(pattern),
        key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"[./]", p)],
    )
    if not found:
        sys.exit(
            "diagram-design skill not found; install the plugin or set DIAGRAM_DESIGN_SKILL_DIR"
        )
    return pathlib.Path(found[-1])


def viewbox(svg_text: str) -> tuple[float, float]:
    root = re.search(r"<svg\b[^>]*>", svg_text, re.S)
    m = root and re.search(
        r'viewBox="\s*([-\d.]+)[\s,]+([-\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)\s*"',
        root.group(0),
    )
    if not m:
        raise ValueError("diagram SVG has no numeric viewBox")
    return float(m.group(3)), float(m.group(4))


def export_one(
    html: pathlib.Path, out_pdf: pathlib.Path, skill: pathlib.Path, page
) -> tuple[float, float]:
    with tempfile.TemporaryDirectory() as tmp:
        svg_path = pathlib.Path(tmp) / (html.stem + ".svg")
        subprocess.run(
            [
                sys.executable,
                str(skill / "scripts" / "export_svg.py"),
                str(html),
                str(svg_path),
                "--system-fonts",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        svg = svg_path.read_text(encoding="utf-8")
    w, h = viewbox(svg)
    # Fix the drawing size to the viewBox (1 user unit = 1 CSS px) and print one page of that size.
    svg = re.sub(r"<svg\b", f'<svg width="{w:g}" height="{h:g}"', svg, count=1)
    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
        "@page{margin:0} html,body{margin:0;padding:0;background:#fff} svg{display:block}"
        f"</style></head><body>{svg}</body></html>"
    )
    page.set_viewport_size({"width": int(round(w)), "height": int(round(h))})
    page.set_content(html_doc, wait_until="load")
    page.evaluate("document.fonts.ready")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    page.pdf(
        path=str(out_pdf),
        width=f"{w:g}px",
        height=f"{h:g}px",
        print_background=True,
        margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        page_ranges="1",
    )
    return w, h


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("html", nargs="+", help="diagram-design HTML files")
    ap.add_argument("--out", required=True, help="output directory for the PDFs")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    skill = skill_dir()
    out = pathlib.Path(args.out)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            for src in args.html:
                html = pathlib.Path(src)
                pdf = out / (html.stem + ".pdf")
                w, h = export_one(html, pdf, skill, page)
                print(
                    f"{pdf}: {w / 96:.2f} x {h / 96:.2f} in ({pdf.stat().st_size} bytes)"
                )
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
