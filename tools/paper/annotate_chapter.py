#!/usr/bin/env python3
"""Annotate the author's camera-ready PDF of the IntelliSys 2026 chapter with the Correction
items in docs/paper_v2/correction/annotations.json.

Every anchor (a verbatim phrase of the chapter) is highlighted on its page; the highlight and a
sticky note at the item's first anchor carry the corrected statement, so a reader at the
publisher sees, at each marked passage, what is wrong and what the corrected text is. A cover
note is placed on page 1. The script exits with status 1 if any anchor is not found, so the
annotation set is complete by construction.

Requires PyMuPDF: pip install pymupdf

Usage:
    python3 tools/paper/annotate_chapter.py --pdf docs/paper_v2/correction/chapter_camera_ready.pdf \
        --out docs/paper_v2/correction/chapter_annotated.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:  # pragma: no cover
    sys.exit("PyMuPDF is required: pip install pymupdf")

HIGHLIGHT = (1.0, 0.85, 0.2)
NOTE_FILL = (1.0, 0.95, 0.80)
NOTE_INK = (0.6, 0.0, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pdf", required=True, help="camera-ready chapter PDF to annotate")
    ap.add_argument("--out", required=True, help="annotated PDF to write")
    ap.add_argument(
        "--annotations",
        default=str(
            Path(__file__).resolve().parents[2]
            / "docs/paper_v2/correction/annotations.json"
        ),
        help="annotation spec (default: docs/paper_v2/correction/annotations.json)",
    )
    args = ap.parse_args()

    spec = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    doc = pymupdf.open(args.pdf)

    first = doc[0]
    box = pymupdf.Rect(first.rect.width - 250, 14, first.rect.width - 18, 112)
    cover = first.add_freetext_annot(
        box, spec["cover"], fontsize=6.5, text_color=NOTE_INK, fill_color=NOTE_FILL
    )
    cover.set_info(title="Correction request")
    cover.update()

    missing: list[tuple[str, int | None, str]] = []
    n_anchors = 0
    for item in spec["items"]:
        content = f"{item['id']} — {item['title']}\n\n{item['note']}"
        placed_note = False
        for anchor in item["anchors"]:
            n_anchors += 1
            page_no = anchor.get("page")
            pages = [doc[page_no - 1]] if page_no else list(doc)
            hit = None
            for page in pages:
                quads = page.search_for(anchor["text"], quads=True)
                if quads:
                    hit = (page, quads)
                    break
            if hit is None:
                missing.append((item["id"], page_no, anchor["text"]))
                continue
            page, quads = hit
            hl = page.add_highlight_annot(quads)
            hl.set_colors(stroke=HIGHLIGHT)
            hl.set_info(title=item["id"], content=content)
            hl.update()
            if not placed_note:
                rect = quads[0].rect
                point = pymupdf.Point(max(rect.x0 - 16, 2), rect.y0 - 2)
                note = page.add_text_annot(point, content, icon="Note")
                note.set_info(title=item["id"])
                note.set_colors(stroke=NOTE_INK)
                note.update()
                placed_note = True

    doc.save(args.out, garbage=3, deflate=True)
    print(
        f"wrote {args.out}: {len(spec['items'])} items, {n_anchors} anchors, "
        f"{n_anchors - len(missing)} highlighted, {len(missing)} missing"
    )
    for item_id, page_no, text in missing:
        print(f"  MISSING {item_id} p{page_no}: {text!r}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
