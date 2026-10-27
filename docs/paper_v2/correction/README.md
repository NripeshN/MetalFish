# Springer Correction package for the IntelliSys 2026 chapter

Author-initiated Correction of Niketan, N., "Hybrid Chess Search: Combining Alpha-Beta and Monte
Carlo Tree Search on Unified Memory Architectures", in K. Arai, P. Lorenz (eds.), *Intelligent
Systems and Applications. IntelliSys 2026*, Lecture Notes in Networks and Systems, vol. 2078,
Springer, Cham, pp. 653–664, DOI [10.1007/978-3-032-32726-0_39](https://doi.org/10.1007/978-3-032-32726-0_39)
(published online 2026-08-14; print year 2027). Basis: `../01_published_paper_audit.md`.

**Status (2026-10-27): draft, not filed.** Filing is outward-facing and is the author's action;
see `support_ticket.md` for the exact steps. Before filing, fill the three remaining fields in
`correction_note.md`:

| Field | Source | State |
| --- | --- | --- |
| `{{LNNS_VOLUME}}` | SpringerLink chapter page, read 2026-10-27: LNNS vol. 2078 | filled |
| `{{RQ1A_RESULT}}` | `results/paper_v2/RQ1a/summary.json` → "Elo ± 95% CI over N games" (cell running) | pending |
| `{{PREPRINT}}` | arXiv identifier of the follow-up paper (Phase B) | pending |
| `{{ZENODO_DOI}}` | Zenodo deposit of `results/paper_v2` (Phase A, after the cells) | pending |

Everything else in the note is final and cites an existing artifact. Items 8–10 were revised on
2026-10-27 after the manuscript's integrity check: the throughput comparison with Lc0 is now
stated per batch size, and the early-stop and arbiter statements are scoped to the records they
come from. Before filing, re-check items 8 and 9 against the final RQ1a/RQ1b telemetry
(`moves.jsonl`) and say so if the complete cells show a different picture.

## Files

| File | Content |
| --- | --- |
| `correction_note.md` | The Correction text in Springer's "Correction to:" form: 14 numbered items, each with the original statement, the corrected statement, and what is withdrawn. |
| `error_list.md` | Working list behind the note: for each item the chapter location (section, camera-ready page), the verbatim statement, the verdict, the corrected statement, the evidence (`commit:file:line` or `results/` artifact) and the audit ID. |
| `support_ticket.md` | Where and how to file (support portal route, publishing-editor route), the ticket fields filled in, and the COPE caveat (the publisher may choose a retraction instrument given items E1–E2). |
| `annotations.json` | Anchor phrases and notes for the annotated PDF; the single source for the highlights. |
| `chapter_camera_ready.pdf` | The author's camera-ready copy (13 pp., 2026-06-16; identical text to the published chapter, Springer pagination differs). Git-ignored (`*.pdf`); copy from `~/Downloads/metalfish.pdf` or the `paper` branch's PDF draft (`git show paper:metalfish/PDF/metalfish.pdf`, May version). |
| `chapter_annotated.pdf` | Generated: every erroneous passage highlighted with the corrected statement as its note (50 anchors, 14 sticky notes, cover note on p. 1). Git-ignored; regenerate with the command below. |

Regenerate the annotated PDF (PyMuPDF required: `pip install pymupdf`):

```bash
python3 tools/paper/annotate_chapter.py --pdf docs/paper_v2/correction/chapter_camera_ready.pdf \
    --out docs/paper_v2/correction/chapter_annotated.pdf
```

The script exits non-zero if any anchor in `annotations.json` is not found in the PDF.

## What the Correction says, in one paragraph

The Elo ratings (∼4048 / ∼3724 / ∼3289), Fig. 2, Table 6's ΔElo column and the "∼435 Elo"
statements are withdrawn (invalid anchor-average over assumed ratings; games not retained). The
alpha-beta engine is Stockfish with eight small edits, so its strength statements are about
Stockfish and the 100% agreement validates nothing. Nodes are 128 bytes, not 64; the 74×
speedup, the 1.26×/5.5× throughput advantage over Lc0, the M1 Ultra figures and the 7.24×
scaling have no records and are withdrawn; the NN-cache, transposition-table and NNUE figures
are corrected; the position classifier does not weight the decision and Fig. 1's routing is
wrong; the agreement stop never fired in the chapter-era telemetry and the "10–40%" saving is
unsupported; the hybrid's final
move was the alpha-beta move in 207/209 logged decisions; two related-work characterisations
are corrected; the hybrid does not use the Lc0-style time manager; and the omitted time-safety
rule is stated. The controlled replacement measurement (hybrid vs AB(8)) is reported in the
follow-up preprint.
