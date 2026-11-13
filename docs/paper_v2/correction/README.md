# Springer Correction package for the IntelliSys 2026 chapter

Author-initiated Correction of Niketan, N., "Hybrid Chess Search: Combining Alpha-Beta and Monte
Carlo Tree Search on Unified Memory Architectures", in K. Arai, P. Lorenz (eds.), *Intelligent
Systems and Applications. IntelliSys 2026*, Lecture Notes in Networks and Systems, vol. 2078,
Springer, Cham, pp. 653–664, DOI [10.1007/978-3-032-32726-0_39](https://doi.org/10.1007/978-3-032-32726-0_39)
(published online 2026-08-14; print year 2027). Basis: `../01_published_paper_audit.md`.

**Status (2026-11-01): draft, not filed.** Filing is outward-facing and is the author's action;
see `support_ticket.md` for the exact steps. Before filing, fill the two remaining fields in
`correction_note.md`:

| Field | Source | State |
| --- | --- | --- |
| `{{LNNS_VOLUME}}` | SpringerLink chapter page, read 2026-10-27: LNNS vol. 2078 | filled |
| `{{RQ1A_RESULT}}` | `results/paper_v2/RQ1a/summary.json` (final reading 2026-10-31, `08_evidence_map.md` §E): −4.3 Elo, 95% CI −10.7 to +2.0, over 2,002 games (corrected 2026-11-01 from −4.4, which had been rounded twice; `08_evidence_map.md` §E "Rounding corrections") | filled |
| `{{PREPRINT}}` | arXiv identifier of the follow-up paper (Phase B) | pending |
| `{{ZENODO_DOI}}` | Zenodo deposit of `results/paper_v2` (Phase A, after the cells) | pending |

Everything else in the note is final and cites an existing artifact. Items 8–10 were revised on
2026-10-27 after the manuscript's integrity check: the throughput comparison with Lc0 is now
stated per batch size, and the early-stop and arbiter statements are scoped to the records they
come from. Items 8 and 9 were re-checked against the complete RQ1a cell on 2026-11-01 and now
cite it (the agreement stop fired on 1.0% of the moves with the transformer active; the MCTS move
was chosen on 7 of 25,138 disagreements); the picture is unchanged. RQ1b was complete on
2026-11-03, and the note was checked against it and against the frozen build's microbenchmark
on 2026-11-10: item 8 now gives both cells (the stop fired on 1.0% of the moves with the
transformer active at 30 s + 0.3 s and on 48.6% at 60 s + 0.6 s); item 9 is unchanged (in RQ1b
the MCTS move was chosen on 1 of 17,338 disagreements); item 10 gives the frozen build's rates
and no longer compares MetalFish's evaluation rate with Lc0's search rate, a comparison the new
evaluation withdraws as not like for like; item 1 calls Lc0's rate on the Apple GPU a search
rate. Item 14 gained the RQ1c-R result on 2026-11-09. On 2026-11-11 item 11 was given the
frozen engine's search rates (1.47, 12.78 and 15.06 million nodes/s on 1, 8 and 12 threads) in
place of the sweep of 13 October, item 14 the transformer-enabled share of the complete
30 s + 0.3 s cell (53%), and item 8 the coordinator's polling interval as the source has it
(0.5 ms, so three polls span about 1.5 ms).

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
wrong; the agreement stop never fired in the July 2026 telemetry and the "10–40%" saving is
unsupported; the hybrid's final
move was the alpha-beta move in 207/209 logged decisions; two related-work characterisations
are corrected; the hybrid does not use the Lc0-style time manager; and the omitted time-safety
rule is stated. The controlled replacement measurement (hybrid vs AB(8)) is reported in the
follow-up preprint.
