# arXiv submission checklist (preprint of the evaluation paper)

Fill-in fields are the author's; everything else is prepared by the build.

| Item | Value / action |
| --- | --- |
| Primary category | cs.AI (Artificial Intelligence); cross-list cs.LG |
| Title | from `main.tex` (candidate 1 of `04_paper_blueprint.md` §4 unless the final numbers favour another wording; outcome-neutral) |
| Authors | Nripesh Niketan (ORCID 0009-0008-2066-1937); affiliation field: "Independent researcher" (or the author's current affiliation) |
| Abstract | ≤ 1,920 characters, plain text (no LaTeX macros except simple math). The manuscript's abstract is longer (2,557 characters as plain text), so the field takes a shortened version that keeps its claims and numbers; the author chooses its wording |
| Comments field | e.g. "38 pages, 6 figures, 29 tables (counts of the build before the author's two fields; recount after `build.sh --final`). Supersedes the measurements of the author's IntelliSys 2026 chapter (LNNS vol. 2078, DOI 10.1007/978-3-032-32726-0_39), whose errors are corrected in Section 8. Data and code: Zenodo DOI …" |
| Journal-ref / DOI fields | leave empty (this is a new paper, not the chapter) |
| License | author's choice at submission: arXiv non-exclusive license (minimum) or CC BY 4.0 (recommended for an evaluation paper with released data) |
| Source upload | `python3 tools/paper/make_arxiv_source.py --final` after `bash build.sh --final` (version 2 reports every measurement, so the build reads `results/paper_v2` as it is; `results/paper_v2/FROZEN.md` lists the cells): packs `main.tex` (figure and table paths rewritten), `PRIMEarxiv.sty`, `main.bbl` (arXiv runs pdflatex but not BibTeX reliably) and exactly the figures and tables `main.tex` references into `arxiv/arxiv_source.tar.gz`, after compiling a scratch copy with pdflatex only and failing on any error, missing file, undefined reference or citation, or `\pending{}` left. Without `--final` it is a dry run. |
| Ancillary files (optional) | none; the Zenodo deposit holds the data |
| Anonymity | not applicable on arXiv; the double-anonymous variant is for IEEE ToG only (`build.sh --anon`) |
| Before upload | `bash build.sh --final` passes; every `%src:` comment resolves to a file in the Zenodo deposit; `08_evidence_map.md` lists every number; `05_related_work.md` has no unverified row; the Correction is filed or its status is stated in Section 8 |
| Generative-AI statement | arXiv's moderation policy asks authors to report significant use of text-generating AI tools; the manuscript holds a `\pending{}` statement in "Use of software tools" that the author decides before `build.sh --final` can pass |
| Correction status | the Section 8 sentence on the Correction states its status at posting (filed on a date, or prepared and not yet filed) |
| Version label | the author footnote says "Version 2" and Appendix B lists the changes from version 1, the repository's first version (pull request #135). arXiv numbers its own versions from v1, so decide before the upload whether to keep that wording if this is the first version posted there |
| After the arXiv ID exists | fill `{{PREPRINT}}` in `docs/paper_v2/correction/correction_note.md`; add the arXiv ID to the repository README; cite the preprint from the journal submission |
