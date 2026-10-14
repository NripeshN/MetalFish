# arXiv submission checklist (preprint of the evaluation paper)

Fill-in fields are the author's; everything else is prepared by the build.

| Item | Value / action |
| --- | --- |
| Primary category | cs.AI (Artificial Intelligence); cross-list cs.LG |
| Title | from `main.tex` (candidate 1 of `04_paper_blueprint.md` §4 unless the final numbers favour another wording; outcome-neutral) |
| Authors | Nripesh Niketan (ORCID 0009-0008-2066-1937); affiliation field: "Independent researcher" (or the author's current affiliation) |
| Abstract | ≤ 1,920 characters, plain text (no LaTeX macros except simple math); written after RQ1a/RQ1b with the headline Elo ± CI |
| Comments field | e.g. "N pages, M figures, K tables. Supersedes the measurements of the IntelliSys 2026 chapter (DOI 10.1007/978-3-032-32726-0_39), whose errors are corrected in Section 8. Data and code: Zenodo DOI …" |
| Journal-ref / DOI fields | leave empty (this is a new paper, not the chapter) |
| License | author's choice at submission: arXiv non-exclusive license (minimum) or CC BY 4.0 (recommended for an evaluation paper with released data) |
| Source upload | tarball of `main.tex`, `PRIMEarxiv.sty`, `main.bbl` (arXiv runs pdflatex but not BibTeX reliably — include the `.bbl`), `../figures/*.pdf` copied next to `main.tex` (adjust `\graphicspath`), `../tables/*.tex` copied likewise; no `\pending{}` left (`build.sh --final` must pass) |
| Ancillary files (optional) | none; the Zenodo deposit holds the data |
| Anonymity | not applicable on arXiv; the double-anonymous variant is for IEEE ToG only (`build.sh --anon`) |
| Before upload | `bash build.sh --final` passes; every `%src:` comment resolves to a file in the Zenodo deposit; `08_evidence_map.md` lists every number; `05_related_work.md` has no unverified row; the Correction is filed or its status is stated in Section 8 |
| After the arXiv ID exists | fill `{{PREPRINT}}` in `docs/paper_v2/correction/correction_note.md`; add the arXiv ID to the repository README; cite the preprint from the journal submission |
