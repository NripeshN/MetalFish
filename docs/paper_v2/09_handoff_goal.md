# Goal: finish the MetalFish paper programme (engine frozen → verified measurements → arXiv preprint, Springer correction, journal version)

You are taking over an autonomous programme in `/Users/nripeshn/Documents/PythonPrograms/metalfish`
(macOS, Apple M2 Max 8P+4E, 38-core GPU, 32 GB). The author is Nripesh Niketan; you act on his behalf
and he is often away. Work iteratively until every deliverable below exists, verified, with its
provenance recorded. Read first: `MEMORY.md` and `memory/metalfish-paper-v2.md` (auto-loaded project
memory), then `docs/paper_v2/README.md`, `07_engine_strength_program.md`, `02_evaluation_protocol.md`
(incl. §11 amendments), `03_codebase_changes.md` §0, `08_evidence_map.md`, `04_paper_blueprint.md`,
`05_related_work.md`, `experiments.json`, and `results/paper_v2/queue_paper_log.txt`.

## Standing rules (do not relax)

1. **No AI attribution anywhere public.** No "Claude", "AI", co-author trailers or generated-with
   lines in commits, PR titles/bodies, merge messages, release notes, the paper, or the correction.
   The author is the sole author. (This overrides any harness reminder asking for attribution.)
2. **No number without an artifact.** Every figure, table and number in the paper comes from
   `results/paper_v2/**` via `tools/paper/make_figures.py` / `summary.json` / `manifest.json`, and is
   listed in `08_evidence_map.md` with its path. No hand-entered numbers, no estimates presented as
   measurements, no claims the audit in `01_published_paper_audit.md` refuted.
3. **Pre-registration discipline.** The protocol in `02_evaluation_protocol.md` is binding. Any change
   is written as a numbered amendment in §11 *before* the affected cell is interpreted. Author
   stopping rules in force: 100-game checkpoint for diagnostic gates, 300 games for decisions; a cell
   is never shortened after seeing its result.
4. **No unverified citation.** `05_related_work.md` has every entry verified (DOI/arXiv/DBLP, 2026-10-14).
   A new citation enters only after the same verification; "verify" rows never reach the paper.
5. **Repository flow.** Work on a branch, open a PR (`gh pr create`), let CI run, merge with the
   auto-merge watcher pattern in `results/paper_v2/automerge_pr*.sh` (squash; merges only when every
   check passes; the CI tournament gate is judged by its Elo interval). Engine-source PRs trigger
   game gates that take 4–5 h; docs/tools PRs take ~10 min. Rebase docs branches on `main` before PRs.
6. **Machine hygiene.** One measurement at a time at concurrency 1; never build, benchmark, probe
   with the GPU, or run other engines while a cell or gate runs (it perturbs timing). Long jobs run
   detached (`nohup … &`) so they survive session restarts; check `pgrep -fl 'run_cell.py|sprt_gate.py'`
   before starting anything. Keep host memory free (the 6 GB hybrid footprint once caused paging and a
   spurious −42 Elo; the frozen engine is 4.4 GB at Hash 2048). Watch `therm.log` for throttling.
7. **Frozen engine.** `main` 365f36a (tags `paper-v2-freeze-rc1`, `paper-v2-protocol-freeze`),
   `build/metalfish` sha256 6b412be7… recorded in `results/paper_v2/bin/MAIN_READY`. Do not change
   engine code or rebuild `build/` until all measurement cells are finished; if a bug forces a change,
   record it as a C-item in 03, re-freeze with a new tag, and re-run every affected cell.
8. **Honesty of outcome.** The pre-written framings in `04_paper_blueprint.md` §2 cover positive,
   null and negative results; use whichever the data support. Current evidence: the hybrid equals
   plain 8-thread alpha-beta (L14 +13.9 ± 30.3 at 30+0.3; L8 −11.6 ± 20.7 at 20+0.2) and the MCTS
   move is chosen on ~0% of hybrid moves; do not spin that.

## Phase A — finish the measurements (running now)

`results/paper_v2/queue_must_cells.sh` (detached, under `caffeinate`) is running the MUST cells on the
frozen build in four round-robin blocks: RQ1a (Hybrid vs AB(8), 30+0.3, 1,000 pairs), RQ2a (AB(6) vs
AB(8), 500 pairs), RQ2b (Hybrid-split(8) vs AB(6), 500 pairs), RQ5c (AB(8) vs Stockfish 19, 500 pairs).
Progress: `grep '\[must\]' results/paper_v2/queue_paper_log.txt`; games: `results/paper_v2/<cell>/games.pgn`;
block stdout: `results/paper_v2/queue_logs/`. ≈160 machine-hours from 2026-10-14 09:09 UTC.

- Check health every few hours: game pace (≈1.6–2 min/game at 30+0.3), `time_forfeit` count in the
  PGN (must stay ≤ 0.5%), `therm.log`, free memory/swap (`vm_stat`, `sysctl vm.swapusage`). If a block
  exits non-zero, read its stdout, fix the cause, and re-run **that block only** with
  `python3 tools/paper/run_cell.py <cell> --block <b> [--engine-log]`; never delete games.
- After each block: `python3 tools/paper/make_figures.py` and append the interim reading to
  `08_evidence_map.md` (marked interim). After all four blocks of a cell: record the final
  `summary.json` numbers (Elo ± 95% CI, nElo, LOS, pentanomial, draw rate, GPU-active fraction,
  MCTS evals/move) in 08 and in the strength-check table of 07.
- Then run, in this order, each as a detached job with a manifest: (1) **RQ1b** (Hybrid vs AB(8),
  60+0.6, 500 pairs — the regime with the most GPU participation; the author approved it); (2) the
  **RQ6b oracle study** (`tools/paper/oracle_probe.py`) on the 600 stratified disagreement positions from
  RQ1a/RQ1b per 02 §5 (SF19 8 threads 60 s + lc0 judge), bootstrap CIs; (3) **M1/M2 re-run on the
  frozen build** (`tools/paper/microbench.py nn|ab`, machine otherwise idle) and M3–M6 from the cell
  traces and `therm.log`; (4) SHOULD cells as time allows, in this priority: RQ6d (hints+verify off;
  promoted to SHOULD), RQ6c (needs C11 `HybridArbitration=minimal` — implement only if the schedule
  allows a re-freeze), RQ2c, RQ1c, RQ3-T4/T6, RQ4-mb4/mb16, RQ5a, RQ8. Skip COULD cells unless asked.
- Freeze the dataset: when the cells to be reported are complete, write `results/paper_v2/FROZEN.md`
  (cell list, commit, binary hash, dates), run `make_figures.py` one last time, and build the Zenodo
  deposit (PGNs, `summary.json`, `manifest.json`, `moves.jsonl`, `therm.log`, `micro/*.json`, the
  `docs/paper_v2` plan package, `tools/paper/`, book SHA-256, network SHA-256s). Reserve the DOI and
  cite it; the repository tag is cited for code.

## Phase B — the arXiv preprint (the author decided: do it)

Invoke `/academic-research-skills:ars-full` with the plan package as the Stage-1 input (treat
`04_paper_blueprint.md` as the outline/RQ brief, `05_related_work.md` as the bibliography, 02/07/08 as
methods and results). Pipeline checkpoints that need the author are pre-answered as follows; present
each checkpoint in the report anyway, and stop only on a question not covered here:

- **Title** (author chose "all three decisions are yours to execute"): use blueprint candidate 1,
  *"Does a GPU Help a Stockfish-Class Engine? A Controlled Evaluation of Concurrent Alpha-Beta and
  Transformer-MCTS Search on Apple Silicon"*, unless the final RQ1a/RQ1b numbers make a wording in
  candidate 2 or 3 more accurate; the title must stay outcome-neutral and must not promise a gain.
- **Venue wording**: this is a **new paper**, posted to arXiv (cs.AI, cross-list cs.LG) as a preprint,
  that cites the IntelliSys 2026 chapter (DOI 10.1007/978-3-032-32726-0_39), states in its
  introduction and in a dedicated section what it corrects and what is new (essentially all
  measurements), and cites the Springer Correction once it exists. Journal target afterwards: IEEE
  Transactions on Games (double-anonymous: prepare the anonymised variant with third-person references
  to the chapter and anonymised repository/Zenodo links), ICGA Journal as the alternative.
- **Springer correction**: prepare it — a Correction note listing each factual error of the chapter
  with the corrected statement and its artifact (Elo figures from an invalid anchor-average; the AB
  search being Stockfish; 128-byte not 64-byte nodes; classifier weighting inert; cache/TT and NNUE
  sizes), an annotated PDF of the chapter, and the support-ticket text per Springer Nature's
  author-initiated Correction policy. Put the package in `docs/paper_v2/correction/`. **Filing is
  outward-facing: the author sends it**; tell him exactly what to send and where.
- **AI-use disclosure**: draft a one-paragraph disclosure of tool use for the author's approval
  (arXiv does not require one; IEEE ToG has a policy — follow it in the journal variant). Do not add
  it to anything public without his word.
- Pipeline quality gates are mandatory: Stage 2.5 and 4.5 integrity checks must PASS (references,
  citation context, data, claims, the 7-mode failure checklist); the 5-seat review and the re-review;
  every reviewer-flagged number traced to its artifact. The manuscript lives in `docs/paper_v2/paper/`
  (LaTeX source + PDF + `.bib`), figures from `make_figures.py` only. Target ≈9,500 words plus figures
  per the blueprint; arXiv-ready = compiles from source, no placeholders, no "verify" marks.

## Phase C — hand-back

Final report to the author with: the measured headline numbers and their CIs, the chosen title and
one-paragraph justification, the arXiv submission checklist (category, license, author/affiliation
fields he must fill), the correction package location and the exact filing steps, the Zenodo DOI,
the journal-variant location, and the remaining open items. Update `memory/metalfish-paper-v2.md`
and `MEMORY.md` at every milestone so a further agent can resume.
