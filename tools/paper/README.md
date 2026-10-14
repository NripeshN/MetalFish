# tools/paper — measurement tooling for the paper rewrite

All scripts are standalone Python 3 (python-chess required for PGN/EPD/engine access) and
are pinned by `tests/test_paper_tools.py`. Protocol: `docs/paper_v2/02_evaluation_protocol.md`;
experiment manifest: `docs/paper_v2/experiments.json`.

| Script | Purpose |
| --- | --- |
| `run_cell.py` | Run one experiment cell with fastchess (fair-resource options, UHO openings, blocks with disjoint openings, manifest with hashes/versions/hardware, thermal log, `summary.json`). `--dry-run` prints the command. |
| `penta_stats.py` | Pentanomial statistics from a PGN: Elo ± 95% CI, nElo, LOS, draw ratios, pairs ratio, WL/DD, terminations; trinomial comparison. Reproduces fastchess's `-report penta=true` numbers. |
| `parse_hybrid_trace.py` | Hybrid telemetry (`HybridTrace:` and time-safety fallback lines) from fastchess/cutechess logs or `results.json` into per-move JSONL with the decision-time FEN; summary of reasons, GPU-active fraction, disagreements. |
| `oracle_probe.py` | Judge AB vs MCTS moves on disagreement positions with reference engines (Stockfish 19, optionally Lc0); loss per candidate, oracle-arbiter upper bound, bootstrap CIs. |
| `suite_runner.py` | EPD suites (bm/am) and Lichess puzzle samples for any UCI engine; Wilson intervals; per-rating-band puzzle accuracy. |
| `microbench.py` | `nn`: BT4 latency/throughput vs batch size (MetalFish FP16/FP32 and Lc0); `ab`: `bench` NPS vs thread count. |
| `annotate_chapter.py` | Highlights every corrected passage of the published chapter's camera-ready PDF with the corrected statement (from `docs/paper_v2/correction/annotations.json`); exits non-zero if an anchor is missing. Needs PyMuPDF. |
| `make_bib.py` | Builds `docs/paper_v2/paper/refs.bib` from the verified reference map (Crossref for DOIs, arXiv API for arXiv IDs, `refs_manual.bib` for web/software/notes); exits non-zero if any citable row has no entry; `--offline` uses the cached fetches. |
| `make_deposit.py` | Assembles the Zenodo deposit directory (cells, gates, microbenchmarks, oracle study, plan package, tools; logs gzipped; CHECKSUMS.sha256; zenodo.json metadata). Nothing is uploaded; `--dry-run` lists the plan. |

Calibration artifacts (fastchess self-match of Stockfish 19 used to validate `penta_stats.py`)
live under `results/paper_v2/calibration/` (gitignored).
