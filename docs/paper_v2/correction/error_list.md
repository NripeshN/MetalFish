# Error list for the Springer Correction — each chapter statement with its corrected statement and evidence

Chapter: Niketan, N., "Hybrid Chess Search: Combining Alpha-Beta and Monte Carlo Tree Search on
Unified Memory Architectures", IntelliSys 2026, LNNS vol. 2078, pp. 653–664,
DOI 10.1007/978-3-032-32726-0_39. Quotations are verbatim from the author's camera-ready PDF
(`chapter_camera_ready.pdf`, 13 pp.; "p." below is the camera-ready page; the published chapter
runs pp. 653–664 with Springer's pagination). The item numbers (E1–E14) match
`correction_note.md` and the annotations in `chapter_annotated.pdf`. Audit IDs refer to
`../01_published_paper_audit.md`. Commit `ec1dee5` (2026-05-11) is the engine the chapter
described; `365f36a` is the frozen engine source of the follow-up paper (tag `paper-v2-freeze-rc1`; the
binary was re-linked from the same source as `paper-v2-freeze-rc2`).

Verdicts: **Wrong** (contradicted by code or records), **Unsupported** (no record exists; the
measurement cannot be reproduced), **Misleading** (true words, wrong inference), **Omission**.

---

### E1 — Elo ratings and derived statements · Wrong (method) / Unsupported (data)

- **Where**: Abstract (p. 1); Sect. 1 contribution 3 (p. 2); Sect. 3.1 (p. 3); Sect. 5.4 Table 5
  caption and Fig. 2 (p. 9); Sect. 5.5 Table 6 (p. 10); Sect. 6.1–6.2 (p. 10); Sect. 7 (p. 11).
- **As published**: "the alpha-beta engine achieved an estimated rating of ∼4048 Elo with zero
  losses in 110 games, while the hybrid engine reached ∼3724 Elo, competitive with engines such
  as Berserk (∼3722) and Lc0 (∼3700)"; "Elo estimates: AB ∼4048, Hybrid ∼3724, MCTS ∼3289";
  Table 6 "AB vs. Hybrid 3-2-0 … +241", "AB vs. MCTS 5-0-0 … >+400", "Hybrid vs. MCTS 5-0-0 …
  >+400"; "The hybrid approach adds ∼435 Elo over pure MCTS".
- **Corrected**: all ratings, the ΔElo column, Fig. 2 and the "∼435 Elo" statements are
  withdrawn. Method: per match, Elo difference = −400·log10(1/p − 1) from a 5-game score p,
  clamped to ±400 at p ∈ {0, 1}; rating = mean over anchor opponents of (assumed anchor rating +
  difference). Anchors were assumed constants (`expected_elo`: Stockfish 3800, Berserk 3722,
  Patricia 3415, Lc0 3700; Stockfish skill levels 5–16 at 2800–3584 as shown in Fig. 2), not
  measurements. Game records were not retained.
  Replacement: the pre-registered hybrid-vs-AB(8) measurement (RQ1a, 30+0.3, 1,000 pairs) —
  **pending**; interim same-binary readings on the frozen engine: L14 +13.9 ± 30.3 Elo over 101
  games at 30+0.3 (`results/paper_v2/sprt/L14_hybrid_lean_vs_ab8_30p03/partial_stats.json`).
- **Evidence**: `ec1dee5:tools/run_tournament_live.py` lines 326–330 (clamped logistic
  difference) and 591–605 (anchor average); `ec1dee5:tools/engines_config.json` (`expected_elo`
  constants for the named engines; the skill-level anchors the chapter's Fig. 2 shows, levels
  5–16 at 2800–3584, are not in that file); no PGN/JSON of the 322-game run anywhere in the
  repository (audit A1, A4);
  Lc0 BT4 Metal throughput `results/paper_v2/micro/nn_latency.json` (165–725 nps). Audit A1–A6.

### E2 — Provenance of the alpha-beta engine · Wrong (overstated) / Misleading

- **Where**: Sect. 2.1 (p. 2); Sect. 3.1 (p. 3); Sect. 5.2 (p. 7); Sect. 1 contribution 1 (p. 2).
- **As published**: "Our alpha-beta engine derives from the Stockfish codebase, inheriting its
  search infrastructure while adding GPU integration and hybrid coordination capabilities";
  "implements iterative deepening with principal variation search, null-move pruning, late move
  reductions, killer move heuristics, and a 512MB transposition table"; "The 100% agreement
  between MetalFish-AB and Stockfish validates our alpha-beta implementation."
- **Corrected**: the AB engine is Stockfish's search and NNUE evaluation (dev `c27c1747` of
  2026-01-03 at the chapter's commit; Stockfish 19 in the frozen engine, `bench` = 2,497,913
  nodes = upstream signature). Eight local edits only (root-order hints from MCTS, keep-last-PV
  on unproven fail-high, AArch64 `yield`, two `[[unlikely]]`, thread QoS, `searchmoves` dedup,
  `TT::is_allocated()`, NEON `SqrClippedReLU`). No GPU integration in the AB search; no killer
  heuristic. The 100% BK agreement is expected, not a validation. All MetalFish-AB strength
  statements are statements about Stockfish.
- **Evidence**: token-level diff vs `reference/stockfish` (audit C1); `../03_codebase_changes.md`
  §0 and C0 (Stockfish 19 port, bench signature); `results/paper_v2/calibration/build_final.log`;
  `results/ab_vs_sf/match.log` (0W-0L-22D vs Stockfish dev at 20+0.5). Audit C1, B2.

### E3 — MCTS node size · Wrong

- **Where**: Abstract (p. 1); Sect. 1 contribution 2 (p. 2); Sect. 4.1 (pp. 5–6); Sect. 6.3 (p. 11).
- **As published**: "reducing MCTS node size from 128 to 64 bytes"; "Reduced sizeof(Node) from
  128 to 64 bytes, fitting two nodes per M-series cache line (128 bytes) instead of one";
  "os_unfair_lock's 4-byte footprint allows two MCTS nodes per 128-byte cache line, doubling
  cache density compared to std::mutex-based implementations".
- **Corrected**: on Apple Silicon `Node` is `alignas(128)` with `static_assert(sizeof(Node) <=
  128)`, hence exactly one 128-byte node per cache line, already at the chapter's commit. 64-byte
  nodes exist only in non-Apple builds. Sect. 6.3 item (3) withdrawn.
- **Evidence**: `ec1dee5:src/mcts/node.h` lines 48–51 (`CACHE_LINE_SIZE = 128` under
  `__APPLE__`) and 96 (`class alignas(CACHE_LINE_SIZE) Node`); `ec1dee5:src/mcts/search.cpp`
  line 28 (`static_assert(sizeof(Node) <= 128)`); same at `365f36a` (node.h 55/102,
  search.cpp 39–48). Audit C7.

### E4 — 74× multi-threaded throughput · Unsupported

- **Where**: Abstract (p. 1); Sect. 1 contribution 2 (p. 2); Table 1 (p. 5); Sect. 4.1 (p. 6);
  Sect. 7 (p. 11).
- **As published**: "improved multi-threaded MCTS throughput from 2 to 148 NPS at 2 threads, a
  74× improvement"; Table 1 row "os_unfair_lock in Node · 2 NPS · 148 NPS · 74× (2T)"; "for a
  74× multi-threaded speedup".
- **Corrected**: the 2 NPS baseline was a lock-convoy defect in an unpreserved development
  build; no record of the measurement exists; the 74× figure and the "speedup" phrasing are
  withdrawn. `os_unfair_lock` is still used for `FinalizeScoreUpdate`/`MakeTerminal`. Other
  Table 1 rows (+7%, +16%, "cache cleanup") are single runs without variance. The hybrid runs one
  MCTS worker by default, so multi-threaded MCTS throughput is not exercised in the evaluated mode.
- **Evidence**: `src/mcts/node.h` (`os_unfair_lock` use); absence of the "before" code and of
  any benchmark record (audit B6, B7); hybrid default AB 8 threads + 1 MCTS worker
  (`../08_evidence_map.md` §A, L5/L9 rows; `../03_codebase_changes.md` §0). Audit B6, B7, C9.

### E5 — NN cache figures · Wrong / Stale

- **Where**: Table 1 (p. 5); Sect. 4.2 (p. 6); Sect. 6.2 (p. 11).
- **As published**: "Shrink NNCache (3.5GB → 38MB) … 92× memory"; "Reducing this to 96
  (covering 99.99% of positions) and the cache from 2M to 500K entries reduced memory from 3.5GB
  to 38MB"; "our full-history key produces unique keys for every path (0% cache hit rate)".
- **Corrected**: entry = 8 B generation + 8 B key + 3 floats + `wdl[3]` + flag + 96 ×
  (`uint16_t` move + `float` policy = 8 B) + `uint16_t` count ≈ 820 B. 500,000 entries ≈ 0.41 GB
  (not 38 MB); 2,000,000 × 218 moves ≈ 3.6 GB; ratio ≈ 9× (not 92×). Full-history key was true
  at the chapter's commit; current-position key since `8a376d2` (2026-05-15,
  `MCTSCacheHistoryLength = 0`). Current defaults: `MCTSNNCacheSize` 2,000,000 (≈1.65 GB) for
  stand-alone MCTS; hybrid capped at 100,000 (C18).
- **Evidence**: `ec1dee5:src/mcts/backend_adapter.h` lines 24 (`kDefaultEntries = 500000`),
  32–49 (`Entry` layout), 40 (`MAX_MOVES = 96`); `ec1dee5:src/mcts/search_params.h` line 98
  (`nn_cache_size = 500000`); `365f36a:src/uci/engine.cpp` lines 224–225;
  `../03_codebase_changes.md` C18; vmmap footprints in `../08_evidence_map.md` §A. Audit C8, B7.

### E6 — Transposition table and NNUE networks · Wrong

- **Where**: Sect. 3.1 (p. 3).
- **As published**: "a 512MB transposition table"; "dual NNUE networks (a 125MB large network
  and a 6MB small network) with HalfKAv2 feature encoding".
- **Corrected**: `Hash` UCI option, default 16 MB (512 MB was the tournament setting of
  Sect. 5.1). Networks at the chapter's commit: `nn-c288c895ea92.nnue` = 108,919,594 B
  (FullThreats + HalfKAv2_hm, L1 = 1024) and `nn-37f18f62d772.nnue` = 3,519,630 B (HalfKAv2_hm,
  L1 = 128). Frozen engine: single `nn-1a298aa575a0.nnue` = 98,511,183 B (Stockfish 19).
- **Evidence**: `ec1dee5:src/uci/engine.cpp` line 77 (`"Hash", Option(16, …)`);
  `ec1dee5:src/eval/evaluate.h` lines 25–26; `ec1dee5:src/eval/nnue/nnue_architecture.h`
  lines 28–29; file sizes `ls -l networks/`; `365f36a:src/eval/evaluate.h` line 25. Audit C2, C3.

### E7 — Position classifier and Fig. 1 · Wrong

- **Where**: Sect. 3.3 (p. 5); Algorithm 1 line 2 (p. 5); Fig. 1 and caption (p. 4); Sect. 6.4
  (p. 11).
- **As published**: "The position classifier analyzes tactical features (checks, captures, piece
  tension) to adjust the decision weighting. In tactical positions, the alpha-beta result is
  preferred; in quiet strategic positions, the MCTS evaluation carries more weight."; Fig. 1
  "UCI Protocol → Position Classifier → {Alpha-Beta, MCTS, Hybrid}", caption "The UCI layer
  routes positions to one of three engines"; "The hybrid engine's decision logic uses a simple
  position classifier".
- **Corrected**: `mcts_weight`, `ab_weight`, `ab_override_threshold` are computed in
  `src/hybrid/classifier.{h,cpp}` and read nowhere else in `src/` (only in `tests/`), at
  `ec1dee5` and at `365f36a`; the decision does not branch on position type. The classifier
  only (i) lengthens the time budget when its multiplier > 1 and (ii) sets `min_mcts_nodes` for
  the agreement stop. Mode selection is by UCI options (`UseHybridSearch`, `UseMCTS`), not by
  the classifier; Fig. 1's routing arrows are wrong.
- **Evidence**: `git grep -nE 'mcts_weight|ab_weight|ab_override_threshold' ec1dee5 -- src`
  → hits only in `src/hybrid/classifier.{h,cpp}` (21 + 3) and `tests/test_hybrid.cpp` (2);
  identical result at `365f36a`; `src/uci/engine.cpp` mode options. Audit C4, C12.

### E8 — Agreement-based early termination · Unsupported

- **Where**: Algorithm 1 lines 8–9 (p. 5); Sect. 6.1 (p. 10).
- **As published**: "if AB and MCTS agree on best move for 3+ iterations then Terminate early
  (10–40% time savings)"; "(1) agreement-based early termination … saving 10–40% of allocated
  time".
- **Corrected**: the stop exists but requires a clock search, `agreement_count ≥ 3` consecutive
  coordinator polls (~1.5 ms apart), elapsed > 25% of budget, MCTS playouts ≥ max(100,
  `min_mcts_nodes`), AB depth ≥ 10. It fired 0 of 209 times in the July 2026 telemetry
  (median 45 playouts < 100); in the current engine it fires rarely (3 of the 3,278 moves with
  the transformer active in the October 2026 30+0.3 pilot; also in some gate matches). No
  time-saving measurement exists; "10–40%" withdrawn.
- **Evidence**: `ec1dee5:src/hybrid/hybrid_search.cpp` lines 640–702 (`agreement_count`,
  gate at 695); `results/tournament_20260701_212250/results.json` parsed with
  `tools/paper/parse_hybrid_trace.py` (0 agreement stops; `../08_evidence_map.md` §C); current
  engine: "engines agree, stopping early" lines in `results/paper_v2/RQ1a_pilot/engines.log.gz`
  (3) and `results/paper_v2/sprt/*/engines.log.gz`; message at
  `365f36a:src/hybrid/hybrid_search.cpp` 4077–4081. Audit C5.

### E9 — Hybrid–Lc0 agreement inference; Bratko–Kopec as a ranking · Misleading

- **Where**: Sect. 5.2 (p. 7); Sect. 6.1 (p. 10); Tables 2–3 (pp. 7–8).
- **As published**: "The 95.8% agreement between the hybrid engine and Lc0 suggests the
  hybrid's decision-making is heavily influenced by its neural network component."; "The 95.8%
  bestmove agreement between the hybrid and Lc0 on the Bratko-Kopec suite suggests the neural
  network component has significant influence on move selection."
- **Corrected**: final move = AB's move in 207/209 hybrid decisions (July 2026, 20+0.2; MCTS
  move chosen 2/209; reasons: engines_agree 131, ab_root_rejects_mcts 49, ab_default 27,
  tie-break 2). Current engine at 30+0.3: MCTS move chosen in 0 of 1,165 disagreements (RQ1a
  pilot, build 8777b99) and 0 times in the frozen build's 101-game gate (L14); re-check against
  the final RQ1a cell before filing. The BK suite
  is at ceiling (2026-06-30 run at 5 s: AB 20, MCTS 24, Hybrid 24, Lc0 18, SF 20) and was used as
  a tuning target (~1,500 BK/puzzle runs under `results/`), so Tables 2–3 cannot rank engines.
- **Evidence**: `results/tournament_20260701_212250/results.json` (+ `parse_hybrid_trace.py`);
  `results/paper_v2/sprt/L14_hybrid_lean_vs_ab8_30p03/` (`engines.log.gz`, `partial_stats.json`)
  and `results/paper_v2/RQ1a_pilot/summary.json`; `results/paper_summary.md` (2026-06-30).
  Audit B1, B3.

### E10 — MCTS throughput vs Lc0 · Wrong / Unsupported

- **Where**: Sect. 5.3 and Table 4 (p. 8); Sect. 6.2 (p. 10).
- **As published**: "MetalFish achieves 1.26× higher average throughput, with the largest
  advantage in middlegame positions (5.5×)"; Table 4 "Complex middlegame 77 14 5.50×",
  "Average 172 136 1.26×"; "our throughput is 1.26× higher at 2 threads".
- **Corrected**: 4 positions × 10 s; Lc0 at 14 nps is a harness fault. Sweep of 2026-10-13
  (no game running; build 8777b99, whose network code is identical to the frozen engine's), same
  BT4 network: MetalFish MPSGraph FP16 b1 7.40 ms = 135 pos/s, b2 224, b4 326, b8 407, b16 446,
  b32 531, b64 538 pos/s; Lc0 0.32.1 Metal `benchmark` (search nodes/s) mb1 165, mb2 163,
  mb4 189, mb8 294, mb16 606, mb32 725, mb64 530. MetalFish is slower at batch 1, the setting it
  plays with (0.82×), and at each program's best batch size (538 vs 725, 0.74×); the two tools
  count different things (network evaluations vs search nodes), so the intermediate sizes, where
  MetalFish's raw rate exceeds Lc0's search rate, are not compared. The chapter's advantage is
  not supported.
- **Evidence**: `results/paper_v2/micro/nn_latency.json` (fields `metalfish[*].median_ms`,
  `median_positions_per_second`; `lc0[*].nps_steady_median`); `../06_environment_and_baselines.md`
  (2026-10-07 sweep, consistent). Audit B4.

### E11 — M1 Ultra figures and 7.24× scaling · Unsupported

- **Where**: Sect. 3.1 (p. 3); Sect. 3.2 (p. 4).
- **As published**: "9.8M NPS on M1 Ultra, demonstrating near-linear scaling (7.24× on 8
  threads)"; "achieving ∼382 NPS on M1 Ultra's 64 GPU cores".
- **Corrected**: no record exists for any M1 Ultra figure or for 7.24×; withdrawn. On M2 Max
  (2026-10-13, build 8777b99, which differs from the frozen engine only in an 8-line option
  default in `src/uci/uci.cpp`), depth 16, Hash 1024, median of 5: 1T 1,532,277 nps; 8T
  11,810,163 (7.71×); 12T 13,518,150. (The abstract's "7.6M nodes per second on 8 threads" is an in-game average and is
  not corrected.)
- **Evidence**: `results/paper_v2/micro/ab_scaling.json` (`aggregate` medians,
  `speedup_vs_first`). Audit B5, C11.

### E12 — Related-work characterisations · Wrong

- **Where**: Sect. 1 (p. 2); Sect. 2.4 (p. 3).
- **As published**: "Unlike previous hybrid approaches that use one paradigm to inform the other
  sequentially [9]"; "Baier and Winands [9] surveyed MCTS enhancements including alpha-beta
  rollouts. Lanctot et al. [10] combined MCTS with minimax in general game playing."
- **Corrected**: Baier & Winands (JAIR 62, 2018) embed shallow minimax inside MCTS selection /
  rollout / backpropagation (not sequential, not a survey). Lanctot et al. (CIG 2014) evaluate
  implicit minimax backups on Kalah, Breakthrough and Lines of Action (not GGP).
- **Evidence**: the cited papers themselves; verified entries in `../05_related_work.md`
  (baier2018, lanctot2014). Audit D1, D2.

### E13 — Time manager not used by the hybrid · Misleading

- **Where**: Sect. 3.2 (p. 4); Sect. 4.3 (p. 6).
- **As published**: "We ported Lc0's smooth time manager … (52% initial, capped at 73%), and a
  piggybank mechanism that saves 12%" (presented as part of the system used in the evaluation).
- **Corrected**: μ = 45.2, σ = 5.93, 0.73 cap and 12% piggybank hold for the stand-alone MCTS
  mode; the hybrid gives its MCTS component a fixed `movetime` budget and does not use this
  manager.
- **Evidence**: `ec1dee5:src/mcts/search.cpp` lines 389–390, 437, 483; hybrid fixed budget
  `src/hybrid/hybrid_search.cpp` (audit C10, lines 586–612 at `b935e66`). Audit C10.

### E14 — Omitted time-safety fallback · Omission

- **Where**: Sect. 3.3 (p. 4–5); affects all hybrid results.
- **As published**: nothing; the hybrid is described as always running both searches.
- **Corrected**: the transformer is disabled (plain AB move) when `movetime` or remaining time
  < 5,000 ms at the chapter's commit; frozen engine: `TransformerLowTimeFallbackMs` 3000 and
  `TransformerMinMoveBudgetMs` 400. GPU active on 51.8% of hybrid moves at 20+0.2 (1,437 moves),
  ~50% at 30+0.3, ≈0% at 10+0.1.
- **Evidence**: `ec1dee5:src/uci/uci.cpp` lines 59, 821–841; `365f36a:src/uci/engine.cpp`
  lines 186–187; telemetry rows in `../08_evidence_map.md` §A (L5, RQ1a pilot, L14) and §C.
  Audit C12.

---

Not corrected (hold or out of scope): the existence of the three modes and the UCI front end;
`os_unfair_lock` in `Node`; 128-byte-aligned `ABSharedState`/`MCTSSharedState` (the
"acquire/release … eliminating lock contention" wording is overstated — audit C6 — but is not
a factual error of the kind a Correction addresses); the Lichess deployment; the Declaration on
Generative AI (already present in the chapter).
