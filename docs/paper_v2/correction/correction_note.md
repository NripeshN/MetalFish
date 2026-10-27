# Correction to: Hybrid Chess Search: Combining Alpha-Beta and Monte Carlo Tree Search on Unified Memory Architectures

**Correction to:** Chapter 39 in: K. Arai, P. Lorenz (eds.), *Intelligent Systems and Applications.
IntelliSys 2026*, Lecture Notes in Networks and Systems, vol. 2078, Springer, Cham, pp. 653–664.
https://doi.org/10.1007/978-3-032-32726-0_39

> Status: **draft for an author-initiated Correction** (prepared 2026-10-14, revised 2026-10-27
> after the manuscript's integrity check: items 8–10). Fields in double
> braces are filled before filing: `{{RQ1A_RESULT}}` (from `results/paper_v2/RQ1a/summary.json` when the cell completes),
> `{{PREPRINT}}` (arXiv identifier of the follow-up paper), `{{ZENODO_DOI}}`. Nothing else
> in this note depends on pending work; every other number cites an existing artifact
> (see `error_list.md`).

The author has re-examined the chapter against the engine's source code at the commit the
chapter described (repository commit `ec1dee5`, 11 May 2026) and against the retained
measurement records, and has found that the following statements in the original version of
the chapter are incorrect or unsupported. They are corrected or withdrawn as stated below.
The software exists and runs in the three modes the chapter describes; the corrections
concern how its components and its results were characterised.

1. **Elo ratings and every statement derived from them** (Abstract; Sect. 1, contribution 3;
   Sect. 3.1; Sect. 5.4, Table 5 caption and Fig. 2; Sect. 5.5, Table 6; Sects. 6.1, 6.2 and 7).
   The ratings "∼4048 Elo" (MetalFish-AB), "∼3724 Elo" (MetalFish-Hybrid) and "∼3289 Elo"
   (MetalFish-MCTS), the comparison "competitive with engines such as Berserk (∼3722) and
   Lc0 (∼3700)", the ΔElo column of Table 6 ("+241", ">+400"), Fig. 2, and the statements
   "The hybrid approach adds ∼435 Elo over pure MCTS" and "The ∼435 Elo gap between the hybrid
   and pure alpha-beta" are withdrawn. They were computed by averaging, over each engine's
   opponents, the opponent's assumed rating plus the logistic Elo difference implied by a
   five-game match score, with the difference clamped to ±400 for 5–0 scores. The assumed
   ratings of the Stockfish skill-level opponents were never measured, and the CCRL ratings
   used for Berserk, Patricia and Lc0 refer to other hardware and time controls; in particular,
   Lc0 with the BT4 network on the Apple GPU evaluates roughly 160–700 positions per second,
   far below the hardware behind its CCRL rating. The procedure is not a valid rating estimate
   and no rating for any of the three modes should be inferred from the chapter. The game
   records of the 322-game tournament were not retained, so the win/draw/loss counts in
   Table 5 cannot be independently verified. A controlled measurement of the hybrid mode
   against the engine's own eight-thread alpha-beta mode under a pre-registered protocol
   (30 s + 0.3 s per game, colour-swapped opening pairs, pentanomial statistics) is reported in
   {{PREPRINT}}: {{RQ1A_RESULT}}.

2. **Provenance of the alpha-beta engine** (Sect. 2.1; Sect. 3.1; Sect. 5.2; Sect. 1,
   contribution 1). "Our alpha-beta engine derives from the Stockfish codebase, inheriting its
   search infrastructure while adding GPU integration and hybrid coordination capabilities"
   should read: the alpha-beta engine *is* Stockfish's search and NNUE evaluation (Stockfish
   development code of 3 January 2026 at the time of the chapter; Stockfish 19 in the current
   release, whose `bench` node count equals the upstream signature 2,497,913). The local
   changes are eight small edits: root-move ordering hints received from the MCTS component,
   keeping the previous principal variation on an unproven fail-high, an AArch64 spin-wait
   hint, two branch-likelihood hints, thread quality-of-service, de-duplication of
   `searchmoves`, a transposition-table allocation check, and a NEON kernel. There is no GPU
   integration in the alpha-beta search, and the search has no "killer move heuristics".
   Consequently the "100% bestmove agreement" with Stockfish reported in Sect. 5.2 does not
   "validate our alpha-beta implementation"; it follows from the two searches being the same
   program, and every statement about the strength of MetalFish-AB is a statement about
   Stockfish on this hardware. Contribution 1 ("achieving competitive strength across all
   three modes") is withdrawn insofar as it rests on the ratings of item 1.

3. **MCTS node size** (Abstract; Sect. 1, contribution 2; Sect. 4.1; Sect. 6.3). "reducing
   node size from 128 to 64 bytes", "Reduced sizeof(Node) from 128 to 64 bytes, fitting two
   nodes per M-series cache line (128 bytes) instead of one", and item (3) of Sect. 6.3
   ("os_unfair_lock's 4-byte footprint allows two MCTS nodes per 128-byte cache line, doubling
   cache density") are incorrect. On Apple Silicon the node class is declared with 128-byte
   alignment and a static assertion that its size does not exceed 128 bytes, so each node
   occupies exactly one 128-byte cache line; this was already the case when the chapter was
   written. A 64-byte node exists only in non-Apple builds. Item (3) of Sect. 6.3 is withdrawn.

4. **The 74× throughput improvement** (Abstract; Sect. 1, contribution 2; Table 1; Sect. 4.1;
   Sect. 7). The measurement "2 NPS → 148 NPS, 74×" cannot be reproduced: the 2 NPS baseline
   was a lock-convoy defect in a development build that was not preserved, not a
   representative implementation, and no measurement record exists. The figure "74×" and the
   phrase "a 74× multi-threaded speedup" are withdrawn; `os_unfair_lock` remains in use for
   per-node score updates. The remaining rows of Table 1 are single-run measurements without
   repetition or variance and should be read as engineering notes, not results. In the
   released configuration the hybrid runs one MCTS worker thread, so multi-threaded MCTS
   throughput is not exercised in the mode the chapter evaluates.

5. **Neural-network cache figures** (Table 1; Sect. 4.2; Sect. 6.2). "Shrink NNCache
   (3.5GB → 38MB), 92× memory" and "reduced memory from 3.5GB to 38MB" are incorrect. With
   96 cached moves per entry each entry occupies about 820 bytes, so the 500,000-entry cache
   of the chapter's engine occupied about 0.41 GB, not 38 MB, and the reduction from the
   previous 218-move, 2,000,000-entry layout (about 3.6 GB) is about 9×, not 92×. The
   statement in Sect. 6.2 that the full-history key gave a 0% hit rate described the key in
   use at the time; since commit `8a376d2` (15 May 2026) the key covers the current position
   only. In the current release the default cache holds 2,000,000 entries (about 1.65 GB) in
   the stand-alone MCTS mode and 100,000 entries in the hybrid mode.

6. **Transposition table and NNUE networks** (Sect. 3.1). "a 512MB transposition table" and
   "dual NNUE networks (a 125MB large network and a 6MB small network) with HalfKAv2 feature
   encoding" are incorrect. The transposition-table size is the UCI `Hash` option, whose
   default is 16 MB; 512 MB was the setting used in the tournament (Sect. 5.1), not an engine
   property. The networks shipped with the chapter's engine were Stockfish's
   `nn-c288c895ea92.nnue` (108.9 MB; FullThreats and HalfKAv2_hm features, 1024-wide first
   layer) and `nn-37f18f62d772.nnue` (3.5 MB; HalfKAv2_hm, 128-wide first layer). The current
   release uses the single Stockfish 19 network `nn-1a298aa575a0.nnue` (98.5 MB).

7. **Position classifier and Fig. 1** (Sect. 3.3; Algorithm 1, line 2; Fig. 1; Sect. 6.4).
   "The position classifier analyzes tactical features (checks, captures, piece tension) to
   adjust the decision weighting. In tactical positions, the alpha-beta result is preferred;
   in quiet strategic positions, the MCTS evaluation carries more weight" is incorrect. The
   classifier computes weighting fields, but no code outside the classifier and its unit tests
   reads them, in the chapter's engine or in the current release, and the final-move decision
   does not depend on the position class. The classifier only scales the per-move time budget
   and sets the minimum MCTS node count required before the agreement-based stop may fire.
   Fig. 1 shows the UCI layer routing positions through the classifier to one of the three
   engines; in fact the engine mode is selected by UCI options (`UseHybridSearch`, `UseMCTS`)
   and the classifier plays no part in routing. The sentence in Sect. 6.4 ("The hybrid
   engine's decision logic uses a simple position classifier") is corrected accordingly.

8. **Agreement-based early termination** (Algorithm 1, lines 8–9; Sect. 6.1). "If AB and
   MCTS agree on best move for 3+ iterations → Terminate early (10–40% time savings)" and
   "saving 10–40% of allocated time" are unsupported. The stop exists but is gated: clock
   games only; three consecutive coordinator polls (about 1.5 ms apart, not search
   iterations); more than a quarter of the budget elapsed; at least 100 MCTS playouts; and
   alpha-beta depth of at least 10. In the game telemetry retained from the chapter's period it
   never fired (0 of 209 hybrid decisions at 20 s + 0.2 s, July 2026; median 45 MCTS playouts
   per move); in the current engine it fires rarely (on 3 of the 3,278 moves with the
   transformer active in a 100-game 30 s + 0.3 s pilot, October 2026). No measurement of a time
   saving exists; the "10–40%" figure is withdrawn.

9. **Influence of the neural-network component** (Sect. 5.2; Sect. 6.1; Tables 2 and 3).
   "The 95.8% agreement between the hybrid engine and Lc0 suggests the hybrid's
   decision-making is heavily influenced by its neural network component" is an incorrect
   inference. In the retained telemetry the hybrid's final move was the alpha-beta move in
   207 of 209 hybrid decisions (July 2026, 20 s + 0.2 s), and in a 100-game pilot of the current
   engine at 30 s + 0.3 s (October 2026) the MCTS move was selected in none of the 1,165 decisions
   in which the two searches disagreed. The agreement with
   Lc0 on the Bratko–Kopec positions reflects agreement between Stockfish's search and Lc0 on
   those positions. The 24-position suite is also at ceiling for these engines (a later run at
   5 s per position gave AB 20, MCTS 24, Hybrid 24, Lc0 18 and Stockfish 20 of 24) and had been
   used while tuning the hybrid's and the MCTS engine's thresholds; Tables 2 and 3 should not
   be read as a ranking.

10. **MCTS throughput relative to Lc0** (Sect. 5.3, Table 4; Sect. 6.2). "MetalFish achieves
    1.26× higher average throughput, with the largest advantage in middlegame positions (5.5×)"
    and "our throughput is 1.26× higher at 2 threads" are withdrawn. Table 4 rests on four
    positions of 10 s each, and the Lc0 value of 14 nodes per second in the "Complex
    middlegame" row indicates a harness fault rather than engine behaviour. Measured on an
    Apple M2 Max with no game running and the same BT4 network (13 October 2026, with the
    engine's network code as later frozen), MetalFish's Metal inference runs at 135 positions/s
    at batch size 1 (7.4 ms per evaluation, FP16), the setting it plays with, and at best
    538 positions/s (batch size 64), whereas a search with the Metal backend of Lc0 0.32.1
    reaches 165 nodes/s at minibatch 1 and at best 725 nodes/s (minibatch 32). MetalFish is
    therefore slower at batch size 1 and at each program's best batch size; the two tools count
    different things (network evaluations against search nodes), so no comparison is made at
    the intermediate sizes. The chapter's throughput advantage is not supported.

11. **M1 Ultra figures and thread scaling** (Sect. 3.1; Sect. 3.2). No measurement record
    exists for "9.8M NPS on M1 Ultra", "near-linear scaling (7.24× on 8 threads)" or
    "∼382 NPS on M1 Ultra's 64 GPU cores"; these figures are withdrawn. On the Apple M2 Max the
    current release searches 1.53 million nodes/s on one thread, 11.81 million on eight
    threads (7.7×) and 13.52 million on twelve threads (fixed depth 16, median of five runs,
    1 GB hash).

12. **Characterisation of related work** (Sect. 1; Sect. 2.4). Baier and Winands [9] did not
    "survey MCTS enhancements including alpha-beta rollouts", and their hybrids do not "use one
    paradigm to inform the other sequentially": they embed shallow minimax searches within the
    MCTS selection, rollout and backpropagation phases. Lanctot et al. [10] did not combine
    MCTS with minimax "in general game playing"; their implicit-minimax backups were evaluated
    on Kalah, Breakthrough and Lines of Action.

13. **Time management within the hybrid** (Sect. 3.2; Sect. 4.3). The Lc0-style smooth time
    manager is used by the stand-alone MCTS mode only. Within the hybrid, the MCTS component
    receives a fixed per-move budget set by the coordinator.

14. **Omitted time-safety rule** (Sect. 3.3). The chapter does not state that the hybrid
    disables its transformer component, and plays the alpha-beta move, whenever the remaining
    clock time or the fixed move time falls below a threshold (5 s in the chapter's engine;
    3 s of remaining time or a move budget below 0.4 s in the current release). At 20 s + 0.2 s
    the transformer was active on 52% of the hybrid's moves and at 30 s + 0.3 s on about 50%;
    at 10 s + 0.1 s it is almost never active. Results attributed to the hybrid at short time
    controls are therefore largely results of the alpha-beta component.

The parts of the chapter that stand are the existence and architecture of the three modes
sharing one binary, the use of `os_unfair_lock` for node score updates and of 128-byte-aligned
shared root state, the smooth time-manager parameters of the stand-alone MCTS mode, and the
Metal (MPSGraph, FP16) inference path. A new evaluation of the same system under a
pre-registered protocol, with all game records, measurement scripts and the engine at a tagged
commit, is reported in {{PREPRINT}} and deposited at {{ZENODO_DOI}}.

The author regrets these errors.
