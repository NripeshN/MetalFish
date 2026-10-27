# Arbiter and timing reference: the MetalFish hybrid at the frozen source (365f36a)

- Commit: `365f36a2bcfb85d3380987f2a49dd248889873f8` ("Cap the hybrid's MCTS NN cache; record the L10-L13 controls and the footprint analysis (#77)", 2026-10-14).
- Every reference is `path:line` inside `git show 365f36a:<path>` (or `git grep ... 365f36a`). Quotes are verbatim, sometimes shortened with "...". This document was compiled by reading the source; nothing in it was measured.
- "Not determinable from source" means the answer depends on runtime behaviour or on another program (fastchess ships only as a binary in `reference/fastchess/`).

## 0. Effective configuration of the "Hybrid" arm

**Set by the arm.** These come from `docs/paper_v2/experiments.json`, engine "Hybrid", plus its defaults block:
- `UseHybridSearch=true`, `UseMCTS=false`, `NNWeights=<BT4>.pb`
- `HybridMCTSThreads=1`, `HybridTrace=true`
- `Threads=8`, `Hash=2048`, `Move Overhead=50`

**Shipped defaults that matter** (`src/uci/engine.cpp`):

| Option | Default | Line |
|---|---|---|
| `HybridOversubscribe` | true | 159 |
| `HybridFollowABTime` | true | 160 |
| `HybridABPolicyWeight` | "0.0" | 161 |
| `HybridMCTSMinimumKLDGainPerNode` | "0.0" | 162 |
| `HybridABRootRejectMCTS` | true | 163 |
| `HybridMCTSRootReject` | false | 164 |
| `HybridMCTSUseSharedTT` | false | 165 |
| `HybridQToCpScale` | 300 | 168 |
| `HybridMCTSABRootHints` | true | 169 |
| `HybridMCTSABRootHintDelayMs` | 0 | 170 |
| `HybridMCTSABRootHintCount` | 8 | 171 |
| `HybridABCandidateVerifyMs` | 240 | 172 |
| `HybridABCandidateVerifyCount` | 5 | 173 |
| `HybridRootPawnLeverTieBreak` | false | 174 |
| `HybridANERootProbe` / `HybridANERootHints` / `HybridANEConfirmMCTSOverride` | false | 175-177 |
| `HybridTrace` | false (the arm sets true) | 185 |
| `TransformerLowTimeFallbackMs` | 3000 | 186 |
| `TransformerMinMoveBudgetMs` | 400 | 187 |
| `MCTSSmartPruningFactor` | "1.33" | 217 |
| `Move Overhead` | 10 (the arm sets 50) | 101 |

**Hard-coded values in `make_hybrid_config`** (`src/uci/uci.cpp`):
- `config.ab_min_depth = 10;` (1271)
- `config.use_position_classifier = true;` (1278)
- `config.decision_mode = ...DecisionMode::DYNAMIC;` (1279). No UCI option exposes the decision mode.
- The NN cache is capped at 100000 entries (1244, 1256-1257).

**UCI options that override header defaults:**
- `config.mcts_root_reject = engine.get_options()["HybridMCTSRootReject"];` (1284) is false. It overrides the struct default `bool mcts_root_reject = true;` (`hybrid_search.h:215`).
- `config.ab_candidate_verify_ms = ...["HybridABCandidateVerifyMs"]` (1295-1296) is 240. It overrides `int ab_candidate_verify_ms = 120;` (`hybrid_search.h:222`).

**Thread split** (`uci.cpp:1215-1218`): `} else if (oversubscribe) { ... ab_threads = available; }`. Alpha-beta gets all 8 threads and the MCTS worker is one extra thread (`HybridMCTSThreads=1`, clamped at 1207).

**Mode dispatch** (`uci.cpp:182-189`): `if (UseMCTS) mcts_mt_go(is); else if (UseHybridSearch) parallel_hybrid_go(is); else go(is);`

**`HybridArbitration` (used by the HybridMin arm) does not exist at 365f36a.**
- `git grep HybridArbitration 365f36a -- src` returns nothing.
- It appears only in `docs/paper_v2/03_codebase_changes.md:76`, as planned change C11.
- An unknown option only triggers `sync_cout << "No such option: " << name` (`src/uci/ucioption.cpp:50-53`).

---

## 1. The override ladder in `make_final_decision`

### 1.0 Shared quantities

All of these are in `src/hybrid/hybrid_search.cpp`.

| Symbol | Definition | Lines |
|---|---|---|
| `budget` | `mcts_decision_budget = HybridHasMCTSDecisionBudget(limits_, time_budget_ms_, ponderhit)`. For a clock `go` (no movetime, nodes, depth, mate or infinite; not pondering; budget > 0) it is **true**: `if (time_budget_ms <= 0) return false; return limits.time[WHITE] > 0 \|\| limits.time[BLACK] > 0;` (532-543). Every predicate parameter named `fixed_budget` receives this flag, so the "...FixedBudget..." predicates are live in clock games despite their names. | 4610-4612 |
| `clock` | `clock_managed_search`: movetime/nodes/depth/mate/infinite all unset and a clock is present. True in clock games. | 4613-4616 |
| R, B, S | R = this move's MCTS root visits (`mcts_total_current_nodes`, which excludes visits reused from the old tree). B = this move's visits of the MCTS best move (`mcts_current_visits`). S = B/R (`visit_share`). | 4617-4618, 4914-4918 |
| R_tot, B_tot, S_abs | The same counts including reused-tree visits; S_abs = B_tot/R_tot (`absolute_visit_share`). | 4326-4331, 4910-4913 |
| `sane` | `HybridMCTSVisitEvidenceSane(playouts, evals, R, B)`. If playouts == 0, require R == 0 and B == 0. Otherwise require B <= R and R <= playouts + max(64, playouts/32); and, if evals > 0, B <= max(512, 64*evals) and R <= max(1024, 96*evals). | 4619-4621; 1659-1678 |
| `reliable` | With `budget`: R >= 180, B >= 72 and S >= 0.24. Without it: R >= max(100, min_mcts_nodes) and B >= max(48, min_mcts_nodes/2). | 4919-4929 |
| `strong` | `reliable && (B >= 512 \|\| S >= 0.55)` | 4930-4931 |
| `overwhelming` | R >= 5000, B >= 512 and S >= 0.30 | 4932-4934 |
| cp | `QToNnueScore(mcts_q, HybridQToCpScale)` = `int(atanh(clamp(q,-0.999,0.999)) * 300)`, so \|cp\| <= 1140. | 4907; `mcts/core.h:139-142` |
| `ab_score` | Alpha-beta's last reported UCI score: WDL-normalised centipawns (`score = InternalUnits{UCIEngine::to_cp(v, pos)}`, `eval/score.cpp:21-22`). Mate becomes ±(30000 − plies) and tablebase ±(20000 − plies) (3878-3888). | 4324 |
| Δ | `eval_delta = mcts_cp - ab_score` | 4938 |
| `ab_verified` | AB completed depth >= max(ab_min_depth=10, classifier ab_verify_depth <= 6), so effectively depth >= 10. | 4935-4937 |
| `clear` | `ab_has_clear_preference = ab_verified && std::abs(ab_score) >= 15` | 4939 |
| G | `root_q_gap`: Q of MCTS top-1 minus the max Q of the other top-10 moves with current visits > 0. It is 0 if top-1 differs from the MCTS best move or fewer than 2 moves exist. | 4582-4609; 2832-2847 |
| Qg | Q(MCTS best) − Q(the AB move as listed in MCTS top-10). The pawn-push and rook predicates pass 0 when the AB move is absent. | per call |
| m-rank, m-score, m-avg, m-LB, m-UB, m-eff | Rank, score, averageScore, inexactLower, inexactUpper and effort (nodes) of the MCTS move in alpha-beta's final root list (the best thread's first 16 moves; 3973-3987; `uci/engine.cpp:568-585`). Scores are in **internal Value units**, not cp. "−∞" means `-VALUE_INFINITE` = −32001 (`core/types.h:143`): AB did not score the move exactly in its last iteration (`search.cpp:1543-1547`, "All other moves but the PV are set to the lowest value"). | 4477-4497 |
| avg gap | AB avg(AB move) − AB avg(MCTS move), in internal units | per call |
| a-rank, a-cur | Rank and current visits of the AB move among MCTS top-10. Rank is −1 if absent. | 4444-4468 |
| `veto` | `ab_root_rejects_mcts` (see §1.2) | 4946-4951 |

Unit caveat: `ab_score` and Δ are in UCI-normalised centipawns, m-score and avg are in internal Value units (about 304 internal = 100 cp at full material, `uci.cpp:620-627, 661-670`), and cp is `300*atanh(Q)`. The thresholds mix all three.

### 1.1 Branches before the disagreement ladder, in evaluation order

| # | Trace `reason` | Code | Move chosen | Condition and thresholds | UCI control (default) | Under Hybrid |
|---|---|---|---|---|---|---|
| P1 | `mcts_only_result` | 4798-4802 | MCTS | `ab_best == Move::none() && mcts_best != Move::none()` | none | enabled |
| P2 | `ab_only_result` / `fallback_legal` | 4803-4811 | AB / first legal | One engine has no move | none | enabled (never picks MCTS) |
| P3 | `root_quiet_attack_tiebreak` | 4813-4818; finder 4703-4760; `HybridRootQuietMinorMajorAttackCandidate` 2210-2241 | An AB root move (not the AB best) that is one of the first 4 MCTS hints and a quiet knight/bishop move attacking an enemy Q or R | See note A below | `HybridRootPawnLeverTieBreak` (false) | **disabled**: `if (!config_.root_pawn_lever_tiebreak \|\| ...) return Move::none();` (4704) |
| P4 | `root_policy_tiebreak` (only when `ab_best == mcts_best`) | 4820-4878; `HybridRootPolicyTieBreak` 1875-1885 | An MCTS alternative that is neither engine's best | See note B below | none | **enabled** |
| P5 | `root_q_conflict_tiebreak` (only when they agree) | 4860-4883; `HybridRootQConflictTieBreak` 1887-1914 | An MCTS alternative | See note C below | none | **enabled** |
| P6 | `root_pawn_lever_tiebreak` (agreement) | 4885-4900; finder 4627-4702 | An AB-root kingside pawn lever | Gated as in P3 | `HybridRootPawnLeverTieBreak` (false) | **disabled** (4629) |
| P7 | `engines_agree` | 4902-4904 | The common move | — | — | — |

**Note A (P3 thresholds).** Gate: `budget`, `sane`, not low-node-primary, and selected ≠ MCTS best. The AB average of the selected move must satisfy \|avg\| <= 1000. The candidate needs MCTS rank 1-2, current visits >= 4, P >= 0.10, AB effort in [1500, 10000], and AB avg gap <= 60. If the candidate has MCTS rank 1, it is rejected when the selected move has MCTS rank 1-3 and Q(sel) − Q(cand) > 0.12. If the candidate has rank 2: when the selected move is not rank 1, it needs selected rank > 2, selected visits <= 2, candidate visits >= 12 and Q(cand) − Q(sel) >= 0.08; otherwise it needs Q(sel) − Q(cand) <= 0.25 and AB avg gap <= 20.

**Note B (P4 thresholds).** Requires `budget && sane && top_count > 1` and MCTS top-1 == the AB move. For a candidate in MCTS top-10, every one of these must hold: R >= 600, top current visits >= 256, candidate current visits >= 256 and >= 82% of the top's, Q(top) − Q(cand) <= 0.05, P(cand) >= 0.30, and P(cand) >= 2.0·P(top). The highest-policy qualifying candidate wins.

**Note C (P5 thresholds).** Same gate as P4. R in [40, 90]. Top and candidate current visits >= 20 each, and the candidate's are >= 75% of the top's. Q(top) <= 0.05, Q(cand) >= 0.10 and Q(cand) − Q(top) >= 0.45. P(top) >= 0.20 and P(cand) <= 0.60·P(top). AB avg(top) − AB avg(cand) <= 60 (a candidate missing from the AB list has avg 0). The largest Q gap wins.

When P4 or P5 fire, the played move differs from both engines' best move, and the reported score is still alpha-beta's (4176-4180).

### 1.2 Veto and veto lifters

**Veto: `HybridABRootRejectsMCTS`** (1860-1873). It is gated by `HybridABRootRejectMCTS` (default true) at 4946-4951:

```
ab_root_rejects_mcts = config_.ab_root_reject_mcts &&
    HybridABRootRejectsMCTS(ab_verified, ab_in_ab.rank, mcts_in_ab.rank,
        ab_in_ab.average_score, mcts_in_ab.average_score,
        ab_in_ab.effort, mcts_in_ab.effort, mcts_in_ab.score);
```

The veto fires when all of the following hold:
- `ab_verified` (depth >= 10)
- the AB move has AB rank 1
- the MCTS move has m-rank >= 2

and at least one of:
- avg gap >= 30 (internal units) and AB-move effort >= 10000; or
- m-score <= −30000 (unscored or mated) and AB-move effort >= 10000 and AB-move effort >= 10·max(1, m-eff).

Code: `if (!ab_verified || ab_rank != 1 || mcts_rank <= 1) return false;`. As a result, an MCTS move outside alpha-beta's first 16 root moves (rank −1) is never vetoed.

**Lifters.** The veto is overridden when `mcts_override_allowed` (5288-5306) is true:

```
low_node_mcts_primary_ready || !ab_root_rejects_mcts || ane_confirmed_mcts_override ||
pawn_only_ane_mcts_override || pawn_only_mcts_override || ane_q_supported_root_override ||
mcts_short_root_tactical || mcts_verified_hint_support || mcts_compact_fixed_budget ||
mcts_compact_pawn_endgame || mcts_ab_lowerbound_confirmed || mcts_low_node_root_confidence ||
mcts_ultra_low_node_root_confidence || mcts_compact_clear_preference ||
mcts_cross_root_confidence_fixed_budget || mcts_root_confidence_reject_override ||
mcts_reused_root_confidence || mcts_root_reject_low_material_push ||
mcts_root_reject_rook_pawn_push || mcts_root_reject_kingside_pawn_push ||
mcts_rook_endgame_quiet_rook || mcts_root_reject_quiet_queen_move ||
mcts_root_reject_quiet_minor_major_attack || mcts_verified_quiet_minor_major_attack ||
mcts_mid_root_tactical_q_gap || mcts_bishop_endgame_retreat || mcts_root_reject_q_gap ||
mcts_clock_root_reject_q_gap || mcts_discovered_pawn_push_override ||
mcts_reused_root_current || mcts_root_rejects_ab || (mcts_overwhelming && eval_delta >= 250)
```

Two items only lift the veto and never select a move:
- `HybridMCTSRootConfidenceRejectOverride` (1041-1076). It requires root-confidence (L10) **and** the veto, plus m-rank >= 1, m-score −∞, no LB/UB and m-eff <= 10000. If m-rank > 3, it also needs m-rank <= 10, avg gap <= 90, a-rank in [2, 4], a-cur in [1, 16] and Qg >= 0.75. Otherwise it needs avg gap <= 120, a-rank >= 2, a-cur >= 10 and Qg >= 0.60.
- `mcts_overwhelming && eval_delta >= 250`

Selectors that do **not** lift the veto, and are therefore blocked when it fires:
- L6 `mcts_overwhelming_delta` (lifted only if Δ >= 250)
- L7 decisive, L8 tactical-gap, L9 root-dominant
- L10 root-confidence (unless the reject-override above holds)
- L35 no-clear
- L36 `mcts_reliable_ab_unverified` can never coincide with the veto, because the veto needs `ab_verified`.

**Other gates that act like vetoes:**
- `sane` is required by most predicates.
- `clear` disables L11, L29, L30, L31 and L35, and parts of L25. It is required by L22 and L23.
- The opposite-direction predicate `mcts_root_rejects_ab` (L34) is off.

**Trace-reading caveat.** All predicates are computed eagerly (4946-5287) and the recorded `reason` is the **first true selector in ladder order**. That can be a non-lifting selector (for example `mcts_decisive_fixed_budget`) whose veto was actually lifted by a later predicate. The trace prints every flag (5510-5557).

### 1.3 Disagreement ladder (DecisionMode::DYNAMIC, 5366-5480), in evaluation order

`const char *reason = "ab_default";` (5309). The **Lift** column shows whether the predicate is in the lifter set. **Needs veto** means the function returns true only when `ab_root_rejects_mcts` is true. Every rung has no individual UCI option; the global controls are listed in §1.6.

| Order | Trace `reason` (picks MCTS unless noted) | Function (definition lines) | Lift | Needs veto | Under Hybrid |
|---|---|---|---|---|---|
| L0 | `ab_root_rejects_mcts` (**AB move**) | `!mcts_override_allowed` (5368-5369) | — | — | enabled |
| L1 | `low_node_mcts_primary` | `HybridLowNodeMCTSPrimaryReady` 558-570 | yes | no | **unreachable in clock games**: needs `go nodes <= 1024` (545-550); uci.cpp:1740-1747 routes such searches to pure MCTS |
| L2 | `ane_confirmed_mcts` | `HybridANEConfirmedMCTSOverride` 1680-1795 | yes | no | **disabled** (`HybridANEConfirmMCTSOverride`=false; `if (!enabled ...) return false;` 1689) |
| L3 | `pawn_only_ane_mcts` | `HybridPawnOnlyANEMCTSOverride` 2508-2656 | yes | no | **disabled** (same option; ANE probe off; 2516) |
| L4 | `pawn_only_mcts` | `HybridPawnOnlyMCTSOverride` 2658-2736 | yes | no | enabled |
| L5 | `ane_q_supported_root` (**picks the ANE top move**) | `HybridANEQSupportedRootOverride` 1797-1849 | yes | yes | **disabled** (1807) |
| L6 | `mcts_overwhelming_delta` | inline: `sane && overwhelming && eval_delta >= 180` (5384-5387) | only if Δ>=250 | no | enabled |
| L7 | `mcts_decisive_fixed_budget` | 653-659 | no | no | enabled |
| L8 | `mcts_tactical_gap_fixed_budget` | 676-682 | no | no | enabled |
| L9 | `mcts_root_dominant_fixed_budget` | 668-674 | no | no | enabled |
| L10 | `mcts_root_confidence_fixed_budget` | 684-702 | via reject-override | no | enabled |
| L11 | `mcts_low_node_root_confidence` | 704-742 | yes | no | enabled |
| L12 | `mcts_ultra_low_node_root_confidence` | 744-778 | yes | no | enabled |
| L13 | `mcts_short_root_tactical` | 780-835 | yes | branch B only | enabled |
| L14 | `mcts_verified_hint_support` | 837-882 | yes | no | enabled (needs the verification of §3 to have run) |
| L15 | `mcts_ab_lowerbound_confirmed` | 884-925 | yes | no | enabled |
| L16 | `mcts_discovered_pawn_push` | 2071-2107 | yes | no | enabled |
| L17 | `mcts_root_reject_low_material_push` | 1078-1110 | yes | yes | enabled |
| L18 | `mcts_root_reject_kingside_pawn_push` | 1112-1152 | yes | yes | enabled |
| L19 | `mcts_root_reject_rook_pawn_push` | 1154-1210 OR 1212-1244 | yes | first part only | enabled |
| L20 | `mcts_rook_endgame_quiet_rook` | 2738-2758 | yes | no | enabled |
| L21 | `mcts_root_reject_quiet_queen_move` | 1246-1301 | yes | yes | enabled |
| L22 | `mcts_root_reject_quiet_minor_major_attack` | 1303-1365 | yes | third branch only | enabled |
| L23 | `mcts_verified_quiet_minor_major_attack` | 1367-1403 | yes | no | enabled |
| L24 | `mcts_mid_root_tactical_q_gap` | 1405-1435 | yes | yes | enabled |
| L25 | `mcts_bishop_endgame_retreat` | 1552-1657 | yes | no | enabled |
| L26 | `mcts_root_reject_q_gap` | 1936-2017 | yes | yes | enabled |
| L27 | `mcts_clock_root_reject_q_gap` | 2019-2033 | yes | yes | enabled (needs `clock`) |
| L28 | `mcts_reused_root_confidence` | 1437-1498 | yes | no | enabled (no `sane` requirement) |
| L29 | `mcts_reused_root_current` | 1500-1550 | yes | third branch only | enabled |
| L30 | `mcts_compact_fixed_budget` | 927-940 | yes | no | enabled |
| L31 | `mcts_compact_pawn_endgame` | 942-974 | yes | no | enabled |
| L32 | `mcts_compact_clear_preference` | 976-1001 | yes | no | enabled |
| L33 | `mcts_cross_root_confidence_fixed_budget` | 1003-1039 | yes | no | enabled |
| L34 | `mcts_root_rejects_ab` | `HybridMCTSRootRejectsAB` 1916-1934; outer gate 5251-5287 | yes | no | **disabled**: `HybridMCTSRootReject`=false (`if (config_.mcts_root_reject && ...` 5256) |
| L35 | `mcts_no_clear_fixed_budget` | 661-666 | no | no | enabled |
| L36 | `mcts_reliable_ab_unverified` | inline: `sane && !ab_verified && reliable && eval_delta >= -80` (5475-5479) | n/a | no | enabled (only when AB depth < 10) |
| — | `ab_default` (**AB move**) | fall-through | | | |

**After the ladder:**
- `Move selected = ane_q_supported_root_override ? ane_top : (choose_mcts ? mcts_best : ab_best); trace_final(reason, selected);` (5587-5590).
- Then `find_root_pawn_lever_tiebreak(selected, true)`, which emits a second trace line with `reason=root_pawn_lever_tiebreak` (5592-5597). It is disabled by `HybridRootPawnLeverTieBreak`=false (4629).
- Otherwise the function returns `mcts_best` if `choose_mcts`, else `ab_best` (5604-5610).
- `MCTS_PRIMARY` (5311-5359) and `AB_PRIMARY` (5360-5365) are unreachable because the mode is hard-coded to DYNAMIC (uci.cpp:1279). So their reasons (`mcts_primary_reliable`, `ab_primary_mcts_overwhelming`, ...) never appear.

### 1.4 Exact conditions of the reachable rungs

Every rung below also requires `budget` (true in clock games) unless stated otherwise. A pair written [a, b] is inclusive; [a, b) excludes b.

**L4 `HybridPawnOnlyMCTSOverride`.**
- Shape (5224-5230): a pawn-only endgame (`non_pawn_material()==0`), where the MCTS move is a pawn lever, a pawn×pawn capture, or a king recapture on the square of the AB pawn capture. Only the lever and king-recapture shapes can return true.
- Base conditions: `sane`, B_tot <= R_tot, B <= R.
- *Ultra-tiny lever:* R_tot in [20, 32], B_tot >= 20, R in [12, 16], B == R, S >= 0.98, G == 0, Qg >= 1.00, cp >= 650, Δ >= 600, a-cur == 0, a-rank == 2, m-rank == 2, m-score −∞, m-eff <= 60000.
- Otherwise it first requires \|AB avg\| <= 170, \|MCTS avg\| <= 220 and \|avg gap\| <= 160.
- *King recapture, form 1:* R_tot in [35, 50], B_tot >= 32, R >= 28, B >= 28, S >= 0.92, G >= 0.50, Qg >= 0.50, cp >= 150, Δ >= 150, a-cur <= 3, a-rank == 2, m-rank == 2, m-score −∞, m-UB, m-eff in [50000, 120000].
- *King recapture, form 2:* R_tot in [20, 32], B_tot >= 18, R >= 12, B >= 11, S >= 0.88, Qg >= 0.41, cp >= 105, Δ >= 115, a-cur <= 2, a-rank == 2, m-rank == 2, m-score −∞, m-eff in [4000, 120000].
- *Pawn lever, form 1:* R_tot in [28, 40], B_tot >= 20, R >= 28, B >= 20, S >= 0.70, G >= 0.40, Qg >= 0.40, cp >= 80, Δ >= 80, a-cur <= 3, a-rank == 3, m-rank in [2, 3], m-score −∞, m-UB, m-eff in [35000, 220000].
- *Pawn lever, form 2:* R_tot in [17, 24], B_tot >= 10, R >= 17, B >= 10, S >= 0.58, G >= 0.13, Qg >= 0.14, Δ >= −10, a-cur <= 3, a-rank >= 3, m-rank in [2, 4], m-score −∞, m-UB, m-eff in [25000, 100000].

**L6 `mcts_overwhelming_delta`.** `sane`, R >= 5000, B >= 512, S >= 0.30, Δ >= 180.

**L7 `HybridMCTSDecisiveFixedBudgetOverride`.** `sane`, `strong`, R >= 300, B >= 210, S >= 0.60, Δ >= 90.

**L8 `HybridMCTSTacticalGapFixedBudgetOverride`.** `sane`, R >= 250, B >= 55, S >= 0.16, G >= 0.12, cp >= 300, Δ >= 250.

**L9 `HybridMCTSRootDominantFixedBudgetOverride`.** `sane`, `strong`, R >= 250, B >= 200, S >= 0.74, cp >= 150, Δ >= 80.

**L10 `HybridMCTSRootConfidenceFixedBudgetOverride`.**
- Base: `sane`, `strong`, R >= 230, B >= 180, cp >= 170.
- Then any one of:
  - S >= 0.65, G >= 0.12, Δ >= 110;
  - S >= 0.74, G >= 0.06, Δ >= 70;
  - R >= 300, B >= 240, S >= 0.64, G >= 0.12, cp >= 200, Δ >= 55.

**L11 `HybridMCTSLowNodeRootConfidenceOverride`.**
- `sane`, `!clear`.
- R in [50, 180], B >= 40, S >= 0.70, G >= 0.30, cp >= 180, Δ >= 150.
- avg gap <= 35. a-rank in [2, 4], a-cur <= 12. Qg >= 0.40.
- Then one of:
  - m-rank in [2, 4], m-score −∞, no LB/UB, m-eff <= 1000;
  - m-rank in [2, 3], LB and not UB, m-score >= −50, m-eff in [50000, 250000].

**L12 `HybridMCTSUltraLowNodeRootConfidenceOverride`.**
- `sane`. R in [20, 40], B >= 20, S >= 0.74, G >= 0.55, cp >= 170, Δ >= 170.
- \|ab_score\| <= 220. avg gap <= 720. m-rank in [1, 4]. m-eff <= 400000.
- If m-score ≠ −∞, then m-score >= ab_score − 90.
- a-rank >= 2, a-cur <= 4. Qg >= 0.45.

**L13 `HybridMCTSShortRootTacticalOverride`.**
- Base: `sane`, G >= 0.16, cp >= 200, Δ >= 45, avg gap <= (veto ? 70 : 25).
- *Branch A* applies when R in [150, 230), B >= 95 and S >= 0.58. It then requires m-rank in [1, 3], m-score −∞, not LB, m-eff in [200000, 1500000], a-rank >= 4, a-cur <= 16 and Qg >= 0.24. If these fail, the predicate is false.
- *Branch B* applies only with the veto: R in [240, 360], B in [150, 260], S in [0.58, 0.72], Δ >= 50, avg gap <= 60, m-rank == 2, m-score −∞, not LB, m-eff in [1e6, 3e6], a-rank == 2, a-cur in [20, 64], Qg >= 0.20.

**L14 `HybridMCTSVerifiedHintSupportOverride`.**
- Gate: `sane` and the first verified hint == the MCTS move (`ab_verified_hint_supports_mcts`, 5002-5004). B <= R.
- R in [120, 320], B in [70, 220], S in [0.48, 0.72], G >= 0.16, cp >= 200, Δ >= 45. avg gap <= 70.
- m-rank == 2, m-score −∞, not LB, m-eff <= 2.5e6.
- Either m-UB, or the "no-bound short root" case: not UB, R in [120, 180], B in [65, 110], S in [0.48, 0.53], m-eff in [1e5, 1e6], Qg >= 0.24.
- a-rank == 2, a-cur in [20, 64]. Qg >= 0.20.

**L15 `HybridMCTSABLowerBoundConfirmedOverride`.** Requires B <= R.
- If `sane`: R in [50, 140], B >= 45, S >= 0.75, G >= 0.30, ab_score <= 500, m-rank in [2, 3], m-LB, m-score >= 250, m-score − ab_score >= 250, m-eff in [1e5, 1.5e6], a-rank > 1, a-cur <= 8.
- If not `sane`: R in [1500, 6000], B >= 1500, S >= 0.95, G >= 0.50, ab_score <= 100, m-rank in [2, 3], m-LB, m-score >= 500, m-score − ab_score >= 450, m-eff in [1e5, 1.5e6], a-rank > 1, a-cur <= 32.

**L16 `HybridMCTSDiscoveredPawnPushOverride`.**
- Shape: the MCTS move is a kingside pawn push (f-h file, 1-2 squares, non-capture; 2764-2778) that newly exposes the enemy king to a rook/bishop/queen line (2780-2809).
- `sane`. R in [18, 64], B >= 17, B <= R, a-cur <= 4, S >= 0.90.
- Qg >= 0.40, cp >= 280, Δ >= 240, \|ab_score\| <= 120. avg gap <= 50.
- m-rank in [1, 3], m-eff <= 500000.
- Then either m-score −∞, or LB and not UB with m-score >= 30.

**L17 `HybridMCTSRootRejectLowMaterialPushOverride`.**
- Gate: `sane`, veto, `non_pawn_material() <= RookValue` (5057), MCTS move is a kingside pawn push.
- R in [700, 1600], B >= 500, S >= 0.68, G >= 0.45, cp >= 140, Δ >= 130. avg gap <= 30.
- m-rank in [2, 3], m-score −∞, m-eff in [20000, 80000].
- a-rank == 2, a-cur >= 120. Qg >= 0.45.

**L18 `HybridMCTSRootRejectKingsidePawnPushOverride`.**
- Gate: `sane`, veto, kingside pawn push.
- R in [15, 22], B in [10, 20], cp >= 240, Δ >= 190.
- m-rank == 2, m-score −∞, m-eff <= 25000. a-rank == 2, a-cur <= 4.
- Then one of:
  - S in [0.60, 0.76], G >= 0.30, Qg >= 0.30;
  - R in [15, 17], B >= 8, S >= 0.50, G >= 0.24, Qg >= 0.24, cp >= 220, Δ >= 190, m-eff <= 50000;
  - S >= 0.99, G == 0, a-cur == 0, B >= 18, cp >= 270, Δ >= 215, m-eff <= 20000, Qg >= 0.32.

**L19, first part: `HybridMCTSRootRejectRookEndgamePawnPushOverride`.**
- Gate: veto; a rook endgame (rooks only besides pawns); the MCTS move is a quiet d/e-file pawn push of 1-2 squares landing on rank >= 4; B <= R.
- If `sane`, the main case is R in [45, 1600], B >= 40, S >= 0.88, G >= 0.60, cp >= 430, Δ >= 300. If that fails, it falls back to the "ultra-low breakthrough" case: R in [30, 48], B >= 28, S >= 0.84, G >= 0.66, cp >= 440, Δ >= 240, m-rank in [2, 6], m-score −∞, m-eff <= 64, a-rank in [2, 3], a-cur in [1, 2], B >= 12·a-cur, Qg >= 0.65.
- If not `sane`: R in [1500, 6000], B >= 1500, S >= 0.95, G >= 0.50, cp >= 450, Δ >= 380.
- Then in all cases: m-rank in [2, 6], m-score −∞, m-eff <= (sane ? 20000 : 2000); a-rank in [2, 3], a-cur in [1, 32]; B >= (sane ? 6 : 50)·a-cur; Qg >= 0.60.

**L19, second part: `HybridMCTSRookEndgamePawnPushLowerBoundOverride`.**
- Gate: `sane`, same shape, B <= R.
- R in [24, 64], B >= 18, S >= 0.70, G >= 0.55, cp >= 180, Δ >= 160, \|ab_score\| <= 80.
- m-rank in [2, 3], LB and not UB, m-score >= −50, m-score − ab_score >= −30, m-eff in [50000, 500000].
- a-rank == 2, a-cur in [4, 8]. Qg >= 0.50.

**L20 `HybridMCTSRookEndgameQuietRookOverride`.**
- Gate: `sane`, rook endgame, the MCTS move is a non-capture rook move.
- Both visit pairs passed are current visits (5117-5118): R in [12, 24], B >= 12, S >= 0.98.
- Qg >= 1.55, cp >= 300, Δ >= 300.
- m-rank in [4, 8], m-score −∞, m-eff <= 100.
- a-rank in [2, 3], a-cur == 0.

**L21 `HybridMCTSRootRejectQuietQueenMoveOverride`.**
- Gate: `sane`, veto; the MCTS move is a quiet queen move one file sideways on its 3rd rank, landing on the d or e file.
- Main case: R in [50, 1200], B >= 47, S >= 0.85, G >= 1.05, cp >= 330, Δ >= 330.
- Alternative "low root tactical" case: R in [7, 24], B >= 5, S >= 0.70, G >= 1.15, cp >= 190, Δ >= 230, m-rank in [3, 7], m-score −∞, not LB, m-eff <= 512, a-rank in [5, 8], a-cur == 0, Qg >= 1.00.
- Then: m-rank in [2, (low ? 7 : 6)], m-score −∞, not LB, not UB (unless low), m-eff <= 512.
- For the AB move: low case needs a-rank in [5, 8] and a-cur == 0; otherwise a-rank in [2, 4] and a-cur in [1, 4].
- B >= (low ? 1 : 12)·max(1, a-cur). Qg >= (low ? 1.00 : 1.05).

**L22 `HybridMCTSRootRejectQuietMinorMajorAttackOverride`.**
- Gate: `sane`, `clear`; the MCTS move is a quiet knight/bishop move attacking an enemy Q or R.
- *Short:* R in [120, 220], B in [65, 125], S in [0.45, 0.60], G >= 0.17, cp >= 220, Δ >= 60, m-rank in [2, 8], m-score −∞, not LB, m-eff <= 1e6, a-rank == 2, a-cur in [20, 40], Qg >= 0.24.
- *Medium:* R in [220, 450], B in [170, 320], S >= 0.55, G >= 0.08, cp >= 170, Δ >= 30, m-rank == 2, m-score −∞, not LB, m-eff <= 2.5e6, a-rank == 2, a-cur in [20, 48], Qg >= 0.18.
- *Otherwise, with the veto:* R in [55, 90], B >= 30, S >= 0.50, G >= 0.19, cp >= 180, Δ >= 75, m-rank in [5, 12], m-score −∞, not LB, m-eff <= 10000. Then either a-rank in [2, 3], a-cur <= 24 and Qg >= 0.19; or a-rank >= 4, a-cur <= 4, B >= 12·a-cur and Qg >= 0.45.

**L23 `HybridMCTSVerifiedQuietMinorMajorAttackOverride`.**
- Gate: `sane`, first verified hint == the MCTS move, `clear`, quiet knight/bishop attack on Q or R.
- R in [120, 180], B in [50, 110], S in [0.42, 0.56], G >= 0.18, cp >= 240, Δ >= 80. avg gap <= 70.
- m-rank == 2, m-score −∞, not LB, m-eff in [1e5, 7e5].
- a-rank in [2, 4], a-cur <= 32. Qg >= 0.18.

**L24 `HybridMCTSMidRootTacticalQGapOverride`.**
- Gate: `sane`, veto, B <= R.
- R in [110, 180], B >= 70, S >= 0.55, G >= 0.80, cp >= 170, Δ >= 220.
- m-rank in [2, 3], m-score −∞, no LB/UB, m-eff <= 6000.
- a-rank == 2, a-cur in [32, 64]. Qg >= 0.80.

**L25 `HybridMCTSBishopEndgameRetreatOverride`.**
- Gate: `sane`; a bishop-only endgame; the MCTS move is a quiet bishop move from rank >= 2 onto its own 1st rank. Requires B_tot <= R_tot and B <= R.
- "Barely touched" means m-score −∞, no LB/UB and m-eff <= 10000.
- Sub-case (i): not `clear`, barely touched, m-rank in [1, 4], R_tot in [80, 190], B_tot >= 30, R >= 12, B >= 12, S_abs >= 0.35, S >= 0.70, cp >= 190, Δ >= 180, a-rank in [3, 4], a-cur <= 12, Qg >= 0.30.
- Sub-case (ii): not `clear`, barely touched, m-rank in [1, 4], R_tot in [100, 130], B_tot >= 45, R >= 55, B >= 36, S_abs >= 0.43, S >= 0.58, cp >= 210, Δ >= 200, a-rank in [2, 3], a-cur <= 12, Qg >= 0.29.
- Sub-case (iii): not `clear`, m-rank in [1, 2], LB and not UB, m-score >= 140, m-eff >= 200000, R_tot in [90, 130], B_tot >= 40, R >= 50, B >= 30, S_abs >= 0.40, S >= 0.58, cp >= 220, Δ >= 210, a-rank >= 3, a-cur <= 12, Qg >= 0.38.
- Sub-case (iv), main path:
  - R_tot >= 200, B_tot >= 180, R >= 24, S_abs >= 0.75, S >= 0.94, cp >= 280, Δ >= 210, a-rank >= 2, a-cur <= 1.
  - B >= 24, unless B >= 23 with R >= 24, cp >= 360, Δ >= 360 and Qg >= 0.58.
  - Qg >= 0.50. It drops to 0.46 when the move is barely touched, m-rank <= 4, a-cur == 0, S_abs >= 0.80, S >= 0.99, cp >= 300 and Δ >= 290.
  - If `clear`: m-rank in [1, 2], LB and not UB, m-score >= 300, m-eff >= 250000 and Δ >= 220.
  - Otherwise any one of: barely touched with m-rank <= 4 (or rank 5 with cp >= 360 and Δ >= 350); m-rank in [1, 2] with LB, not UB, m-score >= 0 and m-eff >= 250000; or m-rank in [1, 2] with LB, not UB, m-score >= 150, m-eff >= 150000, S_abs >= 0.90, cp >= 380, Δ >= 360 and Qg >= 0.60.

**L26 `HybridMCTSRootRejectQGapOverride`.** Requires the veto and B <= R.
- Main case: `sane`, R in [40, 90], B in [25, 40], S in [0.40, 0.60], G >= 0.60, cp >= 80, Δ >= 150. Then m-rank in {2, 3}, m-score −∞, m-eff <= 2500, a-rank == 2, a-cur >= 20 and a-cur >= 75% of B, Qg >= 0.55.
- If the main case fails, any one of these six:
  - (a) `sane`, R in [24, 60], B in [18, 30], S >= 0.45, G >= 0.90, cp >= 250, Δ >= 280, m-rank in [2, 8], m-score −∞, m-eff <= 1000, a-rank in [3, 5], a-cur <= 2, Qg >= 0.90.
  - (b) `sane`, R in [18, 32], B in [16, 24], S >= 0.80, G >= 0.70, cp >= 250, Δ >= 250, m-rank in [2, 4], m-score −∞, m-eff <= 800, a-rank == 2, a-cur <= 2, Qg >= 0.70.
  - (c) `sane`, R in [12, 32], B >= 10, S >= 0.75, G >= 0.30, cp >= 75, Δ >= 75, m-rank in [2, 6], m-score −∞, m-eff <= 10000, a-rank in [2, 4], a-cur <= 3, Qg >= 0.30.
  - (d) `sane`, R in [45, 80], B >= 45, S >= 0.88, G >= 1.20, cp >= 500, Δ >= 600, m-rank in [2, 3], m-score −∞, m-eff <= 100, a-rank == 2, a-cur <= 2, Qg >= 1.20.
  - (e) `sane`, R in [90, 300], B >= 60, S >= 0.60, G >= 0.75, cp >= 140, Δ >= 220, m-rank == 2, m-score −∞, m-eff <= 1500, a-rank == 2, a-cur >= 20, Qg >= 0.75.
  - (f) **not** `sane`, R in [1400, 2600], B >= 1500, S >= 0.94, G >= 1.0, cp >= 300, Δ >= 350, m-rank in [2, 3], m-score −∞, m-eff <= 2600, a-rank == 2, a-cur in [20, 64], Qg >= 1.0.

**L27 `HybridMCTSClockRootRejectQGapOverride`.** `clock`, `sane`, veto. R in [100, 300], B >= 72, S >= 0.65, G >= 0.90, cp >= 220, Δ >= 300. m-rank in [2, 6], m-score −∞, m-eff <= 1000. a-rank in [2, 6], a-cur <= 4. Qg >= 0.90.

**L28 `HybridMCTSReusedRootConfidenceOverride`.** Only `budget` is required; `sane` is not.
- Cache-heavy case:
  - Either R_tot >= 500, B_tot >= 500, R >= 18, B >= 18, m-eff <= 100000; or R_tot >= 400, B_tot >= 400, R >= 20, B >= 20, m-eff <= 150000.
  - And: R_tot >= 3R, B_tot >= 3B, S_abs >= 0.96, S >= 0.96, cp >= 400, Δ >= 400, avg gap <= 75, m-rank in [1, 2], m-score −∞, a-rank == 2, a-cur <= 1, Qg >= 0.85.
- Otherwise: R_tot >= 200, B_tot >= 180, R >= 24, B >= 24, S_abs >= 0.90, S >= 0.90, R_tot >= 3R, B_tot >= 3B, cp >= 300, Δ >= 300, m-rank in [1, 3], m-score −∞, m-eff <= 250000, avg gap <= 75, a-rank >= 2, a-cur <= 2, Qg >= 0.75.

**L29 `HybridMCTSReusedRootCurrentOverride`.**
- Base: `sane`, not `clear`. R_tot >= 350, B_tot >= 300, R >= 24, B >= 24, S_abs >= 0.84, S >= 0.92, cp >= 300, Δ >= 300. R_tot >= 3R, B_tot >= 3B. a-rank >= 2, a-cur <= 1. Qg >= 0.47.
- Then one of:
  - m-rank in [1, 2], LB and not UB, m-score >= 300, m-eff >= 250000;
  - m-rank in [1, 2], LB and not UB, m-eff >= 250000, and either (m-score >= 0, S_abs >= 0.93, cp >= 400, Qg >= 0.58) or (m-score >= −50, S_abs >= 0.95, S >= 0.95, cp >= 400, Δ >= 400, Qg >= 0.85);
  - veto with m-rank in [1, 4], m-score −∞, m-eff <= 10000, G >= 0.75 and Qg >= 0.50.

**L30 `HybridMCTSCompactFixedBudgetOverride`.** `sane`, not `clear`. R in [50, 180), B >= 48, S >= 0.82, G >= 0.55, cp >= 140, Δ >= 140. avg gap <= 90.

**L31 `HybridMCTSCompactPawnEndgameOverride`.** `sane`, pawn-only endgame, not `clear`. R in [35, 80), B >= 32, S >= 0.88, G >= 0.90, cp >= 220, Δ >= 220. avg gap <= 90. m-rank == 2, m-score −∞, not LB, m-eff in [1e5, 6e5]. a-rank == 2, a-cur <= 2. Qg >= 0.90.

**L32 `HybridMCTSCompactClearPreferenceOverride`.** `sane`. R in [50, 180), B >= 48, S >= 0.90, G >= 0.75, cp >= 280, Δ >= 300, \|ab_score\| <= 40. a-rank >= 2, a-cur <= 2. Qg >= 0.75. m-rank == 2, m-score −∞, not LB, m-UB, m-eff in [20000, 200000].

**L33 `HybridMCTSCrossRootConfidenceOverride`.**
- `sane`, `strong`. R >= 250, B >= 170, S >= 0.62, G >= 0.12, cp >= 170, Δ >= 30.
- m-rank in [1, 4], m-score −∞, m-eff >= 1000, ab_score >= −300. avg gap <= 80. a-cur > 0.
- Then either a-rank >= 4, or (a-rank in [2, 3], a-cur <= 48, B >= 6·a-cur, G >= 0.12, S >= 0.64, Δ >= 30, Qg >= 0.18).
- Finally B >= 3·a-cur and Qg >= (rank-2/3 case ? 0.18 : 0.20).

**L35 `HybridMCTSNoClearFixedBudgetOverride`.** `sane`, not `clear`, `strong`, B >= 225, S >= 0.58, Δ >= 30.

**L36 `mcts_reliable_ab_unverified`.** `sane`, AB depth < 10, `reliable`, Δ >= −80.

**Disabled rungs, given for completeness.**
- L34 `HybridMCTSRootRejectsAB`. Outer gate: `config_.mcts_root_reject && top_count > 0 && mcts_reliable && B >= 160 && S >= 0.35 && ab_score in [-120, 40]`, and MCTS top-1 is the MCTS move. Inner test: `sane`, AB visits > 0, and either (AB visits >= 25, top visits >= 3·AB visits, Q gap >= 0.25) or (budget, strong, not clear, top >= 180, top >= 4·AB, AB <= 64, S >= 0.65, Q gap >= 0.12, Δ >= 25).
- The ANE rungs L2, L3 and L5 have their thresholds at 1680-1795, 2508-2656 and 1797-1849. They cannot fire under this configuration, so they are not enumerated here.

### 1.5 Trace reasons the engine can emit

The trace is written by `trace_simple` (4761-4796) or `trace_final` (5483-5585). Every string the engine can write as `reason=`:
- **Before the ladder:** `mcts_only_result`, `ab_only_result`, `fallback_legal`, `root_quiet_attack_tiebreak`, `root_policy_tiebreak`, `root_q_conflict_tiebreak`, `root_pawn_lever_tiebreak`, `engines_agree`.
- **From the ladder:** `ab_default`, `ab_root_rejects_mcts`, plus the reasons of rungs L1-L36 above. `ane_q_supported_root` selects the ANE move.

**`ab_fallback_time_safety` is not emitted by the engine.**
- On fallback moves the engine prints only `info string Time safety: <reason>; using Alpha-Beta without transformer MCTS` (uci.cpp:1731-1733), and does so whatever `HybridTrace` is set to.
- The parser `tools/paper/parse_hybrid_trace.py` turns that line into a record: `FALLBACK_RE = re.compile(r"Time safety:\s*(.*?using Alpha-Beta without transformer MCTS)")` (30-32) and `"reason": "ab_fallback_time_safety"` (167).

### 1.6 Global controls that act on the whole ladder (no per-rung option)

| Control | Default | Effect |
|---|---|---|
| `UseHybridSearch` | false (arm: true) | Enables the hybrid path. |
| `HybridABRootRejectMCTS` | true | The veto. |
| `HybridMCTSRootReject` | false | L34. |
| `HybridRootPawnLeverTieBreak` | false | P3, P6 and the tiebreak after the ladder. |
| `HybridANE*` | false | L2, L3, L5. |
| `HybridQToCpScale` | 300 | Sets cp, hence Δ, for every rung. |
| `HybridABCandidateVerifyMs` / `HybridABCandidateVerifyCount` | 240 / 5 | Feed the "verified hint" flags used by L14 and L23. |
| `HybridMCTSABRootHints` | true | Hints feed verification. |

The decision mode, `ab_min_depth`=10 and every numeric threshold are hard-coded.

---

## 2. MCTS-to-alpha-beta root hints

### When they are read: once per move, at the start of the AB driver thread

- In `run_ab_search`, `ab_limits.root_order_hints = collect_root_order_hints();` (`hybrid_search.cpp:3837`) is the only call site. It runs before the verification `go` (3849) and before the main alpha-beta `go` (3955).
- With `HybridMCTSABRootHintDelayMs=0` the snapshot is taken immediately: `if (delay_ms == 0 || SteadyNowMs() >= deadline_ms) break;` (3165-3172). If that pass found nothing, it retries once (3174-3176).
- Nothing later in the move changes alpha-beta's root ordering:
  - The AB→MCTS channel `update_mcts_policy_from_ab` returns at once: `if (config_.ab_policy_weight <= 0.0f) return;` (3048-3049).
  - `HybridMCTSUseSharedTT` is false.

### How many moves

- `hint_count = clamp(HybridMCTSABRootHintCount=8, 1, 16)`. Because candidate verification is on, it is raised to `max(8, clamp(5,1,10))` = 8 (3102-3107).
- `collect_root_order_hints` truncates to `clamp(8,1,16) + (ane_root_hints ? ... : 0)` = 8 (3522-3527).
- The list is assembled in this order (3114-3163):
  1. Quiet knight/bishop attacks on an enemy Q or R with policy >= 0.08 and >= 0.35·(policy of the most-visited move).
  2. Kingside pawn levers with policy >= 0.25 and >= 1.15·that leader policy (2826-2830).
  3. The most-visited moves, until the list holds max(1, 8/2) = 4.
  4. Policy order for the rest, up to 8.
- The source is `GetRootMoveStats()`, which returns `{}` if the MCTS root has no edges and otherwise sorts by total visits (including reused ones), then Q, then policy (`mcts/search.cpp:3741-3769`).
- The verification sub-search uses the first `HybridABCandidateVerifyCount`=5 distinct hints (3537-3551).

### What the MCTS root contains at that moment (timing)

- The MCTS thread is spawned in the same `start_search` call (313-316). It re-roots the tree in `StartSearch` (`mcts/search.cpp:389-410`, `tree_.TryReuse(fen)`, which keeps a child or grandchild of the old root; `node.h:507-588`).
- The AB thread reads the root under a shared lock. Nothing orders it after the re-root.
- So the snapshot can be:
  - the reused subtree's statistics from the previous move;
  - empty, if the root is not yet expanded, in which case there are no hints and no verification (3848 requires non-empty hints);
  - in principle, if the AB snapshot wins the race, the previous root's moves. `ApplyRootOrderHints` ignores moves that do not match legal moves.
- How often each case occurs is **not determinable from source**. The `ABHints=` field of each HybridTrace records what was used.

### How they are applied: ordering only, plus the verification side effects

- **Main search.** `ThreadPool::start_thinking` calls `Search::ApplyRootOrderHints(rootMoves, limits.root_order_hints);` once, before copying `th->worker->rootMoves = rootMoves;` into every thread (`search/thread.cpp:330, 344-356`).
- `ApplyRootOrderHints` is `std::stable_sort(first, last, [&](...){ return hint_rank(a) < hint_rank(b); });` within each tablebase-rank group. Hinted moves come first in hint order and the others keep generation order (`search/search.cpp:59-98`).
- Nothing else reads `root_order_hints`: `git grep root_order_hints 365f36a -- src` finds only `search.cpp:59-98`, `thread.cpp:330`, `search.h:176` and the hybrid. There is no score bonus, pruning or reduction change, and no `searchmoves` restriction in the main search.

### Interaction with Stockfish internals

**Root move order.**
- The root TT move is `ttData.move = rootNode ? rootMoves[pvIdx].pv[0] : ...` (`search.cpp:864`).
- The root move loop uses the ordinary `MovePicker` (1157). The rootMoves vector only filters which moves are searched: `if (rootNode && !std::count(rootMoves.begin() + pvIdx, ...` (1179).
- So the hint's direct effect is that the first legal hint is the root TT move at iteration 1, in every thread.

**Re-sorting between iterations.**
- After each search call, `std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast);` (433) and the stable sort at 517 move the best move to the front.
- Non-PV moves get `rm.score = -VALUE_INFINITE` (1547). Their hint-induced relative order survives only as a stable tie-break. `RootMove::operator<` compares score, then previousScore (`search.h:134-135`).
- From iteration 2 onwards, `rootMoves[0]` is the previous iteration's best move, not a hint.

**Aspiration windows.**
- `delta = 5 + threadIdx % 8 + std::abs(rootMoves[pvIdx].meanSquaredScore) / 10193; Value avg = rootMoves[pvIdx].averageScore; alpha = std::max(avg - delta, ...)` (403-408).
- Fresh `RootMove`s start with `averageScore = -VALUE_INFINITE` (`search.h:141`). The hints do not change the window except through which move is at index 0.

**Helper threads.**
- All threads receive the same hinted initial order (thread.cpp:350). Afterwards each thread re-sorts independently.
- The best-thread vote (`get_best_thread`, thread.cpp:364ff) is not affected by the hints.

**Indirect effects through the verification sub-search (§3).**
- The verification is a separate `go` restricted to the 5 candidates (`searchmoves`, 3559-3563). It runs on the same 8-thread pool and the shared TT, with `tt.new_search()` at every `go` (search.cpp:227).
- Its result reorders the hint list: verified order first, then the remaining hints (3851-3859). It also feeds `ABVerifiedHints` into the decision (L14, L23).
- It overwrites state carried across `go` calls: `bestPreviousScore` and `bestPreviousAverageScore` (search.cpp:275-277), and `previousTimeReduction` (662, set to 1 for a movetime search).
- Thread histories also persist across `go` calls; `lowPlyHistory` is reset and `mainHistory` decayed at each search start (358-362).

---

## 3. The verification sub-search

### Budget formula

`HybridABCandidateVerifyBudgetMs` (`hybrid_search.cpp:572-597`):

```
if (requested_ms <= 0) return 0;  if (waiting_for_ponderhit) return 0;
if (limits.nodes > 0 || limits.depth > 0 || limits.mate > 0 || limits.infinite) return 0;
if (limits.movetime > 0) { if (limits.movetime < 1000) return 0; budget_cap = movetime / 6; }
else if ((limits.time[WHITE] > 0 || limits.time[BLACK] > 0) && time_budget_ms >= 1000) {
  budget_cap = time_budget_ms / 8; } else { return 0; }
const int budget = std::min(requested_ms, budget_cap); return budget >= 10 ? budget : 0;
```

- `requested_ms` = `HybridABCandidateVerifyMs` = 240.
- `time_budget_ms` is the hybrid coordinator's budget, **not Stockfish's optimum or maximum**. It is `calculate_time_budget()` (468-500), possibly scaled by the classifier (§4):
  - `base = time_left/moves_to_go(30) + 0.75*inc`, capped by `max(500, time_left/4)` and `time_left-100`, floored at 250.
- Verification runs only if the hint list is non-empty: `if (candidate_verify_ms > 0 && !ab_limits.root_order_hints.empty())` (3848).
- For clock games, "up to 240 ms, at most an eighth of the per-move budget, when that budget is at least 1 s" describes this formula, where the per-move budget is the coordinator's budget.
- The 240 ms is the movetime of the **multi-candidate** search only. When a queen is on the board, extra single-candidate searches can follow:
  - `const int single_verify_ms = std::clamp(verify_ms / 2, 80, 160);` (3625).
  - One is run for each other candidate that is a quiet rook or queen check (`HybridIsQuietMajorCheck`, 2377-2392) and whose verify average is within 80 of the leader's (3626-3634).
  - The loop stops at the first candidate it promotes (3663-3669).
- The worst case is therefore 240 + 4·120 ms, plus stop latency.
- Both kinds of search are Stockfish `movetime` searches (3555-3563, 3636-3641), stopped by `(worker.limits.movetime && elapsed >= worker.limits.movetime)` (search.cpp:2157).

### Added on top of alpha-beta's allotment

1. The AB driver sets `ab_limits.startTime = now();` (3829).
2. It collects hints (3837) and runs the verification (3849).
3. It then **resets** `ab_limits.startTime = now();` (3864) before the main `engine_->go(ab_limits)` (3955).
4. With `HybridFollowABTime=true` the AB limits keep the game clock: `if (!config_.follow_ab_time) { ab_limits = HybridBuildABLimits(...) }` (3947-3954) is skipped.
5. Stockfish's time manager then calls `startTime = limits.startTime;` (timeman.cpp:42) and `elapsed_time() const { return now() - startTime; }` (timeman.h:36). Optimum and maximum come from the unchanged `limits.time[us]` received with `go` (timeman.cpp:89-127; `main_manager()->tm.init(limits, ...)` at search.cpp:225).

So the verification time, and everything between `go` and the main AB `go`, is **not charged** to the AB time manager. The AB search still receives its full optimum and maximum afterwards.

### Can the total move time exceed Stockfish's allocation? Yes

Wall time from `go` to `bestmove` is the sum of:
- the pre-AB window: thread start, hint snapshot, verification of up to 240 ms (+ single checks), stop latencies;
- the AB search, bounded by its own `tm.maximum()` measured from the reset `startTime` (search.cpp:641, 2155-2156);
- the post-AB coordinator work: stopping MCTS, waiting up to the join grace (§8), and the decision plus trace output.

There is also a secondary effect: the verification `go` changes the inputs to Stockfish's time manager for the main search.
- It sets `bestPreviousScore` and `bestPreviousAverageScore` (search.cpp:275-277). These are used in `iterValue.fill(...)` (338-341) and in `fallingEval` (607-611).
- It sets `previousTimeReduction` (662) to 1 (`double timeReduction = 1` at 315; it is never updated without time management). This is used in `reduction` (620-621).
- The main search's `totalTime` can therefore differ from what the plain alpha-beta engine would compute in the same position. The size of the difference is not determinable from source.

---

## 4. The position classifier

### What it changes

In `start_search` (`hybrid_search.cpp:267-293`), always on (`use_position_classifier = true`, uci.cpp:1278; there is no UCI option):

```
if (time_budget_ms > 0 && limits_.movetime <= 0 && current_strategy_.time_multiplier > 1.0f) {
  time_budget_ms = std::max(200, static_cast<int>(time_budget_ms * current_strategy_.time_multiplier)); }
```

followed by `time_budget_ms_.store(...)` and `info string Time budget: ...` (295-296). Only multipliers above 1 are applied; the budget is never shortened.

**Multiplier** (`hybrid/classifier.cpp:359-453`):
- Position type: HIGHLY_TACTICAL 1.1, TACTICAL 1.05, BALANCED 1.0, STRATEGIC 0.95, HIGHLY_STRATEGIC 0.9.
- ×1.15 if complexity > 0.7.
- Clock factor: ×0.5 if time_left < 10 s, ×0.8 if < 30 s, ×1.3 if > 300 s.
- ×1.1 if increment > 5 s.
- Worked example (arithmetic from these constants): with 10-30 s left, only HIGHLY_TACTICAL with complexity > 0.7 exceeds 1, at 1.1·1.15·0.8 = 1.012.

**Node minimum.**
- `min_mcts_nodes` defaults to 100 (`classifier.h:91`).
- It becomes 150 if complexity > 0.7 (`classifier.cpp:427-430`) and 300 if time_left > 300 s (446-448).
- It is used only in two places:
  - the coordinator's agreement early stop, `std::max(100, current_strategy_.min_mcts_nodes)` (4070-4071);
  - the reliability thresholds **when there is no MCTS decision budget** (4919-4926). In clock games there is a budget, so the fixed 180/72 apply.

**Other fields.**
- `ab_verify_depth` (2-6) always loses to `max(ab_min_depth=10, ...)` (3052, 4075, 4937), so it has no effect.
- `mcts_weight`, `cpuct`, `ab_depth`, `aspiration_window` and the remaining strategy fields are never read (`grep current_strategy_` finds only 268-290, 3052, 4071, 4075, 4922, 4926, 4937).
- It also prints `info string Parallel search - Position: ...` (283-292).

### Can it lengthen a move beyond Stockfish's allocation under `HybridFollowABTime=true`?

**Not by changing the AB search.**
- The AB limits stay on the game clock (3947-3954 skipped).
- `should_stop()` ignores the budget: `if (time_budget_ms > 0 && !HybridABTimeManagerOwnsClock(limits_, config_.follow_ab_time))` (2881-2889).
- `HybridABTimeManagerOwnsClock` returns true for clock searches when following AB time (511-519).

**What the larger `time_budget_ms_` does change:**
- the MCTS movetime (614-621);
- the verification cap, `time_budget_ms / 8`, including the 1000 ms on/off threshold (588-590). Because verification time is added before the AB clock starts (§3), this can **indirectly** lengthen the move by up to the verification window;
- the agreement-stop minimum time (`time_budget_ms / 4`, 4063-4065), which can only delay an early stop;
- the join grace (`clamp(budget/4, 1000, 5000)`, 2849-2855).

---

## 5. Who ends the move under `HybridFollowABTime=true`, and what bounds it

### Alpha-beta's Stockfish time manager ends the move

- Coordinator loop: `if (ab_done && ab_state_.has_result.load(...)) { if (!HybridShouldContinueMCTSAfterAB(limits_, config_.follow_ab_time) || mcts_done) { break; } }` (4037-4042).
- For clock searches `HybridShouldContinueMCTSAfterAB` returns `!follow_ab_time && (clock)`, i.e. false (502-509). The coordinator therefore stops as soon as the AB thread has exited with a result.
- The coordinator's own deadline is disabled for clock searches (2881-2889, §4).
- The AB search stops through the Stockfish iteration check `if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum())) ...` (search.cpp:641-649) or `check_time`'s `elapsed > tm.maximum()` (2155-2156).

### Two other ways a move can end

- **Agreement early stop.** `HybridCanStopEarlyOnAgreement` is true for clock searches (526-530). The coordinator `break`s when all of these hold:
  - `agreement_count >= 3`: three consecutive 0.5 ms polls with identical non-zero AB and MCTS moves;
  - `ms > min_time`, where `min_time = time_budget_ms / 4`;
  - MCTS nodes >= max(100, min_mcts_nodes);
  - AB completed depth >= max(10, ab_verify_depth).
  - It prints `info string Hybrid: engines agree, stopping early at <ms>ms` (4058-4082). It then sets the stop flag and calls `engine_->stop()` (4117-4122).
- **External `stop`/`quit`**: `stop_active_searches()` (uci.cpp:1664-1666).

### The MCTS side has its own per-move budget

- `HybridBuildMCTSLimits`: `if (!fixed_or_analysis && clock_search && time_budget_ms > 0) { mcts_limits.movetime = time_budget_ms; ... time/inc/movestogo = 0 }` (599-625).
- The MCTS budget is then `movetime - Move Overhead`: `if (limits_.movetime > 0) return std::max<int64_t>(1, limits_.movetime - overhead);` (`mcts/search.cpp:1501-1502`). It is enforced by a `TimeLimitStopper` measured from the MCTS `StartSearch` (372, 1451-1452).
- `SmartPruningStopper` (factor 1.33, 1465-1468) can stop MCTS earlier. The KLD stopper is off (`kld_gain_min=0.0`, uci.cpp:1266-1267).
- When AB finishes first, MCTS is cut: `stop_flag_ = true; mcts_search_->Stop();` (4117-4120). When MCTS finishes first, it idles and does not end the move.

### Hard cap

- There is **no hybrid-level wall-clock cap** for clock games in this mode.
- The effective cap is Stockfish's `maximumTime = max(optimum, min(0.8097*time_left - MoveOverhead, maxScale*optimum))` (timeman.cpp:125-127), measured from the post-verification `startTime`. To that add:
  - the pre-AB window: hints, verification of up to 240 ms (+ single checks);
  - the post-AB stop/join work. The coordinator waits at most `HybridSubsearchJoinGraceMs` = `clamp(budget/4, 1000, 5000)` ms for the subsearch threads before deciding without them (4124-4141).
- `TransformerLowTimeFallbackMs=3000` and `TransformerMinMoveBudgetMs=400` are a separate guard, not a cap: at low clock they bypass the hybrid completely (uci.cpp:938-992, 1730-1737).

---

## 6. Score reported in UCI `info ... score`

### Inside a hybrid move: only the coordinator's lines carry a score

- Stockfish's own UCI callbacks are replaced during both alpha-beta searches:
  - `engine_->set_on_update_full(...)` capture (3873-3921);
  - `set_on_bestmove(...)` (3923-3933);
  - no-op callbacks during verification (3565-3568).
- The MCTS search is started with `info_cb = nullptr` (2941).
- Score-bearing lines come from `send_info` in this form: `ss << "info depth " << depth; ss << " score cp " << score; ... ss << " string " << source;` (5631-5674).

**Periodic lines** (every >= 500 ms) carry `ab_state_.best_score` with `pv <ABmove> string hybrid` (4088-4106).

**Final line before `bestmove`** (4176-4182):

```
const int info_score = final_move == mcts_best && final_move != ab_best
    ? QToNnueScore(mcts_state_.best_q.load(...), config_.q_to_cp_scale)
    : ab_state_.best_score.load(...);
send_info(info_depth, info_score, total_nodes, ..., final_pv_, "hybrid-final");
```

- When the MCTS move is played, the score is `int(atanh(clamp(Q,±0.999))*300)`, bounded at ±1140 cp.
- Otherwise it is alpha-beta's last reported score. That includes the P4/P5 tiebreak moves, which differ from both engines' best moves.
- AB's score is WDL-normalised centipawns (`eval/score.cpp:21-22`). Mate is printed as **`cp` ±(30000 − plies)** and tablebase as ±(20000 − plies) (3878-3888). The hybrid **never prints `score mate`**.
- Line order at the end of a move:
  1. HybridTrace (inside `make_final_decision`, 4147);
  2. `info ... string hybrid-final` (4181);
  3. `info string Final: ...` (4184-4212, no score token);
  4. `bestmove` (4214).
- Which line fastchess uses for adjudication is fastchess behaviour, **not determinable from this repository's source** (fastchess is present only as a binary).

### Time-safety fallback moves: yes, it differs

- `parallel_hybrid_go` prints the `Time safety` line and calls `ensure_full_thread_pool(engine); engine.go(limits);` (uci.cpp:1730-1737).
- This is the plain Stockfish path. `UCIEngine::on_update_full` prints `" score " << format_score(info.score)` with an optional bound, seldepth, multipv, hashfull and tbhits (uci.cpp:737-760).
- `format_score` prints `mate N` (in moves) for mates and `cp` ±(20000 − plies) for tablebase scores (641-657). This matches the AB arm.
- The same plain path is used when there are fewer than 3 threads or the weights are missing (1754-1771).

---

## 7. Network load, MPSGraph build and warm-up before `readyok`

### Yes, at the first `isready`, if the hybrid is enabled and weights are set

- `isready` handler: `else if (token == "isready") { preload_search_objects(engine); sync_cout << "readyok" << sync_endl; }` (uci.cpp:161-164).
- `preload_search_objects` returns early unless `UseHybridSearch` or `UseMCTS` is set and a weights path resolves (1579-1590).
- It can be disabled with `METALFISH_PRELOAD_TRANSFORMER` = 0/false/off/no; preloading is on by default (994-1003).
- With `HybridABThreads=0` the split is not "stable" (1597-1598), so `can_preload_hybrid = ... || !g_parallel_hybrid_search` (1599-1600). It builds once, when no object exists yet:
  - `MCTS::create_parallel_hybrid_search(&engine, config)` (1615-1617)
  - then prints `info string Hybrid search preloaded (transformer ready)` (1620-1622).

**Construction chain:**
1. `ParallelHybridSearch::initialize` creates `Backend(weights, cache, backend_config)` and then `Search(...)` (hybrid_search.cpp:136-141).
2. `NNMCTSEvaluator::Impl` calls `NN::LoadWeights(weights_path)` then `NN::CreateNetwork(weights_, backend_config)` (`mcts/evaluator.cpp:61-69`).
3. `MetalNetwork::MetalNetwork` runs `builder_->init(...)` and `builder_->build(kInputPlanes, weights, ...)`, building the MPSGraph in the constructor (`nn/metal/metal_network.mm:39-59`).
4. The `MCTS::Search` constructor runs a warm-up: `warmup_batch(1, ...)` on the start position, a blocking evaluation via `comp->ComputeBlocking()`, then `backend_->Cache().Clear();` (`mcts/search.cpp:278-322`). The larger warm-up batch only runs if minibatch > 1. On Apple the auto minibatch resolves to 1 (uci.cpp:909-916; `mcts/search.cpp:216-217`), so the warm-up is a single batch-1 evaluation.

**At `go`.** The cached object is reused when `make_hybrid_cache_key(...)` matches (uci.cpp:1782-1807). For clock searches under oversubscription, the config does not depend on the search limits:
- the AB thread count equals `available` (1215-1218);
- the minibatch is only forced for movetime < 1000 (1258-1262).

So no re-initialisation should happen inside the clock. If it did, it would be re-created at 1786-1804 after `limits.startTime` was set (parse_limits, 265).

**Not determinable from source:** whether MPSGraph does further lazy compilation after the batch-1 warm-up, and whether fastchess always waits for `readyok` before the first `go`.

### `ucinewgame` does not preload

- `stop_active_searches(); wait_active_searches(); engine.search_clear(); reset_cached_search_objects();` (uci.cpp:166-171).
- `search_clear` clears the TT and thread state (engine.cpp:268-276).
- `new_game()` clears the MCTS tree and NN cache (`hybrid_search.cpp:424-466`; `mcts/search.cpp:331-354`). The loaded network and graph persist.
- A later `isready` does not rebuild, because the object exists (1599-1600).

---

## 8. Publication of the shared root state

### `ABSharedState` (`hybrid_search.h:36-89`)

- All fields are `std::atomic`.
- `set_best_move` stores `best_move_raw`, `best_score`, `completed_depth` and `nodes_searched` with `memory_order_relaxed`. It then does `has_result.store(true, std::memory_order_release); update_counter.fetch_add(1, std::memory_order_release);` (66-73).
- `publish_pv`: `pv_moves[i]` and `pv_depth` are relaxed; `pv_length.store(len, std::memory_order_release)` (81-88).
- `get_best_move()` uses `best_move_raw.load(std::memory_order_acquire)` (61-64). That is an acquire load on a variable that is stored relaxed, so it does not pair with a release on that variable.
- `ab_running` is stored with release (303, 3782) and loaded with acquire (4034).
- Writers are the alpha-beta callbacks on the AB threads (3917-3919, 3930-3931).

### `MCTSSharedState` (`hybrid_search.h:91-141`), written by `publish_mcts_state` (2994-3045)

- `best_move_raw`, `best_q`, `best_visits`, `best_current_visits`, `total_nodes`, `total_current_nodes` and every `top_moves[i]` field are relaxed (3003-3039).
- Then `num_top_moves.store(top_count, std::memory_order_release); has_result.store(true, std::memory_order_release); update_counter.fetch_add(1, std::memory_order_release);` (3041-3044).
- It is published every `policy_update_interval_ms`=50 from the hybrid MCTS thread (2965-2966), and again at that thread's end (2984).

### How the coordinator reads them

- **In its 0.5 ms poll loop:** `ab_running`, `mcts_running` and `ab_state_.has_result` with acquire (4034-4037). `best_move_raw` of both states with relaxed, for the agreement test (4048-4050). `completed_depth`, `nodes_searched` and `best_score` with relaxed, for info lines (4094-4098).
- **After the loop:** it re-publishes the MCTS state itself (`publish_mcts_state()`, 4143-4145) before `make_final_decision()` (4147).
- **In the decision:** `get_best_move()` uses acquire and the scalars use relaxed (4322-4337). `num_top_moves` is read with acquire, then the entries relaxed (4358-4375, 4449-4465). AB root data, hints and verified hints are copied under `ab_root_mutex_` (4381-4385, 4417-4421, 4482).
- There is no seqlock or version check, so a multi-field read during the search is not an atomic snapshot.
- The decision-time view is consistent when the subsearches have finished, because the done flags are stored with release in `signal_thread_done` (118-121) and loaded with acquire in the join loop (4129-4130).

### Stop/done handshake and join grace

- Each thread's RAII guard clears its `*_running` flag and calls `signal_thread_done` (2899-2907, 3779-3787, 4009-4017). That does `done_flag.store(true, std::memory_order_release); thread_cv_.notify_all();`.
- **Coordinator, after the loop:** `stop_flag_.store(true, release)`, `mcts_search_->Stop()`, `engine_->stop()` (4117-4122). It then polls both done flags until `HybridSubsearchJoinGraceMs` expires (4124-4136): `if (external_stop) return 2500; if (time_budget_ms <= 0) return 1000; return std::clamp(time_budget_ms / 4, 1000, 5000);` (2849-2855).
- **On expiry** it logs `send_info_string("Hybrid: subsearch stop grace expired; returning best available move");`, i.e. UCI `info string ...` (4138-4141). It decides and sends `bestmove`, then blocks until both threads finish (4216-4221).
- In that case `stats_.mcts_nodes` and `transformer_evaluations` are still the zeros from `stats_.reset()` (229), because they are only written at the end of `mcts_thread_main` (2989-2991). So `sane` is false and most rungs are disabled. This is derived from code; frequency is not determinable.
- **`wait()`** has no timeout. After 10 s it prints `[HYB] wait() taking >10s - coord_done=... mcts_done=... ab_done=...` to stderr (363-392).
- **`join_all_threads()`** (destructor path) waits up to 5 s on the condition variable, then joins (98-116).

---

## 9. HybridTrace logging

### Only the hybrid uses it

- `config.trace_decisions = engine.get_options()["HybridTrace"];` (uci.cpp:1319) is read only in `hybrid_search.cpp`. `git grep trace_decisions` gives 3178-3513, 3747, 3763, 4283, 4762, 5484, plus h:213.
- Pure MCTS, plain alpha-beta, time-safety fallback, <3-thread fallback and tiny-node MCTS-route moves produce no HybridTrace.

### What it writes per hybrid move

**The decision line.** There is exactly one `info string HybridTrace: reason=<r> ...` line per hybrid decision. A second line would appear only if the post-ladder pawn-lever tiebreak fired; it is disabled.
- `trace_final` (5483-5585) writes these fields:
  - **Decision and moves:** reason, mode, selected, ABMove, MCTSMove.
  - **Scores and visits:** ABScore, ABDepth, MCTSQ, MCTSCP, EvalDelta, MCTSBestVisits, MCTSBestCurrentVisits, MCTSPlayouts, MCTSEvals, MCTSRootVisits, MCTSRootCurrentVisits, MCTSConfidence(Root)Visits, AbsoluteVisitShare, VisitShare.
  - **Status flags:** MCTSDecisionBudget, MCTSVisitEvidenceSane, LowNodeMCTSPrimary/Ready, RootQGap, MCTSReliable, MCTSStrong.
  - **One 0/1 flag per ladder predicate**, and MCTSOverwhelming, ABVerified, ABClearPreference, ABRootRejectsMCTS, MCTSOverrideAllowed, MCTSRootRejectsAB.
  - **ANE fields.**
  - **Lists:** ABHints, ABVerifiedHints, `MCTSTop=[move:n=..:cn=..:q=..:p=..]` (up to 10), FirstABHint, FirstABVerifiedHint, ABMCTSAgreeOffFirstHint, ABInMCTS*/MCTSInAB* cross ranks, `ABRoot=[move:s=..:ps=..:avg=..:lb=..:ub=..:eff=..]` (up to 16).
- `trace_simple` (4761-4796) writes a subset, without ABDepth, MCTSCP, EvalDelta, shares or flags. It is used for `mcts_only_result`, `ab_only_result`, `fallback_legal`, the tiebreaks and `engines_agree`.

**Other trace-gated lines:**
- `Hybrid: AB root hints from MCTS ...` (3178-3184)
- `Hybrid: preserving leading MCTS root hint after shallow AB verify ...` (3747-3752)
- `Hybrid: AB candidate verify <ms>ms ...` (3763-3769)
- `Hybrid: final PV truncated before illegal move ...` (4283-4286)
- ANE probe lines (disabled)

**Lines printed regardless of HybridTrace:**
- `Starting Parallel Hybrid Search`, `Hybrid MCTS runtime`, `Hybrid thread split` (uci.cpp:1749-1780)
- `Parallel search - Position`, `Time budget` (283-296)
- periodic `... string hybrid` lines, `engines agree, stopping early`, `subsearch stop grace expired`, `... string hybrid-final`, and `Final: MCTSPlayouts=...` (4079-4212)

### Which thread writes it, and when

- **Decision line:** the coordinator thread writes it inside `make_final_decision()` (`trace_final(reason, selected)` at 5590; called at 4147). That is **after** both subsearches have been stopped and joined (or the grace has expired) and **before** `invoke_best_move_callback` prints `bestmove` (4214). Formatting and I/O therefore count toward the move's wall time but cannot affect either search.
- **Hint and verification lines:** the AB driver thread writes them before the main AB `go`.
- **Output path:** everything goes through `info_callback_` to `sync_cout << info << sync_endl` (uci.cpp:1822-1824), i.e. synchronous, mutex-serialised stdout.
- `trace_decisions` is also part of the hybrid cache key (uci.cpp:1351). Toggling it between moves would force a rebuild; it is constant in the experiments.
