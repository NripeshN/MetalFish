# The oracle's text in the v2 manuscript, fixed for each class before the oracle is read

Written on 2026-11-09, after protocol amendment 16 merged (03:11 UTC) and before any RQ6b-R output
was read. At that time 21 of the 150 shards had finished. Only their health lines had been seen:
positions judged, judge errors, rows lacking a judge, judgments of an earlier configuration,
positions judged again and GPU error lines.

The manuscript's oracle sentences are taken from this file. Which variant is used is decided
mechanically by the class of amendment 16 (b), as computed by `oracle_probe.py --summarise-only` and
tabulated by `make_figures.py` (`oracle.md`, `oracle_cells.tex`, `oracle_stability.md`). Numbers in
⟨angle brackets⟩ are filled from those artifacts. Nothing else in a sentence is changed except for
grammar that the numbers require, such as a plural.

The title is not affected. Under 09's rule the title depends on the final RQ1a and RQ1b numbers,
which are final. It stays blueprint candidate 1, whatever the oracle's class.

## Notation

- A bound is in expected-score points per disagreement position, written with its 95% interval as
  `0.004 [0.001, 0.008]` in the body and `0.004 (95\% interval 0.001 to 0.008)` in the abstract.
- ⟨CONS⟩: the consensus bound, on the positions where both judges agree (12 (d)).
- ⟨X_SL⟩: the cross-judge bound when Stockfish 19 selects and Lc0 scores. ⟨X_LS⟩: the reverse (14).
- ⟨SF⟩, ⟨LC⟩: the per-judge bounds, where one judge both selects and scores. They are reported but do
  not decide the class.
- ⟨N⟩: positions with verdicts from both judges. ⟨G⟩: the games they come from.
- Suffixes \_A and \_B: the same quantity on the positions from RQ1a's and from RQ1b's games (16 (a)).
- ⟨CLASS⟩ phrases: *little to gain*, *room under both judges*, *judge-dependent or unresolved*, or
  *no class*.

## Selection rules (mechanical)

1. **Class.** The variant is chosen by the pooled `outcome.class` in the summary: 1 (little to gain),
   2 (room under both judges), 3 (judge-dependent or unresolved) or D (no class: an incomplete run
   with a stratum uncovered, 16 (c)).
2. **Class 3 has two forms.**
   - 3a: the three bands include at least one *within* and at least one *beyond*. The text says the
     gain "depends on the judge".
   - 3b: otherwise. The text says it "is not resolved at this sample size".
3. **RQ1a.** §6 always gives both cells' bounds and classes. If the RQ1a positions' class differs from
   the pooled class, the abstract, §1 and the conclusion each add: "(on the positions from RQ1a's
   games alone: ⟨CLASS_A⟩)".
4. **An incomplete run** (⟨N⟩ < 600 at the cut-off, every stratum covered):
   - §4.4 uses its incomplete form;
   - in the abstract and §1, "600 disagreement positions" becomes "⟨N⟩ of the 600 sampled
     disagreement positions (an incomplete run)";
   - Contribution 4 and the conclusion add: "(⟨N⟩ of the 600 positions; the run is incomplete)".
5. **Stability (16 (e)).**
   - §6 always gives both judges' two statistics and whether each passes.
   - If a judge fails, §6 adds the sentence below. The abstract and the conclusion add: ", with
     ⟨judge⟩'s verdicts those of its runner configuration (it failed the stability check)".
6. **The conversion (16 (g)).**
   - §6 always gives the class at 125 and 500 cp.
   - If either differs from the class at 250 cp, the abstract and the conclusion add: "(the class is
     sensitive to the conversion constant)".
7. **Not used anywhere:** a combined reading of the bound with RQ1a's Elo interval (16 (d)).
   Projected Elo appears only in §6 and the table, labelled as a conversion and not a prediction of
   game results.

## Abstract (replaces `\pending{oracle study RQ6b-R: one sentence with its bound}`)

**1. Little to gain.** On 600 disagreement positions from the workstation's RQ1a and RQ1b games,
judged on the runners by Stockfish~19 and by Lc0 with the BT4 network (RQ6b-R), an arbiter choosing
between the two root candidates at decision time could have gained little: where both judges agree,
the bound is ⟨CONS⟩ expected-score points per disagreement position (95\% interval ⟨lo⟩ to ⟨hi⟩), and
it and both cross-judge bounds lie within the registered tie band of 0.01.

**2. Room under both judges.** On 600 disagreement positions from the workstation's RQ1a and RQ1b
games, judged on the runners by Stockfish~19 and by Lc0 with the BT4 network (RQ6b-R), the bound on
what any arbiter choosing between the two root candidates could gain is ⟨CONS⟩ expected-score points
per disagreement position (95\% interval ⟨lo⟩ to ⟨hi⟩) where both judges agree. It is an upper bound,
not a prediction of game results, and it and both cross-judge bounds exceed the registered tie band
of 0.01.

**3a. Judge-dependent.** On 600 disagreement positions from the workstation's RQ1a and RQ1b games,
judged on the runners (RQ6b-R), what an arbiter choosing between the two root candidates could gain
depends on the judge, in expected-score points per disagreement position:
- ⟨X_SL⟩ when Stockfish~19 selects and Lc0 with the BT4 network scores;
- ⟨X_LS⟩ the other way round;
- ⟨CONS⟩ where both agree (tie band 0.01).
No direction is claimed.

**3b. Unresolved.** As 3a, but "is not resolved at this sample size" replaces "depends on the judge".

**D. No class.** The oracle study on the runners (RQ6b-R) was incomplete at the cut-off: ⟨N⟩ of its
600 positions had verdicts from both judges, and not every stratum was covered. Its results are
reported descriptively.

## §1, Results paragraph (replaces `\pending{the oracle study RQ6b-R: one sentence}`)

- **1.** The oracle study on the runners (RQ6b-R) finds little to gain: the consensus bound is ⟨CONS⟩
  expected-score points per disagreement position, and it and both cross-judge bounds lie within the
  tie band of 0.01 (Section~\ref{sec:oracle}).
- **2.** The oracle study on the runners (RQ6b-R) finds room under both judges: the consensus bound,
  ⟨CONS⟩ expected-score points per disagreement position, is an upper bound on what any arbiter could
  gain, and it and both cross-judge bounds exceed the tie band of 0.01 (Section~\ref{sec:oracle}).
- **3a / 3b.** In the oracle study on the runners (RQ6b-R) the bounds are ⟨CONS⟩ where the judges
  agree, and ⟨X_SL⟩ and ⟨X_LS⟩ across judges, so whether an arbiter could gain depends on the judge
  (3a) / is not resolved at this sample size (3b) (Section~\ref{sec:oracle}).
- **D.** The oracle study on the runners (RQ6b-R) was incomplete at the cut-off and is reported
  descriptively (Section~\ref{sec:oracle}).

## Contribution 4 (replaces `\pending{the oracle study's results on the runners (RQ6b-R)}`)

- **1.** its result on the runners: little to gain under either judge (RQ6b-R)
- **2.** its result on the runners: room for an arbiter under both judges, an upper bound and not a
  prediction of game results (RQ6b-R)
- **3a.** its result on the runners: the bound depends on the judge (RQ6b-R)
- **3b.** its result on the runners: the bound is not resolved at this sample size (RQ6b-R)
- **D.** the descriptive results of its incomplete run on the runners (RQ6b-R)

## §4.4, status (replaces `\pending{the oracle study RQ6b-R: status at the cut-off}`)

- **Complete:** RQ6b-R is complete, with verdicts from both judges on all 600 positions; it was first read on ⟨date⟩ at ⟨time⟩~UTC, after amendment~16 and its tools had merged (Appendix~\ref{app:amend})
- **Incomplete, every stratum covered:** RQ6b-R was cut off on ⟨date⟩ with ⟨N⟩ of its 600 positions
  judged by both judges and every stratum covered, so it is reported as an incomplete run
  (amendment~16 (c)); it was first read on ⟨date⟩ at ⟨time⟩~UTC, after amendment~16 and its tools had
  merged (Appendix~\ref{app:amend})
- **Incomplete, a stratum uncovered (D):** RQ6b-R was cut off on ⟨date⟩ with ⟨N⟩ of its 600 positions
  judged by both judges and not every stratum covered, so its results are descriptive, with no class
  (amendment~16 (c)); it was first read on ⟨date⟩ at ⟨time⟩~UTC, after amendment~16 and its tools had
  merged (Appendix~\ref{app:amend})

## §6 (replaces `\pending{oracle study RQ6b-R on the runners: …}`)

**Common part (every class).** Shares are given as point estimates, with their intervals in the
table; bounds are given with their intervals.

RQ6b-R has verdicts from both judges on ⟨N⟩ of the 600 sampled positions, from ⟨G⟩ games (Stockfish~19
judged ⟨N_SF⟩ and Lc0 ⟨N_LC⟩; a position without both verdicts enters only its judge's results). On the
74 positions the workstation had judged, the runners' Stockfish~19 gave the same verdict on ⟨SF_SAME⟩
and Lc0 on ⟨LC_SAME⟩, with correlations of ⟨SF_R⟩ and ⟨LC_R⟩ between the two machines' score
differences (Table~\ref{tab:oracle-stability}); ⟨STABILITY⟩. Stockfish rated the alpha-beta move
better on ⟨SF_AB⟩ of the disagreement positions and the MCTS move on ⟨SF_MCTS⟩, Lc0 on ⟨LC_AB⟩ and
⟨LC_MCTS⟩, and the judges agreed on ⟨AGREE⟩ (Table~\ref{tab:oracle}). The bounds that decide the class
are ⟨CONS⟩ expected-score points per disagreement position where both judges agree, ⟨X_SL⟩ when
Stockfish selects and Lc0 scores, and ⟨X_LS⟩ the other way round (Table~\ref{tab:oracle-cells}); the
per-judge bounds, in which one judge both selects and scores, are in Table~\ref{tab:oracle} and do not
decide it.

⟨STABILITY⟩ is one of:
- **both pass:** "both pass amendment~16's criterion, and for Stockfish the two statistics also show
  how sensitive its verdicts are to search depth (3 threads against 8)"
- **one fails:** "⟨Judge⟩ fails amendment~16's criterion, so its runner verdicts are those of its
  runner configuration, and the class below carries that qualification" (the other judge's pass is
  stated as above)
- **both fail:** "both fail amendment~16's criterion, so their runner verdicts are those of their
  runner configurations, and the class below carries that qualification"

**The class.**

- **1.** All three lie within the tie band of 0.01, so by amendment~16 the class is *little to gain*:
  judged by two related engines, an arbiter choosing between these two root candidates could have
  gained little on these positions.
- **2.** All three exceed the tie band of 0.01, so by amendment~16 the class is *room under both
  judges*: the consensus bound, ⟨ELO_CONS⟩~Elo per disagreement position as a conversion, is an upper
  bound on what any arbiter between the two root candidates could gain there, not a prediction of game
  results, and the two judges are related (Section~\ref{sec:threats}).
- **3a.** The bounds fall on both sides of the tie band of 0.01, so by amendment~16 the class is
  *judge-dependent or unresolved*: whether an arbiter could gain depends on the judge, and no direction
  is claimed.
- **3b.** At least one bound's interval contains 0.01, so by amendment~16 the class is
  *judge-dependent or unresolved*: whether an arbiter could gain is not resolved at this sample size,
  and no direction is claimed.
- **D.** Not every stratum has a position with verdicts from both judges, so by amendment~16 (c) the
  results are descriptive and no class is assigned.

**Per source cell (every class except D).** "On the positions from RQ1a's games alone the class is
⟨CLASS_A⟩, and on RQ1b's ⟨CLASS_B⟩ (Table~\ref{tab:oracle-cells}, each reweighted by its own cell's
strata)."

**The conversion (every class except D).**
- If no class changes: "With the conversion constant halved or doubled (tie bands of 0.02 and 0.005),
  the classes are unchanged."
- Otherwise: "With the constant at ⟨125 | 500⟩~cp per unit of log-odds, the class ⟨of which positions⟩
  becomes ⟨X⟩, so it is sensitive to the conversion."

## Conclusion (replaces `\pending{the oracle study's result on the runners (RQ6b-R)}`)

The sentence before the slot reads: "The oracle study is the test of the third:"

- **1.** judged by Stockfish~19 and by Lc0 with the BT4 network, an arbiter choosing between the two
  root candidates could have gained little on the positions sampled. Where both judges agree the
  bound is ⟨CONS⟩ expected-score points per disagreement position. The third condition is therefore
  not met by this pair of searches at the budgets tested.
- **2.** both judges leave room. The consensus bound of ⟨CONS⟩ expected-score points per
  disagreement position is an upper bound on what an arbiter between the two root candidates could
  gain, not a prediction of game results. The third condition is therefore open, and an arbiter
  learned from such positions is the next step.
- **3a / 3b.** the bounds are ⟨X_SL⟩ when Stockfish~19 selects and Lc0 scores, ⟨X_LS⟩ the other way
  round, and ⟨CONS⟩ where both agree. Whether an arbiter could gain therefore depends on the judge
  (3a) / is not resolved at this sample size (3b). The third condition is untested in either
  direction.
- **D.** the run was incomplete at the cut-off (⟨N⟩ of 600 positions with verdicts from both judges,
  not every stratum covered), so the third condition is untested here.

## The arXiv abstract

It uses the same class wording as the abstract above, with the consensus bound and its interval.

## Revision before reading (2026-11-09, after the re-review's Phase 1 criteria)

Added before any RQ6b-R output was read:
- §4.4 states when RQ6b-R was first read, so that the timing of the rules fixed before reading can be
  checked in the manuscript (re-review criteria, FIXED-BEFORE-READING and NS-3). The amendments
  appendix gives the same time in its Preceded column.
- §6 gives each judge's count of positions judged, and how a position without both verdicts is handled
  (re-review NS-2).
- §1 and §6 are shortened, so that the main text stays within the length ceiling once the numbers are
  in (a trial fill with placeholder numbers counted 10,570 words against a ceiling of 10,450). The
  shares become point estimates, with their intervals in the table. The per-cell bounds move to the
  table, and the text gives the classes. No class rule, number or wording of a class's reading
  changes.
