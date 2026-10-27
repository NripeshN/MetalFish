# Springer Nature support ticket — author-initiated Correction (text to paste)

**Filing is the author's action.** Nothing in this directory has been sent anywhere. Steps and
the text to paste are below; fill the three remaining `{{…}}` fields in `correction_note.md` first.

## Where to file (verified 2026-10-14)

1. Sign in to the Springer Nature support portal, https://support.springernature.com (the same
   Springer Nature account used on SpringerLink / during production). The books article
   "Post-publication book corrections"
   (https://support.springernature.com/en/support/solutions/articles/6000281741-post-publication-book-corrections)
   is only readable when signed in; its public summary says errors in a published book's text,
   metadata or figures can be reported to the publishing editor or via a support request, that
   the ISBN or DOI, an error description and a justification are required, that multi-change
   requests need a marked-up PDF of the affected pages, and that Springer Nature decides between
   a formal erratum/correction and a change in a later edition.
2. Open "Request Support" (https://support.springernature.com/en/support/tickets/new), choose
   the Books / published-book-correction category offered, and paste the text below. Attach
   `chapter_annotated.pdf` (highlights + corrected text at every passage) and
   `correction_note.md` (export to PDF or DOCX if the form only accepts those).
3. In parallel, email the volume editors (Kohei Arai and Pascal Lorenz, via the IntelliSys / SAI
   Conferences address used for the submission) and the Springer publishing editor named in the
   production correspondence for LNNS, with the same two attachments and the ticket number: the
   books article names the publishing editor as the other route.
4. Keep the ticket number in this directory (`README.md` → Status) and, when the Correction is
   published ("Correction to: …", own DOI, linked to the chapter), add its DOI to
   `../05_related_work.md` and to the preprint's bibliography.

Expectation to be aware of: Springer Nature follows COPE guidance. A Correction is used when a
part of an otherwise reliable publication is misleading; when the publisher judges that the
main results are unreliable it may instead issue a retraction (or a retraction and
replacement). Items E1–E2 withdraw the chapter's headline results, so the publisher may decide
on a retraction notice rather than a correction. The request below asks for a Correction and
leaves the instrument to the publisher; do not argue for a lesser instrument than the facts
support.

## Ticket fields

**Title**
Hybrid Chess Search: Combining Alpha-Beta and Monte Carlo Tree Search on Unified Memory Architectures

**DOI**
10.1007/978-3-032-32726-0_39 (chapter 39 of "Intelligent Systems and Applications. IntelliSys
2026", eds. K. Arai and P. Lorenz, Lecture Notes in Networks and Systems, vol. 2078, Springer,
Cham, pp. 653–664, published online 14 August 2026; book DOI 10.1007/978-3-032-32726-0; ISBN 978-3-032-32725-3 (print),
978-3-032-32726-0 (online))

**Corresponding author name & email address**
Nripesh Niketan — nripesh14@gmail.com (the address on the chapter; ORCID 0009-0008-2066-1937)

**Error description**
I am the sole author of this chapter and request an author-initiated Correction. On
re-examining the chapter against the engine's source code and the retained measurement
records I found that several statements of results and implementation facts are incorrect or
cannot be supported. The attached annotated PDF highlights each passage and gives the corrected
text; the attached Correction note lists all fourteen items. In summary:

1. The Elo ratings (∼4048, ∼3724, ∼3289), Fig. 2, the ΔElo column of Table 6 and the "∼435 Elo"
   statements were computed by an invalid procedure (an average over assumed opponent ratings
   plus clamped five-game logistic differences) and are withdrawn; the tournament's game
   records were not retained.
2. The alpha-beta engine is Stockfish's search and evaluation with eight small local edits, not
   a derivative "adding GPU integration"; the 100% move agreement with Stockfish is therefore
   not a validation, and MetalFish-AB's strength statements are statements about Stockfish.
3. MCTS nodes occupy 128 bytes (one per cache line), not 64 bytes; the "two nodes per cache
   line" statements are wrong.
4. The "2 → 148 NPS, 74×" improvement cannot be reproduced and has no record; withdrawn.
5. The neural-network cache figures (3.5 GB → 38 MB, 92×) are wrong (≈0.41 GB, ≈9×).
6. The transposition-table default (16 MB, not 512 MB) and the NNUE network sizes and feature
   set (108.9 MB and 3.5 MB; FullThreats + HalfKAv2_hm) were misstated.
7. The position classifier does not adjust the decision weighting, and Fig. 1's routing through
   the classifier is wrong (mode selection is by UCI option).
8. The agreement-based early stop never fired in the telemetry retained from the chapter's
   period (0 of 209 decisions) and fires rarely in the current engine; the "10–40% time
   savings" figure is unsupported.
9. The inference that the hybrid is "heavily influenced by its neural network component" is
   wrong: the final move was the alpha-beta move in 207 of 209 logged decisions; the
   Bratko–Kopec suite is at ceiling and was used during tuning.
10. The "1.26× / 5.5× higher throughput than Lc0" result is withdrawn; controlled measurements
    show MetalFish's inference slower than Lc0 at batch size 1, the setting it plays with, and at
    each program's best batch size.
11. The M1 Ultra figures (9.8M NPS, 7.24× scaling, ∼382 NPS) have no record; withdrawn.
12. Two related-work characterisations (Baier & Winands; Lanctot et al.) are wrong.
13. The hybrid does not use the Lc0-style time manager; 14. the chapter omits the time-safety
    rule under which the hybrid plays the alpha-beta move whenever the clock is short.

**Justification for correction**
The errors affect the scientific content of the chapter (its quantitative results and the
description of what was built) and the publication record. The corrected statements are
supported by the engine's source code at the commit the chapter described and by measurement
records that are retained and will be deposited with the follow-up paper
({{PREPRINT}}, data at {{ZENODO_DOI}}). I ask that a Correction be published and linked to the
chapter so that readers are not misled by the withdrawn figures. I will supply any further
material the editors require.
