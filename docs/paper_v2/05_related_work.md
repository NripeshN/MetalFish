# 05 — Related work map and reference list

Status column: **ID verified** = DOI/arXiv/venue confirmed during the 2026-10-07 or 2026-10-14
search; **verify** = found via secondary sources, confirm before citing (IRON RULE:
no unverified citation enters the paper). The earlier chapter had 16 references; the
new paper needs roughly 45–55.

## A. Search paradigms and hybrids

| Key | Reference | Identifier | Use | Status |
| --- | --- | --- | --- | --- |
| knuth1975 | Knuth & Moore, An analysis of alpha-beta pruning, AIJ 6(4) 293–326, 1975 | 10.1016/0004-3702(75)90019-3 | AB foundation | ID verified |
| coulom2006 | Coulom, Efficient selectivity and backup operators in MCTS, CG 2006, LNCS 4630 | 10.1007/978-3-540-75538-8_7 | MCTS origin | ID verified |
| kocsis2006 | Kocsis & Szepesvári, Bandit based Monte-Carlo planning, ECML 2006, LNCS 4212, 282–293 | 10.1007/11871842_29 | UCT | ID verified 2026-10-14 (Springer/SZTAKI) |
| browne2012 | Browne et al., A survey of Monte Carlo tree search methods, IEEE TCIAIG 4(1) 1–43, 2012 | 10.1109/TCIAIG.2012.2186810 | MCTS survey | ID verified 2026-10-14 (IEEE Xplore 6145622) |
| rosin2011 | Rosin, Multi-armed bandits with episode context, AMAI 61(3) 203–230, 2011 | 10.1007/s10472-011-9258-6 | PUCT | ID verified 2026-10-14 (Crossref) |
| silver2018 | Silver et al., A general RL algorithm that masters chess, shogi and Go through self-play, Science 362(6419) 1140–1144, 2018 | 10.1126/science.aar6404 | AlphaZero; the "averaging vs minimax" argument (quote from arXiv:1712.01815 v1) | ID verified |
| baier2013 | Baier & Winands, Monte-Carlo tree search and minimax hybrids, IEEE CIG 2013, 1–8 | 10.1109/CIG.2013.6633630 | Minimax inside MCTS | ID verified 2026-10-14 (Crossref) |
| baier2015 | Baier & Winands, MCTS-minimax hybrids, IEEE TCIAIG 7(2) 167–179, 2015 | 10.1109/TCIAIG.2014.2366555 | Journal version | ID verified 2026-10-14 (Crossref) |
| baier2018 | Baier & Winands, MCTS-minimax hybrids with state evaluations, JAIR 62, 193–231, 2018 | 10.1613/jair.1.11208 | Hybrids with evaluations | ID verified 2026-10-14 (Crossref) |
| lanctot2014 | Lanctot, Winands, Pepels, Sturtevant, Monte Carlo tree search with heuristic evaluations using implicit minimax backups, IEEE CIG 2014, 1–8 | 10.1109/CIG.2014.6932903 (arXiv:1406.0486) | Implicit minimax backups | ID verified 2026-10-14 (Crossref) |
| huang2015 | Huang, Pruning game tree by rollouts, AAAI 2015, 1165–1173 | 10.1609/aaai.v29i1.9371 | Alpha-beta and MT-SSS* as rollout algorithms | ID verified 2026-10-14 (AAAI OJS) |
| baier2017 | Baier, A rollout-based search algorithm unifying MCTS and alpha-beta, in: Computer Games (CGW/GIGA 2016), CCIS, Springer, 57–70, 2017 | 10.1007/978-3-319-57969-6_5 | Unification | ID verified 2026-10-14 (Crossref) |
| ramanujan2010 | Ramanujan, Sabharwal, Selman, On adversarial search spaces and sampling-based planning, ICAPS 2010, 20(1) 242–245 | 10.1609/icaps.v20i1.13437 | Traps in chess vs Go | ID verified 2026-10-14 (AAAI OJS) |
| nguyen2024 | Nguyen & Ramanujan, Lookahead pathology in Monte-Carlo tree search, ICAPS 2024, 34(1) 414–422 | 10.1609/icaps.v34i1.31501 (arXiv:2212.05208) | UCT pathology | ID verified 2026-10-14 (AAAI OJS) |
| ji2023 | Ji & Thielscher, MCTS with dynamic depth minimax, Advances in Computer Games (ACG 2023), 63–75 | DBLP conf/acg/JiT23 (Springer LNCS; DOI to add from the volume) | Dynamic minimax depth inside MCTS | Title/venue/pages verified 2026-10-14 (DBLP, UNSW PDF); DOI pending |
| helfenstein2024 | Helfenstein, Czech, Blüml, Eisel, Kersting, Checkmating one, by using many: combining mixture of experts with MCTS to improve in chess, 2024 | arXiv:2401.16852 | MoE + MCTS in chess (CrazyAra line) | ID verified 2026-10-14 (arXiv API); the earlier key `czech2026` had the wrong year and first author |
| ~~truong2023~~ | "Monte Carlo Tree Search-Minimax Hybrid in AlphaZero" (ml-research PDF) | — | — | **Not located** in two searches on 2026-10-14; do not cite |
| chen2019 | Chen, Chen & Lin, Two-phase-win strategy for improving the AlphaZero's strength, WSCE 2019 (IEEE), 117–121 | 10.1109/WSCE49000.2019.9041112 | AlphaZero + alpha-beta minimax (Connect Four) | ID verified 2026-10-14 (NTNU scholar record) |
| cohen2023 | Cohen-Solal & Cazenave, Minimax strikes back, AAMAS 2023, 1923–1931 | DBLP (arXiv:2012.10700) | Minimax-based learning vs AlphaZero-style MCTS | Venue/pages verified 2026-10-14 (DBLP, LAMSADE CV); cite the AAMAS version |

## B. Engines and systems that combine or compare the paradigms

| Key | Reference | Identifier | Use | Status |
| --- | --- | --- | --- | --- |
| stockfish | Stockfish developers, Stockfish 18 (Jan 2026) and 19 (Sep 2026) release notes | stockfishchess.org/blog | AB+NNUE SOTA; SFNNv10/v16; UHO/pentanomial testing culture | verified pages |
| nasu2018 | Nasu, Efficiently updatable neural-network-based evaluation functions for computer shogi, WCSC28 appeal doc, 2018 | — | NNUE | known |
| lc0 | Leela Chess Zero project; "Transformer progress" blog, Feb 28, 2024 | lczero.org/blog/2024/02/transformer-progress | BT networks; raw policy +300 Elo over T78 | verified page |
| monroe2024 | Monroe & Chalmers, Mastering chess with a transformer model | arXiv:2409.12272 (v2, 2024-10-28) | Lc0 transformer design | ID verified 2026-10-14 (arXiv) |
| ruoss2024 | Ruoss et al., Amortized planning with large-scale transformers: a case study on chess | arXiv:2402.04494 (v2, 2024-10-21) | Searchless transformer chess; ChessBench | ID verified 2026-10-14 (arXiv) |
| scorpio | Shawul, Scorpio chess engine (alpha-beta rollouts in MCTS; NN eval), 2018– | github.com/dshawul/Scorpio | Closest engine-level precedent | software; cite repo + CPW |
| komodo | Komodo/Dragon MCTS mode (tree growth guided by shallow alpha-beta) | komodochess.com; CPW "Komodo" | Commercial hybrid precedent; developer-reported strength | software/web |
| sfmcts2017 | Nicolet, A branch to test the Monte Carlo algorithm in Stockfish, FishCooking, Dec 6, 2017 | groups.google.com/d/msg/fishcooking/AE4EgWQ20dY | Stockfish MCTS experiment | web; verify link |
| lc0sfhybrid | Black, lc0-stockfish-hybrid (Stockfish as a filter in Lc0's selection) | github.com/benblack769/lc0-stockfish-hybrid | Hobbyist hybrid precedent | software |
| ceres | Elliott, Ceres (C# MCTS using Lc0 backends) | github/CPW | Alternative MCTS stack | software |
| allie | Treat, Allie / AllieStein (own PUCT search on Lc0 nets, TCEC S15) | CPW "Allie" | Alternative MCTS stack | software |
| pascutto2025 | Pascutto, Sin dolor no hay gloria: on conquering 2 World Computer Chess Championships in Spain, ICGA Journal 47(3) 136–156, 2025 | 10.1177/13896911251360654 | Stoofvlees II: MCTS + CNN + classical evaluator; WCCC 2023 winner, 2024 shared | ID verified 2026-10-14 (SAGE) |
| carlson2026 | Carlson, Matilda: engine-agnostic search with human policy guidance | arXiv:2606.25176 | Frozen human policy + engine search with a residual correction (adjacent) | ID verified 2026-10-14 (arXiv) |
| maia2 | Tang et al., Maia-2: a unified model for human-AI alignment in chess, NeurIPS 2024 | arXiv:2409.20553 | Adjacent (human-like play) | ID verified 2026-10-14 (arXiv, NeurIPS poster page) |

## C. Evaluation methodology

| Key | Reference | Identifier | Use | Status |
| --- | --- | --- | --- | --- |
| vdbergh_gsprt | Van den Bergh, A practical introduction to the GSPRT (note) | cantate.be/Fishtest/GSPRT_approximation.pdf (archive.org/details/fishtest_mathematics) | GSPRT as used by Fishtest; random-walk model (formula 2.1) | Title/author verified 2026-10-14 from the PDF (first page); hardy.uhasselt.be mirror unreachable that day |
| vdbergh_rw | Van den Bergh, Notes on continuous random walks (note) | cantate.be/Fishtest/brownian_approximation.pdf | Elo from a finished GSPRT (formula 6.1) | Title/author verified 2026-10-14 from the PDF; the earlier entry title ("generalized maximum likelihood ratio…") was not located — do not cite it |
| vdbergh_nelo | Van den Bergh, Comments on normalized Elo (note; refers to the normalized-Elo definition as its ref. [4]) | cantate.be/Fishtest/normalized_elo_practical.pdf | nElo in practice; pentanomial variance | Title/author verified 2026-10-14 from the PDF; cite together with the Fishtest Mathematics wiki page |
| fishtest | Fishtest wiki ("Fishtest Mathematics": pentanomial model, GSPRT, normalized Elo) and `vdbergh/pentanomial`, `vdbergh/simul` | github.com/official-stockfish/fishtest/wiki/Fishtest-Mathematics | Testing framework conventions (game pairs, books, adjudication, nElo bounds) | Page verified 2026-10-14; it links the Van den Bergh documents on cantate.be with archive.org mirrors |
| sfbooks | official-stockfish/books repository (commit 65815cc, 2025-12-20): `UHO_Lichess_4852_v1.epd`, 2,632,036 positions (books.json: 1,457,270 white-to-move / 1,174,766 black-to-move, depth 2–16) | github.com/official-stockfish/books | The opening book used by every cell and gate in this paper (Fishtest's UHO Lichess book) | Verified locally 2026-10-14 (clone + books.json); cite the repository and commit |
| pohl2024 | Pohl, UHO 2024 (Unbalanced Human Openings), sp-cc.de | sp-cc.de/uho_2024.htm | Book construction (Megabase 2024, ≥2200 players, KomodoDragon 3.3 scoring), recommended draw ratio 45–60%, eval bands | Page verified 2026-10-14; the `UHO_Lichess_4852_v1` book used here is a separate UHO variant — source page to confirm |
| uhocritique | Talkchess, "The trouble with UHO" (Kaufman and others) | talkchess.com/viewtopic.php?p=939914 | Book-dependence critique | web |
| coulom2008 | Coulom, Whole-history rating: a Bayesian rating system for players of time-varying strength, Computers and Games 2008, LNCS 5131, 113–124 | 10.1007/978-3-540-87608-3_11 | Rating methods (do not misuse) | ID verified 2026-10-14 (Crossref) |
| bayeselo | Coulom, Bayesian Elo rating | remi-coulom.fr/Bayesian-Elo | Rating tool | web |
| ordo | Ballicora, Ordo 1.2.6 | github.com/michiguel/Ordo | Rating tool used | software |
| szczecinski2020 | Szczecinski & Djebbi, Understanding draws in Elo rating algorithm, J. Quant. Anal. Sports 16(3), 2020 | arXiv:1910.06081 | Draw modelling | verify journal cite |
| glickman1999 | Glickman, Parameter estimation in large dynamic paired comparison experiments, Appl. Stat. 48(3) 377–394, 1999 | 10.1111/1467-9876.00159 | Rating uncertainty | verify |
| ccrl | CCRL 40/15 "About" (reference machine, time-control scaling) | computerchess.org.uk/4040/about.html | Why cross-hardware Elo import is invalid | verified page |
| tcec | TCEC hardware (ICGA Journal reports; S17 split CPU/GPU qualification) | ICGA J articles (verify) | GPU vs CPU fairness precedent | verify |
| levy | Hardware-doubling heuristic (50–70 Elo per doubling), forum-level | — | Background only | web |
| bratko1982 | Bratko & Kopec, Performance analysis of a chess engine advice-giving program, Advances in Computer Chess 3, 1982 | — | BK suite origin | known |
| sts | Corbit & Swaminathan, Strategic Test Suite; Mosca's STS-Rating | sites.google.com/site/strategictestsuite; github.com/fsmosca/STS-Rating | STS | web |
| eret | Eigenmann, Eigenmann Rapid Engine Test, Glarean Magazin, 2017 | glarean-magazin.ch | ERET (111) | web |
| arasan | Dart, Arasan test suite (2023 revision) | github.com/jdart1/arasan-chess | Arasan 2023 | software |
| lichesspuzzles | Lichess open puzzle database | database.lichess.org | Puzzle evaluation | web |

## D. Hardware and inference

| Key | Reference | Identifier | Use | Status |
| --- | --- | --- | --- | --- |
| applem2 | Apple, M2 Max technical overview; unified memory architecture | apple.com newsroom | Platform | web |
| mpsgraph | Apple, Metal Performance Shaders Graph documentation (compiled executables) | developer.apple.com/documentation/metalperformanceshadersgraph | Backend | web |
| mpslatency2026 | Non-monotonic latency in Apple MPS decoding (preprint) | arXiv:2605.08913 | Adjacent evidence on MPS latency regimes | verify |
| fusionml2026 | FusionML: prefill, not decode — CPU+GPU co-execution on unified-memory Apple Silicon | arXiv:2607.22785 | Adjacent: CPU/GPU co-execution on unified memory | verify |
| frameworktax2023 | Fernandez et al., The framework tax: disparities between inference efficiency in NLP research and deployment | arXiv:2302.06117 | Small-batch overhead argument | verify |
| mlxbench2025 | Benchmarking on-device ML on Apple Silicon with MLX | arXiv:2510.18921 | Adjacent | verify |

## E. What the earlier chapter cited (keep/replace)

Keep: knuth1975, nasu2018, silver2018, vaswani2017 (Attention is all you need, NeurIPS
2017), rosin2011, browne2012, baier2018 (fix the characterisation), lanctot2014 (fix the
domain description), bratko1982, uci spec, lc0, stockfish, apple docs. Replace
`coulom2008elo` as the "method" citation (the method was not WHR). Drop `ccrl` as a
source of comparable ratings; cite it only for methodology.
