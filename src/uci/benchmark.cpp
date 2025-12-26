/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "search/search.h"
#include "search/tt.h"
#include "uci/uci.h"
#include <iostream>
#include <vector>

namespace MetalFish {

namespace UCI {

// Standard benchmark positions (same as Stockfish)
const std::vector<std::string> BenchPositions = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbqkb1r/ppppp1pp/5n2/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "r1bqk1nr/ppppbppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqk2r/pppp1ppp/2n1pn2/8/1bPP4/2N1P3/PP3PPP/R1BQKBNR w KQkq - 2 5",
    "r2q1rk1/2p1bppp/p2p1n2/1p2P3/4P1b1/1nP1BN2/PP3PPP/RN1QR1K1 w - - 1 12",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r1bqk2r/ppp2ppp/2n1pn2/3p4/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 2 6",
    "r1b1kb1r/pp1n1ppp/1qn1p3/3pP3/3P4/2N2N2/PP3PPP/R1BQKB1R w KQkq - 0 8",
    "r1bq1rk1/ppp1nppp/4p3/3pP3/3Pn3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 1 10"};

// Extended benchmark with more positions
const std::vector<std::string> ExtendedBenchPositions = {
    // Tactical positions
    "r2qk2r/pp1n1ppp/2pbpn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w kq - 2 9",
    "r1bq1rk1/pp2nppp/2n1p3/2ppP3/3P4/2P2N2/PP3PPP/RNBQR1K1 w - c6 0 10",

    // Endgame positions
    "8/8/4k3/8/2p5/8/1P2K3/8 w - - 0 1", "8/p7/1p6/1P6/5k2/8/5K2/8 w - - 0 1",
    "8/2k5/8/8/8/8/R7/4K3 w - - 0 1",

    // Complex middlegame
    "r1bqr1k1/ppp2ppp/2nb1n2/3pp3/8/2NPBN2/PPP1BPPP/R2Q1RK1 w - - 0 9"};

} // namespace UCI

} // namespace MetalFish
