/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  
  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "uci/uci.h"
#include "search/tt.h"
#include "core/movegen.h"
#include "core/perft.h"
#include "eval/evaluate.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>

namespace MetalFish {

namespace UCI {

OptionsMap Options;

// Option implementations
Option::Option(OnChange f) : on_change(f), type("button"), min(0), max(0), idx(0) {}

Option::Option(bool v, OnChange f) : on_change(f), type("check"), min(0), max(0), idx(0) {
    defaultValue = currentValue = (v ? "true" : "false");
}

Option::Option(const char* v, OnChange f) : on_change(f), type("string"), min(0), max(0), idx(0) {
    defaultValue = currentValue = v;
}

Option::Option(double v, double minv, double maxv, OnChange f)
    : on_change(f), type("spin"), min(int(minv)), max(int(maxv)), idx(0) {
    defaultValue = currentValue = std::to_string(int(v));
}

Option& Option::operator=(const std::string& v) {
    if (type == "button") {
        if (on_change)
            on_change(*this);
    } else {
        currentValue = v;
        if (on_change)
            on_change(*this);
    }
    return *this;
}

Option::operator int() const {
    return type == "check" ? (currentValue == "true") : std::stoi(currentValue);
}

Option::operator std::string() const {
    return currentValue;
}

bool Option::operator==(const char* s) const {
    return currentValue == s;
}

void OptionsMap::init() {
    // Initialize UCI options
    options["Hash"] = Option(64, 1, 33554432, Option::OnChange([](const Option& o) {
        TT.resize(int(o));
    }));
    
    options["Clear Hash"] = Option(Option::OnChange([](const Option&) {
        TT.clear();
    }));
    
    options["Threads"] = Option(1, 1, 512);
    options["UCI_Chess960"] = Option(false);
    options["MultiPV"] = Option(1, 1, 500);
    
    // GPU-specific options
    options["GPU_BatchSize"] = Option(64, 1, 4096);
    options["GPU_UseUnifiedMemory"] = Option(true);
}

bool OptionsMap::contains(const std::string& name) const {
    return options.find(name) != options.end();
}

Option& OptionsMap::operator[](const std::string& name) {
    return options[name];
}

const Option& OptionsMap::operator[](const std::string& name) const {
    return options.at(name);
}

// Square to string
std::string square_to_string(Square s) {
    return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

// Move to UCI string
std::string move_to_uci(Move m, bool chess960) {
    if (m == Move::none())
        return "(none)";
    if (m == Move::null())
        return "0000";

    Square from = m.from_sq();
    Square to = m.to_sq();

    if (m.type_of() == CASTLING && !chess960)
        to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));

    std::string str = square_to_string(from) + square_to_string(to);

    if (m.type_of() == PROMOTION)
        str += " nbrq"[m.promotion_type()];

    return str;
}

// UCI string to move
Move uci_to_move(const Position& pos, const std::string& str) {
    if (str.length() < 4)
        return Move::none();

    Square from = make_square(File(str[0] - 'a'), Rank(str[1] - '1'));
    Square to = make_square(File(str[2] - 'a'), Rank(str[3] - '1'));

    // Check for valid squares
    if (!is_ok(from) || !is_ok(to))
        return Move::none();

    // Find the move in legal moves
    for (const auto& m : MoveList<LEGAL>(pos)) {
        if (m.from_sq() == from && m.to_sq() == to) {
            // Check promotion
            if (m.type_of() == PROMOTION) {
                if (str.length() < 5)
                    continue;
                char promo = str[4];
                if ((promo == 'n' && m.promotion_type() != KNIGHT) ||
                    (promo == 'b' && m.promotion_type() != BISHOP) ||
                    (promo == 'r' && m.promotion_type() != ROOK) ||
                    (promo == 'q' && m.promotion_type() != QUEEN))
                    continue;
            }
            return m;
        }
        // Castling in UCI notation
        if (m.type_of() == CASTLING) {
            Square kto = relative_square(pos.side_to_move(), m.to_sq() > m.from_sq() ? SQ_G1 : SQ_C1);
            if (m.from_sq() == from && kto == to)
                return m;
        }
    }

    return Move::none();
}

// Score to string
std::string score_to_string(Value v) {
    if (std::abs(v) >= VALUE_MATE_IN_MAX_PLY) {
        int ply = VALUE_MATE - std::abs(v);
        int moves = (ply + 1) / 2;
        return (v > 0 ? "mate " : "mate -") + std::to_string(moves);
    }
    return "cp " + std::to_string(v);
}

// PV to string
std::string pv_to_string(const std::vector<Move>& pv, bool chess960) {
    std::string str;
    for (const auto& m : pv) {
        if (!str.empty())
            str += " ";
        str += move_to_uci(m, chess960);
    }
    return str;
}

// UCI command handlers
void uci() {
    std::cout << "id name MetalFish 1.0.0\n";
    std::cout << "id author Nripesh Niketan\n\n";
    
    for (auto& [name, opt] : Options) {
        std::cout << "option name " << name << " type " << opt.type;
        if (opt.type == "spin")
            std::cout << " default " << opt.defaultValue << " min " << opt.min << " max " << opt.max;
        else if (opt.type == "check")
            std::cout << " default " << opt.defaultValue;
        else if (opt.type == "string")
            std::cout << " default " << opt.defaultValue;
        std::cout << "\n";
    }
    
    std::cout << "uciok" << std::endl;
}

void setoption(std::istringstream& is) {
    std::string token, name, value;
    
    is >> token;  // "name"
    
    while (is >> token && token != "value")
        name += (name.empty() ? "" : " ") + token;
    
    while (is >> token)
        value += (value.empty() ? "" : " ") + token;
    
    if (Options.contains(name))
        Options[name] = value;
}

void position(Position& pos, std::istringstream& is, StateListPtr& states) {
    std::string token, fen;
    
    is >> token;
    
    if (token == "startpos") {
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        is >> token;  // Consume "moves" if present
    }
    else if (token == "fen") {
        while (is >> token && token != "moves")
            fen += token + " ";
    }
    
    states = std::make_unique<std::deque<StateInfo>>(1);
    pos.set(fen, Options["UCI_Chess960"] == "true", &states->back());
    
    // Apply moves
    while (is >> token) {
        Move m = uci_to_move(pos, token);
        if (m == Move::none())
            break;
        
        states->emplace_back();
        pos.do_move(m, states->back());
    }
}

void go(Position& pos, std::istringstream& is, StateListPtr& states) {
    Search::LimitsType limits;
    std::string token;
    
    while (is >> token) {
        if (token == "wtime")       is >> limits.time[WHITE];
        else if (token == "btime")  is >> limits.time[BLACK];
        else if (token == "winc")   is >> limits.inc[WHITE];
        else if (token == "binc")   is >> limits.inc[BLACK];
        else if (token == "movestogo") is >> limits.movestogo;
        else if (token == "depth")  is >> limits.depth;
        else if (token == "nodes")  is >> limits.nodes;
        else if (token == "movetime") is >> limits.movetime;
        else if (token == "mate")   is >> limits.mate;
        else if (token == "infinite") limits.infinite = 1;
        else if (token == "ponder")   limits.ponderMode = true;
        else if (token == "perft") {
            int depth = 1;
            is >> depth;
            auto start = std::chrono::steady_clock::now();
            uint64_t nodes = perft(pos, depth, true);
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "\nNodes searched: " << nodes << std::endl;
            if (elapsed > 0) {
                std::cout << "Time: " << elapsed << " ms" << std::endl;
                std::cout << "NPS: " << (nodes * 1000 / elapsed) << std::endl;
            }
            return;
        }
        else if (token == "searchmoves") {
            while (is >> token) {
                Move m = uci_to_move(pos, token);
                if (m != Move::none())
                    limits.searchmoves.push_back(m);
            }
        }
    }
    
    // Create a worker and start searching
    static Search::Worker worker;
    worker.clear();
    
    Search::Signals_stop = false;
    
    // Run search in a separate thread
    std::thread([&pos, limits, &states]() mutable {
        worker.start_searching(pos, limits, states);
        
        // Output best move
        if (!worker.rootMoves.empty()) {
            std::cout << "bestmove " << move_to_uci(worker.rootMoves[0].pv[0], false);
            if (worker.rootMoves[0].pv.size() > 1)
                std::cout << " ponder " << move_to_uci(worker.rootMoves[0].pv[1], false);
            std::cout << std::endl;
        }
    }).detach();
}

void bench(Position& pos, std::istringstream& is, StateListPtr& states) {
    int depth = 12;
    is >> depth;
    
    // Standard benchmark positions
    const char* benchPositions[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
    };
    
    uint64_t totalNodes = 0;
    auto startTime = Search::now();
    
    for (const char* fen : benchPositions) {
        states = std::make_unique<std::deque<StateInfo>>(1);
        pos.set(fen, false, &states->back());
        
        Search::LimitsType limits;
        limits.depth = depth;
        
        Search::Worker worker;
        worker.clear();
        Search::Signals_stop = false;
        worker.start_searching(pos, limits, states);
        worker.wait_for_search_finished();
        
        totalNodes += worker.nodes.load();
    }
    
    auto elapsed = Search::now() - startTime;
    
    std::cout << "\n==========================="
              << "\nTotal time (ms) : " << elapsed
              << "\nNodes searched  : " << totalNodes
              << "\nNodes/second    : " << (elapsed > 0 ? totalNodes * 1000 / elapsed : 0)
              << std::endl;
}

// Main UCI loop
void loop(int argc, char* argv[]) {
    // Initialize options
    Options.init();
    
    // Initialize hash table
    TT.resize(int(Options["Hash"]));
    
    // Initialize search
    Search::init();
    
    Position pos;
    StateListPtr states = std::make_unique<std::deque<StateInfo>>(1);
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &states->back());
    
    std::string line, token;
    
    // Process command line arguments
    for (int i = 1; i < argc; ++i) {
        line += std::string(argv[i]) + " ";
    }
    
    // Main loop
    while (true) {
        if (argc == 1 || line.empty()) {
            if (!std::getline(std::cin, line))
                break;
        }
        
        std::istringstream is(line);
        line.clear();
        
        is >> std::skipws >> token;
        
        if (token == "quit" || token == "stop") {
            Search::Signals_stop = true;
            if (token == "quit")
                break;
        }
        else if (token == "uci")
            uci();
        else if (token == "setoption")
            setoption(is);
        else if (token == "isready")
            std::cout << "readyok" << std::endl;
        else if (token == "ucinewgame") {
            Search::clear();
            states = std::make_unique<std::deque<StateInfo>>(1);
            pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &states->back());
        }
        else if (token == "position")
            position(pos, is, states);
        else if (token == "go")
            go(pos, is, states);
        else if (token == "bench")
            bench(pos, is, states);
        else if (token == "d")
            std::cout << pos << std::endl;
        else if (token == "eval")
            std::cout << "Evaluation: " << Eval::evaluate(pos) << std::endl;
    }
}

} // namespace UCI

} // namespace MetalFish
