/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  
  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

#include "core/types.h"
#include "core/position.h"
#include "search/search.h"
#include <string>
#include <map>
#include <functional>

namespace MetalFish {

namespace UCI {

// UCI option types
enum OptionType {
    OPT_CHECK,
    OPT_SPIN,
    OPT_COMBO,
    OPT_BUTTON,
    OPT_STRING
};

// UCI Option class
class Option {
public:
    using OnChange = std::function<void(const Option&)>;

    Option(OnChange = nullptr);
    Option(bool v, OnChange = nullptr);
    Option(const char* v, OnChange = nullptr);
    Option(double v, double min, double max, OnChange = nullptr);
    Option(const char* v, const char* cur, OnChange = nullptr);

    Option& operator=(const std::string&);
    void operator<<(const Option&);
    
    operator int() const;
    operator std::string() const;
    bool operator==(const char*) const;

    std::string defaultValue, currentValue, type;
    int min, max;
    size_t idx;
    OnChange on_change;
};

// UCI Options map
class OptionsMap {
public:
    void init();
    bool contains(const std::string& name) const;
    Option& operator[](const std::string& name);
    const Option& operator[](const std::string& name) const;
    
    auto begin() { return options.begin(); }
    auto end() { return options.end(); }

private:
    std::map<std::string, Option> options;
};

extern OptionsMap Options;

// UCI command handlers
void loop(int argc, char* argv[]);
void uci();
void setoption(std::istringstream& is);
void position(Position& pos, std::istringstream& is, StateListPtr& states);
void go(Position& pos, std::istringstream& is, StateListPtr& states);
void bench(Position& pos, std::istringstream& is, StateListPtr& states);

// Move conversion utilities
std::string move_to_uci(Move m, bool chess960);
Move uci_to_move(const Position& pos, const std::string& str);
std::string square_to_string(Square s);
std::string score_to_string(Value v);
std::string pv_to_string(const std::vector<Move>& pv, bool chess960);

} // namespace UCI

} // namespace MetalFish

