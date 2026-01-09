/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "search/tune.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>

#include "uci/ucioption.h"

using std::string;

namespace MetalFish {

bool          Tune::update_on_last;
const Option* LastOption = nullptr;
OptionsMap*   Tune::options;
namespace {
std::map<std::string, int> TuneResults;

std::optional<std::string> on_tune(const Option& o) {

    if (!Tune::update_on_last || LastOption == &o)
        Tune::read_options();

    return std::nullopt;
}
}

void Tune::make_option(OptionsMap* opts, const string& n, int v, const SetRange& r) {

    // Do not generate option when there is nothing to tune (ie. min = max)
    if (r(v).first == r(v).second)
        return;

    if (TuneResults.count(n))
        v = TuneResults[n];

    opts->add(n, Option(v, r(v).first, r(v).second, on_tune));
    LastOption = &((*opts)[n]);

    // Print formatted parameters, ready to be copy-pasted in Fishtest
    std::cout << n << ","                                  //
              << v << ","                                  //
              << r(v).first << ","                         //
              << r(v).second << ","                        //
              << (r(v).second - r(v).first) / 20.0 << ","  //
              << "0.0020" << std::endl;
}

string Tune::next(string& names, bool pop) {

    string name;

    do
    {
        string token = names.substr(0, names.find(','));

        if (pop)
            names.erase(0, token.size() + 1);

        std::stringstream ws(token);
        name += (ws >> token, token);  // Remove trailing whitespace

    } while (std::count(name.begin(), name.end(), '(') - std::count(name.begin(), name.end(), ')'));

    return name;
}


template<>
void Tune::Entry<int>::init_option() {
    make_option(options, name, value, range);
}

template<>
void Tune::Entry<int>::read_option() {
    if (options->count(name))
        value = int((*options)[name]);
}

// Instead of a variable here we have a PostUpdate function: just call it
template<>
void Tune::Entry<Tune::PostUpdate>::init_option() {}
template<>
void Tune::Entry<Tune::PostUpdate>::read_option() {
    value();
}

}  // namespace MetalFish


// Init options with tuning session results instead of default values. Useful to
// get correct bench signature after a tuning session or to test tuned values.
// Just copy fishtest tuning results in a result.txt file and extract the
// values with:
//
// cat results.txt | sed 's/^param: \([^,]*\), best: \([^,]*\).*/  TuneResults["\1"] = int(round(\2));/'
//
// Then paste the output below, as the function body


namespace MetalFish {

void Tune::read_results() { /* ...insert your values here... */ }

}  // namespace MetalFish
