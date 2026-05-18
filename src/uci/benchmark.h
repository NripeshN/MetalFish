/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef BENCHMARK_H_INCLUDED
#define BENCHMARK_H_INCLUDED

#include <iosfwd>
#include <string>
#include <vector>

namespace MetalFish::Benchmark {

std::vector<std::string> setup_bench(const std::string &, std::istream &);

struct BenchmarkSetup {
  int ttSize;
  int threads;
  std::vector<std::string> commands;
  std::string originalInvocation;
  std::string filledInvocation;
};

BenchmarkSetup setup_benchmark(std::istream &);

} // namespace MetalFish::Benchmark

#endif // #ifndef BENCHMARK_H_INCLUDED
