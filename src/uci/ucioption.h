/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef UCIOPTION_H_INCLUDED
#define UCIOPTION_H_INCLUDED

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>

namespace MetalFish {
// Define a custom comparator, because the UCI options should be
// case-insensitive
struct CaseInsensitiveLess {
  bool operator()(const std::string &, const std::string &) const;
};

class OptionsMap;

// The Option class implements each option as specified by the UCI protocol
class Option {
public:
  using OnChange = std::function<std::optional<std::string>(const Option &)>;

  Option(const OptionsMap *);
  Option(OnChange = nullptr);
  Option(bool v, OnChange = nullptr);
  Option(const char *v, OnChange = nullptr);
  Option(int v, int minv, int maxv, OnChange = nullptr);
  Option(const char *v, const char *cur, OnChange = nullptr);

  Option &operator=(const std::string &);
  operator int() const;
  operator std::string() const;
  bool operator==(const char *) const;
  bool operator!=(const char *) const;

  friend std::ostream &operator<<(std::ostream &, const OptionsMap &);

  int operator<<(const Option &) = delete;

private:
  friend class OptionsMap;
  friend class Engine;
  friend class Tune;

  std::string defaultValue, currentValue, type;
  int min, max;
  size_t idx;
  OnChange on_change;
  const OptionsMap *parent = nullptr;
};

class OptionsMap {
public:
  using InfoListener = std::function<void(std::optional<std::string>)>;

  OptionsMap() = default;
  OptionsMap(const OptionsMap &) = delete;
  OptionsMap(OptionsMap &&) = delete;
  OptionsMap &operator=(const OptionsMap &) = delete;
  OptionsMap &operator=(OptionsMap &&) = delete;

  void add_info_listener(InfoListener &&);

  void setoption(std::istringstream &);

  const Option &operator[](const std::string &) const;

  void add(const std::string &, const Option &option);

  std::size_t count(const std::string &) const;

private:
  friend class Engine;
  friend class Option;

  friend std::ostream &operator<<(std::ostream &, const OptionsMap &);

  // The options container is defined as a std::map
  using OptionsStore = std::map<std::string, Option, CaseInsensitiveLess>;

  OptionsStore options_map;
  InfoListener info;
};

} // namespace MetalFish
#endif // #ifndef UCIOPTION_H_INCLUDED
