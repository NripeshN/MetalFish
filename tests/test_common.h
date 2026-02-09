/*
  MetalFish Test Framework
  Shared test utilities -- single source of truth for TestCase, EXPECT, etc.
*/

#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace MetalFish {
namespace Test {

struct TestCase {
  std::string name;
  bool passed = true;
  int assertions = 0;
  int failures = 0;
  std::vector<std::string> failure_messages;

  void fail(const std::string &msg) {
    passed = false;
    failures++;
    failure_messages.push_back(msg);
  }

  void check(bool condition, const std::string &msg) {
    assertions++;
    if (!condition) {
      fail(msg);
    }
  }

  void print_result() const {
    if (passed) {
      std::cout << "  PASS: " << name << " (" << assertions << " assertions)"
                << std::endl;
    } else {
      std::cout << "  FAIL: " << name << " (" << failures << "/" << assertions
                << " failed)" << std::endl;
      for (const auto &msg : failure_messages) {
        std::cout << "    - " << msg << std::endl;
      }
    }
  }
};

#define EXPECT(tc, cond)                                                       \
  do {                                                                         \
    (tc).check((cond), std::string(__FILE__) + ":" +                           \
                           std::to_string(__LINE__) + ": " + #cond);           \
  } while (0)

#define EXPECT_EQ(tc, a, b)                                                    \
  do {                                                                         \
    (tc).check((a) == (b), std::string(__FILE__) + ":" +                       \
                               std::to_string(__LINE__) + ": " + #a +          \
                               " == " + #b);                                   \
  } while (0)

#define EXPECT_NE(tc, a, b)                                                    \
  do {                                                                         \
    (tc).check((a) != (b), std::string(__FILE__) + ":" +                       \
                               std::to_string(__LINE__) + ": " + #a +          \
                               " != " + #b);                                   \
  } while (0)

#define EXPECT_GT(tc, a, b)                                                    \
  do {                                                                         \
    (tc).check((a) > (b), std::string(__FILE__) + ":" +                        \
                              std::to_string(__LINE__) + ": " + #a + " > " +   \
                              #b);                                             \
  } while (0)

#define EXPECT_GE(tc, a, b)                                                    \
  do {                                                                         \
    (tc).check((a) >= (b), std::string(__FILE__) + ":" +                       \
                               std::to_string(__LINE__) + ": " + #a +          \
                               " >= " + #b);                                   \
  } while (0)

#define EXPECT_NEAR(tc, a, b, eps)                                             \
  do {                                                                         \
    (tc).check(std::abs((a) - (b)) <= (eps),                                   \
               std::string(__FILE__) + ":" + std::to_string(__LINE__) +        \
                   ": |" + #a + " - " + #b + "| <= " + #eps);                  \
  } while (0)

// Run a named test section and track pass/fail
inline bool run_section(const std::string &name,
                        std::function<bool()> test_fn) {
  std::cout << "\n--- " << name << " ---" << std::endl;
  try {
    bool result = test_fn();
    if (result)
      std::cout << "  Section PASSED" << std::endl;
    else
      std::cout << "  Section FAILED" << std::endl;
    return result;
  } catch (const std::exception &e) {
    std::cout << "  Section CRASHED: " << e.what() << std::endl;
    return false;
  }
}

} // namespace Test
} // namespace MetalFish
