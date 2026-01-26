/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file thread_win32_osx.h
 * @brief MetalFish source file.
 */

  Licensed under GPL-3.0
*/

#ifndef THREAD_WIN32_OSX_H_INCLUDED
#define THREAD_WIN32_OSX_H_INCLUDED

#include <thread>

// On OSX threads other than the main thread are created with a reduced stack
// size of 512KB by default, this is too low for deep searches, which require
// somewhat more than 1MB stack, so adjust it to TH_STACK_SIZE.
// The implementation calls pthread_create() with the stack size parameter
// equal to the Linux 8MB default, on platforms that support it.

#if defined(__APPLE__) || defined(__MINGW32__) || defined(__MINGW64__) ||      \
    defined(USE_PTHREADS)

#include <functional>
#include <pthread.h>

namespace MetalFish {

class NativeThread {
  pthread_t thread;

  static constexpr size_t TH_STACK_SIZE = 8 * 1024 * 1024;

public:
  template <class Function, class... Args>
  explicit NativeThread(Function &&fun, Args &&...args) {
    auto func = new std::function<void()>(
        std::bind(std::forward<Function>(fun), std::forward<Args>(args)...));

    pthread_attr_t attr_storage, *attr = &attr_storage;
    pthread_attr_init(attr);
    pthread_attr_setstacksize(attr, TH_STACK_SIZE);

    auto start_routine = [](void *ptr) -> void * {
      auto f = reinterpret_cast<std::function<void()> *>(ptr);
      // Call the function
      (*f)();
      delete f;
      return nullptr;
    };

    pthread_create(&thread, attr, start_routine, func);
  }

  void join() { pthread_join(thread, nullptr); }
};

} // namespace MetalFish

#else // Default case: use STL classes

namespace MetalFish {

using NativeThread = std::thread;

} // namespace MetalFish

#endif

#endif // #ifndef THREAD_WIN32_OSX_H_INCLUDED