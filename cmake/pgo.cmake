# Profile-guided optimisation for the metalfish executable (clang/LLVM
# instrumentation).
#
# -DMETALFISH_PGO=generate                   instrumented build; run
# tools/build_pgo.sh to train -DMETALFISH_PGO=use
# -DMETALFISH_PGO_PROFILE=<file.profdata>   optimised build using the profile
#
# Only affects the `metalfish` target. Off by default (METALFISH_PGO=off).
set(METALFISH_PGO
    "off"
    CACHE STRING "PGO mode for the metalfish target: off|generate|use")
set(METALFISH_PGO_PROFILE
    ""
    CACHE FILEPATH "Merged .profdata file for METALFISH_PGO=use")

if(METALFISH_PGO STREQUAL "generate")
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "METALFISH_PGO=generate requires clang")
  endif()
  target_compile_options(metalfish PRIVATE -fprofile-instr-generate)
  target_link_options(metalfish PRIVATE -fprofile-instr-generate)
  message(STATUS "PGO: instrumented build (METALFISH_PGO=generate)")
elseif(METALFISH_PGO STREQUAL "use")
  if(NOT EXISTS "${METALFISH_PGO_PROFILE}")
    message(
      FATAL_ERROR
        "METALFISH_PGO=use needs METALFISH_PGO_PROFILE=<existing .profdata>")
  endif()
  target_compile_options(
    metalfish
    PRIVATE "-fprofile-instr-use=${METALFISH_PGO_PROFILE}"
            -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date
            -Wno-backend-plugin)
  target_link_options(metalfish PRIVATE
                      "-fprofile-instr-use=${METALFISH_PGO_PROFILE}")
  message(STATUS "PGO: using profile ${METALFISH_PGO_PROFILE}")
elseif(NOT METALFISH_PGO STREQUAL "off")
  message(
    FATAL_ERROR
      "METALFISH_PGO must be off, generate or use (got '${METALFISH_PGO}')")
endif()
