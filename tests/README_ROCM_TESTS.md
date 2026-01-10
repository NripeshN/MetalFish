# ROCm Backend Test Suite

This directory contains comprehensive tests for the ROCm/HIP GPU backend implementation.

## Test Files

### test_rocm.cpp
Integration tests for ROCm backend, similar to the Metal test suite. Tests the complete backend functionality in real-world scenarios.

**Test Coverage:**
- GPU backend availability and initialization
- Device information queries
- Buffer creation and management
- Memory access patterns (unified memory)
- Buffer creation with initial data
- Multiple memory modes (Shared, Private, Managed)
- Memory tracking and statistics
- Command encoder creation and usage
- GPU synchronization primitives
- GPU operations integration (NNUE, batch operations)
- Kernel management (loading pre-compiled kernels)
- Buffer read/write operations
- Command submission (sync and async)
- Memory barriers

### test_rocm_unit.cpp
Unit tests for individual ROCm backend components. Tests specific functionality and edge cases.

**Test Coverage:**
- Buffer lifecycle (allocation/deallocation)
- Memory modes behavior verification
- Buffer data integrity across data types
- Memory tracking accuracy
- Command encoder creation and management
- Synchronization primitives
- Buffer usage patterns (Default, Transient, Persistent, Streaming)
- Edge cases (zero-size, large buffers, type conversions)
- Concurrent operations (multiple buffers and encoders)
- Device information consistency
- Buffer read/write patterns (sequential, random)
- Memory barrier behavior

## Running Tests

### With ROCm Support

```bash
# Build with ROCm enabled
cd metalfish
cmake -B build -DUSE_ROCM=ON -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# Run all tests
./build/metalfish_tests

# Run specific test
./build/metalfish_tests # Individual tests run automatically
```

### Without ROCm (CPU Fallback)

```bash
# Build without ROCm
cmake -B build -DUSE_ROCM=OFF -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# Tests will be skipped gracefully
./build/metalfish_tests
```

## Test Output

When ROCm is available, you should see output like:

```
=== Testing ROCm GPU Backend ===
GPU Backend: ROCm/HIP
Device: AMD Radeon RX 7900 XTX
Unified Memory: No
Max Buffer Size: 24564 MB
Max Threadgroup Memory: 65536 bytes

=== Testing Buffer Creation ===
Buffer creation: SUCCESS

=== Testing Memory Access ===
Unified memory: Not available (discrete GPU)

=== Testing Buffer with Initial Data ===
Buffer with data creation: SUCCESS

... (more tests)

All ROCm tests passed!

=== ROCm Backend Unit Tests ===
  Testing buffer lifecycle... PASSED
  Testing memory modes... PASSED
  Testing buffer data integrity... PASSED
  Testing memory tracking... PASSED
  ... (more unit tests)

ROCm Unit Tests: 12 passed, 0 failed
```

## Test Requirements

### Hardware
- AMD GPU with GCN 3.0+ architecture
- Radeon RX 6000/7000 series, Radeon Pro, or Instinct MI series
- Ryzen APU with RDNA graphics (for unified memory tests)

### Software
- ROCm 5.0 or later
- HIP runtime libraries
- Properly configured GPU drivers

### Without Hardware
Tests will gracefully skip when:
- ROCm is not compiled (`USE_ROCM=OFF`)
- ROCm runtime is not available
- No AMD GPU is detected

## Test Structure

Each test follows this pattern:

```cpp
bool test_feature() {
  std::cout << "  Testing feature..." << std::flush;
  
  try {
    // Setup
    ROCmTestFixture fixture;
    
    // Test logic
    assert(condition);
    
    // Cleanup (automatic via RAII)
    
    std::cout << " PASSED" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << " FAILED: " << e.what() << std::endl;
    return false;
  }
}
```

## Adding New Tests

To add a new ROCm test:

1. Add test function to `test_rocm_unit.cpp`:
```cpp
bool test_new_feature() {
  std::cout << "  Testing new feature..." << std::flush;
  
  ROCmTestFixture fixture;
  
  // Your test code here
  assert(your_condition);
  
  std::cout << " PASSED" << std::endl;
  return true;
}
```

2. Register in the test array:
```cpp
Test tests[] = {
  // ... existing tests ...
  {"New Feature", ROCmTests::test_new_feature},
};
```

3. Rebuild and run tests.

## Debugging Tests

### Verbose Output
Tests automatically output progress. For more details, add debug prints:

```cpp
std::cout << "Debug: value = " << value << std::endl;
```

### Running Under Debugger
```bash
gdb --args ./build/metalfish_tests
```

### ROCm Profiling
```bash
rocprof --stats ./build/metalfish_tests
```

### Memory Checking
```bash
rocm-smi -d  # Monitor GPU memory during tests
```

## Known Limitations

1. **Runtime Compilation**: HIP runtime compilation (hipRTC) tests are currently skipped as this feature is not yet implemented.

2. **Kernel Execution**: Kernel execution tests require pre-compiled HIP kernels, which need to be built separately.

3. **Unified Memory**: Some tests are skipped on discrete GPUs that don't support unified memory.

4. **CI/CD**: Tests may not run in CI environments without AMD GPU hardware. They gracefully skip in such cases.

## Test Maintenance

- Keep tests aligned with Metal backend tests for consistency
- Update tests when adding new ROCm backend features
- Ensure tests pass on both APU and discrete GPU configurations
- Test on multiple AMD GPU architectures when possible

## Troubleshooting

### Test Fails with "ROCm not available"
- Verify ROCm installation: `rocminfo`
- Check user in render group: `groups $USER`
- Verify GPU detected: `rocm-smi`

### Memory Allocation Failures
- Check available GPU memory: `rocm-smi`
- Reduce test buffer sizes if needed
- Close other GPU applications

### Compilation Errors
- Ensure ROCm is in PATH: `which hipcc`
- Set ROCm path: `export ROCM_PATH=/opt/rocm`
- Check CMake configuration: `cmake -B build -DUSE_ROCM=ON`

## Reference

For more information:
- See `src/gpu/rocm/README.md` for ROCm backend documentation
- See `tests/test_metal.cpp` for Metal backend test reference
- ROCm documentation: https://rocmdocs.amd.com/
