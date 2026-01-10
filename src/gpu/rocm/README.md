# ROCm/HIP Backend for MetalFish

This directory contains the ROCm/HIP backend implementation for MetalFish GPU acceleration.

## Overview

The ROCm backend provides GPU acceleration for AMD GPUs using the HIP (Heterogeneous-compute Interface for Portability) API. HIP is AMD's answer to CUDA and provides a C++ runtime API for GPU computing.

## Architecture

The implementation follows the same pattern as the Metal backend:

- **rocm_backend.cpp**: Main backend implementation
  - `ROCmBackend`: Singleton backend manager
  - `ROCmBuffer`: GPU memory buffer wrapper
  - `ROCmKernel`: Compute kernel (compiled HIP code)
  - `ROCmCommandEncoder`: Command recording and submission

- **kernels/**: HIP kernel implementations
  - `nnue_full.hip`: NNUE evaluation kernels

## Supported Hardware

### AMD GPUs
- **Discrete GPUs**: Radeon RX 6000/7000 series, Radeon Pro
- **APUs**: Ryzen 7000+ series with RDNA graphics
- **Data Center**: Instinct MI series

### Minimum Requirements
- GCN 3.0+ architecture (Fiji, Polaris, Vega, RDNA, RDNA2, RDNA3)
- ROCm 5.0 or later

## Building with ROCm Support

### Prerequisites

1. Install ROCm:
```bash
# Ubuntu/Debian
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_*_all.deb
sudo dpkg -i amdgpu-install_*_all.deb
sudo amdgpu-install --usecase=rocm

# Verify installation
rocminfo
```

2. Add user to render group:
```bash
sudo usermod -a -G render $USER
```

### Build MetalFish

```bash
cd metalfish
cmake -B build -DUSE_ROCM=ON
cmake --build build -j$(nproc)
```

## Implementation Status

| Feature | Status |
|---------|--------|
| Backend abstraction | ✓ Complete |
| Buffer management | ✓ Complete |
| Kernel management | ✓ Complete |
| Command encoding | ✓ Complete |
| Unified memory (APU) | ✓ Complete |
| Runtime compilation | Planned |
| NNUE kernels | Planned |
| Batch operations | Planned |

## Memory Models

The ROCm backend supports three memory modes:

1. **Shared**: Host-accessible memory (pinned)
   - Best for CPU-GPU data transfer
   - Zero-copy on APUs with unified memory

2. **Private**: Device-only memory
   - Fastest for GPU-only data
   - Requires explicit copies

3. **Managed**: Unified/managed memory
   - Automatic migration between CPU/GPU
   - Transparent access from both sides

## Performance Considerations

### APU (Unified Memory)
- Use `MemoryMode::Shared` for zero-copy access
- Minimize data movement
- CPU and GPU share physical memory

### Discrete GPU
- Use `MemoryMode::Private` for GPU-only data
- Batch transfers to minimize PCIe overhead
- Consider async transfers with streams

## Kernel Compilation

HIP kernels can be compiled:

1. **Offline** (recommended):
```bash
hipcc -c nnue_full.hip -o nnue_full.o
hipcc --genco nnue_full.hip -o nnue_full.co
```

2. **Runtime** (future):
Using hipRTC (ROCm Runtime Compiler)

## Testing

Run tests with ROCm backend:
```bash
./build/metalfish_tests
```

The test suite will automatically use the ROCm backend if available.

## Troubleshooting

### "No ROCm devices found"
- Check: `rocminfo` to verify GPU is detected
- Ensure user is in `render` group
- Verify ROCm drivers are loaded: `lsmod | grep amdgpu`

### Build fails with "hip not found"
- Ensure ROCm is installed: `which hipcc`
- Add to PATH: `export PATH=/opt/rocm/bin:$PATH`
- Set ROCm path: `export ROCM_PATH=/opt/rocm`

### Performance issues
- Check GPU clocks: `rocm-smi`
- Monitor usage: `rocm-smi -d`
- Enable GPU profiling: `rocprof ./metalfish`

## References

- [ROCm Documentation](https://rocmdocs.amd.com/)
- [HIP Programming Guide](https://rocmdocs.amd.com/projects/HIP/en/latest/)
- [AMD GPU Architecture](https://www.amd.com/en/technologies/rdna-3)

## Future Work

- [ ] Implement runtime kernel compilation with hipRTC
- [ ] Port NNUE evaluation kernels from Metal
- [ ] Optimize for RDNA3 architecture
- [ ] Add performance benchmarks
- [ ] Support multi-GPU configurations
