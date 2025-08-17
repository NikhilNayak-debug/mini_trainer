# GPU Test Setup Guide

## Quick Start

The simplest way to run GPU tests is using the existing uv environment:

```bash
# From the mini_trainer root directory
./tests/gpu_tests/run_simple.sh diagnostic
```

## Installation Options

### Option 1: Using existing uv environment (Recommended)

```bash
# Install base dependencies
uv sync

# Optional: Install flash-attn for better performance
uv pip install flash-attn>=2.8.2

# Run tests
./tests/gpu_tests/run_simple.sh diagnostic
```

### Option 2: Using tox with automatic setup

```bash
# Tox will create isolated environments
uv run tox -e gpu-overfitting
```

**Note**: Flash-attn installation in tox environments may fail due to build dependencies. If this happens, tests will automatically fall back to eager attention.

## Flash Attention Installation

Flash-attn requires:
- PyTorch installed first
- CUDA 11.6+
- ninja build system
- Compatible GPU architecture (A100, H100, etc.)

### Manual installation:
```bash
# Ensure torch is installed
uv sync

# Install build dependencies
uv pip install ninja packaging psutil einops

# Install flash-attn (may take several minutes to compile)
uv pip install --no-build-isolation flash-attn>=2.8.2
```

### If flash-attn installation fails:
The tests will automatically fall back to eager attention mode, which is slower but functionally equivalent for testing purposes.

## Running Specific Tests

### Quick diagnostic for training accuracy issues:
```bash
./tests/gpu_tests/run_simple.sh diagnostic
```

### Single sample overfitting test:
```bash
./tests/gpu_tests/run_simple.sh quick
```

### All overfitting tests:
```bash
./tests/gpu_tests/run_simple.sh overfitting
```

### Single GPU tests:
```bash
./tests/gpu_tests/run_simple.sh single
```

### Multi-GPU tests (requires 2+ GPUs):
```bash
./tests/gpu_tests/run_simple.sh multi
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch sizes in test configurations
- Use smaller models (tests default to Qwen-0.5B)
- Clear GPU memory: `torch.cuda.empty_cache()`

### Flash-attn build failures
- Tests work without flash-attn (using eager attention)
- For production, consider using pre-built wheels or Docker images

### Multi-GPU test failures
- Ensure at least 2 GPUs are visible: `nvidia-smi`
- Check NCCL installation
- Set CUDA_VISIBLE_DEVICES explicitly: `export CUDA_VISIBLE_DEVICES=0,1`

## Expected Test Results

When running the diagnostic suite on properly functioning code:

1. **Single sample overfitting**: Should achieve < 0.1 loss after ~100 steps
2. **Loss comparison**: Patched and standard losses should match within 1e-3
3. **Batch overfitting**: Should see >90% loss reduction after 50 epochs
4. **Gradient accumulation**: Accumulated and batch gradients should be similar

If all tests pass but training accuracy is still poor, check:
- Learning rate and optimizer settings
- Data preprocessing and tokenization
- Model initialization
- Batch size and gradient accumulation settings
