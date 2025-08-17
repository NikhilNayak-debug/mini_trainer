# GPU Tests for Mini Trainer

This directory contains GPU-based tests designed to diagnose training accuracy issues and verify the training pipeline works correctly with CUDA and flash-attention.

## Purpose

These tests were created to help debug a training accuracy issue where the model cannot achieve the same accuracy as other codebases with identical settings. The tests focus on:

1. **Overfitting behavior** - Verifying the model can memorize single samples
2. **Loss computation** - Comparing different loss calculation methods
3. **Distributed training** - Testing multi-GPU synchronization and scaling
4. **Training mechanics** - Validating gradient accumulation and optimization

## Requirements

- CUDA-capable GPU(s)
- PyTorch with CUDA support
- Flash-attention (optional but recommended)
- At least 2 GPUs for multi-GPU tests

## Installation

The tests work with the standard mini_trainer installation:

```bash
# Using uv (recommended)
uv sync

# Optional: Install flash-attn for better performance
# (Tests work without it, using eager attention)
uv pip install flash-attn>=2.8.2
```

**Note**: Flash-attn is optional. Tests automatically fall back to eager attention if not installed.

## Running Tests

### Using Tox (Recommended)

All GPU tests can be run through tox with isolated environments:

```bash
# Run ALL GPU tests
uv run tox -e gpu-tests

# Run overfitting tests for accuracy diagnosis
uv run tox -e gpu-overfitting

# Run single GPU tests
uv run tox -e gpu-single

# Run multi-GPU tests (requires 2+ GPUs)
uv run tox -e gpu-multi

# Run tests marked with @pytest.mark.gpu
uv run tox -e gpu
```

### Running Specific Tests with Tox

```bash
# Run a specific test by name
uv run tox -e gpu-tests -- -k test_single_sample_overfitting

# Run with verbose output
uv run tox -e gpu-overfitting -- -v

# Run until first failure
uv run tox -e gpu-tests -- -x

# Run specific test file
uv run tox -e gpu-tests -- tests/gpu_tests/test_overfitting.py
```

### Alternative: Direct Execution

If you prefer to use your existing uv environment:

```bash
# Quick overfitting test
uv run pytest tests/gpu_tests/test_overfitting.py::TestSingleSampleOverfitting::test_single_sample_overfitting_loss_decreases -xvs

# All overfitting tests
uv run pytest tests/gpu_tests/test_overfitting.py -v

# All GPU tests
uv run pytest tests/gpu_tests -v
```

### Running Specific Tests

```bash
# Run a specific test by name
python run_gpu_tests.py --overfitting -k test_single_sample_overfitting

# Or with pytest directly
uv run pytest tests/gpu_tests/test_overfitting.py::TestSingleSampleOverfitting::test_single_sample_overfitting_loss_decreases -v -s
```

## Test Structure

### test_overfitting.py
- `TestSingleSampleOverfitting`: Tests if the model can memorize single examples
  - `test_single_sample_overfitting_loss_decreases`: Verifies loss reduction on single sample
  - `test_compare_loss_computation_methods`: Compares patched vs standard loss
  - `test_loss_scaling_with_world_size`: Tests distributed loss scaling logic
- `TestBatchOverfitting`: Tests overfitting on small batches
  - `test_training_pipeline_with_small_batch`: Full pipeline test with small batch

### test_single_gpu_training.py
- `TestSingleGPUTraining`: Single GPU training mechanics
  - `test_single_gpu_training_loop`: Complete training loop test
  - `test_gradient_accumulation`: Verifies gradient accumulation
  - `test_loss_with_liger_kernels`: Tests Liger kernel integration

### test_multi_gpu_training.py
- `TestMultiGPUTraining`: Distributed training tests
  - `test_two_gpu_training`: FSDP2 training on 2 GPUs
  - `test_gradient_synchronization`: Verifies gradient sync across ranks
  - `test_loss_consistency_across_ranks`: Checks loss computation consistency

## Quick Start for Accuracy Debugging

```bash
# Run the diagnostic suite
uv run tox -e gpu-tests -- -k diagnostic

# Or run key overfitting tests
uv run tox -e gpu-overfitting
```

## Debugging Training Accuracy Issues

If the diagnostic suite reveals issues, check:

1. **Loss Computation**
   - Run `test_compare_loss_computation_methods` to verify loss patching
   - Check if reduced vs non-reduced losses match expectations

2. **Overfitting Capability**
   - Run `test_single_sample_overfitting_loss_decreases`
   - Model should achieve near-zero loss on single samples

3. **Distributed Scaling**
   - Run `test_loss_scaling_with_world_size`
   - Verify the `world_size / batch_num_loss_counted_tokens` scaling

4. **Gradient Accumulation**
   - Run `test_gradient_accumulation`
   - Compare accumulated gradients with batch processing

## Expected Behavior

### Successful Tests
- Single sample overfitting should achieve < 0.1 final loss
- Loss should decrease by > 50% when overfitting
- Patched and standard losses should match within 1e-3 tolerance
- Distributed gradients should be synchronized across ranks

### Common Issues
1. **Loss not decreasing**: Check learning rate, optimizer settings
2. **Loss mismatch**: Verify loss patching and reduction settings
3. **Distributed failures**: Check NCCL setup and GPU visibility
4. **Flash attention errors**: Fall back to eager attention for testing

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: Control which GPUs to use
- `NCCL_DEBUG`: Set to `INFO` for distributed debugging
- `TORCH_DISTRIBUTED_DEBUG`: Set to `DETAIL` for more info

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in test configurations
- Use smaller models (GPT2 instead of larger models)
- Clear cache between tests: `torch.cuda.empty_cache()`

### Multi-GPU Tests Fail
- Ensure at least 2 GPUs are available: `nvidia-smi`
- Check NCCL is properly installed
- Verify no other processes are using the GPUs

### Flash Attention Issues
- Tests will fall back to eager attention if flash_attn not available
- Install with: `uv pip install flash-attn>=2.8.2`
- Requires CUDA 11.6+ and compatible GPU architecture

## Next Steps

After running the diagnostic suite:

1. If all tests pass, the training mechanics are working correctly
   - Check hyperparameters (learning rate, batch size)
   - Verify data preprocessing and tokenization
   - Review model initialization

2. If tests fail, focus on the specific failures:
   - Loss computation issues → Check loss patching
   - Overfitting failures → Verify backward pass and optimization
   - Distributed issues → Check FSDP2 configuration

3. Compare with reference implementation:
   - Run the same tests on the working codebase
   - Compare loss values and convergence rates
   - Check for differences in data preprocessing
