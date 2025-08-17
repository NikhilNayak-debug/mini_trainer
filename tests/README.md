# Mini Trainer Test Suite

This directory contains comprehensive unit tests for all critical components of the distributed training pipeline.

## Test Organization

The test suite is organized into the following modules:

### 1. `test_batch_lengths_to_minibatches.py`
Tests for batch packing algorithms including:
- Greedy batch assignment algorithm
- LPT (Longest Processing Time) algorithm  
- Performance comparisons between algorithms
- Edge cases and load balancing metrics

### 2. `test_data_loader.py`
Tests for data loading and sampling components:
- `JsonlDataset`: Tests loading and processing JSONL data files
- `InfiniteSampler`: Tests infinite shuffled sampling for training
- `MaxTokensPerRankCollator`: Tests distributed batching and token limits
- `get_data_loader`: Tests data loader creation and configuration

### 3. `test_model_initialization.py`
Tests for model setup and initialization:
- Model and tokenizer alignment
- FSDP2 wrapping and sharding
- Liger kernels integration
- Optimizer and scheduler setup
- Orthogonal subspace learning (SVD) support

### 4. `test_training_components.py`
Tests for training utilities and components:
- Gradient stepping and clipping
- Batch metrics accumulation and reduction
- Model checkpointing and saving
- Distributed environment setup
- Logging utilities
- Module patching

### 5. `test_training_loop.py`
Tests for the main training loop:
- Training loop execution
- Minibatch processing
- Checkpoint saving based on samples
- Memory management
- Multi-rank training coordination
- CLI parameter handling

### 6. GPU Tests (`gpu_tests/`)
GPU-enabled tests that require CUDA and flash-attention:

#### `test_mixed_precision.py`
Tests for mixed precision training dtype validation:
- FSDP2 mixed precision policy configuration
- Model parameter dtype verification (FP32 vs BF16)
- Gradient dtype validation after training iterations
- Optimizer state dtype checking (exp_avg, exp_avg_sq)
- Tiny Llama model creation (<1M parameters) for testing
- Catches bugs where reduce_dtype is incorrectly set to BF16

#### `test_single_gpu_training.py`
Tests for single GPU training mechanics:
- Complete training loop validation
- Gradient accumulation behavior
- Loss computation with different kernels (Liger vs HuggingFace)
- Model convergence verification

#### `test_multi_gpu_training.py`
Tests for multi-GPU distributed training:
- Distributed FSDP2 training coordination
- Multi-rank gradient synchronization
- Distributed data loading and sampling

#### `test_overfitting.py`
Tests for training accuracy and overfitting behavior:
- Small dataset overfitting validation
- Training convergence on minimal examples
- Loss reduction verification

## Running Tests

### Using Tox (Recommended)
```bash
# Run all tests
tox -e test

# Run specific test module
tox -e test -- tests/test_data_loader.py

# Run with verbose output
tox -e test-verbose

# Run until first failure
tox -e test-quick

# Run with coverage report
tox -e test-coverage
```

### Using pytest directly with uv
```bash
# Run all tests (excluding GPU tests)
uv run pytest tests/ -m "not gpu"

# Run all tests including GPU tests (requires CUDA)
uv run pytest tests/

# Run specific test class
uv run pytest tests/test_data_loader.py::TestJsonlDataset

# Run specific test method
uv run pytest tests/test_data_loader.py::TestJsonlDataset::test_dataset_initialization

# Run only GPU tests
uv run pytest tests/gpu_tests/ -m gpu

# Run specific GPU test
uv run pytest tests/gpu_tests/test_mixed_precision.py::TestMixedPrecisionDtypes::test_fsdp2_mixed_precision_dtypes

# Run with verbose output
uv run pytest -v tests/
```

### Using the test runner script
```bash
# Run all tests including performance comparisons
python tests/run_tests.py
```

## Test Coverage

The test suite covers:
- **Data Pipeline**: Dataset loading, sampling, batching, collation
- **Model Setup**: Initialization, FSDP wrapping, optimizer configuration
- **Training Loop**: Forward/backward passes, gradient accumulation, checkpointing
- **Distributed Training**: Multi-rank coordination, metrics reduction, synchronization
- **Mixed Precision**: FSDP2 dtype policies, parameter/gradient/optimizer state validation
- **GPU Training**: Single and multi-GPU training mechanics, convergence validation
- **Utilities**: Logging, memory management, configuration handling

## Adding New Tests

When adding new functionality, ensure you:
1. Add corresponding unit tests in the appropriate test module
2. Test both normal operation and edge cases
3. Include tests for distributed scenarios where applicable
4. Mock external dependencies (models, tokenizers, etc.)
5. Document what the test verifies

## Test Requirements

### Basic Tests
Tests require the following packages (installed automatically by tox):
- pytest >= 7.0
- pytest-cov (for coverage reports)
- pytest-mock (for advanced mocking)
- All project dependencies

### GPU Tests (`gpu_tests/`)
GPU tests have additional requirements:
- CUDA-capable GPU with sufficient memory
- NVIDIA drivers and CUDA toolkit
- flash-attention >= 2.0 (`pip install flash-attn`)
- PyTorch with CUDA support

GPU tests are marked with the `@pytest.mark.gpu` decorator and can be:
- Skipped with `-m "not gpu"` if CUDA is unavailable
- Run exclusively with `-m gpu` on GPU-enabled systems

## Performance Testing

The `test_batch_lengths_to_minibatches.py` module includes performance benchmarks comparing different batching algorithms. These tests measure:
- Load distribution across ranks
- Number of minibatches generated
- Algorithm execution speed
- Efficiency metrics

Run performance tests separately:
```bash
python tests/test_batch_lengths_to_minibatches.py
```
