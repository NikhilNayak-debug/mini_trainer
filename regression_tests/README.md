# Regression Tests and Benchmarks

This directory contains regression tests, benchmarks, and analysis tools that provide performance insights and detailed analysis rather than binary pass/fail signals.

## Contents

### `benchmark_batching.py`
Performance benchmarks and efficiency analysis for batch packing algorithms:
- Compares greedy vs LPT (Longest Processing Time) algorithms
- Measures load distribution and balance across distributed ranks
- Analyzes edge cases and realistic data distributions
- Provides execution time benchmarks

**Usage:**
```bash
python regression_tests/benchmark_batching.py
```

### `test_osft_fidelity_script.py`
Script for testing OSFT (Orthogonal Subspace Fine-Tuning) decomposition and reconstruction fidelity:
- Verifies that OSFT-decomposed models can accurately reconstruct original parameters
- Tests distributed SVD computation for OSFT initialization
- Measures numerical accuracy and reconstruction errors

**Usage:**
```bash
# Single GPU
python regression_tests/test_osft_fidelity_script.py

# Multiple GPUs (distributed testing)
torchrun --nnodes=1 --nproc-per-node=4 regression_tests/test_osft_fidelity_script.py
```

## Purpose

These tests serve different purposes from unit tests:

1. **Performance Analysis**: Measure and compare algorithm efficiency
2. **Regression Detection**: Identify performance degradations over time
3. **Optimization Insights**: Provide detailed metrics for optimization work
4. **Realistic Scenarios**: Test with realistic data distributions and edge cases

## Running All Benchmarks

To run all regression tests and benchmarks:

```bash
# Run batching benchmarks
python regression_tests/benchmark_batching.py

# Run OSFT fidelity tests (requires model downloads)
python regression_tests/test_osft_fidelity_script.py --model-name-or-path "Qwen/Qwen2.5-1.5B-Instruct"
```

## Interpreting Results

### Batching Benchmarks
- **Load Balance Ratio**: Closer to 1.0 is better (perfect balance)
- **Overall Efficiency**: Higher is better (less wasted capacity)
- **Execution Time**: Lower is better (faster algorithm)

### OSFT Fidelity
- **Success Rate**: Percentage of parameters matching within tolerance
- **Max Difference**: Maximum deviation from original parameters
- **Average Difference**: Mean deviation across all parameters

## Notes

- These tests may take longer to run than unit tests
- Some tests require GPU availability
- Results may vary based on hardware and system load
- These are NOT part of the regular CI/CD test suite
