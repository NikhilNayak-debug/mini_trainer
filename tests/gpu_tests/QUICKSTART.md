# GPU Tests Quick Start

## Fastest Way to Test

```bash
# From mini_trainer root directory
uv run tox -e gpu-tests -- -k test_single_sample_overfitting
```

This will:
1. Create an isolated test environment
2. Install all dependencies
3. Run a quick overfitting test to verify training works

## Full Diagnostic Suite

To diagnose training accuracy issues:

```bash
# Run comprehensive overfitting tests
uv run tox -e gpu-overfitting
```

## What These Tests Check

- **Overfitting capability**: Can the model memorize single samples?
- **Loss computation**: Is the loss calculation correct?
- **Gradient flow**: Are gradients computed and accumulated properly?
- **Training pipeline**: Does the full training loop work?

## Expected Results

✅ **PASS**: Model achieves near-zero loss on single samples
✅ **PASS**: Loss decreases significantly (>90%) when overfitting
✅ **PASS**: Patched and standard loss computations match

If all tests pass but accuracy is still poor, the issue is likely in:
- Hyperparameters (learning rate, batch size)
- Data preprocessing
- Model initialization
