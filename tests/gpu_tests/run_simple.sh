#!/bin/bash
# Simple runner for GPU tests that uses the existing uv environment
# This avoids the complexity of building flash-attn in tox environments

echo "============================================================"
echo "GPU Tests Runner (Simple Mode)"
echo "============================================================"
echo ""
echo "This runner uses your existing uv environment with pre-installed"
echo "dependencies. To install flash-attn if not already installed:"
echo "  uv pip install flash-attn>=2.8.2"
echo ""
echo "============================================================"

# Check if we're in the mini_trainer directory
if [ ! -f "train.py" ]; then
    echo "Error: Please run this script from the mini_trainer root directory"
    exit 1
fi

# Parse arguments
TEST_TYPE="${1:-diagnostic}"
EXTRA_ARGS="${@:2}"

case "$TEST_TYPE" in
    diagnostic)
        echo "Running diagnostic suite for training accuracy issues..."
        uv run python tests/gpu_tests/run_gpu_tests.py --diagnostic $EXTRA_ARGS
        ;;
    overfitting)
        echo "Running overfitting tests..."
        uv run pytest tests/gpu_tests/test_overfitting.py -v -s $EXTRA_ARGS
        ;;
    single)
        echo "Running single GPU tests..."
        uv run pytest tests/gpu_tests/test_single_gpu_training.py -v -s $EXTRA_ARGS
        ;;
    multi)
        echo "Running multi-GPU tests (requires 2+ GPUs)..."
        CUDA_VISIBLE_DEVICES=0,1 uv run pytest tests/gpu_tests/test_multi_gpu_training.py -v -s -m multigpu $EXTRA_ARGS
        ;;
    all)
        echo "Running all GPU tests..."
        uv run pytest tests/gpu_tests -v -m gpu $EXTRA_ARGS
        ;;
    quick)
        echo "Running quick overfitting test..."
        uv run pytest tests/gpu_tests/test_overfitting.py::TestSingleSampleOverfitting::test_single_sample_overfitting_loss_decreases -xvs $EXTRA_ARGS
        ;;
    *)
        echo "Usage: $0 [diagnostic|overfitting|single|multi|all|quick] [extra pytest args]"
        echo ""
        echo "Examples:"
        echo "  $0 diagnostic              # Run diagnostic suite"
        echo "  $0 overfitting            # Run all overfitting tests"
        echo "  $0 single                 # Run single GPU tests"
        echo "  $0 multi                  # Run multi-GPU tests"
        echo "  $0 all                    # Run all tests"
        echo "  $0 quick                  # Run a quick test"
        echo "  $0 overfitting -k test_name  # Run specific test"
        exit 1
        ;;
esac
