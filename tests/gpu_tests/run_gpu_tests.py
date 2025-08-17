#!/usr/bin/env python
"""GPU test runner for diagnosing training accuracy issues.

This script provides a convenient way to run GPU tests with proper setup
and environment configuration.
"""
import argparse
import sys
import os
import subprocess


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA is not available. GPU tests require CUDA.")
            return False
        
        num_gpus = torch.cuda.device_count()
        print(f"✓ Found {num_gpus} GPU(s)")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_flash_attn():
    """Check if flash attention is available."""
    try:
        import flash_attn
        print("✓ Flash attention is installed")
        return True
    except ImportError:
        print("⚠️  Flash attention not installed (some tests will be skipped)")
        return False


def run_overfitting_tests(args):
    """Run overfitting tests to diagnose accuracy issues."""
    print("\n" + "="*60)
    print("Running Overfitting Tests")
    print("="*60)
    
    cmd = ["uv", "run", "pytest", "tests/gpu_tests/test_overfitting.py", "-v", "-s"]
    
    if args.specific_test:
        cmd.extend(["-k", args.specific_test])
    
    if args.verbose:
        cmd.append("-vv")
    
    return subprocess.run(cmd).returncode


def run_single_gpu_tests(args):
    """Run single GPU training tests."""
    print("\n" + "="*60)
    print("Running Single GPU Tests")
    print("="*60)
    
    cmd = ["uv", "run", "pytest", "tests/gpu_tests/test_single_gpu_training.py", "-v", "-s"]
    
    if args.specific_test:
        cmd.extend(["-k", args.specific_test])
    
    if args.verbose:
        cmd.append("-vv")
    
    # Set CUDA device if specified
    env = os.environ.copy()
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Using GPU {args.gpu_id}")
    
    return subprocess.run(cmd, env=env).returncode


def run_multi_gpu_tests(args):
    """Run multi-GPU distributed tests."""
    import torch
    if torch.cuda.device_count() < 2:
        print("❌ Multi-GPU tests require at least 2 GPUs")
        return 1
    
    print("\n" + "="*60)
    print("Running Multi-GPU Tests")
    print("="*60)
    
    cmd = ["uv", "run", "pytest", "tests/gpu_tests/test_multi_gpu_training.py", "-v", "-s", "-m", "multigpu"]
    
    if args.specific_test:
        cmd.extend(["-k", args.specific_test])
    
    if args.verbose:
        cmd.append("-vv")
    
    # Set GPUs to use
    env = os.environ.copy()
    if args.num_gpus:
        gpu_ids = ",".join(str(i) for i in range(args.num_gpus))
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"Using GPUs: {gpu_ids}")
    
    return subprocess.run(cmd, env=env).returncode


def run_all_tests(args):
    """Run all GPU tests."""
    results = []
    
    # Run overfitting tests
    print("\n🔍 Testing overfitting behavior...")
    results.append(("Overfitting", run_overfitting_tests(args)))
    
    # Run single GPU tests
    print("\n💻 Testing single GPU training...")
    results.append(("Single GPU", run_single_gpu_tests(args)))
    
    # Run multi-GPU tests if available
    import torch
    if torch.cuda.device_count() >= 2:
        print("\n🖥️  Testing multi-GPU training...")
        results.append(("Multi-GPU", run_multi_gpu_tests(args)))
    else:
        print("\n⚠️  Skipping multi-GPU tests (need 2+ GPUs)")
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, returncode in results:
        status = "✓ PASSED" if returncode == 0 else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
        if returncode != 0:
            all_passed = False
    
    return 0 if all_passed else 1


def run_diagnostic_suite(args):
    """Run a comprehensive diagnostic suite for training accuracy issues."""
    print("\n" + "="*60)
    print("Running Diagnostic Suite for Training Accuracy Issues")
    print("="*60)
    
    tests = [
        "test_single_sample_overfitting_loss_decreases",
        "test_compare_loss_computation_methods",
        "test_loss_scaling_with_world_size",
        "test_training_pipeline_with_small_batch",
        "test_gradient_accumulation",
    ]
    
    if check_flash_attn():
        tests.append("test_loss_with_liger_kernels")
    
    print(f"\nRunning {len(tests)} diagnostic tests:")
    for test in tests:
        print(f"  • {test}")
    
    failed_tests = []
    for test in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test}")
        print('='*60)
        
        cmd = ["uv", "run", "pytest", 
               f"tests/gpu_tests/test_overfitting.py::{test}",
               "-v", "-s"]
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed_tests.append(test)
    
    # Summary
    print("\n" + "="*60)
    print("Diagnostic Suite Results")
    print("="*60)
    
    if not failed_tests:
        print("✓ All diagnostic tests passed!")
        print("\nThe training mechanics appear to be working correctly.")
        print("Consider checking:")
        print("  • Learning rate and optimizer settings")
        print("  • Data preprocessing and tokenization")
        print("  • Model initialization")
        print("  • Batch size and gradient accumulation settings")
    else:
        print(f"✗ {len(failed_tests)} test(s) failed:")
        for test in failed_tests:
            print(f"  • {test}")
        print("\nThese failures may indicate issues with:")
        print("  • Loss computation or scaling")
        print("  • Gradient accumulation")
        print("  • Distributed training setup")
    
    return 0 if not failed_tests else 1


def main():
    parser = argparse.ArgumentParser(
        description="Run GPU tests for mini_trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all GPU tests
  python run_gpu_tests.py --all
  
  # Run diagnostic suite for accuracy issues
  python run_gpu_tests.py --diagnostic
  
  # Run only overfitting tests
  python run_gpu_tests.py --overfitting
  
  # Run single GPU tests on GPU 1
  python run_gpu_tests.py --single --gpu-id 1
  
  # Run specific test
  python run_gpu_tests.py --overfitting -k test_single_sample_overfitting
"""
    )
    
    parser.add_argument("--all", action="store_true",
                       help="Run all GPU tests")
    parser.add_argument("--diagnostic", action="store_true",
                       help="Run diagnostic suite for training accuracy issues")
    parser.add_argument("--overfitting", action="store_true",
                       help="Run overfitting tests")
    parser.add_argument("--single", action="store_true",
                       help="Run single GPU tests")
    parser.add_argument("--multi", action="store_true",
                       help="Run multi-GPU tests")
    parser.add_argument("--gpu-id", type=int,
                       help="GPU ID to use for single GPU tests")
    parser.add_argument("--num-gpus", type=int,
                       help="Number of GPUs to use for multi-GPU tests")
    parser.add_argument("-k", "--specific-test", type=str,
                       help="Run specific test by name pattern")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Check prerequisites
    print("Checking environment...")
    if not check_cuda():
        return 1
    
    check_flash_attn()
    
    # Default to diagnostic if no specific test selected
    if not any([args.all, args.diagnostic, args.overfitting, args.single, args.multi]):
        args.diagnostic = True
    
    # Run requested tests
    if args.diagnostic:
        return run_diagnostic_suite(args)
    elif args.all:
        return run_all_tests(args)
    elif args.overfitting:
        return run_overfitting_tests(args)
    elif args.single:
        return run_single_gpu_tests(args)
    elif args.multi:
        return run_multi_gpu_tests(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
