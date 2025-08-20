#!/usr/bin/env python
"""Helper script to install flash-attn with proper dependency order.

This ensures torch and ninja are installed before attempting to build flash-attn.
Comes with an optional `--strict` flag that lets GPU-oriented tests enforce its presence,
while allowing those tests to be skippable on non-GPU tests that need to run the broader suite (e.g. CI).
"""
import argparse
import subprocess
import sys
import importlib.util


def is_package_installed(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def install_flash_attn(strict=False):
    """Install flash-attn with proper dependency order.
    
    Args:
        strict: If True, exit with error code when installation fails.
                If False, exit with success even on failure (default).
    """
    print("=" * 60)
    print("Setting up flash-attn for GPU tests")
    print(f"Strict mode: {strict}")
    print("=" * 60)

    # Check if flash-attn is already installed and uninstall it
    if is_package_installed("flash_attn"):
        print("⚠️  flash-attn is already installed, uninstalling first...")
        result = subprocess.run(
            ["uv", "pip", "uninstall", "flash-attn"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"❌ Failed to uninstall flash-attn: {result.stderr}")
            return False if strict else True
        print("✓ flash-attn uninstalled")

    # Check if torch is installed
    if not is_package_installed("torch"):
        print("❌ PyTorch must be installed first")
        print("   In tox: Add 'torch>=2.6' to deps section")
        print("   In uv: Run 'uv sync'")
        return False if strict else True
    
    # Install build dependencies for flash-attn
    print("\nInstalling build dependencies for flash-attn...")
    build_deps = ["packaging", "psutil", "einops", "ninja"]
    for dep in build_deps:
        if not is_package_installed(dep):
            print(f"  Installing {dep}...")
            result = subprocess.run(
                ["uv", "pip", "install", dep],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"  ⚠️  Failed to install {dep}")
    
    # Install flash-attn with no-build-isolation since torch is already installed
    print("\nInstalling flash-attn (this may take a few minutes)...")
    print("Note: Using --no-build-isolation since torch is already installed")
    result = subprocess.run(
        ["uv", "pip", "install", "--no-build-isolation", "flash-attn>=2.8.2"],
        capture_output=False,  # Show output for long build process
        text=True
    )
    
    if result.returncode != 0:
        print("⚠️  Failed to install flash-attn")
        if strict:
            print("   ❌ Exiting with error (strict mode enabled)")
        else:
            print("   ✓ Continuing anyway (strict mode disabled)")
            print("   GPU tests will run with eager attention instead")
        return False if strict else True
    
    print("✓ flash-attn installed successfully")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install flash-attn for GPU tests")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if flash-attn installation fails (default: False)"
    )
    args = parser.parse_args()
    
    success = install_flash_attn(strict=args.strict)
    sys.exit(0 if success else 1)
