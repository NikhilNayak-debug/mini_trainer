#!/usr/bin/env python
"""Helper script to install flash-attn with proper dependency order.

This ensures torch and ninja are installed before attempting to build flash-attn.
"""
import subprocess
import sys
import importlib.util


def is_package_installed(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def install_flash_attn():
    """Install flash-attn with proper dependency order."""
    print("=" * 60)
    print("Setting up flash-attn for GPU tests")
    print("=" * 60)
    
    # Check if torch is installed
    if not is_package_installed("torch"):
        print("❌ PyTorch must be installed first")
        print("   Run: uv sync")
        return False
    
    # Check if ninja is installed
    if not is_package_installed("ninja"):
        print("⚠️  Ninja not installed, installing...")
        result = subprocess.run(
            ["uv", "pip", "install", "ninja"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"❌ Failed to install ninja: {result.stderr}")
            return False
        print("✓ Ninja installed")
    
    # Check if flash-attn is already installed
    if is_package_installed("flash_attn"):
        print("✓ flash-attn is already installed")
        return True
    
    # Install build dependencies for flash-attn
    print("\nInstalling build dependencies for flash-attn...")
    build_deps = ["packaging", "psutil", "einops"]
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
        print("   GPU tests will run with eager attention instead")
        return False
    
    print("✓ flash-attn installed successfully")
    return True


if __name__ == "__main__":
    success = install_flash_attn()
    sys.exit(0 if success else 1)
