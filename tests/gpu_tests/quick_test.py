#!/usr/bin/env python
"""Quick test to verify GPU test setup is working."""
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    print("=" * 60)
    print("GPU Test Environment Check")
    print("=" * 60)
    
    # Check CUDA
    print("\n1. CUDA Availability:")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA is available")
        print(f"   ✓ Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"     GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        print("   ✗ CUDA is not available")
        return 1
    
    # Check flash-attn
    print("\n2. Flash Attention:")
    try:
        import flash_attn
        print("   ✓ Flash attention is installed")
    except ImportError:
        print("   ⚠ Flash attention not installed")
        print("     Install with: uv pip install flash-attn>=2.8.2")
    
    # Check transformers
    print("\n3. Transformers:")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("   ✓ Transformers is installed")
        
        # Try loading a small model
        print("\n4. Testing model loading:")
        print("   Loading GPT2 (small model for testing)...")
        model = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2",
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        print("   ✓ Model loaded successfully")
        
        # Quick forward pass
        print("\n5. Testing forward pass:")
        model = model.cuda()
        inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"   ✓ Forward pass successful (logits shape: {outputs.logits.shape})")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return 1
    
    # Check mini_trainer modules
    print("\n6. Mini Trainer modules:")
    try:
        import setup_model_for_training
        import sampler
        import train
        import utils
        print("   ✓ All mini_trainer modules importable")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("✓ Environment check complete! Ready to run GPU tests.")
    print("=" * 60)
    print("\nTo run the diagnostic suite:")
    print("  cd tests/gpu_tests")
    print("  python run_gpu_tests.py --diagnostic")
    print("\nOr using tox:")
    print("  uv run tox -e gpu-overfitting")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
