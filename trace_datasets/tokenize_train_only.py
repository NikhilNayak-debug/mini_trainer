#!/usr/bin/env python3
"""
Tokenize only the training datasets for mini_trainer training.
Test datasets are kept in original format for TRACE evaluation.
"""

import os
import subprocess
import sys

def tokenize_training_dataset(input_file, output_file):
    """Tokenize a single training dataset using mini_trainer's process_data.py"""
    print(f"Tokenizing {os.path.basename(input_file)}...")

    cmd = [
        "python", "../scripts/process_data.py",
        "--input-file", input_file,
        "--output-file", output_file,
        "--model-name-or-path", "meta-llama/Llama-2-7b-chat-hf",
        "--max-sample-num-tokens", "128000"
    ]

    # Set environment variable to disable tokenizer parallelism warnings
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully tokenized {os.path.basename(input_file)}")
            return True
        else:
            print(f"❌ Error tokenizing {os.path.basename(input_file)}:")
            print(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception tokenizing {os.path.basename(input_file)}: {e}")
        return False

def main():
    """Tokenize only training datasets for TRACE benchmark"""
    print("Tokenizing TRACE training datasets for mini_trainer...")
    print("=" * 60)
    print("Note: Test datasets will remain in original format for TRACE evaluation")
    print("=" * 60)

    # Create tokenized directory
    os.makedirs('tokenized', exist_ok=True)

    # Get only training files from converted directory
    training_files = []
    for filename in os.listdir('converted'):
        if filename.endswith('_train.jsonl'):
            input_path = os.path.join('converted', filename)
            output_path = os.path.join('tokenized', filename.replace('.jsonl', '_tokenized.jsonl'))
            training_files.append((input_path, output_path))

    # Sort for consistent processing
    training_files.sort()

    print(f"Found {len(training_files)} training datasets to tokenize:")
    for input_path, _ in training_files:
        print(f"  - {os.path.basename(input_path)}")
    print()

    # Tokenize each training dataset
    success_count = 0
    for input_path, output_path in training_files:
        if tokenize_training_dataset(input_path, output_path):
            success_count += 1

    # Summary
    print("=" * 60)
    print("TOKENIZATION SUMMARY:")
    print("=" * 60)
    print(f"Successfully tokenized: {success_count}/{len(training_files)} training datasets")

    if success_count == len(training_files):
        print("🎉 All training datasets tokenized successfully!")
        print()
        print("Next steps:")
        print("1. Use tokenized files for mini_trainer training")
        print("2. Use original test files (converted/*_test.jsonl) for TRACE evaluation")
    else:
        print(f"⚠️  {len(training_files) - success_count} datasets failed to tokenize")

    print(f"\nTokenized training files saved in 'tokenized/' directory")
    print(f"Original test files remain in 'converted/' directory")

if __name__ == "__main__":
    main()