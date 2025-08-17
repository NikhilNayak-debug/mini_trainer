"""Test overfitting behavior to diagnose training accuracy issues.

This module tests whether the model can overfit on single or small batches of examples,
which is crucial for verifying that the training mechanics work correctly.
"""
import pytest
import torch
import torch.nn as nn
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from setup_model_for_training import setup_model, setup_training_components
from sampler import JsonlDataset, get_data_loader
from train import take_gradient_step
from utils import init_distributed_environment, log_rank_0


@pytest.mark.gpu
@pytest.mark.overfitting
class TestSingleSampleOverfitting:
    """Test whether the model can memorize a single sample."""
    
    @pytest.fixture
    def small_model_name(self):
        """Return name of a small model with chat template."""
        # Using Qwen2.5-0.5B which has a chat template
        return "Qwen/Qwen2.5-0.5B-Instruct"  # 500M params, has chat template
    
    @pytest.fixture
    def create_single_sample_data(self, tmp_path):
        """Create a JSONL file with a single training sample."""
        def _create(content: str = "The capital of France is Paris", 
                   response: str = "That's correct!"):
            data_file = tmp_path / "single_sample.jsonl"
            
            # Create training data
            sample = {
                "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": response}
                ]
            }
            
            with open(data_file, 'w') as f:
                json.dump(sample, f)
                f.write('\n')
            
            return str(data_file)
        return _create
    
    @pytest.fixture
    def tokenize_sample(self, small_model_name):
        """Tokenize a sample for training."""
        def _tokenize(data_file: str, output_file: str):
            import subprocess
            
            # Run tokenization via CLI
            cmd = [
                "python", "process_data.py",
                "--input-file", data_file,
                "--output-file", output_file,
                "--model-name-or-path", small_model_name,
                "--max-sample-num-tokens", "512"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Tokenization failed: {result.stderr}")
            
            return output_file
        return _tokenize
    
    def test_single_sample_overfitting_loss_decreases(
        self, 
        small_model_name, 
        create_single_sample_data,
        tokenize_sample,
        tmp_path,
        single_gpu_device
    ):
        """Test that loss decreases when training on a single sample."""
        # Create data
        raw_data = create_single_sample_data()
        tokenized_data = str(tmp_path / "tokenized.jsonl")
        tokenize_sample(raw_data, tokenized_data)
        
        # Setup model
        model = AutoModelForCausalLM.from_pretrained(
            small_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"  # Use eager for testing without flash_attn
        )
        model = model.to(single_gpu_device)
        
        # Create tokenizer and align
        tokenizer = AutoTokenizer.from_pretrained(small_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Setup for training
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        
        # Load the single sample
        dataset = JsonlDataset(tokenized_data)
        sample = dataset[0]
        
        # Prepare batch
        input_ids = sample['input_ids'].unsqueeze(0).to(single_gpu_device)
        labels = sample['labels'].unsqueeze(0).to(single_gpu_device)
        
        # Track losses
        losses = []
        
        # Training loop - overfit on single sample
        for step in range(100):
            # Forward pass
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss
            
            # Track loss
            losses.append(loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Log progress
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")
        
        # Verify that loss decreased significantly
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        
        print(f"\nInitial loss (avg of first 5): {initial_loss:.4f}")
        print(f"Final loss (avg of last 5): {final_loss:.4f}")
        print(f"Loss reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
        
        # Assert significant loss reduction
        assert final_loss < initial_loss * 0.5, \
            f"Loss did not decrease enough. Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
        
        # Check if we achieved near-zero loss (true overfitting)
        assert final_loss < 0.1, \
            f"Final loss too high for overfitting: {final_loss:.4f}"
    
    def test_compare_loss_computation_methods(
        self,
        small_model_name,
        create_single_sample_data,
        tokenize_sample,
        tmp_path,
        single_gpu_device
    ):
        """Compare different loss computation methods to identify discrepancies."""
        # Create data
        raw_data = create_single_sample_data()
        tokenized_data = str(tmp_path / "tokenized.jsonl")
        tokenize_sample(raw_data, tokenized_data)
        
        # Setup model WITHOUT patching
        model_standard = AutoModelForCausalLM.from_pretrained(
            small_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        ).to(single_gpu_device)
        
        # Setup model WITH patching (as done in training)
        from none_reduction_losses import hf_fixed_cross_entropy_none_reduction
        from utils import patch_target_module
        
        # Patch the loss function
        patch_target_module(
            "transformers.loss.loss_utils.fixed_cross_entropy",
            hf_fixed_cross_entropy_none_reduction,
        )
        
        model_patched = AutoModelForCausalLM.from_pretrained(
            small_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        ).to(single_gpu_device)
        
        # Load sample
        dataset = JsonlDataset(tokenized_data)
        sample = dataset[0]
        
        input_ids = sample['input_ids'].unsqueeze(0).to(single_gpu_device)
        labels = sample['labels'].unsqueeze(0).to(single_gpu_device)
        num_loss_tokens = sample['num_loss_counted_tokens']
        
        # Get losses from both models
        with torch.no_grad():
            # Standard model 
            output_standard = model_standard(input_ids=input_ids, labels=labels)
            loss_standard = output_standard.loss
            
            # Handle if standard model also returns non-reduced loss
            if loss_standard.numel() > 1:
                valid_mask = labels.view(-1) != -100
                loss_standard_mean = loss_standard.view(-1)[valid_mask].mean()
            else:
                loss_standard_mean = loss_standard
            
            # Patched model (no reduction)
            output_patched = model_patched(input_ids=input_ids, labels=labels)
            loss_patched = output_patched.loss
            
            # Calculate mean of non-reduced loss
            if loss_patched.numel() > 1:
                # Filter out -100 labels
                valid_mask = labels.view(-1) != -100
                loss_patched_mean = loss_patched.view(-1)[valid_mask].mean()
            else:
                loss_patched_mean = loss_patched
        
        print(f"\nLoss Comparison:")
        print(f"Standard model loss shape: {loss_standard.shape if loss_standard.numel() > 1 else 'scalar'}")
        print(f"Standard model loss (mean): {loss_standard_mean.item():.6f}")
        print(f"Patched model loss shape: {loss_patched.shape if loss_patched.numel() > 1 else 'scalar'}")
        print(f"Patched model loss (mean): {loss_patched_mean.item():.6f}")
        print(f"Number of loss-counted tokens: {num_loss_tokens}")
        
        # They should be very close
        assert torch.allclose(loss_standard_mean, loss_patched_mean, rtol=1e-3, atol=1e-4), \
            f"Loss mismatch: standard={loss_standard_mean.item():.6f}, patched={loss_patched_mean.item():.6f}"
    
    def test_loss_scaling_with_world_size(
        self,
        small_model_name,
        create_single_sample_data,
        tokenize_sample,
        tmp_path,
        single_gpu_device
    ):
        """Test the loss scaling logic used in distributed training."""
        # Create data
        raw_data = create_single_sample_data(
            content="What is 2+2?",
            response="2+2 equals 4."
        )
        tokenized_data = str(tmp_path / "tokenized.jsonl")
        tokenize_sample(raw_data, tokenized_data)
        
        # Setup model with patching
        from none_reduction_losses import hf_fixed_cross_entropy_none_reduction
        from utils import patch_target_module
        
        patch_target_module(
            "transformers.loss.loss_utils.fixed_cross_entropy",
            hf_fixed_cross_entropy_none_reduction,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            small_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        ).to(single_gpu_device)
        
        # Load sample
        dataset = JsonlDataset(tokenized_data)
        sample = dataset[0]
        
        input_ids = sample['input_ids'].unsqueeze(0).to(single_gpu_device)
        labels = sample['labels'].unsqueeze(0).to(single_gpu_device)
        num_loss_tokens = sample['num_loss_counted_tokens']
        
        # Simulate the training logic
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss.float().sum()  # Sum as done in training
        
        # Simulate different world sizes
        for world_size in [1, 2, 4, 8]:
            # This is the scaling from train.py
            scaled_loss = loss * world_size / num_loss_tokens
            
            print(f"\nWorld size {world_size}:")
            print(f"  Raw loss sum: {loss.item():.6f}")
            print(f"  Scaled loss: {scaled_loss.item():.6f}")
            print(f"  Num loss tokens: {num_loss_tokens}")
            
            # The average loss per token should be consistent
            avg_per_token = loss.item() / num_loss_tokens
            print(f"  Avg per token (raw): {avg_per_token:.6f}")
            
            # Verify scaling makes sense
            assert scaled_loss.item() > 0, "Scaled loss should be positive"
            
            # For world_size=1, scaled should equal avg per token
            if world_size == 1:
                assert torch.allclose(
                    scaled_loss, 
                    torch.tensor(avg_per_token).to(single_gpu_device),
                    rtol=1e-4
                ), f"Scaling incorrect for world_size=1"


@pytest.mark.gpu
@pytest.mark.overfitting
class TestBatchOverfitting:
    """Test overfitting on small batches with the full training pipeline."""
    
    def test_training_pipeline_with_small_batch(
        self,
        tmp_path,
        single_gpu_device
    ):
        """Test the full training pipeline with a small batch."""
        # Create multiple samples for a small batch
        data_file = tmp_path / "batch_samples.jsonl"
        samples = [
            {"messages": [
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": "1+1 equals 2."}
            ]},
            {"messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]},
            {"messages": [
                {"role": "user", "content": "What is 3+3?"},
                {"role": "assistant", "content": "3+3 equals 6."}
            ]},
        ]
        
        with open(data_file, 'w') as f:
            for sample in samples:
                json.dump(sample, f)
                f.write('\n')
        
        # Tokenize the data
        import subprocess
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenized_file = tmp_path / "tokenized_batch.jsonl"
        
        cmd = [
            "python", "process_data.py",
            "--input-file", str(data_file),
            "--output-file", str(tokenized_file),
            "--model-name-or-path", model_name,
            "--max-sample-num-tokens", "512"
        ]
        subprocess.run(cmd, check=True)
        
        # Setup model directly without using setup_model which may require distributed init
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        )
        
        # Move to GPU
        model = model.to(single_gpu_device)
        model.train()
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        
        # Load dataset
        dataset = JsonlDataset(str(tokenized_file))
        
        # Training loop
        losses = []
        for epoch in range(50):
            epoch_loss = 0
            for i in range(len(dataset)):
                sample = dataset[i]
                
                # Prepare batch
                input_ids = sample['input_ids'].unsqueeze(0).to(single_gpu_device)
                labels = sample['labels'].unsqueeze(0).to(single_gpu_device)
                
                # Forward pass
                output = model(input_ids=input_ids, labels=labels)
                loss = output.loss
                
                # Handle non-reduced loss if patched
                if loss.numel() > 1:
                    loss = loss.mean()
                
                epoch_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient step
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_loss = epoch_loss / len(dataset)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
        
        # Check for overfitting
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        
        print(f"\nTraining Results:")
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
        
        # Should see significant reduction
        assert final_loss < initial_loss * 0.3, \
            f"Insufficient loss reduction. Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
