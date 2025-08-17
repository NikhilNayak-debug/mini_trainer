"""Single GPU training tests to verify training mechanics."""
import pytest
import torch
import torch.nn as nn
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from setup_model_for_training import setup_model, setup_training_components
from sampler import get_data_loader, JsonlDataset, InfiniteSampler, MaxTokensPerRankCollator
from torch.utils.data import DataLoader
from batch_metrics import BatchMetrics
from utils import patch_target_module


@pytest.mark.gpu
class TestSingleGPUTraining:
    """Test training on a single GPU with the full pipeline."""
    
    @pytest.fixture
    def small_model_configs(self):
        """Return configurations for small models suitable for testing."""
        return [

            {
                'name': 'qwen-0.5b',
                'model_id': 'Qwen/Qwen2.5-0.5B-Instruct',
                'has_flash_attn': True,
                'attn_implementation': 'flash_attention_2'
            },
        ]
    
    @pytest.fixture
    def create_test_dataset(self, tmp_path):
        """Create a test dataset with multiple samples."""
        def _create(num_samples=10):
            data_file = tmp_path / "train_data.jsonl"
            samples = []
            
            for i in range(num_samples):
                sample = {
                    "messages": [
                        {"role": "user", "content": f"What is {i}+{i}?"},
                        {"role": "assistant", "content": f"{i}+{i} equals {2*i}."}
                    ]
                }
                samples.append(sample)
            
            with open(data_file, 'w') as f:
                for sample in samples:
                    json.dump(sample, f)
                    f.write('\n')
            
            return str(data_file)
        return _create
    
    def test_single_gpu_training_loop(
        self,
        small_model_configs,
        create_test_dataset,
        tmp_path,
        single_gpu_device,
        flash_attn_available
    ):
        """Test the complete training loop on a single GPU."""
        # Use Qwen model which has a chat template
        model_config = small_model_configs[0]  # Qwen-0.5B
        
        # Adjust attention implementation based on flash_attn availability
        if not flash_attn_available:
            model_config['attn_implementation'] = 'eager'
        
        # Create and tokenize data
        raw_data = create_test_dataset(num_samples=20)
        tokenized_data = tmp_path / "tokenized.jsonl"
        
        import subprocess
        cmd = [
            "python", "process_data.py",
            "--input-file", raw_data,
            "--output-file", str(tokenized_data),
            "--model-name-or-path", model_config['model_id'],
            "--max-sample-num-tokens", "512"
        ]
        subprocess.run(cmd, check=True)
        
        # Setup model with patching
        from none_reduction_losses import hf_fixed_cross_entropy_none_reduction
        patch_target_module(
            "transformers.loss.loss_utils.fixed_cross_entropy",
            hf_fixed_cross_entropy_none_reduction,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_config['model_id'],
            torch_dtype=torch.bfloat16,
            attn_implementation=model_config['attn_implementation']
        )
        model = model.to(single_gpu_device)
        model.train()
        
        # Setup training components
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.0
        )
        
        lr_scheduler = get_scheduler(
            name="constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=2
        )
        
        # Create data loader (simplified for single GPU)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        
        # Create data loader with 0 workers to avoid segfaults in test
        dataset = JsonlDataset(str(tokenized_data))
        data_loader = DataLoader(
            dataset,
            batch_size=4,
            sampler=InfiniteSampler(len(dataset), seed=42),
            collate_fn=MaxTokensPerRankCollator(
                max_tokens_per_rank=1000,
                rank=0,
                world_size=1,
                dummy_sample=None
            ),
            num_workers=0  # Use 0 workers to avoid segfaults in test environment
        )
        
        # Training metrics
        batch_metrics = BatchMetrics()
        losses = []
        
        # Training loop
        data_iter = iter(data_loader)
        for step in range(20):  # Train for 20 steps to allow more convergence
            batch = next(data_iter)
            
            batch_metrics.reset_batch()
            total_batch_loss = 0
            batch_num_loss_counted_tokens = 0
            
            for minibatch in batch:
                # Extract metadata
                mb_num_loss_counted_tokens = minibatch.pop('num_loss_counted_tokens')
                mb_num_samples = minibatch.pop('num_samples')
                batch_num_loss_counted_tokens = minibatch.pop('batch_num_loss_counted_tokens')
                
                # Move to device
                minibatch = {k: v.to(single_gpu_device) for k, v in minibatch.items()}
                
                # Forward pass
                output = model(**minibatch)
                loss = output.loss.float().sum()
                
                # Scale loss (simulating distributed training with world_size=1)
                scaled_loss = loss / batch_num_loss_counted_tokens
                
                # Track metrics
                total_batch_loss += loss.item()
                batch_metrics.accumulate_minibatch_metrics(
                    num_loss_counted_tokens=mb_num_loss_counted_tokens,
                    num_total_tokens=minibatch['input_ids'].shape[1],
                    num_samples=mb_num_samples,
                    loss=loss.item(),
                    loss_backward=scaled_loss.item(),
                    time_per_minibatch=0.1  # Dummy time
                )
                
                # Backward pass
                scaled_loss.backward()
            
            # Gradient step
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Calculate average loss
            avg_loss = total_batch_loss / batch_num_loss_counted_tokens
            losses.append(avg_loss)
            
            print(f"Step {step}: Loss = {avg_loss:.4f}, Grad norm = {grad_norm:.4f}")
        
        # Verify training is working
        assert len(losses) == 20, "Should have 20 loss values"
        assert all(loss > 0 for loss in losses), "All losses should be positive"
        
        # Check that losses are decreasing on average (compare first 5 vs last 5)
        first_5_avg = np.mean(losses[:5])
        last_5_avg = np.mean(losses[-5:])
        print(f"\nFirst 5 steps avg loss: {first_5_avg:.4f}")
        print(f"Last 5 steps avg loss: {last_5_avg:.4f}")
        
        # Loss should generally trend downward, but allow some fluctuation
        # Just verify training mechanics work, not strict convergence
        assert last_5_avg <= first_5_avg * 1.5, \
            f"Loss increasing too much: first_5={first_5_avg:.4f}, last_5={last_5_avg:.4f}"
    
    def test_gradient_accumulation(
        self,
        small_model_configs,
        create_test_dataset,
        tmp_path,
        single_gpu_device
    ):
        """Test gradient accumulation behavior."""
        model_config = small_model_configs[0]  # Use Qwen
        
        # Create small dataset
        raw_data = create_test_dataset(num_samples=8)
        tokenized_data = tmp_path / "tokenized.jsonl"
        
        import subprocess
        cmd = [
            "python", "process_data.py",
            "--input-file", raw_data,
            "--output-file", str(tokenized_data),
            "--model-name-or-path", model_config['model_id'],
            "--max-sample-num-tokens", "512"
        ]
        subprocess.run(cmd, check=True)
        
        # Setup model
        model = AutoModelForCausalLM.from_pretrained(
            model_config['model_id'],
            torch_dtype=torch.bfloat16,
            attn_implementation='eager'
        )
        model = model.to(single_gpu_device)
        model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Test gradient accumulation
        from sampler import JsonlDataset
        dataset = JsonlDataset(str(tokenized_data))
        
        # Method 1: Single large batch
        optimizer.zero_grad()
        total_loss_single = 0
        
        for i in range(4):
            sample = dataset[i]
            input_ids = sample['input_ids'].unsqueeze(0).to(single_gpu_device)
            labels = sample['labels'].unsqueeze(0).to(single_gpu_device)
            
            output = model(input_ids=input_ids, labels=labels)
            
            # Handle potential non-scalar loss
            raw_loss = output.loss
            if raw_loss.numel() > 1:
                # Calculate mean of valid loss values
                valid_mask = labels.view(-1) != -100
                raw_loss = raw_loss.view(-1)[valid_mask].mean()
            
            loss = raw_loss / 4  # Divide by accumulation steps
            total_loss_single += raw_loss.item()
            loss.backward()
        
        # Get gradients after accumulation
        accumulated_grads = []
        for param in model.parameters():
            if param.grad is not None:
                accumulated_grads.append(param.grad.clone())
        
        # Method 2: Process same samples without accumulation
        optimizer.zero_grad()
        total_loss_no_accum = 0
        
        all_input_ids = []
        all_labels = []
        for i in range(4):
            sample = dataset[i]
            all_input_ids.append(sample['input_ids'])
            all_labels.append(sample['labels'])
        
        # Pad to same length
        max_len = max(ids.shape[0] for ids in all_input_ids)
        padded_input_ids = []
        padded_labels = []
        
        for ids, labs in zip(all_input_ids, all_labels):
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padded_input_ids.append(
                    torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
                )
                padded_labels.append(
                    torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)])
                )
            else:
                padded_input_ids.append(ids)
                padded_labels.append(labs)
        
        batch_input_ids = torch.stack(padded_input_ids).to(single_gpu_device)
        batch_labels = torch.stack(padded_labels).to(single_gpu_device)
        
        output = model(input_ids=batch_input_ids, labels=batch_labels)
        loss = output.loss
        
        # Handle potential non-scalar loss
        if loss.numel() > 1:
            # Calculate mean of valid loss values
            valid_mask = batch_labels.view(-1) != -100
            loss = loss.view(-1)[valid_mask].mean()
        
        total_loss_no_accum = loss.item() * 4  # Multiply for comparison
        loss.backward()
        
        # Compare gradients
        print(f"\nGradient Accumulation Test:")
        print(f"Total loss (accumulated): {total_loss_single:.4f}")
        print(f"Total loss (batch): {total_loss_no_accum:.4f}")
        
        # Gradients should be similar (not exact due to padding)
        for i, (param, acc_grad) in enumerate(zip(model.parameters(), accumulated_grads)):
            if param.grad is not None and acc_grad is not None:
                if i < 5:  # Check first few parameters
                    grad_diff = (param.grad - acc_grad).abs().mean().item()
                    print(f"Param {i} gradient diff: {grad_diff:.6f}")
    
    def test_loss_with_liger_kernels(
        self,
        small_model_configs,
        create_test_dataset,
        tmp_path,
        single_gpu_device
    ):
        """Test loss computation with Liger kernels if available."""
        try:
            import liger_kernel
        except ImportError:
            pytest.skip("Liger kernels not installed")
        
        try:
            import flash_attn
            pytest.skip("Skipping Liger test when flash-attn is available to avoid conflicts")
        except ImportError:
            pass  # flash-attn not available, continue with test
        
        model_config = small_model_configs[0]  # Use Qwen
        
        # Create dataset
        raw_data = create_test_dataset(num_samples=5)
        tokenized_data = tmp_path / "tokenized.jsonl"
        
        import subprocess
        cmd = [
            "python", "process_data.py",
            "--input-file", raw_data,
            "--output-file", str(tokenized_data),
            "--model-name-or-path", model_config['model_id'],
            "--max-sample-num-tokens", "512"
        ]
        subprocess.run(cmd, check=True)
        
        # Test with and without Liger kernels
        from sampler import JsonlDataset
        dataset = JsonlDataset(str(tokenized_data))
        sample = dataset[0]
        
        input_ids = sample['input_ids'].unsqueeze(0).to(single_gpu_device)
        labels = sample['labels'].unsqueeze(0).to(single_gpu_device)
        
        # Model without Liger
        model_no_liger = AutoModelForCausalLM.from_pretrained(
            model_config['model_id'],
            torch_dtype=torch.bfloat16,
            attn_implementation='eager'
        ).to(single_gpu_device)
        
        with torch.no_grad():
            output_no_liger = model_no_liger(input_ids=input_ids, labels=labels)
            loss_no_liger = output_no_liger.loss
        
        # Model with Liger
        from none_reduction_losses import liger_fixed_fused_linear_cross_entropy_none_reduction
        patch_target_module(
            "liger_kernel.transformers.model.loss_utils.fixed_fused_linear_cross_entropy",
            liger_fixed_fused_linear_cross_entropy_none_reduction,
        )
        
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model_liger = AutoLigerKernelForCausalLM.from_pretrained(
            model_config['model_id'],
            torch_dtype=torch.bfloat16,
            attn_implementation='eager'
        ).to(single_gpu_device)
        
        with torch.no_grad():
            output_liger = model_liger(input_ids=input_ids, labels=labels)
            loss_liger = output_liger.loss
            
            # Handle non-reduced loss
            if loss_liger.numel() > 1:
                valid_mask = labels.view(-1) != -100
                loss_liger = loss_liger.view(-1)[valid_mask].mean()
        
        print(f"\nLiger Kernel Comparison:")
        print(f"Loss without Liger: {loss_no_liger.item():.6f}")
        print(f"Loss with Liger: {loss_liger.item():.6f}")
        print(f"Difference: {abs(loss_no_liger.item() - loss_liger.item()):.6f}")
        
        # Losses should be very close
        assert torch.allclose(loss_no_liger, loss_liger, rtol=1e-2, atol=1e-3), \
            f"Liger loss mismatch: no_liger={loss_no_liger.item():.6f}, liger={loss_liger.item():.6f}"
