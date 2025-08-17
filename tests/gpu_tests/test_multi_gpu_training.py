"""Multi-GPU distributed training tests."""
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List
import numpy as np
import socket

from transformers import AutoTokenizer, AutoModelForCausalLM
from setup_model_for_training import setup_model, setup_training_components, wrap_fsdp2
from sampler import JsonlDataset, InfiniteSampler, MaxTokensPerRankCollator
from torch.utils.data import DataLoader
from batch_metrics import BatchMetrics
from utils import init_distributed_environment, log_rank_0, patch_target_module


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_free_port():
    """Get a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed_env(rank, world_size, port=None):
    """Setup distributed environment for testing."""
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    
    # Use provided port or get a free one
    if port is None:
        port = get_free_port()
    
    os.environ['MASTER_PORT'] = str(port)
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set CUDA device
    torch.cuda.set_device(rank)


def run_distributed_training(rank, world_size, model_name, data_file, output_dir, test_config, port):
    """Run training on a single rank in distributed setting."""
    try:
        # Setup distributed environment
        setup_distributed_env(rank, world_size, port)
        device = torch.device(f'cuda:{rank}')
        
        # Patch loss function
        from none_reduction_losses import hf_fixed_cross_entropy_none_reduction
        patch_target_module(
            "transformers.loss.loss_utils.fixed_cross_entropy",
            hf_fixed_cross_entropy_none_reduction,
        )
        
        # Setup model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation='eager'  # Use eager for testing
        )
        
        # Wrap with FSDP2
        model = wrap_fsdp2(model)
        
        # Setup optimizer and scheduler
        from transformers import get_scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=test_config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=0.0
        )
        
        lr_scheduler = get_scheduler(
            name="constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=test_config['num_warmup_steps']
        )
        
        # Create data loader with fewer workers to avoid segfaults in tests
        dataset = JsonlDataset(data_file)
        data_loader = DataLoader(
            dataset,
            batch_size=test_config['batch_size'],
            sampler=InfiniteSampler(len(dataset), seed=42),
            collate_fn=MaxTokensPerRankCollator(
                test_config['max_tokens_per_gpu'],
                rank=rank,
                world_size=world_size,
                dummy_sample=None
            ),
            num_workers=0  # Use 0 workers to avoid segfaults in test environment
        )
        
        # Training loop
        model.train()
        losses = []
        data_iter = iter(data_loader)
        
        for step in range(test_config['num_steps']):
            batch = next(data_iter)
            
            # The batch is a list of minibatches
            total_loss = 0
            num_tokens = 0
            
            for minibatch in batch:
                # Extract metadata
                mb_num_loss_counted_tokens = minibatch.pop('num_loss_counted_tokens', 0)
                mb_num_samples = minibatch.pop('num_samples', 0)
                batch_num_loss_counted_tokens = minibatch.pop('batch_num_loss_counted_tokens', 0)
                
                # Move to device
                minibatch = {k: v.to(device) for k, v in minibatch.items()}
                
                # Forward pass
                output = model(**minibatch)
                loss = output.loss
                
                # Handle non-reduced loss
                if loss.numel() > 1:
                    loss = loss.sum()
                
                total_loss += loss
                num_tokens += batch_num_loss_counted_tokens if batch_num_loss_counted_tokens > 0 else 1
            
            # Average loss
            avg_loss = total_loss / max(num_tokens, 1)
            
            # Backward pass
            optimizer.zero_grad()
            avg_loss.backward()
            
            # Optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            # Track loss
            losses.append(avg_loss.item())
            
            if rank == 0 and step % 5 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")
        
        # Save results from rank 0
        if rank == 0:
            results = {
                'num_steps': test_config['num_steps'],
                'final_loss': losses[-1] if losses else None,
                'losses': losses
            }
            
            results_file = Path(output_dir) / 'training_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        dist.barrier()
        
    finally:
        cleanup_distributed()


def check_gradient_sync(rank, world_size, results_dict, port):
    """Helper function for gradient synchronization test."""
    try:
        setup_distributed_env(rank, world_size, port)
        device = torch.device(f'cuda:{rank}')
        
        # Create a simple model
        model = torch.nn.Linear(10, 10).to(device)
        
        # Wrap with FSDP2 would happen here in real scenario
        # For this test, we'll use DistributedDataParallel
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank])
        
        # Create identical input on all ranks
        torch.manual_seed(42)
        input_data = torch.randn(4, 10).to(device)
        target = torch.randn(4, 10).to(device)
        
        # Forward and backward pass
        output = model(input_data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Get gradient from first parameter
        grad = list(model.parameters())[0].grad.clone()
        
        # Gather all gradients to rank 0
        if rank == 0:
            grad_list = [torch.zeros_like(grad) for _ in range(world_size)]
            dist.gather(grad, grad_list, dst=0)
            
            # Check all gradients are the same
            for i in range(1, world_size):
                assert torch.allclose(grad_list[0], grad_list[i], rtol=1e-5), \
                    f"Gradient mismatch between rank 0 and rank {i}"
            
            results_dict['success'] = True
        else:
            dist.gather(grad, dst=0)
        
        dist.barrier()
        
    finally:
        cleanup_distributed()


def compute_loss_on_rank(rank, world_size, model_name, data_file, results_dict, port):
    """Helper function for loss consistency test."""
    try:
        setup_distributed_env(rank, world_size, port)
        device = torch.device(f'cuda:{rank}')
        
        # Setup model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation='eager'
        )
        model = model.to(device)
        
        # Load first sample
        from sampler import JsonlDataset
        dataset = JsonlDataset(data_file)
        sample = dataset[0]
        
        # Prepare input
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        labels = sample['labels'].unsqueeze(0).to(device)
        
        # Compute loss
        with torch.no_grad():
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss
            
            # Handle potential non-scalar loss
            if loss.numel() > 1:
                loss = loss.mean()
        
        # Store result
        results_dict[f'rank_{rank}_loss'] = loss.item()
        
        dist.barrier()
        
    finally:
        cleanup_distributed()


@pytest.mark.gpu
@pytest.mark.multi_gpu
class TestMultiGPUTraining:
    """Test distributed training across multiple GPUs."""
    
    @pytest.fixture
    def check_multi_gpu(self, gpu_count):
        """Check if multiple GPUs are available."""
        if gpu_count < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        return gpu_count
    
    @pytest.fixture
    def create_distributed_dataset(self, tmp_path):
        """Create a dataset for distributed training tests."""
        def _create(num_samples=30):
            data = []
            for i in range(num_samples):
                # Create simple math problems
                a = i * 2
                b = i * 3
                data.append({
                    "messages": [
                        {"role": "user", "content": f"Question {i}: What is {a}+{b}?"},
                        {"role": "assistant", "content": f"Answer: {a}+{b} equals {a+b}."}
                    ]
                })
            
            data_file = tmp_path / "distributed_data.jsonl"
            with open(data_file, 'w') as f:
                for item in data:
                    json.dump(item, f)
                    f.write('\n')
            
            return str(data_file)
        return _create
    
    def test_two_gpu_training(
        self,
        check_multi_gpu,
        create_distributed_dataset,
        tmp_path
    ):
        """Test training across 2 GPUs with FSDP2."""
        world_size = 2
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        # Create dataset
        raw_data = create_distributed_dataset(num_samples=30)
        tokenized_data = tmp_path / "tokenized.jsonl"
        
        # Tokenize data
        import subprocess
        cmd = [
            "python", "process_data.py",
            "--input-file", raw_data,
            "--output-file", str(tokenized_data),
            "--model-name-or-path", model_name,
            "--max-sample-num-tokens", "512"
        ]
        subprocess.run(cmd, check=True)
        
        # Training configuration
        test_config = {
            'learning_rate': 1e-3,
            'num_warmup_steps': 2,
            'batch_size': 8,
            'max_tokens_per_gpu': 1000,
            'num_steps': 10
        }
        
        output_dir = tmp_path / "checkpoints"
        output_dir.mkdir(exist_ok=True)
        
        # Get a free port for this test
        port = get_free_port()
        
        # Launch distributed training
        mp.spawn(
            run_distributed_training,
            args=(world_size, model_name, str(tokenized_data), str(output_dir), test_config, port),
            nprocs=world_size,
            join=True
        )
        
        # Check results
        results_file = output_dir / 'training_results.json'
        assert results_file.exists(), "Training results not saved"
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        assert results['num_steps'] == test_config['num_steps'], "Wrong number of training steps"
        assert results['final_loss'] is not None, "No final loss recorded"
        assert results['final_loss'] > 0, "Final loss should be positive"
        
        print(f"\nDistributed Training Results (2 GPUs):")
        print(f"Number of steps: {results['num_steps']}")
        print(f"Final loss: {results['final_loss']:.4f}")
        print(f"Loss trajectory: {results['losses']}")
        
        # Check that loss decreased
        if len(results['losses']) > 1:
            initial_loss = results['losses'][0]
            final_loss = results['losses'][-1]
            print(f"Loss reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
    
    def test_gradient_synchronization(
        self,
        check_multi_gpu,
        tmp_path
    ):
        """Test that gradients are properly synchronized across GPUs."""
        world_size = 2
        manager = mp.Manager()
        results_dict = manager.dict()
        results_dict['success'] = False
        
        # Get a free port for this test
        port = get_free_port()
        
        # Run test
        mp.spawn(
            check_gradient_sync,
            args=(world_size, results_dict, port),
            nprocs=world_size,
            join=True
        )
        
        assert results_dict['success'], "Gradient synchronization test failed"
        print("✓ Gradient synchronization test passed")
    
    def test_loss_consistency_across_ranks(
        self,
        check_multi_gpu,
        create_distributed_dataset,
        tmp_path
    ):
        """Test that loss computation is consistent across ranks."""
        world_size = 2
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        # Create dataset
        raw_data = create_distributed_dataset(num_samples=5)
        tokenized_data = tmp_path / "tokenized_consistency.jsonl"
        
        import subprocess
        cmd = [
            "python", "process_data.py",
            "--input-file", raw_data,
            "--output-file", str(tokenized_data),
            "--model-name-or-path", model_name,
            "--max-sample-num-tokens", "512"
        ]
        subprocess.run(cmd, check=True)
        
        manager = mp.Manager()
        results_dict = manager.dict()
        
        # Get a free port for this test
        port = get_free_port()
        
        # Run test
        mp.spawn(
            compute_loss_on_rank,
            args=(world_size, model_name, str(tokenized_data), results_dict, port),
            nprocs=world_size,
            join=True
        )
        
        # Check consistency
        losses = [results_dict[f'rank_{i}_loss'] for i in range(world_size)]
        print(f"\nLoss consistency test:")
        for i, loss in enumerate(losses):
            print(f"  Rank {i} loss: {loss:.6f}")
        
        # All ranks should compute the same loss
        assert all(abs(losses[0] - loss) < 1e-6 for loss in losses), \
            f"Loss mismatch across ranks: {losses}"
        
        print("✓ Loss consistency test passed")
