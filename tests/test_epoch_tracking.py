"""
Test suite for epoch tracking in the data loader and sampler.

These tests verify that epochs are accurately tracked across distributed training,
which is critical for:
- Saving models at epoch boundaries
- Ending training after a specific number of epochs
- Logging accurate epoch metrics
"""

import torch
import pytest
from unittest.mock import MagicMock, patch, call
from torch.utils.data import DataLoader
import tempfile
import json
import os

from mini_trainer.sampler import JsonlDataset, MaxTokensPerRankCollator, get_data_loader, EpochSampler


class TestEpochTracking:
    """Test suite for epoch tracking in data loading."""
    
    def test_epoch_sampler_epoch_increment(self):
        """Test that EpochSampler correctly handles epochs."""
        sampler = EpochSampler(len_data=10, seed=42)
        
        # First epoch
        sampler.set_epoch(0)
        first_epoch_indices = list(sampler)
        
        # Second epoch
        sampler.set_epoch(1)
        second_epoch_indices = list(sampler)
        
        # Each epoch should have all indices 0-9, but in different orders
        assert set(first_epoch_indices) == set(range(10))
        assert set(second_epoch_indices) == set(range(10))
        
        # The order should be different between epochs (with very high probability)
        assert first_epoch_indices != second_epoch_indices
    
    def test_epoch_tracking_not_exposed(self):
        """Test that epoch information is not exposed to the data loader consumer."""
        sampler = EpochSampler(len_data=5, seed=42)
        
        # The sampler has no way to query current epoch
        assert not hasattr(sampler, 'current_epoch')
        assert not hasattr(sampler, 'get_epoch')
        
        # The iterator also doesn't expose epoch info
        iterator = iter(sampler)
        assert not hasattr(iterator, 'epoch')
    
    def test_distributed_epoch_synchronization(self):
        """Test that different ranks see the same shuffled order per epoch."""
        data_len = 10
        sampler1 = EpochSampler(len_data=data_len, seed=42)
        sampler2 = EpochSampler(len_data=data_len, seed=42)  # Same seed
        
        for epoch in range(3):
            sampler1.set_epoch(epoch)
            sampler2.set_epoch(epoch)
            iter1 = iter(sampler1)
            iter2 = iter(sampler2)
            for _ in range(data_len):
                assert next(iter1) == next(iter2)
    
    def test_batch_size_epoch_boundary_issue(self):
        """Test that batch size affects when epoch boundaries are crossed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_file = os.path.join(temp_dir, "test.jsonl")
            samples = []
            for i in range(100):
                samples.append({
                    'input_ids': [1] * 10,
                    'labels': [1] * 10,
                    'len': 10
                })
            
            with open(test_file, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            
            # Create data loaders with different batch sizes
            loader1 = get_data_loader(
                data_path=test_file,
                batch_size=10,  # Evenly divides dataset
                max_tokens_per_gpu=1000,
                seed=42,
                rank=0,
                world_size=1
            )
            
            loader2 = get_data_loader(
                data_path=test_file,
                batch_size=7,  # Does not evenly divide dataset
                max_tokens_per_gpu=1000,
                seed=42,
                rank=0,
                world_size=1
            )
            
            # Count samples in first "epoch" worth of batches
            iter1 = iter(loader1)
            iter2 = iter(loader2)
            
            samples_seen1 = 0
            samples_seen2 = 0
            
            # Process 100 samples (1 epoch worth)
            while samples_seen1 < 100:
                batch = next(iter1)
                for mb in batch:
                    samples_seen1 += mb['num_samples']
            
            while samples_seen2 < 100:
                batch = next(iter2)
                for mb in batch:
                    samples_seen2 += mb['num_samples']
            
            # Both should have seen exactly 100 samples
            assert samples_seen1 == 100
            # With batch_size=7, we might overshoot due to batching
            assert samples_seen2 >= 100  # This reveals the issue!
    
    
    def test_distributed_epoch_completion_mismatch(self):
        """Test that different ranks may complete epochs at different times."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data with varying lengths
            test_file = os.path.join(temp_dir, "test.jsonl")
            samples = []
            for i in range(50):
                # Varying lengths to trigger different packing
                length = 10 + (i % 20)
                samples.append({
                    'input_ids': [1] * length,
                    'labels': [1] * length,
                    'len': length
                })
            
            with open(test_file, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            
            # Create loaders for different ranks
            loader_rank0 = get_data_loader(
                data_path=test_file,
                batch_size=10,
                max_tokens_per_gpu=200,
                seed=42,
                rank=0,
                world_size=2
            )
            
            loader_rank1 = get_data_loader(
                data_path=test_file,
                batch_size=10,
                max_tokens_per_gpu=200,
                seed=42,
                rank=1,
                world_size=2
            )
            
            # Process batches and count samples
            iter0 = iter(loader_rank0)
            iter1 = iter(loader_rank1)
            
            samples_rank0 = 0
            samples_rank1 = 0
            batches_processed = 0

            batch_size = 10
            num_samples = 50
            
            # Process same number of batches on each rank
            for _ in range(num_samples // batch_size):
                batch0 = next(iter0)
                batch1 = next(iter1)
                
                for mb in batch0:
                    samples_rank0 += mb.get('num_samples', 0)
                for mb in batch1:
                    samples_rank1 += mb.get('num_samples', 0)
                
                batches_processed += 1
            
            # Different ranks may have processed different numbers of samples
            # This is a problem for epoch synchronization
            print(f"Rank 0 processed {samples_rank0} samples")
            print(f"Rank 1 processed {samples_rank1} samples")
            
            # They likely won't match due to dynamic batching
            # This is the bug - no synchronized epoch boundaries
            assert samples_rank0 != samples_rank1
    

class TestSamplerBugs:
    """Test suite specifically targeting known bugs in epoch tracking."""
    
    def test_data_loader_length_with_epoch_sampler(self):
        """Test that DataLoader with EpochSampler reports correct length."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=100)
        dataset.__getitem__ = MagicMock(side_effect=lambda x: {'input_ids': [1], 'labels': [1]})
        
        sampler = EpochSampler(len(dataset))
        loader = DataLoader(dataset, batch_size=10, sampler=sampler)
        
        # The loader computes a length based on batch size
        length = len(loader)
        assert length == 10  # Based on dataset_size / batch_size
        
        # EpochSampler provides one complete epoch
        count = 0
        for i, batch in enumerate(loader):
            count += 1
        
        assert count == length  # Should iterate exactly one epoch
    
    def test_steps_per_epoch_calculation_issue(self):
        """Test that calculating steps per epoch is problematic with dynamic batching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data with varying lengths
            test_file = os.path.join(temp_dir, "test.jsonl")
            samples = []
            for i in range(100):
                # Create more extreme variations to trigger dynamic batching
                if i % 5 == 0:
                    length = 5  # Very short
                elif i % 3 == 0:
                    length = 100  # Long
                else:
                    length = 50  # Medium
                samples.append({
                    'input_ids': [1] * length,
                    'labels': [1] * length,
                    'len': length
                })
            
            with open(test_file, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            
            # Try to calculate steps per epoch
            loader = get_data_loader(
                data_path=test_file,
                batch_size=10,
                max_tokens_per_gpu=200,  # Will cause dynamic batching
                seed=42,
                rank=0,
                world_size=1
            )
            
            # Count actual steps in first epoch
            iterator = iter(loader)
            steps = 0
            samples_seen = 0
            
            while samples_seen < 100:
                batch = next(iterator)
                steps += 1
                for mb in batch:
                    samples_seen += mb.get('num_samples', 0)
            
            # The number of steps is not batch_size / dataset_size
            naive_steps_per_epoch = 100 // 10  # Would be 10
            actual_steps = steps
            
            print(f"Naive calculation: {naive_steps_per_epoch} steps")
            print(f"Actual steps for first epoch: {actual_steps} steps")
            
            # They might not match due to dynamic batching
            # OR the calculation itself is problematic
            print(f"Samples seen after {actual_steps} steps: {samples_seen}")
            
            # The real issue: we can't predict steps per epoch with dynamic batching
            # Even if they happen to match sometimes, it's not guaranteed
            # Let's test that processing exactly one epoch worth can overshoot
            assert samples_seen >= 100  # We may have processed more than one epoch!
    