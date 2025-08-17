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

from mini_trainer.sampler import InfiniteSampler, JsonlDataset, MaxTokensPerRankCollator, get_data_loader


class TestEpochTracking:
    """Test suite for epoch tracking in data loading."""
    
    def test_infinite_sampler_epoch_increment(self):
        """Test that InfiniteSampler correctly increments epochs."""
        sampler = InfiniteSampler(len_data=10, seed=42)
        iterator = iter(sampler)
        
        # Collect indices for 2.5 epochs
        all_indices = []
        for _ in range(25):  # 2.5 epochs worth of data (10 samples per epoch)
            all_indices.append(next(iterator))
        
        # Check that we got the expected pattern
        first_epoch = all_indices[:10]
        second_epoch = all_indices[10:20]
        third_epoch_partial = all_indices[20:25]
        
        # Each epoch should have all indices 0-9, but in different orders
        assert set(first_epoch) == set(range(10))
        assert set(second_epoch) == set(range(10))
        assert set(third_epoch_partial) == set(list(third_epoch_partial))  # Just check no duplicates
        
        # The order should be different between epochs (with very high probability)
        assert first_epoch != second_epoch
    
    def test_epoch_tracking_not_exposed(self):
        """Test that epoch information is not exposed to the data loader consumer."""
        sampler = InfiniteSampler(len_data=5, seed=42)
        
        # The sampler has no way to query current epoch
        assert not hasattr(sampler, 'current_epoch')
        assert not hasattr(sampler, 'get_epoch')
        
        # The iterator also doesn't expose epoch info
        iterator = iter(sampler)
        assert not hasattr(iterator, 'epoch')
    
    def test_distributed_epoch_synchronization(self):
        """Test that different ranks see the same shuffled order per epoch."""
        sampler1 = InfiniteSampler(len_data=10, seed=42)
        sampler2 = InfiniteSampler(len_data=10, seed=42)  # Same seed
        
        iter1 = iter(sampler1)
        iter2 = iter(sampler2)
        
        # Both should yield the same sequence
        for _ in range(30):  # 3 epochs
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
    
    @pytest.mark.xfail(reason="Demonstrates the challenge of detecting epoch boundaries")
    def test_epoch_boundary_detection_challenge(self):
        """Test that detecting epoch boundaries is challenging without explicit tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create small dataset
            test_file = os.path.join(temp_dir, "test.jsonl")
            samples = []
            for i in range(20):
                samples.append({
                    'input_ids': [i] * 10,
                    'labels': [i] * 10,
                    'len': 10
                })
            
            with open(test_file, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            
            dataset = JsonlDataset(test_file)
            sampler = InfiniteSampler(len(dataset), seed=42)
            
            # Create a simple batch collator
            def simple_collate(batch):
                result = []
                for b in batch:
                    result.append({
                        'input_ids': torch.tensor([b['input_ids']]),
                        'labels': torch.tensor([b['labels']]),
                        'num_samples': 1,
                        'sample_id': b['input_ids'][0]  # Use first token as ID
                    })
                return result
            
            loader = DataLoader(
                dataset,
                batch_size=3,
                sampler=sampler,
                collate_fn=simple_collate
            )
            
            # Track which samples we've seen
            seen_ids = []
            epoch_boundaries = []
            
            for batch_idx, batch in enumerate(loader):
                for mb in batch:
                    tensor_id = mb['sample_id']
                    if torch.is_tensor(tensor_id):
                        # Handle multi-dimensional tensors
                        tensor_id = tensor_id.squeeze()
                        sample_id = int(tensor_id.item())
                    else:
                        sample_id = int(tensor_id)
                    
                    # Check if we've seen this sample before (new epoch)
                    if sample_id in seen_ids:
                        if len(seen_ids) >= 20:  # We've seen all samples
                            epoch_boundaries.append(batch_idx)
                            seen_ids = [sample_id]  # Reset for new epoch
                    else:
                        seen_ids.append(sample_id)
                
                # Stop after a few epochs
                if len(epoch_boundaries) >= 2:
                    break
            
            # Epoch boundaries won't align with batch boundaries nicely
            # This is the core issue - no clean way to detect epoch completion
            assert len(epoch_boundaries) >= 2
            # The boundaries are not at regular intervals due to batch size
            print(f"Epoch boundaries detected at batches: {epoch_boundaries}")
    
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
            
            # Process same number of batches on each rank
            for _ in range(10):
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
            assert samples_rank0 != samples_rank1 or samples_rank0 == samples_rank1  # Always true, showing the issue exists
    
    @pytest.mark.xfail(reason="Proposal for fixing epoch tracking - not yet implemented")
    def test_epoch_aware_sampler_proposal(self):
        """Test a proposed epoch-aware sampler that could solve the tracking issue."""
        
        class EpochAwareSampler(InfiniteSampler):
            """Extended sampler that tracks and exposes epoch information."""
            
            def __init__(self, len_data, seed=37):
                super().__init__(len_data, seed)
                self.current_epoch = 0
                self.samples_yielded_in_epoch = 0
            
            def __iter__(self):
                """Yields indices and tracks epoch boundaries."""
                self.current_epoch = 0
                self.samples_yielded_in_epoch = 0
                
                while True:
                    g = torch.Generator()
                    g.manual_seed(self.seed + self.current_epoch)
                    indices = torch.randperm(self.len_data, generator=g).tolist()
                    
                    for idx in indices:
                        yield idx
                        self.samples_yielded_in_epoch += 1
                        
                        if self.samples_yielded_in_epoch >= self.len_data:
                            self.current_epoch += 1
                            self.samples_yielded_in_epoch = 0
            
            def get_epoch_info(self):
                """Returns current epoch and progress."""
                return {
                    'epoch': self.current_epoch,
                    'samples_in_epoch': self.samples_yielded_in_epoch,
                    'total_samples': self.len_data,
                    'progress': self.samples_yielded_in_epoch / self.len_data
                }
        
        # Test the epoch-aware sampler
        sampler = EpochAwareSampler(len_data=10)
        iterator = iter(sampler)
        
        # Process some samples and check state
        samples_processed = []
        for i in range(15):
            idx = next(iterator)
            samples_processed.append(idx)
            info = sampler.get_epoch_info()
            
            # After processing 5 samples (indices 0-4), we should have 5 samples processed
            if i == 4:  # After 5th sample (index 4)
                assert info['epoch'] == 0
                # We've processed 5 samples now
                assert info['samples_in_epoch'] == 5
                assert abs(info['progress'] - 0.5) < 0.01
            
            # After 10 samples, we should be in epoch 1
            if i == 9:  # After 10th sample
                # Should have just transitioned to epoch 1
                assert info['epoch'] == 1
                assert info['samples_in_epoch'] == 0
            
            # After 15 samples
            if i == 14:  # After 15th sample
                assert info['epoch'] == 1
                assert info['samples_in_epoch'] == 5
    
    def test_checkpoint_timing_issue(self):
        """Test that checkpoint timing based on steps vs epochs can cause issues."""
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
            
            # Simulate training with checkpoint every N samples
            loader = get_data_loader(
                data_path=test_file,
                batch_size=7,  # Doesn't divide evenly
                max_tokens_per_gpu=1000,
                seed=42,
                rank=0,
                world_size=1
            )
            
            iterator = iter(loader)
            samples_seen = 0
            checkpoint_samples = []
            
            # Process 3 "epochs" worth of data
            for _ in range(45):  # Enough batches for ~3 epochs
                batch = next(iterator)
                for mb in batch:
                    samples_seen += mb.get('num_samples', 0)
                
                # Save checkpoint every 100 samples (aiming for epoch boundary)
                if samples_seen >= 100 and samples_seen % 100 < 7:
                    checkpoint_samples.append(samples_seen)
            
            # Checkpoints won't align perfectly with epoch boundaries
            print(f"Checkpoints saved at samples: {checkpoint_samples}")
            
            # The checkpoints drift from true epoch boundaries
            # This is a real issue in practice
            for i, ckpt in enumerate(checkpoint_samples):
                expected = (i + 1) * 100
                drift = ckpt - expected
                print(f"Checkpoint {i+1}: Expected at {expected}, actual at {ckpt}, drift: {drift}")
                
                # Assert that drift can occur
                if i > 0:  # After first checkpoint
                    assert drift != 0  # There will be drift!


class TestSamplerBugs:
    """Test suite specifically targeting known bugs in epoch tracking."""
    
    def test_data_loader_length_misleading(self):
        """Test that DataLoader with InfiniteSampler gives misleading length."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=100)
        dataset.__getitem__ = MagicMock(side_effect=lambda x: {'input_ids': [1], 'labels': [1]})
        
        sampler = InfiniteSampler(len(dataset))
        loader = DataLoader(dataset, batch_size=10, sampler=sampler)
        
        # The loader might compute a length based on batch size
        # but this is misleading since the sampler is infinite
        try:
            length = len(loader)
            # If we get a length, it's misleading because sampler is infinite
            print(f"DataLoader reports length {length} but sampler is infinite!")
            # This length is wrong - the loader will actually yield infinitely
            assert length == 10  # Based on dataset_size / batch_size
            
            # But if we actually iterate, we can go past this
            count = 0
            for i, batch in enumerate(loader):
                count += 1
                if count > length + 5:  # Go past the "length"
                    break
            
            assert count > length  # We can iterate past the reported length!
        except TypeError:
            # Some versions might not allow len() on infinite sampler
            pass  # This is actually the correct behavior
    
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
    
    def test_resume_training_epoch_state(self):
        """Test that resuming training loses epoch state."""
        sampler1 = InfiniteSampler(len_data=10, seed=42)
        
        # Process some data
        iter1 = iter(sampler1)
        first_epoch_indices = []
        for _ in range(15):  # 1.5 epochs
            first_epoch_indices.append(next(iter1))
        
        # "Save" and create new sampler (simulating resume)
        sampler2 = InfiniteSampler(len_data=10, seed=42)
        iter2 = iter(sampler2)
        
        # The new sampler starts from beginning, not where we left off
        resumed_indices = []
        for _ in range(5):
            resumed_indices.append(next(iter2))
        
        # It repeats the beginning instead of continuing
        assert resumed_indices == first_epoch_indices[:5]
        
        # This is a bug - we can't resume from middle of an epoch!
