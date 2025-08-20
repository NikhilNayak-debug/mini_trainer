"""
Test suite for data loading and sampling components.

Tests the JsonlDataset and MaxTokensPerRankCollator
to ensure correct data loading, sampling, and batching behavior.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
from torch.utils.data import DataLoader

from mini_trainer.sampler import (
    JsonlDataset,
    MaxTokensPerRankCollator,
    get_data_loader,
    mb_collate_fn,
    reset_minibatches,
    EpochSampler
)


class TestJsonlDataset:
    """Test suite for the JsonlDataset class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample JSONL data for testing."""
        return [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "labels": [10, 20, -100, 40, 50],
                "len": 5,
                "num_loss_counted_tokens": 4
            },
            {
                "input_ids": [6, 7, 8, 9],
                "labels": [-100, -100, 80, 90],
                "len": 4,
                "num_loss_counted_tokens": 2
            },
            {
                "input_ids": [11, 12, 13],
                "labels": [110, 120, 130],
                "len": 3
                # Missing num_loss_counted_tokens to test fallback
            }
        ]
    
    @pytest.fixture
    def temp_jsonl_file(self, sample_data):
        """Create a temporary JSONL file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in sample_data:
                json.dump(item, f)
                f.write('\n')
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_dataset_initialization(self, temp_jsonl_file):
        """Test dataset initialization with valid JSONL file."""
        dataset = JsonlDataset(path=temp_jsonl_file)
        assert len(dataset) == 3
    
    def test_dataset_getitem(self, temp_jsonl_file):
        """Test retrieving items from dataset."""
        dataset = JsonlDataset(path=temp_jsonl_file)
        
        # Test first item
        item = dataset[0]
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['labels'], torch.Tensor)
        assert item['input_ids'].tolist() == [1, 2, 3, 4, 5]
        assert item['labels'].tolist() == [10, 20, -100, 40, 50]
        assert item['len'] == 5
        assert item['num_loss_counted_tokens'] == 4
        
        # Test second item
        item = dataset[1]
        assert item['input_ids'].tolist() == [6, 7, 8, 9]
        assert item['num_loss_counted_tokens'] == 2
    
    def test_dataset_missing_loss_counted_tokens(self, temp_jsonl_file):
        """Test fallback calculation when num_loss_counted_tokens is missing."""
        dataset = JsonlDataset(path=temp_jsonl_file)
        
        # Third item has missing num_loss_counted_tokens
        item = dataset[2]
        assert item['input_ids'].tolist() == [11, 12, 13]
        assert item['labels'].tolist() == [110, 120, 130]
        # Should calculate from labels (all non -100)
        assert item['num_loss_counted_tokens'] == 3
    
    def test_dataset_index_types(self, temp_jsonl_file):
        """Test dataset accepts different index types."""
        dataset = JsonlDataset(path=temp_jsonl_file)
        
        # Test with int
        item = dataset[0]
        assert item is not None
        
        # Test with numpy int
        item = dataset[np.int64(1)]
        assert item is not None



class TestMbCollateFn:
    """Test suite for the minibatch collate function."""
    
    def test_collate_single_sample(self):
        """Test collating a single sample."""
        minibatch = [{
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'labels': torch.tensor([10, -100, 30, 40, 50]),
            'num_loss_counted_tokens': 4
        }]
        
        result = mb_collate_fn(minibatch, batch_num_loss_counted_tokens=4)
        
        assert result['input_ids'].shape == (1, 5)
        assert result['labels'].shape == (1, 5)
        assert result['position_ids'].shape == (1, 5)
        assert result['position_ids'].tolist() == [[0, 1, 2, 3, 4]]
        assert result['num_loss_counted_tokens'] == 4
        assert result['num_samples'] == 1
        assert result['batch_num_loss_counted_tokens'] == 4
    
    def test_collate_multiple_samples(self):
        """Test collating multiple samples into packed format."""
        minibatch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([10, 20, 30]),
                'num_loss_counted_tokens': 3
            },
            {
                'input_ids': torch.tensor([4, 5, 6, 7]),
                'labels': torch.tensor([40, -100, 60, 70]),
                'num_loss_counted_tokens': 3
            }
        ]
        
        result = mb_collate_fn(minibatch, batch_num_loss_counted_tokens=6)
        
        # Check concatenation
        assert result['input_ids'].shape == (1, 7)
        assert result['input_ids'].tolist() == [[1, 2, 3, 4, 5, 6, 7]]
        assert result['labels'].tolist() == [[10, 20, 30, 40, -100, 60, 70]]
        
        # Check position_ids reset for each sequence
        assert result['position_ids'].tolist() == [[0, 1, 2, 0, 1, 2, 3]]
        
        assert result['num_loss_counted_tokens'] == 6
        assert result['num_samples'] == 2
    
    def test_collate_with_dummy_sample(self):
        """Test collating with dummy samples (padding)."""
        minibatch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([10, 20, 30]),
                'num_loss_counted_tokens': 3
            },
            {
                'input_ids': torch.tensor([99, 99, 99]),
                'labels': torch.tensor([-100, -100, -100]),
                'num_loss_counted_tokens': 0  # Dummy sample
            }
        ]
        
        result = mb_collate_fn(minibatch, batch_num_loss_counted_tokens=3)
        
        # Dummy samples shouldn't count
        assert result['num_samples'] == 1
        assert result['num_loss_counted_tokens'] == 3


class TestMaxTokensPerRankCollator:
    """Test suite for the MaxTokensPerRankCollator class."""
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        return [
            {'input_ids': torch.tensor([1]*100), 'labels': torch.tensor([1]*100), 
             'len': 100, 'num_loss_counted_tokens': 100},
            {'input_ids': torch.tensor([2]*200), 'labels': torch.tensor([2]*200), 
             'len': 200, 'num_loss_counted_tokens': 200},
            {'input_ids': torch.tensor([3]*300), 'labels': torch.tensor([3]*300), 
             'len': 300, 'num_loss_counted_tokens': 300},
            {'input_ids': torch.tensor([4]*400), 'labels': torch.tensor([4]*400), 
             'len': 400, 'num_loss_counted_tokens': 400},
        ]
    
    @patch('mini_trainer.sampler.dist.get_rank', return_value=0)
    @patch('mini_trainer.sampler.dist.get_world_size', return_value=2)
    def test_collator_initialization(self, mock_world_size, mock_rank):
        """Test collator initialization with distributed settings."""
        collator = MaxTokensPerRankCollator(max_tokens_per_rank=1000)
        
        assert collator.max_tokens_per_rank == 1000
        assert collator.rank == 0
        assert collator.world_size == 2
        assert collator.dummy_sample is not None
    
    def test_collator_custom_dummy_sample(self):
        """Test collator with custom dummy sample."""
        dummy = {
            'input_ids': torch.tensor([999, 999]),
            'labels': torch.tensor([-100, -100]),
            'len': 2,
            'num_loss_counted_tokens': 0
        }
        
        collator = MaxTokensPerRankCollator(
            max_tokens_per_rank=1000,
            rank=0,
            world_size=2,
            dummy_sample=dummy
        )
        
        assert collator.dummy_sample == dummy
    
    @patch('mini_trainer.sampler.batch_lengths_to_minibatches_lpt')
    def test_collator_filters_long_sequences(self, mock_batch_fn, sample_batch, capsys):
        """Test that collator filters sequences longer than max_tokens."""
        mock_batch_fn.return_value = [[0, 1]]
        
        # Add a very long sequence
        sample_batch.append({
            'input_ids': torch.tensor([5]*2000),
            'labels': torch.tensor([5]*2000),
            'len': 2000,
            'num_loss_counted_tokens': 2000
        })
        
        collator = MaxTokensPerRankCollator(
            max_tokens_per_rank=1000,
            rank=0,
            world_size=2
        )
        
        result = collator(sample_batch)
        
        # Check that warning was printed
        captured = capsys.readouterr()
        assert "removed 1 samples" in captured.out
        
        # Check that the long sequence was properly filtered
        mock_batch_fn.assert_called_once()
        batch_lengths = mock_batch_fn.call_args[0][0]
        # The filtered batch should only have the original 4 sequences
        assert len(batch_lengths) == 4
        assert 2000 not in batch_lengths
        assert all(length <= 1000 for length in batch_lengths)
    
    @patch('mini_trainer.sampler.batch_lengths_to_minibatches_lpt')
    def test_collator_returns_minibatches(self, mock_batch_fn, sample_batch):
        """Test that collator returns properly formatted minibatches."""
        mock_batch_fn.return_value = [[0, 1], [2, -1]]  # Two minibatches
        
        collator = MaxTokensPerRankCollator(
            max_tokens_per_rank=500,
            rank=0,
            world_size=2
        )
        
        result = collator(sample_batch)
        
        assert len(result) == 2  # Two minibatches
        
        # Each result should be a dictionary with required keys
        for mb in result:
            assert 'input_ids' in mb
            assert 'labels' in mb
            assert 'position_ids' in mb
            assert 'num_loss_counted_tokens' in mb
            assert 'num_samples' in mb
            assert 'batch_num_loss_counted_tokens' in mb


class TestGetDataLoader:
    """Test suite for the get_data_loader function."""
    
    @pytest.fixture
    def temp_data_file(self):
        """Create a temporary data file for testing."""
        data = [
            {
                "input_ids": list(range(100)),
                "labels": list(range(100)),
                "len": 100,
                "num_loss_counted_tokens": 100
            }
            for _ in range(10)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    @patch('mini_trainer.sampler.dist.get_rank', return_value=0)
    @patch('mini_trainer.sampler.dist.get_world_size', return_value=1)
    def test_get_data_loader_basic_epoch_sampler(self, mock_world_size, mock_rank, temp_data_file):
        """Test basic data loader creation."""
        expected_batch_size = 4
        expected_grad_accum_steps = 1

        loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=4,
            max_tokens_per_gpu=500,
            seed=42,
        )
        # asserts that loading the model works properly
        assert loader is not None
        assert loader.batch_size == expected_batch_size
        assert loader.sampler.epoch == 0

        # get an iterator and fetch one batch
        loader_it = iter(loader)
        batch = next(loader_it)

        # we expect there to only be a single grad accum step, since the
        # temp data file creates 100-token samples, so batch-size of 4
        # should only return a single batch
        assert len(batch) == expected_grad_accum_steps
        microbatch = batch[0]
        assert microbatch['num_samples'] == expected_batch_size
        assert loader.sampler.epoch == 0


        # now pop another 2 batches off, the last should either not exist or only have 2 samples
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch['num_samples'] == expected_batch_size
        assert loader.sampler.epoch == 0

        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch['num_samples'] == 2  # we now expect the last 2 samples to be here

        # now we should have seen all samples, but we need to incremenet the epoch
        assert loader.sampler.epoch == 0
        loader.sampler.set_epoch(1)
        assert loader.sampler.epoch == 1


    @patch('mini_trainer.sampler.dist.get_rank', return_value=0)
    @patch('mini_trainer.sampler.dist.get_world_size', return_value=1)
    def test_get_data_loader_epoch_wraparound(self, mock_world_size, mock_rank, temp_data_file):
        """Test data loader behavior at epoch boundaries with EpochSampler."""
        expected_batch_size = 8
        expected_grad_accum_steps = 1

        loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=expected_batch_size,
            max_tokens_per_gpu=50000,  # so we dont accumulate in this test
            seed=42,
        )
        # asserts that loading the model works properly
        assert loader is not None
        assert loader.batch_size == expected_batch_size
        assert loader.sampler.epoch == 0

        # get an iterator and fetch one batch
        loader_it = iter(loader)
        batch = next(loader_it)
        assert len(batch) == expected_grad_accum_steps

        # we expect there to only be a single grad accum step, since the
        # temp data file creates 10 100-token samples, so batch-size 8 w/ 100 tokens per sample
        # should only return a single batch
        assert len(batch) == expected_grad_accum_steps
        microbatch = batch[0]
        assert microbatch['num_samples'] == expected_batch_size
        assert loader.sampler.epoch == 0

        # The next batch should only have 2 samples (10 total - 8 already seen = 2 remaining)
        # After this, the epoch should be complete
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch['num_samples'] == 2  # Only 2 samples left in the epoch
        assert loader.sampler.epoch == 0  # Epoch counter doesn't auto-increment

        # Iterator should be exhausted now - need to create a new one for next epoch
        # This would typically happen in the training loop
        loader.sampler.set_epoch(1)  # Manually set to next epoch
        assert loader.sampler.epoch == 1  # Now it's incremented
        loader_it = iter(loader)  # Create new iterator for the new epoch
        
        # Now we should get a full batch again from the new epoch
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch['num_samples'] == expected_batch_size
        assert loader.sampler.epoch == 1  # Still at epoch 1



    @patch('mini_trainer.sampler.dist.get_rank', return_value=0)
    @patch('mini_trainer.sampler.dist.get_world_size', return_value=2)
    def test_get_data_loader_two_gpus(self, mock_world_size, mock_rank, temp_data_file):
        """Test basic data loader creation."""
        expected_batch_size = 8
        expected_grad_accum_steps = 1
        expected_leftover_samples = 2

        # first try with an even batch size, so the last step will have 2 samples
        loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=8,
            max_tokens_per_gpu=500,
            seed=42,
        )
        # asserts that loading the model works properly
        assert loader is not None
        assert loader.batch_size == expected_batch_size
        assert loader.sampler.epoch == 0

        # get an iterator and fetch one batch
        loader_it = iter(loader)
        batch = next(loader_it)

        # we expect there to only be a single grad accum step, since the
        # temp data file creates 100-token samples, so batch-size of 4
        # should only return a single batch
        assert len(batch) == expected_grad_accum_steps
        microbatch = batch[0]
        assert microbatch['num_samples'] == expected_batch_size // mock_world_size()
        assert loader.sampler.epoch == 0

        # so now the next batch should have 1 sample on each rank
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch['num_samples'] == expected_leftover_samples // mock_world_size()
        assert loader.sampler.epoch == 0


    @patch('mini_trainer.sampler.dist.get_rank', return_value=1)
    @patch('mini_trainer.sampler.dist.get_world_size', return_value=2)
    def test_mock_rank(self, mock_world_size, mock_rank, temp_data_file):
        assert mock_rank() == 1
        mock_rank.return_value = 0
        assert mock_rank() == 0


    @patch('mini_trainer.sampler.dist.get_rank', return_value=0)
    @patch('mini_trainer.sampler.dist.get_world_size', return_value=2)
    def test_padded_samples_on_last_rank(self, mock_world_size, mock_rank, temp_data_file):
        """Test basic data loader creation."""
        expected_batch_size = 9
        expected_leftover_samples = 1

        # first try with an even batch size, so the last step will have 2 samples
        loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=9,
            max_tokens_per_gpu=500,
            seed=42,
        )
        # asserts that loading the model works properly
        assert loader is not None
        assert loader.batch_size == expected_batch_size
        assert loader.sampler.epoch == 0

        # get an iterator and fetch one batch
        loader_it = iter(loader)
        batch = next(loader_it)

        # this will burn up most of the samples
        minibatch = batch[0]
        assert minibatch['num_samples'] == (expected_batch_size // mock_world_size()) + 1

        # now there should be 1 batch left, it will most likely be on the first rank
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch['num_samples'] > 0

        # next, change our rank to the last to see the difference
        mock_rank.return_value = 1
        loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=9,
            max_tokens_per_gpu=500,
            seed=42,
        )

        # asserts that loading the model works properly
        assert loader is not None
        assert loader.batch_size == expected_batch_size
        assert loader.sampler.epoch == 0

        # get an iterator and fetch one batch
        loader_it = iter(loader)
        batch = next(loader_it)

        # this will burn up most of the samples
        minibatch = batch[0]
        assert minibatch['num_samples'] == expected_batch_size // mock_world_size()

        # we expect nothing to be there on the last rank
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch['num_samples'] == 0
        # but we still expect there to be an item
        assert microbatch["labels"] is not None
        assert microbatch["input_ids"] is not None
        assert microbatch["batch_num_loss_counted_tokens"] > 0
        assert microbatch["num_loss_counted_tokens"] == 0

    
    def test_get_data_loader_with_custom_params(self, temp_data_file):
        """Test data loader with custom parameters."""
        dummy_sample = {
            'input_ids': torch.tensor([0, 0]),
            'labels': torch.tensor([-100, -100]),
            'len': 2,
            'num_loss_counted_tokens': 0
        }
        
        loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=8,
            max_tokens_per_gpu=1000,
            seed=123,
            rank=1,
            world_size=4,
            dummy_sample=dummy_sample
        )
        
        assert loader.batch_size == 8
        assert loader.collate_fn.rank == 1
        assert loader.collate_fn.world_size == 4
        assert loader.collate_fn.dummy_sample == dummy_sample
    
    @patch('mini_trainer.sampler.dist.get_rank', return_value=0)
    @patch('mini_trainer.sampler.dist.get_world_size', return_value=2)
    def test_data_loader_iteration(self, mock_world_size, mock_rank, temp_data_file):
        """Test that data loader can be iterated."""
        loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=2,
            max_tokens_per_gpu=500,
            seed=42
        )
        
        # Get an iterator and fetch one batch
        data_iter = iter(loader)
        batch = next(data_iter)
        
        assert isinstance(batch, list)
        # Each element should be a minibatch dictionary
        if len(batch) > 0:
            assert isinstance(batch[0], dict)
            assert 'input_ids' in batch[0]


class TestResetMinibatches:
    """Test suite for the reset_minibatches utility function."""
    
    def test_reset_minibatches_basic(self):
        """Test basic reset functionality."""
        ids, loads = reset_minibatches(4)
        
        assert len(ids) == 4
        assert len(loads) == 4
        assert all(isinstance(lst, list) for lst in ids)
        assert all(len(lst) == 0 for lst in ids)
        assert np.array_equal(loads, np.zeros(4))
    
    def test_reset_minibatches_single_rank(self):
        """Test reset with single rank."""
        ids, loads = reset_minibatches(1)
        
        assert len(ids) == 1
        assert len(loads) == 1
        assert ids[0] == []
        assert loads[0] == 0

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
    
    def test_epoch_boundary_with_uneven_batch_size(self):
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
            assert samples_seen2 == 100  # This reveals the issue!

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
    
    
    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
