"""
Test suite for data loading and sampling components.

Tests the JsonlDataset, InfiniteSampler, and MaxTokensPerRankCollator
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

from mini_trainer.sampler import (
    JsonlDataset,
    InfiniteSampler,
    MaxTokensPerRankCollator,
    get_data_loader,
    mb_collate_fn,
    reset_minibatches
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


class TestInfiniteSampler:
    """Test suite for the InfiniteSampler class."""
    
    def test_sampler_initialization(self):
        """Test sampler initialization."""
        sampler = InfiniteSampler(len_data=100, seed=42)
        assert len(sampler) == 100
        assert sampler.seed == 42
    
    def test_sampler_infinite_iteration(self):
        """Test that sampler provides infinite iteration."""
        sampler = InfiniteSampler(len_data=5, seed=42)
        iterator = iter(sampler)
        
        # Collect indices for multiple epochs
        indices = []
        for _ in range(15):  # 3 epochs worth
            indices.append(next(iterator))
        
        assert len(indices) == 15
        # Check all indices are within range
        assert all(0 <= idx < 5 for idx in indices)
        
        # Check that we have seen all indices
        unique_indices = set(indices[:5])
        assert len(unique_indices) == 5
    
    def test_sampler_deterministic_with_seed(self):
        """Test that same seed produces same sequence."""
        sampler1 = InfiniteSampler(len_data=10, seed=42)
        sampler2 = InfiniteSampler(len_data=10, seed=42)
        
        iter1 = iter(sampler1)
        iter2 = iter(sampler2)
        
        # Compare first 20 indices
        for _ in range(20):
            assert next(iter1) == next(iter2)
    
    def test_sampler_different_seeds(self):
        """Test that different seeds produce different sequences."""
        sampler1 = InfiniteSampler(len_data=10, seed=42)
        sampler2 = InfiniteSampler(len_data=10, seed=43)
        
        iter1 = iter(sampler1)
        iter2 = iter(sampler2)
        
        indices1 = [next(iter1) for _ in range(10)]
        indices2 = [next(iter2) for _ in range(10)]
        
        # Sequences should be different (with high probability)
        assert indices1 != indices2


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
    @patch('mini_trainer.sampler.dist.get_world_size', return_value=2)
    def test_get_data_loader_basic(self, mock_world_size, mock_rank, temp_data_file):
        """Test basic data loader creation."""
        loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=4,
            max_tokens_per_gpu=500,
            seed=42
        )
        
        assert loader is not None
        assert loader.batch_size == 4
        assert loader.num_workers == 4
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
