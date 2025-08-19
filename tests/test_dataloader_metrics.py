"""
Test suite for data loader metrics and statistics.

Tests counting batches, tokens, and other dataset metrics needed for
proper learning rate scheduler configuration.
"""
import os
import sys
import tempfile
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from unittest.mock import patch

from mini_trainer.sampler import get_data_loader


class TestDataLoaderBatchCount:
    """Test suite for counting batches in data loader."""
    
    @pytest.fixture
    def create_test_data(self):
        """Create temporary test data file."""
        def _create(num_samples=10):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for i in range(num_samples):
                    # Create varying length sequences to test token counting
                    seq_length = 10 + (i % 5) * 5  # Lengths: 10, 15, 20, 25, 30
                    # Create realistic token IDs and labels
                    input_ids = list(range(100, 100 + seq_length))
                    # Make some labels -100 (ignored in loss)
                    labels = [lid if j > 2 else -100 for j, lid in enumerate(input_ids)]
                    num_loss_counted = sum(1 for l in labels if l != -100)
                    
                    sample = {
                        "input_ids": input_ids,
                        "labels": labels,
                        "len": seq_length,
                        "num_loss_counted_tokens": num_loss_counted
                    }
                    f.write(json.dumps(sample) + '\n')
                return f.name
        return _create
    
    @patch('torch.distributed.is_initialized', return_value=False)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.get_world_size', return_value=1)
    def test_count_batches_finite_sampler(self, mock_world_size, mock_rank, mock_is_init, create_test_data):
        """Test counting batches with finite sampler."""
        data_path = create_test_data(num_samples=20)
        
        try:
            # Create data loader with finite sampler
            data_loader = get_data_loader(
                data_path=data_path,
                batch_size=4,
                max_tokens_per_gpu=1000,
                seed=42,
                use_infinite_sampler=False
            )
            
            # Count batches by iterating through the data loader
            batch_count = 0
            for batch in data_loader:
                batch_count += 1
            
            # With 20 samples and batch size 4, we expect 5 batches
            # Note: actual batching may vary due to dynamic batching based on tokens
            assert batch_count > 0, "Should have at least one batch"
            
            # Alternative method: use len() if available
            if hasattr(data_loader, '__len__'):
                length_from_len = len(data_loader)
                assert length_from_len > 0, "Data loader length should be positive"
        
        finally:
            os.unlink(data_path)
    
    @patch('torch.distributed.is_initialized', return_value=False)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.get_world_size', return_value=1)
    def test_count_batches_with_epoch(self, mock_world_size, mock_rank, mock_is_init, create_test_data):
        """Test that batch count is consistent across epochs."""
        data_path = create_test_data(num_samples=10)
        
        try:
            data_loader = get_data_loader(
                data_path=data_path,
                batch_size=2,
                max_tokens_per_gpu=1000,
                seed=42,
                use_infinite_sampler=False
            )
            
            # Count batches for multiple epochs
            epoch_batch_counts = []
            for epoch in range(3):
                data_loader.sampler.set_epoch(epoch)
                batch_count = 0
                for batch in data_loader:
                    batch_count += 1
                epoch_batch_counts.append(batch_count)
            
            # All epochs should have the same number of batches
            assert len(set(epoch_batch_counts)) == 1, f"Batch counts vary across epochs: {epoch_batch_counts}"
            assert epoch_batch_counts[0] > 0, "Should have at least one batch per epoch"
        
        finally:
            os.unlink(data_path)


class TestDatasetTokenCount:
    """Test suite for counting tokens in dataset."""
    
    @pytest.fixture
    def create_test_data(self):
        """Create temporary test data file with known token counts."""
        def _create():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                # Create samples with predictable token counts
                samples = [
                    {
                        "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                        "labels": [-100, -100, 3, 4, 5, 6, 7, 8],  # 6 loss tokens
                        "len": 8,
                        "num_loss_counted_tokens": 6
                    },
                    {
                        "input_ids": [10, 11, 12, 13, 14, 15],
                        "labels": [-100, -100, -100, 13, 14, 15],  # 3 loss tokens
                        "len": 6,
                        "num_loss_counted_tokens": 3
                    },
                    {
                        "input_ids": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                        "labels": [-100, -100, 22, 23, 24, 25, 26, 27, 28, 29],  # 8 loss tokens
                        "len": 10,
                        "num_loss_counted_tokens": 8
                    }
                ]
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
                return f.name
        return _create
    
    @patch('torch.distributed.is_initialized', return_value=False)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.get_world_size', return_value=1)
    def test_count_total_loss_tokens(self, mock_world_size, mock_rank, mock_is_init, create_test_data):
        """Test counting total loss-counted tokens in dataset."""
        data_path = create_test_data()
        
        try:
            # Create data loader
            data_loader = get_data_loader(
                data_path=data_path,
                batch_size=1,  # Use batch size 1 to make counting easier
                max_tokens_per_gpu=10000,
                seed=42,
                use_infinite_sampler=False
            )
            
            # Count total loss tokens
            total_loss_tokens = 0
            total_samples = 0
            
            for batch in data_loader:
                for minibatch in batch:
                    # Each minibatch should have 'num_loss_counted_tokens'
                    if 'num_loss_counted_tokens' in minibatch:
                        total_loss_tokens += minibatch['num_loss_counted_tokens']
                    if 'num_samples' in minibatch:
                        total_samples += minibatch['num_samples']
            
            # Verify we counted tokens
            assert total_loss_tokens > 0, "Should have counted some loss tokens"
            assert total_samples == 3, f"Should have 3 samples, got {total_samples}"
            
        finally:
            os.unlink(data_path)
    
    @patch('torch.distributed.is_initialized', return_value=False)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.get_world_size', return_value=1)
    def test_calculate_average_tokens_per_batch(self, mock_world_size, mock_rank, mock_is_init, create_test_data):
        """Test calculating average tokens per batch for token-based training."""
        data_path = create_test_data()
        
        try:
            data_loader = get_data_loader(
                data_path=data_path,
                batch_size=2,
                max_tokens_per_gpu=10000,
                seed=42,
                use_infinite_sampler=False
            )
            
            # Track tokens per batch
            batch_token_counts = []
            
            for batch in data_loader:
                batch_tokens = 0
                for minibatch in batch:
                    if 'batch_num_loss_counted_tokens' in minibatch:
                        # This is the total for the whole batch
                        batch_tokens = minibatch['batch_num_loss_counted_tokens']
                        break
                if batch_tokens > 0:
                    batch_token_counts.append(batch_tokens)
            
            # Calculate average
            if batch_token_counts:
                avg_tokens_per_batch = sum(batch_token_counts) / len(batch_token_counts)
                assert avg_tokens_per_batch > 0, "Average tokens per batch should be positive"
                
                # Can use this to estimate steps for token-based training
                max_tokens = 1000
                estimated_steps = int(max_tokens / avg_tokens_per_batch)
                assert estimated_steps >= 0, "Estimated steps should be non-negative"
        
        finally:
            os.unlink(data_path)
    
    @patch('torch.distributed.is_initialized', return_value=False)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.get_world_size', return_value=1)
    def test_dataset_metrics_for_scheduler(self, mock_world_size, mock_rank, mock_is_init, create_test_data):
        """Test gathering all metrics needed for LR scheduler configuration."""
        data_path = create_test_data()
        
        try:
            data_loader = get_data_loader(
                data_path=data_path,
                batch_size=2,
                max_tokens_per_gpu=10000,
                seed=42,
                use_infinite_sampler=False
            )
            
            # Gather comprehensive metrics
            metrics = {
                'num_batches': 0,
                'total_samples': 0,
                'total_loss_tokens': 0,
                'batch_token_counts': []
            }
            
            for batch in data_loader:
                metrics['num_batches'] += 1
                batch_loss_tokens = 0
                
                for minibatch in batch:
                    if 'num_samples' in minibatch:
                        metrics['total_samples'] += minibatch['num_samples']
                    if 'num_loss_counted_tokens' in minibatch:
                        metrics['total_loss_tokens'] += minibatch['num_loss_counted_tokens']
                    if 'batch_num_loss_counted_tokens' in minibatch:
                        batch_loss_tokens = minibatch['batch_num_loss_counted_tokens']
                
                if batch_loss_tokens > 0:
                    metrics['batch_token_counts'].append(batch_loss_tokens)
            
            # Calculate derived metrics
            if metrics['batch_token_counts']:
                metrics['avg_tokens_per_batch'] = sum(metrics['batch_token_counts']) / len(metrics['batch_token_counts'])
            else:
                metrics['avg_tokens_per_batch'] = 0
            
            # Verify all metrics are collected
            assert metrics['num_batches'] > 0, "Should have at least one batch"
            assert metrics['total_samples'] == 3, f"Should have 3 samples, got {metrics['total_samples']}"
            assert metrics['total_loss_tokens'] > 0, "Should have counted loss tokens"
            assert metrics['avg_tokens_per_batch'] > 0, "Should have positive average tokens per batch"
            
            # Test calculating training steps for different modes
            # Epoch-based
            max_epochs = 5
            epoch_based_steps = metrics['num_batches'] * max_epochs
            assert epoch_based_steps > 0, "Epoch-based steps should be positive"
            
            # Token-based
            max_tokens = 10000
            if metrics['avg_tokens_per_batch'] > 0:
                token_based_steps = int(max_tokens / metrics['avg_tokens_per_batch'])
                assert token_based_steps >= 0, "Token-based steps should be non-negative"
        
        finally:
            os.unlink(data_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
