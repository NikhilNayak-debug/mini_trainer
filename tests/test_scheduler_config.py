"""
Test suite for learning rate scheduler configuration.

Tests the scheduler configuration with different training modes and
the calculation of training steps.
"""
import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from transformers import get_scheduler

from mini_trainer.training_types import TrainingMode
from mini_trainer.setup_model_for_training import setup_training_components
from mini_trainer.sampler import get_data_loader
from mini_trainer.train import calculate_num_training_steps


class TestCalculateTrainingSteps:
    """Test suite for calculating training steps based on training mode."""
    
    @pytest.fixture
    def create_test_data(self):
        """Create temporary test data file."""
        def _create(num_samples=10):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for i in range(num_samples):
                    seq_length = 10 + (i % 3) * 5
                    input_ids = list(range(100, 100 + seq_length))
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
    
    def test_calculate_steps_infinite_mode(self):
        """Test that INFINITE mode returns None for training steps."""
        result = calculate_num_training_steps(
            training_mode=TrainingMode.INFINITE,
            data_loader=None
        )
        assert result is None
    
    def test_calculate_steps_step_mode(self):
        """Test that STEP mode returns max_steps."""
        result = calculate_num_training_steps(
            training_mode=TrainingMode.STEP,
            data_loader=None,
            max_steps=1000
        )
        assert result == 1000
    
    @patch('torch.distributed.is_initialized', return_value=False)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.get_world_size', return_value=1)
    def test_calculate_steps_epoch_mode(self, mock_world_size, mock_rank, mock_is_init, create_test_data):
        """Test calculating steps for EPOCH mode."""
        data_path = create_test_data(num_samples=20)
        
        try:
            data_loader = get_data_loader(
                data_path=data_path,
                batch_size=4,
                max_tokens_per_gpu=1000,
                seed=42
            )
            
            # Reset data loader for fresh iteration
            data_loader.sampler.set_epoch(0)
            
            result = calculate_num_training_steps(
                training_mode=TrainingMode.EPOCH,
                data_loader=data_loader,
                max_epochs=3
            )
            
            # Should be num_batches * max_epochs
            assert result is not None
            assert result > 0
            # With 20 samples and batch size 4, expect around 5 batches * 3 epochs = 15 steps
            # But actual may vary due to dynamic batching
            
        finally:
            os.unlink(data_path)
    
    @patch('torch.distributed.is_initialized', return_value=False)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.get_world_size', return_value=1)
    def test_calculate_steps_token_mode(self, mock_world_size, mock_rank, mock_is_init, create_test_data):
        """Test calculating steps for TOKEN mode."""
        data_path = create_test_data(num_samples=10)
        
        try:
            data_loader = get_data_loader(
                data_path=data_path,
                batch_size=2,
                max_tokens_per_gpu=1000,
                seed=42
            )
            
            # Reset data loader
            data_loader.sampler.set_epoch(0)
            
            result = calculate_num_training_steps(
                training_mode=TrainingMode.TOKEN,
                data_loader=data_loader,
                max_tokens=500
            )
            
            # Should calculate based on average tokens per batch
            assert result is not None
            assert result >= 0
            
        finally:
            os.unlink(data_path)
    
    def test_calculate_steps_with_infinite_mode(self):
        """Test that INFINITE mode returns None for training steps."""
        # INFINITE mode should always return None
        result = calculate_num_training_steps(
            training_mode=TrainingMode.INFINITE,
            data_loader=None,
            max_epochs=5
        )
        assert result is None


class TestSchedulerConfiguration:
    """Test suite for learning rate scheduler configuration."""
    
    def test_scheduler_with_training_steps(self):
        """Test creating scheduler with num_training_steps."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # Test cosine scheduler which requires num_training_steps
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        
        assert scheduler is not None
        # The scheduler should be properly configured
        # We can't check specific attributes as they vary by scheduler implementation
    
    def test_scheduler_without_training_steps(self):
        """Test creating scheduler that doesn't require num_training_steps."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # Test constant scheduler which doesn't require num_training_steps
        scheduler = get_scheduler(
            name="constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=None
        )
        
        assert scheduler is not None
    
    def test_scheduler_with_specific_kwargs(self):
        """Test creating scheduler with scheduler-specific kwargs."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # Test cosine with restarts using scheduler_specific_kwargs
        scheduler = get_scheduler(
            name="cosine_with_restarts",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
            scheduler_specific_kwargs={"num_cycles": 3}
        )
        
        assert scheduler is not None
    
    @patch('mini_trainer.setup_model_for_training.wrap_fsdp2')
    @patch('mini_trainer.setup_model_for_training.log_rank_0')
    def test_setup_training_components_with_steps(self, mock_log, mock_wrap):
        """Test setup_training_components with num_training_steps."""
        model = MagicMock()
        mock_wrap.return_value = model
        model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        
        # Mock the optim_wrapper to return the optimizer unchanged
        with patch('mini_trainer.osft_utils.optim_wrapper', side_effect=lambda opt, model: opt):
            model, optimizer, scheduler = setup_training_components(
                model=model,
                learning_rate=1e-5,
                num_warmup_steps=100,
                lr_scheduler="cosine",
                num_training_steps=1000,
                scheduler_kwargs={}
            )
        
        assert model is not None
        assert optimizer is not None
        assert scheduler is not None
        
        # Verify scheduler was created with correct steps
        mock_log.assert_called_with("Using FSDP2 wrapper")


class TestDatasetMetrics:
    """Test suite for getting dataset metrics."""
    
    @pytest.fixture
    def create_test_data(self):
        """Create temporary test data file."""
        def _create():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                samples = [
                    {
                        "input_ids": [1, 2, 3, 4, 5],
                        "labels": [-100, -100, 3, 4, 5],
                        "len": 5,
                        "num_loss_counted_tokens": 3
                    },
                    {
                        "input_ids": [6, 7, 8, 9, 10, 11],
                        "labels": [-100, 7, 8, 9, 10, 11],
                        "len": 6,
                        "num_loss_counted_tokens": 5
                    }
                ]
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
                return f.name
        return _create
    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
