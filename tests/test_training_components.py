"""
Test suite for training utilities and components.

Tests gradient stepping, batch metrics, model saving, distributed setup,
and other critical training loop components.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call, mock_open
from collections import defaultdict

from mini_trainer.train import take_gradient_step, save_model
from mini_trainer.batch_metrics import BatchMetrics
from mini_trainer.utils import (
    init_distributed_environment,
    check_distributed_is_synchronized,
    log_rank_0,
    setup_logger,
    patch_target_module,
    get_caller
)


class TestTakeGradientStep:
    """Test suite for the gradient step function."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model with parameters."""
        model = MagicMock()
        # Create mock parameters with grad attribute
        param1 = MagicMock()
        param1.grad = torch.randn(10, 10)
        param2 = MagicMock() 
        param2.grad = torch.randn(5, 5)
        model.parameters.return_value = [param1, param2]
        return model
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer."""
        optimizer = MagicMock()
        optimizer.step = MagicMock()
        optimizer.zero_grad = MagicMock()
        return optimizer
    
    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock learning rate scheduler."""
        scheduler = MagicMock()
        scheduler.step = MagicMock()
        scheduler.get_last_lr = MagicMock(return_value=[1e-5])
        return scheduler
    
    @patch('mini_trainer.train.torch.nn.utils.clip_grad_norm_')
    def test_gradient_step_basic(self, mock_clip, mock_model, mock_optimizer, mock_scheduler):
        """Test basic gradient step execution."""
        mock_clip.return_value = torch.tensor(2.5)
        
        grad_norm = take_gradient_step(mock_model, mock_optimizer, mock_scheduler)
        
        # Check gradient clipping was applied
        mock_clip.assert_called_once_with(mock_model.parameters(), 1.0)
        
        # Check optimizer and scheduler steps
        mock_optimizer.step.assert_called_once()
        mock_scheduler.step.assert_called_once()
        mock_optimizer.zero_grad.assert_called_once()
        
        # Check return value
        assert grad_norm.item() == 2.5
    
    @patch('mini_trainer.train.torch.nn.utils.clip_grad_norm_')
    def test_gradient_step_order(self, mock_clip, mock_model, mock_optimizer, mock_scheduler):
        """Test that operations happen in correct order."""
        mock_clip.return_value = torch.tensor(1.0)
        
        # Track call order
        call_order = []
        mock_clip.side_effect = lambda *args, **kwargs: (call_order.append('clip'), torch.tensor(1.0))[1]
        mock_optimizer.step.side_effect = lambda: call_order.append('opt_step')
        mock_scheduler.step.side_effect = lambda: call_order.append('sched_step')
        mock_optimizer.zero_grad.side_effect = lambda: call_order.append('zero_grad')
        
        take_gradient_step(mock_model, mock_optimizer, mock_scheduler)
        
        # Verify correct order
        assert call_order == ['clip', 'opt_step', 'sched_step', 'zero_grad']
    
    @patch('mini_trainer.train.torch.nn.utils.clip_grad_norm_')
    def test_gradient_step_high_grad_norm(self, mock_clip, mock_model, mock_optimizer, mock_scheduler):
        """Test handling of high gradient norm."""
        mock_clip.return_value = torch.tensor(100.0)  # Very high grad norm
        
        grad_norm = take_gradient_step(mock_model, mock_optimizer, mock_scheduler)
        
        assert grad_norm.item() == 100.0
        # Should still proceed with optimization
        mock_optimizer.step.assert_called_once()


class TestBatchMetrics:
    """Test suite for BatchMetrics class."""
    
    def test_batch_metrics_initialization(self):
        """Test BatchMetrics initialization."""
        metrics = BatchMetrics()
        
        assert isinstance(metrics.totals, defaultdict)
        assert isinstance(metrics.minibatch_metrics, defaultdict)
        assert len(metrics.totals) == 0
        assert len(metrics.minibatch_metrics) == 0
    
    def test_accumulate_minibatch_metrics(self):
        """Test accumulating minibatch metrics."""
        metrics = BatchMetrics()
        
        # Accumulate first minibatch
        metrics.accumulate_minibatch_metrics(
            num_loss_counted_tokens=100,
            num_total_tokens=150,
            num_samples=2,
            loss=2.5,
            time_per_minibatch=0.5
        )
        
        assert metrics.minibatch_metrics['num_loss_counted_tokens'] == 100
        assert metrics.minibatch_metrics['num_total_tokens'] == 150
        assert metrics.minibatch_metrics['num_samples'] == 2
        assert metrics.minibatch_metrics['loss'] == 2.5
        assert metrics.minibatch_metrics['time_per_minibatch'] == 0.5
        
        # Accumulate second minibatch
        metrics.accumulate_minibatch_metrics(
            num_loss_counted_tokens=50,
            num_total_tokens=75,
            num_samples=1,
            loss=1.5,
            time_per_minibatch=0.3
        )
        
        assert metrics.minibatch_metrics['num_loss_counted_tokens'] == 150
        assert metrics.minibatch_metrics['num_total_tokens'] == 225
        assert metrics.minibatch_metrics['num_samples'] == 3
        assert metrics.minibatch_metrics['loss'] == 4.0
        assert metrics.minibatch_metrics['time_per_minibatch'] == 0.8
    
    @patch('mini_trainer.batch_metrics.torch.distributed.all_reduce')
    def test_reduce_batch_metrics(self, mock_all_reduce):
        """Test reducing metrics across distributed processes."""
        metrics = BatchMetrics()
        
        # Setup minibatch metrics
        metrics.minibatch_metrics['num_samples'] = 4
        metrics.minibatch_metrics['loss'] = 10.0
        metrics.minibatch_metrics['num_tokens'] = 500
        
        # Mock all_reduce to simulate summing across 2 processes
        def all_reduce_side_effect(tensor, op):
            tensor.mul_(2)  # Simulate 2 processes
            return None
        
        mock_all_reduce.side_effect = all_reduce_side_effect
        
        device = torch.device('cpu')
        metrics.reduce_batch_metrics(device)
        
        # Check that metrics were reduced
        assert metrics.totals['num_samples'] == 8  # 4 * 2
        assert metrics.totals['loss'] == 20.0  # 10.0 * 2
        assert metrics.totals['num_tokens'] == 1000  # 500 * 2
        
        # Check minibatch metrics were cleared
        assert len(metrics.minibatch_metrics) == 0
    
    def test_reset_batch(self):
        """Test resetting batch metrics."""
        metrics = BatchMetrics()
        
        # Add some metrics
        metrics.totals['test'] = 100
        metrics.minibatch_metrics['test'] = 50
        
        metrics.reset_batch()
        
        assert len(metrics.totals) == 0
        assert len(metrics.minibatch_metrics) == 0
    
    def test_accumulate_with_custom_metrics(self):
        """Test accumulating custom metric names."""
        metrics = BatchMetrics()
        
        # Test with numeric values
        metrics.accumulate_minibatch_metrics(
            custom_metric_1=42,
            custom_metric_2=3.14,
            another_metric=100
        )
        
        assert metrics.minibatch_metrics['custom_metric_1'] == 42
        assert metrics.minibatch_metrics['custom_metric_2'] == 3.14
        assert metrics.minibatch_metrics['another_metric'] == 100
        
        # Test accumulation
        metrics.accumulate_minibatch_metrics(
            custom_metric_1=8,
            custom_metric_2=0.86,
            another_metric=50
        )
        
        assert metrics.minibatch_metrics['custom_metric_1'] == 50
        assert abs(metrics.minibatch_metrics['custom_metric_2'] - 4.0) < 1e-6
        assert metrics.minibatch_metrics['another_metric'] == 150


class TestSaveModel:
    """Test suite for model saving functionality."""
    
    @pytest.fixture
    def mock_fsdp_model(self):
        """Create a mock FSDP model."""
        model = MagicMock()
        model.module = MagicMock()
        model.module.config = MagicMock()
        model.module.config.to_json_file = MagicMock()
        model.module.prepare_state_dict_for_save = MagicMock(side_effect=lambda x: x)
        return model
    
    @patch('mini_trainer.train.torch.distributed.get_rank', return_value=0)
    @patch('mini_trainer.train.torch.distributed.barrier')
    @patch('torch.distributed.checkpoint.state_dict.get_model_state_dict')
    @patch('huggingface_hub.split_torch_state_dict_into_shards')
    @patch('safetensors.torch.save_file')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('mini_trainer.train.os.makedirs')
    @patch('mini_trainer.train.log_rank_0')
    def test_save_model_rank_0(self, mock_log, mock_makedirs, mock_tokenizer_cls,
                               mock_save_file, mock_split, mock_get_state_dict,
                               mock_barrier, mock_rank, mock_fsdp_model):
        """Test model saving on rank 0."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.save_pretrained = MagicMock()
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        state_dict = {
            'layer1.weight': torch.randn(10, 10),
            'layer2.weight': torch.randn(5, 5)
        }
        mock_get_state_dict.return_value = state_dict
        
        mock_split_result = MagicMock()
        mock_split_result.filename_to_tensors = {'model.safetensors': ['layer1.weight', 'layer2.weight']}
        mock_split_result.is_sharded = False
        mock_split_result.metadata = {}
        mock_split_result.tensor_to_filename = {}
        mock_split.return_value = mock_split_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_model(
                mock_fsdp_model,
                samples_seen=1000,
                output_dir=temp_dir,
                model_name_or_path="test/model"
            )
        
        # Check directory creation
        mock_makedirs.assert_called()
        
        # Check state dict retrieval
        mock_get_state_dict.assert_called_once()
        
        # Check sharding
        mock_split.assert_called_once()
        
        # Check file saving
        mock_save_file.assert_called()
        
        # Check config and tokenizer saving
        mock_fsdp_model.module.config.to_json_file.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
        
        # Check barrier for synchronization
        mock_barrier.assert_called()
    
    @patch('mini_trainer.train.torch.distributed.get_rank', return_value=1)
    @patch('mini_trainer.utils.get_rank', return_value=1)  # Also patch the utils version
    @patch('mini_trainer.utils.is_initialized', return_value=True)
    @patch('mini_trainer.train.torch.distributed.barrier')
    @patch('torch.distributed.checkpoint.state_dict.get_model_state_dict')
    def test_save_model_non_rank_0(self, mock_get_state_dict, mock_barrier, mock_is_init, mock_utils_rank, mock_rank, mock_fsdp_model):
        """Test that non-rank-0 processes wait at barrier."""
        # Mock state dict to avoid processing MagicMock
        mock_get_state_dict.return_value = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_model(
                mock_fsdp_model,
                samples_seen=1000,
                output_dir=temp_dir,
                model_name_or_path="test/model"
            )
        
        # Non-rank-0 should only call barrier
        mock_barrier.assert_called()
        # Should still get state dict (all ranks do this)
        mock_get_state_dict.assert_called_once()
    
    @patch('mini_trainer.train.torch.distributed.get_rank', return_value=0)
    @patch('mini_trainer.train.torch.distributed.barrier')
    @patch('torch.distributed.checkpoint.state_dict.get_model_state_dict')
    @patch('huggingface_hub.split_torch_state_dict_into_shards')
    @patch('safetensors.torch.save_file')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('builtins.open', new_callable=mock_open)
    @patch('mini_trainer.train.os.makedirs')
    @patch('mini_trainer.train.log_rank_0')
    def test_save_model_sharded(self, mock_log, mock_makedirs, mock_file_open,
                                mock_tokenizer_cls, mock_save_file, mock_split,
                                mock_get_state_dict, mock_barrier, mock_rank, mock_fsdp_model):
        """Test saving sharded model."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        state_dict = {f'layer{i}.weight': torch.randn(100, 100) for i in range(10)}
        mock_get_state_dict.return_value = state_dict
        
        # Simulate sharded model
        mock_split_result = MagicMock()
        mock_split_result.filename_to_tensors = {
            'model-00001.safetensors': ['layer0.weight', 'layer1.weight'],
            'model-00002.safetensors': ['layer2.weight', 'layer3.weight'],
        }
        mock_split_result.is_sharded = True
        mock_split_result.metadata = {'total_size': 1000000}
        mock_split_result.tensor_to_filename = {
            'layer0.weight': 'model-00001.safetensors',
            'layer1.weight': 'model-00001.safetensors',
        }
        mock_split.return_value = mock_split_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_model(
                mock_fsdp_model,
                samples_seen=1000,
                output_dir=temp_dir,
                model_name_or_path="test/model"
            )
        
        # Should save multiple shards
        assert mock_save_file.call_count == 2
        
        # Should save index file for sharded model
        mock_file_open.assert_called()
        written_content = ''.join(call.args[0] for call in mock_file_open().write.call_args_list)
        assert 'metadata' in written_content
        assert 'weight_map' in written_content


class TestDistributedUtils:
    """Test suite for distributed training utilities."""
    
    @patch.dict(os.environ, {'LOCAL_RANK': '0'})
    @patch('mini_trainer.utils.torch.distributed.init_process_group')
    @patch('mini_trainer.utils.torch.cuda.set_device')
    @patch('mini_trainer.utils.check_distributed_is_synchronized')
    @patch('mini_trainer.utils.torch.distributed.barrier')
    @patch('mini_trainer.utils.log_rank_0')
    def test_init_distributed_environment(self, mock_log, mock_barrier, mock_check,
                                         mock_set_device, mock_init_pg):
        """Test distributed environment initialization."""
        init_distributed_environment()
        
        # Check process group initialization
        mock_init_pg.assert_called_once()
        args, kwargs = mock_init_pg.call_args
        assert args[0] == "nccl"
        assert 'timeout' in kwargs
        
        # Check device setting
        mock_set_device.assert_called_once_with(0)
        
        # Check synchronization check
        mock_check.assert_called_once()
        
        # Check barrier
        mock_barrier.assert_called_once()
    
    @patch('mini_trainer.utils.dist.get_rank', return_value=0)
    @patch('mini_trainer.utils.dist.get_world_size', return_value=4)
    @patch('mini_trainer.utils.dist.all_reduce')
    def test_check_distributed_synchronized(self, mock_all_reduce, mock_world_size, mock_rank):
        """Test distributed synchronization check."""
        # Mock successful all_reduce
        def all_reduce_side_effect(tensor, op):
            tensor.fill_(4)  # Simulate 4 processes each adding 1
            return None
        
        mock_all_reduce.side_effect = all_reduce_side_effect
        
        # Should not raise
        check_distributed_is_synchronized()
        
        mock_all_reduce.assert_called_once()
    
    @patch('mini_trainer.utils.dist.get_rank', return_value=0)
    @patch('mini_trainer.utils.dist.get_world_size', return_value=4)
    @patch('mini_trainer.utils.dist.all_reduce')
    def test_check_distributed_not_synchronized(self, mock_all_reduce, mock_world_size, mock_rank):
        """Test distributed synchronization check failure."""
        # Mock failed all_reduce
        def all_reduce_side_effect(tensor, op):
            tensor.fill_(3)  # Wrong value
            return None
        
        mock_all_reduce.side_effect = all_reduce_side_effect
        
        with pytest.raises(AssertionError, match="distributed check failed"):
            check_distributed_is_synchronized()


class TestPatchTargetModule:
    """
    Test suite for module patching utility.
    This is very important to validate, because we use it to patch the liger and transformers loss functions.
    """
    
    def test_patch_target_module(self):
        """Test patching a module attribute."""
        # Create a mock module
        import types
        test_module = types.ModuleType('test_module')
        test_module.original_function = lambda x: x * 2
        
        # Add to sys.modules temporarily
        import sys
        sys.modules['test_module'] = test_module
        
        try:
            # Define replacement function
            def replacement_function(x):
                return x * 3
            
            # Patch the module
            patch_target_module('test_module.original_function', replacement_function)
            
            # Verify patch worked
            assert test_module.original_function(5) == 15  # 5 * 3
            
        finally:
            # Cleanup
            del sys.modules['test_module']
    
    def test_patch_target_module_invalid(self):
        """Test patching with invalid target."""
        with pytest.raises(AssertionError, match="must have an object to patch"):
            patch_target_module('invalid', lambda: None)


class TestTrainingIntegration:
    """Integration tests for training components."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradient_accumulation_simulation(self):
        """Simulate gradient accumulation with batch metrics."""
        metrics = BatchMetrics()
        
        # Simulate 3 gradient accumulation steps
        for i in range(3):
            metrics.accumulate_minibatch_metrics(
                num_samples=2,
                loss=1.5 * (i + 1),
                num_loss_counted_tokens=100,
                time_per_minibatch=0.1
            )
        
        # Check accumulated values
        assert metrics.minibatch_metrics['num_samples'] == 6
        assert metrics.minibatch_metrics['loss'] == 9.0  # 1.5 + 3.0 + 4.5
        assert metrics.minibatch_metrics['num_loss_counted_tokens'] == 300
        assert np.isclose(metrics.minibatch_metrics['time_per_minibatch'], 0.3)
    
    def test_mock_training_step(self):
        """Test a mock training step with all components."""
        # Create mock components
        model = MagicMock()
        model.parameters.return_value = [MagicMock()]
        
        optimizer = MagicMock()
        scheduler = MagicMock()
        scheduler.get_last_lr.return_value = [1e-5]
        
        metrics = BatchMetrics()
        
        # Simulate minibatch processing
        metrics.accumulate_minibatch_metrics(
            num_samples=4,
            loss=2.5,
            num_loss_counted_tokens=200
        )
        
        # Take gradient step
        with patch('mini_trainer.train.torch.nn.utils.clip_grad_norm_') as mock_clip:
            mock_clip.return_value = torch.tensor(1.0)
            grad_norm = take_gradient_step(model, optimizer, scheduler)
        
        assert grad_norm.item() == 1.0
        optimizer.step.assert_called_once()
        scheduler.step.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
