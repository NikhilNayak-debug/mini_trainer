"""
Test suite for model initialization and training setup.

Tests the setup_model, setup_training_components, and related functions
to ensure correct model initialization, FSDP wrapping, and optimizer setup.

TODO: This file needs to be combined with `test_integration_small_models.py`
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call
from transformers import AutoConfig

from mini_trainer.setup_model_for_training import (
    wrap_fsdp2,
    align_model_and_tokenizer,
    setup_model,
    setup_training_components,
)


class TestAlignModelAndTokenizer:
    """Test suite for model and tokenizer alignment."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.vocab_size = 32000
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.resize_token_embeddings = MagicMock()
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=32000)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        return tokenizer
    
    def test_align_matching_vocab_size(self, mock_model, mock_tokenizer):
        """Test alignment when vocab sizes match."""
        result = align_model_and_tokenizer(mock_model, mock_tokenizer)
        
        assert result == mock_model
        mock_model.resize_token_embeddings.assert_not_called()
    
    def test_align_resize_vocab(self, mock_model, mock_tokenizer):
        """Test vocab resizing when tokenizer has more tokens."""
        mock_tokenizer.__len__ = MagicMock(return_value=32005)
        
        with patch('mini_trainer.setup_model_for_training.log_rank_0') as mock_log:
            result = align_model_and_tokenizer(mock_model, mock_tokenizer)
        
        # Should resize to next multiple of 8
        mock_model.resize_token_embeddings.assert_called_once_with(32008)
    
    def test_align_fix_special_tokens(self, mock_model, mock_tokenizer):
        """Test fixing mismatched special tokens."""
        mock_model.config.pad_token_id = 999
        mock_model.config.bos_token_id = 998
        mock_model.config.eos_token_id = 997
        
        with patch('mini_trainer.setup_model_for_training.log_rank_0') as mock_log:
            result = align_model_and_tokenizer(mock_model, mock_tokenizer)
        
        # Special tokens should be aligned
        assert mock_model.config.pad_token_id == 0
        assert mock_model.config.bos_token_id == 1
        assert mock_model.config.eos_token_id == 2
        
        # Should have logged warnings
        assert mock_log.call_count == 3
    
    def test_align_none_special_tokens(self, mock_model, mock_tokenizer):
        """Test handling of None special tokens."""
        mock_model.config.pad_token_id = None
        mock_tokenizer.pad_token_id = None
        
        with patch('mini_trainer.setup_model_for_training.log_rank_0') as mock_log:
            result = align_model_and_tokenizer(mock_model, mock_tokenizer)
        
        # None values should be left alone
        assert mock_model.config.pad_token_id is None
        mock_log.assert_not_called()


class TestWrapFSDP2:
    """Test suite for FSDP2 wrapping."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model with transformer layers."""
        model = MagicMock()
        model.device = torch.device('cpu')
        model.config = MagicMock()
        model.config.use_cache = True
        
        # Mock transformer layers
        layers = [MagicMock() for _ in range(4)]
        model.model = MagicMock()
        model.model.layers = layers
        
        return model
    
    @patch('mini_trainer.setup_model_for_training.dist.get_rank', return_value=0)
    @patch('mini_trainer.setup_model_for_training.dist.get_world_size', return_value=2)
    @patch('mini_trainer.setup_model_for_training.init_device_mesh')
    @patch('mini_trainer.setup_model_for_training.fully_shard')
    @patch('mini_trainer.setup_model_for_training.ptd_checkpoint_wrapper')
    def test_wrap_fsdp2_basic(self, mock_checkpoint, mock_fully_shard, 
                               mock_init_mesh, mock_world_size, mock_rank, mock_model):
        """Test basic FSDP2 wrapping."""
        mock_mesh = MagicMock()
        mock_init_mesh.return_value = mock_mesh
        mock_checkpoint.side_effect = lambda x, **kwargs: x
        
        result = wrap_fsdp2(mock_model)
        
        # Should disable cache
        assert mock_model.config.use_cache == False
        
        # Should wrap each layer with checkpoint wrapper
        assert mock_checkpoint.call_count == 4
        
        # Should create device mesh
        mock_init_mesh.assert_called_once_with("cuda", [2], mesh_dim_names=["fsdp"])
        
        # Should fully shard each layer and the model
        assert mock_fully_shard.call_count == 5  # 4 layers + 1 model
    
    @patch('mini_trainer.setup_model_for_training.dist.get_rank', return_value=1)
    @patch('mini_trainer.setup_model_for_training.dist.get_world_size', return_value=4)
    @patch('mini_trainer.setup_model_for_training.init_device_mesh')
    @patch('mini_trainer.setup_model_for_training.fully_shard')
    @patch('mini_trainer.setup_model_for_training.ptd_checkpoint_wrapper')
    def test_wrap_fsdp2_multi_gpu(self, mock_checkpoint, mock_fully_shard,
                                   mock_init_mesh, mock_world_size, mock_rank, mock_model):
        """Test FSDP2 wrapping with multiple GPUs."""
        mock_mesh = MagicMock()
        mock_init_mesh.return_value = mock_mesh
        mock_checkpoint.side_effect = lambda x, **kwargs: x
        
        # Model should be moved to correct GPU
        mock_model.device = torch.device('cpu')
        
        result = wrap_fsdp2(mock_model)
        
        # Should move to correct GPU
        mock_model.to.assert_called_once()
        to_call = mock_model.to.call_args[0][0]
        assert to_call.type == 'cuda'
        assert to_call.index == 1
    
    @patch('mini_trainer.setup_model_for_training.dist.get_rank', return_value=0)
    def test_wrap_fsdp2_no_layers_found(self, mock_rank, mock_model):
        """Test error handling when transformer layers not found."""
        mock_model.model = None  # No model attribute
        
        with pytest.raises(ValueError, match="Cannot find transformer block container"):
            wrap_fsdp2(mock_model)


class TestSetupModel:
    """Test suite for model setup."""
    
    @patch('mini_trainer.setup_model_for_training.AutoTokenizer.from_pretrained')
    @patch('mini_trainer.setup_model_for_training.AutoModelForCausalLM.from_pretrained')
    @patch('mini_trainer.setup_model_for_training.align_model_and_tokenizer')
    def test_setup_model_standard(self, mock_align, mock_model_cls, mock_tokenizer_cls):
        """Test standard model setup without special features."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "LlamaForCausalLM"
        mock_model_cls.return_value = mock_model
        mock_align.return_value = mock_model
        
        with patch('mini_trainer.setup_model_for_training.log_rank_0'):
            result = setup_model(
                model_name_or_path="meta-llama/Llama-2-7b",
                use_liger_kernels=False,
                osft=False,
                rank=0
            )
        
        assert result == mock_model
        mock_model_cls.assert_called_once()
        mock_align.assert_called_once_with(mock_model, mock_tokenizer)
    
    
class TestSetupTrainingComponents:
    """Test suite for training components setup."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.parameters = MagicMock(return_value=[MagicMock()])
        return model
    
    @patch('mini_trainer.setup_model_for_training.wrap_fsdp2')
    @patch('transformers.get_scheduler')
    @patch('mini_trainer.setup_model_for_training.torch.optim.AdamW')
    @patch('mini_trainer.osft_utils.optim_wrapper')
    @patch('mini_trainer.setup_model_for_training.log_rank_0')
    def test_setup_training_components_basic(self, mock_log, mock_optim_wrapper, 
                                            mock_adamw, mock_scheduler, mock_wrap, mock_model):
        """Test basic training components setup."""
        mock_wrapped_model = MagicMock()
        mock_wrap.return_value = mock_wrapped_model
        
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer
        
        mock_wrapped_optimizer = MagicMock()
        mock_optim_wrapper.return_value = mock_wrapped_optimizer
        
        mock_lr_scheduler = MagicMock()
        mock_lr_scheduler.get_last_lr = MagicMock(return_value=[1e-5])
        mock_scheduler.return_value = mock_lr_scheduler
        
        model, optimizer, lr_scheduler = setup_training_components(
            mock_model,
            learning_rate=1e-5,
            num_warmup_steps=10,
            lr_scheduler="constant_with_warmup"
        )
        
        assert model == mock_wrapped_model
        assert optimizer == mock_wrapped_optimizer
        assert lr_scheduler == mock_lr_scheduler
        
        # Check FSDP2 wrapping
        mock_wrap.assert_called_once_with(mock_model)
        
        # Check optimizer creation
        mock_adamw.assert_called_once_with(
            mock_wrapped_model.parameters(),
            lr=1e-5,
            betas=(0.9, 0.95),
            weight_decay=0.0
        )
        
        # Check optimizer wrapping
        mock_optim_wrapper.assert_called_once_with(mock_optimizer, mock_wrapped_model)
        
        # Check scheduler creation with new parameters
        mock_scheduler.assert_called_once_with(
            name="constant_with_warmup",
            optimizer=mock_wrapped_optimizer,
            num_warmup_steps=10,
            num_training_steps=None,
            scheduler_specific_kwargs={}
        )
        
        # Check scheduler properties and step
        assert lr_scheduler.split_batches == True
        lr_scheduler.step.assert_called_once()
    
    @patch('mini_trainer.setup_model_for_training.wrap_fsdp2')
    @patch('transformers.get_scheduler')
    @patch('mini_trainer.setup_model_for_training.torch.optim.AdamW')
    @patch('mini_trainer.osft_utils.optim_wrapper')
    @patch('mini_trainer.setup_model_for_training.log_rank_0')
    def test_setup_training_components_different_scheduler(self, mock_log, mock_optim_wrapper,
                                                          mock_adamw, mock_scheduler, mock_wrap, mock_model):
        """Test setup with different scheduler type."""
        mock_wrap.return_value = mock_model
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer
        mock_optim_wrapper.return_value = mock_optimizer
        
        mock_lr_scheduler = MagicMock()
        mock_scheduler.return_value = mock_lr_scheduler
        
        model, optimizer, lr_scheduler = setup_training_components(
            mock_model,
            learning_rate=5e-6,
            num_warmup_steps=100,
            lr_scheduler="cosine"
        )
        
        mock_scheduler.assert_called_once_with(
            name="cosine",
            optimizer=mock_optimizer,
            num_warmup_steps=100,
            num_training_steps=None,
            scheduler_specific_kwargs={}
        )


class TestIntegration:
    """Integration tests for model initialization and training setup."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch('mini_trainer.setup_model_for_training.dist.is_initialized', return_value=False)
    @patch('mini_trainer.setup_model_for_training.AutoTokenizer.from_pretrained')
    def test_model_device_placement(self, mock_tokenizer, mock_dist_init):
        """Test that model is correctly placed on GPU."""
        # This test would require actual model loading
        # Skipping for unit tests to avoid downloading models
        pytest.skip("Integration test requiring actual model download")
    
    def test_end_to_end_mock(self):
        """Test end-to-end flow with mocks."""
        with patch('mini_trainer.setup_model_for_training.AutoTokenizer.from_pretrained') as mock_tok:
            with patch('mini_trainer.setup_model_for_training.AutoModelForCausalLM.from_pretrained') as mock_model_cls:
                with patch('mini_trainer.setup_model_for_training.wrap_fsdp2') as mock_wrap:
                    with patch('mini_trainer.setup_model_for_training.log_rank_0'):
                        mock_tokenizer = MagicMock()
                        mock_tokenizer.__len__ = MagicMock(return_value=32000)  # Set proper length
                        mock_tok.return_value = mock_tokenizer
                        
                        mock_model = MagicMock()
                        mock_model.__class__.__name__ = "LlamaForCausalLM"
                        mock_model.config = MagicMock()
                        mock_model.config.vocab_size = 32000
                        mock_model.parameters = MagicMock(return_value=[MagicMock()])
                        mock_model_cls.return_value = mock_model
                        
                        mock_wrap.return_value = mock_model
                        
                        # Setup model
                        model = setup_model(
                            model_name_or_path="test/model",
                            use_liger_kernels=False,
                            osft=False,
                            rank=0
                        )
                        
                        # Setup training components
                        with patch('mini_trainer.setup_model_for_training.torch.optim.AdamW') as mock_adamw:
                            with patch('transformers.get_scheduler') as mock_sched:
                                with patch('mini_trainer.osft_utils.optim_wrapper') as mock_opt_wrap:
                                    mock_optimizer = MagicMock()
                                    mock_adamw.return_value = mock_optimizer
                                    mock_opt_wrap.return_value = mock_optimizer
                                    
                                    mock_scheduler = MagicMock()
                                    mock_sched.return_value = mock_scheduler
                                    
                                    model, optimizer, scheduler = setup_training_components(
                                        model,
                                        learning_rate=1e-5,
                                        num_warmup_steps=10,
                                        lr_scheduler="constant"
                                    )
                                    
                                    assert model is not None
                                    assert optimizer is not None
                                    assert scheduler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
