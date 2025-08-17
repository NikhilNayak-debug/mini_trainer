"""
Integration tests using actual small language models from Transformers.

These tests use tiny model configurations to test real functionality
without requiring large amounts of memory or computation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import torch
import torch.nn as nn
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from setup_model_for_training import align_model_and_tokenizer, setup_model, setup_training_components
from sampler import JsonlDataset, InfiniteSampler, MaxTokensPerRankCollator, get_data_loader
from batch_metrics import BatchMetrics
from train import take_gradient_step


# TODO: add tests to validate our codebase works with these models 
def create_tiny_llama_model():
    """Create a tiny Llama model with ~50k parameters."""
    config = LlamaConfig(
        vocab_size=500,  # Very small vocabulary
        hidden_size=32,   # Tiny hidden size
        intermediate_size=64,  # Small FFN
        num_hidden_layers=2,  # Only 2 layers
        num_attention_heads=2,  # Few attention heads
        num_key_value_heads=1,  # GQA
        max_position_embeddings=64,  # Short sequences
        rope_theta=10000.0,
        hidden_act="silu",
    )
    model = LlamaForCausalLM(config)
    return model, config


def create_tiny_mistral_model():
    """Create a tiny Mistral model with ~50k parameters."""
    config = MistralConfig(
        vocab_size=500,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        sliding_window=32,
    )
    model = MistralForCausalLM(config)
    return model, config


def create_tiny_qwen2_model():
    """Create a tiny Qwen2 model with ~50k parameters."""
    config = Qwen2Config(
        vocab_size=500,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config)
    return model, config


def create_test_data_file(num_samples=10, max_length=50):
    """Create a temporary JSONL file with test data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(num_samples):
            length = min(max_length, 10 + i * 3)
            sample = {
                "input_ids": list(range(length)),
                "labels": list(range(length)),
                "len": length,
                "num_loss_counted_tokens": length - 5  # Some tokens not counted
            }
            json.dump(sample, f)
            f.write('\n')
        return f.name

class TestModelInitialization:
    """Test model initialization with tiny models.
    
    TODO: needs more rigorous testing of the setup functions
    """
    def test_model_tokenizer_alignment(self):
        """Test aligning a tiny model with a tokenizer."""
        model, config = create_tiny_llama_model()
        
        # Set initial config values that differ from tokenizer to test alignment
        model.config.pad_token_id = 999  # Different from tokenizer
        model.config.bos_token_id = 998  # Different from tokenizer
        model.config.eos_token_id = 997  # Different from tokenizer
        
        # Create a mock tokenizer
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=500)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        
        with patch('setup_model_for_training.log_rank_0'):  # Mock logging
            aligned_model = align_model_and_tokenizer(model, tokenizer)
        
        # Check alignment happened - model should now match tokenizer
        assert aligned_model.config.pad_token_id == 0
        assert aligned_model.config.bos_token_id == 1
        assert aligned_model.config.eos_token_id == 2
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch('setup_model_for_training.dist.get_rank', return_value=0)
    @patch('setup_model_for_training.dist.get_world_size', return_value=1)
    @patch('setup_model_for_training.dist.is_initialized', return_value=False)
    def test_wrap_tiny_model_fsdp(self, mock_dist_init, mock_world_size, mock_rank):
        """Test FSDP wrapping with a tiny model."""
        from setup_model_for_training import wrap_fsdp2
        
        model, config = create_tiny_llama_model()
        model = model.cuda()
        
        # Wrap with FSDP2
        with patch('setup_model_for_training.init_device_mesh') as mock_mesh:
            with patch('setup_model_for_training.fully_shard') as mock_shard:
                mock_mesh.return_value = MagicMock()
                mock_shard.side_effect = lambda x, **kwargs: x
                
                wrapped_model = wrap_fsdp2(model)
                
                assert wrapped_model is not None
                # Check that sharding was attempted
                # TODO: make sure transformer blocks were also wrapped
                assert mock_shard.called
    
    def test_training_components_setup_with_tiny_model(self):
        """Test setting up training components with a tiny model."""
        model, config = create_tiny_llama_model()
        
        # TODO: ensure proper functions were also called
        with patch('setup_model_for_training.wrap_fsdp2') as mock_wrap:
            with patch('transformers.get_scheduler') as mock_sched_fn:
                with patch('svd_utils.optim_wrapper') as mock_opt_wrap:
                    with patch('setup_model_for_training.log_rank_0'):
                        mock_wrap.return_value = model
                        mock_opt_wrap.side_effect = lambda opt, m: opt
                        
                        # Create mock scheduler
                        mock_scheduler = MagicMock()
                        mock_scheduler.split_batches = False
                        mock_scheduler.step = MagicMock()
                        mock_sched_fn.return_value = mock_scheduler
                        
                        model, optimizer, scheduler = setup_training_components(
                            model,
                            learning_rate=1e-4,
                            num_warmup_steps=10,
                            lr_scheduler="constant"
                        )
                        
                        assert model is not None
                        assert optimizer is not None
                        assert scheduler is not None


class TestSVDModelInitialization:
    """Test SVD/orthogonal subspace learning with tiny models."""
    
    @pytest.mark.skipif(True, reason="SVD initialization requires significant setup")
    def test_svd_model_creation(self):
        """Test creating an SVD model from a tiny base model."""
        # This would require implementing SVD for tiny models
        # Skipping for now as it's complex and optional
        pytest.skip("SVD model tests require full SVD implementation")



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
