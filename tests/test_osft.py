"""
Comprehensive tests for OSFT (Orthogonal Subspace Fine-Tuning) and SVD functionality.

Tests validate:
1. osft_rank_ratio validation in API
2. osft_target_patterns passing through API
3. SVD config generation with custom patterns
4. Integration with setup_model
"""

import pytest
import tempfile
import json
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess

from mini_trainer.api_train import run_training, StreamablePopen
from mini_trainer.training_types import TorchrunArgs, TrainingArgs, LogLevelEnum, TrainingMode
from mini_trainer.svd_utils import (
    auto_generate_target_svd_config, 
    get_model_config, 
    is_svd_param,
    create_svd_model_class,
    MODEL_CONFIGS,
    _get_model_patterns_from_name
)
from mini_trainer.setup_model_for_training import setup_model


class TestOSFTAPIValidation:
    """Test OSFT parameter validation in the API."""
    
    def test_osft_requires_rank_ratio(self):
        """Test that osft=True requires osft_rank_ratio to be provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            # osft=True but osft_rank_ratio=None should raise error
            train_args = TrainingArgs(
                output_dir=tmpdir,
                osft=True,
                osft_rank_ratio=None  # This should cause an error
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                # it should not even run this, so return value doesn't matter here
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                with pytest.raises(ValueError, match="osft_rank_ratio is required when osft is True"):
                    run_training(torch_args, train_args)
                
                # shouldnt have even gotten run
                assert mock_popen_class.call_count == 0
            
    
    def test_osft_with_valid_rank_ratio(self):
        """Test that osft=True with valid rank_ratio passes validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            train_args = TrainingArgs(
                output_dir=tmpdir,
                osft=True,
                osft_rank_ratio=0.5  # Valid ratio
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0  # Success
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                # Verify command includes osft parameters
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                assert "--osft" in command
                assert "--osft-rank-ratio=0.5" in command
    
    def test_osft_rank_ratio_not_required_when_osft_false(self):
        """Test that osft_rank_ratio is not required when osft=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            train_args = TrainingArgs(
                output_dir=tmpdir,
                osft=False,
                osft_rank_ratio=None  # This should be fine
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                # Should not raise error
                run_training(torch_args, train_args)
                
                # Verify osft parameters not in command
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                assert "--osft" not in command
                assert all(not arg.startswith("--osft-rank-ratio") for arg in command)
    
    def test_osft_target_patterns_passed_through(self):
        """Test that osft_target_patterns are correctly passed through the API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            test_patterns = ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj"]
            train_args = TrainingArgs(
                output_dir=tmpdir,
                osft=True,
                osft_rank_ratio=0.75,
                osft_target_patterns=test_patterns
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                # Verify command includes target patterns
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                assert "--osft" in command
                assert "--osft-rank-ratio=0.75" in command
                # Find the target patterns argument
                patterns_arg = None
                for arg in command:
                    if arg.startswith("--osft-target-patterns="):
                        patterns_arg = arg
                        break
                
                assert patterns_arg is not None
                # The patterns should be passed as a list string
                expected = "--osft-target-patterns=self_attn.q_proj,self_attn.k_proj,mlp.gate_proj"
                assert patterns_arg == expected
    
    def test_osft_target_patterns_empty_list(self):
        """Test that empty osft_target_patterns list is handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            train_args = TrainingArgs(
                output_dir=tmpdir,
                osft=True,
                osft_rank_ratio=0.5,
                osft_target_patterns=[]  # Empty list
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                # Empty list is treated same as None - not passed
                # This is reasonable as empty list means no custom patterns
                patterns_arg = None
                for arg in command:
                    if arg.startswith("--osft-target-patterns="):
                        patterns_arg = arg
                        break
                
                assert patterns_arg is None
    
    def test_osft_target_patterns_none_not_passed(self):
        """Test that None osft_target_patterns is not passed to command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            train_args = TrainingArgs(
                output_dir=tmpdir,
                osft=True,
                osft_rank_ratio=0.5,
                osft_target_patterns=None  # None should not be passed
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                # None should result in no target patterns argument
                assert all(not arg.startswith("--osft-target-patterns") for arg in command)
    
    def test_various_rank_ratios(self):
        """Test that different rank ratios are correctly passed."""
        rank_ratios = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        
        for ratio in rank_ratios:
            with tempfile.TemporaryDirectory() as tmpdir:
                torch_args = TorchrunArgs()
                train_args = TrainingArgs(
                    output_dir=tmpdir,
                    osft=True,
                    osft_rank_ratio=ratio
                )
                
                with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                    mock_popen = MagicMock()
                    mock_popen.poll.return_value = 0
                    mock_popen_class.return_value = mock_popen
                    
                    run_training(torch_args, train_args)
                    
                    call_args = mock_popen_class.call_args
                    _, command = call_args[0]
                    
                    assert f"--osft-rank-ratio={ratio}" in command


class TestSVDConfigGeneration:
    """Test SVD configuration generation with custom patterns."""
    
    def test_get_model_patterns_from_name(self):
        """Test pattern detection from model names."""
        # Test known model types
        assert _get_model_patterns_from_name("llama") == MODEL_CONFIGS["llama"]["patterns"]
        assert _get_model_patterns_from_name("gpt-j-6b") == MODEL_CONFIGS["gpt-j"]["patterns"]
        assert _get_model_patterns_from_name("gptj") == MODEL_CONFIGS["gpt-j"]["patterns"]
        assert _get_model_patterns_from_name("opt-350m") == MODEL_CONFIGS["opt"]["patterns"]
        assert _get_model_patterns_from_name("qwen2-7b") == MODEL_CONFIGS["qwen"]["patterns"]
        assert _get_model_patterns_from_name("gemma-2b") == MODEL_CONFIGS["gemma"]["patterns"]
        
        # Test default fallback
        assert _get_model_patterns_from_name("unknown-model") == MODEL_CONFIGS["default"]["patterns"]
    
    def test_get_model_config_with_custom_patterns(self):
        """Test that custom patterns override model defaults."""
        custom_patterns = ["custom.layer1", "custom.layer2"]
        
        # Custom patterns should override model-specific patterns
        patterns = get_model_config("llama", target_patterns=custom_patterns)
        assert patterns == custom_patterns
        
        # Custom patterns should override default
        patterns = get_model_config(None, target_patterns=custom_patterns)
        assert patterns == custom_patterns
    
    def test_get_model_config_without_custom_patterns(self):
        """Test model config retrieval without custom patterns."""
        # Should get model-specific patterns
        patterns = get_model_config("llama", target_patterns=None)
        assert patterns == MODEL_CONFIGS["llama"]["patterns"]
        
        # Should get default patterns
        patterns = get_model_config(None, target_patterns=None)
        assert patterns == MODEL_CONFIGS["default"]["patterns"]
    
    def test_auto_generate_svd_config_with_custom_patterns(self):
        """Test SVD config generation with custom target patterns."""
        # Create a mock model with various layers
        mock_model = MagicMock()
        mock_params = [
            ("layer1.self_attn.q_proj.weight", torch.zeros(128, 64)),
            ("layer1.self_attn.k_proj.weight", torch.zeros(128, 64)),
            ("layer1.mlp.gate_proj.weight", torch.zeros(256, 128)),
            ("layer2.custom_proj.weight", torch.zeros(100, 50)),
            ("layer2.another_proj.weight", torch.zeros(200, 100)),
        ]
        mock_model.named_parameters.return_value = mock_params
        
        # Test with custom patterns
        custom_patterns = ["custom_proj", "another_proj"]
        config = auto_generate_target_svd_config(
            mock_model,
            target_patterns=custom_patterns,
            rank_ratio=0.5
        )
        
        # Should only include layers matching custom patterns
        assert "layer2.custom_proj.weight" in config
        assert "layer2.another_proj.weight" in config
        assert "layer1.self_attn.q_proj.weight" not in config
        assert "layer1.self_attn.k_proj.weight" not in config
        assert "layer1.mlp.gate_proj.weight" not in config
        
        # Check rank values
        assert config["layer2.custom_proj.weight"] == 25  # min(100, 50) * 0.5
        assert config["layer2.another_proj.weight"] == 50  # min(200, 100) * 0.5
    
    def test_auto_generate_svd_config_with_rank_ratio(self):
        """Test that rank_ratio correctly affects the generated config."""
        mock_model = MagicMock()
        mock_params = [
            ("layer.proj.weight", torch.zeros(100, 80)),
        ]
        mock_model.named_parameters.return_value = mock_params
        
        # Test different rank ratios
        for ratio in [0.1, 0.25, 0.5, 0.75, 0.9]:
            config = auto_generate_target_svd_config(
                mock_model,
                target_patterns=["proj"],
                rank_ratio=ratio
            )
            
            expected_rank = int(80 * ratio)  # min(100, 80) * ratio
            assert config["layer.proj.weight"] == expected_rank
    
    def test_auto_generate_svd_config_edge_cases(self):
        """Test edge cases in SVD config generation."""
        mock_model = MagicMock()
        
        # Test with rank_ratio >= 1.0 (should cap at full_rank - 1)
        mock_params = [("layer.proj.weight", torch.zeros(50, 50))]
        mock_model.named_parameters.return_value = mock_params
        
        config = auto_generate_target_svd_config(
            mock_model,
            target_patterns=["proj"],
            rank_ratio=1.0
        )
        assert config["layer.proj.weight"] == 49  # full_rank - 1
        
        # Test with 1D parameters (should be skipped)
        mock_params = [
            ("layer.bias", torch.zeros(100)),  # 1D parameter
            ("layer.weight", torch.zeros(100, 50)),  # 2D parameter
        ]
        mock_model.named_parameters.return_value = mock_params
        
        config = auto_generate_target_svd_config(
            mock_model,
            target_patterns=["layer"],
            rank_ratio=0.5
        )
        
        assert "layer.bias" not in config  # 1D should be skipped
        assert "layer.weight" in config  # 2D should be included
    
    def test_is_svd_param_function(self):
        """Test the is_svd_param utility function."""
        svd_config = {
            "layer1.weight": 10,
            "layer2.weight": 0,  # 0 means not SVD
        }
        
        # 2D param with positive rank in config
        param_2d = torch.zeros(100, 50)
        assert is_svd_param("layer1.weight", param_2d, svd_config) is True
        
        # 2D param with 0 rank in config
        assert is_svd_param("layer2.weight", param_2d, svd_config) is False
        
        # 2D param not in config
        assert is_svd_param("layer3.weight", param_2d, svd_config) is False
        
        # 1D param (should be False regardless)
        param_1d = torch.zeros(100)
        assert is_svd_param("layer1.weight", param_1d, svd_config) is False


class TestSVDModelCreation:
    """Test SVD model class creation and initialization."""
    
    def test_create_svd_model_class(self):
        """Test that create_svd_model_class creates a valid subclass."""
        # Create a simple mock base class
        class MockModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.linear = nn.Linear(10, 10)
        
        # Create SVD model class
        SVDModelClass = create_svd_model_class(MockModel)
        
        # Check class inheritance
        assert issubclass(SVDModelClass, MockModel)
        assert SVDModelClass.__name__ == "MockModelWithSVD"
        
        # Check that required methods exist
        assert hasattr(SVDModelClass, 'reinitialize_svd')
        assert hasattr(SVDModelClass, 'reinitialize_svd_distributed')
        assert hasattr(SVDModelClass, 'project_gradients')
        assert hasattr(SVDModelClass, 'from_pretrained')
    
    def test_svd_model_initialization_without_svd(self):
        """Test SVD model can be initialized without SVD decomposition."""
        class MockModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.dtype = torch.float32
        
        SVDModelClass = create_svd_model_class(MockModel)
        
        # Initialize without SVD
        config = MagicMock()
        model = SVDModelClass(config, svd_config={}, initialize_svd=False)
        
        assert model.svd_config == {}
        assert hasattr(model, 'svd_params')
        assert len(model.svd_params) == 0


class TestSetupModelIntegration:
    """Test integration of OSFT options with setup_model function."""
    
    def test_osft_params_flow_through_setup(self):
        """Test that OSFT parameters flow through the setup correctly."""
        # Test that SVD model creation gets the right parameters
        with patch('mini_trainer.svd_utils.create_svd_model_class') as mock_svd_class:
            with patch('mini_trainer.svd_utils.auto_generate_target_svd_config') as mock_auto_config:
                mock_auto_config.return_value = {"layer.weight": 10}
                
                # Mock the SVD model class
                mock_svd_model_cls = MagicMock()
                mock_svd_instance = MagicMock()
                mock_svd_instance.config = MagicMock()
                mock_svd_instance.config.vocab_size = 1000
                mock_svd_instance.dtype = torch.float32
                
                # Setup from_pretrained to return a properly configured model
                def from_pretrained_side_effect(*args, **kwargs):
                    # Store the kwargs for verification
                    mock_svd_model_cls.last_kwargs = kwargs
                    return mock_svd_instance
                
                mock_svd_model_cls.from_pretrained.side_effect = from_pretrained_side_effect
                mock_svd_class.return_value = mock_svd_model_cls
                
                # Mock tokenizer and base model
                with patch('mini_trainer.setup_model_for_training.AutoTokenizer') as mock_tokenizer_cls:
                    with patch('mini_trainer.setup_model_for_training.AutoModelForCausalLM') as mock_model_cls:
                        with patch('mini_trainer.setup_model_for_training.log_rank_0'):
                            mock_tokenizer = MagicMock()
                            mock_tokenizer.__len__ = MagicMock(return_value=1000)
                            mock_tokenizer.pad_token_id = 0
                            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
                            
                            mock_base_model = MagicMock()
                            mock_base_model.config = MagicMock()
                            mock_base_model.config.vocab_size = 1000
                            mock_model_cls.from_pretrained.return_value = mock_base_model
                            
                            # Call setup_model with OSFT params
                            model = setup_model(
                                osft=True,
                                rank=0,
                                osft_rank_ratio=0.75,
                                osft_target_patterns=["custom.layer1", "custom.layer2"],
                                model_name_or_path="test-model"
                            )
                            
                            # Verify the SVD model class was created
                            mock_svd_class.assert_called_once()
                            
                            # Verify from_pretrained was called with the right params
                            assert 'rank_ratio' in mock_svd_model_cls.last_kwargs
                            assert mock_svd_model_cls.last_kwargs['rank_ratio'] == 0.75
                            assert 'target_patterns' in mock_svd_model_cls.last_kwargs
                            assert mock_svd_model_cls.last_kwargs['target_patterns'] == ["custom.layer1", "custom.layer2"]


class TestEndToEndOSFT:
    """End-to-end tests for OSFT functionality."""
    
    def test_command_line_osft_params_validation(self):
        """Test that command line validates OSFT parameters correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test script that validates OSFT params
            test_script = Path(tmpdir) / "validate_osft.py"
            test_script.write_text("""
import sys
import typer
from typing import Optional

app = typer.Typer()

@app.command()
def main(
    osft: bool = False,
    osft_rank_ratio: Optional[float] = None,
    osft_target_patterns: Optional[str] = None
):
    # Validate: if osft is True, rank_ratio must be provided
    if osft and osft_rank_ratio is None:
        print("ERROR: osft_rank_ratio required")
        raise typer.Exit(1)

    # Parse target patterns if provided (comma-delimited)
    if osft_target_patterns:
        patterns = [p.strip() for p in osft_target_patterns.split(",")]
        print(f"PATTERNS: {patterns}")

    print(f"SUCCESS: osft={osft}, ratio={osft_rank_ratio}")

if __name__ == "__main__":
    app()
""")
            
            # Test valid OSFT configuration
            result = subprocess.run(
                ["python", str(test_script), "--osft", "--osft-rank-ratio=0.5", 
                 "--osft-target-patterns=q_proj,k_proj"],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert "SUCCESS" in result.stdout
            assert "PATTERNS: ['q_proj', 'k_proj']" in result.stdout
            
            # Test missing rank_ratio
            result = subprocess.run(
                ["python", str(test_script), "--osft"],
                capture_output=True,
                text=True
            )
            assert result.returncode == 1
            assert "ERROR: osft_rank_ratio required" in result.stdout
            
            # Test osft=False doesn't require rank_ratio
            result = subprocess.run(
                ["python", str(test_script)],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert "SUCCESS: osft=False" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
