"""Unit tests for the mini_trainer API wrapper."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess

from mini_trainer.api_train import (
    TorchrunArgs,
    TrainingArgs,
    LogLevel,
    run_training,
    StreamablePopen
)


class TestDataclasses:
    """Test the dataclass configurations."""
    
    def test_torchrun_args_defaults(self):
        """Test TorchrunArgs default values."""
        args = TorchrunArgs()
        assert args.nnodes == 1
        assert args.nproc_per_node == 8
        assert args.node_rank == 0
        assert args.rdzv_id == 420
        assert args.rdzv_endpoint == "0.0.0.0:12345"
    
    def test_torchrun_args_custom(self):
        """Test TorchrunArgs with custom values."""
        args = TorchrunArgs(
            nnodes=2,
            nproc_per_node=4,
            node_rank=1,
            rdzv_id=123,
            rdzv_endpoint="localhost:9999"
        )
        assert args.nnodes == 2
        assert args.nproc_per_node == 4
        assert args.node_rank == 1
        assert args.rdzv_id == 123
        assert args.rdzv_endpoint == "localhost:9999"
    
    def test_training_args_defaults(self):
        """Test TrainingArgs default values."""
        args = TrainingArgs()
        assert args.model_name_or_path == "Qwen/Qwen2.5-1.5B-Instruct"
        assert args.data_path == "test.jsonl"
        assert args.batch_size == 1024
        assert args.max_tokens_per_gpu == 10000
        assert args.learning_rate == 5e-6
        assert args.num_warmup_steps == 10
        assert args.lr_scheduler == "constant_with_warmup"
        assert args.seed == 42
        assert args.use_liger_kernels is False
        assert args.orthogonal_subspace_learning is False
        assert args.output_dir == "./output"
        assert args.logging_level == LogLevel.INFO
        assert args.min_samples_per_checkpoint == 1000
    
    def test_training_args_custom(self):
        """Test TrainingArgs with custom values."""
        args = TrainingArgs(
            model_name_or_path="gpt2",
            data_path="/path/to/data.jsonl",
            batch_size=512,
            learning_rate=1e-4,
            use_liger_kernels=True,
            logging_level=LogLevel.DEBUG
        )
        assert args.model_name_or_path == "gpt2"
        assert args.data_path == "/path/to/data.jsonl"
        assert args.batch_size == 512
        assert args.learning_rate == 1e-4
        assert args.use_liger_kernels is True
        assert args.logging_level == LogLevel.DEBUG


class TestStreamablePopen:
    """Test the StreamablePopen wrapper."""
    
    def test_streamable_popen_success(self):
        """Test StreamablePopen with successful command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            command = ["echo", "test output"]
            
            popen = StreamablePopen(str(log_file), command)
            popen.listen()
            
            # Check log file was created and contains output
            assert log_file.exists()
            with open(log_file) as f:
                content = f.read()
                assert "test output" in content
            
            # Check process finished successfully
            assert popen.poll() == 0
    
    def test_streamable_popen_failure(self):
        """Test StreamablePopen with failing command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            command = ["ls", "/nonexistent/path/that/should/not/exist"]
            
            popen = StreamablePopen(str(log_file), command)
            popen.listen()
            
            # Check process failed
            assert popen.poll() != 0


class TestRunTraining:
    """Test the run_training function."""
    
    @patch('mini_trainer.api_train.StreamablePopen')
    def test_run_training_command_construction(self, mock_popen_class):
        """Test that run_training constructs the correct command."""
        mock_popen = MagicMock()
        mock_popen.poll.return_value = 0  # Success
        mock_popen_class.return_value = mock_popen
        
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(
                nnodes=2,
                nproc_per_node=4,
                node_rank=1,
                rdzv_id=999,
                rdzv_endpoint="master:1234"
            )
            train_args = TrainingArgs(
                model_name_or_path="my-model",
                data_path="/data/train.jsonl",
                batch_size=256,
                max_tokens_per_gpu=5000,
                learning_rate=2e-5,
                num_warmup_steps=100,
                lr_scheduler="cosine",
                seed=123,
                output_dir=tmpdir,
                use_liger_kernels=True,
                orthogonal_subspace_learning=True,
                min_samples_per_checkpoint=5000
            )
            
            run_training(torch_args, train_args)
            
            # Check that StreamablePopen was called with correct arguments
            assert mock_popen_class.called
            call_args = mock_popen_class.call_args
            log_file, command = call_args[0]
            
            # Verify log file path
            assert tmpdir in log_file
            assert "training_log_node1.log" in log_file
            
            # Verify command structure
            assert command[0] == "torchrun"
            assert "--nnodes=2" in command
            assert "--node_rank=1" in command
            assert "--nproc_per_node=4" in command
            assert "--rdzv_id=999" in command
            assert "--rdzv_endpoint=master:1234" in command
            
            # Verify training arguments
            assert "--model-name-or-path=my-model" in command
            assert "--data-path=/data/train.jsonl" in command
            assert "--batch-size=256" in command
            assert "--max-tokens-per-gpu=5000" in command
            assert "--learning-rate=2e-05" in command
            assert "--num-warmup-steps=100" in command
            assert "--lr-scheduler=cosine" in command
            assert "--seed=123" in command
            assert f"--output-dir={tmpdir}" in command
            assert "--min-samples-per-checkpoint=5000" in command
            assert "--use-liger-kernels" in command
            assert "--orthogonal-subspace-learning" in command
            
            # Verify listen was called
            mock_popen.listen.assert_called_once()
    
    @patch('mini_trainer.api_train.StreamablePopen')
    def test_run_training_keyboard_interrupt(self, mock_popen_class):
        """Test that run_training handles keyboard interrupt properly."""
        mock_popen = MagicMock()
        mock_popen.poll.return_value = 1  # Failure
        mock_popen.listen.side_effect = KeyboardInterrupt()
        mock_popen_class.return_value = mock_popen
        
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            train_args = TrainingArgs(output_dir=tmpdir)
            
            with pytest.raises(KeyboardInterrupt):
                run_training(torch_args, train_args)
            
            # Verify cleanup was attempted
            mock_popen.terminate.assert_called_once()
            mock_popen.wait.assert_called_once()
    
    @patch('mini_trainer.api_train.StreamablePopen')
    def test_run_training_process_failure(self, mock_popen_class):
        """Test that run_training raises error on process failure."""
        mock_popen = MagicMock()
        mock_popen.poll.return_value = 1  # Failure
        mock_popen_class.return_value = mock_popen
        
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            train_args = TrainingArgs(output_dir=tmpdir)
            
            with pytest.raises(RuntimeError) as exc_info:
                run_training(torch_args, train_args)
            
            assert "Training failed" in str(exc_info.value)
            mock_popen.terminate.assert_called_once()


class TestLogLevel:
    """Test the LogLevel enum."""
    
    def test_log_level_values(self):
        """Test that LogLevel enum has correct values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"
    
    def test_log_level_string_comparison(self):
        """Test that LogLevel can be compared with strings."""
        assert LogLevel.INFO == "INFO"
        assert LogLevel.DEBUG == "DEBUG"
