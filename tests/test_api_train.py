"""Unit tests for the mini_trainer API wrapper."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess

from mini_trainer.api_train import (
    run_training,
    StreamablePopen
)
from mini_trainer.training_types import (
    TorchrunArgs,
    TrainingArgs,
    LogLevelEnum,
    TrainingMode
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
        assert args.logging_level == LogLevelEnum.INFO
        assert args.min_samples_per_checkpoint == 1000
        assert args.use_infinite_sampler is True
        assert args.training_mode == TrainingMode.INFINITE
        assert args.max_epochs == 0
        assert args.max_steps == 0
        assert args.max_tokens == 0
        assert args.checkpoint_at_epoch is False
        assert args.save_final_checkpoint is False
    
    def test_training_args_custom(self):
        """Test TrainingArgs with custom values."""
        args = TrainingArgs(
            model_name_or_path="gpt2",
            data_path="/path/to/data.jsonl",
            batch_size=512,
            learning_rate=1e-4,
            use_liger_kernels=True,
            logging_level=LogLevelEnum.DEBUG
        )
        assert args.model_name_or_path == "gpt2"
        assert args.data_path == "/path/to/data.jsonl"
        assert args.batch_size == 512
        assert args.learning_rate == 1e-4
        assert args.use_liger_kernels is True
        assert args.logging_level == LogLevelEnum.DEBUG


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


    def test_streamable_popen_stdout_capture(self):
        """Test StreamablePopen captures stdout correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "stdout_test.log"
            command = ["python", "-c", "print('Hello stdout'); print('Line 2 stdout')"]
            
            popen = StreamablePopen(str(log_file), command)
            popen.listen()
            
            # Check log file contains stdout
            assert log_file.exists()
            with open(log_file) as f:
                content = f.read()
                assert "Hello stdout" in content
                assert "Line 2 stdout" in content
            
            assert popen.poll() == 0

    def test_streamable_popen_stderr_capture(self):
        """Test StreamablePopen captures stderr correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "stderr_test.log"
            command = ["python", "-c", "import sys; print('Error message', file=sys.stderr); print('Another error', file=sys.stderr)"]
            
            popen = StreamablePopen(str(log_file), command)
            popen.listen()
            
            # Check log file contains stderr
            assert log_file.exists()
            with open(log_file) as f:
                content = f.read()
                assert "Error message" in content
                assert "Another error" in content
            
            assert popen.poll() == 0

    def test_streamable_popen_mixed_output(self):
        """Test StreamablePopen captures both stdout and stderr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "mixed_test.log"
            command = ["python", "-c", 
                      "import sys; "
                      "print('stdout line 1'); "
                      "print('stderr line 1', file=sys.stderr); "
                      "print('stdout line 2'); "
                      "print('stderr line 2', file=sys.stderr)"]
            
            popen = StreamablePopen(str(log_file), command)
            popen.listen()
            
            # Check log file contains both outputs
            assert log_file.exists()
            with open(log_file) as f:
                content = f.read()
                assert "stdout line 1" in content
                assert "stdout line 2" in content
                assert "stderr line 1" in content
                assert "stderr line 2" in content
            
            assert popen.poll() == 0

    def test_streamable_popen_stdout_realtime(self):
        """Test StreamablePopen captures stdout in real-time."""
        import time
        import threading
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "stdout_realtime_test.log"
            # Script that outputs "1", "2", "3" with delays
            command = ["python", "-c", 
                      "import time, sys; "
                      "print('1', flush=True); "
                      "time.sleep(0.5); "
                      "print('2', flush=True); "
                      "time.sleep(0.5); "
                      "print('3', flush=True)"]
            
            popen = StreamablePopen(str(log_file), command)
            
            # Start listening in a separate thread
            listen_thread = threading.Thread(target=popen.listen)
            listen_thread.start()
            
            # Check that "1" appears first
            time.sleep(0.2)  # Give it time to start and print "1"
            with open(log_file) as f:
                content = f.read()
                assert "1" in content
                assert "2" not in content
                assert "3" not in content
            
            # Check that "2" appears next
            time.sleep(0.5)  # Wait for "2" to be printed
            with open(log_file) as f:
                content = f.read()
                assert "1" in content
                assert "2" in content
                assert "3" not in content
            
            # Check that "3" appears last
            time.sleep(0.5)  # Wait for "3" to be printed
            with open(log_file) as f:
                content = f.read()
                assert "1" in content
                assert "2" in content
                assert "3" in content
            
            listen_thread.join()
            assert popen.poll() == 0

    def test_streamable_popen_stderr_realtime(self):
        """Test StreamablePopen captures stderr in real-time."""
        import time
        import threading
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "stderr_realtime_test.log"
            # Script that outputs "1", "2", "3" to stderr with delays
            command = ["python", "-c", 
                      "import time, sys; "
                      "print('1', file=sys.stderr, flush=True); "
                      "time.sleep(0.5); "
                      "print('2', file=sys.stderr, flush=True); "
                      "time.sleep(0.5); "
                      "print('3', file=sys.stderr, flush=True)"]
            
            popen = StreamablePopen(str(log_file), command)
            
            # Start listening in a separate thread
            listen_thread = threading.Thread(target=popen.listen)
            listen_thread.start()
            
            # Check that "1" appears first
            time.sleep(0.2)  # Give it time to start and print "1"
            with open(log_file) as f:
                content = f.read()
                assert "1" in content
                assert "2" not in content
                assert "3" not in content
            
            # Check that "2" appears next
            time.sleep(0.5)  # Wait for "2" to be printed
            with open(log_file) as f:
                content = f.read()
                assert "1" in content
                assert "2" in content
                assert "3" not in content
            
            # Check that "3" appears last
            time.sleep(0.5)  # Wait for "3" to be printed
            with open(log_file) as f:
                content = f.read()
                assert "1" in content
                assert "2" in content
                assert "3" in content
            
            listen_thread.join()
            assert popen.poll() == 0

    def test_streamable_popen_mixed_realtime(self):
        """Test StreamablePopen captures mixed stdout/stderr in real-time."""
        import time
        import threading
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "mixed_realtime_test.log"
            # Script that alternates between stdout and stderr with delays
            command = ["python", "-c", 
                      "import time, sys; "
                      "print('stdout-1', flush=True); "
                      "time.sleep(0.3); "
                      "print('stderr-1', file=sys.stderr, flush=True); "
                      "time.sleep(0.3); "
                      "print('stdout-2', flush=True); "
                      "time.sleep(0.3); "
                      "print('stderr-2', file=sys.stderr, flush=True)"]
            
            popen = StreamablePopen(str(log_file), command)
            
            # Start listening in a separate thread
            listen_thread = threading.Thread(target=popen.listen)
            listen_thread.start()
            
            # Check first stdout appears
            time.sleep(0.15)
            with open(log_file) as f:
                content = f.read()
                assert "stdout-1" in content
                assert "stderr-1" not in content
                assert "stdout-2" not in content
                assert "stderr-2" not in content
            
            # Check first stderr appears
            time.sleep(0.3)
            with open(log_file) as f:
                content = f.read()
                assert "stdout-1" in content
                assert "stderr-1" in content
                assert "stdout-2" not in content
                assert "stderr-2" not in content
            
            # Check second stdout appears
            time.sleep(0.3)
            with open(log_file) as f:
                content = f.read()
                assert "stdout-1" in content
                assert "stderr-1" in content
                assert "stdout-2" in content
                assert "stderr-2" not in content
            
            # Check second stderr appears
            time.sleep(0.3)
            with open(log_file) as f:
                content = f.read()
                assert "stdout-1" in content
                assert "stderr-1" in content
                assert "stdout-2" in content
                assert "stderr-2" in content
            
            listen_thread.join()
            assert popen.poll() == 0


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


class TestEnums:
    """Test the enum types."""
    
    def test_log_level_values(self):
        """Test that LogLevelEnum has correct values."""
        assert LogLevelEnum.DEBUG.value == "DEBUG"
        assert LogLevelEnum.INFO.value == "INFO"
        assert LogLevelEnum.WARNING.value == "WARNING"
        assert LogLevelEnum.ERROR.value == "ERROR"
        assert LogLevelEnum.CRITICAL.value == "CRITICAL"
    
    def test_log_level_string_comparison(self):
        """Test that LogLevelEnum can be compared with strings."""
        assert LogLevelEnum.INFO == "INFO"
        assert LogLevelEnum.DEBUG == "DEBUG"
    
    def test_training_mode_values(self):
        """Test that TrainingMode enum has correct values."""
        assert TrainingMode.EPOCH.value == "epoch"
        assert TrainingMode.STEP.value == "step"
        assert TrainingMode.TOKEN.value == "token"
        assert TrainingMode.INFINITE.value == "infinite"
    
    def test_training_mode_string_comparison(self):
        """Test that TrainingMode can be compared with strings."""
        assert TrainingMode.EPOCH == "epoch"
        assert TrainingMode.INFINITE == "infinite"
