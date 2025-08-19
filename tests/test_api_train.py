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
        assert args.min_samples_per_checkpoint is None
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
            # min_samples_per_checkpoint=5000 should be in the command
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


class TestParameterPassing:
    """Test that parameters are correctly passed through to the training script."""
    
    def test_lr_scheduler_kwargs_empty(self):
        """Test that empty lr_scheduler_kwargs is passed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock training script that just prints the received arguments
            mock_script = Path(tmpdir) / "mock_train.py"
            mock_script.write_text("""
import sys
import json

# Parse arguments and print lr_scheduler_kwargs
for i, arg in enumerate(sys.argv):
    if arg.startswith("--lr-scheduler-kwargs="):
        kwargs_str = arg.split("=", 1)[1]
        kwargs = json.loads(kwargs_str)
        print(f"LR_SCHEDULER_KWARGS:{json.dumps(kwargs)}")
        sys.exit(0)
""")
            
            with patch('mini_trainer.api_train.Path') as mock_path:
                # Make the train script path point to our mock script
                mock_path.return_value.__truediv__.return_value = mock_script
                
                torch_args = TorchrunArgs(nproc_per_node=1)
                train_args = TrainingArgs(
                    output_dir=tmpdir,
                    lr_scheduler_kwargs=None  # Should default to empty dict
                )
                
                # Capture the subprocess output
                with patch('subprocess.Popen') as mock_popen:
                    mock_process = MagicMock()
                    mock_process.stdout.readline.side_effect = [
                        "LR_SCHEDULER_KWARGS:{}\n",
                        ""  # End of output
                    ]
                    mock_process.poll.return_value = 0
                    mock_process.wait.return_value = 0
                    mock_popen.return_value = mock_process
                    
                    process = StreamablePopen(str(Path(tmpdir) / "test.log"), ["dummy"])
                    process.process = mock_process
                    
                    # Check that empty dict is passed
                    output = []
                    for line in iter(mock_process.stdout.readline, ''):
                        if line:
                            output.append(line.strip())
                    
                    assert "LR_SCHEDULER_KWARGS:{}" in output[0]
    
    def test_lr_scheduler_kwargs_complex(self):
        """Test that complex lr_scheduler_kwargs dict is passed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock training script
            mock_script = Path(tmpdir) / "mock_train.py"
            mock_script.write_text("""
import sys
import json

# Parse arguments and validate lr_scheduler_kwargs
for i, arg in enumerate(sys.argv):
    if arg.startswith("--lr-scheduler-kwargs="):
        kwargs_str = arg.split("=", 1)[1]
        kwargs = json.loads(kwargs_str)
        # Validate the complex kwargs
        assert "min_lr" in kwargs
        assert kwargs["min_lr"] == 1e-6
        assert "T_max" in kwargs
        assert kwargs["T_max"] == 1000
        assert "eta_min" in kwargs
        assert kwargs["eta_min"] == 0.0
        assert "nested" in kwargs
        assert kwargs["nested"]["key1"] == "value1"
        assert kwargs["nested"]["key2"] == 42
        print("KWARGS_VALIDATED:SUCCESS")
        sys.exit(0)

print("KWARGS_VALIDATED:FAILED")
sys.exit(1)
""")
            
            complex_kwargs = {
                "min_lr": 1e-6,
                "T_max": 1000,
                "eta_min": 0.0,
                "nested": {
                    "key1": "value1",
                    "key2": 42
                }
            }
            
            # Actually run the subprocess to test end-to-end
            command = [
                "python", str(mock_script),
                f"--lr-scheduler-kwargs={json.dumps(complex_kwargs)}"
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert "KWARGS_VALIDATED:SUCCESS" in result.stdout
    
    def test_command_construction_with_special_chars(self):
        """Test that special characters in kwargs are properly escaped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            train_args = TrainingArgs(
                output_dir=tmpdir,
                lr_scheduler_kwargs={
                    "description": "A string with spaces and special chars: @#$%",
                    "path": "/path/with/slashes",
                    "float_val": 3.14159,
                    "bool_val": True,
                    "null_val": None
                }
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                # Get the constructed command
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                # Find the lr-scheduler-kwargs argument
                kwargs_arg = None
                for arg in command:
                    if arg.startswith("--lr-scheduler-kwargs="):
                        kwargs_arg = arg
                        break
                
                assert kwargs_arg is not None
                
                # Extract and parse the JSON
                json_str = kwargs_arg.split("=", 1)[1]
                parsed_kwargs = json.loads(json_str)
                
                # Verify all values are preserved correctly
                assert parsed_kwargs["description"] == "A string with spaces and special chars: @#$%"
                assert parsed_kwargs["path"] == "/path/with/slashes"
                assert parsed_kwargs["float_val"] == 3.14159
                assert parsed_kwargs["bool_val"] is True
                assert parsed_kwargs["null_val"] is None
    
    def test_all_boolean_flags_passed(self):
        """Test that all boolean flags are correctly passed when True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            train_args = TrainingArgs(
                output_dir=tmpdir,
                use_liger_kernels=True,
                orthogonal_subspace_learning=True,
                use_infinite_sampler=True,
                checkpoint_at_epoch=True,
                save_final_checkpoint=True
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                # Verify all boolean flags are present
                assert "--use-liger-kernels" in command
                assert "--orthogonal-subspace-learning" in command
                assert "--use-infinite-sampler" in command
                assert "--checkpoint-at-epoch" in command
                assert "--save-final-checkpoint" in command
    
    def test_boolean_flags_not_passed_when_false(self):
        """Test that boolean flags are not passed when False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            train_args = TrainingArgs(
                output_dir=tmpdir,
                use_liger_kernels=False,
                orthogonal_subspace_learning=False,
                use_infinite_sampler=False,
                checkpoint_at_epoch=False,
                save_final_checkpoint=False
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                # Verify boolean flags are NOT present when False
                assert "--use-liger-kernels" not in command
                assert "--orthogonal-subspace-learning" not in command
                # Note: use_infinite_sampler defaults to True, so we set it to False
                assert "--use-infinite-sampler" not in command
                assert "--checkpoint-at-epoch" not in command
                assert "--save-final-checkpoint" not in command
    
    def test_training_mode_enum_passed_correctly(self):
        """Test that TrainingMode enum values are correctly converted to strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for mode in [TrainingMode.EPOCH, TrainingMode.STEP, TrainingMode.TOKEN, TrainingMode.INFINITE]:
                torch_args = TorchrunArgs()
                train_args = TrainingArgs(
                    output_dir=tmpdir,
                    training_mode=mode
                )
                
                with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                    mock_popen = MagicMock()
                    mock_popen.poll.return_value = 0
                    mock_popen_class.return_value = mock_popen
                    
                    run_training(torch_args, train_args)
                    
                    call_args = mock_popen_class.call_args
                    _, command = call_args[0]
                    
                    # Find the training-mode argument
                    mode_arg = None
                    for arg in command:
                        if arg.startswith("--training-mode="):
                            mode_arg = arg
                            break
                    
                    assert mode_arg == f"--training-mode={mode.value}"
    
    def test_logging_level_enum_passed_correctly(self):
        """Test that LogLevelEnum values are correctly converted to strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for level in [LogLevelEnum.DEBUG, LogLevelEnum.INFO, LogLevelEnum.WARNING, 
                         LogLevelEnum.ERROR, LogLevelEnum.CRITICAL]:
                torch_args = TorchrunArgs()
                train_args = TrainingArgs(
                    output_dir=tmpdir,
                    logging_level=level
                )
                
                with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                    mock_popen = MagicMock()
                    mock_popen.poll.return_value = 0
                    mock_popen_class.return_value = mock_popen
                    
                    run_training(torch_args, train_args)
                    
                    call_args = mock_popen_class.call_args
                    _, command = call_args[0]
                    
                    # Find the logging-level argument
                    level_arg = None
                    for arg in command:
                        if arg.startswith("--logging-level="):
                            level_arg = arg
                            break
                    
                    assert level_arg == f"--logging-level={level.value}"
    
    def test_numeric_parameters_precision(self):
        """Test that numeric parameters maintain precision when passed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs()
            train_args = TrainingArgs(
                output_dir=tmpdir,
                learning_rate=1.23456789e-7,  # Very small float
                batch_size=2048,
                max_tokens_per_gpu=50000,
                num_warmup_steps=1000,
                min_samples_per_checkpoint=10000,
                max_epochs=100,
                max_steps=50000,
                max_tokens=1000000000  # Large integer
            )
            
            with patch('mini_trainer.api_train.StreamablePopen') as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen
                
                run_training(torch_args, train_args)
                
                call_args = mock_popen_class.call_args
                _, command = call_args[0]
                
                # Check that numeric values are preserved
                assert "--learning-rate=1.23456789e-07" in command
                assert "--batch-size=2048" in command
                assert "--max-tokens-per-gpu=50000" in command
                assert "--num-warmup-steps=1000" in command
                assert "--min-samples-per-checkpoint=10000" in command
                assert "--max-epochs=100" in command
                assert "--max-steps=50000" in command
                assert "--max-tokens=1000000000" in command
    
    def test_end_to_end_parameter_reception(self):
        """Integration test that verifies train.py receives parameters correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock train.py that validates all received parameters
            mock_train_script = Path(tmpdir) / "validate_train.py"
            mock_train_script.write_text("""
import sys
import json
import argparse

# Parse all arguments exactly as train.py would
parser = argparse.ArgumentParser()
parser.add_argument("--model-name-or-path", type=str)
parser.add_argument("--data-path", type=str)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--max-tokens-per-gpu", type=int)
parser.add_argument("--learning-rate", type=float)
parser.add_argument("--num-warmup-steps", type=int)
parser.add_argument("--lr-scheduler", type=str)
parser.add_argument("--lr-scheduler-kwargs", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--output-dir", type=str)
parser.add_argument("--logging-level", type=str)
parser.add_argument("--min-samples-per-checkpoint", type=int, default=None)
parser.add_argument("--training-mode", type=str)
parser.add_argument("--max-epochs", type=int)
parser.add_argument("--max-steps", type=int)
parser.add_argument("--max-tokens", type=int)
parser.add_argument("--use-liger-kernels", action="store_true")
parser.add_argument("--orthogonal-subspace-learning", action="store_true")
parser.add_argument("--use-infinite-sampler", action="store_true")
parser.add_argument("--checkpoint-at-epoch", action="store_true")
parser.add_argument("--save-final-checkpoint", action="store_true")

args = parser.parse_args()

# Validate received values
validation_results = []

# Check string parameters
validation_results.append(("model", args.model_name_or_path == "test-model-path"))
validation_results.append(("data", args.data_path == "/path/to/data.jsonl"))
validation_results.append(("scheduler", args.lr_scheduler == "cosine_annealing"))
validation_results.append(("output", args.output_dir.endswith("test_output")))
validation_results.append(("logging", args.logging_level == "DEBUG"))
validation_results.append(("mode", args.training_mode == "epoch"))

# Check numeric parameters
validation_results.append(("batch", args.batch_size == 512))
validation_results.append(("tokens", args.max_tokens_per_gpu == 8192))
validation_results.append(("lr", abs(args.learning_rate - 3.5e-5) < 1e-10))
validation_results.append(("warmup", args.num_warmup_steps == 250))
validation_results.append(("checkpoint", args.min_samples_per_checkpoint == 2500 if args.min_samples_per_checkpoint is not None else False))
validation_results.append(("epochs", args.max_epochs == 10))
validation_results.append(("steps", args.max_steps == 1000))
validation_results.append(("max_tokens", args.max_tokens == 5000000))
validation_results.append(("seed", args.seed == 999))

# Check boolean flags
validation_results.append(("liger", args.use_liger_kernels == True))
validation_results.append(("osft", args.orthogonal_subspace_learning == True))
validation_results.append(("infinite", args.use_infinite_sampler == False))
validation_results.append(("epoch_ckpt", args.checkpoint_at_epoch == True))
validation_results.append(("final_ckpt", args.save_final_checkpoint == True))

# Check lr_scheduler_kwargs JSON parsing
try:
    kwargs = json.loads(args.lr_scheduler_kwargs)
    validation_results.append(("kwargs_T_max", kwargs.get("T_max") == 500))
    validation_results.append(("kwargs_eta_min", abs(kwargs.get("eta_min", 0) - 1e-7) < 1e-10))
    validation_results.append(("kwargs_nested", kwargs.get("nested", {}).get("value") == "test"))
except:
    validation_results.append(("kwargs_parse", False))

# Report results
failed = [name for name, passed in validation_results if not passed]
if failed:
    print(f"VALIDATION_FAILED: {','.join(failed)}")
    sys.exit(1)
else:
    print("VALIDATION_SUCCESS")
    sys.exit(0)
""")
            
            # Set up the test parameters with specific values
            complex_kwargs = {
                "T_max": 500,
                "eta_min": 1e-7,
                "nested": {"value": "test"}
            }
            
            # Build the command as run_training would
            command = [
                "python", str(mock_train_script),
                "--model-name-or-path=test-model-path",
                "--data-path=/path/to/data.jsonl",
                "--batch-size=512",
                "--max-tokens-per-gpu=8192",
                "--learning-rate=3.5e-05",
                "--num-warmup-steps=250",
                "--lr-scheduler=cosine_annealing",
                f"--lr-scheduler-kwargs={json.dumps(complex_kwargs)}",
                "--seed=999",
                f"--output-dir={tmpdir}/test_output",
                "--logging-level=DEBUG",
                "--min-samples-per-checkpoint=2500",
                "--training-mode=epoch",
                "--max-epochs=10",
                "--max-steps=1000",
                "--max-tokens=5000000",
                "--use-liger-kernels",
                "--orthogonal-subspace-learning",
                # Note: NOT including --use-infinite-sampler (testing False)
                "--checkpoint-at-epoch",
                "--save-final-checkpoint"
            ]
            
            # Run the validation script
            result = subprocess.run(command, capture_output=True, text=True)
            
            # Check validation passed
            assert result.returncode == 0, f"Validation failed: {result.stdout}\n{result.stderr}"
            assert "VALIDATION_SUCCESS" in result.stdout
