"""API wrapper for mini_trainer that provides programmatic training interface."""

import subprocess
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum


logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Logging level enum."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class TorchrunArgs:
    """Arguments for torchrun distributed training."""
    nnodes: int = 1
    nproc_per_node: int = 8
    node_rank: int = 0
    rdzv_id: int = 420
    rdzv_endpoint: str = "0.0.0.0:12345"


@dataclass
class TrainingArgs:
    """Arguments for training configuration."""
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path: str = "test.jsonl"
    batch_size: int = 1024
    max_tokens_per_gpu: int = 10000
    learning_rate: float = 5e-6
    num_warmup_steps: int = 10
    lr_scheduler: str = "constant_with_warmup"
    seed: int = 42
    use_liger_kernels: bool = False
    orthogonal_subspace_learning: bool = False
    output_dir: str = "./output"
    logging_level: LogLevel = LogLevel.INFO
    min_samples_per_checkpoint: int = 1000


class StreamablePopen:
    """A wrapper for subprocess.Popen that streams output in real-time."""
    
    def __init__(self, log_file: str, command: list):
        self.log_file = log_file
        self.command = command
        self.process = None
        
    def listen(self):
        """Start the process and stream output."""
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else ".", exist_ok=True)
        
        with open(self.log_file, "w") as f:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    print(line, end='')
                    f.write(line)
                    f.flush()
            
            self.process.wait()
    
    def poll(self):
        """Check if process has finished."""
        if self.process:
            return self.process.poll()
        return None
    
    def terminate(self):
        """Terminate the process."""
        if self.process:
            self.process.terminate()
    
    def wait(self, timeout=None):
        """Wait for process to finish."""
        if self.process:
            return self.process.wait(timeout=timeout)
    
    def kill(self):
        """Kill the process."""
        if self.process:
            self.process.kill()


def run_training(torch_args: TorchrunArgs, train_args: TrainingArgs) -> None:
    """
    Wrapper around the main training job that calls torchrun.
    
    Args:
        torch_args: Torchrun configuration for distributed training
        train_args: Training configuration parameters
    """
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, train_args.logging_level.value),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting training setup...")
    
    # Ensure output directory exists
    output_path = Path(train_args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build torchrun command
    train_script = Path(__file__).parent / "train.py"
    
    command = [
        "torchrun",
        f"--nnodes={torch_args.nnodes}",
        f"--node_rank={torch_args.node_rank}",
        f"--nproc_per_node={torch_args.nproc_per_node}",
        f"--rdzv_id={torch_args.rdzv_id}",
        f"--rdzv_endpoint={torch_args.rdzv_endpoint}",
        str(train_script),
        f"--model-name-or-path={train_args.model_name_or_path}",
        f"--data-path={train_args.data_path}",
        f"--batch-size={train_args.batch_size}",
        f"--max-tokens-per-gpu={train_args.max_tokens_per_gpu}",
        f"--learning-rate={train_args.learning_rate}",
        f"--num-warmup-steps={train_args.num_warmup_steps}",
        f"--lr-scheduler={train_args.lr_scheduler}",
        f"--seed={train_args.seed}",
        f"--output-dir={train_args.output_dir}",
        f"--logging-level={train_args.logging_level.value}",
        f"--min-samples-per-checkpoint={train_args.min_samples_per_checkpoint}",
    ]
    
    # Add optional boolean flags
    if train_args.use_liger_kernels:
        command.append("--use-liger-kernels")
    
    if train_args.orthogonal_subspace_learning:
        command.append("--orthogonal-subspace-learning")
    
    logger.info("Running training command as subprocess: %s", " ".join(command))
    
    # Run the training process
    log_file = output_path / f"training_log_node{torch_args.node_rank}.log"
    process = None
    interrupt = None
    failure = False
    
    try:
        process = StreamablePopen(str(log_file), command)
        print(f"Command: {' '.join(command)}")
        print(f"Logs will be saved to: {log_file}")
        process.listen()
    except KeyboardInterrupt as e:
        logger.info("Training subprocess interrupted by user.")
        interrupt = e
    except Exception as e:
        logger.error("Unexpected exception during training", exc_info=e)
        interrupt = e
    finally:
        if process is None:
            return
        
        failure = process.poll() != 0
        if not failure:
            logger.info("Training completed successfully! 🎉")
        else:
            logger.error("Training subprocess failed. Check logs for details.")
        
        process.terminate()
        try:
            logger.info("Waiting for process to exit (60s timeout)...")
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logger.error("Training subprocess did not terminate, sending SIGKILL.")
            process.kill()
        
        if interrupt:
            raise interrupt
        if failure:
            raise RuntimeError(
                f"Training failed. Please check the logs at {log_file} for details."
            )

