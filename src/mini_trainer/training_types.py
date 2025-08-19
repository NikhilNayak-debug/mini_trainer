"""
Shared type definitions and dataclasses for mini_trainer.

This module consolidates all common type definitions, enums, and dataclasses
used across the mini_trainer package to avoid duplication and ensure consistency.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class TrainingMode(str, Enum):
    """Training mode determines the stopping criterion for training."""
    EPOCH = "epoch"
    STEP = "step"
    TOKEN = "token"
    INFINITE = "infinite"


class LogLevelEnum(str, Enum):
    """Logging level configuration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class TorchrunArgs:
    """Arguments for torchrun distributed training configuration."""
    nnodes: int = 1
    nproc_per_node: int = 8
    node_rank: int = 0
    rdzv_id: int = 420
    rdzv_endpoint: str = "0.0.0.0:12345"


@dataclass
class TrainingArgs:
    """Complete training configuration arguments."""
    # Model and data
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path: str = "test.jsonl"
    
    # Training hyperparameters
    batch_size: int = 1024
    max_tokens_per_gpu: int = 10000
    learning_rate: float = 5e-6
    num_warmup_steps: int = 10
    lr_scheduler: str = "constant_with_warmup"
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    seed: int = 42
    
    # Model configuration
    use_liger_kernels: bool = False
    orthogonal_subspace_learning: bool = False
    
    # Output and logging
    output_dir: str = "./output"
    logging_level: LogLevelEnum = LogLevelEnum.INFO
    min_samples_per_checkpoint: int = 1000
    
    # Sampling configuration
    use_infinite_sampler: bool = True
    
    # Training mode and stopping criteria
    training_mode: TrainingMode = TrainingMode.INFINITE
    max_epochs: int = 0  # For EPOCH mode
    max_steps: int = 0   # For STEP mode  
    max_tokens: int = 0  # For TOKEN mode
    
    # Checkpointing
    checkpoint_at_epoch: bool = False
    save_final_checkpoint: bool = False
