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


@dataclass
class TorchrunArgs:
    """Arguments for torchrun distributed training configuration."""
    nnodes: int = 1
    nproc_per_node: int = 1
    node_rank: int = 0
    rdzv_id: int = 123
    rdzv_endpoint: str = "127.0.0.1"


@dataclass
class TrainingArgs:
    """Complete training configuration arguments."""
    # Required fields (no defaults)
    model_name_or_path: str
    data_path: str
    batch_size: int
    max_tokens_per_gpu: int
    learning_rate: float
    output_dir: str
    
    # Optional fields (with defaults)
    num_warmup_steps: int = 0
    lr_scheduler: str = "constant_with_warmup"
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    seed: int = 42
    
    # Model configuration
    use_liger_kernels: bool = False
    osft: bool = False
    osft_unfreeze_rank_ratio: float | None = None
    osft_target_patterns: list[str] | None = None
    osft_upcast_dtype: str | None = "float32"
    osft_output_dtype: str | None = None
    
    # Output options
    min_samples_per_checkpoint: Optional[int] = None
    
    # Training mode and stopping criteria
    training_mode: TrainingMode = TrainingMode.EPOCH
    max_epochs: int = 1  # For EPOCH mode
    max_steps: int = 0   # For STEP mode  
    max_tokens: int = 0  # For TOKEN mode
    
    # Checkpointing
    checkpoint_at_epoch: bool = False
    save_final_checkpoint: bool = True

