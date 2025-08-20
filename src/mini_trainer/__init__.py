"""Mini Trainer - A simple training library for PyTorch models.

This package provides reference implementations of emerging training algorithms,
including Orthogonal Subspace Fine Tuning (OSFT).
"""

__version__ = "0.1.0"

from . import api_train
from . import async_structured_logger
from . import batch_metrics
from . import batch_packer
from . import none_reduction_losses
from . import sampler
from . import setup_model_for_training
from . import osft_utils
from . import train
from . import utils

# Export main API functions for convenience
from .api_train import run_training
from .training_types import TorchrunArgs, TrainingArgs, TrainingMode

__all__ = [
    "api_train",
    "async_structured_logger",
    "batch_metrics", 
    "batch_packer",
    "none_reduction_losses",
    "sampler",
    "setup_model_for_training",
    "osft_utils",
    "train",
    "utils",
    # Main API exports
    "run_training",
    "TorchrunArgs",
    "TrainingArgs",
    "TrainingMode",
]
