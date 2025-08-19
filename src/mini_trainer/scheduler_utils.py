"""
Utilities for learning rate scheduler configuration.

This module provides functions for calculating training steps and configuring
learning rate schedulers based on different training modes.
"""

from typing import Optional, Dict, Any
import logging

from mini_trainer.training_types import TrainingMode

logger = logging.getLogger(__name__)


def calculate_num_training_steps(
    training_mode: TrainingMode,
    data_loader,
    max_epochs: int = 0,
    max_steps: int = 0,
    max_tokens: int = 0,
    use_infinite_sampler: bool = True,
) -> Optional[int]:
    """
    Calculate the number of training steps based on the training mode.
    
    Args:
        training_mode: The training mode (EPOCH, STEP, TOKEN, or INFINITE)
        data_loader: The data loader to get dataset statistics from
        max_epochs: Maximum epochs for EPOCH mode
        max_steps: Maximum steps for STEP mode
        max_tokens: Maximum tokens for TOKEN mode
        use_infinite_sampler: Whether using infinite sampler
    
    Returns:
        Number of training steps, or None for INFINITE mode or when it can't be calculated
    """
    
    if training_mode == TrainingMode.INFINITE:
        logger.info("INFINITE training mode: num_training_steps is None")
        return None
    
    elif training_mode == TrainingMode.STEP:
        logger.info(f"STEP training mode: num_training_steps = {max_steps}")
        return max_steps
    
    elif training_mode == TrainingMode.EPOCH:
        if use_infinite_sampler:
            logger.warning("Cannot calculate training steps for EPOCH mode with infinite sampler")
            return None
        
        # Count the number of batches in one epoch
        try:
            # Try to get length directly if available
            if hasattr(data_loader, '__len__'):
                num_batches_per_epoch = len(data_loader)
            else:
                # Otherwise iterate through one epoch to count
                logger.info("Counting batches by iterating through data loader...")
                num_batches_per_epoch = 0
                for _ in data_loader:
                    num_batches_per_epoch += 1
        except Exception as e:
            logger.error(f"Failed to count batches: {e}")
            return None
        
        num_training_steps = num_batches_per_epoch * max_epochs
        logger.info(f"EPOCH training mode: {num_batches_per_epoch} batches/epoch * {max_epochs} epochs = {num_training_steps} steps")
        return num_training_steps
    
    elif training_mode == TrainingMode.TOKEN:
        if use_infinite_sampler:
            logger.warning("Cannot calculate training steps for TOKEN mode with infinite sampler")
            return None
        
        # Calculate average tokens per batch
        logger.info("Calculating average tokens per batch...")
        batch_token_counts = []
        total_loss_tokens = 0
        
        try:
            for batch in data_loader:
                batch_tokens = 0
                for minibatch in batch:
                    if 'num_loss_counted_tokens' in minibatch:
                        total_loss_tokens += minibatch['num_loss_counted_tokens']
                    if 'batch_num_loss_counted_tokens' in minibatch:
                        batch_tokens = minibatch['batch_num_loss_counted_tokens']
                
                if batch_tokens > 0:
                    batch_token_counts.append(batch_tokens)
            
            if not batch_token_counts:
                logger.error("No token counts found in batches")
                return None
            
            avg_tokens_per_batch = sum(batch_token_counts) / len(batch_token_counts)
            num_training_steps = int(max_tokens / avg_tokens_per_batch)
            
            logger.info(f"TOKEN training mode: {max_tokens} tokens / {avg_tokens_per_batch:.1f} avg tokens/batch = {num_training_steps} steps")
            return num_training_steps
            
        except Exception as e:
            logger.error(f"Failed to calculate token-based steps: {e}")
            return None
    
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")


def get_dataset_metrics(data_loader, use_infinite_sampler: bool = True) -> Dict[str, Any]:
    """
    Get comprehensive metrics from the data loader for scheduler configuration.
    
    Args:
        data_loader: The data loader to analyze
        use_infinite_sampler: Whether using infinite sampler
    
    Returns:
        Dictionary with dataset metrics
    """
    metrics = {
        'num_batches': 0,
        'total_samples': 0,
        'total_loss_tokens': 0,
        'avg_tokens_per_batch': 0,
        'batch_token_counts': []
    }
    
    if use_infinite_sampler:
        logger.warning("Cannot get complete dataset metrics with infinite sampler")
        return metrics
    
    try:
        for batch in data_loader:
            metrics['num_batches'] += 1
            batch_loss_tokens = 0
            
            for minibatch in batch:
                if 'num_samples' in minibatch:
                    metrics['total_samples'] += minibatch['num_samples']
                if 'num_loss_counted_tokens' in minibatch:
                    metrics['total_loss_tokens'] += minibatch['num_loss_counted_tokens']
                if 'batch_num_loss_counted_tokens' in minibatch:
                    batch_loss_tokens = minibatch['batch_num_loss_counted_tokens']
            
            if batch_loss_tokens > 0:
                metrics['batch_token_counts'].append(batch_loss_tokens)
        
        # Calculate average
        if metrics['batch_token_counts']:
            metrics['avg_tokens_per_batch'] = sum(metrics['batch_token_counts']) / len(metrics['batch_token_counts'])
        
        logger.info(f"Dataset metrics: {metrics['num_batches']} batches, "
                   f"{metrics['total_samples']} samples, "
                   f"{metrics['total_loss_tokens']} total loss tokens, "
                   f"{metrics['avg_tokens_per_batch']:.1f} avg tokens/batch")
        
    except Exception as e:
        logger.error(f"Failed to get dataset metrics: {e}")
    
    return metrics
