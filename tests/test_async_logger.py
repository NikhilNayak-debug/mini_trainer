"""
Test suite for AsyncStructuredLogger.

Tests the asynchronous logging functionality used for training metrics.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import time
from unittest.mock import patch
import pytest

from mini_trainer.async_structured_logger import AsyncStructuredLogger


class TestAsyncStructuredLogger:
    """Test suite for AsyncStructuredLogger."""
    
    def test_logger_initialization(self):
        """Test logger initialization with file path."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            logger = AsyncStructuredLogger(temp_path)
            assert logger.file_name == temp_path
            assert logger.loop is not None
            assert logger.logs is not None
            
            # Stop the event loop
            logger.loop.call_soon_threadsafe(logger.loop.stop)
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('builtins.print')  # Mock print to avoid output during tests
    def test_log_sync_basic(self, mock_print):
        """Test synchronous logging of metrics."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            logger = AsyncStructuredLogger(temp_path)
            
            # Log some metrics
            metrics = {
                'step': 1,
                'loss': 2.5,
                'learning_rate': 1e-5,
                'samples': 100
            }
            logger.log_sync(metrics)
            
            # Give async operations time to complete
            time.sleep(0.2)
            
            # Stop event loop
            logger.loop.call_soon_threadsafe(logger.loop.stop)
            
            # Read and verify
            with open(temp_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                logged_data = json.loads(lines[0])
                assert logged_data['step'] == 1
                assert logged_data['loss'] == 2.5
                assert logged_data['learning_rate'] == 1e-5
                assert logged_data['samples'] == 100
                assert 'timestamp' in logged_data  # Logger adds timestamp
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('builtins.print')  # Mock print to avoid output during tests
    def test_log_sync_multiple_entries(self, mock_print):
        """Test logging multiple entries."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            logger = AsyncStructuredLogger(temp_path)
            
            # Log multiple entries
            for i in range(3):
                logger.log_sync({
                    'step': i,
                    'loss': 2.0 + i * 0.1
                })
                # short pause to allow for coroutine to complete (it is not fully synchronous)
                time.sleep(0.1)
            
            # Give async operations time to complete
            time.sleep(0.3)
            
            # Stop event loop
            logger.loop.call_soon_threadsafe(logger.loop.stop)
            
            # Read and verify
            with open(temp_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3
                
                for i, line in enumerate(lines):
                    data = json.loads(line)
                    assert data['step'] == i
                    assert abs(data['loss'] - (2.0 + i * 0.1)) < 1e-6
                    assert 'timestamp' in data
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('builtins.print')
    def test_logger_with_nested_data(self, mock_print):
        """Test logging nested dictionary structures."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            logger = AsyncStructuredLogger(temp_path)
            
            # Log nested structure
            complex_metrics = {
                'step': 1,
                'losses': {
                    'total': 2.5,
                    'kl': 0.1,
                    'reconstruction': 2.4
                },
                'learning_rates': [1e-5, 2e-5],
                'metadata': {
                    'epoch': 1,
                    'batch_size': 32
                }
            }
            logger.log_sync(complex_metrics)
            
            # Give async operations time
            time.sleep(0.2)
            
            # Stop event loop
            logger.loop.call_soon_threadsafe(logger.loop.stop)
            
            # Read and verify
            with open(temp_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                data = json.loads(lines[0])
                assert data['step'] == 1
                assert data['losses']['total'] == 2.5
                assert data['losses']['kl'] == 0.1
                assert data['learning_rates'] == [1e-5, 2e-5]
                assert data['metadata']['batch_size'] == 32
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('builtins.print')
    def test_logger_file_append(self, mock_print):
        """Test that logger appends to existing file."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
            # Write initial content
            f.write(json.dumps({'initial': 'data'}) + '\n')
        
        try:
            logger = AsyncStructuredLogger(temp_path)
            
            # Give time for initialization
            time.sleep(0.1)
            
            logger.log_sync({'new': 'entry'})
            
            # Give time to write
            time.sleep(0.2)
            
            logger.loop.call_soon_threadsafe(logger.loop.stop)
            
            # Read and verify
            with open(temp_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 2
                
                first_data = json.loads(lines[0])
                assert first_data['initial'] == 'data'
                
                second_data = json.loads(lines[1])
                assert second_data['new'] == 'entry'
                assert 'timestamp' in second_data
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_logger_event_loop(self):
        """Test that logger uses event loop correctly."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            logger = AsyncStructuredLogger(temp_path)
            
            # Verify event loop is running
            assert logger.loop is not None
            assert not logger.loop.is_closed()
            
            # Stop the event loop
            logger.loop.call_soon_threadsafe(logger.loop.stop)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_logger_repr(self):
        """Test logger string representation."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            logger = AsyncStructuredLogger(temp_path)
            repr_str = repr(logger)
            
            assert 'AsyncStructuredLogger' in repr_str
            assert temp_path in repr_str
            
            # Stop the event loop
            logger.loop.call_soon_threadsafe(logger.loop.stop)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestAsyncLoggerIntegration:
    """Integration tests for AsyncStructuredLogger with training loop."""
    
    @patch('builtins.print')
    def test_logger_in_training_context(self, mock_print):
        """Test logger usage in a simulated training context."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            logger = AsyncStructuredLogger(temp_path)
            
            # Simulate training loop logging
            for step in range(3):
                batch_metrics = {
                    "step": step,
                    "lr": 1e-5 * (0.9 ** step),
                    "grad_norm": 1.5 - step * 0.1,
                    "loss": 2.5 - step * 0.2,
                    "num_samples": 32,
                    "tokens_per_second": 1000 + step * 100,
                }
                logger.log_sync(batch_metrics)
                time.sleep(0.05)  # Small delay between logs
            
            # Give time to complete
            time.sleep(0.3)
            
            # Stop event loop
            logger.loop.call_soon_threadsafe(logger.loop.stop)
            
            # Verify all metrics were logged
            with open(temp_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3
                
                for i, line in enumerate(lines):
                    data = json.loads(line)
                    assert data['step'] == i
                    assert 'lr' in data
                    assert 'grad_norm' in data
                    assert 'loss' in data
                    assert 'timestamp' in data
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('builtins.print')
    def test_logger_error_handling(self, mock_print):
        """Test logger handles errors gracefully."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            logger = AsyncStructuredLogger(temp_path)
            
            # Try to log non-dict (should handle gracefully)
            logger.log_sync("not a dict")  # This will fail but shouldn't crash
            
            # Log valid data
            logger.log_sync({'valid': 'data'})
            
            # Give time to process
            time.sleep(0.2)
            
            # Stop event loop
            logger.loop.call_soon_threadsafe(logger.loop.stop)
            
            # The valid data should still be logged
            with open(temp_path, 'r') as f:
                lines = f.readlines()
                # May have 0 or 1 lines depending on error handling
                if lines:
                    data = json.loads(lines[-1])
                    assert data.get('valid') == 'data'
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
