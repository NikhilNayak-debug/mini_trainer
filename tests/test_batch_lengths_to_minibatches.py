"""
Test suite for batch_lengths_to_minibatches function.

Tests both correctness and efficiency of the batching algorithm,
measuring load distribution and number of minibatches generated.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import time
from sampler import batch_lengths_to_minibatches
from batch_packer import batch_lengths_to_minibatches_lpt


class TestBatchLengthsToMinibatches:
    
    def test_empty_batch(self):
        """Test with empty batch."""
        result = batch_lengths_to_minibatches([], 130000, 4, 0)
        assert result == []
    
    def test_single_sequence(self):
        """Test with single sequence."""
        result = batch_lengths_to_minibatches([5000], 130000, 4, 0)
        assert len(result) == 1
        assert result[0] == [0]
        
        # Other ranks should get padding
        result_rank1 = batch_lengths_to_minibatches([5000], 130000, 4, 1)
        assert result_rank1 == [[-1]]
    
    def test_basic_distribution(self):
        """Test basic sequence distribution across ranks."""
        batch_lengths = [10000, 20000, 30000, 40000]
        max_tokens = 50000
        num_ranks = 2
        
        # Get results for both ranks
        rank0_result = batch_lengths_to_minibatches(batch_lengths, max_tokens, num_ranks, 0)
        rank1_result = batch_lengths_to_minibatches(batch_lengths, max_tokens, num_ranks, 1)
        
        # Should have same number of minibatches
        assert len(rank0_result) == len(rank1_result)
        
        # Check no sequence exceeds max_tokens per rank
        for minibatch in rank0_result:
            total_tokens = sum(batch_lengths[i] for i in minibatch if i != -1)
            assert total_tokens <= max_tokens
            
        for minibatch in rank1_result:
            total_tokens = sum(batch_lengths[i] for i in minibatch if i != -1)
            assert total_tokens <= max_tokens
    
    def test_no_duplicates_across_ranks(self):
        """Test that no sequence appears in multiple ranks."""
        batch_lengths = [10000, 20000, 30000, 40000, 50000]
        max_tokens = 60000
        num_ranks = 3
        
        all_assigned_indices = set()
        for rank in range(num_ranks):
            rank_result = batch_lengths_to_minibatches(batch_lengths, max_tokens, num_ranks, rank)
            for minibatch in rank_result:
                for idx in minibatch:
                    if idx != -1:
                        assert idx not in all_assigned_indices, f"Index {idx} assigned to multiple ranks"
                        all_assigned_indices.add(idx)
        
        # All non-padding indices should be covered
        expected_indices = set(range(len(batch_lengths)))
        assert all_assigned_indices == expected_indices
    
    def test_token_limit_enforcement(self):
        """Test that token limits are strictly enforced."""
        batch_lengths = [80000, 40000, 30000, 20000]
        max_tokens = 100000
        num_ranks = 2
        
        for rank in range(num_ranks):
            rank_result = batch_lengths_to_minibatches(batch_lengths, max_tokens, num_ranks, rank)
            for minibatch in rank_result:
                total_tokens = sum(batch_lengths[i] for i in minibatch if i != -1)
                assert total_tokens <= max_tokens, f"Rank {rank} exceeded token limit: {total_tokens} > {max_tokens}"

"""
NEED TO ADD: test the new batch_lengths_to_minibatches_lpt function
"""
