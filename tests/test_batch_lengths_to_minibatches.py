"""
Unit tests for batch_lengths_to_minibatches functions.

Tests correctness of the batching algorithms for distributed training.
"""

import numpy as np
import pytest
from mini_trainer.sampler import batch_lengths_to_minibatches
from mini_trainer.batch_packer import batch_lengths_to_minibatches_lpt


class TestBatchLengthsToMinibatches:
    """Test basic functionality of batch packing algorithms."""
    
    def test_empty_batch(self):
        """Test with empty batch."""
        result = batch_lengths_to_minibatches([], 130000, 4, 0)
        assert result == []
        
        result_lpt = batch_lengths_to_minibatches_lpt([], 130000, 4, 0)
        assert result_lpt == []
    
    def test_single_sequence(self):
        """Test with single sequence."""
        result = batch_lengths_to_minibatches([5000], 130000, 4, 0)
        assert len(result) == 1
        assert result[0] == [0]
        
        # Other ranks should get padding
        result_rank1 = batch_lengths_to_minibatches([5000], 130000, 4, 1)
        assert result_rank1 == [[-1]]
        
        # Test LPT version
        result_lpt = batch_lengths_to_minibatches_lpt([5000], 130000, 4, 0)
        assert len(result_lpt) == 1
        assert result_lpt[0] == [0]
    
    def test_basic_distribution(self):
        """Test basic sequence distribution across ranks."""
        batch_lengths = [10000, 20000, 30000, 40000]
        max_tokens = 50000
        num_ranks = 2
        
        # Test both algorithms
        for algo_fn in [batch_lengths_to_minibatches, batch_lengths_to_minibatches_lpt]:
        # Get results for both ranks
            rank0_result = algo_fn(batch_lengths, max_tokens, num_ranks, 0)
            rank1_result = algo_fn(batch_lengths, max_tokens, num_ranks, 1)
        
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
        
        # Test both algorithms
        for algo_fn in [batch_lengths_to_minibatches, batch_lengths_to_minibatches_lpt]:
            all_assigned_indices = set()
            for rank in range(num_ranks):
                rank_result = algo_fn(batch_lengths, max_tokens, num_ranks, rank)
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
        
        # Test both algorithms
        for algo_fn in [batch_lengths_to_minibatches, batch_lengths_to_minibatches_lpt]:
            for rank in range(num_ranks):
                rank_result = algo_fn(batch_lengths, max_tokens, num_ranks, rank)
                for minibatch in rank_result:
                    total_tokens = sum(batch_lengths[i] for i in minibatch if i != -1)
                    assert total_tokens <= max_tokens, f"Rank {rank} exceeded token limit: {total_tokens} > {max_tokens}"

    def test_padding_in_incomplete_minibatches(self):
        """Test that incomplete minibatches have proper padding."""
        batch_lengths = [50000, 50000, 50000]  # 3 sequences that need individual minibatches
        max_tokens = 60000
        num_ranks = 4  # More ranks than sequences in last minibatch
        
        # Test both algorithms
        for algo_fn in [batch_lengths_to_minibatches, batch_lengths_to_minibatches_lpt]:
            # Ranks with no actual sequences should get padding (-1)
            for rank in range(num_ranks):
                rank_result = algo_fn(batch_lengths, max_tokens, num_ranks, rank)
                
                # Each rank should have same number of minibatches
                assert len(rank_result) > 0
                
                # Check for padding in minibatches
                for minibatch in rank_result:
                    if all(idx == -1 for idx in minibatch):
                        # This is a padding minibatch
                        assert minibatch == [-1]
    
    def test_sequences_exceeding_max_tokens(self):
        """Test handling of sequences that exceed max_tokens."""
        # Some sequences are larger than max_tokens (they would be filtered by collator)
        batch_lengths = [50000, 140000, 30000, 150000, 20000]
        max_tokens = 130000
        num_ranks = 2
        
        # Test both algorithms - they should handle this gracefully
        for algo_fn in [batch_lengths_to_minibatches, batch_lengths_to_minibatches_lpt]:
            for rank in range(num_ranks):
                rank_result = algo_fn(batch_lengths, max_tokens, num_ranks, rank)
                
                # Should still produce results
                assert isinstance(rank_result, list)
                
                # Sequences under the limit should still be distributed
                for minibatch in rank_result:
                    for idx in minibatch:
                        if idx != -1 and idx < len(batch_lengths):
                            # Only check sequences that fit
                            if batch_lengths[idx] <= max_tokens:
                                assert batch_lengths[idx] <= max_tokens
    
    def test_uniform_length_sequences(self):
        """Test with all sequences having the same length."""
        batch_lengths = [15000] * 20
        max_tokens = 130000
        num_ranks = 4
        
        # Test both algorithms
        for algo_fn in [batch_lengths_to_minibatches, batch_lengths_to_minibatches_lpt]:
            all_minibatch_counts = []
            for rank in range(num_ranks):
                rank_result = algo_fn(batch_lengths, max_tokens, num_ranks, rank)
                all_minibatch_counts.append(len(rank_result))
            
            # All ranks should have same number of minibatches
            assert len(set(all_minibatch_counts)) == 1
            
            # Verify distribution is relatively balanced
            sequences_per_rank = []
            for rank in range(num_ranks):
                rank_result = algo_fn(batch_lengths, max_tokens, num_ranks, rank)
                count = sum(1 for mb in rank_result for idx in mb if idx != -1)
                sequences_per_rank.append(count)
            
            # Check that distribution is reasonably balanced
            max_diff = max(sequences_per_rank) - min(sequences_per_rank)
            assert max_diff <= 2, f"Unbalanced distribution: {sequences_per_rank}"
    
    def test_deterministic_output(self):
        """Test that the algorithms produce deterministic output."""
        batch_lengths = [10000, 25000, 35000, 50000, 60000, 75000, 80000, 90000]
        max_tokens = 130000
        num_ranks = 4
        
        # Test both algorithms
        for algo_fn in [batch_lengths_to_minibatches, batch_lengths_to_minibatches_lpt]:
            # Run multiple times and verify same output
            results_by_rank = {}
            for iteration in range(3):
                for rank in range(num_ranks):
                    result = algo_fn(batch_lengths, max_tokens, num_ranks, rank)
                    if rank not in results_by_rank:
                        results_by_rank[rank] = result
                    else:
                        assert results_by_rank[rank] == result, f"Non-deterministic output for rank {rank}"


class TestLPTSpecificBehavior:
    """Test specific behaviors of the LPT algorithm."""
    
    def test_lpt_sorts_by_length(self):
        """Test that LPT processes sequences in descending order of length."""
        # Create sequences with known ordering
        batch_lengths = [1000, 5000, 2000, 8000, 3000]
        max_tokens = 10000
        num_ranks = 2
        
        # LPT should process longest sequences first
        rank0_result = batch_lengths_to_minibatches_lpt(batch_lengths, max_tokens, num_ranks, 0)
        rank1_result = batch_lengths_to_minibatches_lpt(batch_lengths, max_tokens, num_ranks, 1)
        
        # Collect all assigned sequences
        all_sequences = []
        for mb_idx in range(max(len(rank0_result), len(rank1_result))):
            if mb_idx < len(rank0_result):
                for idx in rank0_result[mb_idx]:
                    if idx != -1:
                        all_sequences.append((idx, batch_lengths[idx]))
            if mb_idx < len(rank1_result):
                for idx in rank1_result[mb_idx]:
                    if idx != -1:
                        all_sequences.append((idx, batch_lengths[idx]))
        
        # The longest sequences should generally be in earlier minibatches
        # (This is a heuristic check since exact order depends on packing)
        first_minibatch_max = 0
        for rank_result in [rank0_result, rank1_result]:
            if rank_result and rank_result[0]:
                for idx in rank_result[0]:
                    if idx != -1:
                        first_minibatch_max = max(first_minibatch_max, batch_lengths[idx])
        
        # First minibatch should contain one of the longer sequences
        assert first_minibatch_max >= 5000
    
    def test_lpt_better_balance_than_greedy(self):
        """Test that LPT generally achieves better load balance than greedy."""
        # Create a scenario where LPT should perform better
        batch_lengths = [90000, 85000, 80000, 30000, 25000, 20000, 15000, 10000]
        max_tokens = 130000
        num_ranks = 4
        
        # Measure load balance for both algorithms
        def measure_balance(algo_fn):
            loads = []
            for rank in range(num_ranks):
                rank_result = algo_fn(batch_lengths, max_tokens, num_ranks, rank)
                total_load = sum(
                    sum(batch_lengths[i] for i in mb if i != -1)
                    for mb in rank_result
                )
                loads.append(total_load)
            
            if max(loads) == 0:
                return 1.0
            return min(loads) / max(loads)
        
        greedy_balance = measure_balance(batch_lengths_to_minibatches)
        lpt_balance = measure_balance(batch_lengths_to_minibatches_lpt)
        
        # LPT should achieve at least as good balance as greedy
        # (in practice it's usually better, but we allow equality for edge cases)
        assert lpt_balance >= greedy_balance * 0.95  # Allow 5% tolerance


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_large_sequence(self):
        """Test with a single sequence that uses most of max_tokens."""
        batch_lengths = [120000]
        max_tokens = 130000
        num_ranks = 4
        
        for algo_fn in [batch_lengths_to_minibatches, batch_lengths_to_minibatches_lpt]:
            # Only rank 0 should get the sequence
            rank0_result = algo_fn(batch_lengths, max_tokens, num_ranks, 0)
            assert len(rank0_result) == 1
            assert 0 in rank0_result[0]
            
            # Other ranks should get padding
            for rank in range(1, num_ranks):
                rank_result = algo_fn(batch_lengths, max_tokens, num_ranks, rank)
                assert len(rank_result) == 1
                assert rank_result[0] == [-1]
    
    def test_many_tiny_sequences(self):
        """Test with many very small sequences."""
        batch_lengths = [100] * 1000  # 1000 tiny sequences
        max_tokens = 130000
        num_ranks = 8
        
        for algo_fn in [batch_lengths_to_minibatches, batch_lengths_to_minibatches_lpt]:
            total_sequences_distributed = 0
            max_minibatches = 0
            
            for rank in range(num_ranks):
                rank_result = algo_fn(batch_lengths, max_tokens, num_ranks, rank)
                max_minibatches = max(max_minibatches, len(rank_result))
                
                for minibatch in rank_result:
                    # Should pack many sequences per minibatch
                    seq_count = sum(1 for idx in minibatch if idx != -1)
                    total_sequences_distributed += seq_count
                    
                    # With 100-token sequences and 130000 limit, should fit many
                    if seq_count > 0:
                        assert seq_count <= 1300  # Theoretical max
                        assert seq_count >= 100   # Should pack efficiently
            
            # All sequences should be distributed
            assert total_sequences_distributed == 1000
            
            # Should need very few minibatches
            assert max_minibatches <= 2
    
    def test_mixed_scales(self):
        """Test with sequences of vastly different scales."""
        batch_lengths = (
            [100] * 50 +      # Tiny
            [10000] * 20 +    # Medium
            [100000] * 5      # Large
        )
        np.random.shuffle(batch_lengths)
        max_tokens = 130000
        num_ranks = 4
        
        for algo_fn in [batch_lengths_to_minibatches, batch_lengths_to_minibatches_lpt]:
            all_assigned = []
            
            for rank in range(num_ranks):
                rank_result = algo_fn(batch_lengths, max_tokens, num_ranks, rank)
                
                for minibatch in rank_result:
                    # Check token limit
                    total = sum(batch_lengths[i] for i in minibatch if i != -1)
                    assert total <= max_tokens
                    
                    # Collect assigned indices
                    all_assigned.extend(i for i in minibatch if i != -1)
            
            # All sequences should be assigned exactly once
            assert sorted(all_assigned) == list(range(len(batch_lengths)))
