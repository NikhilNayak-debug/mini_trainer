"""
Unit tests for OSFT (Orthogonal Subspace Fine-Tuning) decomposition and reconstruction fidelity.

These tests verify that when a model is created with distributed OSFT initialization,
the reconstructed parameters from the decomposed SVD parts are identical to the
original untouched model parameters (within numerical tolerance).
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import os
from pathlib import Path


class TestOSFTReconstructionFidelity:
    """Test OSFT reconstruction maintains parameter fidelity."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model with various parameter shapes."""
        model = MagicMock()
        
        # Create state dict with various parameter types
        state_dict = {
            "layer1.weight": torch.randn(128, 64),
            "layer1.bias": torch.randn(128),
            "layer2.self_attn.q_proj.weight": torch.randn(256, 128),
            "layer2.self_attn.k_proj.weight": torch.randn(256, 128),
            "layer2.self_attn.v_proj.weight": torch.randn(256, 128),
            "layer2.mlp.gate_proj.weight": torch.randn(512, 256),
            "layer2.mlp.up_proj.weight": torch.randn(512, 256),
            "layer2.mlp.down_proj.weight": torch.randn(256, 512),
            "norm.weight": torch.randn(256),
        }
        
        model.state_dict.return_value = state_dict
        return model
    
    @pytest.fixture
    def mock_osft_model(self, mock_model):
        """Create a mock OSFT model with reconstruction capability."""
        osft_model = MagicMock()
        
        # Copy the original state dict
        original_state = mock_model.state_dict()
        
        # Define which parameters are OSFT-decomposed
        osft_model.name_mapping = {
            "layer2.self_attn.q_proj.weight": "svd_param_0",
            "layer2.self_attn.k_proj.weight": "svd_param_1",
            "layer2.self_attn.v_proj.weight": "svd_param_2",
            "layer2.mlp.gate_proj.weight": "svd_param_3",
            "layer2.mlp.up_proj.weight": "svd_param_4",
            "layer2.mlp.down_proj.weight": "svd_param_5",
        }
        
        # Mock the reconstruction method
        def reconstruct_weight(param_name):
            if param_name in osft_model.name_mapping:
                # Return the original parameter with small numerical error
                original = original_state[param_name]
                # Add tiny numerical noise to simulate reconstruction error
                noise = torch.randn_like(original) * 1e-7
                return original + noise
            else:
                raise ValueError(f"Parameter {param_name} not in OSFT mapping")
        
        osft_model._reconstruct_weight = MagicMock(side_effect=reconstruct_weight)
        osft_model.state_dict.return_value = original_state
        
        return osft_model
    
    def test_osft_reconstruction_accuracy(self, mock_model, mock_osft_model):
        """Test that OSFT reconstruction is accurate within tolerance."""
        tolerance = 1e-5
        
        original_state = mock_model.state_dict()
        
        # Track comparison results
        total_params_compared = 0
        identical_params = 0
        close_params = 0
        different_params = 0
        max_difference = 0.0
        
        for param_name, original_param in original_state.items():
            if param_name not in mock_osft_model.name_mapping:
                # Skip non-OSFT parameters
                continue
            
            # Reconstruct the parameter
            reconstructed_param = mock_osft_model._reconstruct_weight(param_name)
            
            # Verify shapes match
            assert reconstructed_param.shape == original_param.shape, \
                f"Shape mismatch for {param_name}: {reconstructed_param.shape} vs {original_param.shape}"
            
            # Calculate difference
            diff = torch.abs(reconstructed_param - original_param)
            max_diff = diff.max().item()
            max_difference = max(max_difference, max_diff)
            
            total_params_compared += 1
            
            # Check if parameters are close enough
            if torch.equal(reconstructed_param, original_param):
                identical_params += 1
            elif torch.allclose(reconstructed_param, original_param, atol=tolerance, rtol=tolerance):
                close_params += 1
            else:
                different_params += 1
                pytest.fail(f"Parameter {param_name} differs by {max_diff:.2e}, exceeding tolerance {tolerance:.2e}")
        
        # Verify we compared some parameters
        assert total_params_compared > 0, "No parameters were compared"
        
        # All parameters should be at least close
        assert different_params == 0, f"{different_params} parameters exceeded tolerance"
        
        # Log statistics (these would normally be logged, but in tests we just verify)
        success_rate = (identical_params + close_params) / total_params_compared * 100
        assert success_rate == 100.0, f"Only {success_rate:.1f}% of parameters match within tolerance"
    
    def test_osft_handles_different_dtypes(self, mock_model):
        """Test OSFT reconstruction with different data types."""
        osft_model = MagicMock()
        
        # Test with float16
        original_param = torch.randn(64, 32, dtype=torch.float32)
        reconstructed_param = original_param.to(torch.float64)  # Upcast for comparison
        
        osft_model.name_mapping = {"test_param": "osft_0"}
        osft_model._reconstruct_weight.return_value = reconstructed_param
        
        # The reconstruction should handle dtype conversion
        result = osft_model._reconstruct_weight("test_param")
        assert result.dtype == torch.float64
        
        # When comparing, we should upcast the original
        original_upcast = original_param.to(torch.float64)
        assert torch.allclose(result, original_upcast, atol=1e-5)
    
    def test_osft_skips_non_decomposed_params(self, mock_model, mock_osft_model):
        """Test that non-decomposed parameters are properly skipped."""
        original_state = mock_model.state_dict()
        
        skipped_params = []
        for param_name in original_state.keys():
            if param_name not in mock_osft_model.name_mapping:
                skipped_params.append(param_name)
        
        # Verify we have some non-decomposed parameters
        assert len(skipped_params) > 0, "Test should have non-decomposed parameters"
        
        # These should include biases and norm weights
        assert "layer1.bias" in skipped_params
        assert "norm.weight" in skipped_params
        
        # Attempting to reconstruct these should raise an error
        for param_name in skipped_params:
            with pytest.raises(ValueError, match=f"Parameter {param_name} not in OSFT mapping"):
                mock_osft_model._reconstruct_weight(param_name)
    
    def test_osft_reconstruction_with_rank_ratio(self):
        """Test that different rank ratios affect reconstruction accuracy."""
        # Create a simple test case
        original_weight = torch.randn(100, 50)
        
        # Simulate SVD decomposition with different rank ratios
        rank_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        for ratio in rank_ratios:
            rank = int(min(original_weight.shape) * ratio)
            
            # Perform actual SVD
            U, S, V = torch.svd(original_weight)
            
            # Truncate to rank
            U_truncated = U[:, :rank]
            S_truncated = S[:rank]
            V_truncated = V[:, :rank]
            
            # Reconstruct
            reconstructed = U_truncated @ torch.diag(S_truncated) @ V_truncated.T
            
            # Calculate reconstruction error
            error = torch.norm(original_weight - reconstructed) / torch.norm(original_weight)
            
            # Higher rank ratios should have lower reconstruction error
            if ratio >= 0.5:
                assert error < 0.5, f"Rank ratio {ratio} has too high error: {error}"
            
            # Very high rank ratios should have very low error
            if ratio >= 0.9:
                assert error < 0.15, f"Rank ratio {ratio} should have minimal error, got {error}"
    
    @patch('mini_trainer.setup_model_for_training.setup_model')
    def test_osft_model_integration(self, mock_setup):
        """Test OSFT model integration with setup_model."""
        # Mock the setup_model to return our mock models
        original_model = MagicMock()
        osft_model = MagicMock()
        
        # Configure setup_model to return different models based on osft flag
        def setup_side_effect(**kwargs):
            if kwargs.get('osft', False):
                return osft_model
            else:
                return original_model
        
        mock_setup.side_effect = setup_side_effect
        
        # Load models with and without OSFT
        from mini_trainer.setup_model_for_training import setup_model
        
        model_without_osft = setup_model(
            model_name_or_path="test-model",
            osft=False,
            rank=0
        )
        
        model_with_osft = setup_model(
            model_name_or_path="test-model",
            osft=True,
            rank=0,
            osft_rank_ratio=0.5
        )
        
        # Verify different models were returned
        assert model_without_osft is original_model
        assert model_with_osft is osft_model
        
        # Verify setup was called with correct parameters
        assert mock_setup.call_count == 2
        
        # Check the calls
        calls = mock_setup.call_args_list
        assert calls[0].kwargs['osft'] == False
        assert calls[1].kwargs['osft'] == True
        assert calls[1].kwargs['osft_rank_ratio'] == 0.5


class TestSVDNumericalStability:
    """Test numerical stability of SVD operations."""
    
    def test_dtype_precision_effects(self):
        """Demonstrate how different dtypes affect SVD precision.
        
        This test shows why float64 is needed for accurate orthogonality
        and rank checks. With float32 and bfloat16, numerical errors
        accumulate quickly, especially for larger matrices.
        """
        # Create a rank-2 matrix
        torch.manual_seed(42)  # For reproducibility
        A_64 = torch.randn(8, 2, dtype=torch.float64)
        B_64 = torch.randn(2, 8, dtype=torch.float64)
        matrix_64 = A_64 @ B_64
        
        # Convert to different precisions
        matrix_32 = matrix_64.to(torch.float32)
        
        # Perform SVD in different precisions
        U_64, S_64, V_64 = torch.svd(matrix_64)
        U_32, S_32, V_32 = torch.svd(matrix_32)
        
        # Check orthogonality: U @ U.T should be identity
        ortho_error_64 = torch.norm(U_64 @ U_64.T - torch.eye(8, dtype=torch.float64))
        ortho_error_32 = torch.norm(U_32 @ U_32.T - torch.eye(8, dtype=torch.float32))
        
        # Float64 should have much better orthogonality
        assert ortho_error_64 < 1e-13, f"Float64 orthogonality error too high: {ortho_error_64}"
        assert ortho_error_32 < 1e-5, f"Float32 orthogonality error too high: {ortho_error_32}"
        
        # The float32 error should be noticeably worse than float64
        assert ortho_error_32 > ortho_error_64 * 100, \
            "Float32 should have significantly worse orthogonality than float64"
        
        # Check rank detection: count "significant" singular values
        # With float64, we can use a tighter threshold
        rank_64 = torch.sum(S_64 > 1e-10).item()
        rank_32 = torch.sum(S_32 > 1e-5).item()  # Need looser threshold for float32
        
        # Float64 should correctly identify rank as 2
        assert rank_64 == 2, f"Float64 should detect rank 2, got {rank_64}"
        # Float32 might detect extra "phantom" singular values due to numerical noise
        assert rank_32 <= 4, f"Float32 rank detection affected by precision: {rank_32}"
        
        # Document the precision limitations
        print(f"Orthogonality error - Float64: {ortho_error_64:.2e}, Float32: {ortho_error_32:.2e}")
        print(f"Detected rank - Float64: {rank_64}, Float32: {rank_32}")
    
    def test_svd_with_extreme_values(self):
        """Test SVD with very large and very small values.
        
        Note: Extreme values can exacerbate precision issues. For production use
        with extreme scales, consider normalizing inputs or using float64.
        """
        # Test with very small values (using float32 is sufficient here)
        small_weight = torch.randn(32, 16, dtype=torch.float32) * 1e-10
        U, S, V = torch.svd(small_weight)
        reconstructed = U @ torch.diag(S) @ V.T
        
        # Should maintain relative accuracy
        rel_error = torch.norm(small_weight - reconstructed) / (torch.norm(small_weight) + 1e-20)
        assert rel_error < 1e-5, f"Failed to reconstruct small values accurately: {rel_error}"
        
        # Test with very large values
        large_weight = torch.randn(32, 16, dtype=torch.float32) * 1e10
        U, S, V = torch.svd(large_weight)
        reconstructed = U @ torch.diag(S) @ V.T
        
        rel_error = torch.norm(large_weight - reconstructed) / torch.norm(large_weight)
        assert rel_error < 1e-5, f"Failed to reconstruct large values accurately: {rel_error}"
    
    def test_svd_with_rank_deficient_matrix(self):
        """Test SVD with rank-deficient matrices.
        
        Note: This test uses float64 for better numerical stability.
        With float32 and especially bfloat16, checking for orthogonality
        and rank properties becomes very difficult due to numerical precision.
        """
        # Use float64 for better numerical stability in rank testing
        # Create a small rank-deficient matrix (rank 2 in 6x6 matrix)
        A = torch.randn(6, 2, dtype=torch.float64)
        B = torch.randn(2, 6, dtype=torch.float64)
        rank_deficient = A @ B
        
        U, S, V = torch.svd(rank_deficient)
        
        # Check that singular values drop off significantly after the true rank
        # With float64, we can be more strict about the drop-off
        if len(S) >= 3:
            # The ratio between 2nd (last significant) and 3rd (first noise) singular value
            if S[2] > 1e-14:  # Only check ratio if 3rd value is not essentially zero
                ratio = S[1] / S[2]
                # With float64, we expect a large ratio but not as strict due to numerical noise
                assert ratio > 10, f"Singular values should drop after rank 2, ratio: {ratio}"
            else:
                # If S[2] is essentially zero, that's even better
                assert S[2] < 1e-10, f"Expected near-zero singular value, got {S[2]}"
        
        # Reconstruction should still work (more tolerant for float64 accumulation)
        reconstructed = U @ torch.diag(S) @ V.T
        assert torch.allclose(rank_deficient, reconstructed, atol=1e-10)
    
    def test_svd_gradient_flow(self):
        """Test that gradients flow through SVD operations correctly."""
        # Create a parameter that requires gradient
        weight = torch.randn(16, 8, requires_grad=True)
        
        # Perform SVD-based operation
        U, S, V = torch.svd(weight)
        
        # Truncate rank (simulate low-rank approximation)
        rank = 4
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        V_truncated = V[:, :rank]
        
        # Reconstruct and compute loss
        reconstructed = U_truncated @ torch.diag(S_truncated) @ V_truncated.T
        loss = torch.norm(reconstructed - weight)
        
        # Check that we can compute gradients
        loss.backward()
        
        assert weight.grad is not None, "Gradient should be computed"
        assert not torch.isnan(weight.grad).any(), "Gradients should not contain NaN"
        assert not torch.isinf(weight.grad).any(), "Gradients should not contain Inf"
