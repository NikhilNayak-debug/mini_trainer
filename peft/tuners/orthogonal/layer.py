import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import decompose_weight_matrix, reconstruct_weight_matrix, project_gradient_to_orthogonal_space


class OSLLinear(nn.Module):
    """Linear layer with orthogonal subspace learning via SVD decomposition."""

    def __init__(self, base_layer: nn.Linear, top_k: int):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        svd = decompose_weight_matrix(base_layer.weight.data, top_k)
        self.register_buffer("U_high", svd["U_high"])
        self.register_buffer("S_high", svd["S_high"])
        self.register_buffer("V_high", svd["V_high"])
        self.U_low = svd["U_low"]
        self.S_low = svd["S_low"]
        self.V_low = svd["V_low"]
        self.bias = base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = reconstruct_weight_matrix(
            {
                "U_high": self.U_high,
                "S_high": self.S_high,
                "V_high": self.V_high,
                "U_low": self.U_low,
                "S_low": self.S_low,
                "V_low": self.V_low,
            }
        )
        if W.dtype != x.dtype:
            W = W.to(x.dtype)
        return F.linear(x, W, self.bias)

    def project_gradients(self):
        project_gradient_to_orthogonal_space(
            {
                "U_high": self.U_high,
                "S_high": self.S_high,
                "V_high": self.V_high,
                "U_low": self.U_low,
                "S_low": self.S_low,
                "V_low": self.V_low,
            }
        )
