import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def decompose_weight_matrix(weight: torch.Tensor, top_k: int):
    """Decompose a 2D weight matrix with SVD splitting the top ``top_k`` singular
    components as frozen buffers and the rest as trainable parameters."""
    device_local = weight.device
    W = weight.to(torch.float32)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(top_k, S.shape[0])

    svd = {
        "U_high": U[:, :k].contiguous().detach().to(device=device_local),
        "S_high": S[:k].contiguous().detach().to(device=device_local),
        "V_high": Vt[:k, :].contiguous().detach().to(device=device_local),
        "U_low": nn.Parameter(U[:, k:].contiguous().detach().to(device=device_local)),
        "S_low": nn.Parameter(S[k:].contiguous().detach().to(device=device_local)),
        "V_low": nn.Parameter(Vt[k:, :].contiguous().detach().to(device=device_local)),
        "rank_high": k,
    }
    return svd


def reconstruct_weight_matrix(svd_dict):
    """Reconstruct the weight matrix from its SVD components."""
    U_high = svd_dict["U_high"]
    S_high = svd_dict["S_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    S_low = svd_dict["S_low"]
    V_low = svd_dict["V_low"]

    if U_high.numel() > 0 and S_high.numel() > 0:
        high_part = torch.mm(U_high * S_high.unsqueeze(0), V_high)
    else:
        high_part = torch.zeros(U_low.size(0), V_low.size(1), device=U_high.device)

    if U_low.numel() > 0 and S_low.numel() > 0:
        low_part = torch.mm(U_low * S_low.unsqueeze(0), V_low)
    else:
        low_part = torch.zeros(U_high.size(0), V_high.size(1), device=U_low.device)

    return high_part + low_part


def project_gradient_to_orthogonal_space(svd_dict):
    """Project gradients of the low-rank subspace to be orthogonal to the high
    subspace."""
    if (
        svd_dict["U_low"].grad is None
        and svd_dict["S_low"].grad is None
        and svd_dict["V_low"].grad is None
    ):
        return

    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]

    if svd_dict["U_low"].grad is not None:
        dU = svd_dict["U_low"].grad
        # Support distributed tensors by operating on the local shard
        local_U_high = getattr(U_high, "to_local", lambda: U_high)()
        local_dU = getattr(dU, "to_local", lambda: dU)()
        if local_U_high.size(0) != local_dU.size(0):
            rank = torch.distributed.get_rank()
            start = rank * local_dU.size(0)
            end = start + local_dU.size(0)
            local_U_high = local_U_high[start:end]
        proj = local_U_high @ (local_U_high.transpose(0, 1) @ local_dU)
        local_dU.sub_(proj)
        if hasattr(dU, "_local_tensor"):
            dU._local_tensor.copy_(local_dU)
        else:
            dU.copy_(local_dU)

    if svd_dict["V_low"].grad is not None:
        dV = svd_dict["V_low"].grad
        local_V_high = getattr(V_high, "to_local", lambda: V_high)()
        local_dV = getattr(dV, "to_local", lambda: dV)()
        if local_V_high.size(1) != local_dV.size(1):
            rank = torch.distributed.get_rank()
            start = rank * local_dV.size(1)
            end = start + local_dV.size(1)
            local_V_high = local_V_high[:, start:end]
        proj = (local_dV @ local_V_high.transpose(0, 1)) @ local_V_high
        local_dV.sub_(proj)
        if hasattr(dV, "_local_tensor"):
            dV._local_tensor.copy_(local_dV)
        else:
            dV.copy_(local_dV)


def auto_generate_target_svd_config(model):
    """Generate an SVD configuration for attention and MLP projection weights."""
    target_patterns = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj",
    ]
    config = {}
    for name, param in model.named_parameters():
        if any(pat in name for pat in target_patterns) and len(param.shape) == 2:
            top_k = int(np.floor(min(param.shape) * 0.5))
            full_rank = min(param.shape)
            if top_k >= full_rank:
                top_k = full_rank - 1
            config[name] = top_k
    return config


def create_svd_model_class(base_cls):
    """Dynamically create a subclass of ``base_cls`` that performs SVD based
    decomposition on selected linear weights."""

    class ModelWithSVD(base_cls):
        def __init__(self, config, svd_config=None, initialize_svd=True, **kwargs):
            super().__init__(config, **kwargs)
            self.svd_config = svd_config or {}
            self.name_mapping = {}
            self.svd_params = nn.ModuleDict()
            if initialize_svd:
                self._initialize_svd_parameters()

        def reinitialize_svd(self):
            self.name_mapping = {}
            self.svd_params = nn.ModuleDict()
            self._initialize_svd_parameters()
        def _get_module_by_name(self, name):
            parts = name.split(".")
            attr = parts[-1]
            mod = self
            for p in parts[:-1]:
                if hasattr(mod, p):
                    mod = getattr(mod, p)
                elif p.isdigit():
                    mod = mod[int(p)]
                else:
                    return None, None
            return mod, attr

        def _initialize_svd_parameters(self):
            for name, param in list(self.named_parameters()):
                if len(param.shape) == 2 and name in self.svd_config and self.svd_config[name] > 0:
                    top_k = self.svd_config[name]
                    print(f"[SVD Init] Decomposing {name} with top_k={top_k}")
                    svd_dict = decompose_weight_matrix(param.data, top_k=top_k)
                    safe_name = name.replace(".", "_")
                    self.name_mapping[name] = safe_name

                    self.register_buffer(f"{safe_name}_U_high", svd_dict["U_high"])
                    self.register_buffer(f"{safe_name}_S_high", svd_dict["S_high"])
                    self.register_buffer(f"{safe_name}_V_high", svd_dict["V_high"])

                    module_svd = nn.Module()
                    module_svd.U_low = svd_dict["U_low"]
                    module_svd.S_low = svd_dict["S_low"]
                    module_svd.V_low = svd_dict["V_low"]
                    module_svd.rank_high = svd_dict["rank_high"]
                    module_svd.safe_name = safe_name
                    self.svd_params[safe_name] = module_svd

                    mod, attr = self._get_module_by_name(name)
                    bias = mod.bias if hasattr(mod, "bias") else None

                    def make_forward(sn, bias):
                        def forward(x):
                            W = self._reconstruct_weight_by_safe_name(sn)
                            if W.dtype != x.dtype:
                                W = W.to(x.dtype)
                            return F.linear(x, W, bias)
                        return forward

                    mod.forward = make_forward(safe_name, bias)
                    param.requires_grad = False
                    mod._parameters.pop(attr, None)

        def _reconstruct_weight_by_safe_name(self, safe_name):
            U_high = getattr(self, f"{safe_name}_U_high")
            S_high = getattr(self, f"{safe_name}_S_high")
            V_high = getattr(self, f"{safe_name}_V_high")
            module_svd = self.svd_params[safe_name]
            svd_dict = {
                "U_high": U_high,
                "S_high": S_high,
                "V_high": V_high,
                "U_low": module_svd.U_low,
                "S_low": module_svd.S_low,
                "V_low": module_svd.V_low,
            }
            return reconstruct_weight_matrix(svd_dict)

        def _reconstruct_weight(self, original_name):
            return self._reconstruct_weight_by_safe_name(self.name_mapping[original_name])

        def project_gradients(self):
            for safe_name, module_svd in self.svd_params.items():
                svd_dict = {
                    "U_high": getattr(self, f"{safe_name}_U_high"),
                    "S_high": getattr(self, f"{safe_name}_S_high"),
                    "V_high": getattr(self, f"{safe_name}_V_high"),
                    "U_low": module_svd.U_low,
                    "S_low": module_svd.S_low,
                    "V_low": module_svd.V_low,
                }
                project_gradient_to_orthogonal_space(svd_dict)

    ModelWithSVD.__name__ = f"{base_cls.__name__}WithSVD"
    return ModelWithSVD

