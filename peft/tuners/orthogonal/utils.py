import torch
import torch.nn as nn


def decompose_weight_matrix(weight: torch.Tensor, top_k: int):
    device_local = weight.device
    orig_dtype = weight.dtype
    W = weight.to(torch.float32)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(top_k, S.shape[0])
    svd = {
        "U_high": U[:, :k].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "S_high": S[:k].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "V_high": Vt[:k, :].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "U_low": nn.Parameter(U[:, k:].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "S_low": nn.Parameter(S[k:].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "V_low": nn.Parameter(Vt[k:, :].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
    }
    return svd


def reconstruct_weight_matrix(svd_dict):
    U_high = svd_dict["U_high"]
    S_high = svd_dict["S_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    S_low = svd_dict["S_low"]
    V_low = svd_dict["V_low"]

    high_part = torch.mm(U_high * S_high.unsqueeze(0), V_high) if S_high.numel() > 0 else 0
    low_part = torch.mm(U_low * S_low.unsqueeze(0), V_low) if S_low.numel() > 0 else 0
    return high_part + low_part


def project_gradient_to_orthogonal_space(svd_dict):
    if all((svd_dict[name].grad is None for name in ["U_low", "S_low", "V_low"])):
        return
    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]
    if svd_dict["U_low"].grad is not None:
        dU = svd_dict["U_low"].grad
        proj = U_high @ (U_high.transpose(0, 1) @ dU)
        dU.sub_(proj)
    if svd_dict["V_low"].grad is not None:
        dV = svd_dict["V_low"].grad
        proj = (dV @ V_high.transpose(0, 1)) @ V_high
        dV.sub_(proj)


def auto_generate_target_svd_config(model):
    patterns = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ]
    config = {}
    for name, param in model.named_parameters():
        if any(p in name for p in patterns) and len(param.shape) == 2:
            top_k = int(min(param.shape) * 0.5)
            full_rank = min(param.shape)
            if top_k >= full_rank:
                top_k = full_rank - 1
            config[name] = top_k
    return config


def optim_wrapper(optimizer, model):
    if not hasattr(model, "project_gradients"):
        return optimizer
    orig_step = optimizer.step
    def step(*args, **kwargs):
        model.project_gradients()
        return orig_step(*args, **kwargs)
    optimizer.step = step
    return optimizer
