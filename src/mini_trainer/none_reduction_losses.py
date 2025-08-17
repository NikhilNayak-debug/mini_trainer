import torch
import torch.nn as nn
from typing import Optional

def liger_fixed_fused_linear_cross_entropy_none_reduction(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    final_logit_softcapping: Optional[float] = None,
    **kwargs,
):
    import liger_kernel.transformers.functional as F
    # torch.distributed.breakpoint()
    loss = F.liger_fused_linear_cross_entropy(
        hidden_states,
        lm_head_weight,
        target,
        reduction='none',
        ignore_index=ignore_index,
        softcap=final_logit_softcapping,
    )

    return loss

def hf_fixed_cross_entropy_none_reduction(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    # torch.distributed.breakpoint()
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction='none')
    return loss
