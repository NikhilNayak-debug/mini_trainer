import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner

from .config import OrthogonalSubspaceConfig
from .layer import OSLLinear
from .utils import auto_generate_target_svd_config


class OSLModel(BaseTuner):
    """Applies orthogonal subspace adapters to a pretrained model."""

    prefix: str = "osl_"

    def __init__(self, model, config: OrthogonalSubspaceConfig, adapter_name: str, low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _prepare_adapter_config(self, peft_config: OrthogonalSubspaceConfig, model_config: dict) -> OrthogonalSubspaceConfig:
        if peft_config.automatic_target and not peft_config.svd_config:
            peft_config.svd_config = auto_generate_target_svd_config(self.model)
        return peft_config

    def _check_target_module_exists(self, peft_config: OrthogonalSubspaceConfig, key: str) -> bool:
        return key in peft_config.svd_config

    def _create_and_replace(self, peft_config: OrthogonalSubspaceConfig, adapter_name: str, target: nn.Module, target_name: str, parent: nn.Module, current_key: str) -> None:
        top_k = peft_config.svd_config[current_key]
        if isinstance(target, nn.Linear):
            new_module = OSLLinear(target, top_k)
            setattr(parent, target_name, new_module)
            self.targeted_module_names.append(current_key)

    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        for n, p in model.named_parameters():
            p.requires_grad = False
        for module in model.modules():
            if isinstance(module, OSLLinear):
                module.U_low.requires_grad = True
                module.S_low.requires_grad = True
                module.V_low.requires_grad = True

    def disable_adapter_layers(self) -> None:
        for module in self.model.modules():
            if isinstance(module, OSLLinear):
                module.requires_grad_(False)

    def enable_adapter_layers(self) -> None:
        for module in self.model.modules():
            if isinstance(module, OSLLinear):
                module.requires_grad_(True)

    def project_gradients(self):
        for module in self.model.modules():
            if isinstance(module, OSLLinear):
                module.project_gradients()
