from peft.utils import register_peft_method

from .config import OrthogonalSubspaceConfig
from .model import OSLModel
from .layer import OSLLinear

__all__ = ["OrthogonalSubspaceConfig", "OSLModel", "OSLLinear"]

register_peft_method(name="osl", config_cls=OrthogonalSubspaceConfig, model_cls=OSLModel)
