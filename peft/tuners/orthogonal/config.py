from dataclasses import dataclass, field
from typing import Dict, Optional

from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class OrthogonalSubspaceConfig(PeftConfig):
    """Configuration for orthogonal subspace learning adapters."""

    svd_config: Optional[Dict[str, int]] = field(default_factory=dict, metadata={"help": "Parameter name to top_k."})
    automatic_target: bool = field(default=True, metadata={"help": "Infer target modules automatically when True."})

    def __post_init__(self):
        self.peft_type = PeftType.OSL
