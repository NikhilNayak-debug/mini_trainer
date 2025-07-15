import enum
from typing import Optional


class PeftType(str, enum.Enum):
    """Enum class for the different types of adapters in PEFT."""

    PROMPT_TUNING = "PROMPT_TUNING"
    MULTITASK_PROMPT_TUNING = "MULTITASK_PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    BOFT = "BOFT"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"
    LOHA = "LOHA"
    LOKR = "LOKR"
    OFT = "OFT"
    POLY = "POLY"
    LN_TUNING = "LN_TUNING"
    VERA = "VERA"
    FOURIERFT = "FOURIERFT"
    XLORA = "XLORA"
    HRA = "HRA"
    VBLORA = "VBLORA"
    CPT = "CPT"
    BONE = "BONE"
    RANDLORA = "RANDLORA"
    TRAINABLE_TOKENS = "TRAINABLE_TOKENS"
    C3A = "C3A"
    OSL = "OSL"


class TaskType(str, enum.Enum):
    """Enum class for the different task types supported by PEFT."""

    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"

