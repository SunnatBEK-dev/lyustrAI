from enum import Enum


class AssistantMode(str, Enum):
    """User-visible assistant operating modes."""

    SINGLE_MODEL = "single_model"
    ADAPTIVE_MULTI_MODEL = "adaptive"
