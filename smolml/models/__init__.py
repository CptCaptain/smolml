"""Model interface, registry, and baseline.

Importing this package registers the built-in models (their ``@register_model``
decorators run on import), so ``list_models()`` is populated.
"""

from smolml.models.registry import (
    LanguageModel,
    build_model,
    get_model,
    list_models,
    register_model,
)
from smolml.models.transformer import Transformer, TransformerConfig

__all__ = [
    "LanguageModel",
    "Transformer",
    "TransformerConfig",
    "build_model",
    "get_model",
    "list_models",
    "register_model",
]
