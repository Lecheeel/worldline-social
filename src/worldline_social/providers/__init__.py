"""Model-provider adapters used by social controllers."""

from .base import (
    CompletionRequest,
    CompletionResponse,
    ModelMessage,
    ModelProvider,
    ModelToolCall,
    ProviderError,
)
from .deepseek import DeepSeekProvider, create_deepseek_provider
from .registry import ProviderRegistry


def builtin_provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("deepseek", create_deepseek_provider)
    return registry

__all__ = [
    "CompletionRequest",
    "CompletionResponse",
    "DeepSeekProvider",
    "ModelMessage",
    "ModelProvider",
    "ModelToolCall",
    "ProviderError",
    "ProviderRegistry",
    "builtin_provider_registry",
    "create_deepseek_provider",
]
