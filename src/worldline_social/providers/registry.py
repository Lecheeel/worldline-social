"""Explicit provider registration; no runtime package discovery is required."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import ModelProvider

ProviderFactory = Callable[[Mapping[str, Any]], ModelProvider]


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, provider_id: str, factory: ProviderFactory) -> None:
        if not provider_id:
            raise ValueError("provider_id must not be empty")
        if provider_id in self._factories:
            raise ValueError(f"provider already registered: {provider_id}")
        self._factories[provider_id] = factory

    def create(self, provider_id: str, config: Mapping[str, Any]) -> ModelProvider:
        try:
            factory = self._factories[provider_id]
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "none"
            raise KeyError(f"unknown provider {provider_id}; available: {available}") from error
        return factory(config)

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
