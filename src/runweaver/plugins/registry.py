"""Explicit plugin registry with Python entry-point discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.metadata import entry_points

from runweaver.builtins import BUILTIN_BLOCKS
from runweaver.exceptions import ConfigurationError


class PluginRegistry:
    """Resolve implementations by registered ids, never by executing config paths."""

    def __init__(self, *, discover: bool = True) -> None:
        self.blocks: dict[str, Callable[[dict[str, object]], object]] = dict(BUILTIN_BLOCKS)
        self.planners: dict[str, Callable[[dict[str, object]], object]] = {}
        self.serializers: dict[str, Callable[[], object]] = {}
        self.executors: dict[str, Callable[[dict[str, object]], object]] = {}
        if discover:
            self.discover()

    def register_block(
        self,
        name: str,
        factory: Callable[[dict[str, object]], object],
    ) -> None:
        if name in self.blocks:
            raise ConfigurationError(f"block plugin already registered: {name}")
        self.blocks[name] = factory

    def resolve_block(self, name: str, parameters: Mapping[str, object] | None = None) -> object:
        try:
            factory = self.blocks[name]
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown block plugin {name!r}; install/register an entry point "
                "in group 'runweaver.blocks'"
            ) from exc
        return factory(dict(parameters or {}))

    def discover(self) -> None:
        groups = {
            "runweaver.blocks": self.blocks,
            "runweaver.planners": self.planners,
            "runweaver.serializers": self.serializers,
            "runweaver.executors": self.executors,
        }
        for group, target in groups.items():
            for entry_point in entry_points(group=group):
                target.setdefault(entry_point.name, entry_point.load())
