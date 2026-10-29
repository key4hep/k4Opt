"""Strict YAML loading, shared by the components that read YAML files.

``yaml.load(text, Loader=UniqueKeyLoader)`` is ``yaml.safe_load`` that rejects duplicate mapping
keys: PyYAML silently keeps the last one, so a repeated key would drop part of what was written.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any, cast

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that raises :class:`yaml.YAMLError` for a duplicate mapping key."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue  # `<<: *anchor` is expanded by the base class; explicit keys win
            # some PyYAML stubs leave `node` unannotated, which strict mypy calls untyped
            construct = cast(Callable[..., Any], self.construct_object)
            key = construct(key_node, deep=True)
            if isinstance(key, Hashable):
                if key in seen:
                    raise yaml.YAMLError(
                        f"duplicate key {key!r} (line {key_node.start_mark.line + 1})"
                    )
                seen.add(key)
        return super().construct_mapping(node, deep)  # rejects unhashable keys itself
