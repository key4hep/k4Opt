"""Loading of the k4opt YAML configuration file.

::

    geometry: FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml   # relative to $K4GEO, or absolute
    root: ../k4geo                                          # optional, relative to this YAML file
    overrides:                                              # optional
      - target: constants.InnerTracker_Barrel_radius_1
        value: "405*mm"
      - target: constants.HCal_cell_size
        file: FCCee/CLD/compact/CLD_o2_v09/HCalBarrel_o1_v01_01.xml   # only for ambiguous targets
        value: "30*mm"

Override values must be YAML strings; numbers and booleans are rejected so the
replacement text is always exactly what the user wrote.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from k4opt.errors import ConfigError
from k4opt.geometry.errors import InvalidTargetError
from k4opt.geometry.overrides import Override
from k4opt.geometry.source import GeometrySource

_TOP_KEYS = {"geometry", "root", "overrides"}
_OVERRIDE_KEYS = {"target", "value", "file"}


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """A parsed configuration: the geometry source plus the requested overrides."""

    path: Path
    source: GeometrySource
    overrides: tuple[Override, ...]


def _require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], what: str) -> None:
    unknown = sorted(str(k) for k in mapping if k not in allowed)
    if unknown:
        raise ConfigError(f"{what} has unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _require_str(mapping: Mapping[str, Any], key: str, what: str) -> str:
    if key not in mapping:
        raise ConfigError(f"{what} is missing required key {key!r}")
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{what}.{key} must be a non-empty string")
    return value


def load_config(path: str | os.PathLike[str]) -> GeometryConfig:
    """Parse ``path`` and build the :class:`GeometrySource` and overrides it describes."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"configuration file {config_path} not found")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{config_path}: cannot read: {exc}") from exc
    return parse_config(data, config_path)


def parse_config(data: Any, config_path: Path) -> GeometryConfig:
    """Validate an already-loaded YAML document (``root`` resolves against ``config_path``)."""
    top = _require_mapping(data, "configuration")
    _reject_unknown(top, _TOP_KEYS, "configuration")
    geometry = _require_str(top, "geometry", "configuration")

    root: Path | None = None
    if top.get("root") is not None:
        root_text = _require_str(top, "root", "configuration")
        root = Path(os.path.expandvars(os.path.expanduser(root_text)))
        if not root.is_absolute():
            root = config_path.resolve().parent / root

    source = GeometrySource(compact=geometry.strip(), root=root)

    raw_overrides = top.get("overrides")
    overrides: list[Override] = []
    if raw_overrides is not None:
        if not isinstance(raw_overrides, list):
            raise ConfigError("'overrides' must be a list")
        for index, item in enumerate(raw_overrides):
            what = f"overrides[{index}]"
            mapping = _require_mapping(item, what)
            _reject_unknown(mapping, _OVERRIDE_KEYS, what)
            target = _require_str(mapping, "target", what)
            if "value" not in mapping:
                raise ConfigError(f"{what} is missing required key 'value'")
            value = mapping["value"]
            if not isinstance(value, str):
                raise ConfigError(
                    f"{what}.value must be a quoted string, got {type(value).__name__} "
                    f'({value!r}); write value: "{value}"'
                )
            file = mapping.get("file")
            if file is not None and (not isinstance(file, str) or not file.strip()):
                raise ConfigError(f"{what}.file must be a non-empty string when given")
            try:
                overrides.append(Override(target=target.strip(), value=value, file=file))
            except InvalidTargetError as exc:
                raise ConfigError(f"{what}: {exc}") from exc
    return GeometryConfig(path=config_path, source=source, overrides=tuple(overrides))
