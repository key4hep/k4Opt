"""Geometry component: parameters of DD4hep/Key4hep XML detector descriptions.

Discover every addressable XML attribute reachable from a compact file, override
selected ones with literal values and materialize an isolated, modified copy of
the geometry with a manifest. See ``docs/geometry.md``.

Typical use (inside a Key4hep environment, where ``$K4GEO`` is set)::

    from k4opt.geometry import GeometrySource, Override

    source = GeometrySource("FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml")
    catalog = source.discover()
    result = source.materialize(
        overrides=[Override(target="constants.InnerTracker_Barrel_radius_1", value="405*mm")],
        output="./generated/cld-test",
    )
"""

from k4opt.errors import ConfigError
from k4opt.geometry.config import GeometryConfig, load_config
from k4opt.geometry.discovery import Catalog, Parameter, Unaddressable
from k4opt.geometry.errors import (
    AmbiguousTargetError,
    DuplicateTargetError,
    GeometryError,
    IncludeError,
    InvalidTargetError,
    MaterializationError,
    SourceError,
    TargetError,
    UnknownTargetError,
)
from k4opt.geometry.includes import ExternalInclude, IncludeEdge
from k4opt.geometry.materialize import MANIFEST_NAME, ChangedFile, MaterializationResult
from k4opt.geometry.overrides import Override, ResolvedOverride
from k4opt.geometry.references import ParameterReference, Step, parse_reference
from k4opt.geometry.source import K4GEO_ENV, GeometrySource, GitInfo, resolve_geometry

__all__ = [
    "K4GEO_ENV",
    "MANIFEST_NAME",
    "AmbiguousTargetError",
    "Catalog",
    "ChangedFile",
    "ConfigError",
    "DuplicateTargetError",
    "ExternalInclude",
    "GeometryConfig",
    "GeometryError",
    "GeometrySource",
    "GitInfo",
    "IncludeEdge",
    "IncludeError",
    "InvalidTargetError",
    "MaterializationError",
    "MaterializationResult",
    "Override",
    "Parameter",
    "ParameterReference",
    "ResolvedOverride",
    "SourceError",
    "Step",
    "TargetError",
    "Unaddressable",
    "UnknownTargetError",
    "load_config",
    "parse_reference",
    "resolve_geometry",
]
