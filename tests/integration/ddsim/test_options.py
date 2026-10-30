"""The ddsim options of a problem reach ddsim as written (needs the Key4hep stack).

``ddsim --dumpParameter`` prints what ddsim parsed from its command line and exits without a
geometry, in well under a second; skipped without ``ddsim``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any

import pytest

from k4opt.ddsim.command import option_args

pytestmark = pytest.mark.integration


def parsed_by_ddsim(*args: str) -> dict[str, str]:
    """``option -> value`` as ``ddsim --dumpParameter`` prints them for ``args``."""
    if shutil.which("ddsim") is None:
        pytest.skip("ddsim from the Key4hep stack is required")
    dump = subprocess.run(
        ["ddsim", "--dumpParameter", *args], capture_output=True, text=True, check=True
    ).stdout
    parsed: dict[str, str] = {}
    group = None
    for line in dump.splitlines():
        if match := re.fullmatch(r" '(\w+)': ", line):  # a group: its options follow, indented
            group = match[1]
        elif group and (match := re.fullmatch(r"\t'(\w+)': '(.*)',?", line)):
            parsed[f"{group}.{match[1]}"] = match[2]
        elif match := re.fullmatch(r"[ {]'(\w+)': (.*?),?", line):
            group = None
            parsed[match[1]] = match[2]
    return parsed


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"physics.list": "QGSP_BERT"}, {"physics.list": "QGSP_BERT"}),
        ({"physics.rangecut": 0.5}, {"physics.rangecut": "0.5"}),
        ({"part.minimalKineticEnergy": "2*MeV"}, {"part.minimalKineticEnergy": "2*MeV"}),
        ({"part.keepAllParticles": True}, {"part.keepAllParticles": "True"}),
        ({"enableDetailedShowerMode": True}, {"enableDetailedShowerMode": "True"}),
        ({"geometry.enableDebugVolumes": True}, {"geometry.enableDebugVolumes": "True"}),
        ({"action.run": ["A", "B"]}, {"action.run": "[{'name': 'A'}, {'name': 'B'}]"}),
        (
            {"meta.eventParameters": ["a/F=1", "b/I=2"]},
            {"meta.eventParameters": "['a/F=1', 'b/I=2']"},
        ),
    ],
)
def test_ddsim_reads_each_kind_of_option_as_k4opt_writes_it(
    options: dict[str, Any], expected: dict[str, str]
) -> None:
    parsed = parsed_by_ddsim(*option_args(options))
    assert {key: parsed.get(key) for key in expected} == expected
