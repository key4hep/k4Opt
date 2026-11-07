"""Internal: what a study keeps of what it ran with, as its sessions open and its trials are told.

The study at the top of an optimization keeps it all in its ``software/`` directory, for every
study of the levels below too (:mod:`k4opt.software.keep`): when a session opens, the software and
where its local builds come from; as each trial's record is accepted, the inputs its stages were
observed to read and which files of local builds they read. Keeping never stops a study: what
cannot be kept is said once, and recorded.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from k4opt.record.write import SUMMARY_NAME, recorded_footprints, recorded_inputs
from k4opt.software import Software, SoftwareError
from k4opt.software.keep import DIRECTORY, keep_inputs, keep_used, take

_SAID = 3  # how many problems a warning names


class Keeper:
    """Keeps, in ``study/software``, what the trials of the study in ``study`` ran with."""

    def __init__(self, study: Path, warn: Callable[[str], None]) -> None:
        self.directory = study / DIRECTORY
        self._warn = warn
        self._roots: list[str] = []
        self._snapshot: str | None = None  # the state of the software this session kept

    def open(self, identity: Mapping[str, Any]) -> None:
        """Keep what a session with ``identity`` runs on (:func:`~k4opt.software.keep.take`)."""
        try:
            software = Software.from_dict(identity["software"])
            self._roots = [entry.path for entry in software.local]
            taken = take(self.directory, software, identity)
        except (SoftwareError, OSError, KeyError) as exc:
            self._warn(f"the study keeps nothing of what it runs with: {exc}")
            return
        self._snapshot = taken.path.name
        if os.path.realpath(self.directory).startswith("/afs/"):
            self._warn(
                f"{self.directory} keeps private copies of local code and inputs; on AFS its "
                "access list decides who can read them (fs listacl), not its permissions"
            )
        self._say(list(taken.problems))

    def trial(self, run: Path) -> None:
        """Keep what the stages of the trial in ``run`` read that its record names."""
        try:
            summary = json.loads((run / SUMMARY_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(summary, Mapping):
            return
        inputs = {
            path: (then, [stage], "observed")
            for path, (then, stage) in recorded_inputs(summary).items()
        }
        try:
            where = run.relative_to(self.directory.parent).as_posix()
        except ValueError:
            where = str(run)
        try:
            problems = keep_inputs(self.directory, inputs, trial=where)
            footprints = recorded_footprints(summary)
            keep_used(self.directory, run, footprints, self._roots, where, self._snapshot)
        except (SoftwareError, OSError) as exc:
            problems = [str(exc)]
        self._say(problems)

    def _say(self, problems: list[str]) -> None:
        if problems:
            more = f" (and {len(problems) - _SAID} more)" if len(problems) > _SAID else ""
            said = "; ".join(problems[:_SAID])
            self._warn(f"not kept for a rerun: {said}{more}; `k4opt study show` lists them")
