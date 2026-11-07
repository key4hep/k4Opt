"""A rerun's material, made from nothing but what a study kept (``software/``) and its
``study.json``: the acceptance harness of what k4opt keeps. ``k4opt reproduce`` is deferred; this
is what it would do for one candidate, in tests only.

:func:`restore` writes, below a new directory: each reconstruction's job directory, made from the
stack's package plus the kept changed files; each checkout, cloned from its remote at its base
commit with the kept patch applied (a build among them is to be built again by the caller); the
files of local code in no checkout; each kept input; each evaluator's files; and the problem, with
its job directories pointing at the restored ones. Every kept byte is read through
:meth:`k4opt.software.keep.Store.get`, which refuses altered bytes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from k4opt.software.keep import INPUTS, Store
from k4opt.software.readiness import kept_directory


@dataclass
class Restored:
    """Where :func:`restore` put each thing, by what it was in the original run."""

    problem: Path
    steering: dict[str, Path] = field(default_factory=dict)  # stage label: job directory
    checkouts: dict[str, Path] = field(default_factory=dict)  # original prefix: clone
    code: dict[str, Path] = field(default_factory=dict)  # original directory: restored one
    inputs: dict[str, Path] = field(default_factory=dict)  # original path: restored file
    evaluators: dict[str, Path] = field(default_factory=dict)  # name: restored file


def _git(*args: str, cwd: Path | None = None, stdin: bytes | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, input=stdin, check=True, capture_output=True)


def restore(
    study: Path, into: Path, remotes: dict[str, str] | None = None, k4opt: bool = False
) -> Restored:
    """Restore into the new directory ``into`` what a rerun of the study in ``study`` needs
    (see the module); ``remotes`` replaces the remote of a checkout (by its path then); k4opt's
    own checkout only with ``k4opt`` (a test reruns with the k4opt it runs)."""
    directory = kept_directory(study)
    store = Store(directory)
    (document,) = (json.loads(p.read_text()) for p in directory.glob("software-*.json"))
    into.mkdir(parents=True)
    definition: dict[str, Any] = json.loads((study / "study.json").read_text())["problem"][
        "definition"
    ]
    restored = Restored(into / "problem.yaml")
    for entry in document["steering"]:
        job = into / "steering" / entry["stage"].replace("/", "-") / Path(entry["directory"]).name
        shutil.copytree(entry["original"], job, symlinks=True)
        os.chmod(job, 0o755)
        for root, dirs, files in os.walk(job):
            for name in dirs + files:
                os.chmod(os.path.join(root, name), 0o755 if name in dirs else 0o644)
        for name in entry["removed"]:
            (job / name).unlink()
        for name, file in {**entry["changed"], **entry["added"]}.items():
            (job / name).parent.mkdir(parents=True, exist_ok=True)
            (job / name).write_bytes(store.get(file["kept"]))
        restored.steering[entry["stage"]] = job
        sample, stage = entry["stage"].split("/")
        definition["samples"][sample][stage]["directory"] = str(job)
    for origin in document["origins"]:
        git = origin.get("git")
        name = Path(origin["prefix"].rstrip("/")).name
        if git is not None and not k4opt and git["checkout"] == document.get("k4opt"):
            continue
        if git is not None:
            remote = (remotes or {}).get(origin["prefix"]) or git["remote"] or git["checkout"]
            clone = into / "src" / Path(git["checkout"]).name
            if clone.exists():
                continue
            _git("clone", "-q", remote, str(clone))
            if git["base"]:
                _git("checkout", "-q", git["base"], cwd=clone)
            if git["patch"]:
                _git("apply", "--whitespace=nowarn", "-", cwd=clone, stdin=store.get(git["patch"]))
            restored.checkouts[origin["prefix"]] = clone
        elif origin["kind"] == "directory":
            target = into / "code" / name
            for relative, file in origin["files"].items():
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                (target / relative).write_bytes(store.get(file["kept"]))
            restored.code[origin["prefix"]] = target
    kept = directory / INPUTS
    inputs = json.loads(kept.read_text())["inputs"] if kept.is_file() else {}
    for path, entry in inputs.items():
        if entry["kept"]:
            target = into / "inputs" / Path(path).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(store.get(entry["kept"]))
            restored.inputs[path] = target
    for name, entry in document["evaluators"].items():
        target = into / "evaluators" / name
        for file, sha256 in entry["files"].items():
            (target / file).parent.mkdir(parents=True, exist_ok=True)
            (target / file).write_bytes(store.get(sha256))
        restored.evaluators[name] = target / Path(entry["path"]).name
    restored.problem.write_text(yaml.safe_dump(definition, allow_unicode=True))
    return restored
