"""The Software tab: what the study ran on and what a rerun would need, with the evidence and its
date. It says *not rerun* until someone has rerun it, and keeps apart a local build's source, which
can be built again, and the binary that ran, of which only a fingerprint is left.

Everything shown comes from the study's software sheet: a published file's text is shown as text,
never run, imported or rendered."""

from __future__ import annotations

import streamlit as st

from k4opt.errors import K4OptError
from k4opt.publication import CatalogEntry, PublishedArtifact, SoftwareSheet

import data
import labels
import tables
import widgets

SHOWN_BYTES = 2**20
"""The largest published file shown here; a larger one is named, with where it is."""


def render(where: str, entry: CatalogEntry) -> None:
    try:
        sheet = data.software(where, entry)
    except K4OptError as exc:
        st.error(f"The study's software sheet does not hold up: {labels.plain(exc)}")
        return
    if sheet is None:
        st.info("This publication says nothing of the software this study ran on.")
        return
    _status(sheet)
    _changes(sheet)
    _published(where, sheet)
    _inputs(where, sheet)
    _rerun(sheet)
    _known(sheet)


def _status(sheet: SoftwareSheet) -> None:
    with st.container(border=True, key="software-status"):
        st.markdown(f"**{labels.plain(sheet.status)}**")
        base = tables.requirement(sheet, "base")
        if base is not None:
            st.caption(f"{labels.plain(base.label)} · {labels.plain(base.identified)}")
        st.caption(labels.plain(tables.checked(sheet)))
        st.caption(labels.plain(tables.at_a_glance(sheet)))
        for line in tables.software_warnings(sheet):
            st.warning(labels.plain(line), icon=":material/warning:")


def _changes(sheet: SoftwareSheet) -> None:
    rows = tables.local_changes(sheet)
    if not rows:
        return
    st.subheader("Changes to the nightly")
    widgets.table(rows)
    st.caption(tables.REBUILD_NOTE)
    line = tables.stages_used(sheet)
    if line:
        st.caption(labels.plain(line))


def _published(where: str, sheet: SoftwareSheet) -> None:
    code = [artifact for artifact in sheet.artifacts if artifact.group == "code"]
    for artifact in code:
        title = f"{artifact.owner} · {artifact.name}" if artifact.owner else artifact.name
        with st.expander(labels.plain(title), icon=":material/difference:"):
            _artifact(where, artifact)


def _artifact(where: str, artifact: PublishedArtifact) -> None:
    if artifact.kind == "patch":
        st.caption(labels.plain(f"Against {artifact.base}"))
    if artifact.bytes > SHOWN_BYTES:
        st.caption(f"{labels.size(artifact.bytes)}: too large to show here.")
    else:
        try:
            content = data.artifact(where, artifact)
        except K4OptError as exc:
            st.error(f"This file does not hold up: {labels.plain(exc)}")
            return
        text = labels.text(content)
        if text is None:
            st.caption(f"{labels.size(artifact.bytes)} of binary data.")
        else:
            st.code(text, language="diff" if artifact.kind == "patch" else None)
    _download(where, artifact)


def _key(artifact: PublishedArtifact) -> str:
    """A widget's key for ``artifact``: two can hold equal bytes, none answers the same things."""
    parts = ("get", artifact.sha256[:16], artifact.owner, artifact.name, *artifact.requirements)
    return "-".join(parts)


def _download(where: str, artifact: PublishedArtifact) -> None:
    """A way to get a published file: its bytes, checked against its SHA-256, or, too large to
    hold in the page, its link or its place in the publication."""
    if artifact.bytes <= SHOWN_BYTES:
        try:
            content = data.artifact(where, artifact)
        except K4OptError as exc:
            st.error(f"This file does not hold up: {labels.plain(exc)}")
            return
        name = artifact.name.rpartition("/")[2]
        st.download_button(artifact.name, content, file_name=name, key=_key(artifact))
        return
    link = tables.artifact_link(where, artifact)
    if link is not None:
        st.markdown(f"[Download {labels.plain(artifact.name)}]({link})")
    else:
        st.caption(labels.plain(f"{artifact.name}: {artifact.path} in the publication."))


def _inputs(where: str, sheet: SoftwareSheet) -> None:
    rows = tables.inputs(sheet)
    if not rows:
        return
    st.subheader("Inputs")
    widgets.table(rows)
    for artifact in (a for a in sheet.artifacts if a.group == "inputs"):
        _download(where, artifact)
    st.caption(
        "A commitment binds this publication to a file without showing anything of it: someone "
        "given the file and its salt can check it is the one that was read."
    )


def _rerun(sheet: SoftwareSheet) -> None:
    steps = tables.rerun_steps(sheet)
    st.subheader("To rerun")
    st.markdown("\n".join(f"{n}. {labels.plain(step)}" for n, step in enumerate(steps, start=1)))
    missing = tables.rerun_missing(sheet)
    if missing:
        st.warning(
            "What is published is not enough to do this alone; not published: "
            f"{labels.plain(', '.join(missing))}.",
            icon=":material/lock:",
        )


def _known(sheet: SoftwareSheet) -> None:
    rows = tables.other_requirements(sheet)
    with st.expander("Everything a rerun needs", icon=":material/checklist:"):
        widgets.table(rows)
