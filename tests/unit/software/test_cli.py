"""``k4opt software show``: what candidates would run on in this shell."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from k4opt.cli import main
from k4opt.software import Software
from tests.support.software import PLATFORM, lcg_view


def show() -> str:
    out = io.StringIO()
    assert main(["software", "show"], stdout=out, stderr=io.StringIO()) == 0
    return out.getvalue()


def test_it_says_which_build_this_shell_runs_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEY4HEP_STACK", str(lcg_view(tmp_path)))
    text = show()
    assert "LCG devkey-head, built Sat Nov 7 04:17:32 2026" in text
    assert PLATFORM in text
    assert Software.now().id in text
    assert f"{len(Software.now().variables)} compared" in text


def test_an_unrecognized_stack_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEY4HEP_STACK", raising=False)
    text = show()
    assert "unrecognized stack" in text and "platform" not in text
