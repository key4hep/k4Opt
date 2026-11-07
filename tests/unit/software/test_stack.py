"""The official build in full: what an LCG view's manifests and build information say of each
package, wherever the view is."""

from __future__ import annotations

from pathlib import Path

from k4opt.software import Build
from k4opt.software.stack import Release
from tests.support.software import lcg_view, manifest


def test_a_view_says_each_package_and_the_revision_of_each_head_one(tmp_path: Path) -> None:
    view = lcg_view(tmp_path)
    release = Release.of(str(view))
    assert release is not None
    assert release.build == Build.of(str(view))
    assert set(release.packages) == {"DD4hep", "ROOT", "cldconfig", "ddmarlinpandora"}
    root = release.packages["ROOT"]
    assert (root.version, root.hash, root.revision) == ("6.38.00", "4c5d6", None)
    # in the copy's own release, not where the manifest says it was published
    nightlies = view.parents[4] / "nightlies" / "devkey-head" / "Sat"
    assert root.prefix == str(nightlies / "ROOT" / "6.38.00" / "x86_64-el9-gcc16-opt")
    pandora = release.packages["ddmarlinpandora"]
    assert (pandora.revision, pandora.lcgcmake) == ("821384f", "5cc9f1ea0")
    # "4644ad2|7": the commit, then what lcgcmake adds to it
    assert release.packages["DD4hep"].revision == "4644ad2"
    assert release.manifests["LCG_externals"] == manifest(view).read_bytes()


def test_a_package_is_found_by_its_name_as_k4_local_repo_spells_a_checkout(
    tmp_path: Path,
) -> None:
    release = Release.of(str(lcg_view(tmp_path)))
    assert release is not None
    assert release.package("DDMarlinPandora") is release.packages["ddmarlinpandora"]
    assert release.package("dd4hep") is release.packages["DD4hep"]
    assert release.package("Gaudi") is None
    cldconfig = release.packages["cldconfig"]
    assert release.owner(f"{cldconfig.prefix}/share/CLDConfig/CLDReconstruction.py") is cldconfig
    assert release.owner(str(tmp_path)) is None


def test_a_head_package_without_build_information_has_no_revision(tmp_path: Path) -> None:
    view = lcg_view(tmp_path)
    nightlies = view.parents[4] / "nightlies" / "devkey-head" / "Sat"
    (nightlies / "DD4hep" / "HEAD" / "x86_64-el9-gcc16-opt" / ".buildinfo_DD4hep.txt").unlink()
    release = Release.of(str(view))
    assert release is not None and release.packages["DD4hep"].revision is None


def test_no_release_for_a_stack_that_is_no_lcg_view(tmp_path: Path) -> None:
    other = tmp_path / "setup.sh"
    other.write_text("export PATH=/opt/stack/bin:$PATH\n")
    assert Release.of(str(other)) is None
    assert Release.of(None) is None
    view = lcg_view(tmp_path / "view")
    manifest(view, "contrib").unlink()
    assert Release.of(str(view)) is None


def test_the_latest_link_is_the_release_of_its_weekday(tmp_path: Path) -> None:
    view = lcg_view(tmp_path)
    latest = view.parents[2] / "latest" / "x86_64-el9-gcc16-opt" / "setup.sh"
    first, second = Release.of(str(view)), Release.of(str(latest))
    assert first is not None and second is not None
    assert first.packages == second.packages and first.setup == second.setup
