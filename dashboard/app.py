"""The k4Opt dashboard: the detector optimization studies of a publication, read where it is.

Run it on a publication directory or URL (``docs/dashboard.md``)::

    K4OPT_DASHBOARD_DATA=published/ streamlit run dashboard/app.py

The sidebar says what is shown: the study, chosen once there, and where the studies are read
from. Two pages: *Studies* (``/``), a card for each published study, and *Study*
(``/study?study=<id>``), one of them.
"""

from __future__ import annotations

import html
from datetime import date
from urllib.parse import urlsplit

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from k4opt import __version__
from k4opt.publication import CatalogEntry

import data
import labels

GITHUB = "https://github.com/key4hep/k4Opt"
DOCS = f"{GITHUB}/blob/main/docs/dashboard.md"
STUDY = "study"  # the URL's parameter of the study shown
# layout only, no theme: the page starts just below Streamlit's header (3.75rem, its menu), as
# k4Bench's does, rather than below its much taller default margin; the dialog of the model shown
# full screen takes the screen's width
CSS = """
<style>
.stMainBlockContainer { padding-top: 3.75rem !important; }
div[role="dialog"]:has(.st-key-model-full) { width: 96vw !important; max-width: 96vw !important; }
/* the numbers of a study's status whole, never cut short */
.st-key-status [data-testid="stMetricValue"], .st-key-status [data-testid="stMetricValue"] > div {
  overflow: visible !important; text-overflow: clip !important; }
/* the card of the data source, as k4Bench's */
.k4-source { background: rgba(91,155,213,0.08); border: 1px solid rgba(91,155,213,0.28);
  border-radius: 8px; padding: 0.45rem 0.75rem; display: flex; align-items: center; gap: 0.55rem;
  margin-bottom: 0.25rem; }
.k4-source-text a, .k4-source-text a:hover { text-decoration: none !important; }
.k4-source-icon { font-size: 1.1rem; line-height: 1; }
.k4-source-body { overflow: hidden; }
.k4-source-kind { font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.07em;
  color: #7a9fbf; font-weight: 600; margin-bottom: 0.1rem; }
/* the footer at the bottom of the window when the page is short, after it when it is longer,
   as k4Bench's: the main column is a full-height flex column whose footer takes the free space */
.stMainBlockContainer { min-height: 100vh; display: flex; flex-direction: column; }
.stMainBlockContainer > div:first-child,
.stMainBlockContainer > div:first-child > [data-testid="stVerticalBlock"] {
  flex: 1 0 auto; display: flex; flex-direction: column; }
[data-testid="stElementContainer"]:has(.k4-page-footer) { margin-top: auto; }
.k4-page-rule { border: none; border-top: 1px solid rgba(128,128,128,0.25); margin: 0; }
.k4-page-footer { display: flex; justify-content: center; align-items: center; gap: 1.2rem;
  padding: 0.2rem 0 1.2rem 0; font-size: 0.80rem; color: #9a9a9a; line-height: 1.7;
  text-align: center; }
.k4-page-footer a { color: #5b9bd5; text-decoration: none; font-weight: 600; }
.k4-page-footer strong { color: #c0c0c0; }
.k4-footer-rule { border: none; border-top: 1px solid rgba(128,128,128,0.2);
  margin: 1.5rem 0 0.6rem 0; }
.k4-footer { font-size: 0.72rem; color: #888; text-align: center; line-height: 1.6;
  padding-bottom: 0.4rem; }
.k4-source-text { font-size: 0.72rem; color: #5b9bd5; font-weight: 500; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
</style>
"""

st.set_page_config(
    page_title="k4Opt",
    page_icon=":material/tune:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": DOCS,
        "Report a bug": f"{GITHUB}/issues",
        "About": f"k4Opt {__version__}, detector optimization for Key4hep · [GitHub]({GITHUB})",
    },
)
st.markdown(CSS, unsafe_allow_html=True)


EVERY = "*"  # a filter of the sidebar that keeps every study


def _source(found: data.Source) -> None:
    """Where the studies are read from, as k4Bench says it: a button to read the catalog again
    now, and a card naming the publication, a link where it is served."""
    st.markdown("### Data Source")
    where = found.location
    st.button(
        "Refresh Data",
        on_click=data.refresh,
        disabled=where is None,
        help="Read the catalog again now",
    )
    st.markdown(_card(where), unsafe_allow_html=True)  # why nothing is shown: the page says


def _card(where: str | None) -> str:
    """The card of the publication ``where`` names: what kind of place it is, and where, a link
    if it is served."""
    served = where is not None and urlsplit(where).scheme in ("http", "https")
    if where is None:
        kind = "not configured"
    elif served:
        kind = "WebEOS data" if urlsplit(where).netloc.endswith(".web.cern.ch") else "web data"
    else:
        kind = "EOS data" if where.startswith("/eos/") else "local data"
    shown = html.escape(where or f"set {data.ENVIRONMENT}")
    if served:
        assert where is not None
        shown = (
            f'<a href="{html.escape(where)}" target="_blank" rel="noopener noreferrer" '
            f'style="color:inherit;text-decoration:none">{shown} ↗</a>'
        )
    return (
        '<div class="k4-source"><span class="k4-source-icon">🗄️</span>'
        f'<div class="k4-source-body"><div class="k4-source-kind">{html.escape(kind)}</div>'
        f'<div class="k4-source-text">{shown}</div></div></div>'
    )


def _scope(found: data.Source) -> list[CatalogEntry]:
    """The studies the sidebar keeps: of the detector and the Key4hep stack chosen (every one
    unless chosen), each kept in the URL (``?detector=CLD_o2_v09&key4hep=nightly 2026-10-20``)."""
    catalog = found.catalog
    if catalog is None or not catalog.studies:
        return []
    entries = data.newest_first(catalog)
    detectors = sorted({labels.detector(e.geometry) for e in entries})
    detector = _filter("Detector", "detector", detectors, "Every detector")
    kept = [e for e in entries if detector in (EVERY, labels.detector(e.geometry))]
    stacks = sorted({e.key4hep for e in kept}, reverse=True)
    stack = _filter("Key4hep", "key4hep", stacks, "Every release")
    return [e for e in kept if stack in (EVERY, e.key4hep)]


def _filter(label: str, key: str, options: list[str], every: str) -> str:
    """A filter of the sidebar, bound to the URL's parameter ``key``; a choice the options no
    longer hold (another detector chosen above) is dropped, as k4Bench drops it."""
    if st.session_state.get(key) not in (EVERY, *options):
        st.session_state.pop(key, None)
    chosen = st.selectbox(
        label,
        [EVERY, *options],
        format_func=lambda option: every if option == EVERY else option,
        key=key,
        bind="query-params",
    )
    return str(chosen)


def _picker(
    found: data.Source, kept: list[CatalogEntry], study: StreamlitPage, *, on_study: bool
) -> None:
    """The study shown, chosen among those the sidebar keeps, the latest published first; on the
    Study page without one, the latest. Choosing another opens it, with nothing of the last one's
    view but its tab."""
    catalog = found.catalog
    if catalog is None or not catalog.studies:
        return
    every = {entry.id: entry for entry in data.newest_first(catalog)}
    entries = {entry.id: entry for entry in kept}
    held = st.query_params.get(STUDY) if on_study else None
    if on_study and held is None:
        held = next(iter(entries or every))
        st.query_params[STUDY] = held
    current = held if held in entries else None
    chosen = st.selectbox(
        "Study",
        list(entries),
        index=None if current is None else list(entries).index(current),
        format_func=lambda study: entries[study].name,
        placeholder="Choose a study" if entries else "No study of these",
        key=f"sidebar-study-{current}",  # a widget of its own for each study shown
    )
    if chosen is not None and chosen != current:
        # the tab open stays open: its widget holds it (and puts it in the URL), not the URL
        st.switch_page(study, query_params={STUDY: chosen})


def _footer() -> None:
    """The attribution at the bottom of the sidebar, as k4Bench's."""
    st.markdown(
        f"""
        <hr class="k4-footer-rule">
        <div class="k4-footer">
            <strong style="color:#a0a0a0;">© {date.today().year} CERN</strong><br>
            For the benefit of the<br>
            <a href="https://fcc.web.cern.ch/" target="_blank" rel="noopener noreferrer"
               style="color:#5b9bd5;text-decoration:none;">FCC project</a><br>
            <span style="opacity:0.7;">J. F. Beirer (CERN)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _page_footer() -> None:
    """The attribution at the bottom of the page, as k4Bench's."""
    st.markdown(
        f"""
        <hr class="k4-page-rule">
        <div class="k4-page-footer">
            <span style="font-size:1.8rem;opacity:0.75;">⚛️</span>
            <div>
                <strong style="letter-spacing:0.02em;">© {date.today().year} CERN</strong>
                &nbsp;·&nbsp; For the benefit of the&nbsp;<a href="https://fcc.web.cern.ch/"
                target="_blank" rel="noopener noreferrer">FCC project</a>
                <br>
                Created by <strong>Joshua Falco Beirer</strong>
                &nbsp;<span style="opacity:0.6;">(CERN)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


found = data.resolve()
studies = st.Page("studies.py", title="Studies", icon=":material/grid_view:", default=True)
study = st.Page("study.py", title="Study", icon=":material/scatter_plot:", url_path="study")
page = st.navigation([studies, study], position="hidden")
with st.sidebar:
    _source(found)
    kept = _scope(found)
    st.session_state[data.KEPT] = [entry.id for entry in kept]  # for the Studies page
    _picker(found, kept, study, on_study=page.url_path == study.url_path)
    st.page_link(studies, label="All studies", icon=":material/table_rows:")
    _footer()
page.run()
_page_footer()
