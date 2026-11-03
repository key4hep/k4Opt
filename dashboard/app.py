"""The k4Opt dashboard: the detector optimization studies of a publication, read where it is.

Run it on a publication directory or URL (``docs/dashboard.md``)::

    K4OPT_DASHBOARD_DATA=published/ streamlit run dashboard/app.py

Two pages, in the top bar: *Studies* (``/``), every published study, and *Study*
(``/study?study=<id>``), one of them. The sidebar, closed at first, says where the studies are
read from.
"""

from __future__ import annotations

import streamlit as st

from k4opt import __version__

import data

st.set_page_config(
    page_title="k4Opt",
    page_icon=":material/tune:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

found = data.resolve()

with st.sidebar:
    st.markdown("### k4Opt")
    st.caption("Detector optimization for Key4hep")
    st.markdown("**Publication**")
    st.code(found.location or "not configured", language=None)
    if found.read is not None:
        count = len(found.read.catalog.studies)
        st.caption(
            f"{count} {'study' if count == 1 else 'studies'}; catalog read "
            f"{found.read.at:%H:%M:%S} UTC, again after {data.CATALOG_TTL} s"
        )
    elif found.problem is not None:
        st.caption(
            {"info": "Nothing to show", "warning": "Unavailable", "error": "Unreadable"}[
                found.level
            ]
        )
    st.button(
        "Refresh",
        on_click=data.refresh,
        icon=":material/refresh:",
        disabled=found.location is None,
        help="Read the catalog again now",
    )
    st.caption(f"k4opt {__version__}")

pages = [
    st.Page("overview.py", title="Studies", icon=":material/table_rows:", default=True),
    st.Page("study.py", title="Study", icon=":material/scatter_plot:", url_path="study"),
]
st.navigation(pages, position="top").run()
