"""The Study page: one study, chosen by its id in the URL (``/study?study=<id>``), so that a link
names it, in tabs, the one open in the URL too (``&view=Trials``): *Overview*, its trials in the
plane of two axes and in the order they were proposed; *Trials*, each trial and all it recorded;
*Analysis*, how its parameters act on an objective; *Pipeline*, how a candidate is evaluated;
*Problem*, what it optimizes and what it was made with."""

from __future__ import annotations

from collections.abc import Callable

import plotly.graph_objects as go
import streamlit as st

from k4opt.errors import K4OptError
from k4opt.optimization import RULE_KINDS, Rule, StudyResults, TradeOffs, TrialResult, trade_offs
from k4opt.publication import CatalogEntry, PublicationUnavailableError
from k4opt.visualization import (
    TRIAL,
    GraphNode,
    TaskGraph,
    axis_choices,
    color_choices,
    default_axes,
    edf_figure,
    history_figure,
    parallel_figure,
    pipeline_figure,
    rank_figure,
    slice_figure,
    task_graph,
    trials_figure,
)

import data
import tables

# the URL's parameters: the study by its id, the tab open, the figure's axes and what colors it,
# and the trial shown (a page script shows a bare string, so nothing here is a docstring but the
# module's)
STUDY, VIEW, X, Y, COLOR, SHOWN = "study", "view", "x", "y", "color", "trial"
ANALYZED = "objective"  # the URL's parameter of the objective the Analysis tab shows
TABS = ("Overview", "Trials", "Analysis", "Pipeline", "Problem")
BY_OUTCOME = "outcome"  # the figure's colors unless chosen: each trial by its outcome
# the trial a click on a figure opened, and counters bumped after each click on a figure and each
# closing of a trial, so that the chart or table comes back without a selection
OPENED, PICKS, ROWS = "trial-opened", "trial-picks", "trial-rows"
# the URL's parameters of a trade-off: the reference trial, a rule for each objective by its
# index (``0:worsen:5:%,1:improve:10:u``) and ``noise=0`` to compare values without their
# uncertainties; and what each kind of rule is called
REFERENCE, RULES, NOISE = "ref", "rules", "noise"
ANY = "any"
KINDS = {
    ANY: "may do anything",
    "improve": "must improve by at least",
    "worsen": "may worsen by at most",
    "keep": "must not worsen",
}
# the color of each outcome's badge; every other outcome is a trial without a result
BADGES = {"feasible": "green", "infeasible": "orange", "running": "blue"}
# the task selected, the stacks of samples drawn sample by sample (by key; ALL: every one), the
# study they are of, and a counter bumped after each click, so that the chart comes back without
# a selection and the next click on the same node is a click again
SELECTED, EXPANDED, OF, CHART = "tasks-selected", "tasks-expanded", "tasks-of", "tasks-chart"
ALL = "*"
# drag to pan, scroll to zoom, buttons for both and for going back to the whole graph
CHART_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtons": [["zoomIn2d", "zoomOut2d", "resetScale2d"]],
}


def _chosen(
    label: str,
    options: list[str],
    key: str,
    default: str | None,
    format_func: Callable[[str], str],
    placeholder: str | None = None,
) -> str | None:
    """A choice kept in the URL's parameter ``key``, so that a link reopens it; a value the options
    do not hold gives ``default``."""
    held = st.query_params.get(key)
    current = held if held in options else default
    chosen = st.selectbox(
        label,
        options,
        index=None if current is None else options.index(current),
        format_func=format_func,
        placeholder=placeholder,
        key=f"choice-{key}",
    )
    if chosen is not None and chosen != held:
        st.query_params[key] = chosen
    return chosen


def main(found: data.Source) -> None:
    catalog = found.catalog
    if catalog is None or not catalog.studies or found.location is None:
        st.title("Study")
        data.show(found)
        return
    entries = {entry.id: entry for entry in _newest_first(catalog.studies)}
    requested = st.query_params.get(STUDY)
    if requested is not None and requested not in entries:
        st.warning(
            f"The catalog of {data.plain(found.location)} has no study "
            f"{data.plain(requested)}; choose one."
        )
    names = {study: entry.name for study, entry in entries.items()}
    narrow, _ = st.columns([1, 2])
    with narrow:
        chosen = _chosen(
            "Study",
            list(entries),
            STUDY,
            None if requested is not None else next(iter(entries)),
            format_func=lambda study: names[study],
            placeholder="Choose a study",
        )
    if not chosen:
        return
    entry = entries[chosen]
    try:
        results = data.study(found.location, entry)
    except PublicationUnavailableError as exc:
        st.warning(
            f"The study cannot be read just now ({data.plain(exc)}); it is read again on your "
            "next interaction."
        )
        return
    except K4OptError as exc:
        st.error(f"The study's snapshot does not hold up: {data.plain(exc)}")
        return
    _render(entry, results)


def _newest_first(entries: tuple[CatalogEntry, ...]) -> list[CatalogEntry]:
    return sorted(entries, key=lambda entry: entry.published, reverse=True)


def _render(entry: CatalogEntry, results: StudyResults) -> None:
    st.title(data.plain(entry.name), anchor=False)
    st.caption(  # an id is a UUID and a SHA-256 hex digits, checked: nothing in them formats
        f"Problem **{data.plain(entry.problem or 'unnamed')}** · created "
        f"{data.plain(entry.created)} · published {data.plain(entry.published)} · study "
        f"`{entry.id}` · snapshot `{entry.snapshot.sha256[:16]}`"
    )
    outcomes = entry.outcomes
    if outcomes["running"]:
        st.info(
            f"{outcomes['running']} trial(s) were running when this snapshot was published; "
            "publish the study again to see them end."
        )
    columns = st.columns(6)
    shown = [
        ("Trials", entry.trials),
        ("Feasible", outcomes["feasible"]),
        ("Infeasible", outcomes["infeasible"]),
        ("Without a result", sum(outcomes[o] for o in tables.WITHOUT_RESULT)),
        ("Running", outcomes["running"]),
        ("Pareto front", len(results.pareto)),
    ]
    for column, (label, value) in zip(columns, shown, strict=True):
        column.metric(label, value, border=True)
    # only the open tab runs (the graph needs Graphviz), and the URL says which it is
    views = st.tabs(TABS, key=VIEW, bind="query-params")
    pages = (_overview, _trials, _analysis, _tasks, _problem)
    for view, page in zip(views, pages, strict=True):
        if view.open is not False:
            with view:
                page(results)


def _overview(results: StudyResults) -> None:
    choices = list(axis_choices(results))
    default_x, default_y = default_axes(results)
    left, middle, right = st.columns(3)
    with left:
        x = _chosen("Horizontal axis", choices, X, default_x, format_func=_axis_name)
    with middle:
        y = _chosen("Vertical axis", choices, Y, default_y, format_func=_axis_name)
    with right:
        shades = [BY_OUTCOME, *color_choices(results)]
        color = _chosen("Color", shades, COLOR, BY_OUTCOME, format_func=_axis_name)
    dark = st.context.theme.type == "dark"
    found = _trade_offs(results)
    try:
        shade = None if color == BY_OUTCOME else color
        figure = trials_figure(
            results,
            x,
            y,
            color=shade,
            highlight=None if found is None else found.matches,
            reference=None if found is None else found.reference.number,
            dark=dark,
        )
    except K4OptError as exc:
        st.warning(data.plain(exc))
    else:
        figure.update_layout(title=None)  # the page names the study
        _pickable(figure, "trials")
    if found is not None:
        _matches(results, found, noisy=st.query_params.get(NOISE) != "0")
    if results.pareto:
        st.markdown(f"**On the Pareto front** ({len(results.pareto)} trials)")
        st.dataframe(tables.trials(results, results.pareto), hide_index=True, height=300)
    st.subheader("History", anchor=False)
    st.caption("Each objective in the order the trials were proposed; click a trial to open it.")
    objectives = [f"objective:{o.label}" for o in results.problem.objectives]
    columns = st.columns(min(len(objectives), 2))
    for index, objective in enumerate(objectives):
        with columns[index % len(columns)]:
            try:
                figure = history_figure(results, objective, dark=dark)
            except K4OptError as exc:
                st.warning(data.plain(exc))
            else:
                _pickable(figure, f"history-{index}")
    opened = st.session_state.pop(OPENED, None)
    if opened is not None:
        _dialog(results, opened)


def _trade_offs(results: StudyResults) -> TradeOffs | None:
    """The trade-off asked against a reference trial, its controls in an expander, open once a
    reference is chosen; ``None`` until a reference and a rule are."""
    valued = [t.number for t in results.trials if t.objectives is not None]
    held = st.query_params.get(REFERENCE, "")
    linked = int(held) if held.isdigit() and int(held) in valued else None
    key = f"trade-reference-{results.id}"  # another study's reference is not this one's
    state = st.session_state
    if key not in state:
        state[key] = linked
    outcome = {t.number: t.outcome for t in results.trials}
    with st.expander(
        "Trade-offs against a reference trial",
        expanded=state[key] is not None,
        icon=":material/balance:",
    ):
        st.caption(
            "Ask what each objective must give or may give up against a reference trial: the "
            "feasible trials that meet every rule are highlighted, with how each objective "
            "changed. Open a trial to compare with it."
        )
        reference = st.selectbox(
            "Reference trial",
            valued,
            format_func=lambda n: (
                f"trial {n} · {outcome[n]}" + (" · Pareto front" if n in results.pareto else "")
            ),
            placeholder="Choose a trial",
            key=key,
        )
        held_rules = _held_rules(st.query_params.get(RULES, ""))
        rules: list[Rule] = []
        encoded = []
        for index, objective in enumerate(results.problem.objectives):
            kind0, amount0, relative0 = held_rules.get(index, (ANY, 0.0, True))
            name, how, much, scale = st.columns([3, 3, 1, 1], vertical_alignment="center")
            arrow = "▼" if objective.direction == "minimize" else "▲"
            name.markdown(f"{arrow} **{data.plain(objective.metric)}**")
            name.caption(data.plain(objective.label))
            kind = how.selectbox(
                "Rule",
                list(KINDS),
                index=list(KINDS).index(kind0),
                format_func=KINDS.__getitem__,
                key=f"trade-kind-{results.id}-{index}",
                label_visibility="collapsed",
            )
            amount = much.number_input(
                "Amount",
                min_value=0.0,
                value=amount0,
                step=1.0,
                key=f"trade-amount-{results.id}-{index}",
                label_visibility="collapsed",
                disabled=kind in (ANY, "keep"),
            )
            unit = tables.units(results).get(objective.label) or "in its unit"
            relative = (
                scale.selectbox(
                    "In",
                    ["%", unit],
                    index=0 if relative0 else 1,
                    key=f"trade-scale-{results.id}-{index}",
                    label_visibility="collapsed",
                    disabled=kind in (ANY, "keep"),
                )
                == "%"
            )
            for each in RULE_KINDS:  # the kind chosen, as a rule's kind
                if each != kind:
                    continue
                rules.append(
                    Rule(objective.label, each, amount / 100 if relative else amount, relative)
                )
                encoded.append(f"{index}:{kind}:{amount:g}:{'%' if relative else 'u'}")
        noise = st.toggle(
            "Differences within the recorded uncertainties count as none",
            value=st.query_params.get(NOISE) != "0",
            key=f"trade-noise-{results.id}",
            help="A trial improves only beyond the uncertainty of its value and of the "
            "reference's (added in quadrature), and worsening within it is no worsening.",
        )
    _keep_in_url(REFERENCE, None if reference is None else str(reference))
    _keep_in_url(RULES, ",".join(encoded) or None)
    _keep_in_url(NOISE, None if noise else "0")
    if reference is None or not rules:
        return None
    try:
        return trade_offs(results, reference, rules, within_uncertainty=noise)
    except K4OptError as exc:
        st.warning(data.plain(exc))
        return None


def _held_rules(text: str) -> dict[int, tuple[str, float, bool]]:
    """The rules a URL holds, by objective index; whatever does not read as one is left out."""
    held: dict[int, tuple[str, float, bool]] = {}
    for part in text.split(","):
        index, _, rest = part.partition(":")
        kind, _, rest = rest.partition(":")
        amount, _, scale = rest.partition(":")
        try:
            value = float(amount)
        except ValueError:
            continue
        if index.isdigit() and kind in RULE_KINDS and value >= 0 and scale in ("%", "u"):
            held[int(index)] = (kind, value, scale == "%")
    return held


def _keep_in_url(key: str, value: str | None) -> None:
    """``value`` as the URL's parameter ``key``, none if ``None``."""
    if value is None:
        if key in st.query_params:
            del st.query_params[key]
    elif st.query_params.get(key) != value:
        st.query_params[key] = value


def _matches(results: StudyResults, found: TradeOffs, *, noisy: bool) -> None:
    """How many trials meet the trade-off, and each with how it changed; why none, if none."""
    reference = found.reference.number
    if not found.matches:
        hardest = ", ".join(
            f"{data.plain(label.rsplit('/', 1)[-1])} excludes {count}"
            for label, count in sorted(found.excludes.items(), key=lambda item: -item[1])
        )
        noise = " Differences within the uncertainties count as none." if noisy else ""
        st.warning(
            f"No feasible trial meets this against trial {reference}, of {found.compared}: "
            f"{hardest}.{noise}"
        )
        return
    st.markdown(
        f"**{len(found.matches)} of {found.compared} feasible trials** meet this against "
        f"trial {reference}; select one to open it."
    )
    if noisy:
        st.caption(
            "The changes are as measured; the rules compared them less the uncertainty of the two "
            "values, so a change a little beyond a rule's amount can still meet it."
        )
    rows = tables.trade_offs(results, found)
    state = st.session_state
    state.setdefault(PICKS, 0)
    event = st.dataframe(
        rows,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"trade-table-{state[PICKS]}",
        column_config={
            column: st.column_config.NumberColumn(format="%+.1f%%")
            for column in rows[0]
            if column.startswith("Δ ")
        },
    )
    picked = event.selection.rows if event is not None else []
    if picked:
        state[OPENED], state[PICKS] = rows[picked[0]]["Trial"], state[PICKS] + 1
        st.rerun()


def _pickable(figure: go.Figure, name: str) -> None:
    """``figure``, each of whose points is a trial (its ``customdata``): a click on one opens it."""
    state = st.session_state
    state.setdefault(PICKS, 0)
    event = st.plotly_chart(
        figure,
        theme="streamlit",
        key=f"{name}-{state[PICKS]}",
        on_select="rerun",
        selection_mode="points",
    )
    selection = getattr(event, "selection", None) or {}
    for point in selection.get("points", []):
        curve, index = point.get("curve_number"), point.get("point_index")
        numbers = None if curve is None else figure.data[curve].customdata
        if numbers is not None and index is not None:
            state[OPENED], state[PICKS] = int(numbers[index]), state[PICKS] + 1
            st.rerun()


@st.dialog("Trial", width="large")
def _dialog(results: StudyResults, number: int) -> None:
    _trial(results, results.trial(number))


def _compare(results: StudyResults, number: int) -> None:
    """Make trial ``number`` the reference of the trade-off, and open the Overview to show it."""
    st.session_state[f"trade-reference-{results.id}"] = number
    st.session_state[VIEW] = TABS[0]


def _trials(results: StudyResults) -> None:
    counts = {outcome: n for outcome, n in results.outcomes.items() if n}
    shown = st.pills(
        "Outcome",
        list(counts),
        selection_mode="multi",
        default=list(counts),
        format_func=lambda outcome: f"{outcome} · {counts[outcome]}",
        key="trials-outcomes",
    )
    rows = [row for row in tables.trials(results) if row["Outcome"] in shown]
    state = st.session_state
    state.setdefault(ROWS, 0)
    event = st.dataframe(
        rows,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"trials-table-{state[ROWS]}",
    )
    picked = event.selection.rows if event is not None else []
    number = rows[picked[0]]["Trial"] if picked else _linked(results)
    if number is None:
        st.caption("Select a trial for all it recorded; a link to the page names it.")
        return
    assert isinstance(number, int)
    if st.query_params.get(SHOWN) != str(number):
        st.query_params[SHOWN] = str(number)
    with st.container(border=True):
        if _trial(results, results.trial(number), closable=True):
            del st.query_params[SHOWN]
            state[ROWS] += 1
            st.rerun()


def _linked(results: StudyResults) -> int | None:
    """The trial the URL names, if the study has it."""
    held = st.query_params.get(SHOWN, "")
    numbers = {trial.number for trial in results.trials}
    return int(held) if held.isdigit() and int(held) in numbers else None


def _trial(results: StudyResults, trial: TrialResult, *, closable: bool = False) -> bool:
    """All ``trial`` recorded: how it ended, its objectives, candidate, limits and stages; whether
    it was closed, if it can be."""
    title, close = st.columns([6, 1], vertical_alignment="center")
    with title:
        badges = [f":{BADGES.get(trial.outcome, 'red')}-badge[{trial.outcome}]"]
        if trial.number in results.pareto:
            badges.append(":blue-badge[:material/star: Pareto front]")
        st.markdown(f"#### Trial {trial.number} &nbsp; " + " ".join(badges))
    closed = False
    with close:
        if closable:
            closed = st.button("Close", key="trial-close", width="stretch")
    if trial.objectives is not None and st.button(
        "Compare with this trial",
        icon=":material/balance:",
        key=f"trial-compare-{trial.number}",
        on_click=_compare,
        args=(results, trial.number),
        help="Make it the reference of a trade-off, on the Overview",
    ):
        st.rerun()  # from a dialog too: the whole page shows the new reference
    objectives = tables.trial_objectives(trial)
    if objectives:
        for start in range(0, len(objectives), 4):
            columns = st.columns(4)
            for column, row in zip(columns, objectives[start : start + 4], strict=False):
                column.metric(
                    data.plain(str(row["Objective"]).rsplit("/", 1)[-1]),
                    _value(row),
                    help=data.plain(f"{row['Objective']}, {row['Direction']}"),
                    border=True,
                )
    if trial.message:
        st.code(trial.message, language=None, wrap_lines=True)
    left, right = st.columns(2)
    with left:
        st.dataframe(tables.trial_candidate(results, trial), hide_index=True)
        st.dataframe(tables.trial_facts(results, trial), hide_index=True)
    with right:
        for rows in (
            tables.trial_limits(trial),
            tables.trial_stages(trial),
            tables.trial_diagnostics(trial),
        ):
            if rows:
                st.dataframe(rows, hide_index=True)
    return closed


def _value(row: tables.Row) -> str:
    """An objective's value as a tile shows it, with its uncertainty and unit."""
    value, uncertainty, unit = row["Value"], row["±"], row["Unit"]
    text = f"{value:.4g}" if isinstance(value, float) else str(value)
    if isinstance(uncertainty, (int, float)):
        text += f" ± {uncertainty:.2g}"
    return f"{text} {unit}" if unit and unit != "1" else text  # 1: a number, of no unit


def _analysis(results: StudyResults) -> None:
    objectives = [f"objective:{o.label}" for o in results.problem.objectives]
    narrow, _ = st.columns([1, 2])
    with narrow:
        objective = _chosen(
            "Objective", objectives, ANALYZED, objectives[0], format_func=_axis_name
        )
    dark = st.context.theme.type == "dark"
    st.subheader("Parallel coordinates", anchor=False)
    st.caption(
        "Each feasible trial is a line through its parameters, objectives and constraint "
        "margins. Drag along an axis to keep the trials within a range; click the range to drop it."
    )
    _shown(lambda: parallel_figure(results, objective, dark=dark))
    st.subheader("Slice", anchor=False)
    st.caption(
        "The objective against each parameter, colored by trial number; click a trial to open it."
    )
    _shown(lambda: slice_figure(results, objective, dark=dark), "slice")
    st.subheader("Rank", anchor=False)
    if len(results.problem.parameters) < 2:
        st.caption("Ranking the trials over pairs of parameters needs two parameters at least.")
    else:
        st.caption(
            "Each pair of parameters, the feasible trials colored by how they rank in the "
            "objective; nothing is interpolated between them."
        )
        _shown(lambda: rank_figure(results, objective, dark=dark), "rank")
    st.subheader("Distribution", anchor=False)
    st.caption("The share of the trials with a value of each objective at most each value.")
    _shown(lambda: edf_figure(results, dark=dark))
    opened = st.session_state.pop(OPENED, None)
    if opened is not None:
        _dialog(results, opened)


def _shown(make: Callable[[], go.Figure], pickable: str | None = None) -> None:
    """The figure ``make`` makes, or why there is none; its trials opened by a click if
    ``pickable`` names it."""
    try:
        figure = make()
    except K4OptError as exc:
        st.warning(data.plain(exc))
        return
    if pickable is None:
        st.plotly_chart(figure, theme="streamlit")
    else:
        _pickable(figure, pickable)


def _problem(results: StudyResults) -> None:
    st.markdown("**Parameters**")
    st.dataframe(tables.parameters(results), hide_index=True)
    st.markdown("**Objectives**")
    st.dataframe(tables.objectives(results), hide_index=True)
    if results.problem.constraints:
        st.markdown("**Constraints**")
        st.dataframe(tables.constraints(results), hide_index=True)
    st.markdown("**What it was made with**")
    st.dataframe(tables.provenance(results), hide_index=True)
    st.dataframe(tables.evaluators(results), hide_index=True)
    st.caption(
        "Each trial's record stays in the study directory; the snapshot names it by its id "
        "and SHA-256, and holds no file of it."
    )


def _tasks(results: StudyResults) -> None:
    state = st.session_state
    if state.get(OF) != results.id:  # another study: nothing of the last one stays selected
        state[OF], state[SELECTED], state[EXPANDED] = results.id, None, set()
    state.setdefault(CHART, 0)
    expanded: set[str] = state[EXPANDED]
    stack = ALL not in expanded
    graph = task_graph(results, stack=stack, expanded=expanded)
    stacks = [key for key, box in graph.samples.items() if box.is_stack]
    caption, each, again = st.columns([4, 1, 1], vertical_alignment="bottom")
    with caption:
        st.caption(
            "Left to right: the parameters (blue) and what each sets (dashed), the geometry, "
            "each sample (violet) with its stages, the evaluations (aqua), and the objectives "
            "and constraints (orange) they measure. Samples alike but for their options are "
            "stacked in one box: click it to draw each. A dotted task is a diagnostic: recorded, "
            "but no objective or constraint needs it. Times are medians over the trials."
        )
    with each:
        if st.button("Draw each sample", disabled=not stacks, width="stretch"):
            state[EXPANDED] = {ALL}
            st.rerun()
    with again:
        if st.button("Stack alike samples", disabled=not expanded, width="stretch"):
            state[EXPANDED] = set()
            st.rerun()
    try:
        figure = pipeline_figure(results, stack=stack, expanded=expanded, selected=state[SELECTED])
    except K4OptError as exc:
        st.warning(data.plain(exc))
        return
    event = st.plotly_chart(
        figure,
        theme="streamlit",
        key=f"tasks-{state[CHART]}",
        on_select="rerun",
        selection_mode="points",
        config=CHART_CONFIG,
    )
    clicked = _clicked(event, figure)
    if clicked is not None:
        node = next((n for n in graph.nodes if n.id == clicked), None)
        box = graph.samples.get(clicked if node is None else node.sample or "")
        if box is not None and box.is_stack and (node is None or node.samples):
            state[EXPANDED] = {*expanded, box.members[0]}  # a stack: draw each of its samples
        else:
            state[SELECTED] = clicked
        state[CHART] += 1
        st.rerun()
    _selected(results, graph)


def _clicked(event: object, figure: go.Figure) -> str | None:
    """The id of the node clicked, if the chart reports one: a point of the figure's last trace,
    the nodes, whose ``customdata[0]`` is the id."""
    nodes = len(figure.data) - 1
    selection = getattr(event, "selection", None) or {}
    for point in selection.get("points", []):
        index = point.get("point_index")
        if point.get("curve_number") == nodes and index is not None:
            return str(figure.data[nodes].customdata[index][0])
    return None


def _selected(results: StudyResults, graph: TaskGraph) -> None:
    """What the problem says of the task clicked, and how it went in each trial."""
    chosen = st.session_state.get(SELECTED)
    node: GraphNode | None = next((n for n in graph.nodes if n.id == chosen), None)
    if node is None:
        st.caption("Click a task for what it is and how it went in each trial.")
        return
    title, clear = st.columns([5, 1], vertical_alignment="bottom")
    with title:
        st.markdown(f"##### {data.plain(node.title)}")
        where = node.kind if node.sample is None else f"{node.kind} of {node.sample}"
        st.caption(data.plain(where + ("" if node.required else ", a diagnostic")))
    with clear:
        if st.button("Clear", width="stretch", key="tasks-clear"):
            st.session_state[SELECTED] = None
            st.session_state[CHART] += 1
            st.rerun()
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(tables.facts(node), hide_index=True)
    rows = tables.task_trials(results, node)
    with right:
        if rows:
            st.dataframe(rows, hide_index=True)
        else:
            st.caption("Its value in each trial is in the table of the Trials tab.")


def _axis_name(key: str) -> str:
    if key == BY_OUTCOME:
        return "outcome"
    if key == TRIAL:
        return "trial number"
    kind, _, name = key.partition(":")
    return f"{name} ({kind})"


main(data.current())
