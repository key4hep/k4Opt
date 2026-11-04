"""The Trade-offs tab: which trials give up at most so much for at least so much against a
reference trial. The rules are beside the figure they highlight, the trials that meet them below.

The URL holds the trade-off: the reference trial (``ref``), a rule for each objective by its index
(``rules=0:worsen:5:%,1:improve:10:u``) and ``noise=0`` to compare values without their
uncertainties."""

from __future__ import annotations

import streamlit as st

from k4opt.errors import K4OptError
from k4opt.optimization import RULE_KINDS, Rule, StudyResults, TradeOffs, trade_offs
from k4opt.visualization import names, trials_figure

import labels
import tables
import trial
import widgets

REFERENCE, RULES, NOISE = "ref", "rules", "noise"
ANY = "any"
# what each kind of rule is called, read after the objective's name
KINDS = {
    ANY: "may do anything",
    "improve": "improves by ≥",
    "worsen": "worsens by ≤",
    "keep": "does not worsen",
}


def render(results: StudyResults) -> None:
    rules, drawn = st.columns([2, 3], gap="large")
    with rules:
        found = _trade_offs(results)
    with drawn:
        x, y, _ = widgets.axes(results, color=False)
        try:
            figure = trials_figure(
                results,
                x,
                y,
                highlight=None if found is None else found.matches,
                reference=None if found is None else found.reference.number,
                dark=st.context.theme.type == "dark",
            )
        except K4OptError as exc:
            st.warning(labels.plain(exc))
        else:
            figure.update_layout(title=None)  # the page names the study
            clicked = widgets.chart(figure, "trade")
            if clicked is not None:
                trial.show(clicked)
    if found is not None:
        _matches(results, found, noisy=st.query_params.get(NOISE) != "0")


def _trade_offs(results: StudyResults) -> TradeOffs | None:
    """The trade-off asked against a reference trial, from its controls; ``None`` until a
    reference and a rule are chosen."""
    valued = [t.number for t in results.trials if t.objectives is not None]
    held = st.query_params.get(REFERENCE, "")
    linked = int(held) if held.isdigit() and int(held) in valued else None
    key = trial.reference_key(results)
    state = st.session_state
    if key not in state:
        state[key] = linked
    outcome = {t.number: t.outcome for t in results.trials}
    called, shown = names(results.problem), tables.units(results)
    with st.container(border=True):
        st.markdown("**Compare against a trial**")
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
            arrow = "▼" if objective.direction == "minimize" else "▲"
            st.markdown(
                f"{arrow} **{labels.plain(called.metric(objective.label))}**",
                help=labels.plain(f"{objective.label}, {objective.direction}"),
            )
            how, much, scale = st.columns([4, 3, 2], vertical_alignment="center")
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
            unit = labels.unit(shown.get(objective.label)) or "absolute"
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
            "Differences within the uncertainties count as none",
            value=st.query_params.get(NOISE) != "0",
            key=f"trade-noise-{results.id}",
            help="A trial improves only beyond the uncertainty of its value and of the "
            "reference's (added in quadrature), and worsening within it is no worsening.",
        )
    widgets.keep_in_url(REFERENCE, None if reference is None else str(reference))
    widgets.keep_in_url(RULES, ",".join(encoded) or None)
    widgets.keep_in_url(NOISE, None if noise else "0")
    if reference is None:
        st.caption("Choose a reference trial, or open a trial and compare with it.")
        return None
    if not rules:
        st.caption("Give a rule to at least one objective.")
        return None
    try:
        return trade_offs(results, reference, rules, within_uncertainty=noise)
    except K4OptError as exc:
        st.warning(labels.plain(exc))
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


def _matches(results: StudyResults, found: TradeOffs, *, noisy: bool) -> None:
    """How many trials meet the trade-off, and each with how it changed; why none, if none."""
    reference = found.reference.number
    if not found.matches:
        called = names(results.problem)
        hardest = ", ".join(
            f"{labels.plain(called.metric(label))} excludes {count}"
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
    picked = widgets.table(
        rows,
        "trade-table",
        config={
            column: st.column_config.NumberColumn(column, format="%+.1f%%")
            for column in rows[0]
            if column.startswith("Δ ")
        },
    )
    if picked is not None:
        trial.show(int(str(rows[picked]["Trial"])))
