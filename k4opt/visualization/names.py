"""What the figures and the dashboard call the parameters, samples, objectives and constraints of a
problem: the ``title`` the problem gives one, its name otherwise.

A parameter or a sample is called by its title, or its name. An objective or a constrained metric
is called by its title, or its metric, and only where two would be called alike is more said: the
sample it is measured on (by that sample's own title or name), then its whole label. A title names
what is measured, the sample is the sample's to name: two objectives titled ``Angular residual``
on the samples ``10 GeV γ`` and ``50 GeV γ`` are ``Angular residual · 10 GeV γ`` and ``Angular
residual · 50 GeV γ``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from k4opt.problem import Constraint, Objective, ProblemSpec


@dataclass(frozen=True, slots=True)
class Names:
    """What each thing of a problem is called: each parameter and each sample by its name, each
    objective and constrained metric by its label (one of each, where an objective and a
    constraint name one metric)."""

    parameters: Mapping[str, str]
    samples: Mapping[str, str]
    measured: Mapping[str, str]

    def parameter(self, name: str) -> str:
        return self.parameters.get(name, name)

    def sample(self, name: str) -> str:
        return self.samples.get(name, name)

    def metric(self, label: str) -> str:
        return self.measured.get(label, label)


def names(spec: ProblemSpec) -> Names:
    """What the things of ``spec`` are called."""
    samples = {name: sample.title or name for name, sample in spec.samples.items()}
    measured: dict[str, Objective | Constraint] = {}
    items: list[Objective | Constraint] = [*spec.objectives, *spec.constraints]
    for item in items:
        if item.label not in measured or measured[item.label].title is None:
            measured[item.label] = item  # the first titled of those of one metric
    options = {}
    for label, item in measured.items():
        called = item.title or item.metric
        on = [] if item.sample is None else [f"{called} · {samples.get(item.sample, item.sample)}"]
        options[label] = list(dict.fromkeys([called, *on, label]))
    return Names({p.name: p.title or p.name for p in spec.parameters}, samples, _distinct(options))


def _distinct(options: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """For each key, the first of its ``options`` no other key is called by: each key called alike
    with another takes its next option, until none is (the last option of each, a label, is
    distinct)."""
    level = dict.fromkeys(options, 0)
    while True:
        called = {key: options[key][level[key]] for key in options}
        alike = Counter(called.values())
        clash = [k for k in options if alike[called[k]] > 1 and level[k] < len(options[k]) - 1]
        if not clash:
            return called
        for key in clash:
            level[key] += 1
