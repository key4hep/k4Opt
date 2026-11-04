"""What the figures and the dashboard call the parameters, samples, objectives and constraints of a
problem: the ``title`` the problem gives one, its name otherwise, and never two of a kind alike.

A sample is called by its title, or its name. A parameter is called by its title, or its name;
two parameters titled alike are each called by the title and the name (``Cell size
(ecal_barrel_size)``). An objective is called by its title, or its metric, and only where two
objectives would be called alike is more said: the sample it is measured on (by that sample's own
title or name), then its whole label; so is a constraint among the constraints. A title names
what is measured, the sample is the sample's to name: two objectives titled ``Angular residual``
on the samples ``10 GeV γ`` and ``50 GeV γ`` are ``Angular residual · 10 GeV γ`` and ``Angular
residual · 50 GeV γ``. An objective and a constraint on one metric are each called by their own
title.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from k4opt.problem import Constraint, Objective, ProblemSpec


@dataclass(frozen=True, slots=True)
class Names:
    """What each thing of a problem is called: each parameter and each sample by its name, each
    objective and each constraint by the label of its metric."""

    parameters: Mapping[str, str]
    samples: Mapping[str, str]
    objectives: Mapping[str, str]
    constraints: Mapping[str, str]

    def parameter(self, name: str) -> str:
        return self.parameters.get(name, name)

    def sample(self, name: str) -> str:
        return self.samples.get(name, name)

    def objective(self, label: str) -> str:
        return self.objectives.get(label, label)

    def constraint(self, label: str) -> str:
        return self.constraints.get(label, label)


def names(spec: ProblemSpec) -> Names:
    """What the things of ``spec`` are called."""
    samples = {name: sample.title or name for name, sample in spec.samples.items()}
    parameters = {
        p.name: [p.name] if p.title is None else [p.title, f"{p.title} ({p.name})"]
        for p in spec.parameters
    }
    return Names(
        _distinct(parameters),
        samples,
        _distinct(_measured(spec.objectives, samples)),
        _distinct(_measured(spec.constraints, samples)),
    )


def _measured(
    items: Sequence[Objective | Constraint], samples: Mapping[str, str]
) -> dict[str, list[str]]:
    """What each item may be called, by its label, from the first choice on."""
    options = {}
    for item in items:
        called = item.title or item.metric
        on = [] if item.sample is None else [f"{called} · {samples.get(item.sample, item.sample)}"]
        options[item.label] = list(dict.fromkeys([called, *on, item.label]))
    return options


def _distinct(options: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """For each key, the first of its ``options`` no other key is called by: each key called alike
    with another takes its next option, until none is (the last option of each holds the key,
    which is distinct)."""
    level = dict.fromkeys(options, 0)
    while True:
        called = {key: options[key][level[key]] for key in options}
        alike = Counter(called.values())
        clash = [k for k in options if alike[called[k]] > 1 and level[k] < len(options[k]) - 1]
        if not clash:
            return called
        for key in clash:
            level[key] += 1
