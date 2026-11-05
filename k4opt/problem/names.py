"""What the parameters, samples, objectives and constraints of a problem are called, wherever they
are shown to a person (a study's outline, the figures, the dashboard): the ``title`` the problem
gives one, its name otherwise, and never two of a kind alike.

A parameter or a sample is called by its title, or its name; two parameters (or two samples)
titled alike are each called by the title and the name (``Cell size (ecal_barrel_size)``). An
objective is called by its title, or its metric, and only where two objectives would be called
alike is more said: the sample it is measured on (by that sample's own title or name), then its
whole label; so is a constraint among the constraints. A title names
what is measured, the sample is the sample's to name: two objectives titled ``Angular residual``
on the samples ``10 GeV γ`` and ``50 GeV γ`` are ``Angular residual · 10 GeV γ`` and ``Angular
residual · 50 GeV γ``. An objective and a constraint on one metric are each called by their own
title.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from k4opt.problem.models import Constraint, Objective, ProblemSpec


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
    samples = _distinct({name: _titled(name, s.title) for name, s in spec.samples.items()})
    return Names(
        _distinct({p.name: _titled(p.name, p.title) for p in spec.parameters}),
        samples,
        _distinct(_measured(_objectives(spec), samples)),
        _distinct(_measured(spec.constraints, samples)),
    )


def _objectives(spec: ProblemSpec) -> list[Objective]:
    """The objectives of ``spec``, then those only an inner level names."""
    inner = [level.objective for level in spec.levels if level.objective is not None]
    return list({o.label: o for o in [*inner, *spec.objectives]}.values())


def _titled(name: str, title: str | None) -> list[str]:
    """What a parameter or sample may be called: its title, then the title and its name."""
    return [name] if title is None else [title, f"{title} ({name})"]


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
