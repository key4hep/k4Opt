"""``test.toy_significance``, a combined evaluator: the expected significance of a signal over
its backgrounds, from the selected and generated sums of weights each sample's ``test.yields``
measures and the cross section the problem gives each sample. For the framework's tests.

A TOY, not physics-grade: one bin, the Asimov significance ``sqrt(2((s + b) ln(1 + s/b) - s))``,
no systematic uncertainty. The cross sections are the problem's ``properties`` of each sample,
which k4opt delivers and never reads; normalizing is this evaluator's own work.
"""

import math
from collections.abc import Mapping

from k4opt.evaluation import Evaluator, Role

SELECTED = ("analysis", "selected.sum_weights")
GENERATED = ("analysis", "generated.sum_weights")


class ToySignificance(Evaluator):
    name = "test.toy_significance"
    version = "1"
    roles = {
        "signal": Role(metrics=[SELECTED, GENERATED]),
        "backgrounds": Role(many=True, metrics=[SELECTED, GENERATED]),
    }
    settings = {"luminosity_invab": float}
    metrics = {("analysis", "expected_significance"): "1", ("analysis", "expected_background"): "1"}

    def evaluate(self, request):
        luminosity = 1e3 * request.settings["luminosity_invab"]  # fb^-1

        def expected(item):
            cross_section = item.properties.get("cross_section")
            if not isinstance(cross_section, Mapping) or cross_section.get("unit") != "fb":
                self.fail(f"{item.sample}: properties.cross_section must be given in fb")
            generated = item.measurement(*GENERATED).value
            if generated <= 0:  # signed weights can sum to zero or below
                self.invalid(f"{item.sample}: the generator weights sum to {generated}")
            return (
                cross_section["value"] * luminosity * item.measurement(*SELECTED).value / generated
            )

        (signal,) = request.roles["signal"]
        s = expected(signal)
        b = sum(expected(item) for item in request.roles["backgrounds"])
        if b <= 0:
            self.invalid("no expected background: the toy significance is undefined")
        z = math.sqrt(2 * ((s + b) * math.log1p(s / b) - s))
        return {("analysis", "expected_significance"): z, ("analysis", "expected_background"): b}
