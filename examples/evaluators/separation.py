"""Toy binned pi0 rejection at a requested photon efficiency, without systematics.

Accept widths through the first whole bin reaching the requested signal efficiency. Report
rejection as the fraction of background rejected (1 - efficiency), finite even for zero passes.
There is no interpolation within a bin and no claim of independent validation of the cut.
"""

import json
import math
from itertools import pairwise

from k4opt.evaluation import Evaluator, Role


class Separation(Evaluator):
    name, version = "calo.separation", "1"
    roles = {"signal": Role(artifacts=("histogram",)), "background": Role(artifacts=("histogram",))}
    settings = {"signal_efficiency": float}
    metrics = {
        ("analysis", "background_rejection"): "1",
        ("analysis", "signal_efficiency"): "1",
        ("analysis", "width_cut"): "mm",
    }

    def check(self, request):
        if not 0 < request.settings["signal_efficiency"] <= 1:
            self.fail("signal_efficiency must be in (0, 1]")

    def _histogram(self, item):
        try:
            data = json.loads(item.artifacts["histogram"].path.read_text())
            edges, counts = data["edges"], data["counts"]
            under, over = data["underflow"], data["overflow"]
            if (
                not isinstance(edges, list)
                or not isinstance(counts, list)
                or len(edges) != len(counts) + 1
                or not counts
                or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                    for x in edges
                )
                or any(b <= a for a, b in pairwise(edges))
                or any(
                    isinstance(n, bool) or not isinstance(n, int) or n < 0
                    for n in [*counts, under, over]
                )
            ):
                self.fail(f"{item.label}: malformed width histogram")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.fail(f"{item.label}: cannot read width histogram: {exc}")
        if under:
            self.invalid(f"{item.label}: histogram underflow; lower the first bin edge")
        if sum(counts) + over == 0:
            self.invalid(f"{item.label}: empty width histogram")
        return edges, counts, over

    def evaluate(self, request):
        (signal,), (background,) = request.roles["signal"], request.roles["background"]
        edges, s, signal_overflow = self._histogram(signal)
        bedges, b, background_overflow = self._histogram(background)
        if edges != bedges:
            self.fail("signal and background histograms need identical bin edges")
        total = sum(s) + signal_overflow
        selected, accepted = 0, 0
        for index, count in enumerate(s):
            selected += count
            accepted += b[index]
            if selected / total >= request.settings["signal_efficiency"]:
                return {
                    ("analysis", "background_rejection"): 1
                    - accepted / (sum(b) + background_overflow),
                    ("analysis", "signal_efficiency"): selected / total,
                    ("analysis", "width_cut"): edges[index + 1],
                }
        self.invalid("signal efficiency needs overflow events; increase the last bin edge")
