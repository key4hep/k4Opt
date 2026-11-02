"""``test.lifecycle``: writes a line to the file the environment variable ``LIFECYCLE_LOG`` names
for each of its ``check`` (``check <pid> <x>``), each instance it is created as
(``init <pid> <id>``) and each ``evaluate`` (``evaluate <pid> <id> <x> <calls before>``), so that
a test sees which process did what, for which candidate, on which instance, whatever process it
ran in. It reads no stage and measures its setting ``x``, a length that a parameter of the problem
sets."""

import os

from k4opt.evaluation import Evaluator, Quantity


def _log(line):
    with open(os.environ["LIFECYCLE_LOG"], "a") as log:  # one write per line: lines never mix
        log.write(f"{line}\n")


class Lifecycle(Evaluator):
    name = "test.lifecycle"
    version = "1"
    settings = {"x": Quantity("mm")}
    metrics = {("analysis", "x"): "mm"}

    def __init__(self):
        self.calls = 0  # what an instance keeps: a new one for every evaluation starts from 0
        _log(f"init {os.getpid()} {id(self)}")

    @classmethod
    def check(cls, request):
        super().check(request)
        _log(f"check {os.getpid()} {request.settings['x']}")

    def evaluate(self, request):
        _log(f"evaluate {os.getpid()} {id(self)} {request.settings['x']} {self.calls}")
        self.calls += 1
        return {("analysis", "x"): request.settings["x"]}
