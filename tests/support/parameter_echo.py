"""``test.parameter_echo``: the value of its setting ``x``, a length in mm that a parameter of the
problem sets, reported as an ``analysis`` metric. It reads no pipeline stage, so it needs no
simulation; for the framework's tests."""

from k4opt.evaluation import Evaluator, Quantity


class ParameterEcho(Evaluator):
    name = "test.parameter_echo"
    version = "1"
    settings = {"x": Quantity("mm")}
    metrics = {("analysis", "candidate.first_value"): "1"}

    def evaluate(self, request):
        return {("analysis", "candidate.first_value"): request.settings["x"]}
