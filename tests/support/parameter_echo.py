"""``test.parameter_echo``: the value of the candidate's first parameter, reported as an
``analysis`` metric. It requires no pipeline stage, so it needs no simulation; for the
framework's tests."""

from k4opt.evaluation import Evaluator


class ParameterEcho(Evaluator):
    name = "test.parameter_echo"
    version = "1"
    metrics = {("analysis", "candidate.first_value"): "1"}

    def evaluate(self, request):
        first, *_ = request.candidate.values()
        return {("analysis", "candidate.first_value"): first}
