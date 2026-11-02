"""``test.candidate_check``: records the value of its setting ``x``, which a parameter of the
problem sets, in every request its ``check()`` and ``evaluate()`` see, and refuses in ``check()``
a value equal to its ``reject`` setting, for the optimizer's tests. It reads no stage, so it needs
no simulation. Every instance is kept, in order, in ``CandidateCheck.instances``."""

from k4opt.evaluation import EvaluationContractError, Evaluator


class CandidateCheck(Evaluator):
    name = "test.candidate_check"
    version = "1"
    settings = {"x": float, "reject": -1.0}
    metrics = {("analysis", "seen"): "1"}

    instances = []

    def __init__(self):
        self.checked = []
        self.evaluated = []
        CandidateCheck.instances.append(self)

    def check(self, request):
        super().check(request)
        x = request.settings["x"]
        self.checked.append(x)
        if x == request.settings["reject"]:
            raise EvaluationContractError(f"this evaluator cannot measure {x}")

    def evaluate(self, request):
        x = request.settings["x"]
        self.evaluated.append(x)
        return {("analysis", "seen"): x}
