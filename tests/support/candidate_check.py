"""``test.candidate_check``: records the value of its setting ``x``, which a parameter of the
problem sets, in every request its ``check()`` (in ``CandidateCheck.checked``) and ``evaluate()``
see, and rejects in ``check()`` a value equal to its ``reject`` setting (raises an
``EvaluationContractError`` for one equal to its ``refuse`` setting), for the optimizer's tests.
It reads no stage, so it needs no simulation. Every instance is kept, in order, in
``CandidateCheck.instances``."""

from k4opt.evaluation import EvaluationContractError, Evaluator


class CandidateCheck(Evaluator):
    name = "test.candidate_check"
    version = "1"
    settings = {"x": float, "reject": -1.0, "refuse": -1.0}
    metrics = {("analysis", "seen"): "1"}

    instances = []
    checked = []

    def __init__(self):
        self.evaluated = []
        CandidateCheck.instances.append(self)

    @classmethod
    def check(cls, request):
        super().check(request)
        x = request.settings["x"]
        CandidateCheck.checked.append(x)
        if x == request.settings["refuse"]:
            raise EvaluationContractError(f"this evaluator cannot work with {x}")
        if x == request.settings["reject"]:
            cls.reject(f"this evaluator cannot measure {x}")

    def evaluate(self, request):
        x = request.settings["x"]
        self.evaluated.append(x)
        return {("analysis", "seen"): x}
