"""``test.candidate_check``: records the candidate of every request its ``check()`` and
``evaluate()`` see, and refuses in ``check()`` a first parameter equal to its ``reject``
setting, for the optimizer's tests. It reads no stage, so it needs no simulation. Every instance
is kept, in order, in ``CandidateCheck.instances``."""

from k4opt.evaluation import EvaluationContractError, Evaluator


class CandidateCheck(Evaluator):
    name = "test.candidate_check"
    version = "1"
    settings = {"reject": -1.0}
    metrics = {("analysis", "seen"): "1"}

    instances = []

    def __init__(self):
        self.checked = []
        self.evaluated = []
        CandidateCheck.instances.append(self)

    def check(self, request):
        super().check(request)
        self.checked.append(dict(request.candidate))
        first, *_ = request.candidate.values()
        if first == request.settings["reject"]:
            raise EvaluationContractError(f"this evaluator cannot measure {first}")

    def evaluate(self, request):
        self.evaluated.append(dict(request.candidate))
        first, *_ = request.candidate.values()
        return {("analysis", "seen"): first}
