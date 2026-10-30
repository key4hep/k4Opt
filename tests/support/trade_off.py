"""``test.trade_off``: two objectives that pull apart, ``a = x`` and ``b = 1/x`` of the candidate's
first parameter, for the optimizer's tests. It reads no stage, so it needs no simulation. Below
``invalid_below`` its result is ``invalid``, below ``failed_below`` ``failed``."""

from k4opt.evaluation import Evaluator


class TradeOff(Evaluator):
    name = "test.trade_off"
    version = "1"
    settings = {"invalid_below": 0.0, "failed_below": 0.0}
    metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

    def evaluate(self, request):
        x, *_ = request.candidate.values()
        if x < request.settings["failed_below"]:
            self.fail(f"x = {x} is below failed_below")
        if x < request.settings["invalid_below"]:
            self.invalid(f"x = {x} is below invalid_below")
        return {("analysis", "a"): x, ("analysis", "b"): 1 / x}
