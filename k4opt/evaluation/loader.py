"""Load an evaluator class from a file or an importable module, as :mod:`k4opt.usercode` loads and
identifies any class of the user's code."""

from __future__ import annotations

from k4opt.errors import CodeLoadError
from k4opt.evaluation.errors import EvaluatorLoadError
from k4opt.evaluation.protocol import EvaluatorProtocol
from k4opt.usercode import load_class


def load_evaluator(reference: str) -> type[EvaluatorProtocol]:
    """The evaluator class that ``reference`` names, ``FILE.py:Class`` or ``module:Class``.

    Loaded and identified by :func:`~k4opt.usercode.load_class`: ``FILE.py`` is a path (a leading
    ``~`` is expanded) whose directory is added to ``sys.path``, anything else an importable
    module name; a file is executed once per process, and a file or a module it imports that
    changed since is refused, as is a module beside it of a name the process imported from
    elsewhere. :func:`~k4opt.usercode.source_of` gives the source it is identified by.

    The class must have an ``evaluate`` method; it is not created here. Whoever evaluates with
    it creates an instance (without arguments) whenever it needs one, which must have ``name``
    and ``version`` too. The file is executed as ordinary code: only name evaluators you trust.
    Raises :class:`EvaluatorLoadError`.
    """
    try:
        cls = load_class(reference, "evaluator", ("evaluate",))
    except CodeLoadError as exc:
        raise EvaluatorLoadError(str(exc)) from exc
    return cls  # a class with evaluate(), as checked
