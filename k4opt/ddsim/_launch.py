"""Internal: start a program that dies with the thread that started this launcher (Linux).

Run by its file path, ``python -I _launch.py PARENT PROGRAM ARGS...``, from the thread of the
process PARENT that starts the program (:func:`k4opt.ddsim.run._execute`). It has the kernel send
it SIGKILL once that thread ends (``PR_SET_PDEATHSIG``), checks that PARENT is still its parent
and becomes PROGRAM: ``exec`` keeps the process, its group and that signal. So no code runs
between fork and exec in PARENT, whose libraries (ROOT) may have started threads of their own.
It needs nothing but the standard library: ``-I`` keeps the environment's packages out.
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys
from collections.abc import Sequence

PR_SET_PDEATHSIG = 1  # prctl(2): the signal a process gets when its parent thread ends


def main(argv: Sequence[str]) -> str:
    """Become the program ``argv[1:]``, to be killed once the parent ``argv[0]`` ends; why it
    cannot, if it cannot (it does not return otherwise)."""
    parent, program = int(argv[0]), list(argv[1:])
    if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        error = os.strerror(ctypes.get_errno())
        return f"k4opt: {program[0]} cannot be made to end with its worker process (prctl: {error})"
    if os.getppid() != parent:
        return f"k4opt: the worker process that starts {program[0]} ended before it started"
    os.execv(program[0], program)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
