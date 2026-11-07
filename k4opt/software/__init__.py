"""Software component: what candidates run on, as one identity.

``KEY4HEP_STACK`` names a place, and a nightly's slot holds another build every week. So what a
session runs on is identified by what is there: the build of the stack (an LCG view's manifests
and when it was built), the system, and every environment variable not known to be
irrelevant. Two sessions with equal :attr:`Software.id` run on the same software, wherever the
stack is; :meth:`Software.differences` says what differs, by name. See ``docs/software.md``.

Typical use::

    from k4opt.software import Software

    opened = Software.now()
    print(opened.build.label, opened.id)
    print(opened.differences(Software.now()))  # [] unless something changed since

The command line is ``k4opt software show``.
"""

from __future__ import annotations

from k4opt.software.errors import SoftwareError
from k4opt.software.identity import IGNORED, Build, Software, System, compared_variables

__all__ = ["IGNORED", "Build", "Software", "SoftwareError", "System", "compared_variables"]
