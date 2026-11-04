"""Internal: the marks k4opt sets on a trial of the optimizer, named once.

A trial is told in several writes to the storage, any of which an interruption (Ctrl-C,
SIGTERM, SIGKILL) can be the last. The writes are ordered so that every prefix of them reads
back as the trial it was meant to be, or as an ``interrupted`` one: the mark that says what kind
of trial it is (``STATUS``) is written after what that kind needs, and an unrecorded trial is read
from the marks below alone (:func:`~k4opt.optimization._optuna.read.trial_result`).
"""

from __future__ import annotations

PREFIX = "k4opt."

DIR = "k4opt.dir"
"""The trial's directory, relative to the study: set as soon as the trial is asked for."""
SAMPLER_SEED = "k4opt.sampler_seed"
"""The seed of the sampler that proposed it."""
STATUS = "k4opt.status"
"""The status of its record, or ``rejected``."""
RECORD = "k4opt.record"
"""Its record: id, SHA-256 and path."""
MESSAGE = "k4opt.message"
"""Its message, shortened."""
DIAGNOSTICS = "k4opt.diagnostics"
"""How many diagnostics did not succeed."""
RECOVERED = "k4opt.recovered"
"""Finished by a resume, not by the session that asked for it."""
FEASIBLE = "k4opt.feasible"
"""Whether a valid trial keeps every limit."""
REJECTED_BY = "k4opt.rejected_by"
"""The evaluation that rejected its candidate."""
ERROR = "k4opt.error"
"""What stopped the study at this trial."""

REJECTION = (REJECTED_BY, MESSAGE, STATUS)
"""What a rejected trial is told, in this order: ``STATUS`` last, so a trial interrupted before
it has no status and reads as interrupted."""

UNRECORDED = frozenset({DIR, SAMPLER_SEED, RECOVERED, ERROR})
"""What a trial without a record carries of k4opt's: an interrupted one, or one that stopped the
study before its record was told."""

REJECTED = UNRECORDED | set(REJECTION)
"""What a rejected trial carries of k4opt's, and nothing else of it."""

PARTLY_REJECTED = UNRECORDED | set(REJECTION[:-1])
"""What a trial interrupted while it was told its rejection may carry: a prefix of
:data:`REJECTION` without ``STATUS``."""
