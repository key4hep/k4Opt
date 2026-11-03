# Publication: `k4opt publish`

A study lives in its directory, with Optuna's storage, on the machine it ran on. A
**publication** is a directory of portable, read-only snapshots of studies that anyone, or the
[dashboard](dashboard.md), can read without Optuna, without the study directory and without the
Key4hep stack. It can be served as static files (WebEOS, any web server, an object store's web
endpoint) or copied anywhere.

```bash
k4opt publish runs/cld-ecal-study --output published/
```

```text
cld-ecal-granularity-3f9c2a1b: 30 trials (28 feasible, 1 invalid, 1 failed), published 2026-11-03T10:00:00+00:00 as published/studies/3f9c2a1b-.../snapshots/9b1c0e7d2a4f5e68.json
```

## What a publication holds

```text
published/
├── catalog.json                                  k4opt.catalog/1: the one file rewritten
└── studies/
    └── 3f9c2a1b-5d0e-4f43-9a52-77c1d2e0b6a4/     a study, by its id
        └── snapshots/
            └── 9b1c0e7d2a4f5e68.json             the study as published, never changed
```

**A snapshot** is the study's [JSON form](optimization.md#export-formats), byte for byte what
`k4opt study export --format json` writes: the problem definition, every trial with its
candidate, objectives (values, units, uncertainties), limits (values, margins, verdicts),
outcome, times and messages, the Pareto front, and `study.json` without its locations. It is
named by the first 16 hex digits of its SHA-256, so a name stands for one content: a snapshot is
never changed, a later state of the study is another file, and a reader that has a snapshot by
its name has exactly what was published.

What leaves the machine with a snapshot, and what does not:

| Published | Not published |
| --- | --- |
| the problem's definition, as written | the problem file's path |
| each trial's candidate, assessment, outcome, times, message and diagnostics | the trial directories and every file in them (records, events, geometries, logs) |
| each record's id and SHA-256, to identify it | the record's path or any link to it: the record stays in the study |
| size and SHA-256 of every file the problem names, the evaluators' versions, source hashes and git commits, the geometry's compact file, fingerprint, targets and the hashes of what its external references read | the paths of those files, the geometry's source root, `K4GEO` |
| `KEY4HEP_STACK` (part of the study's identity) and k4opt's version, hash and commit | the optimizer's storage (a journal name, or a database URL with its host and user) |
| the optimizer, its sampler and seed, its directions and distributions | anything of `study.json` not on the list (`PORTABLE` in `k4opt/optimization/_study.py`) |

**Publishing makes all of it readable to whoever can read the publication.** The problem
definition and the messages are published as they are: a stage `file:` given as an absolute
path, an evaluator setting naming a file, or an evaluator that writes a path into its message
puts that path into the snapshot. `k4opt publish` therefore names, before it writes anything,
every absolute path outside `/cvmfs` that the problem's definition and the trials' messages
hold:

```text
k4opt: warning: trial 7's message holds the local path /home/me/logs/ddsim.log; it is published
```

Publish into a staging directory first if a study may hold what should not be public, look at
what it says, and serve or copy the directory only then. Nothing is published automatically: a
publication holds only what someone ran `k4opt publish` for.

**The catalog** names each study once, by its id, with what a list of studies shows and sorts
by (its problem, the compact file of the detector it starts from, the Key4hep stack) and the
snapshot it is published at:

```json
{
  "format": "k4opt.catalog/1",
  "studies": [
    {
      "id": "3f9c2a1b-5d0e-4f43-9a52-77c1d2e0b6a4",
      "name": "cld-ecal-granularity-3f9c2a1b",
      "problem": "cld-ecal-granularity",
      "geometry": "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml",
      "key4hep": "nightly 2026-10-20",
      "created": "2026-10-30T12:00:00+00:00",
      "published": "2026-11-03T10:00:00+00:00",
      "trials": 30,
      "outcomes": {"feasible": 28, "infeasible": 0, "invalid": 1, "failed": 1, "rejected": 0, "interrupted": 0, "error": 0, "running": 0},
      "parameters": 1,
      "objectives": 2,
      "constraints": 0,
      "snapshot": {"path": "studies/3f9c2a1b-.../snapshots/9b1c0e7d2a4f5e68.json", "sha256": "9b1c0e7d...", "bytes": 48213}
    }
  ]
}
```

`running` counts the trials that were running when the snapshot was published: a study in
progress can be published, and published again as it goes on. Nothing says whether a study is
"finished": a study can always be resumed.

## Publishing

`k4opt publish DIR --output PUBLICATION [--storage URL]` reads the study as
[`k4opt study`](optimization.md#inspecting-and-exporting-a-study) does, checked in full and
unchanged (every record against the study and the optimizer, every assessment recomputed from
its measurements), then:

1. refuses a publication directory inside the study (also through a symlink); one that does not
   exist is made;
2. names the local paths the snapshot carries (above);
3. writes the snapshot, whole and on disk before it gets its name, as every k4opt record is
   written; one already there with the same bytes is kept, one with other bytes is refused;
4. under `.catalog.lock`, reads `catalog.json` (none yet: an empty catalog; a malformed one is
   refused and left as it is) and writes it again whole with the study's entry, replacing the
   old one by a rename, so a reader sees the old catalog or the new one.

Publishing the same state of a study again changes nothing, not even the catalog's time. A
**later state** (more trials, or more of them ended) replaces the study's entry; its earlier
snapshot stays, unnamed, so a reader that was reading it still reads it whole. An **earlier
state** is refused: a publisher that read the study before another published it never takes the
catalog back. One publisher per study is the expected way of working, and this is its
safeguard. `.catalog.lock` keeps two publications of different studies into one catalog from
losing each other's entry; a publication killed while it held the lock leaves it, and the next
one is refused with the file named, to remove once none runs.

A publication is a plain directory, so it can be published straight into its served location
(an EOS directory behind WebEOS) or published locally and copied there; copy the snapshots
before the catalog, and a reader never meets a catalog entry whose snapshot is not there yet.

## Reading a publication

```python
from k4opt.publication import open_publication

publication = open_publication("https://k4opt-data.web.cern.ch")  # or a directory
catalog = publication.catalog()
for entry in catalog.studies:
    print(entry.name, entry.trials, entry.outcomes["feasible"])
study = publication.study(catalog.studies[0])  # a StudyResults
study.problem, study.trials, study.pareto
```

A directory and its URL give the same studies: each file is read by the path its catalog gives,
never by listing a directory. Over HTTP(S) the server's certificate is verified, a redirect is
refused (a login page is no catalog, so serve the publication without one), and every read is
bounded in size and in time: 30 s to connect and between two parts of an answer, 120 s for the
whole of it, however steadily it trickles in.

**Size.** A snapshot holds about 1.5 kB per trial of a study of one parameter and two
objectives, and is at most 32 MiB (`MAX_SNAPSHOT_BYTES`), some 20000 such trials: a study read
back takes about 1.5 times its snapshot's size in memory, and reading it about 5 times at its
peak. `k4opt publish` refuses a larger one before writing anything, and a reader refuses one a
catalog names as larger. `Cache-Control: no-cache` asks caches between for the current
catalog; a snapshot never changes.

**What is checked, and where.** `k4opt publish` publishes only what `load_study` has read and
checked against the records. A reader checks the published form, not the measurements, whose
records are not published:

- the catalog: every entry well formed, the study's id a UUID, and the snapshot path exactly the
  one the entry's id and SHA-256 give, so that no entry can name a file outside the publication
  or another URL; a malformed entry refuses the whole catalog, so no study goes missing
  unnoticed;
- the snapshot: exactly the size and SHA-256 its entry names (a partial copy or a replaced file
  is never read as a study);
- the study: its JSON form reads back
  ([`StudyResults.from_dict`](optimization.md#export-formats)): the identity digest of its
  `study.json` part, each trial's status, outcome, candidate (in its parameters' domains),
  assessment and record against the problem and each other, every limit judged again on its
  recorded value (margin, normalized margin and verdict, so feasibility too), and the outcome
  counts and the Pareto front, recomputed;
- and it is the study its catalog entry describes (id, name, counts).

| Raised | When | What to do |
| --- | --- | --- |
| `NothingPublishedError` | the location holds no `catalog.json` (or the directory does not exist) | publish a study there, or check the location |
| `PublicationUnavailableError` | a timeout, a refused connection, a server error (5xx, 408, 429), an answer cut short, an unreadable file | read again later |
| `PublicationError` | a malformed catalog or snapshot, a snapshot that is missing or other than its entry names, a redirect, another HTTP error | fix the publication |

All three are `K4OptError`s. Nothing is cached by the reader: every call reads again (the
dashboard caches, and says how).

## Python API

```python
from k4opt.optimization import load_study
from k4opt.publication import local_paths, publish

results = load_study("runs/cld-ecal-study")
local_paths(results)  # [(where, path), ...]: what k4opt publish warns about
entry = publish(results, "published/", warn=print)  # its CatalogEntry
entry.snapshot.path, entry.snapshot.sha256, entry.published
```

## Not done

Publishing files of a trial (events, plots, logs) beside its snapshot, which needs a contract of
its own; uploading to EOS or an object store (publish into a mounted directory, or copy);
access control and authentication; a database-backed catalog for searching many studies; removing
or renaming published studies. None of them is blocked: snapshots are named by content and
studies by their own ids, and a reader only ever follows the paths a catalog gives.
