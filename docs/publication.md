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
code        nothing local to publish
scan        nothing found
to          published/

published   cld-ecal-granularity-3f9c2a1b (snapshot 9b1c0e7d): code of 0, 0 inputs, 0 large
            30 trials (28 feasible, 1 invalid, 1 failed), published 2026-11-03T10:00:00+00:00 as published/studies/3f9c2a1b-.../snapshots/9b1c0e7d2a4f5e68.json
```

A study that used local code or files lists them first and asks: see
[What else is published](#what-else-is-published).

## What a publication holds

```text
published/
├── catalog.json                                  k4opt.catalog/1: the one file rewritten
└── studies/
    └── 3f9c2a1b-5d0e-4f43-9a52-77c1d2e0b6a4/     a study, by its id
        ├── snapshots/
        │   └── 9b1c0e7d2a4f5e68.json             the study as published, never changed
        ├── software/
        │   └── 4e1a7730c2d95b16.json             k4opt.software-sheet/1: what a rerun needs
        └── artifacts/
            └── a31f06c88e7d4b02                  a file its author approved, named by its content
```

**A snapshot** is the study's [JSON form](optimization.md#export-formats), byte for byte what
`k4opt study export --format json` writes, but for a message [withheld](#what-else-is-published)
because it looks like it holds a secret: the problem definition, every trial with its
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
| size and SHA-256 of every file the problem names, the evaluators' versions, source hashes and git commits, the geometry's compact file, fingerprint, targets and the hashes of what its external references read | the paths of those files, the geometry's source root, `K4GEO`, `KEY4HEP_STACK` (the build identifies the stack) |
| the [software](software.md)'s `id`, build and system, the programs the stages ran (size and SHA-256 of `ddsim`, `k4run`), k4opt's version, hash and commit, and the [software sheet](#what-else-is-published) | the optimizer's storage (a journal name, or a database URL with its host and user) |
| the optimizer, its sampler and seed, its directions and distributions | the names and digests of the software's variables, which can be private, and anything of `study.json` not on the list (`PORTABLE` in `k4opt/optimization/_study.py`) |

**Publishing makes all of it readable to whoever can read the publication.** The problem
definition and the messages are published as they are: a stage `file:` given as an absolute
path, an evaluator setting naming a file, or an evaluator that writes a path into its message
puts that path into the snapshot. `k4opt publish` therefore lists, before it asks or writes
anything, every absolute path outside `/cvmfs` that the problem's definition and the trials'
messages hold, at every level of a nested study:

```text
warning:    trial 7's message holds the local path /home/me/logs/ddsim.log; it is published
```

Publish into a staging directory first if a study may hold what should not be public, look at
what it says, and serve or copy the directory only then. Nothing is published automatically: a
publication holds only what someone ran `k4opt publish` for.

**The catalog** names each study once, by its id, with what a list of studies shows and sorts
by (its problem, the compact file of the detector it starts from, the Key4hep build it ran on) and the
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
      "key4hep": "LCG devkey-head, built Sat Nov 7 04:17:32 2026",
      "created": "2026-10-30T12:00:00+00:00",
      "published": "2026-11-03T10:00:00+00:00",
      "trials": 30,
      "outcomes": {"feasible": 28, "infeasible": 0, "invalid": 1, "failed": 1, "rejected": 0, "interrupted": 0, "error": 0, "running": 0},
      "parameters": 1,
      "objectives": 2,
      "constraints": 0,
      "snapshot": {"path": "studies/3f9c2a1b-.../snapshots/9b1c0e7d2a4f5e68.json", "sha256": "9b1c0e7d...", "bytes": 48213},
      "software": {"path": "studies/3f9c2a1b-.../software/4e1a7730c2d95b16.json", "sha256": "4e1a7730...", "bytes": 9120}
    }
  ]
}
```

`running` counts the trials that were running when the snapshot was published: a study in
progress can be published, and published again as it goes on. Nothing says whether a study is
"finished": a study can always be resumed.

## Publishing

`k4opt publish DIR --output PUBLICATION [--storage URL] [--yes] [--include-inputs NAME]
[--include NAME] [--exclude NAME] [--include-large]` reads the study as
[`k4opt study`](optimization.md#inspecting-and-exporting-a-study) does, checked in full and
unchanged (every record against the study and the optimizer, every assessment recomputed from
its measurements), then:

1. refuses a publication directory inside the study (also through a symlink); one that does not
   exist is made;
2. lists what else would be published, the local paths the snapshot carries (above) and what
   the secret scan found, and asks ([below](#what-else-is-published));
3. writes the snapshot and the files approved, whole and on disk before each gets its name, as
   every k4opt record is written; a file already there with the same bytes is kept, one with
   other bytes is refused;
4. under `.catalog.lock`, reads `catalog.json` (none yet: an empty catalog; a malformed one is
   refused and left as it is), writes the study's software sheet and writes the catalog again
   whole with the study's entry, replacing the old one by a rename, so a reader sees the old
   catalog or the new one.

Publishing the same state of a study again adds no snapshot and does not change when it was
published: what a rerun needs is **checked again**, and the study's software sheet says when
(a nightly may have been overwritten since). A sheet lists what this publication approved, and what
an earlier one published that is still of the study ([below](#what-else-is-published)). A
publication that fails removes every file it wrote, and writes nothing until its software sheet is
made and checked. Publishing a study's JSON form again keeps its sheet. A
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

## What else is published

The study keeps privately what a rerun needs ([software](software.md#what-a-rerun-needs)).
What of it leaves the machine is decided here. Recording is automatic; **publishing is not**.

Local code is approved in **units**, each published whole or not at all: which files of it a rerun
needs is never guessed, since what is compiled in is never read at run time.

| Unit | What it holds |
| --- | --- |
| a local build | its checkout's whole state against the revision the nightly built (or every file, if no remote has its history): edits, untracked files and binaries a checkout holds alike; its `cmake-options.txt`; what it installed that a stage read, as `installed/...`, since that can differ from what its source installs |
| a directory of local code | its every kept file, or its checkout's whole state; offered if a stage read from it, or if that is not known; held back (`--include` offers it) if every stage was observed and none read from it |
| k4opt's own checkout, an evaluator, the geometry | their kept files (the geometry with an `aliases.json` of its files' other names) |
| a Python environment a stage used | its distributions, `name==version`, as a `requirements.txt` to `pip install -r`; whether an index still has them is not checked, so the status keeps saying *availability not established* |
| a changed job directory | its changed and added files and a `job-files.json`: the stack's files it removed, whether each file is executable (`false` too), its links within it |
| an input | one file; a job file that leads out of its directory, through any chain of links, is an input of its own (`link:<stage>/<file>`), not code |

| What | Published |
| --- | --- |
| the study's **software sheet**: the status, each requirement with its state and who can get it | always, made field by field: no path, size, variable name, host, or digest of an unpublished file |
| the **code** units | after the first question |
| the **input** units | after a second, separate question that names where they go; up to 10 MB in all |
| a unit past the 10 MB of its group | a warning, not published; `--include-large` publishes it |
| a unit you leave out | `--exclude NAME` of it or of any of its files: the dashboard shows its name, *on request* |
| a file under `~/.` a stage read, `.rootrc` among them | an input like any other, asked for with the inputs |
| what holds the account's credentials (`~/.ssh/`, `~/.gnupg/`, `~/.aws/`, `~/.kube/`, `~/.docker/`, `~/.netrc`, `~/.pgpass`, `~/.git-credentials`) | never offered, and on request: `--include-inputs NAME` names one |
| a venv's `pyvenv.cfg` | listed only: it says where an interpreter is, and the environment is a requirement of its own |
| a build no stage read | not offered |

A file counts as read where it was read (equal bytes elsewhere say nothing), a Python module also
when its `__pycache__` was. Of a build's CMake cache, `cmake-options.txt` holds the options that name
no local path and whose name says no credential (an empty one too; a path on the stack or the
system as it is); the others, `CMAKE_TOOLCHAIN_FILE` and `CMAKE_PREFIX_PATH` among them, are counted
in its first line, and the build's options then read *partially published*: the build is on request.
Where it installed is no setting. Never published: a library or program (only its SHA-256 and
build-id), the CMake cache itself, or an environment's list of distributions (only how many).

A file that is an input is published only as that input, under the input question, never inside a
unit of code that holds it too (a data file in a checkout, a job file's link target): it is left out
of that unit's patch or files, and the input answers for that unit as well. `--exclude` leaves out a
file wherever it is: every unit that holds it, and every one that answers for a unit left out.

Every file is known by its unit, its kind and its name: a file of the unit as it is, a patch, one
k4opt generates (`cmake-options.txt`, `job-files.json`, `aliases.json`, `requirements.txt`), or a file
the geometry reached out of its tree (`externals/<sha256[:16]>/<name>`, as the snapshot's geometry
names each reference's file by its SHA-256, so two references to equal bytes are one file). So a
real file called like a generated one is another file, and two pieces of one identity and other
bytes are refused. A Python environment's `requirements.txt` lists the distributions installed from
an index, read from their metadata; one installed from a local path, an archive or a repository, or
whose metadata cannot be read, is left to the author, and the environment is then on request.

`--include` names code and `--include-inputs` inputs, each only of its own kind, so that naming an
input never opts in code. A name is a file's path or its own name, a unit's name, or its id
(`package:ddmarlinpandora:3f2a9c1e`, `input:7c1e09a2b4f3`, shown beside inputs that share a name). One that
names nothing is refused, an empty one always (a script's unset variable selects nothing), and so is
one that fits several units when approving, with how to name each: approving one must never approve
two. A release is bound to the study and the state it was reviewed in.

**Publishing again.** Every unit keeps its id between publications: an input's comes from its
commitment, a local build's or directory's from the study's private salt with where it is (so it
holds whatever else is installed, and says nothing of where), a job link's from its place in the
job. A later sheet knows what an earlier one published by that identity, never by bytes alone. What
was published stays listed and *public*, since a publication is never taken back (`k4opt publish`
says which), if its unit still holds it with the same bytes (a patch against the same commit), or
if the study still needs it and its kept copy is gone, which the checked published copy then stands
in for. A file its unit no longer holds belongs to another state of the study, and is not listed.

```text
code        ddmarlinpandora   1.2 kB
              patch against 821384f   3 files  +2 −1
                src/DDCaloHitCreator.cc  +1 −1
                notes.txt  +1 −0
                New.h  +1 −0
              cmake-options.txt  187 B
            helpers   4.3 kB
              ecal_tweak.py  4.2 kB  read by pi0_50gev/reconstruction
              __init__.py  0 B
inputs      ecal_scale.json   1.1 kB   read by pi0_50gev/reconstruction   1.1 kB of 10 MB
warning:    pi0_overlay.root, 48 MB, over 10 MB: not published (--include-large to publish it)
scan        nothing found
to          /eos/project/k4opt/pub

Publish the code above? [y/N]
Also publish the contents of these input files? Anyone who can read
/eos/project/k4opt/pub can read them. [y/N]
published   pi0-cells (snapshot 7c1e09a2): code of 2, 1 input, 0 large
```

Both questions default to no, and the questions are separate since a file published by mistake
cannot be taken back while a file forgotten is one more publish. **In a script:** `--yes` answers
for the code only; inputs are published only if each is named with `--include-inputs NAME`, so a
script never publishes data it does not name. Without a terminal and without `--yes`, if there is
code to approve, `k4opt publish` refuses. A study with a local geometry always has code to approve.

**The secret scan** looks through everything that would be published for private keys, access
tokens, passwords in URLs and in assignments, and files named like credentials: the files listed,
the snapshot and the software sheet, each text as a reader reads it. Where it finds one decides
what happens:

- **in a file listed**, it **blocks** with the file and line, nothing is written, and only removing
  or `--exclude`-ing the file lifts it;
- **in a trial's message** or a diagnostic's, at any level of a nested study, the message is
  **withheld**: the snapshot holds `[withheld by k4opt publish: it looks like it holds a GitHub
  token]` in its place, and the list says so. A message is what a program printed (the last lines
  of its error output) or an evaluator wrote, which no one can change once the study ran;
- **anywhere else in the snapshot** (the problem's definition, `study.json`), it **blocks**,
  naming the field (`the study's problem.definition.evaluators[0].settings.calibration`): that is
  the study itself, whose identity digest a reader checks, so it is published as it is or not at
  all. Such a study is not published;
- **in the software sheet**, which says only what k4opt found of the study's software, it blocks
  as well.

It is best effort: it supports your review of the list and does not replace it.

**The status** of a published study says what a reader can do, not what the author can: what is
only on request makes it *needs access*. It is never *reproduced*, which takes a rerun.

## Reading a publication

```python
from k4opt.publication import open_publication

publication = open_publication("https://k4opt-data.web.cern.ch")  # or a directory
catalog = publication.catalog()
for entry in catalog.studies:
    print(entry.name, entry.trials, entry.outcomes["feasible"])
study = publication.study(catalog.studies[0])  # a StudyResults
study.problem, study.trials, study.pareto
sheet = publication.software(catalog.studies[0])  # a SoftwareSheet, None for one without
sheet.status, sheet.requirements, sheet.artifacts
publication.artifact(sheet.artifacts[0])  # its bytes, checked against its SHA-256
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
- and it is the study its catalog entry describes (id, name, counts);
- a software sheet: the size and SHA-256 its entry names, of that study and that snapshot, well
  formed in every part, each artifact's path the one its study and SHA-256 give; an artifact:
  exactly the size and SHA-256 its sheet names.

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
from k4opt.publication import local_paths, offer, publish

results = load_study("runs/cld-ecal-study")
local_paths(results)  # [(where, path), ...]: what k4opt publish lists
entry = publish(results, "published/", warn=print)  # its CatalogEntry; nothing but the public form
entry.snapshot.path, entry.snapshot.sha256, entry.published

found = offer(results, exclude=["private_notes.py"])  # what may be published, for you to review
found.code, found.inputs, found.large, found.held, found.findings, found.withheld, found.paths
release = found.release(code=True, inputs=["ecal_scale.json"])  # refused if a secret was found
publish(results, "published/", release=release)  # exactly that list
```

## Not done

Publishing a trial's own files (events, plots, logs) beside its snapshot, which needs a contract of
its own; recording that someone reran a study (nothing says *reproduced* yet); uploading to EOS or an object store (publish into a mounted directory, or copy);
access control and authentication; a database-backed catalog for searching many studies; removing
or renaming published studies. None of them is blocked: snapshots are named by content and
studies by their own ids, and a reader only ever follows the paths a catalog gives.
