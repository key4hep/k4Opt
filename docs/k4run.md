# k4run

`k4opt.k4run` runs a Key4hep reconstruction (a Gaudi job, started with `k4run`) once, on the
candidate's geometry and every event of a simulation. A job is one or more Gaudi options files,
such as CLDConfig's `CLDReconstruction.py`, which are Python programs that read arguments of
their own. k4opt passes the stage's options on as the job's arguments, gives it the geometry,
the input and the output, has it reconstruct every event, and checks from the job's resolved
options that it read what it was given. It does not read the output; an evaluator does.
`k4opt evaluate` runs it as the `reconstruction` [stage](evaluation.md#stages) of each sample
that defines one, once per candidate, so evaluators never call it. Checked against the `k4run`
of the nightly 2026-10-20 (k4FWCore's `k4run` and `ApplicationMgr.py`) and the reconstructions
of CLD, ALLEGRO, IDEA and ILD as [k4Bench](https://github.com/key4hep/k4Bench) runs them.

```yaml
samples:
  pi0_50gev:
    simulation: {executor: ddsim, events: 200, seed: 42}
    reconstruction:
      executor: k4run
      directory: $CLDCONFIG/share/CLDConfig   # the job runs from a copy of it
      options: CLDReconstruction.py           # one file or a list, relative to directory
      geometry: compactFile                   # the job's own argument for the compact file
      cms: 240                                # --cms=240
      truthTracking: true                     # --truthTracking
```

## The job

| Key | Meaning |
| --- | --- |
| `options` | Required. The job's options files, one or a list, in the order k4run loads them. With `directory`, each is named relative to it and lies inside it; without, a relative path is read from the directory of the problem file. |
| `directory` | Optional. A directory the job runs from a writable copy of, in the stage's own directory (`job/`). A reconstruction such as CLD's runs only from its own directory: its Pandora settings name files relative to where it runs. `$NAME` and `${NAME}` are environment variables (`$CLDCONFIG`, set by the Key4hep environment), `~` is the home directory. |
| `geometry` | Optional. The job's own argument that takes the compact file (CLD's and ILD's `compactFile`), for a job that derives more than the geometry service from it. Without it, k4opt sets `GeoSvc.detectors` (ALLEGRO, IDEA). |
| `input` | Optional. The job's own argument that takes the input files (CLD's `inputFiles`). Without it, k4opt sets `IOSvc.Input`. |
| any other key | An argument of the job or a Gaudi property, named as `k4run OPTIONS.py --help` names it, without the dashes: `cms: 91` is `--cms=91`, `GeoSvc.OutputLevel: 3` is `--GeoSvc.OutputLevel=3`, a list is `--key v1 v2`, `true` the flag `--key`, `false` `--key=False` (as a Gaudi property reads it). A mapping or `null` is refused, as is a list value starting with `-`, which k4run would read as an argument. |

A reconstruction takes no `events` or `seed`: it reconstructs every event of the simulation it
reads, whose `events` and `seed` made them (an evaluator of the reconstruction gets those as
`request.events` and `request.seed`). A seed the job itself reads is one of its arguments.
`timeout` limits it as any stage ([Time limits](problem.md#time-limits)).

## What k4opt sets, and checks

k4opt runs

    k4run <options files> <stage>/k4opt.opts.py --num-events=-1 --GeoSvc.detectors=<compact>
          --IOSvc.Input <simulation> --IOSvc.Output=<stage>/events.edm4hep.root <the job's arguments>

from the job's copy of its directory, with `--<geometry>=<compact>` or `--<input> <simulation>`
in place of the service where the stage names the job's own argument. k4opt's own options file
comes after the job's, so that it changes nothing the job set; it has Gaudi write the job's
resolved options (`JobOptionsSvc().DUMPFILE`) to `joboptions.txt`. It comes before the job's
arguments: a file after `--inputFiles a.root` would be read as one more input file.

- `--num-events=-1`: every event of the input. A job may set its own number of events
  (`CLDReconstruction.py` sets 3), which would otherwise reconstruct only some.
- The arguments k4opt sets are refused in a problem, in both of k4run's spellings of a property
  (`GeoSvc.detectors` and `detectors.GeoSvc`): `num-events`, `n`, `evt-max`,
  `GeoSvc.detectors`, `IOSvc.Input`, `IOSvc.Output`, `ApplicationMgr.EvtMax`, every property of
  `JobOptionsSvc`, the argument `geometry` or `input` names, and k4run's own `dry-run`, `list`,
  `gdb`, `interactive`, `interactive-root` and `help`.
- After the job ran, its resolved options must show `GeoSvc.detectors` = the candidate's compact
  file, `IOSvc.Input` = the simulation and `IOSvc.Output` = the output, or the reconstruction
  fails with what the job used. CLD's reconstruction, for one, sets its geometry from its own
  `--compactFile`, whose default is CLD_o2_v07: a reconstruction that forgot `geometry:
  compactFile` would otherwise reconstruct the candidate's events on another detector without a
  word.
- The output must hold every event of the simulation (when the simulation was run or supplied
  for it), checked as for a simulation; an empty output fails.

## The dry run

The job's options files define its arguments, so k4opt keeps no table of them: whether the job
takes its arguments is the job's to say. When a session opens, before anything is simulated, it
runs `k4run --dry-run` with the arguments of each reconstruction that runs, once with the
options that parameters set at the first end of each domain and once for each other end, from a
copy of the job's directory, on the problem's own geometry and an input of one empty event (which
k4FWCore requires to exist; a multithreaded job reads its first event). The job's options files
run and k4run parses every argument, strictly; nothing is reconstructed. The first dry run writes
the input, in about five seconds; each other takes about a second. A job that refuses an
argument refuses the problem:

    samples.pi0_50gev.reconstruction: the job does not take its arguments: k4run: error:
    argument --cms: invalid choice: '250' (choose from 91, 160, 240, 365)

## Files and records

The stage's directory, `samples/<sample>/reconstruction/`, holds the output
`events.edm4hep.root`, `k4run.stdout.log`, `k4run.stderr.log`, `k4run.json` (argument list,
directory, exit code, time limit, peak memory), `joboptions.txt` and `k4opt.opts.py`, and the
job's working directory `job/`, where whatever else the job writes stays (histograms,
`cdb.log`). The record of the stage keeps the argument list and, under `files`, the path, size
and SHA-256 of every file of `directory` (or of each options file), by where the problem names
it (`directory/PandoraSettingsCLD/PandoraSettingsDefault.xml`). The session identifies the same
files when it opens, so `verify_inputs` and a study's resume refuse another job; a file that
changes while k4run runs fails the reconstruction. The record's `environment.observed.k4run` is
the `k4run` that was started.

## From Python

```python
from pathlib import Path
from k4opt.k4run import dry_run, reconstruct

cld = Path("/cvmfs/.../share/CLDConfig")  # $CLDCONFIG/share/CLDConfig
options = {
    "options": "CLDReconstruction.py",
    "directory": str(cld),
    "geometry": "compactFile",
    "cms": 91,
    "truthTracking": True,
}
job = {"files": [cld / "CLDReconstruction.py"], "directory": cld, "options": options}
print(dry_run(compact=compact, workdir=Path("dry"), **job) or "takes its arguments")
run = reconstruct(compact=compact, input_file=simulated, workdir=Path("reco"), **job)
print(run.error or run.output)
```

`reconstruct` raises `K4runError` (a `k4opt.program.ProgramError`, as `DdsimError` is) for
options it cannot pass on, a `timeout` that is not a positive number of seconds or an existing
`workdir`, before anything is run. A `k4run` that cannot run or fails is reported in
`ReconstructionRun.error`: not found, not started, nonzero exit with the tail of its standard
error, killed when it ran longer than `timeout` seconds, no resolved options, a geometry, input
or output other than k4opt gave it, or no output file. It runs as ddsim does
([ddsim](ddsim.md)), in a process group of its own, with `started` and `die_with_parent`.
`properties` reads a `joboptions.txt`, `verify` checks one, `job_files` lists the files of a
directory as the record names them.

## Not yet

A reconstruction of part of the events; a multithreaded job's threads counted against the
`--workers` (a Gaudi Hive job counts as one processor); a parameter with a unit that sets a
reconstruction option (the job decides the unit of what it reads); fetching a job's data files
(ALLEGRO's `--dataFolder`: give a local directory); LCIO outputs; branches, such as truth and
pattern-recognition tracking as two reconstructions of one simulation.
