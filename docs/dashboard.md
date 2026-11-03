# Dashboard

The dashboard shows published k4opt studies in a browser: every study of a
[publication](publication.md), and for each its problem, how its trials ended, its trials in the
plane of two objectives or parameters with the Pareto front, and what it was made with. It is a
[Streamlit](https://streamlit.io) app in `dashboard/`, built into a container and meant to run on
CERN OpenShift at `k4opt-dashboard.app.cern.ch`, after the pattern of
[k4Bench's dashboard](https://github.com/key4hep/k4Bench/tree/main/dashboard).

## How it is put together

```text
 study directory          publication (a directory, served or copied)         dashboard
 ───────────────          ───────────────────────────────────────────         ─────────
 study.json, journal ──►  k4opt publish ──► catalog.json                 ──►  k4opt.publication
 trial-*/summary.json     (load_study:      studies/<id>/snapshots/*.json      (directory or URL)
                           Optuna, checks)                                         │
                                                                           StudyResults.from_dict
                                                                                   │
                                                                     k4opt.visualization (Plotly)
                                                                                   │
                                                                         dashboard/*.py (Streamlit)
```

| Part | Where | Depends on |
| --- | --- | --- |
| Reading and checking a study | `k4opt.optimization` (`load_study`, `StudyResults`) | Optuna, only to read a study directory |
| Publishing and reading publications | `k4opt.publication` | `k4opt.optimization`'s `StudyResults` and its JSON form; the standard library |
| Figures | `k4opt.visualization` | `StudyResults`, Plotly |
| Pages | `dashboard/` | the two above, Streamlit |
| Container, deployment | `dashboard/Dockerfile`, `openshift/`, `.github/workflows/` | |

The scientific results stay those of k4opt: a published study reads back as the same
`StudyResults` that `load_study` gives, and the dashboard shows what it holds (outcomes,
feasibility, the Pareto front, values with units and uncertainties) without judging anything
again. Nothing in the dashboard depends on Optuna or on where a publication is kept:

- **No optimizer in it.** Optuna is k4opt's first optimizer, not its data model. The figures are
  built from `StudyResults` with Plotly, never with `optuna.visualization` or Optuna Dashboard,
  so a study made by another optimizer draws alike, and the container has no Optuna. A unit test
  (`tests/unit/test_components.py`) refuses an import of Optuna in `dashboard/`, of Streamlit
  anywhere in k4opt and of Plotly outside `k4opt.visualization`.
- **No storage in it.** The dashboard reads a publication through `open_publication`: a
  directory or an `https://` URL, the same files and the same checks. EOS behind WebEOS is one
  way to serve it, as any web server or a copy on disk is.
- **Streamlit stays outside k4opt.** `k4opt optimize`, `k4opt evaluate` and the rest need neither
  Streamlit nor Plotly; `pip install 'k4opt[plot]'` adds Plotly for figures in a notebook.

## Running it locally

```bash
pip install -r dashboard/requirements.txt && pip install --no-deps -e .
# and Graphviz, whose dot lays out how a candidate is evaluated: apt/dnf/brew install graphviz
k4opt publish runs/cld-ecal-study --output published/      # needs Optuna, which reads the study
K4OPT_DASHBOARD_DATA=published/ streamlit run dashboard/app.py
```

`K4OPT_DASHBOARD_DATA` names the publication: a directory, or the URL it is served at
(`K4OPT_DASHBOARD_DATA=https://k4opt-data.web.cern.ch`). Unset, the dashboard starts and says how
to set it; a publication with nothing in it, one that does not answer and one that does not hold
up each have their own message, and *Refresh* in the sidebar reads the catalog again. The pages:

- **Studies** (`/`): every published study, the latest first, with its problem, when it was
  created and published, how many trials ended how (and the feasible share), and how many were
  running when it was published (a study in progress is published as far as it got). Selecting
  one opens it.
- **Study** (`/study?study=<id>`): one study, by its id, so that the link can be shared, with its
  outcome counts and four tabs; the tab open, the figure's axes and colors and the trial shown
  are in the URL too (`&view=Trials&trial=12`, `&x=objective:...&y=parameter:...&color=trial`).
  Only the open tab runs.
  - **Overview**: [`trials_figure`](visualization.md) with its axes and what colors it, the
    trials on the Pareto front, and [`history_figure`](visualization.md#the-history-of-an-objective-history_figure)
    for each objective. Clicking a trial in a figure opens it in a dialog.
  - **Trials**: every trial (values, uncertainties, limit margins, times, record ids, messages),
    filtered by outcome; selecting one shows all it recorded: its objectives, candidate, limits
    with their margins, stage times, diagnostics that did not succeed, and its record.
  - **Pipeline**: how a candidate is evaluated
    ([`pipeline_figure`](visualization.md#how-a-candidate-is-evaluated-pipeline_figure): drag to
    pan, scroll to zoom; click a stack of samples alike to draw each, *Draw each sample* and
    *Stack alike samples* for all of them, and click a task for what the problem says of it and
    its time or failures in each trial).
  - **Problem**: its parameters, objectives and constraints, and what it was made with (k4opt,
    the Key4hep stack, the geometry, each evaluator's version, source hash and commit, the
    optimizer and its seed).

**Caching.** The catalog changes when a study is published: it is kept 60 s for all viewers, and
*Refresh* drops it. A snapshot never changes, since its name is its content: a study is kept by
its snapshot's path and SHA-256, so it is never stale, and the latest read are kept while their
snapshots add up to 96 MiB (`STUDY_BYTES` in `dashboard/data.py`), some 150 MB in memory, so
that the cache stays bounded by size, not by a number of studies, within the deployment's 1 GiB.
Each is still checked against its catalog entry on every page. What is kept is immutable, so viewers share it safely. A read that fails or arrives short
raises, and Streamlit keeps no exception: nothing of it is kept, and the next interaction reads
again.

## The container

```bash
docker build -f dashboard/Dockerfile -t k4opt-dashboard .     # from the repository's root
docker run -p 8080:8080 -e K4OPT_DASHBOARD_DATA=https://... k4opt-dashboard
```

`python:3.13-slim` (pinned by digest), Graphviz from Debian with Liberation Sans (its `dot` lays
out and draws the graph of a study's tasks, measuring text as wide as a browser's Helvetica or
Arial), the packages of `dashboard/requirements.txt` (every one pinned; Streamlit, Plotly, lxml,
PyYAML and what they need, no Optuna), k4opt without its optional dependencies, and
`dashboard/`. It holds no study: the data is read where it is published,
so publishing a study needs no new image. Streamlit listens on 8080 and answers
`/_stcore/health`. It runs as any user of group 0, as OpenShift assigns one (and as 1001
elsewhere); `HOME` is `/tmp`, where Streamlit keeps its settings.

## On CERN OpenShift

`openshift/` holds the deployment, the service and the route, all named `k4opt-dashboard` in the
project `k4opt`: one replica (each viewer's session lives in its process), 100m/256Mi requested
and 500m/1Gi at most, readiness and liveness probes on `/_stcore/health` with k4Bench's timings,
no root, no privilege escalation, no capabilities, and `K4OPT_DASHBOARD_DATA` naming the
publication at `https://k4opt-data.web.cern.ch`. The route serves
`https://k4opt-dashboard.app.cern.ch` with TLS ended at the router, to the whole Internet: the
CERN routers admit only CERN's networks unless a route empties its IP whitelist
(`haproxy.router.openshift.io/ip_whitelist: ""`), as this one does.

**What is provisioned once, by hand.** All of it is in place; it is what a new deployment needs.

1. The OpenShift project `k4opt` at [paas.cern.ch](https://paas.cern.ch), owned by the
   maintainers' e-group.
2. A service account that deploys, with the `edit` role in the project, and a long-lived token
   for it:

   ```bash
   oc create serviceaccount k4opt-deployer -n k4opt
   oc policy add-role-to-user edit -z k4opt-deployer -n k4opt
   oc apply -n k4opt -f - <<'EOF'
   apiVersion: v1
   kind: Secret
   metadata:
     name: k4opt-deployer-token
     annotations:
       kubernetes.io/service-account.name: k4opt-deployer
   type: kubernetes.io/service-account-token
   EOF
   oc get secret k4opt-deployer-token -n k4opt -o jsonpath='{.data.token}' | base64 -d
   ```

3. In the GitHub repository: an environment `production` (with required reviewers, if every
   deployment should be approved) holding the secrets `OC_SERVER`
   (`https://api.paas.okd.cern.ch`) and `OC_TOKEN` (the token above), and the repository variable
   `DASHBOARD_DEPLOY` set to `true` once all of this is in place.
4. Runners that reach CERN's OpenShift API: the deploy job runs on `[self-hosted, fcc-ironic]`,
   as k4Bench's does. k4Bench's runners are registered to key4hep/k4Bench, so one more is
   registered for key4hep/k4Opt on the same machines, supervised as theirs are. The repository
   is public: fork pull requests of all outside contributors wait for approval before any
   workflow runs, since one could otherwise run on these runners.
5. After the first image is pushed, the package `ghcr.io/key4hep/k4opt-dashboard` made public in
   the organization's package settings, so OpenShift pulls it without credentials (or an image
   pull secret in the project).
6. The data site: an EOS directory served by the WebEOS site `https://k4opt-data.web.cern.ch`
   (a `UserProvidedDirectory`, as k4Bench's data is served), named in `K4OPT_DASHBOARD_DATA` in
   `openshift/deployment.yaml`. It needs anonymous access: a site behind CERN SSO answers with a
   login page, which the dashboard refuses. Studies are published straight into the directory
   (`k4opt publish DIR --output <the EOS directory>`); that needs no deployment.

**Deploying.** `.github/workflows/deploy-dashboard.yml` runs on every push to `main` that changes
`dashboard/`, `k4opt/`, `pyproject.toml`, `openshift/` or itself, and on demand
(`workflow_dispatch`). Its first job builds the image on a GitHub runner and pushes it to GHCR,
tagged with the full commit SHA (and `latest` on main). The second runs only with
`DASHBOARD_DEPLOY` set, from `main`, in the `production` environment:

1. it logs in to the API with the token, the certificate verified (it is issued by a public
   authority; nothing skips a TLS check);
2. it applies the service and the route;
3. it applies the deployment with its image replaced by the digest just pushed, so exactly that
   image rolls out, once;
4. it waits for the rollout (5 minutes at most) and reports the route.

Pull requests and the CI never need CERN credentials: CI's `dashboard` job builds the image,
starts it as an arbitrary user of group 0, probes `/_stcore/health`, checks Optuna is not in it,
and validates the manifests (kubeconform; the Route has no public schema, and
`tests/unit/test_deployment.py` checks it with the rest).

**Access.** Treat everything the deployed dashboard reads as public. A data site with anonymous
access serves its files to anyone, whatever the dashboard's route does, so restricting the route
would not restrict the data, and the route is open to the Internet. Only studies meant to be
public belong in its publication (see what a snapshot holds in
[publication](publication.md#what-a-publication-holds)).
Non-public studies need access control on both the data and the app, such as CERN SSO in front of
the route and an authenticated data source; that is not built.

## Developing it

```bash
python -m pytest -p xdist -n auto tests/unit/publication tests/unit/visualization tests/unit/dashboard
(cd dashboard && MYPYPATH=.. mypy --strict --follow-imports=silent --config-file ../pyproject.toml \
   app.py data.py overview.py study.py tables.py)
```

The tests make real studies of the mini geometry with the test evaluator `test.trade_off`
(problem `trade`, clearly not a scientific study), publish them, and run the app with Streamlit's
`AppTest`, from a directory and from a local HTTP server that also answers with the faults a real
one shows (a missing file, a server error, a redirect, an answer cut short or late). They need
Streamlit, Plotly and Optuna, which the Key4hep stack lacks: CI runs them in its `dashboard` job.
`dashboard/` holds plain modules Streamlit imports from its directory: `app.py` (the page
configuration, the sidebar, the navigation), `data.py` (where the studies are read, and the
caches), `overview.py` and `study.py` (the pages) and `tables.py` (the tables' rows, without
Streamlit).

**A new figure** is a function of `StudyResults` in `k4opt.visualization`
([how](visualization.md#adding-a-figure)), shown on the Study page with `st.plotly_chart`.

**A database-backed catalog**, for searching many studies, can replace `catalog.json` without
changing the snapshots: snapshots stay files (named by content, so any store serves them), and a
catalog of rows (id, name, problem, created, published, counts, snapshot path and SHA-256, and
whatever is to be searched) is what `Publication.catalog()` would read instead. The dashboard
reads only `Catalog`, `CatalogEntry` and `StudyResults`.

## Not done

Access control and CERN SSO, a database of studies, comparisons between studies, files of a
trial (events, plots, logs), live progress of a running study, and controlling studies from the
browser.
