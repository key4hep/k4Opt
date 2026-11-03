# The k4opt dashboard: Streamlit on port 8080, reading the publication K4OPT_DASHBOARD_DATA names
# (a directory or an https:// URL). No Key4hep stack, DD4hep, Geant4, podio or Optuna, but Graphviz
# to lay graphs out; no data in the image. Built from the repository's root:
#
#   docker build -f dashboard/Dockerfile -t k4opt-dashboard .
#   docker run -p 8080:8080 -e K4OPT_DASHBOARD_DATA=https://... k4opt-dashboard
FROM docker.io/library/python:3.13-slim@sha256:aaf08c5490d52cd92bc27142a5215b1e47355a5eaf9adfad2a0fbc022d0b5490

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HOME=/tmp \
    K4OPT_DASHBOARD_DATA=

WORKDIR /app

# Graphviz's dot lays out the graph of a study's tasks (k4opt.visualization.pipeline_figure)
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

# the packages first, the code last: what changes most often is rebuilt alone
COPY dashboard/requirements.txt dashboard/requirements.txt
RUN pip install -r dashboard/requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY k4opt/ k4opt/
RUN pip install --no-deps . && rm -rf build k4opt.egg-info

COPY dashboard/ dashboard/

# OpenShift runs the container as an arbitrary user of group 0: everything stays readable, HOME
# (/tmp, where Streamlit keeps its settings) writable; another runtime runs it as user 1001
RUN chgrp -R 0 /app && chmod -R g=u /app
USER 1001

ARG REVISION=unknown
LABEL org.opencontainers.image.title="k4opt-dashboard" \
      org.opencontainers.image.description="Dashboard of published k4opt detector optimization studies" \
      org.opencontainers.image.source="https://github.com/key4hep/k4Opt" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${REVISION}"

EXPOSE 8080
# the health endpoint for probes: GET /_stcore/health
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false", \
     "--client.toolbarMode=viewer"]
