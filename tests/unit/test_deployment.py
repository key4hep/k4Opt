"""The dashboard's container, its OpenShift manifests and its workflows agree with each other: one
name, one port, one health path, the exact image built, nothing run as root, no TLS check skipped,
and no deployment without its credentials."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
NAME, NAMESPACE, PORT, HEALTH = "k4opt-dashboard", "k4opt", 8080, "/_stcore/health"
IMAGE = "ghcr.io/key4hep/k4opt-dashboard"


def manifest(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / "openshift" / name).read_text())
    return data


def workflow(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())
    return data


def test_the_manifests_name_one_service_on_one_port_with_its_health_probes() -> None:
    deployment, service, route = (manifest(f"{n}.yaml") for n in ("deployment", "service", "route"))
    for document in (deployment, service, route):
        assert (document["metadata"]["name"], document["metadata"]["namespace"]) == (
            NAME,
            NAMESPACE,
        )
    pod = deployment["spec"]["template"]["spec"]
    (container,) = pod["containers"]
    assert container["name"] == "dashboard" and container["image"] == f"{IMAGE}:latest"
    assert container["ports"] == [{"containerPort": PORT, "protocol": "TCP"}]
    for probe in ("readinessProbe", "livenessProbe"):
        assert container[probe]["httpGet"] == {"path": HEALTH, "port": PORT}
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    }
    assert set(container["resources"]) == {"requests", "limits"}
    assert {e["name"]: e["value"] for e in container["env"]} == {"K4OPT_DASHBOARD_DATA": ""}
    assert deployment["spec"]["selector"]["matchLabels"] == service["spec"]["selector"]
    assert service["spec"]["ports"][0]["targetPort"] == PORT == route["spec"]["port"]["targetPort"]
    assert route["spec"]["to"] == {"kind": "Service", "name": NAME}
    assert route["spec"]["tls"] == {
        "termination": "edge",
        "insecureEdgeTerminationPolicy": "Redirect",
    }
    assert not route["metadata"].get("annotations")  # no claim about who can reach it


def test_the_image_serves_the_port_without_optuna_and_as_any_user() -> None:
    dockerfile = (ROOT / "dashboard" / "Dockerfile").read_text()
    assert f"EXPOSE {PORT}" in dockerfile and f'"--server.port={PORT}"' in dockerfile
    assert re.search(r"^FROM \S+:3\.13-slim@sha256:[0-9a-f]{64}$", dockerfile, re.MULTILINE)
    assert re.search(r"^USER 1001$", dockerfile, re.MULTILINE)
    assert "chmod -R g=u /app" in dockerfile and "HOME=/tmp" in dockerfile
    requirements = (ROOT / "dashboard" / "requirements.txt").read_text().splitlines()
    pinned = [line for line in requirements if line and not line.startswith("#")]
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==\S+", line) for line in pinned)
    names = {line.split("==")[0].lower() for line in pinned}
    assert {"streamlit", "plotly", "lxml", "pyyaml"} <= names
    assert not names & {"optuna", "optuna-dashboard", "podio", "sqlalchemy"}


def test_the_deployment_runs_the_exact_image_built_and_only_once_provisioned() -> None:
    deploy = workflow("deploy-dashboard.yml")
    assert deploy["env"] == {"IMAGE": IMAGE, "NAMESPACE": NAMESPACE}
    on = deploy[True]  # YAML reads the key "on" as true
    assert set(on) == {"push", "workflow_dispatch"} and on["push"]["branches"] == ["main"]
    image, job = deploy["jobs"]["image"], deploy["jobs"]["deploy"]
    assert image["permissions"] == {"contents": "read", "packages": "write"}
    assert "permissions" not in job and deploy["permissions"] == {"contents": "read"}
    assert job["needs"] == "image"
    assert job["if"] == "vars.DASHBOARD_DEPLOY == 'true' && github.ref == 'refs/heads/main'"
    assert job["environment"]["name"] == "production"
    steps = "\n".join(step.get("run", "") for step in job["steps"])
    assert "RepoDigests" in "\n".join(step.get("run", "") for step in image["steps"])
    assert 'oc set image --local -f openshift/deployment.yaml "dashboard=${NEW_IMAGE}"' in steps
    assert f"deployment/{NAME}" in steps and "--timeout=" in steps


def test_no_workflow_skips_a_tls_check_or_puts_a_secret_in_a_command() -> None:
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        text = path.read_text()
        assert "insecure-skip-tls-verify" not in text, path.name
        assert not re.search(r"curl[^\n]*\s-k\b|--insecure", text), path.name
        for line in text.splitlines():
            if "secrets." in line and "run:" in line:
                raise AssertionError(f"{path.name}: a secret inline in a command: {line}")
