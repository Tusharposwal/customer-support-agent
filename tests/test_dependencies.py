"""constraints.txt must pin every installed dependency exactly, agree with
pyproject.toml, and match what is actually installed (in CI this proves the
pinned set was used). Reads installed package metadata only; no network."""

from importlib import metadata
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "customer-support-agent"


def read_constraints() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        (spec,) = requirement.specifier  # exactly one specifier...
        assert spec.operator == "==" and "*" not in spec.version, f"not an exact pin: {line}"  # ...an exact one
        assert requirement.marker is None and not requirement.extras, f"keep pins plain: {line}"
        name = canonicalize_name(requirement.name)
        assert name not in pins, f"listed twice: {line}"
        pins[name] = spec.version
    return pins


def project_requirements() -> list[Requirement]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    raw = project["dependencies"] + [r for group in project["optional-dependencies"].values() for r in group]
    return [Requirement(r) for r in raw]


def installed_tree() -> dict[str, str]:
    """Every distribution the project (with its dev extra) needs on this
    platform, by following installed metadata: {name: installed version}."""
    environment = default_environment()
    found = {}
    pending = [(r, set(r.extras)) for r in project_requirements()]
    while pending:
        requirement, extras = pending.pop()
        name = canonicalize_name(requirement.name)
        if name in found:
            continue
        dist = metadata.distribution(name)
        found[name] = dist.version
        for raw in dist.requires or []:
            child = Requirement(raw)
            if child.marker is None or any(
                child.marker.evaluate({**environment, "extra": extra}) for extra in ["", *extras]
            ):
                pending.append((child, set(child.extras)))
    return found


def test_constraints_are_exact_pins():
    assert len(read_constraints()) >= 30


def test_every_direct_dependency_is_pinned_within_its_range():
    pins = read_constraints()
    for requirement in project_requirements():
        name = canonicalize_name(requirement.name)
        assert name in pins, f"{requirement.name} is in pyproject.toml but not constraints.txt"
        assert requirement.specifier.contains(pins[name]), (
            f"{requirement.name}=={pins[name]} is outside pyproject.toml's range {requirement.specifier}"
        )


def test_every_installed_dependency_is_pinned():
    """Includes dependencies of dependencies: a new one must be added."""
    missing = sorted(set(installed_tree()) - set(read_constraints()))
    assert missing == [], f"add these to constraints.txt: {missing}"


def test_installed_versions_match_the_pins():
    pins = read_constraints()
    wrong = {
        name: (version, pins[name]) for name, version in installed_tree().items() if version != pins[name]
    }
    assert wrong == {}, f"installed (version, pinned): {wrong}; reinstall with -c constraints.txt"


@pytest.mark.parametrize("path", ["README.md", ".github/workflows/tests.yml"])
def test_install_commands_use_the_constraints(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    assert "-c constraints.txt" in text
