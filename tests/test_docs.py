"""
Documentation checks for the public docs.

Two cheap gates: the API reference names every public symbol, and the
tracked public docs never point readers at private, untracked files.
"""

import re
from pathlib import Path

import pytest

import qaoa_portfolio

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
API_REFERENCE = REPO_ROOT / "docs" / "api_reference.md"

PUBLIC_DOCS = (
    "README.md",
    "docs/algorithm.md",
    "docs/usage_guide.md",
    "docs/api_reference.md",
    "docs/benchmarks.md",
    "docs/rust_core.md",
    "docs/quantum_backend.md",
    "docs/dataloader.md",
    "docs/visualization.md",
    "docs/testing_manual.md",
    "docs/running-on-dgx-spark.md",
)

# Private files are gitignored; a public doc that links them is a dead link.
# The patterns avoid bare "PROJECT_", which matches UV_PROJECT_ENVIRONMENT.
PRIVATE_REFERENCES = re.compile(
    r"PROJECT_PHASE|PROJECT_ROADMAP|docs/PROJECT_|EXPERIMENT_LOG|CODE_REVIEW_"
    r"|AGENTS\.md|CLAUDE\.md"
)


def _api_reference_text() -> str:
    return API_REFERENCE.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", qaoa_portfolio.__all__)
def test_api_reference_covers_package_exports(name):
    assert name in _api_reference_text(), f"{name} missing from api_reference.md"


def test_api_reference_covers_rust_bridge():
    core = pytest.importorskip("qaoa_portfolio_core")
    text = _api_reference_text()
    missing = [name for name in core.__all__ if name not in text]
    assert not missing, f"bridge names missing from api_reference.md: {missing}"


@pytest.mark.parametrize("relative_path", PUBLIC_DOCS)
def test_public_docs_do_not_reference_private_files(relative_path):
    path = REPO_ROOT / relative_path
    assert path.exists(), f"{relative_path} is listed as a public doc but missing"
    hits = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if PRIVATE_REFERENCES.search(line)
    ]
    assert not hits, f"{relative_path} references private files:\n" + "\n".join(hits)
