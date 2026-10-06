"""Contract tests for the downstream base pin and Renovate policy."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile.konflux"
pytestmark = pytest.mark.skipif(not DOCKERFILE.exists(), reason="The checkout has no downstream Dockerfile.")


def test_base_image_has_a_default_tag_and_digest():
    text = DOCKERFILE.read_text()
    match = re.search(r"^ARG BASE_IMAGE=(\S+)$", text, re.MULTILINE)
    assert match, "The Dockerfile must contain the authoritative base pin."
    assert re.fullmatch(r"quay\.io/aipcc/base-images/[\w.-]+:[\w.-]+@sha256:[a-f0-9]{64}", match.group(1))
    assert text.index(match.group(0)) < text.index("FROM ${BASE_IMAGE}")


def test_stage_argument_retains_the_image_label():
    text = DOCKERFILE.read_text()
    stage = text.split("FROM ${BASE_IMAGE}", 1)[1]
    assert re.search(r"^ARG BASE_IMAGE$", stage, re.MULTILINE)
    assert 'com.redhat.aiplatform.image="${BASE_IMAGE}"' in stage


def test_legacy_argument_file_cannot_override_the_pin():
    path = ROOT / ".konflux" / "cpu-ubi9.conf"
    if path.exists():
        assert path.read_text().strip() == "", "The compatibility file must remain empty."


def test_complete_image_build_tests_native_imports():
    text = DOCKERFILE.read_text()
    assert 'RUN pip install --no-cache-dir ".[inline]" &&' in text
    assert 'python -c "import pyarrow; import llama_stack_provider_trustyai_garak"' in text


def test_renovate_waits_for_checks_before_a_merge():
    config = json.loads((ROOT / ".github" / "renovate.json").read_text())
    assert config["automerge"] is True
    assert config["ignoreTests"] is False
    assert config["platformAutomerge"] is False
    assert config["automergeType"] == "pr"
    assert config["automergeStrategy"] == "squash"
    assert config["rebaseWhen"] == "behind-base-branch"
    assert config["pinDigests"] is True


def test_renovate_preserves_mintmaker_defaults():
    config = json.loads((ROOT / ".github" / "renovate.json").read_text())
    assert "enabledManagers" not in config
    assert "branchPrefix" not in config
    assert "baseBranches" not in config
    assert "baseBranchPatterns" not in config
    assert "extends" not in config
