"""Exercise installed entry points and package data away from the source directory."""

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("entry_point", [["atlasir"], [sys.executable, "-m", "atlasir"]])
def test_installed_smoke_reads_packaged_fixture(entry_point, tmp_path):
    completed = subprocess.run(
        [*entry_point, "smoke"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "ok"
    assert result["scope"] == "installation-only"
    assert (result["documents"], result["queries"], result["qrels"]) == (3, 2, 2)
    assert result["python"].startswith(("3.11.", "3.12."))
    assert list(tmp_path.iterdir()) == []


def test_unknown_command_fails(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "atlasir", "not-a-command"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "invalid choice" in completed.stderr
