"""Exercise the real offline Kaggle handoff with a tiny preserved reference."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from atlasir.artifacts import export_run
from atlasir.baseline import run_baseline
from atlasir.reproduction import compare_runs

ROOT = Path(__file__).resolve().parents[1]


def test_handoff_roundtrip_and_comparison_guard(tmp_path):
    reference = run_baseline(
        data=ROOT / "src/atlasir/fixtures/beir",
        name="fixture",
        role="train",
        selection=None,
        config={},
        root=tmp_path / "reference",
        source_root=ROOT,
    )
    run = Path(reference["path"])
    assert compare_runs(run, run)["scores_exact"]
    bundle = export_run(run, tmp_path / "reference.bundle")
    attribution = tmp_path / "attribution.md"
    attribution.write_text("Synthetic fixture: no real dataset included.\n")
    package = tmp_path / "handoff.zip"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_kaggle_day05.py"),
            "--reference",
            str(bundle),
            "--attribution",
            str(attribution),
            "--destination",
            str(package),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inputs = tmp_path / "inputs"
    with zipfile.ZipFile(package) as archive:
        archive.extractall(inputs)
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/kaggle_day05.py"),
            "--input",
            str(inputs),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["comparison"]["metrics_and_ranked_ids"] == "exact match"
    assert (output / "day05-baseline-export.zip").is_file()
    # Input identity must be checked before creating any new output or executing retrieval.
    (inputs / "SCIFACT-ATTRIBUTION.md").write_text("corrupted transfer")
    broken = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/kaggle_day05.py"),
            "--input",
            str(inputs),
            "--output",
            str(tmp_path / "must-not-exist"),
        ],
        capture_output=True,
        text=True,
    )
    assert broken.returncode != 0
    assert "hash mismatch" in broken.stderr
    assert not (tmp_path / "must-not-exist").exists()
    different = run_baseline(
        data=ROOT / "src/atlasir/fixtures/beir",
        name="fixture",
        role="train",
        selection=None,
        config={"k1": 2},
        root=tmp_path / "different",
        source_root=ROOT,
    )
    with pytest.raises(ValueError, match="config.resolved"):
        compare_runs(run, Path(different["path"]))


def test_notebook_contains_executable_python_without_saved_outputs():
    notebook = json.loads((ROOT / "notebooks/kaggle-day05.ipynb").read_bytes())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "kaggle-day05", "exec")
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
