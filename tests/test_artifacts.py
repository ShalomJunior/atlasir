"""Behavioral checks for preservation, failed attempts, isolation and corrupted transfers."""

import hashlib
import io
import json
import shutil
import tarfile
from importlib.resources import files
from pathlib import Path

import pytest

from atlasir.artifacts import (
    RunWriter,
    export_run,
    resolved_fixture_config,
    restore_run,
    source_snapshot,
    verify_run,
)
from atlasir.cli import main


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "src" / "atlasir").mkdir(parents=True)
    (root / "src" / "atlasir" / "__init__.py").write_text("# preserved source\n")
    (root / "pyproject.toml").write_text('[project]\nname = "fixture"\n')
    (root / "uv.lock").write_text("# fixture lockfile\n")
    return root


def writer(root, source, run_id="e00-test"):
    return RunWriter(
        root,
        source,
        resolved_fixture_config(),
        files("atlasir").joinpath("fixtures", "smoke.json").read_bytes(),
        run_id,
    )


def finished(root, source, run_id="e00-test"):
    with writer(root, source, run_id) as run:
        run.write_json("results.json", {"documents": 3, "queries": 2, "qrels": 2})
        return run.seal()


def test_roundtrip_restores_without_original_source_or_run(tmp_path, source):
    run = finished(tmp_path, source)
    expected = verify_run(run)
    saved_source = (run / "source.tar.gz").read_bytes()
    bundle = export_run(run, tmp_path / "backup.bundle")
    shutil.rmtree(run)
    shutil.rmtree(source)
    destination = tmp_path / "independent-workspace" / "restored"
    assert restore_run(bundle, destination) == expected
    assert (destination / "source.tar.gz").read_bytes() == saved_source
    assert verify_run(destination)["status"] == "success"


def test_same_spec_new_run_and_changed_source_new_spec(tmp_path, source):
    first = finished(tmp_path, source, "first")
    second = finished(tmp_path, source, "second")
    assert verify_run(first)["spec_id"] == verify_run(second)["spec_id"]
    (source / "src/atlasir/__init__.py").write_text("# corrected source\n")
    third = finished(tmp_path, source, "third")
    assert verify_run(first)["spec_id"] != verify_run(third)["spec_id"]


def test_duplicate_id_and_output_cannot_overwrite(tmp_path, source):
    with writer(tmp_path, source) as run:
        run.write_json("results.json", {"value": 1})
        with pytest.raises(FileExistsError):
            run.write_json("results.json", {"value": 2})
        saved = run.seal()
        with pytest.raises(ValueError, match="closed"):
            run.write_json("anything.json", {})
    with pytest.raises(FileExistsError):
        writer(tmp_path, source)
    assert json.loads((saved / "results.json").read_bytes()) == {"value": 1}


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failed_or_interrupted_attempt_is_preserved(tmp_path, source, failure):
    with pytest.raises(failure), writer(tmp_path, source) as run:
        run.write_json("partial.json", {"completed_shards": [1]})
        raise failure()
    scratch = tmp_path / ".scratch/e00-test"
    assert (scratch / "partial.json").exists()
    assert not (scratch / "COMPLETE.json").exists()
    manifest = json.loads((scratch / "manifest.json").read_bytes())
    assert manifest["status"] == ("interrupted" if failure is KeyboardInterrupt else "failed")
    with pytest.raises(ValueError):
        verify_run(scratch)
    with pytest.raises(FileExistsError):
        writer(tmp_path, source)


def test_unsealed_exit_is_not_success(tmp_path, source):
    with pytest.raises(ValueError, match="without being sealed"), writer(tmp_path, source):
        pass
    assert not (tmp_path / "artifacts/runs/e00-test").exists()


@pytest.mark.parametrize("change", ["modified", "missing", "extra"])
def test_verification_rejects_changed_files(tmp_path, source, change):
    run = finished(tmp_path, source)
    if change == "modified":
        (run / "results.json").write_text("{}")
    elif change == "missing":
        (run / "results.json").unlink()
    else:
        (run / "extra.txt").write_text("unlisted")
    with pytest.raises(ValueError):
        verify_run(run)


def test_export_and_restore_refuse_existing_destinations(tmp_path, source):
    run = finished(tmp_path, source)
    bundle = export_run(run, tmp_path / "copy.bundle")
    with pytest.raises(FileExistsError):
        export_run(run, bundle)
    restored = tmp_path / "restored"
    restore_run(bundle, restored)
    with pytest.raises(FileExistsError):
        restore_run(bundle, restored)
    assert verify_run(restored)["status"] == "success"


def test_corrupt_archive_is_rejected_without_publishing(tmp_path, source):
    bundle = export_run(finished(tmp_path, source), tmp_path / "corrupt.bundle")
    with (bundle / "run.tar.gz").open("ab") as stream:
        stream.write(b"corruption")
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="checksum or size"):
        restore_run(bundle, destination)
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["traversal", "symlink", "duplicate"])
def test_unsafe_archive_is_rejected_even_with_matching_outer_hash(tmp_path, kind):
    bundle = tmp_path / "unsafe.bundle"
    bundle.mkdir()
    archive = bundle / "run.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("../escape" if kind == "traversal" else "file")
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "../escape"
        stream.addfile(member, io.BytesIO())
        if kind == "duplicate":
            stream.addfile(member, io.BytesIO())
    (bundle / "archive.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        )
    )
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="unsafe or duplicate"):
        restore_run(bundle, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape").exists()


def test_source_snapshot_is_stable_and_excludes_generated_files_and_secrets(source):
    before, _ = source_snapshot(source)
    (source / ".env").write_text("SECRET=not-for-export")
    (source / "artifacts").mkdir()
    (source / "artifacts" / "large.bin").write_bytes(b"unrelated")
    (source / "private" / "docs").mkdir(parents=True)
    (source / "private" / "docs" / "PLAN.md").write_text("Personal working notes")
    after, _ = source_snapshot(source)
    assert before == after


@pytest.mark.parametrize("overrides", [{"seed": True}, {"typo": 1}, {"seed": -1}])
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        resolved_fixture_config(overrides)


@pytest.mark.parametrize("run_id", ["../escape", "a/b", ".", "UPPERCASE"])
def test_invalid_run_id_is_rejected(tmp_path, source, run_id):
    with pytest.raises(ValueError, match="Invalid run ID"):
        writer(tmp_path, source, run_id)
    assert not (tmp_path / ".scratch").exists()


def test_cli_fixture_export_restore(tmp_path, source, capsys):
    actual_source = Path(__file__).resolve().parents[1]
    assert main(["run-fixture", "--root", str(tmp_path), "--source-root", str(actual_source)]) == 0
    result = json.loads(capsys.readouterr().out)
    run = Path(result["path"])
    assert not result["reportable"]
    bundle = tmp_path / "cli.bundle"
    assert main(["export", str(run), str(bundle)]) == 0
    capsys.readouterr()
    restored = tmp_path / "restored"
    assert main(["restore", str(bundle), str(restored)]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == result["run_id"]


def test_cli_rejects_unrelated_source_tree(tmp_path, source):
    with pytest.raises(SystemExit) as failure:
        main(["run-fixture", "--root", str(tmp_path), "--source-root", str(source)])
    assert failure.value.code == 1
    assert not (tmp_path / ".scratch").exists()


def test_active_writer_reserves_id_and_missing_outputs_cannot_seal(tmp_path, source):
    with writer(tmp_path, source) as run:
        with pytest.raises(FileExistsError):
            writer(tmp_path, source)
        with pytest.raises(ValueError, match="missing required outputs"):
            run.seal()
        assert not (run.path / "COMPLETE.json").exists()
        run.write_json("results.json", {})
        final = run.seal()
    catalog = json.loads((tmp_path / "experiments/catalog/e00-test.json").read_bytes())
    assert catalog["run_path"] == str(final.relative_to(tmp_path))
    assert catalog["state"] == "local-sealed"
