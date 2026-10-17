"""Interrupted encoding, portable checkpoints, cache identity and corruption guards."""

import hashlib
import json

import numpy as np
import pytest

from atlasir.checkpoint_io import export_checkpoint, materialize_model, restore_checkpoint
from atlasir.data import stable_hash
from atlasir.encoding import EncodingBatch, resolved_encoder_config
from atlasir.objects import BlobStore, export_objects, import_objects
from atlasir.shards import EmbeddingJob


class FakeEncoder:
    def __init__(self, fail_on=None):
        self.config = resolved_encoder_config()
        self.calls = []
        self.fail_on = fail_on

    def checkpoint_profile(self):
        return getattr(
            self,
            "identity",
            {
                "config": dict(self.config),
                "implementation_sha256": "a" * 64,
                "runtime": {"profile": "synthetic test encoder"},
                "model_files": {"fixture-weights": hashlib.sha256(b"fake model").hexdigest()},
            },
        )

    def encode(self, ids, texts, *, kind):
        self.calls.append(list(ids))
        if len(self.calls) == self.fail_on:
            raise RuntimeError("simulated interruption")
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256((kind + text).encode()).hexdigest()[:8], 16)
            row = np.random.default_rng(seed).normal(size=384).astype(np.float32)
            row /= np.linalg.norm(row)
            vectors.append(row)
        return EncodingBatch(
            tuple(ids),
            np.array(vectors, dtype=np.float32),
            tuple(5 for _ in ids),
            kind,
            stable_hash({"kind": kind, "rows": list(zip(ids, texts, strict=True))}),
            dict(self.config),
        )


def setup(tmp_path):
    store = BlobStore(tmp_path / "objects")
    encoder = FakeEncoder()
    profile = {
        "config": encoder.config,
        "implementation_sha256": "a" * 64,
        "runtime": {"profile": "synthetic test encoder"},
        "model_files": {"fixture-weights": store.put_bytes(b"fake model")},
    }
    rows = {f"d{i}": f"document {i}" for i in range(5)}
    job = EmbeddingJob(tmp_path / "job", store, rows, kind="passage", profile=profile, shard_size=2)
    return store, encoder, profile, rows, job


def test_interrupted_job_resumes_and_matches_clean(tmp_path):
    store, encoder, profile, rows, job = setup(tmp_path)
    encoder.fail_on = 2
    with pytest.raises(RuntimeError, match="interruption"):
        job.run(encoder)
    shard = job.directory / "shards/00000000.json"
    saved = shard.read_bytes()
    attempts = list((job.directory / "attempts").glob("*.json"))
    assert json.loads(attempts[0].read_bytes())["status"] == "failed"
    resumed = FakeEncoder()
    outcome = job.run(resumed)
    assert (
        outcome["status"] == "complete"
        and outcome["reused_shards"] == 1
        and outcome["new_shards"] == 2
    )
    assert resumed.calls == [["d2", "d3"], ["d4"]]
    assert shard.read_bytes() == saved
    clean = EmbeddingJob(
        tmp_path / "clean", store, rows, kind="passage", profile=profile, shard_size=2
    )
    clean.run(FakeEncoder())
    assert clean.arrays()[0] == job.arrays()[0]
    assert np.array_equal(clean.arrays()[1], job.arrays()[1])
    cached = FakeEncoder()
    assert job.run(cached)["reused_shards"] == 3
    assert not cached.calls


def test_partial_export_restores_into_empty_store(tmp_path):
    store, encoder, profile, rows, job = setup(tmp_path)
    assert job.run(encoder, stop_after_new_shards=1)["status"] == "paused"
    with pytest.raises(ValueError, match="incomplete"):
        job.arrays()
    checkpoint = export_checkpoint(job, tmp_path / "checkpoint.zip")
    objects = export_objects(store, job.object_ids(), tmp_path / "objects.zip")
    empty = BlobStore(tmp_path / "elsewhere/objects")
    import_objects(objects, empty)
    restored = restore_checkpoint(checkpoint, tmp_path / "elsewhere/job", empty)
    new = FakeEncoder()
    assert restored.run(new)["reused_shards"] == 1
    assert new.calls == [["d2", "d3"], ["d4"]]
    assert restored.arrays()[0] == sorted(rows)
    with pytest.raises(FileExistsError):
        restore_checkpoint(checkpoint, tmp_path / "elsewhere/job", empty)


@pytest.mark.parametrize("change", ["text", "profile", "shard_size"])
def test_incompatible_resume_rejected(tmp_path, change):
    store, encoder, profile, rows, job = setup(tmp_path)
    job.run(encoder, stop_after_new_shards=1)
    size = 2
    if change == "text":
        rows["d1"] = "changed document"
    elif change == "profile":
        profile = profile | {"runtime": {"profile": "different runtime"}}
    else:
        size = 3
    with pytest.raises(ValueError, match="Incompatible"):
        EmbeddingJob(job.directory, store, rows, kind="passage", profile=profile, shard_size=size)


def test_corrupt_object_and_unknown_shard_fail_before_compute(tmp_path):
    store, encoder, profile, rows, job = setup(tmp_path)
    job.run(encoder, stop_after_new_shards=1)
    record = json.loads((job.directory / "shards/00000000.json").read_bytes())
    payload = store.verify(record["vectors_object"])
    payload.write_bytes(b"broken")
    fresh = FakeEncoder()
    with pytest.raises(ValueError):
        job.run(fresh)
    assert not fresh.calls


def test_single_writer_guard(tmp_path):
    store, encoder, profile, rows, job = setup(tmp_path)
    with job._lock(), pytest.raises(ValueError, match="Another writer"):
        job.run(encoder)


def test_object_dedup_and_offline_model_materialization(tmp_path):
    store = BlobStore(tmp_path / "objects")
    digest = store.put_bytes(b"same bytes")
    assert store.put_bytes(b"same bytes") == digest
    files = {
        name: store.put_bytes(name.encode())
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    }
    model = materialize_model(store, files, tmp_path / "model")
    for name, digest in files.items():
        assert (model / name).read_bytes() == store.read(digest)
    with pytest.raises(ValueError):
        materialize_model(store, files | {"../escape": digest}, tmp_path / "bad-model")


def test_complete_pilot_session_and_fresh_object_restore(tmp_path, monkeypatch):
    import sys
    import types
    from pathlib import Path

    pytest.importorskip("faiss")
    import atlasir.day07 as pilot
    from atlasir.artifacts import restore_run
    from atlasir.data import load_beir

    source = Path(__file__).resolve().parents[1]
    weights = tmp_path / "model-files"
    weights.mkdir()
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        (weights / name).write_bytes(("fake unit-test " + name).encode())
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda *args, **kwargs: str(weights)),
    )

    def factory(config, **kwargs):
        encoder = FakeEncoder()
        encoder.config = dict(config)
        return encoder

    monkeypatch.setattr(pilot, "E5Encoder", factory)

    def fake_profile(encoder, files):
        encoder.identity = {
            "config": dict(encoder.config),
            "implementation_sha256": "a" * 64,
            "runtime": {"profile": "synthetic test encoder"},
            "model_files": files,
        }
        return encoder.identity

    monkeypatch.setattr(pilot, "encoder_profile", fake_profile)
    config = pilot.resolved_shard_pilot_config(
        {"document_limit": 4, "query_limit": 2, "shard_size": 2}
    )
    dataset = load_beir(source / "src/atlasir/fixtures/beir", name="fixture", role="train")
    inputs = tmp_path / "pilot-inputs.json"
    inputs.write_text(json.dumps(pilot.prepare_pilot_inputs(dataset, config)))
    result = pilot.run_shard_pilot(
        inputs_path=inputs, config=config, root=tmp_path / "output", source_root=source
    )
    session = Path(result["session_exports"])
    empty = BlobStore(tmp_path / "restored-objects")
    import_objects(session / "embedding-objects.zip", empty)
    restored = tmp_path / "restored-run"
    restore_run(session / f"{result['run_id']}.bundle", restored)
    check = pilot.verify_shard_pilot(restored, empty)
    assert check["documents"] == 4 and check["queries"] == 2
    assert check["resume"] == "passed" and check["index_rankings"] == "exact match"
    from atlasir.vector_index import ExactVectorIndex

    original_search = ExactVectorIndex.search

    def rounded_search(self, *args, **kwargs):
        return [
            type(hit)(hit.doc_id, hit.score + 1e-8)
            for hit in original_search(self, *args, **kwargs)
        ]

    monkeypatch.setattr(ExactVectorIndex, "search", rounded_search)
    assert (
        "scores within recorded tolerance"
        in pilot.verify_shard_pilot(restored, empty)["index_rankings"]
    )

    def wrong_search(self, *args, **kwargs):
        return [
            type(hit)(hit.doc_id, hit.score + 0.01)
            for hit in original_search(self, *args, **kwargs)
        ]

    monkeypatch.setattr(ExactVectorIndex, "search", wrong_search)
    with pytest.raises(ValueError, match="scores exceed"):
        pilot.verify_shard_pilot(restored, empty)


def test_object_transport_corruption_is_rejected(tmp_path):
    import zipfile

    store = BlobStore(tmp_path / "objects")
    digest = store.put_bytes(b"valuable vector bytes")
    original = export_objects(store, [digest], tmp_path / "good.zip")
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(original) as incoming, zipfile.ZipFile(broken, "x") as outgoing:
        for name in incoming.namelist():
            outgoing.writestr(
                name,
                b"x" * len(incoming.read(name))
                if name.endswith("payload.bin")
                else incoming.read(name),
            )
    with pytest.raises(ValueError):
        import_objects(broken, BlobStore(tmp_path / "empty"))


def test_unknown_checkpoint_shard_rejected_before_encoding(tmp_path):
    store, encoder, profile, rows, job = setup(tmp_path)
    job.run(encoder, stop_after_new_shards=1)
    (job.directory / "shards/99999999.json").write_text("{}")
    new = FakeEncoder()
    with pytest.raises(ValueError, match="unexpected shard"):
        job.run(new)
    assert not new.calls


def test_day07_notebook_uses_actual_archive_safe_bootstrap(tmp_path):
    import base64
    from pathlib import Path

    from atlasir.artifacts import source_snapshot

    source = Path(__file__).resolve().parents[1]
    data, _ = source_snapshot(source)
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    manifest = {
        "schema_version": 2,
        "job": "day07-test",
        "base64_files": ["source.tar.gz"],
        "sha256": {"source.tar.gz": hashlib.sha256(data).hexdigest()},
    }
    (uploaded / "source.tar.gz.b64").write_bytes(base64.b64encode(data))
    (uploaded / "launcher-input.json").write_text(json.dumps(manifest))
    notebook = json.loads((source / "notebooks/kaggle-day07.ipynb").read_bytes())
    namespace = {"ATLASIR_INPUT_ROOT": uploaded, "ATLASIR_WORKING_ROOT": tmp_path / "work"}
    exec(compile("".join(notebook["cells"][1]["source"]), "day07-bootstrap", "exec"), namespace)
    assert (namespace["SOURCE"] / "configs/dense/day07-pilot.json").is_file()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "day07-cell", "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_actual_encoder_profile_is_checked_at_execution(tmp_path):
    store, encoder, profile, rows, job = setup(tmp_path)
    encoder.identity = profile | {"runtime": {"profile": "unexpected runtime"}}
    with pytest.raises(ValueError, match="runtime differ"):
        job.run(encoder)
    assert not encoder.calls
