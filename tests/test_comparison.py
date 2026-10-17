"""Known fusion/headroom outcomes and a full portable comparison without model downloads."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from atlasir.artifacts import export_run, restore_run
from atlasir.comparison import (
    METHODS,
    opportunity,
    reevaluate_comparison,
    resolved_comparison_config,
    rrf,
    run_comparison,
)
from atlasir.data import load_beir, stable_hash
from atlasir.encoding import EncodingBatch, resolved_encoder_config
from atlasir.evaluation import ScoredDocument as Hit
from atlasir.splits import make_selection

ROOT = Path(__file__).resolve().parents[1]


def test_rrf_missing_candidates_ties_and_score_scale():
    hits = rrf([Hit("b", 1000), Hit("a", 1)], [Hit("a", 0.1), Hit("c", 0.09)])
    assert [h.doc_id for h in hits] == ["a", "b", "c"]
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert rrf([Hit("b", 1)], [Hit("a", 1)])[0].doc_id == "a"
    assert rrf([], []) == []


def test_oracle_distinguishes_unique_wins_and_ties():
    values = {"bm25": [1.0, 0.0, 0.5], "dense": [0.0, 1.0, 0.5], "rrf": [0.5, 0.5, 0.5]}
    evaluations = {
        m: {
            "per_query": {str(i): {"nDCG@10": v} for i, v in enumerate(scores)},
            "metrics": {"nDCG@10": sum(scores) / 3},
        }
        for m, scores in values.items()
    }
    result = opportunity(evaluations)
    assert result["oracle_ndcg10"] == pytest.approx(2.5 / 3)
    assert result["oracle_gap"] == pytest.approx(1 / 3)
    assert result["unique_wins"] == {"bm25": 1, "dense": 1, "rrf": 0}
    assert result["tied_queries"] == 1


class FixtureEncoder:
    def __init__(self, directory):
        self.config = resolved_encoder_config()
        self._checkpoint_model_directory = directory
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"test weights only")

    def checkpoint_profile(self):
        return {
            "config": self.config,
            "implementation_sha256": "a" * 64,
            "runtime": {"scope": "synthetic integration test, no model inference"},
            "model_files": {"model.safetensors": hashlib.sha256(b"test weights only").hexdigest()},
        }

    def encode(self, ids, texts, *, kind):
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256((kind + text).encode()).hexdigest()[:8], 16)
            vector = np.random.default_rng(seed).normal(size=384).astype(np.float32)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return EncodingBatch(
            tuple(ids),
            np.array(vectors, dtype=np.float32),
            tuple(7 for _ in ids),
            kind,
            stable_hash({"kind": kind, "rows": list(zip(ids, texts, strict=True))}),
            self.config,
        )


def test_full_session_recalculates_from_portable_inputs_and_reuses_shards(tmp_path):
    pytest.importorskip("faiss")
    data = tmp_path / "data"
    shutil.copytree(ROOT / "src/atlasir/fixtures/beir", data)
    pool = load_beir(data, name="fixture", role="train")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(make_selection(pool)))
    encoder = FixtureEncoder(tmp_path / "model")
    kwargs = dict(
        data=data,
        selection=selection,
        name="fixture",
        config=resolved_comparison_config(),
        root=tmp_path / "output",
        source_root=ROOT,
        encoder=encoder,
    )
    result = run_comparison(**kwargs)
    run = Path(result["path"])
    assert reevaluate_comparison(run)["status"] == "recalculated"
    summary = json.loads((run / "results.json").read_bytes())
    assert set(summary["metrics"]) == {"train", "validation"}
    assert set(summary["metrics"]["validation"]) == set(METHODS)
    bundle = export_run(run, tmp_path / "portable.bundle")
    shutil.rmtree(data)
    shutil.rmtree(run)
    restored = tmp_path / "restored"
    restore_run(bundle, restored)
    assert reevaluate_comparison(restored)["status"] == "recalculated"
    # Inputs restored for a distinct attempt; every verified vector shard is reused.
    shutil.copytree(ROOT / "src/atlasir/fixtures/beir", data)
    retry = run_comparison(**kwargs)
    timings = json.loads((Path(retry["path"]) / "timing.json").read_bytes())
    assert retry["run_id"] != result["run_id"]
    assert all(a["new_shards"] == 0 for a in timings["attempts"].values())


def test_real_upload_package_and_notebook_bootstrap(tmp_path):
    import subprocess
    import sys
    import zipfile

    data = tmp_path / "data"
    shutil.copytree(ROOT / "src/atlasir/fixtures/beir", data)
    pool = load_beir(data, name="scifact", role="train")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(make_selection(pool)))
    archive = tmp_path / "input.zip"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_kaggle_comparison.py"),
            "--data",
            str(data),
            "--selection",
            str(selection),
            "--destination",
            str(archive),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    uploaded = tmp_path / "uploaded"
    with zipfile.ZipFile(archive) as bundle:
        assert "data/qrels/test.tsv" not in bundle.namelist()
        assert "source.tar.gz.b64" in bundle.namelist()
        bundle.extractall(uploaded)
    notebook = json.loads((ROOT / "notebooks/kaggle-scifact-comparison.ipynb").read_bytes())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<comparison-notebook>", "exec")
    scope = {"ATLASIR_INPUT_ROOT": str(uploaded), "ATLASIR_WORKING_ROOT": str(tmp_path / "work")}
    exec(compile("".join(notebook["cells"][1]["source"]), "<bootstrap>", "exec"), scope)
    assert (scope["SOURCE"] / "scripts/run_comparison.py").is_file()
    assert (scope["INPUT"] / "data/corpus.jsonl").read_bytes() == (
        data / "corpus.jsonl"
    ).read_bytes()
    assert scope["manifest"]["datasets"]["train"]["documents"] == 4
    # Once selected, a wrong job must fail before model/setup execution.
    manifest_path = uploaded / "launcher-input.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["job"] = "day07-shards-index-pilot"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="comparison input"):
        exec(compile("".join(notebook["cells"][1]["source"]), "<bootstrap>", "exec"), scope)
