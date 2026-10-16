"""Scoring, historical-index reuse and self-contained baseline restoration."""

import json
import math
import shutil
from pathlib import Path

import pytest

from atlasir.artifacts import export_run, restore_run
from atlasir.baseline import read_config, reevaluate, run_baseline
from atlasir.bm25 import BM25Retriever, resolved_bm25_config
from atlasir.data import Document

ROOT = Path(__file__).resolve().parents[1]


def corpus():
    return {
        "z": Document("z", "", "alpha beta"),
        "a": Document("a", "", "alpha beta"),
        "b": Document("b", "", "other term"),
    }


def test_score_ties_self_filter_and_empty_queries():
    model = BM25Retriever(corpus(), {"top_k": 1})
    assert model.search("ALPHA")[0].doc_id == "a"
    # Native Lucene variant uses tf/(tf + k1 * length normalization).
    expected = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5)) / (1 + 1.2)
    assert model.search("ALPHA")[0].score == pytest.approx(expected, rel=1e-6)
    assert model.search("alpha", query_id="a")[0].doc_id == "z"
    assert model.search("unknownword") == []
    assert model.search("! a") == []
    assert model.tokenize("A, HELLO 42.") == ["hello", "42"]


def test_saved_index_reproduces_rankings():
    model = BM25Retriever(corpus())
    metadata, arrays = model.snapshot()
    # JSON round-trip catches serialization changes to vocabulary/ID types.
    restored = BM25Retriever.from_snapshot(json.loads(json.dumps(metadata)), arrays)
    for query in ("alpha beta", "other", "nothing", ""):
        assert restored.search(query) == model.search(query)
    metadata["version"] = "historical-other-version"
    with pytest.raises(ValueError, match="historical"):
        BM25Retriever.from_snapshot(metadata, arrays)


@pytest.mark.parametrize(
    "config",
    [
        {"k1": 0},
        {"b": 2},
        {"k1": float("nan")},
        {"top_k": True},
        {"stemmer": "english"},
        {"unexpected": 1},
        {"self_matches": []},
    ],
)
def test_bad_configuration(config):
    with pytest.raises(ValueError):
        resolved_bm25_config(config)


def test_yaml_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("k1: 1.2\nk1: 2.0\n")
    with pytest.raises(ValueError, match="unique"):
        read_config(path)


def test_baseline_restores_without_working_inputs(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(ROOT / "src/atlasir/fixtures/beir", data)
    result = run_baseline(
        data=data,
        name="fixture",
        role="train",
        selection=None,
        config=read_config(ROOT / "configs/baselines/bm25.yaml"),
        root=tmp_path,
        source_root=ROOT,
        run_id="e01-roundtrip",
    )
    run = Path(result["path"])
    assert len(json.loads((run / "per_query.json").read_bytes())) == 4
    metadata = json.loads((run / "index.json").read_bytes())
    restored_index = BM25Retriever.from_snapshot(metadata, (run / "index.npz").read_bytes())
    assert restored_index.search("retrieval") == BM25Retriever(
        {
            d["_id"]: Document(d["_id"], d.get("title", ""), d["text"])
            for d in map(json.loads, (data / "corpus.jsonl").read_text().splitlines())
        }
    ).search("retrieval")
    bundle = export_run(run, tmp_path / "backup.bundle")
    shutil.rmtree(run)
    shutil.rmtree(data)
    destination = tmp_path / "restored"
    restore_run(bundle, destination)
    assert reevaluate(destination)["metrics"] == result["metrics"]
    with pytest.raises(ValueError):
        run_baseline(
            data=ROOT / "src/atlasir/fixtures/beir",
            name="fixture",
            role="final",
            selection=None,
            config={},
            root=tmp_path,
            source_root=ROOT,
        )
