"""BEIR-format integrity checks and source/target partition access rules."""

import hashlib
import json
import shutil
from importlib.resources import files

import pytest

from atlasir.data import load_beir, require_development
from atlasir.splits import apply_selection, audit_disjoint, make_selection


@pytest.fixture
def beir(tmp_path):
    root = tmp_path / "beir"
    shutil.copytree(files("atlasir").joinpath("fixtures", "beir"), root)
    return root


def test_loads_complete_corpus_and_only_judged_partition(beir):
    dataset = load_beir(beir, name="fixture", role="train")
    assert set(dataset.corpus) == {"a", "b", "c", "d"}
    assert set(dataset.queries) == {"q-001", "q-002", "q-003", "q-004"}
    assert dataset.qrels["q-001"] == {"a": 3, "b": 1}
    assert dataset.corpus["a"].contents == "Lexical retrieval\nMatch exact query terms with BM25."
    for name, digest in dataset.file_hashes.items():
        assert digest == hashlib.sha256((beir / name).read_bytes()).hexdigest()
    with pytest.raises(TypeError):
        dataset.qrels["q-001"]["a"] = 0


def test_ids_remain_strings_with_leading_zeros(beir):
    (beir / "corpus.jsonl").write_text('{"_id":"001","text":"document"}\n')
    (beir / "queries.jsonl").write_text('{"_id":"002","text":"query"}\n')
    (beir / "qrels/train.tsv").write_text("query-id\tcorpus-id\tscore\n002\t001\t1\n")
    dataset = load_beir(beir, name="fixture", role="train")
    assert list(dataset.corpus) == ["001"]
    assert list(dataset.queries) == ["002"]
    assert dataset.corpus["001"].contents == "document"


@pytest.mark.parametrize(
    "name,role,split,allow",
    [
        ("fixture", "train", "test", False),
        ("fixture", "final", None, False),
        ("fiqa", "train", None, False),
        ("scidocs", "validation", None, True),
        ("scifact", "validation", None, False),
        ("unknown", "train", None, False),
        ("fixture", "typo", None, False),
    ],
)
def test_wrong_role_is_rejected_before_reading_files(tmp_path, name, role, split, allow):
    with pytest.raises(ValueError):
        load_beir(tmp_path / "nonexistent", name=name, role=role, split=split, allow_final=allow)


def test_final_requires_explicit_loading_and_development_guard_rejects_it(beir):
    final = load_beir(beir, name="fiqa", role="final", allow_final=True)
    assert list(final.queries) == ["q-006"]
    with pytest.raises(ValueError, match="development"):
        require_development(final)


@pytest.mark.parametrize("filename", ["corpus.jsonl", "queries.jsonl"])
def test_duplicate_ids_are_rejected(beir, filename):
    path = beir / filename
    with path.open("a") as stream:
        stream.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_beir(beir, name="fixture", role="train")


@pytest.mark.parametrize(
    "row",
    [
        "q-001\tmissing-doc\t1",
        "missing-query\ta\t1",
        "q-001\ta\t-1",
        "q-001\ta\t1.5",
        "q-001\ta",
        "q-001\ta\t1\textra",
    ],
)
def test_malformed_or_dangling_judgments_are_rejected(beir, row):
    (beir / "qrels/train.tsv").write_text("query-id\tcorpus-id\tscore\n" + row + "\n")
    with pytest.raises(ValueError):
        load_beir(beir, name="fixture", role="train")


def test_duplicate_judgment_is_rejected(beir):
    with (beir / "qrels/train.tsv").open("a") as stream:
        stream.write("q-001\ta\t3\n")
    with pytest.raises(ValueError, match="Duplicate query/document"):
        load_beir(beir, name="fixture", role="train")


@pytest.mark.parametrize(
    "line", ['{"_id":1,"text":"x"}', '{"_id":"a","text":null}', "[]", "bad json"]
)
def test_invalid_corpus_records_are_rejected(beir, line):
    (beir / "corpus.jsonl").write_text(line + "\n")
    with pytest.raises(ValueError):
        load_beir(beir, name="fixture", role="train")


def test_split_is_stable_covers_pool_and_groups_duplicate_text(beir):
    pool = load_beir(beir, name="scifact", role="train")
    with pytest.raises(ValueError, match="Partition SciFact"):
        require_development(pool)
    manifest = make_selection(pool)
    assert make_selection(pool) == manifest
    assert set(manifest["train_ids"]) | set(manifest["validation_ids"]) == set(pool.queries)
    assert set(manifest["train_ids"]).isdisjoint(manifest["validation_ids"])
    assert any(group == ["q-003", "q-004"] for group in manifest["groups"])
    train = load_beir(beir, name="scifact", role="train", selection=manifest)
    validation = load_beir(beir, name="scifact", role="validation", selection=manifest)
    assert len(train.corpus) == len(validation.corpus) == 4
    assert train.selection_hash == validation.selection_hash
    assert audit_disjoint(train, validation)["train_queries"] + len(validation.queries) == 4
    require_development(train)
    require_development(validation)


def test_family_grouping_is_transitive_with_duplicate_text(beir):
    path = beir / "queries.jsonl"
    queries = [json.loads(line) for line in path.read_text().splitlines()]
    for query in queries:
        if query["_id"] in {"q-001", "q-003"}:
            query["metadata"] = {"family_id": "claim-family"}
    path.write_text("".join(json.dumps(query) + "\n" for query in queries))
    manifest = make_selection(load_beir(beir, name="fixture", role="train"))
    assert ["q-001", "q-003", "q-004"] in manifest["groups"]


@pytest.mark.parametrize("change", ["overlap", "missing", "hash", "group-crossing"])
def test_invalid_or_stale_split_is_rejected(beir, change):
    pool = load_beir(beir, name="scifact", role="train")
    manifest = make_selection(pool)
    if change == "overlap":
        manifest["train_ids"].append(manifest["validation_ids"][0])
    elif change == "missing":
        manifest["train_ids"].pop()
    elif change == "hash":
        manifest["file_hashes"]["corpus.jsonl"] = "wrong"
    else:
        manifest["train_ids"] = ["q-001", "q-003"]
        manifest["validation_ids"] = ["q-002", "q-004"]
    with pytest.raises(ValueError):
        apply_selection(pool, manifest, "train")


def test_official_source_partitions_can_be_audited_and_duplicate_text_fails(beir):
    train = load_beir(beir, name="nfcorpus", role="train")
    dev = load_beir(beir, name="nfcorpus", role="validation")
    assert audit_disjoint(train, dev) == {"train_queries": 4, "validation_queries": 1}
    with pytest.raises(ValueError, match="official train/dev"):
        make_selection(train)
    path = beir / "queries.jsonl"
    path.write_text(
        path.read_text().replace("Encode a query as a vector.", "PRESERVE experiment records.")
    )
    dev = load_beir(beir, name="nfcorpus", role="validation")
    with pytest.raises(ValueError, match="overlap"):
        audit_disjoint(train, dev)
