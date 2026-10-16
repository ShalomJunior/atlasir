"""Independent hand-calculated expectations, coverage and ranking boundary cases."""

import json
import math
import shutil
from importlib.resources import files

import pytest

from atlasir.data import load_beir
from atlasir.evaluation import ScoredDocument, evaluate


@pytest.fixture
def beir(tmp_path):
    root = tmp_path / "beir"
    shutil.copytree(files("atlasir").joinpath("fixtures", "beir"), root)
    return root


def test_hand_calculated_graded_binary_missing_and_zero_positive_cases(beir):
    dataset = load_beir(beir, name="fixture", role="train")
    raw = json.loads((beir / "rankings.json").read_text())
    rankings = {q: [ScoredDocument(**row) for row in rows] for q, rows in raw.items()}
    result = evaluate(dataset, rankings)
    graded = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    binary = 1 / math.log2(3)
    assert result["per_query"]["q-001"]["nDCG@10"] == pytest.approx(graded)
    assert result["per_query"]["q-003"]["nDCG@10"] == pytest.approx(binary)
    assert result["per_query"]["q-002"] == dict.fromkeys(result["metrics"], 0.0)
    assert result["metrics"]["nDCG@10"] == pytest.approx((graded + binary + 1) / 4)
    assert result["metrics"]["Recall@100"] == pytest.approx(0.75)
    assert result["metrics"]["MRR@10"] == pytest.approx(0.625)
    assert result["coverage"]["missing_queries"] == ["q-002"]
    assert result["coverage"]["no_positive_judgments"] == 1
    assert result["evaluator"]["provider"] == "pytrec_eval"


def test_all_missing_rankings_still_evaluates_every_query(beir):
    result = evaluate(load_beir(beir, name="fixture", role="train"), {})
    assert result["metrics"] == dict.fromkeys(result["metrics"], 0.0)
    assert len(result["per_query"]) == 4
    assert result["coverage"]["empty_rankings"] == 4


def test_provider_cannot_reverse_explicit_tie_order(beir):
    dataset = load_beir(beir, name="fixture", role="train")
    result = evaluate(dataset, {"q-001": [ScoredDocument("b", 5), ScoredDocument("a", 5)]})
    assert result["per_query"]["q-001"]["nDCG@10"] == pytest.approx(1)


@pytest.mark.parametrize("position", [10, 11, 100, 101, 1000, 1001])
def test_cutoff_boundaries(beir, position):
    docs = [{"_id": f"d{x:03}", "text": "distractor"} for x in range(position - 1)]
    docs.append({"_id": "relevant", "text": "answer"})
    (beir / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    (beir / "queries.jsonl").write_text('{"_id":"query","text":"find answer"}\n')
    (beir / "qrels/train.tsv").write_text("query-id\tcorpus-id\tscore\nquery\trelevant\t1\n")
    rankings = {"query": [ScoredDocument(d["_id"], position - i) for i, d in enumerate(docs)]}
    result = evaluate(load_beir(beir, name="fixture", role="train"), rankings)["metrics"]
    assert result["Recall@100"] == (1 if position <= 100 else 0)
    assert result["Recall@1000"] == (1 if position <= 1000 else 0)
    assert result["MRR@10"] == (1 / position if position <= 10 else 0)
    assert result["nDCG@10"] == pytest.approx(1 / math.log2(position + 1) if position <= 10 else 0)


@pytest.mark.parametrize(
    "hits",
    [
        [ScoredDocument("a", float("nan"))],
        [ScoredDocument("a", float("inf"))],
        [ScoredDocument("a", True)],
        [ScoredDocument("missing", 1)],
        [ScoredDocument("a", 2), ScoredDocument("a", 1)],
    ],
)
def test_bad_scores_unknown_documents_and_duplicates_are_rejected(beir, hits):
    with pytest.raises(ValueError):
        evaluate(load_beir(beir, name="fixture", role="train"), {"q-001": hits})


def test_wrong_query_partition_and_final_role_are_rejected(beir):
    train = load_beir(beir, name="fixture", role="train")
    with pytest.raises(ValueError, match="wrong-partition"):
        evaluate(train, {"q-006": []})
    final = load_beir(beir, name="fixture", role="final", allow_final=True)
    with pytest.raises(ValueError, match="development"):
        evaluate(final, {})
    assert evaluate(final, {}, purpose="final")["metrics"]["nDCG@10"] == 0
    with pytest.raises(ValueError):
        evaluate(train, {}, purpose="final")


def test_self_match_policy_is_explicit_and_counted(beir):
    path = beir / "queries.jsonl"
    path.write_text(path.read_text().replace('"q-001"', '"a"'))
    path = beir / "qrels/train.tsv"
    path.write_text(path.read_text().replace("q-001", "a"))
    dataset = load_beir(beir, name="fixture", role="train")
    ranking = {"a": [ScoredDocument("a", 2), ScoredDocument("b", 1)]}
    excluded = evaluate(dataset, ranking)
    kept = evaluate(dataset, ranking, self_matches="keep")
    assert excluded["coverage"]["self_matches_excluded"] == 1
    assert excluded["per_query"]["a"]["Recall@100"] == 0.5
    assert kept["per_query"]["a"]["nDCG@10"] == 1
