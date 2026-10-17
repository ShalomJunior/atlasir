"""Exact FAISS scores, stable cutoff ties, self-filtering and saved row mapping."""

import numpy as np
import pytest

from atlasir.vector_index import ExactVectorIndex

faiss = pytest.importorskip("faiss")


def test_exact_index_matches_hand_scores_and_reload():
    ids = ["z", "a", "b", "c"]
    vectors = np.array([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    index = ExactVectorIndex(ids, vectors)
    query = np.array([[1, 0]], dtype=np.float32)
    result = index.search(query, top_k=10)
    assert [r.doc_id for r in result] == ["a", "z", "b", "c"]
    assert [r.score for r in result] == [1, 1, 0, -1]
    assert index.search(query, top_k=1)[0].doc_id == "a"
    assert index.search(query, top_k=1, query_id="a")[0].doc_id == "z"
    metadata, data = index.snapshot()
    restored = ExactVectorIndex.from_snapshot(metadata, data)
    assert restored.search(query, top_k=10) == result
    metadata["doc_ids"][0] = "tampered"
    with pytest.raises(ValueError, match="mapping"):
        ExactVectorIndex.from_snapshot(metadata, data)


def test_cutoff_ties_expand_past_native_topk():
    ids = [f"d{i:03}" for i in range(40)][::-1]
    vectors = np.tile(np.array([[1, 0]], dtype=np.float32), (len(ids), 1))
    index = ExactVectorIndex(ids, vectors)
    assert [r.doc_id for r in index.search(vectors[:1], top_k=3)] == ["d000", "d001", "d002"]


def test_vectors_and_parameters_rejected():
    matrix = np.array([[1, 0]], dtype=np.float32)
    with pytest.raises(ValueError):
        ExactVectorIndex(["a"], matrix.astype(np.float64))
    with pytest.raises(ValueError):
        ExactVectorIndex(["a"], np.zeros_like(matrix))
    with pytest.raises(ValueError):
        ExactVectorIndex(["a", "a"], np.tile(matrix, (2, 1)))
    index = ExactVectorIndex(["a"], matrix)
    for query in (
        matrix.astype(np.float64),
        np.zeros_like(matrix),
        np.ones((1, 3), dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            index.search(query)
    with pytest.raises(ValueError):
        index.search(matrix, top_k=True)
