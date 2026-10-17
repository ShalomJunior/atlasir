"""Exact CPU FAISS inner-product search with explicit IDs and complete boundary ties."""

import hashlib
from importlib.metadata import version
from typing import Any

import numpy as np

from atlasir.data import identifier, stable_hash
from atlasir.evaluation import ScoredDocument


def _vectors(values: np.ndarray, dimension: int | None = None) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.ndim != 2:
        raise ValueError("Vectors must be a float32 matrix")
    if values.shape[1] < 1 or (dimension is not None and values.shape[1] != dimension):
        raise ValueError("Vector dimension mismatch")
    if not np.isfinite(values).all() or not np.allclose(
        np.linalg.norm(values, axis=1), 1, atol=1e-5
    ):
        raise ValueError("Vectors must be finite and unit normalized")
    return np.ascontiguousarray(values)


class ExactVectorIndex:
    def __init__(self, doc_ids: list[str], vectors: np.ndarray):
        import faiss

        for key in doc_ids:
            identifier(key)
        if not doc_ids or len(set(doc_ids)) != len(doc_ids) or len(doc_ids) != len(vectors):
            raise ValueError("Index requires unique IDs and one vector per document")
        values = _vectors(vectors)
        order = sorted(range(len(doc_ids)), key=lambda row: doc_ids[row])
        self.doc_ids = [doc_ids[row] for row in order]
        self.dimension = values.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(np.ascontiguousarray(values[order]))
        self.row_ids_sha256 = stable_hash(self.doc_ids)

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 1000,
        query_id: str | None = None,
        exclude_self: bool = True,
    ) -> list[ScoredDocument]:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if type(exclude_self) is not bool:
            raise ValueError("exclude_self must be a boolean")
        if query_id is not None:
            identifier(query_id)
        value = _vectors(query, self.dimension)
        if len(value) != 1:
            raise ValueError("Search takes exactly one query vector")
        count = len(self.doc_ids)
        requested = min(count, top_k + 1 + int(exclude_self and query_id in self.doc_ids))
        while True:
            scores, positions = self.index.search(value, requested)
            hits = [
                ScoredDocument(self.doc_ids[int(row)], float(score))
                for score, row in zip(scores[0], positions[0], strict=True)
                if row >= 0 and not (exclude_self and self.doc_ids[int(row)] == query_id)
            ]
            hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
            # Expand until every document tied at the cutoff is present. Sorting
            # only FAISS's initial k results would select arbitrary boundary ties.
            if requested == count or (
                len(hits) >= top_k and hits[top_k - 1].score > float(scores[0][-1])
            ):
                return hits[:top_k]
            requested = min(count, max(requested + 1, requested * 2))

    def snapshot(self) -> tuple[dict[str, Any], bytes]:
        import faiss

        data = faiss.serialize_index(self.index).tobytes()
        vectors = self.index.reconstruct_n(0, len(self.doc_ids))
        return {
            "schema_version": 1,
            "type": "IndexFlatIP",
            "metric": "inner product of unit vectors",
            "faiss_version": version("faiss-cpu"),
            "compile_options": faiss.get_compile_options(),
            "cross_machine_score_tolerance": {"rtol": 1e-6, "atol": 1e-7},
            "dimension": self.dimension,
            "doc_ids": list(self.doc_ids),
            "row_ids_sha256": self.row_ids_sha256,
            "vector_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
            "index_sha256": hashlib.sha256(data).hexdigest(),
            "index_bytes": len(data),
            "ties": "score descending then original string ID ascending; expand cutoff ties",
            "query_batch_size": 1,
        }, data

    @classmethod
    def from_snapshot(cls, metadata: dict[str, Any], data: bytes):
        """Load a verified trusted index; use original vectors to rebuild on new versions."""
        import faiss

        if metadata["faiss_version"] != version("faiss-cpu"):
            raise ValueError("Use historical FAISS or rebuild from saved vector objects")
        if (
            metadata["type"] != "IndexFlatIP"
            or hashlib.sha256(data).hexdigest() != metadata["index_sha256"]
        ):
            raise ValueError("FAISS index identity mismatch")
        ids = metadata["doc_ids"]
        if ids != sorted(set(ids)) or stable_hash(ids) != metadata["row_ids_sha256"]:
            raise ValueError("Index row mapping checksum mismatch")
        for key in ids:
            identifier(key)
        index = faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8).copy())
        if (
            type(index).__name__ != "IndexFlatIP"
            or index.d != metadata["dimension"]
            or index.ntotal != len(ids)
        ):
            raise ValueError("Unexpected FAISS index layout")
        vectors = index.reconstruct_n(0, len(ids))
        _vectors(vectors, metadata["dimension"])
        if hashlib.sha256(vectors.tobytes()).hexdigest() != metadata["vector_sha256"]:
            raise ValueError("Index vectors differ from descriptor")
        instance = cls.__new__(cls)
        instance.doc_ids = list(ids)
        instance.dimension = index.d
        instance.index = index
        instance.row_ids_sha256 = metadata["row_ids_sha256"]
        return instance
