"""Explicit BM25S lexical scoring and deterministic positive-score top-k ordering."""

import io
import math
import re
from collections.abc import Mapping
from importlib.metadata import version
from typing import Any

from atlasir.data import Document, stable_hash
from atlasir.evaluation import ScoredDocument


def resolved_bm25_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    defaults = {
        "schema_version": 2,
        "family": "E01",
        "protocol": "day04-bm25-v1",
        "seed": 17,
        "method": "lucene",
        "idf_method": "lucene",
        "backend": "numpy",
        "dtype": "float32",
        "k1": 1.2,
        "b": 0.75,
        "top_k": 1000,
        "lowercase": True,
        "token_pattern": r"(?u)\b\w\w+\b",
        "stopwords": "none",
        "stemmer": "none",
        "self_matches": "exclude",
    }
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, Mapping) or overrides.keys() - defaults.keys():
        raise ValueError("BM25 config must contain only the declared fields")
    config = defaults | dict(overrides)
    variable = {"k1", "b", "top_k", "seed", "self_matches"}
    for key in defaults.keys() - variable:
        if type(config[key]) is not type(defaults[key]) or config[key] != defaults[key]:
            raise ValueError(f"Unsupported BM25 setting: {key}")
    for key in ("k1", "b"):
        if type(config[key]) not in {int, float} or not math.isfinite(config[key]):
            raise ValueError(f"{key} must be a finite number")
    if config["k1"] <= 0 or not 0 <= config["b"] <= 1:
        raise ValueError("k1 must be positive and b between zero and one")
    if type(config["top_k"]) is not int or not 1 <= config["top_k"] <= 10000:
        raise ValueError("top_k must be an integer between 1 and 10000")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be a uint32 integer")
    if not isinstance(config["self_matches"], str) or config["self_matches"] not in {
        "exclude",
        "keep",
    }:
        raise ValueError("Invalid self-match policy")
    return config


class BM25Retriever:
    """CPU baseline. Full score sorting is deliberate and included in query latency."""

    def __init__(self, corpus: Mapping[str, Document], config: Mapping[str, Any] | None = None):
        import bm25s

        self.config = resolved_bm25_config(config)
        self.pattern = re.compile(self.config["token_pattern"])
        self.doc_ids = sorted(corpus)
        if not self.doc_ids:
            raise ValueError("Cannot index an empty corpus")
        tokens = [self.tokenize(corpus[doc_id].contents) for doc_id in self.doc_ids]
        if not any(tokens):
            raise ValueError("Corpus contains no indexable terms")
        self.model = bm25s.BM25(
            method=self.config["method"],
            idf_method=self.config["idf_method"],
            k1=self.config["k1"],
            b=self.config["b"],
            backend="numpy",
            dtype="float32",
        )
        self.model.index(tokens, show_progress=False)
        self.corpus_text_hash = stable_hash(
            [[doc_id, corpus[doc_id].contents] for doc_id in self.doc_ids]
        )

    def tokenize(self, text: str) -> list[str]:
        return self.pattern.findall(text.lower())

    def search(self, text: str, *, query_id: str | None = None) -> list[ScoredDocument]:
        tokens = [token for token in self.tokenize(text) if token in self.model.vocab_dict]
        if not tokens:
            return []
        scores = self.model.get_scores(tokens)
        hits = [
            ScoredDocument(doc_id, float(score))
            for doc_id, score in zip(self.doc_ids, scores, strict=True)
            if score > 0 and not (self.config["self_matches"] == "exclude" and query_id == doc_id)
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.doc_id))[: self.config["top_k"]]

    def snapshot(self) -> tuple[dict[str, Any], bytes]:
        """Save the actual eager-score arrays, vocabulary and original document mapping."""
        import numpy as np

        scores = self.model.scores
        output = io.BytesIO()
        np.savez_compressed(
            output, data=scores["data"], indices=scores["indices"], indptr=scores["indptr"]
        )
        arrays = output.getvalue()
        metadata = {
            "backend": "bm25s",
            "version": version("bm25s"),
            "config": self.config,
            "doc_ids": self.doc_ids,
            "vocabulary": self.model.vocab_dict,
            "corpus_text_sha256": self.corpus_text_hash,
            "sparse_array_bytes": sum(scores[key].nbytes for key in ("data", "indices", "indptr")),
            "compressed_array_bytes": len(arrays),
            "documents": len(self.doc_ids),
            "terms": len(self.model.vocab_dict) - ("" in self.model.vocab_dict),
            "ranking": "positive scores only; descending score then ascending string ID",
        }
        return metadata, arrays

    @classmethod
    def from_snapshot(cls, metadata: Mapping[str, Any], arrays: bytes) -> "BM25Retriever":
        import bm25s
        import numpy as np

        if metadata["version"] != version("bm25s"):
            raise ValueError("Restore the historical BM25S environment before loading its index")
        instance = cls.__new__(cls)
        instance.config = resolved_bm25_config(metadata["config"])
        instance.pattern = re.compile(instance.config["token_pattern"])
        instance.doc_ids = list(metadata["doc_ids"])
        instance.corpus_text_hash = metadata["corpus_text_sha256"]
        instance.model = bm25s.BM25(
            method="lucene",
            idf_method="lucene",
            backend="numpy",
            dtype="float32",
            k1=instance.config["k1"],
            b=instance.config["b"],
        )
        with np.load(io.BytesIO(arrays), allow_pickle=False) as saved:
            instance.model.scores = {
                key: saved[key].copy() for key in ("data", "indices", "indptr")
            }
        instance.model.scores["num_docs"] = len(instance.doc_ids)
        instance.model.vocab_dict = dict(metadata["vocabulary"])
        instance.model.nonoccurrence_array = None
        return instance
