"""BEIR-format loading and explicit partition roles, without dataset downloads."""

import csv
import hashlib
import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

Role = Literal["train", "validation", "final"]
SOURCE_DATASETS = {"scifact", "nfcorpus", "fixture"}
HELD_OUT_DATASETS = {"fiqa", "scidocs"}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError("IDs must be nonempty strings without whitespace")
    return value


def normalized_query(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def contents(self) -> str:
        return "\n".join(part for part in (self.title, self.text) if part)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    family_id: str | None = None


@dataclass(frozen=True)
class Dataset:
    name: str
    role: Role
    official_split: str
    corpus: Mapping[str, Document]
    queries: Mapping[str, Query]
    qrels: Mapping[str, Mapping[str, int]]
    file_hashes: Mapping[str, str]
    selection_hash: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "role": self.role,
            "official_split": self.official_split,
            "documents": len(self.corpus),
            "queries": len(self.queries),
            "judgments": sum(map(len, self.qrels.values())),
            "file_hashes": dict(self.file_hashes),
            "query_ids_sha256": stable_hash(sorted(self.queries)),
            "selection_hash": self.selection_hash,
        }


def require_development(dataset: Dataset) -> None:
    if dataset.role == "final" or dataset.name not in SOURCE_DATASETS:
        raise ValueError("Final or held-out data cannot be used for development")
    if dataset.name == "scifact" and dataset.selection_hash is None:
        raise ValueError("Partition SciFact's training pool before development evaluation")


def _lines(path: Path, digest: Any):
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            try:
                yield number, raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{path.name}:{number}: expected UTF-8") from error


def _jsonl(path: Path, digest: Any):
    for number, line in _lines(path, digest):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}:{number}: invalid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path.name}:{number}: expected an object")
        yield record


def load_beir(
    root: Path,
    *,
    name: str,
    role: Role,
    split: str | None = None,
    selection: Mapping[str, Any] | None = None,
    allow_final: bool = False,
) -> Dataset:
    """Read a complete corpus and only the official judged query partition.

    An unpartitioned SciFact train pool may be loaded to construct a split, but
    require_development() rejects using that pool in development evaluation.
    """
    if name not in SOURCE_DATASETS | HELD_OUT_DATASETS or role not in {
        "train",
        "validation",
        "final",
    }:
        raise ValueError("Unknown dataset name or partition role")
    if role == "final" and not allow_final:
        raise ValueError("Final data requires explicit allow_final=True")
    if name in HELD_OUT_DATASETS and role != "final":
        raise ValueError("Held-out datasets are final-only")
    expected = {"train": "train", "validation": "dev", "final": "test"}[role]
    if role == "validation" and (
        name == "scifact" or (name == "fixture" and selection is not None)
    ):
        expected = "train"
        if selection is None:
            raise ValueError("SciFact validation requires a training-pool split manifest")
    split = expected if split is None else split
    if split != expected:
        raise ValueError(f"Role {role} requires official split {expected}")
    if selection is not None and (split != "train" or role == "final"):
        raise ValueError("Training-pool selections cannot be applied to this official split")

    corpus_digest, queries_digest, qrels_digest = (hashlib.sha256() for _ in range(3))
    corpus = {}
    for record in _jsonl(root / "corpus.jsonl", corpus_digest):
        doc_id = identifier(record.get("_id"))
        if doc_id in corpus:
            raise ValueError(f"Duplicate corpus ID: {doc_id}")
        title, text = record.get("title", ""), record.get("text")
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError(f"Document {doc_id} must have string title/text fields")
        corpus[doc_id] = Document(doc_id, title, text)
    if not corpus:
        raise ValueError("Corpus is empty")

    qrels: dict[str, dict[str, int]] = {}
    reader = csv.reader(
        (line for _, line in _lines(root / "qrels" / f"{split}.tsv", qrels_digest)), delimiter="\t"
    )
    if next(reader, None) != ["query-id", "corpus-id", "score"]:
        raise ValueError("Qrels header must be query-id, corpus-id, score (tab-separated)")
    for row in reader:
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("Each qrels row must have exactly three fields")
        query_id, doc_id = identifier(row[0]), identifier(row[1])
        try:
            grade = int(row[2])
        except ValueError as error:
            raise ValueError("Relevance grades must be nonnegative integers") from error
        if grade < 0 or doc_id not in corpus:
            raise ValueError("Qrels contains a negative grade or unknown document")
        judgments = qrels.setdefault(query_id, {})
        if doc_id in judgments:
            raise ValueError("Duplicate query/document judgment")
        judgments[doc_id] = grade
    if not qrels:
        raise ValueError("Selected official qrels partition is empty")

    queries, seen = {}, set()
    for record in _jsonl(root / "queries.jsonl", queries_digest):
        query_id = identifier(record.get("_id"))
        if query_id in seen:
            raise ValueError(f"Duplicate query ID: {query_id}")
        seen.add(query_id)
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Query {query_id} must have nonempty string text")
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Query metadata must be an object when provided")
        family = metadata.get("family_id")
        if family is not None:
            family = identifier(family)
        if query_id in qrels:
            queries[query_id] = Query(query_id, text, family)
    if qrels.keys() - queries.keys():
        raise ValueError("Qrels references a missing query")
    dataset = Dataset(
        name,
        role,
        split,
        MappingProxyType(corpus),
        MappingProxyType(queries),
        MappingProxyType({q: MappingProxyType(rows) for q, rows in qrels.items()}),
        MappingProxyType(
            {
                "corpus.jsonl": corpus_digest.hexdigest(),
                "queries.jsonl": queries_digest.hexdigest(),
                f"qrels/{split}.tsv": qrels_digest.hexdigest(),
            }
        ),
    )
    if selection is not None:
        from atlasir.splits import apply_selection

        dataset = apply_selection(dataset, selection, role)
    return dataset
