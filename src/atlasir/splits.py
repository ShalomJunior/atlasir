"""Deterministic source-query partitions with duplicate/family grouping and frozen IDs."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from atlasir.data import Dataset, Role, normalized_query, stable_hash


def _groups(dataset: Dataset) -> list[list[str]]:
    parents = {q: q for q in dataset.queries}

    def find(query_id):
        while parents[query_id] != query_id:
            parents[query_id] = parents[parents[query_id]]
            query_id = parents[query_id]
        return query_id

    seen = {}
    for query_id, query in sorted(dataset.queries.items()):
        keys = [("text", normalized_query(query.text))]
        if query.family_id is not None:
            keys.append(("family", query.family_id))
        for key in keys:
            if key in seen:
                parents[find(query_id)] = find(seen[key])
            seen[key] = query_id
    grouped: dict[str, list[str]] = {}
    for query_id in sorted(parents):
        grouped.setdefault(find(query_id), []).append(query_id)
    return sorted(grouped.values())


def make_selection(
    dataset: Dataset,
    *,
    seed: int = 17,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    if dataset.role != "train" or dataset.official_split != "train" or dataset.selection_hash:
        raise ValueError("Create a selection only from an unpartitioned source training pool")
    if dataset.name not in {"scifact", "fixture"}:
        raise ValueError("NFCorpus uses official train/dev; do not invent a replacement split")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in the uint32 range")
    if type(validation_fraction) not in {float, int} or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    groups = _groups(dataset)
    if len(groups) < 2:
        raise ValueError("At least two independent query groups are required")
    target = max(1, round(len(dataset.queries) * validation_fraction))
    order = sorted(groups, key=lambda group: stable_hash([seed, group]))
    anchor = min(order, key=lambda group: (abs(len(group) - target), stable_hash([seed, group])))
    validation = set(anchor)
    for group in order:
        if group == anchor:
            continue
        size = len(validation) + len(group)
        if size < len(dataset.queries) and abs(size - target) < abs(len(validation) - target):
            validation.update(group)
    return {
        "schema_version": 1,
        "name": dataset.name,
        "official_split": "train",
        "file_hashes": dict(dataset.file_hashes),
        "seed": seed,
        "strategy": "sha256-group-order-v1",
        "normalization": "NFKC-casefold-whitespace-v1",
        "target_validation_fraction": validation_fraction,
        "actual_validation_fraction": len(validation) / len(dataset.queries),
        "groups": groups,
        "train_ids": sorted(dataset.queries.keys() - validation),
        "validation_ids": sorted(validation),
    }


def apply_selection(dataset: Dataset, manifest: Mapping[str, Any], role: Role) -> Dataset:
    if dataset.official_split != "train" or role not in {"train", "validation"}:
        raise ValueError("Selection is restricted to source train/validation queries")
    try:
        if (
            manifest["schema_version"] != 1
            or manifest["name"] != dataset.name
            or manifest["official_split"] != "train"
            or manifest["file_hashes"] != dict(dataset.file_hashes)
        ):
            raise ValueError("Selection does not match this exact dataset version")
        if (
            manifest["strategy"] != "sha256-group-order-v1"
            or manifest["normalization"] != "NFKC-casefold-whitespace-v1"
            or type(manifest["seed"]) is not int
            or not 0 <= manifest["seed"] < 2**32
        ):
            raise ValueError("Unsupported split strategy, normalization or seed")
        train, validation = manifest["train_ids"], manifest["validation_ids"]
        if not isinstance(train, list) or not isinstance(validation, list):
            raise ValueError("Selection IDs must be lists")
        train_set, validation_set = set(train), set(validation)
        if (
            not train_set
            or not validation_set
            or train_set & validation_set
            or len(train_set) != len(train)
            or len(validation_set) != len(validation)
            or train_set | validation_set != dataset.queries.keys()
        ):
            raise ValueError("Selection must cover the pool exactly with disjoint nonempty IDs")
        groups = _groups(dataset)
        if manifest["groups"] != groups or any(
            set(group) & train_set and set(group) & validation_set for group in groups
        ):
            raise ValueError("Selection crosses related query groups")
        expected = make_selection(
            replace(dataset, role="train"),
            seed=manifest["seed"],
            validation_fraction=manifest["target_validation_fraction"],
        )
        if dict(manifest) != expected:
            raise ValueError("Split membership/metadata does not reproduce from its saved recipe")
        chosen = train_set if role == "train" else validation_set
        return replace(
            dataset,
            role=role,
            queries=MappingProxyType({q: dataset.queries[q] for q in sorted(chosen)}),
            qrels=MappingProxyType({q: dataset.qrels[q] for q in sorted(chosen)}),
            selection_hash=stable_hash(dict(manifest)),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed split selection manifest") from error


def audit_disjoint(train: Dataset, validation: Dataset) -> dict[str, int]:
    """Fail on source-partition ID, normalized text or declared family overlap."""
    if train.name != validation.name or train.role != "train" or validation.role != "validation":
        raise ValueError("Audit train/validation partitions of the same dataset")
    texts = {normalized_query(q.text) for q in train.queries.values()}
    families = {q.family_id for q in train.queries.values() if q.family_id is not None}
    if (
        train.queries.keys() & validation.queries.keys()
        or texts & {normalized_query(q.text) for q in validation.queries.values()}
        or families & {q.family_id for q in validation.queries.values() if q.family_id is not None}
    ):
        raise ValueError("Training and validation queries overlap by ID, text or family")
    if any(
        train.file_hashes[name] != validation.file_hashes[name]
        for name in ("corpus.jsonl", "queries.jsonl")
    ):
        raise ValueError("Train/validation partitions refer to different source file versions")
    return {"train_queries": len(train.queries), "validation_queries": len(validation.queries)}
