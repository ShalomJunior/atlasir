"""Common metrics with fixed provider, query coverage and explicit tie/self-match rules."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Literal

import ir_measures

from atlasir.data import Dataset, identifier, require_development


@dataclass(frozen=True)
class ScoredDocument:
    doc_id: str
    score: float


def evaluate(
    dataset: Dataset,
    rankings: Mapping[str, Sequence[ScoredDocument]],
    *,
    purpose: Literal["development", "final"] = "development",
    self_matches: Literal["exclude", "keep"] = "exclude",
) -> dict[str, Any]:
    if not dataset.queries:
        raise ValueError("Evaluation requires a nonempty query partition")
    if purpose == "development":
        require_development(dataset)
    elif purpose != "final" or dataset.role != "final":
        raise ValueError("Final evaluation requires a dataset explicitly loaded with final role")
    if self_matches not in {"exclude", "keep"}:
        raise ValueError("Choose an explicit self-match policy")
    if rankings.keys() - dataset.queries.keys():
        raise ValueError("Rankings contain an unknown or wrong-partition query")

    ordered = {}
    excluded = 0
    for query_id in dataset.queries:
        seen, hits = set(), []
        for hit in rankings.get(query_id, []):
            doc_id = identifier(hit.doc_id)
            if doc_id in seen:
                raise ValueError("Duplicate document in a query ranking")
            seen.add(doc_id)
            if doc_id not in dataset.corpus:
                raise ValueError("Ranking references a document outside the corpus")
            if isinstance(hit.score, bool) or not isinstance(hit.score, (int, float)):
                raise ValueError("Ranking scores must be finite numbers")
            if not math.isfinite(hit.score):
                raise ValueError("Ranking scores must be finite numbers")
            if self_matches == "exclude" and query_id == doc_id:
                excluded += 1
                continue
            hits.append(hit)
        ordered[query_id] = sorted(hits, key=lambda hit: (-hit.score, hit.doc_id))

    # Synthetic descending scores encode our explicit total ordering, avoiding backend tie rules.
    provider_run = {
        query_id: {hit.doc_id: float(len(hits) - rank) for rank, hit in enumerate(hits)}
        for query_id, hits in ordered.items()
    }
    measures = {
        ir_measures.nDCG(dcg="log2") @ 10: "nDCG@10",
        ir_measures.Recall(rel=1) @ 100: "Recall@100",
        ir_measures.Recall(rel=1) @ 1000: "Recall@1000",
    }
    metric_names = [*measures.values(), "MRR@10"]
    per_query = {q: dict.fromkeys(metric_names, 0.0) for q in dataset.queries}
    qrels = {q: dict(rows) for q, rows in dataset.qrels.items()}
    provider = ir_measures.providers.registry["pytrec_eval"]
    for metric in provider.iter_calc(list(measures), qrels, provider_run):
        per_query[metric.query_id][measures[metric.measure]] = metric.value
    # The installed adapter accepts RR@10 but delegates to uncut recip_rank.
    # Explicitly truncate before uncut RR so rank 11 can never contribute to MRR@10.
    top_ten = {q: dict(list(hits.items())[:10]) for q, hits in provider_run.items()}
    for metric in provider.iter_calc([ir_measures.RR(rel=1)], qrels, top_ten):
        per_query[metric.query_id]["MRR@10"] = metric.value
    count = len(per_query)
    return {
        "schema_version": 1,
        "dataset": dataset.describe(),
        "purpose": purpose,
        "metrics": {
            metric: sum(row[metric] for row in per_query.values()) / count
            for metric in metric_names
        },
        "per_query": per_query,
        "coverage": {
            "expected_queries": count,
            "submitted_queries": len(rankings),
            "missing_queries": sorted(dataset.queries.keys() - rankings.keys()),
            "empty_rankings": sum(not hits for hits in ordered.values()),
            "no_positive_judgments": sum(
                not any(g > 0 for g in row.values()) for row in qrels.values()
            ),
            "self_matches_excluded": excluded,
        },
        "evaluator": {
            "ir-measures": version("ir-measures"),
            "provider": "pytrec_eval",
            "provider_distribution": "pytrec-eval-terrier",
            "provider_version": version("pytrec-eval-terrier"),
            "gain": "linear relevance grade",
            "discount": "log2(rank + 1)",
            "binary_relevance": "grade >= 1",
            "mrr_cutoff": "truncate to 10 before backend reciprocal rank",
            "unjudged": "nonrelevant",
            "zero_positive_queries": "zero, included in denominator",
            "tie_order": "score descending, original document ID ascending",
            "self_matches": self_matches,
        },
    }
