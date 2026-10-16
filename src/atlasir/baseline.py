"""Preserved development-only BM25 runs and independent saved-ranking evaluation."""

import gzip
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from atlasir.artifacts import RunWriter, verify_run
from atlasir.bm25 import BM25Retriever, resolved_bm25_config
from atlasir.data import load_beir, require_development
from atlasir.evaluation import ScoredDocument, evaluate


def read_config(path: Path) -> dict[str, Any]:
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise ValueError("Configuration keys must be unique strings")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        return resolved_bm25_config(yaml.load(path.read_text(), Loader=UniqueLoader))
    except yaml.YAMLError as error:
        raise ValueError("Invalid safe YAML configuration") from error


def run_baseline(
    *,
    data: Path,
    name: str,
    role: str,
    selection: Path | None,
    config: dict[str, Any],
    root: Path,
    source_root: Path,
    run_id: str | None = None,
    invocation: list[str] | None = None,
) -> dict[str, Any]:
    installed = Path(__file__).resolve().parent
    source = source_root.resolve() / "src" / "atlasir"
    if {p.relative_to(installed): p.read_bytes() for p in installed.rglob("*.py")} != {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*.py")
    }:
        raise ValueError("source root does not match the installed AtlasIR code")
    config = resolved_bm25_config(config)
    membership = json.loads(selection.read_bytes()) if selection else None
    dataset = load_beir(data, name=name, role=role, selection=membership)
    require_development(dataset)
    payloads = {
        "dataset.json": (json.dumps(dataset.describe(), sort_keys=True, indent=2) + "\n").encode(),
        "selection.json": (json.dumps(membership, sort_keys=True, indent=2) + "\n").encode(),
    }
    for original, saved in (
        ("corpus.jsonl", "data.corpus.jsonl"),
        ("queries.jsonl", "data.queries.jsonl"),
        (f"qrels/{dataset.official_split}.tsv", "data.qrels.tsv"),
    ):
        content = (data / original).read_bytes()
        if hashlib.sha256(content).hexdigest() != dataset.file_hashes[original]:
            raise ValueError("Input changed between loading and snapshotting")
        payloads[saved] = content
    with RunWriter(
        root, source_root, config, None, run_id, invocation or [], baseline_payloads=payloads
    ) as writer:
        start = time.perf_counter_ns()
        retriever = BM25Retriever(dataset.corpus, config)
        indexing_ms = (time.perf_counter_ns() - start) / 1e6
        queries = sorted(dataset.queries)
        for query_id in queries[:5]:
            retriever.search(dataset.queries[query_id].text, query_id=query_id)
        rankings, latency = {}, {}
        for query_id in queries:
            start = time.perf_counter_ns()
            rankings[query_id] = retriever.search(dataset.queries[query_id].text, query_id=query_id)
            latency[query_id] = (time.perf_counter_ns() - start) / 1e6
        evaluation = evaluate(dataset, rankings, self_matches=config["self_matches"])
        index, arrays = retriever.snapshot()
        writer.write_json("index.json", index)
        writer.write_bytes("index.npz", arrays)
        writer.write_json(
            "rankings.json",
            {
                q: [{"doc_id": hit.doc_id, "score": hit.score} for hit in hits]
                for q, hits in rankings.items()
            },
        )
        trec = "".join(
            f"{q} Q0 {hit.doc_id} {rank + 1} {len(hits) - rank} atlasir-bm25\n"
            for q, hits in rankings.items()
            for rank, hit in enumerate(hits)
        )
        writer.write_bytes("rankings.trec.gz", gzip.compress(trec.encode(), mtime=0))
        writer.write_json("per_query.json", evaluation.pop("per_query"))
        writer.write_json("metrics.json", evaluation)
        writer.write_json(
            "timing.json",
            {
                "scope": "CPU pilot; one timed pass after up to five warm-up queries",
                "includes": (
                    "tokenization, scoring, full sort, self-match filtering; excludes IO/evaluation"
                ),
                "query_order": queries,
                "warmup_queries": queries[:5],
                "indexing_ms": indexing_ms,
                "per_query_ms": latency,
                "p50_ms": float(np.percentile(list(latency.values()), 50)),
                "p95_ms": float(np.percentile(list(latency.values()), 95)),
                "trec_scores": "rank-preserving integers; original BM25 scores in rankings.json",
            },
        )
        writer.write_json(
            "results.json",
            {
                "scope": "exploratory-bm25-baseline",
                "metrics": evaluation["metrics"],
                "reportable": False,
            },
        )
        path = writer.seal()
    return {
        "status": "success",
        "run_id": writer.run_id,
        "path": str(path),
        "metrics": evaluation["metrics"],
        "reportable": False,
    }


def reevaluate(run: Path) -> dict[str, Any]:
    manifest = verify_run(run)
    if manifest["schema_version"] != 2:
        raise ValueError("Saved-ranking evaluation requires a baseline run")
    descriptor = json.loads((run / "dataset.json").read_bytes())
    selection = json.loads((run / "selection.json").read_bytes())
    config = json.loads((run / "config.resolved.json").read_bytes())
    with tempfile.TemporaryDirectory(prefix="atlasir-evaluate-") as temporary:
        data = Path(temporary)
        (data / "qrels").mkdir()
        for saved, original in (
            ("data.corpus.jsonl", "corpus.jsonl"),
            ("data.queries.jsonl", "queries.jsonl"),
            ("data.qrels.tsv", f"qrels/{descriptor['official_split']}.tsv"),
        ):
            (data / original).write_bytes((run / saved).read_bytes())
        dataset = load_beir(
            data, name=descriptor["name"], role=descriptor["role"], selection=selection
        )
        if dataset.describe() != descriptor:
            raise ValueError("Saved dataset descriptor does not match its raw inputs")
        rankings = {
            q: [ScoredDocument(**row) for row in rows]
            for q, rows in json.loads((run / "rankings.json").read_bytes()).items()
        }
        evaluation = evaluate(dataset, rankings, self_matches=config["self_matches"])
    per_query = evaluation.pop("per_query")
    if evaluation != json.loads((run / "metrics.json").read_bytes()) or per_query != json.loads(
        (run / "per_query.json").read_bytes()
    ):
        raise ValueError("Recalculated metrics or evaluator profile differ from the saved run")
    return {
        "status": "recalculated",
        "run_id": manifest["run_id"],
        "metrics": evaluation["metrics"],
    }
