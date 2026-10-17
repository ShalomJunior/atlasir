"""Full source-only fixed retrieval comparison and descriptive routing opportunity."""

import gzip
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np

from atlasir.artifacts import RunWriter, export_run, verify_run
from atlasir.bm25 import BM25Retriever, resolved_bm25_config
from atlasir.checkpoint_io import export_checkpoint
from atlasir.data import load_beir, require_development, stable_hash
from atlasir.encoding import E5Encoder, resolved_encoder_config
from atlasir.evaluation import ScoredDocument, evaluate
from atlasir.objects import BlobStore, export_objects
from atlasir.shards import EmbeddingJob
from atlasir.vector_index import ExactVectorIndex

METHODS = ("bm25", "dense", "rrf")
INPUT_FILES = {
    "dataset.json",
    "selection.json",
    "data.corpus.jsonl",
    "data.queries.jsonl",
    "data.qrels.tsv",
}


def resolved_comparison_config(overrides=None):
    defaults = {
        "schema_version": 5,
        "family": "E03",
        "protocol": "source-fixed-comparison-v1",
        "encoder": resolved_encoder_config(),
        "bm25": resolved_bm25_config(),
        "rrf_k": 60,
        "top_k": 1000,
        "shard_size": 64,
        "faiss_threads": 1,
    }
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or overrides.keys() - defaults.keys():
        raise ValueError("Unknown comparison setting")
    config = defaults | overrides
    config["encoder"] = resolved_encoder_config(config["encoder"])
    config["bm25"] = resolved_bm25_config(config["bm25"])
    for key in defaults.keys() - {"encoder", "bm25"}:
        if type(config[key]) is not type(defaults[key]) or config[key] != defaults[key]:
            raise ValueError(f"Unsupported comparison setting: {key}")
    if config["bm25"] != defaults["bm25"]:
        raise ValueError("This comparison freezes the original BM25 reference")
    return config


def rrf(lexical, dense, *, k=60, top_k=1000):
    """Equal-weight RRF on each base's top 1000; missing documents contribute zero."""
    scores = {}
    for ranking in (lexical, dense):
        for rank, hit in enumerate(ranking, 1):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(
        [ScoredDocument(key, score) for key, score in scores.items()],
        key=lambda hit: (-hit.score, hit.doc_id),
    )[:top_k]


def opportunity(evaluations):
    """Label-dependent upper bound only; never an executable routing policy."""
    ids = sorted(evaluations["bm25"]["per_query"])
    best_fixed = max(METHODS, key=lambda m: evaluations[m]["metrics"]["nDCG@10"])
    rows = []
    for key in ids:
        values = {m: evaluations[m]["per_query"][key]["nDCG@10"] for m in METHODS}
        best = max(values.values())
        rows.append(
            {
                "query_id": key,
                "ndcg10": values,
                "best_methods": [m for m in METHODS if abs(values[m] - best) <= 1e-12],
                "oracle_ndcg10": best,
                "gain_over_best_fixed": best - values[best_fixed],
            }
        )
    oracle = sum(row["oracle_ndcg10"] for row in rows) / len(rows)
    fixed = evaluations[best_fixed]["metrics"]["nDCG@10"]
    return {
        "scope": "descriptive relevance-label oracle; no cost constraint or deployable policy",
        "best_fixed_method_on_this_partition": best_fixed,
        "best_fixed_ndcg10": fixed,
        "oracle_ndcg10": oracle,
        "oracle_gap": oracle - fixed,
        "queries_with_strict_gain_over_best_fixed": sum(
            row["gain_over_best_fixed"] > 1e-12 for row in rows
        ),
        "unique_wins": {m: sum(row["best_methods"] == [m] for row in rows) for m in METHODS},
        "tied_queries": sum(len(row["best_methods"]) > 1 for row in rows),
        "per_query": rows,
    }


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _datasets(data, name, selection):
    membership = json.loads(selection.read_bytes()) if selection else None
    datasets = {
        role: load_beir(data, name=name, role=role, selection=membership)
        for role in ("train", "validation")
    }
    for dataset in datasets.values():
        require_development(dataset)
    if datasets["train"].queries.keys() & datasets["validation"].queries.keys():
        raise ValueError("Training and validation queries overlap")
    # This version deliberately supports SciFact's single frozen training-pool input.
    if name not in ("scifact", "fixture") or membership is None:
        raise ValueError("This session requires a frozen SciFact-style training-pool selection")
    return membership, datasets


def run_comparison(*, data, selection, name, config, root, source_root, encoder=None):
    import faiss

    config = resolved_comparison_config(config)
    installed = Path(__file__).resolve().parent
    source = source_root.resolve() / "src/atlasir"
    if {p.relative_to(installed): p.read_bytes() for p in installed.rglob("*.py")} != {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*.py")
    }:
        raise ValueError("Source root differs from installed code")
    membership, datasets = _datasets(data, name, selection)
    train = datasets["train"]
    payloads = {
        "dataset.json": _json({role: ds.describe() for role, ds in datasets.items()}),
        "selection.json": _json(membership),
    }
    for original, saved in (
        ("corpus.jsonl", "data.corpus.jsonl"),
        ("queries.jsonl", "data.queries.jsonl"),
        ("qrels/train.tsv", "data.qrels.tsv"),
    ):
        content = (data / original).read_bytes()
        if hashlib.sha256(content).hexdigest() != train.file_hashes[original]:
            raise ValueError("Input changed during snapshot")
        payloads[saved] = content
    faiss.omp_set_num_threads(config["faiss_threads"])
    root = root.resolve()
    with RunWriter(
        root,
        source_root,
        config,
        None,
        invocation=["scripts/run_comparison.py"],
        comparison_payloads=payloads,
    ) as writer:
        print(f"Run {writer.run_id}: encoding full corpus", flush=True)
        encoder = encoder if encoder is not None else E5Encoder(config["encoder"])
        if encoder.config != config["encoder"]:
            raise ValueError("Encoder config mismatch")
        profile = encoder.checkpoint_profile()
        store = BlobStore(root / "artifacts/objects")
        # Encoder reports hashes of the actual loaded snapshot, not a new download.
        for filename, digest in profile["model_files"].items():
            if store.put_file(encoder._checkpoint_model_directory / filename) != digest:
                raise ValueError("Loaded model changed during snapshot")
        queries = {key: query for ds in datasets.values() for key, query in ds.queries.items()}
        workspace = root / ".scratch/comparison-jobs"
        jobs = {}
        attempts = {}
        for kind, rows, size in (
            ("passage", {k: d.contents for k, d in train.corpus.items()}, config["shard_size"]),
            ("query", {k: q.text for k, q in queries.items()}, 1),
        ):
            # Content/runtime identity, independent of run ID: retry reuses verified shards.
            identity = stable_hash({"kind": kind, "rows": rows, "profile": profile, "size": size})
            job = EmbeddingJob(
                workspace / identity, store, rows, kind=kind, profile=profile, shard_size=size
            )
            if kind == "passage" and not job.verify():
                job.run(encoder, stop_after_new_shards=1)
            attempts[kind] = job.run(encoder)
            jobs[kind] = job
            print(f"{kind}: {len(rows)} rows complete", flush=True)
        doc_ids, vectors = jobs["passage"].arrays()
        query_ids, query_vectors = jobs["query"].arrays()
        start = time.perf_counter()
        dense = ExactVectorIndex(doc_ids, vectors)
        dense_build = time.perf_counter() - start
        index_meta, index_bytes = dense.snapshot()
        index_object = store.put_bytes(index_bytes)
        loaded = ExactVectorIndex.from_snapshot(index_meta, store.read(index_object))
        start = time.perf_counter()
        lexical = BM25Retriever(train.corpus, config["bm25"])
        lexical_build = time.perf_counter() - start
        lexical_meta, lexical_arrays = lexical.snapshot()
        writer.write_json("bm25-index.json", lexical_meta)
        writer.write_bytes("bm25-index.npz", lexical_arrays)
        writer.write_json("dense-index.json", index_meta)
        rankings = {method: {} for method in METHODS}
        timings = {}
        for number in range(min(5, len(query_ids))):
            key = query_ids[number]
            lexical.search(queries[key].text, query_id=key)
            dense.search(query_vectors[number : number + 1], query_id=key)
        for number, key in enumerate(query_ids):
            start = time.perf_counter()
            bm = lexical.search(queries[key].text, query_id=key)
            bm_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            de = dense.search(query_vectors[number : number + 1], query_id=key)
            dense_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            hybrid = rrf(bm, de, k=config["rrf_k"], top_k=config["top_k"])
            fusion_ms = (time.perf_counter() - start) * 1000
            if (
                number == 0
                and loaded.search(query_vectors[number : number + 1], query_id=key) != de
            ):
                raise ValueError("FAISS reload differs")
            for method, hits in zip(METHODS, (bm, de, hybrid), strict=True):
                rankings[method][key] = hits
            timings[key] = {
                "bm25_search_ms": bm_ms,
                "dense_search_ms": dense_ms,
                "rrf_fusion_ms": fusion_ms,
            }
        evaluations = {}
        for role, dataset in datasets.items():
            evaluations[role] = {}
            for method in METHODS:
                subset = {key: rankings[method][key] for key in dataset.queries}
                evaluations[role][method] = evaluate(dataset, subset)
                writer.write_bytes(
                    f"{role}-{method}-rankings.json.gz",
                    gzip.compress(
                        _json({q: [vars(h) for h in hits] for q, hits in subset.items()}), mtime=0
                    ),
                )
        writer.write_json("evaluation.json", evaluations)
        analysis = {role: opportunity(values) for role, values in evaluations.items()}
        writer.write_json("opportunity.json", analysis)
        writer.write_json(
            "timing.json",
            {
                "scope": "exploratory isolated stages; not service latency or router cost",
                "query_encoding": "batch size 1; raw times in query checkpoint shards",
                "search": "one CPU pass after up to five warmups; precomputed query vectors",
                "excludes": "model load, profile hashing, checkpoint IO, export, router, reranking",
                "per_query_ms": timings,
                "bm25_index_seconds": lexical_build,
                "dense_index_seconds": dense_build,
                "attempts": attempts,
            },
        )
        dependencies = sorted(
            set(jobs["passage"].object_ids() + jobs["query"].object_ids() + [index_object])
        )
        writer.write_json(
            "objects.json",
            {
                "objects": dependencies,
                "index_object": index_object,
                "encoder_profile": profile,
            },
        )
        for kind, job in jobs.items():
            path = export_checkpoint(job, writer.path / f"{kind}.checkpoint.zip")
            assert path.is_file()
        buffer = io.BytesIO()
        np.savez_compressed(buffer, ids=np.array(query_ids), vectors=query_vectors)
        writer.write_bytes("query-vectors.npz", buffer.getvalue())
        summary = {
            "scope": f"exploratory {train.name} comparison; no final tests or adaptive claims",
            "reportable": False,
            "metrics": {
                role: {m: e["metrics"] for m, e in methods.items()}
                for role, methods in evaluations.items()
            },
            "next": "fixed reranker action outcomes, then confidence/random/learned policies",
        }
        writer.write_json("results.json", summary)
        run = writer.seal()
    session = root / "session-exports" / writer.run_id
    session.mkdir(parents=True)
    export_run(run, session / f"{writer.run_id}.bundle")
    export_objects(store, dependencies, session / "embedding-objects.zip")
    (session / "results.json").write_bytes(_json(summary))
    (session / "opportunity.json").write_bytes(_json(analysis))
    report = [
        f"# {train.name} fixed retrieval comparison",
        "",
        "Exploratory source results; no final tests or adaptive-gain claim.",
        "",
        "| Partition | Method | nDCG@10 | Recall@100 | Recall@1000 | MRR@10 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for role, methods in summary["metrics"].items():
        for method, metrics in methods.items():
            values = " | ".join(
                f"{metrics[key]:.6f}" for key in ("nDCG@10", "Recall@100", "Recall@1000", "MRR@10")
            )
            report.append(f"| {role} | {method} | {values} |")
    report += ["", "## Descriptive opportunity", ""]
    for role, observed in analysis.items():
        report.append(
            f"{role}: best fixed method {observed['best_fixed_method_on_this_partition']}; "
            f"label-oracle nDCG gap {observed['oracle_gap']:.6f}; "
            f"{observed['queries_with_strict_gain_over_best_fixed']} queries with strict gain."
        )
        report.append("")
    report += [
        "Oracle uses relevance labels and ignores cost; it is not a routing policy.",
        "Timing is isolated-stage exploratory data, not end-to-end service latency.",
        "Next: fixed reranker action matrix and simple selective-reranking baselines.",
    ]
    (session / "comparison.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return {"run_id": writer.run_id, "path": str(run), "session_exports": str(session)}


def reevaluate_comparison(run):
    """Recompute all quality tables from sealed raw inputs and saved rankings; no torch."""
    import tempfile

    manifest = verify_run(run)
    if manifest["schema_version"] != 5:
        raise ValueError("Expected a source comparison")
    saved = json.loads((run / "evaluation.json").read_bytes())
    descriptors = json.loads((run / "dataset.json").read_bytes())
    with tempfile.TemporaryDirectory(prefix="atlasir-comparison-") as temporary:
        data = Path(temporary)
        (data / "qrels").mkdir()
        for source, target in (
            ("data.corpus.jsonl", "corpus.jsonl"),
            ("data.queries.jsonl", "queries.jsonl"),
            ("data.qrels.tsv", "qrels/train.tsv"),
        ):
            (data / target).write_bytes((run / source).read_bytes())
        _, datasets = _datasets(data, descriptors["train"]["name"], run / "selection.json")
        observed = {}
        for role, dataset in datasets.items():
            if dataset.describe() != descriptors[role]:
                raise ValueError("Saved dataset identity differs")
            observed[role] = {}
            for method in METHODS:
                rows = json.loads(
                    gzip.decompress((run / f"{role}-{method}-rankings.json.gz").read_bytes())
                )
                observed[role][method] = evaluate(
                    dataset,
                    {q: [ScoredDocument(**hit) for hit in hits] for q, hits in rows.items()},
                )
        if observed != saved or {
            role: opportunity(values) for role, values in observed.items()
        } != json.loads((run / "opportunity.json").read_bytes()):
            raise ValueError("Recalculated results differ")
    return {"status": "recalculated", "run_id": manifest["run_id"]}
