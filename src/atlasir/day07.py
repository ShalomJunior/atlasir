"""Source-training embedding pilot: portable interruption recovery and exact indexing."""

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from atlasir.artifacts import RunWriter, export_run
from atlasir.checkpoint_io import (
    MODEL_FILES,
    export_checkpoint,
    materialize_model,
    restore_checkpoint,
)
from atlasir.data import Dataset, require_development
from atlasir.encoding import E5Encoder, resolved_encoder_config
from atlasir.objects import BlobStore, export_objects, file_hash, import_objects
from atlasir.shards import EmbeddingJob, encoder_profile
from atlasir.vector_index import ExactVectorIndex


def resolved_shard_pilot_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    defaults = {
        "schema_version": 4,
        "family": "E02",
        "protocol": "day07-shards-index-pilot-v1",
        "encoder": resolved_encoder_config(),
        "document_limit": 256,
        "query_limit": 16,
        "shard_size": 64,
        "sample_seed": 17,
        "faiss_threads": 1,
    }
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or overrides.keys() - defaults.keys():
        raise ValueError("Unknown shard-pilot configuration field")
    result = defaults | overrides
    result["encoder"] = resolved_encoder_config(result["encoder"])
    for key in ("schema_version", "family", "protocol", "faiss_threads"):
        if type(result[key]) is not type(defaults[key]) or result[key] != defaults[key]:
            raise ValueError(f"Unsupported shard-pilot setting: {key}")
    for key, limit in [("document_limit", 1024), ("query_limit", 128), ("shard_size", 256)]:
        if type(result[key]) is not int or not 1 <= result[key] <= limit:
            raise ValueError(f"Invalid shard-pilot {key}")
    if type(result["sample_seed"]) is not int or not 0 <= result["sample_seed"] < 2**32:
        raise ValueError("Invalid pilot sampling seed")
    if result["shard_size"] >= result["document_limit"]:
        raise ValueError("Pilot needs at least two passage shards to test interruption")
    return result


def prepare_pilot_inputs(dataset: Dataset, config: dict[str, Any]) -> dict[str, Any]:
    require_development(dataset)
    if dataset.role != "train":
        raise ValueError("Day 7 samples only source training data")
    config = resolved_shard_pilot_config(config)

    def select(ids, limit, kind):
        return sorted(
            sorted(
                ids,
                key=lambda key: hashlib.sha256(
                    f"day07:{config['sample_seed']}:{kind}:{key}".encode()
                ).hexdigest(),
            )[:limit]
        )

    documents = select(dataset.corpus, config["document_limit"], "passage")
    queries = select(dataset.queries, config["query_limit"], "query")
    if len(documents) <= config["shard_size"]:
        raise ValueError("Need more source documents than one shard")
    return {
        "schema_version": 1,
        "scope": "source-training-encoding-pilot",
        "source": dataset.describe(),
        "sampling": {
            "recipe": "sha256(day07:seed:kind:ID), take smallest hashes, encode by string ID",
            "seed": config["sample_seed"],
            "uses_relevance_labels": False,
        },
        "documents": [{"id": key, "text": dataset.corpus[key].contents} for key in documents],
        "queries": [{"id": key, "text": dataset.queries[key].text} for key in queries],
        "attribution": "Wadden et al., SciFact (EMNLP 2020); Lo et al., S2ORC (ACL 2020); "
        "Thakur et al., BEIR (NeurIPS 2021). Claims/evidence CC BY 4.0; "
        "S2ORC abstracts ODC-By 1.0. https://github.com/allenai/scifact/blob/master/LICENSE.md",
    }


def _inputs(content: bytes, config: dict[str, Any]) -> dict[str, Any]:
    inputs = json.loads(content)
    if inputs["schema_version"] != 1 or inputs["scope"] != "source-training-encoding-pilot":
        raise ValueError("Wrong Day 7 input scope")
    source = inputs["source"]
    if (
        source["name"] not in ("scifact", "fixture")
        or source["role"] != "train"
        or source["official_split"] != "train"
    ):
        raise ValueError("Day 7 requires source training inputs")
    if source["name"] == "scifact" and not source["selection_hash"]:
        raise ValueError("Source training selection must be frozen")
    for kind, limit in [("documents", "document_limit"), ("queries", "query_limit")]:
        rows = inputs[kind]
        if len(rows) != min(config[limit], source[kind]):
            raise ValueError("Pilot sample size differs from configuration/source descriptor")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate pilot row IDs")
    if inputs["sampling"]["seed"] != config["sample_seed"]:
        raise ValueError("Sampling seed mismatch")
    return inputs


def run_shard_pilot(
    *,
    inputs_path: Path,
    config: dict[str, Any],
    root: Path,
    source_root: Path,
    local_files_only: bool = False,
) -> dict[str, Any]:
    config = resolved_shard_pilot_config(config)
    content = inputs_path.read_bytes()
    inputs = _inputs(content, config)
    installed = Path(__file__).resolve().parent
    source = source_root.resolve() / "src/atlasir"
    if {p.relative_to(installed): p.read_bytes() for p in installed.rglob("*.py")} != {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*.py")
    }:
        raise ValueError("source root does not match the installed AtlasIR code")
    root = root.resolve()
    with RunWriter(
        root,
        source_root,
        config,
        None,
        invocation=["atlasir", "shard-pilot", "--inputs", str(inputs_path)],
        shard_payloads={"pilot-inputs.json": content},
    ) as writer:
        import faiss
        from huggingface_hub import snapshot_download

        faiss.omp_set_num_threads(config["faiss_threads"])
        encoder = E5Encoder(config["encoder"], local_files_only=local_files_only)
        snapshot = Path(
            snapshot_download(
                config["encoder"]["model"],
                revision=config["encoder"]["model_revision"],
                local_files_only=True,
            )
        )
        original_store = BlobStore(root / "artifacts/objects")
        model_files = {
            name: original_store.put_file(snapshot / name)
            for name in sorted(MODEL_FILES)
            if (snapshot / name).is_file()
        }
        profile = encoder_profile(encoder, model_files)
        workspace = root / ".scratch/day07-jobs" / writer.run_id
        passages = {row["id"]: row["text"] for row in inputs["documents"]}
        queries = {row["id"]: row["text"] for row in inputs["queries"]}
        original = EmbeddingJob(
            workspace / "original-passages",
            original_store,
            passages,
            kind="passage",
            profile=profile,
            shard_size=config["shard_size"],
        )
        first = original.run(encoder, stop_after_new_shards=1)
        if first["status"] != "paused":
            raise ValueError("Controlled interruption did not leave an incomplete job")
        first_shard = file_hash(original.directory / "shards/00000000.json")
        partial = export_checkpoint(original, workspace / "partial.checkpoint.zip")
        partial_objects = export_objects(
            original_store, original.object_ids(), workspace / "partial-objects.zip"
        )
        # Resume using only portable checkpoint/object exports in an empty store.
        recovered_store = BlobStore(workspace / "recovered-objects")
        import_objects(partial_objects, recovered_store)
        resumed = restore_checkpoint(partial, workspace / "resumed-passages", recovered_store)
        offline = materialize_model(recovered_store, model_files, workspace / "offline-model")
        del encoder
        encoder = E5Encoder(config["encoder"], model_directory=offline, expected_files=model_files)
        if encoder_profile(encoder, model_files) != profile:
            raise ValueError("Restored model execution profile differs")
        second = resumed.run(encoder)
        if (
            second["reused_shards"] != 1
            or file_hash(resumed.directory / "shards/00000000.json") != first_shard
        ):
            raise ValueError("Resume did not retain the original verified shard")
        clean = EmbeddingJob(
            workspace / "clean-passages",
            recovered_store,
            passages,
            kind="passage",
            profile=profile,
            shard_size=config["shard_size"],
        )
        clean.run(encoder)
        ids, vectors = resumed.arrays()
        clean_ids, clean_vectors = clean.arrays()
        if ids != clean_ids or not np.allclose(vectors, clean_vectors, rtol=1e-5, atol=1e-6):
            raise ValueError("Resumed encoding differs from uninterrupted encoding")
        query_job = EmbeddingJob(
            workspace / "queries",
            recovered_store,
            queries,
            kind="query",
            profile=profile,
            shard_size=config["shard_size"],
        )
        query_job.run(encoder)
        query_ids, query_vectors = query_job.arrays()
        start = time.perf_counter()
        index = ExactVectorIndex(ids, vectors)
        index_seconds = time.perf_counter() - start
        index_meta, index_bytes = index.snapshot()
        index_object = recovered_store.put_bytes(index_bytes)
        loaded = ExactVectorIndex.from_snapshot(index_meta, recovered_store.read(index_object))
        rankings = {}
        search_seconds = []
        for number, key in enumerate(query_ids):
            vector = query_vectors[number : number + 1]
            start = time.perf_counter()
            hits = index.search(vector, top_k=min(10, len(ids)), query_id=key)
            search_seconds.append(time.perf_counter() - start)
            if loaded.search(vector, top_k=min(10, len(ids)), query_id=key) != hits:
                raise ValueError("Saved FAISS index did not reproduce rankings")
            rankings[key] = [{"doc_id": hit.doc_id, "score": hit.score} for hit in hits]
        records = resumed.verify(require_complete=True)
        elapsed = sum(record["encoding_seconds"] for record in records)
        projected = elapsed * inputs["source"]["documents"] / len(ids)
        objects = sorted(
            set(resumed.object_ids() + clean.object_ids() + query_job.object_ids() + [index_object])
        )
        # Publish verified recovered objects to the coordinator store for reuse.
        for digest in objects:
            if original_store.put_file(recovered_store.verify(digest)) != digest:
                raise ValueError("Object ingestion identity mismatch")
        writer.write_json(
            "objects.json",
            {
                "schema_version": 1,
                "objects": objects,
                "index_object": index_object,
                "model_files": model_files,
                "passage_job": resumed.job_id,
                "query_job": query_job.job_id,
            },
        )
        writer.write_json("index.json", index_meta)
        writer.write_json("rankings.json", rankings)
        writer.write_bytes("partial.checkpoint.zip", partial.read_bytes())
        for name, job in [("passages", resumed), ("queries", query_job), ("clean", clean)]:
            archive = export_checkpoint(job, workspace / f"{name}.checkpoint.zip")
            writer.write_bytes(f"{name}.checkpoint.zip", archive.read_bytes())
        writer.write_json(
            "resume.json",
            {
                "first_attempt": first,
                "resumed_attempt": second,
                "fresh_object_store": True,
                "offline_model_restored_from_objects": True,
                "first_shard_sha256": first_shard,
                "first_shard_bytes_preserved": True,
                "clean_vs_resumed_max_abs_difference": float(
                    np.max(np.abs(vectors - clean_vectors))
                ),
                "clean_vs_resumed_exact": bool(np.array_equal(vectors, clean_vectors)),
                "tolerance": {"rtol": 1e-5, "atol": 1e-6},
                "saved_index_rankings": "exact match",
            },
        )
        lengths = [
            length
            for record in records
            for length in record["encoding"]["token_lengths_before_truncation"]
        ]
        forecast = {
            "scope": "single small source pilot; projection, not final latency or quota guarantee",
            "sample_documents": len(ids),
            "source_documents": inputs["source"]["documents"],
            "encoding_seconds_by_shard": [record["encoding_seconds"] for record in records],
            "sample_encoding_seconds": elapsed,
            "projected_source_encoding_seconds": projected,
            "projection_with_2x_margin_seconds": projected * 2,
            "excludes": "model download/load, checksums, checkpoint IO, repeats and export",
            "full_source_float32_embedding_bytes": inputs["source"]["documents"] * 384 * 4,
            "approx_full_source_flat_index_vector_bytes": inputs["source"]["documents"] * 384 * 4,
            "model_object_bytes": sum(
                recovered_store.verify(digest).stat().st_size
                for digest in set(model_files.values())
            ),
            "pilot_index_bytes": len(index_bytes),
            "index_build_seconds": index_seconds,
            "query_search_seconds": search_seconds,
            "query_batch_size": 1,
            "document_token_lengths": {
                "mean": float(np.mean(lengths)),
                "max": max(lengths),
                "truncated": sum(n > 512 for n in lengths),
            },
        }
        writer.write_json("forecast.json", forecast)
        writer.write_json(
            "results.json",
            {
                "scope": "source-training-shard/index pilot; no relevance metrics",
                "status": "passed",
                "reportable": False,
                "documents": len(ids),
                "queries": len(query_ids),
                "resume": "passed",
                "index_reload": "passed",
            },
        )
        run = writer.seal()
    session = root / "session-exports" / writer.run_id
    session.mkdir(parents=True)
    export_run(run, session / f"{writer.run_id}.bundle")
    export_objects(original_store, objects, session / "embedding-objects.zip")
    shutil.copyfile(run / "pilot-inputs.json", session / "pilot-inputs-and-attribution.json")
    receipt = {
        "schema_version": 1,
        "run_id": writer.run_id,
        "objects": len(objects),
        "objects_zip_sha256": file_hash(session / "embedding-objects.zip"),
        "objects_zip_bytes": (session / "embedding-objects.zip").stat().st_size,
        "weights_in_object_export": True,
    }
    with (session / "session.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return {
        "status": "success",
        "run_id": writer.run_id,
        "path": str(run),
        "session_exports": str(session),
        "scope": "source-training infrastructure pilot",
        "reportable": False,
    }


def verify_shard_pilot(run: Path, store: BlobStore) -> dict[str, Any]:
    """Check saved checkpoints/vectors/index with no torch or model execution."""
    import tempfile

    from atlasir.artifacts import verify_run

    manifest = verify_run(run)
    if manifest["schema_version"] != 4:
        raise ValueError("Expected a Day 7 shard/index pilot")
    config = resolved_shard_pilot_config(json.loads((run / "config.resolved.json").read_bytes()))
    inputs = _inputs((run / "pilot-inputs.json").read_bytes(), config)
    objects = json.loads((run / "objects.json").read_bytes())
    for digest in objects["objects"]:
        store.verify(digest)
    resume = json.loads((run / "resume.json").read_bytes())
    with tempfile.TemporaryDirectory(prefix="atlasir-check-shards-") as temporary:
        root = Path(temporary)
        partial = restore_checkpoint(run / "partial.checkpoint.zip", root / "partial", store)
        jobs = {
            kind: restore_checkpoint(run / f"{kind}.checkpoint.zip", root / kind, store)
            for kind in ("passages", "queries", "clean")
        }
        for kind in ("passages", "queries"):
            job = jobs[kind]
            job.verify(require_complete=True)
            rows = {
                row["id"]: row["text"]
                for row in inputs["documents" if kind == "passages" else "queries"]
            }
            if dict(job.rows) != rows or job.profile["config"] != config["encoder"]:
                raise ValueError("Saved checkpoint inputs/config differ from the pilot")
        passage_ids, vectors = jobs["passages"].arrays()
        query_ids, queries = jobs["queries"].arrays()
        clean_ids, clean = jobs["clean"].arrays()
        if passage_ids != clean_ids or not np.allclose(vectors, clean, rtol=1e-5, atol=1e-6):
            raise ValueError("Saved clean/resumed vectors differ")
        observed = float(np.max(np.abs(vectors - clean)))
        if (
            observed != resume["clean_vs_resumed_max_abs_difference"]
            or bool(np.array_equal(vectors, clean)) != resume["clean_vs_resumed_exact"]
        ):
            raise ValueError("Saved resume summary differs from vector evidence")
        if partial.job_id != jobs["passages"].job_id or len(partial.verify()) != 1:
            raise ValueError("Wrong partial checkpoint identity")
        original_first = partial.directory / "shards/00000000.json"
        resumed_first = jobs["passages"].directory / "shards/00000000.json"
        if (
            file_hash(original_first) != resume["first_shard_sha256"]
            or original_first.read_bytes() != resumed_first.read_bytes()
        ):
            raise ValueError("Resume replaced its original shard")
        used = sorted(
            set(
                jobs["passages"].object_ids()
                + jobs["clean"].object_ids()
                + jobs["queries"].object_ids()
                + [objects["index_object"]]
            )
        )
        if (
            used != objects["objects"]
            or jobs["passages"].profile["model_files"] != objects["model_files"]
        ):
            raise ValueError("Pilot object dependency inventory differs")
        metadata = json.loads((run / "index.json").read_bytes())
        index = ExactVectorIndex.from_snapshot(metadata, store.read(objects["index_object"]))
        if index.doc_ids != passage_ids or not np.array_equal(
            index.index.reconstruct_n(0, len(passage_ids)), vectors
        ):
            raise ValueError("FAISS row mapping/vectors differ from passage checkpoint")
        saved = json.loads((run / "rankings.json").read_bytes())
        actual = {
            key: [
                {"doc_id": hit.doc_id, "score": hit.score}
                for hit in index.search(
                    queries[number : number + 1], top_k=min(10, len(passage_ids)), query_id=key
                )
            ]
            for number, key in enumerate(query_ids)
        }
        if saved.keys() != actual.keys() or any(
            [hit["doc_id"] for hit in saved[key]] != [hit["doc_id"] for hit in actual[key]]
            for key in actual
        ):
            raise ValueError("Restored index ranked document IDs differ")
        tolerance = metadata.get("cross_machine_score_tolerance", {"rtol": 0, "atol": 0})
        for key in actual:
            if not np.allclose(
                [hit["score"] for hit in saved[key]],
                [hit["score"] for hit in actual[key]],
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
            ):
                raise ValueError("Restored index scores exceed their declared tolerance")
    return {
        "status": "verified",
        "run_id": manifest["run_id"],
        "documents": len(passage_ids),
        "queries": len(query_ids),
        "resume": "passed",
        "index_rankings": "exact match"
        if actual == saved
        else "IDs exact; scores within recorded tolerance",
        "objects": len(used),
        "model_files_in_objects": True,
    }
