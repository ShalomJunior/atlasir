"""Preserve a small synthetic E5 compatibility pilot; no benchmark quality claims."""

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from atlasir.artifacts import RunWriter
from atlasir.encoding import E5Encoder, resolved_encoder_config


def pilot_inputs(content: bytes) -> dict[str, Any]:
    inputs = json.loads(content)
    if not isinstance(inputs, dict) or set(inputs) != {
        "schema_version",
        "scope",
        "queries",
        "passages",
    }:
        raise ValueError("Invalid encoding pilot inputs")
    if inputs["schema_version"] != 1 or inputs["scope"] != "synthetic-encoding-check":
        raise ValueError("Day 6 pilot accepts only the declared synthetic fixture")
    for kind in ("queries", "passages"):
        rows = inputs[kind]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
            raise ValueError("Pilot requires 1–32 rows per kind")
        ids = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "text"}:
                raise ValueError("Pilot rows require id and text")
            from atlasir.data import identifier

            identifier(row["id"])
            if row["id"] in ids or not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("Pilot rows need unique IDs and nonblank text")
            if len(row["text"]) > 100000:
                raise ValueError("Pilot text exceeds the small-run bound")
            ids.add(row["id"])
    return inputs


def run_encoding_pilot(
    *,
    inputs_path: Path,
    config: dict[str, Any],
    root: Path,
    source_root: Path,
    run_id: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    installed = Path(__file__).resolve().parent
    source = source_root / "src/atlasir"
    if {p.relative_to(installed): p.read_bytes() for p in installed.rglob("*.py")} != {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*.py")
    }:
        raise ValueError("source root does not match the installed AtlasIR code")
    content = inputs_path.read_bytes()
    inputs = pilot_inputs(content)
    config = resolved_encoder_config(config)
    with RunWriter(
        root,
        source_root,
        config,
        None,
        run_id,
        ["atlasir", "encode-pilot", "--inputs", str(inputs_path)],
        encoder_payloads={"pilot-inputs.json": content},
    ) as writer:
        start = time.perf_counter()
        encoder = E5Encoder(config, local_files_only=local_files_only)
        load_seconds = time.perf_counter() - start
        import torch
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                config["model"], revision=config["model_revision"], local_files_only=True
            )
        )
        files = {}
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "README.md",
        ):
            path = snapshot / name
            if path.is_file():
                files[name] = {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
        if not {"config.json", "model.safetensors", "tokenizer_config.json"} <= files.keys():
            raise ValueError("Missing cached model identity files")
        writer.write_json(
            "model-files.json",
            {
                "model": config["model"],
                "revision": config["model_revision"],
                "files": files,
                "license": "MIT",
                "attribution": "Wang et al., Text Embeddings by Weakly-Supervised "
                "Contrastive Pre-training (2022)",
                "source": "https://huggingface.co/intfloat/e5-small-v2",
                "weights_in_bundle": False,
            },
        )
        batches, timings, deltas = {}, {}, {}
        for plural, kind in [("queries", "query"), ("passages", "passage")]:
            rows = sorted(inputs[plural], key=lambda row: row["id"])
            ids, texts = [row["id"] for row in rows], [row["text"] for row in rows]
            first = encoder.encode(ids, texts, kind=kind)
            if config["device"].startswith("cuda"):
                torch.cuda.synchronize(config["device"])
            start = time.perf_counter()
            repeated = encoder.encode(ids, texts, kind=kind)
            if config["device"].startswith("cuda"):
                torch.cuda.synchronize(config["device"])
            timings[plural] = time.perf_counter() - start
            if not np.allclose(first.vectors, repeated.vectors, rtol=1e-5, atol=1e-6):
                raise ValueError("Repeated encodings exceed the declared tolerance")
            deltas[plural] = float(np.max(np.abs(first.vectors - repeated.vectors)))
            batches[plural] = first
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **{kind: batch.vectors for kind, batch in batches.items()})
        writer.write_bytes("vectors.npz", buffer.getvalue())
        device = {
            "device": config["device"],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "deterministic_algorithms": True,
            "tf32": False,
            "precision": "float32",
        }
        if config["device"].startswith("cuda"):
            device.update(
                {
                    "name": torch.cuda.get_device_name(config["device"]),
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(config["device"]),
                }
            )
        writer.write_json(
            "encoding.json",
            {
                "batches": {kind: batch.describe() for kind, batch in batches.items()},
                "repeat_max_abs_difference": deltas,
                "repeat_rtol": 1e-5,
                "repeat_atol": 1e-6,
                "device": device,
                "model_load_seconds": load_seconds,
                "warm_repeat_seconds": timings,
                "timing_scope": "tiny compatibility pilot including preprocessing",
            },
        )
        writer.write_json(
            "results.json",
            {
                "scope": "synthetic-e5-encoding-pilot",
                "status": "passed",
                "reportable": False,
                "dimension": config["dimension"],
                "repeat_max_abs_difference": deltas,
            },
        )
        path = writer.seal()
    return {
        "status": "success",
        "run_id": writer.run_id,
        "path": str(path),
        "scope": "synthetic E5 compatibility only",
        "reportable": False,
    }
