"""Pinned E5 encoding, explicit preprocessing and immutable ID/vector batches."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atlasir.data import Document, Query, identifier, stable_hash

MODEL_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"


def resolved_encoder_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    defaults = {
        "schema_version": 3,
        "family": "E02",
        "protocol": "day06-e5-pilot-v1",
        "model": "intfloat/e5-small-v2",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "pooling": "attention-mask mean",
        "normalize": True,
        "dtype": "float32",
        "attention": "eager",
        "max_length": 512,
        "dimension": 384,
        "batch_size": 8,
        "seed": 17,
        "device": "cpu",
    }
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, Mapping) or overrides.keys() - defaults.keys():
        raise ValueError("Encoder configuration contains undeclared fields")
    config = defaults | dict(overrides)
    for key in defaults.keys() - {"batch_size", "seed", "device"}:
        if type(config[key]) is not type(defaults[key]) or config[key] != defaults[key]:
            raise ValueError(f"Unsupported E5 setting: {key}")
    if type(config["batch_size"]) is not int or not 1 <= config["batch_size"] <= 128:
        raise ValueError("batch_size must be an integer in [1, 128]")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be a uint32 integer")
    if config["device"] not in ("cpu", "cuda:0", "cuda:1"):
        raise ValueError("Choose cpu, cuda:0 or cuda:1 explicitly")
    return config


@dataclass(frozen=True)
class EncodingBatch:
    ids: tuple[str, ...]
    vectors: np.ndarray
    token_lengths: tuple[int, ...]
    kind: str
    input_sha256: str
    config: dict[str, Any]

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "ids": list(self.ids),
            "shape": list(self.vectors.shape),
            "dtype": str(self.vectors.dtype),
            "input_sha256": self.input_sha256,
            "config": self.config,
            "token_lengths_before_truncation": list(self.token_lengths),
            "truncated_ids": [
                key
                for key, length in zip(self.ids, self.token_lengths, strict=True)
                if length > self.config["max_length"]
            ],
            "length_definition": "prefixed text including tokenizer special tokens",
            "truncation": "right; longest texts cut at max_length including special tokens",
            "document_text": "nonempty title and text joined by newline",
        }


def mean_pool_normalize(hidden, attention_mask):
    """E5 model-card pooling, including attended special tokens and excluding padding."""
    import torch
    import torch.nn.functional as functional

    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden states and attention mask have incompatible shapes")
    mask = attention_mask.bool()
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Cannot pool a sequence without attended tokens")
    pooled = hidden.float().masked_fill(~mask[..., None], 0).sum(dim=1) / counts[:, None]
    if not torch.isfinite(pooled).all() or torch.any(torch.linalg.vector_norm(pooled, dim=1) == 0):
        raise ValueError("Encoder produced nonfinite or zero-norm pooled vectors")
    return functional.normalize(pooled, p=2, dim=1)


class E5Encoder:
    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        local_files_only: bool = False,
        model_directory: Path | None = None,
        expected_files: Mapping[str, str] | None = None,
    ):
        self.config = resolved_encoder_config(config)
        model_source = self.config["model"]
        if model_directory is not None:
            from atlasir.checkpoint_io import MODEL_FILES, REQUIRED_MODEL_FILES
            from atlasir.objects import file_hash

            if (
                expected_files is None
                or not REQUIRED_MODEL_FILES <= expected_files.keys()
                or expected_files.keys() - MODEL_FILES
            ):
                raise ValueError("Offline model directory requires verified model-file identities")
            if {p.name for p in model_directory.iterdir()} != set(expected_files):
                raise ValueError("Offline model directory has unexpected files")
            for name, digest in expected_files.items():
                path = model_directory / name
                if path.is_symlink() or file_hash(path) != digest:
                    raise ValueError("Offline model file checksum mismatch")
            model_source = str(model_directory.resolve())
            local_files_only = True
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ValueError(
                "Install the locked dense extra: uv sync --locked --extra dense"
            ) from error
        import os

        if self.config["device"].startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("Requested CUDA device is unavailable; no automatic CPU fallback")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(self.config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            revision=self.config["tokenizer_revision"],
            trust_remote_code=False,
            local_files_only=local_files_only,
            use_fast=True,
        )
        self.tokenizer.truncation_side = "right"
        self.model = (
            AutoModel.from_pretrained(
                model_source,
                revision=self.config["model_revision"],
                trust_remote_code=False,
                local_files_only=local_files_only,
                use_safetensors=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            .to(self.config["device"])
            .eval()
        )
        if self.model.config.hidden_size != self.config["dimension"]:
            raise ValueError("Unexpected model embedding dimension")
        if model_directory is not None:
            self._checkpoint_model_directory = model_directory.resolve()
        else:
            from huggingface_hub import snapshot_download

            self._checkpoint_model_directory = Path(
                snapshot_download(
                    self.config["model"],
                    revision=self.config["model_revision"],
                    local_files_only=True,
                )
            )
        from atlasir.checkpoint_io import MODEL_FILES
        from atlasir.objects import file_hash

        self._checkpoint_model_files = {
            name: file_hash(self._checkpoint_model_directory / name)
            for name in sorted(MODEL_FILES)
            if (self._checkpoint_model_directory / name).is_file()
        }

    def checkpoint_profile(self) -> dict[str, Any]:
        """Measure actual loaded-file/runtime identity before checkpoint reuse."""
        from atlasir.checkpoint_io import MODEL_FILES
        from atlasir.objects import file_hash
        from atlasir.shards import encoder_profile

        files = {
            name: file_hash(self._checkpoint_model_directory / name)
            for name in sorted(MODEL_FILES)
            if (self._checkpoint_model_directory / name).is_file()
        }
        if files != self._checkpoint_model_files:
            raise ValueError("Loaded model cache files changed after initialization")
        return encoder_profile(self, files)

    def encode(self, ids: Sequence[str], texts: Sequence[str], *, kind: str) -> EncodingBatch:
        import torch

        if kind not in {"query", "passage"}:
            raise ValueError("Choose query or passage encoding explicitly")
        ids, texts = tuple(ids), tuple(texts)
        if len(ids) != len(texts) or len(set(ids)) != len(ids):
            raise ValueError("Encoding needs unique IDs and one text per ID")
        for key in ids:
            identifier(key)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Cannot encode blank or non-string texts")
        prefix = self.config[f"{kind}_prefix"]
        values = [prefix + text for text in texts]
        vectors, lengths = [], []
        size = self.config["batch_size"]
        with torch.inference_mode():
            for start in range(0, len(values), size):
                batch = values[start : start + size]
                full = self.tokenizer(batch, truncation=False, padding=False)
                lengths.extend(len(row) for row in full["input_ids"])
                tokens = self.tokenizer(
                    batch,
                    max_length=self.config["max_length"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config["device"])
                hidden = self.model(**tokens).last_hidden_state
                vectors.append(mean_pool_normalize(hidden, tokens["attention_mask"]).cpu().numpy())
        array = np.concatenate(vectors) if vectors else np.empty((0, self.config["dimension"]))
        array = np.array(array, dtype=np.float32, order="C", copy=True)
        if array.shape != (len(ids), self.config["dimension"]) or not np.isfinite(array).all():
            raise ValueError("Unexpected vector shape or nonfinite values")
        if len(ids) and not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5):
            raise ValueError("Vectors are not unit normalized")
        array.flags.writeable = False
        return EncodingBatch(
            ids,
            array,
            tuple(lengths),
            kind,
            stable_hash({"kind": kind, "rows": list(zip(ids, texts, strict=True))}),
            dict(self.config),
        )

    def documents(self, corpus: Mapping[str, Document]) -> EncodingBatch:
        ids = sorted(corpus)
        return self.encode(ids, [corpus[key].contents for key in ids], kind="passage")

    def queries(self, queries: Mapping[str, Query]) -> EncodingBatch:
        ids = sorted(queries)
        return self.encode(ids, [queries[key].text for key in ids], kind="query")
