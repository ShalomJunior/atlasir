"""Single-writer embedding checkpoints; completed shards are verified and never replaced."""

import contextlib
import fcntl
import io
import json
import os
import tempfile
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from atlasir.data import identifier, stable_hash
from atlasir.encoding import EncodingBatch, resolved_encoder_config
from atlasir.objects import BlobStore, file_hash


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _publish(path: Path, content: bytes) -> None:
    """Publish by exclusive hard-link so an existing checkpoint can never be overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, path)


def encoder_profile(encoder, model_files: Mapping[str, str]) -> dict[str, Any]:
    import os
    import platform
    import subprocess
    from importlib.metadata import version

    import torch

    import atlasir.encoding

    runtime = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": {
            name: version(name) for name in ("torch", "transformers", "safetensors", "numpy")
        },
        "device": encoder.config["device"],
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if encoder.config["device"].startswith("cuda"):
        properties = torch.cuda.get_device_properties(encoder.config["device"])
        runtime.update(
            {
                "gpu": properties.name,
                "cuda": torch.version.cuda,
                "compute_capability": [properties.major, properties.minor],
            }
        )
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        runtime["driver_versions"] = sorted(set(driver.stdout.strip().splitlines()))
    else:
        cpuinfo = Path("/proc/cpuinfo")
        runtime["cpu_model"] = (
            next(
                (
                    line.split(":", 1)[1].strip()
                    for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            )
            if cpuinfo.exists()
            else platform.processor()
        )
    return {
        "config": dict(encoder.config),
        "implementation_sha256": file_hash(Path(atlasir.encoding.__file__)),
        "runtime": runtime,
        "model_files": dict(model_files),
    }


class EmbeddingJob:
    def __init__(
        self,
        directory: Path,
        store: BlobStore,
        rows: Mapping[str, str],
        *,
        kind: str,
        profile: dict[str, Any],
        shard_size: int = 64,
    ):
        if kind not in ("query", "passage") or not isinstance(rows, Mapping) or not rows:
            raise ValueError("Choose a nonempty query or passage mapping")
        for key, text in rows.items():
            identifier(key)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Embedding rows need nonblank string text")
        if type(shard_size) is not int or not 1 <= shard_size <= 100000:
            raise ValueError("Invalid embedding shard size")
        if set(profile) != {"config", "implementation_sha256", "runtime", "model_files"}:
            raise ValueError("Incomplete encoder profile")
        resolved_encoder_config(profile["config"])
        if directory.is_symlink():
            raise ValueError("Checkpoint directory cannot be a symlink")
        self.directory, self.store = directory.resolve(), store
        self.ids = sorted(rows)
        self.rows = [(key, rows[key]) for key in self.ids]
        self.kind, self.profile, self.shard_size = kind, profile, shard_size
        inputs_object = store.put_bytes(_json(self.rows))
        self.spec = {
            "schema_version": 1,
            "protocol": "day07-embedding-shards-v1",
            "kind": kind,
            "encoder": profile,
            "shard_size": shard_size,
            "ids": self.ids,
            "rows_sha256": stable_hash(self.rows),
            "inputs_object": inputs_object,
        }
        self.job_id = stable_hash(self.spec)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = {"job_id": self.job_id, "spec": self.spec}
        manifest_path = self.directory / "job.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_bytes()) != self.manifest:
                raise ValueError("Incompatible checkpoint inputs, encoder, runtime or shard size")
        else:
            _publish(manifest_path, _json(self.manifest))

    @contextlib.contextmanager
    def _lock(self):
        with (self.directory / ".writer.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another writer is using this checkpoint") from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _parts(self):
        for start in range(0, len(self.rows), self.shard_size):
            yield start // self.shard_size, self.rows[start : start + self.shard_size]

    def _path(self, number: int) -> Path:
        return self.directory / "shards" / f"{number:08d}.json"

    def _verify_shard(self, number: int, rows) -> tuple[dict[str, Any], np.ndarray]:
        if self._path(number).is_symlink():
            raise ValueError("Shard metadata cannot be a symlink")
        metadata = json.loads(self._path(number).read_bytes())
        expected_input = stable_hash({"kind": self.kind, "rows": rows})
        if metadata["job_id"] != self.job_id or metadata["number"] != number:
            raise ValueError("Shard belongs to a different job")
        description = metadata["encoding"]
        if (
            description["ids"] != [row[0] for row in rows]
            or description["input_sha256"] != expected_input
            or description["kind"] != self.kind
            or description["config"] != self.profile["config"]
        ):
            raise ValueError("Shard row mapping or encoder identity differs")
        data = np.load(io.BytesIO(self.store.read(metadata["vectors_object"])), allow_pickle=False)
        dimension = self.profile["config"]["dimension"]
        if (
            data.dtype != np.float32
            or data.shape != (len(rows), dimension)
            or not np.isfinite(data).all()
        ):
            raise ValueError("Invalid shard vector array")
        if not np.allclose(np.linalg.norm(data, axis=1), 1, atol=1e-5):
            raise ValueError("Shard vectors are not unit normalized")
        if description["shape"] != list(data.shape) or description["dtype"] != "float32":
            raise ValueError("Shard vector descriptor differs")
        lengths = description["token_lengths_before_truncation"]
        if len(lengths) != len(rows) or any(type(n) is not int or n <= 0 for n in lengths):
            raise ValueError("Invalid shard token lengths")
        truncated = [
            key
            for (key, _), length in zip(rows, lengths, strict=True)
            if length > self.profile["config"]["max_length"]
        ]
        if description["truncated_ids"] != truncated:
            raise ValueError("Invalid shard truncation metadata")
        return metadata, data

    def verify(self, *, require_complete: bool = False) -> list[dict[str, Any]]:
        if (self.directory / "job.json").is_symlink() or (self.directory / "shards").is_symlink():
            raise ValueError("Checkpoint metadata cannot use symlinks")
        if json.loads((self.directory / "job.json").read_bytes()) != self.manifest:
            raise ValueError("Checkpoint manifest changed")
        self.store.verify(self.spec["inputs_object"])
        for digest in self.profile["model_files"].values():
            self.store.verify(digest)
        expected = {f"{number:08d}.json" for number, _ in self._parts()}
        shards = self.directory / "shards"
        if shards.exists() and any(
            p.name not in expected for p in shards.iterdir() if not p.name.startswith(".pending-")
        ):
            raise ValueError("Checkpoint contains an unexpected shard")
        verified = []
        for number, rows in self._parts():
            if self._path(number).exists():
                metadata, _ = self._verify_shard(number, rows)
                verified.append(metadata)
            elif require_complete:
                raise ValueError("Checkpoint is incomplete")
        complete = self.directory / "COMPLETE.json"
        if complete.exists():
            saved = json.loads(complete.read_bytes())
            checksums = {
                self._path(number).name: file_hash(self._path(number))
                for number, _ in self._parts()
            }
            if saved != {"job_id": self.job_id, "shard_sha256": checksums}:
                raise ValueError("Completed checkpoint inventory differs")
        elif require_complete:
            raise ValueError("Checkpoint completion marker missing")
        return verified

    def run(self, encoder, *, stop_after_new_shards: int | None = None) -> dict[str, Any]:
        if stop_after_new_shards is not None and (
            type(stop_after_new_shards) is not int or stop_after_new_shards < 1
        ):
            raise ValueError("stop_after_new_shards must be positive")
        if dict(encoder.config) != self.profile["config"]:
            raise ValueError("Encoder differs from checkpoint profile")
        if (
            not hasattr(encoder, "checkpoint_profile")
            or encoder.checkpoint_profile() != self.profile
        ):
            raise ValueError(
                "Encoder implementation, model files or runtime differ from checkpoint"
            )
        with self._lock():
            verified = self.verify()
            previous = []
            attempts = self.directory / "attempts"
            attempts.mkdir(exist_ok=True)
            for path in sorted(attempts.glob("*.json")):
                previous.append(json.loads(path.read_bytes()))
            attempt_id = uuid.uuid4().hex
            attempt = {
                "attempt_id": attempt_id,
                "job_id": self.job_id,
                "started_utc": datetime.now(UTC).isoformat(),
                "parent_attempt_id": max(previous, key=lambda x: x["started_utc"])["attempt_id"]
                if previous
                else None,
                "reused_shards": len(verified),
                "new_shards": 0,
                "status": "running",
            }
            path = attempts / f"{attempt_id}.json"
            _publish(path, _json(attempt))
            try:
                for number, rows in self._parts():
                    if self._path(number).exists():
                        continue
                    start = time.perf_counter()
                    batch = encoder.encode(
                        [row[0] for row in rows], [row[1] for row in rows], kind=self.kind
                    )
                    elapsed = time.perf_counter() - start
                    if not isinstance(batch, EncodingBatch):
                        raise ValueError("Encoder must return an EncodingBatch")
                    output = io.BytesIO()
                    np.save(output, batch.vectors, allow_pickle=False)
                    digest = self.store.put_bytes(output.getvalue())
                    record = {
                        "job_id": self.job_id,
                        "number": number,
                        "vectors_object": digest,
                        "encoding": batch.describe(),
                        "encoding_seconds": elapsed,
                        "attempt_id": attempt_id,
                    }
                    # Validate encoder output before publishing the immutable shard metadata.
                    if batch.ids != tuple(
                        row[0] for row in rows
                    ) or batch.input_sha256 != stable_hash({"kind": self.kind, "rows": rows}):
                        raise ValueError("Encoder returned wrong IDs or input identity")
                    if batch.kind != self.kind or batch.config != self.profile["config"]:
                        raise ValueError("Encoder returned incompatible metadata")
                    if len(batch.token_lengths) != len(rows) or any(
                        type(n) is not int or n <= 0 for n in batch.token_lengths
                    ):
                        raise ValueError("Encoder returned invalid token lengths")
                    if (
                        batch.vectors.shape != (len(rows), self.profile["config"]["dimension"])
                        or batch.vectors.dtype != np.float32
                        or not np.isfinite(batch.vectors).all()
                        or not np.allclose(np.linalg.norm(batch.vectors, axis=1), 1, atol=1e-5)
                    ):
                        raise ValueError("Encoder returned invalid vectors")
                    _publish(self._path(number), _json(record))
                    self._verify_shard(number, rows)
                    attempt["new_shards"] += 1
                    if stop_after_new_shards and attempt["new_shards"] >= stop_after_new_shards:
                        break
                records = self.verify()
                count = sum(1 for _ in self._parts())
                if len(records) == count:
                    marker = {
                        "job_id": self.job_id,
                        "shard_sha256": {
                            self._path(number).name: file_hash(self._path(number))
                            for number, _ in self._parts()
                        },
                    }
                    if not (self.directory / "COMPLETE.json").exists():
                        _publish(self.directory / "COMPLETE.json", _json(marker))
                    self.verify(require_complete=True)
                    attempt["status"] = "complete"
                else:
                    attempt["status"] = "paused"
            except BaseException as error:
                attempt["status"] = "failed"
                attempt["error_type"] = type(error).__name__
                raise
            finally:
                attempt["finished_utc"] = datetime.now(UTC).isoformat()
                temporary = path.with_suffix(".pending")
                with temporary.open("xb") as stream:
                    stream.write(_json(attempt))
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
            return attempt

    def arrays(self) -> tuple[list[str], np.ndarray]:
        self.verify(require_complete=True)
        blocks = [self._verify_shard(number, rows)[1] for number, rows in self._parts()]
        return list(self.ids), np.concatenate(blocks)

    def object_ids(self) -> list[str]:
        records = self.verify()
        return sorted(
            {
                self.spec["inputs_object"],
                *self.profile["model_files"].values(),
                *(record["vectors_object"] for record in records),
            }
        )
