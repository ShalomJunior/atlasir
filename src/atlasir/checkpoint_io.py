"""Portable checkpoint metadata and offline materialization of verified model objects."""

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from atlasir.objects import BlobStore
from atlasir.shards import EmbeddingJob

MODEL_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "README.md",
}
REQUIRED_MODEL_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}


def materialize_model(store: BlobStore, files: dict[str, str], destination: Path) -> Path:
    if not REQUIRED_MODEL_FILES <= files.keys() or files.keys() - MODEL_FILES:
        raise ValueError("Unexpected model file set")
    destination.mkdir(parents=True, exist_ok=False)
    for name, digest in files.items():
        source = store.verify(digest)
        with source.open("rb") as incoming, (destination / name).open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    return destination


def _member(name: str) -> bool:
    return name in {"job.json", "COMPLETE.json"} or bool(
        re.fullmatch(r"(shards/[0-9]{8}|attempts/[a-f0-9]{32})\.json", name)
    )


def export_checkpoint(job: EmbeddingJob, destination: Path) -> Path:
    with job._lock():
        job.verify()
        paths = [
            p
            for p in job.directory.rglob("*.json")
            if _member(p.relative_to(job.directory).as_posix())
        ]
        if any(p.is_symlink() for p in paths):
            raise ValueError("Cannot export symlinked metadata")
        files = {p.relative_to(job.directory).as_posix(): p.read_bytes() for p in paths}
        inventory = {
            "schema_version": 1,
            "job_id": job.job_id,
            "object_ids": job.object_ids(),
            "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "checkpoint-inventory.json", json.dumps(inventory, sort_keys=True, indent=2) + "\n"
            )
            for name, data in sorted(files.items()):
                archive.writestr(name, data)
    return destination


def restore_checkpoint(archive_path: Path, destination: Path, store: BlobStore) -> EmbeddingJob:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.getinfo("checkpoint-inventory.json").file_size > 16 * 1024 * 1024:
            raise ValueError("Checkpoint inventory exceeds its metadata size limit")
        inventory = json.loads(archive.read("checkpoint-inventory.json"))
        if inventory["schema_version"] != 1:
            raise ValueError("Unknown checkpoint inventory")
        files = inventory["sha256"]
        names = archive.namelist()
        if (
            len(names) != len(set(names))
            or set(names) != set(files) | {"checkpoint-inventory.json"}
            or "job.json" not in files
            or any(not _member(name) for name in files)
            or sum(info.file_size for info in archive.infolist()) > 128 * 1024 * 1024
        ):
            raise ValueError("Invalid checkpoint archive members")
        for digest in inventory["object_ids"]:
            store.verify(digest)
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=".checkpoint-restore-"
        ) as temporary:
            work = Path(temporary) / "work"
            work.mkdir()
            for name, digest in files.items():
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError("Checkpoint metadata checksum mismatch")
                path = work / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            manifest = json.loads((work / "job.json").read_bytes())
            spec = manifest["spec"]
            rows = dict(json.loads(store.read(spec["inputs_object"])))
            job = EmbeddingJob(
                work,
                store,
                rows,
                kind=spec["kind"],
                profile=spec["encoder"],
                shard_size=spec["shard_size"],
            )
            if job.job_id != inventory["job_id"] or job.object_ids() != sorted(
                inventory["object_ids"]
            ):
                raise ValueError("Checkpoint job/object identity mismatch")
            job.verify(require_complete=(work / "COMPLETE.json").exists())
            work.rename(destination)
    return EmbeddingJob(
        destination,
        store,
        rows,
        kind=spec["kind"],
        profile=spec["encoder"],
        shard_size=spec["shard_size"],
    )
