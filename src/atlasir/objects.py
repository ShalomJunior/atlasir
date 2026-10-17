"""Content-addressed binary objects with verified, portable ZIP transport."""

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

MAX_OBJECT_BUNDLE = 1024 * 1024 * 1024


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("Object identity must be a lowercase SHA256")
    return value


class BlobStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def verify(self, digest: str) -> Path:
        directory = self.root / _digest(digest)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Missing or unsafe object: {digest}")
        if {p.name for p in directory.iterdir()} != {"payload.bin", "object.json"}:
            raise ValueError("Unexpected object contents")
        if any(p.is_symlink() for p in directory.iterdir()):
            raise ValueError("Object cannot contain symlinks")
        path = directory / "payload.bin"
        metadata = json.loads((directory / "object.json").read_bytes())
        if metadata != {"schema_version": 1, "sha256": digest, "bytes": path.stat().st_size}:
            raise ValueError("Invalid object descriptor")
        if file_hash(path) != digest:
            raise ValueError(f"Object checksum mismatch: {digest}")
        return path

    def put_file(self, source: Path) -> str:
        digest = file_hash(source)
        target = self.root / digest
        if target.exists():
            self.verify(digest)
            return digest
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        try:
            payload = temporary / "payload.bin"
            with source.open("rb") as incoming, payload.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if file_hash(payload) != digest:
                raise ValueError("Source object changed during ingestion")
            (temporary / "object.json").write_text(
                json.dumps(
                    {"schema_version": 1, "sha256": digest, "bytes": payload.stat().st_size},
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            )
            try:
                temporary.rename(target)
            except OSError:
                if not target.exists():
                    raise
                self.verify(digest)
            return digest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def put_bytes(self, content: bytes) -> str:
        with tempfile.NamedTemporaryFile(dir=self.root, prefix=".input-") as stream:
            stream.write(content)
            stream.flush()
            return self.put_file(Path(stream.name))

    def read(self, digest: str) -> bytes:
        return self.verify(digest).read_bytes()


def export_objects(store: BlobStore, digests: list[str], destination: Path) -> Path:
    """Export only requested objects. The output file is never overwritten."""
    identities = sorted(set(digests))
    paths = {digest: store.verify(digest) for digest in identities}
    inventory = {digest: path.stat().st_size for digest, path in paths.items()}
    if sum(inventory.values()) > MAX_OBJECT_BUNDLE:
        raise ValueError("Split exports into bundles of at most 1 GiB uncompressed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A partial export remains visible after failure and cannot be mistaken for a valid ZIP.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "object-inventory.json",
            json.dumps({"schema_version": 1, "objects": inventory}, sort_keys=True, indent=2)
            + "\n",
        )
        for digest, path in paths.items():
            archive.write(path, f"objects/{digest}/payload.bin")
    return destination


def import_objects(archive_path: Path, store: BlobStore) -> list[str]:
    """Read only named payload members; no archive-controlled paths are extracted."""
    with zipfile.ZipFile(archive_path) as archive:
        if archive.getinfo("object-inventory.json").file_size > 16 * 1024 * 1024:
            raise ValueError("Object inventory exceeds its metadata size limit")
        inventory = json.loads(archive.read("object-inventory.json"))
        if inventory.get("schema_version") != 1 or not isinstance(inventory.get("objects"), dict):
            raise ValueError("Invalid object inventory")
        objects = inventory["objects"]
        for digest, size in objects.items():
            _digest(digest)
            if type(size) is not int or size < 0:
                raise ValueError("Invalid object size")
        expected = {"object-inventory.json"} | {
            f"objects/{digest}/payload.bin" for digest in objects
        }
        members = archive.namelist()
        if len(members) != len(set(members)) or set(members) != expected:
            raise ValueError("Unexpected or duplicate object archive members")
        if sum(objects.values()) > MAX_OBJECT_BUNDLE:
            raise ValueError("Object archive exceeds restore bound")
        for digest, size in objects.items():
            member = f"objects/{digest}/payload.bin"
            if archive.getinfo(member).file_size != size:
                raise ValueError("Object size differs from inventory")
            with tempfile.NamedTemporaryFile(dir=store.root, prefix=".restore-") as temporary:
                with archive.open(member) as incoming:
                    shutil.copyfileobj(incoming, temporary)
                temporary.flush()
                if file_hash(Path(temporary.name)) != digest:
                    raise ValueError("Object archive checksum mismatch")
                if store.put_file(Path(temporary.name)) != digest:
                    raise ValueError("Restored object identity mismatch")
    return sorted(objects)
