"""Small, portable run bundles with explicit completion and fail-on-overwrite semantics."""

import contextlib
import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
REQUIRED_FILES = {
    "manifest.json",
    "config.resolved.json",
    "environment.json",
    "inputs.json",
    "source.tar.gz",
    "fixture.json",
    "results.json",
    "events.jsonl",
}
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
BASELINE_REQUIRED_FILES = (REQUIRED_FILES - {"fixture.json"}) | {
    "dataset.json",
    "selection.json",
    "data.corpus.jsonl",
    "data.queries.jsonl",
    "data.qrels.tsv",
    "rankings.json",
    "rankings.trec.gz",
    "metrics.json",
    "per_query.json",
    "timing.json",
    "index.json",
    "index.npz",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write(path: Path, content: bytes) -> None:
    """Exclusively create and flush a file; never replace a previous artifact."""
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_metadata(path: Path, value: Any) -> None:
    """Atomically update metadata only while its run is unfinished."""
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    _write(temporary, _encoded(value))
    temporary.replace(path)


def _run_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,119}", value):
        raise ValueError(
            "Invalid run ID: use lowercase letters, digits, dots, dashes or underscores"
        )
    return value


def resolved_fixture_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve and validate the deliberately small E00 configuration schema."""
    defaults = {"schema_version": 1, "family": "E00", "protocol": "day02-fixture-v1", "seed": 17}
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or overrides.keys() - defaults.keys():
        raise ValueError("E00 configuration must be an object containing only declared fields")
    config = defaults | overrides
    for key in ("schema_version", "family", "protocol"):
        if type(config[key]) is not type(defaults[key]) or config[key] != defaults[key]:
            raise ValueError(f"Unsupported E00 configuration field: {key}")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    return config


def environment() -> dict[str, Any]:
    """Capture installed package versions and public runtime facts, never environment secrets."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "device": "cpu",
        "packages": sorted(
            {f"{package.metadata['Name']}=={package.version}" for package in distributions()}
        ),
    }


def source_snapshot(root: Path) -> tuple[bytes, dict[str, Any]]:
    """Archive only declared project source/config/docs; exclude credentials and generated data."""
    root = root.resolve()
    required = ("pyproject.toml", "uv.lock", "src/atlasir")
    if any(not (root / name).exists() for name in required):
        raise ValueError("source root must contain pyproject.toml, uv.lock and src/atlasir")
    names = (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".gitignore",
        "README.md",
        "LICENSE",
        "src",
        "tests",
        "scripts",
        "notebooks",
        "configs",
        "docs",
        ".github",
    )
    candidates = []
    for name in names:
        path = root / name
        if path.is_symlink():
            raise ValueError(f"Source snapshot cannot include a symlink: {name}")
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            for current, dirs, files in os.walk(path, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
                for filename in sorted(files):
                    candidate = Path(current) / filename
                    if filename.startswith(".") or candidate.suffix in {".pyc", ".pyo"}:
                        continue
                    if candidate.is_symlink():
                        raise ValueError("Source snapshot cannot include symlinked files")
                    candidates.append(candidate)
                if any((Path(current) / directory).is_symlink() for directory in dirs):
                    raise ValueError("Source snapshot cannot include symlinked directories")
    output = io.BytesIO()
    total = 0
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in sorted(candidates):
                content = path.read_bytes()
                total += len(content)
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError("Source snapshot exceeds the size limit")
                member = tarfile.TarInfo(path.relative_to(root).as_posix())
                member.size = len(content)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(content))
    git: dict[str, Any] = {"commit": None, "dirty": None}
    try:

        def query(*args: str) -> str:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True, timeout=10
            ).strip()

        git = {
            "commit": query("rev-parse", "HEAD"),
            "dirty": bool(query("status", "--porcelain", "--untracked-files=all")),
        }
    except (OSError, subprocess.SubprocessError):
        pass
    return output.getvalue(), git


def _files(root: Path) -> dict[str, dict[str, Any]]:
    inventory = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Run bundles cannot contain symlinks")
        if path.is_file():
            content = path.read_bytes()
            inventory[path.relative_to(root).as_posix()] = {
                "sha256": _digest(content),
                "bytes": len(content),
            }
    return inventory


def verify_run(path: Path) -> dict[str, Any]:
    """Verify every saved byte and reject missing, additional or inconsistent files."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Run must be a real directory")
    actual = _files(path)
    try:
        checksum_bytes = (path / "artifact_checksums.json").read_bytes()
        checksums = json.loads(checksum_bytes)
        complete = json.loads((path / "COMPLETE.json").read_bytes())
        manifest = json.loads((path / "manifest.json").read_bytes())
        if complete["checksums_sha256"] != _digest(checksum_bytes):
            raise ValueError("Checksum inventory does not match completion marker")
        for name in ("artifact_checksums.json", "COMPLETE.json"):
            actual.pop(name)
        schema = manifest["schema_version"]
        required = BASELINE_REQUIRED_FILES if schema == 2 else REQUIRED_FILES
        if actual != checksums or not required <= actual.keys():
            raise ValueError("Run files do not match the saved checksum inventory")
        if (
            type(schema) is not int
            or schema not in {1, 2}
            or manifest["status"] != "success"
            or complete["run_id"] != manifest["run_id"]
        ):
            raise ValueError("Unsupported or incomplete run manifest")
        _run_id(manifest["run_id"])
        raw_config = json.loads((path / "config.resolved.json").read_bytes())
        if schema == 2:
            from atlasir.bm25 import resolved_bm25_config

            config = resolved_bm25_config(raw_config)
        else:
            config = resolved_fixture_config(raw_config)
        spec = {
            "config": config,
            "source_sha256": _digest((path / "source.tar.gz").read_bytes()),
            "inputs": json.loads((path / "inputs.json").read_bytes()),
            "environment": json.loads((path / "environment.json").read_bytes()),
        }
        if manifest["spec_id"] != _digest(_encoded(spec)):
            raise ValueError("Run specification does not match its saved provenance")
        if manifest["source_sha256"] != spec["source_sha256"]:
            raise ValueError("Source identity does not match the saved snapshot")
        if schema == 1:
            if spec["inputs"]["fixture_sha256"] != _digest((path / "fixture.json").read_bytes()):
                raise ValueError("Fixture does not match the recorded input identity")
        else:
            payloads = spec["inputs"]["payload_files"]
            expected = {
                "dataset.json",
                "selection.json",
                "data.corpus.jsonl",
                "data.queries.jsonl",
                "data.qrels.tsv",
            }
            if set(payloads) != expected or any(
                _digest((path / name).read_bytes()) != digest for name, digest in payloads.items()
            ):
                raise ValueError("Baseline input snapshot does not match its specification")
            if manifest["family"] != "E01" or manifest["protocol"] != config["protocol"]:
                raise ValueError("Unsupported baseline manifest")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid or unfinished run bundle: {path}") from error


class RunWriter:
    """Reserve, write and seal E00 fixtures or schema-2 BM25 baselines."""

    def __init__(
        self,
        root: Path,
        source_root: Path,
        config: dict[str, Any],
        fixture: bytes | None,
        run_id: str | None = None,
        invocation: list[str] | None = None,
        *,
        baseline_payloads: dict[str, bytes] | None = None,
    ) -> None:
        self.root = root.resolve()
        if baseline_payloads is None:
            config = resolved_fixture_config(config)
            if fixture is None:
                raise ValueError("A fixture input is required for E00")
            inputs = {"fixture_sha256": _digest(fixture), "fixture": "synthetic-v1"}
            payloads = {"fixture.json": fixture}
            schema, scope = 1, "artifact-lifecycle-fixture"
            self.required_files = REQUIRED_FILES
        else:
            from atlasir.bm25 import resolved_bm25_config

            config = resolved_bm25_config(config)
            payloads = dict(baseline_payloads)
            expected = {
                "dataset.json",
                "selection.json",
                "data.corpus.jsonl",
                "data.queries.jsonl",
                "data.qrels.tsv",
            }
            if set(payloads) != expected or fixture is not None:
                raise ValueError("BM25 requires its declared input snapshot and no E00 fixture")
            inputs = {
                "payload_files": {name: _digest(content) for name, content in payloads.items()}
            }
            schema, scope = 2, "exploratory-bm25-baseline"
            self.required_files = BASELINE_REQUIRED_FILES
        source, git = source_snapshot(source_root)
        runtime = environment()
        spec = {
            "config": config,
            "source_sha256": _digest(source),
            "inputs": inputs,
            "environment": runtime,
        }
        spec_id = _digest(_encoded(spec))
        self.run_id = _run_id(
            run_id
            or (
                f"{datetime.now(UTC):%Y%m%dt%H%M%Sz}-{config['family'].lower()}-{spec_id[:8]}-{uuid.uuid4().hex[:12]}"
            )
        )
        reservations = self.root / ".scratch" / "reservations"
        reservations.mkdir(parents=True, exist_ok=True)
        # Permanent reservations prevent concurrent writers and reuse of failed attempt IDs.
        (reservations / self.run_id).mkdir()
        self.path = self.root / ".scratch" / self.run_id
        self.final_path = self.root / "artifacts" / "runs" / self.run_id
        if self.final_path.exists():
            raise FileExistsError(self.final_path)
        self.path.mkdir()
        self.closed = False
        self.manifest = {
            "schema_version": schema,
            "run_id": self.run_id,
            "spec_id": spec_id,
            "family": config["family"],
            "protocol": config["protocol"],
            "started_at": _now(),
            "status": "running",
            "reportable": False,
            "scope": scope,
            "git": git,
            "source_sha256": _digest(source),
            "invocation": invocation,
        }
        self.write_json("manifest.json", self.manifest)
        self.write_json("config.resolved.json", config)
        self.write_json("environment.json", runtime)
        self.write_json("inputs.json", inputs)
        self.write_bytes("source.tar.gz", source)
        for name, content in payloads.items():
            self.write_bytes(name, content)
        self.write_bytes(
            "events.jsonl",
            (
                _encoded({"event": "created", "at": _now()}).decode().replace("\n", "") + "\n"
            ).encode(),
        )

    def __enter__(self) -> "RunWriter":
        return self

    def write_bytes(self, name: str, content: bytes) -> None:
        if self.closed:
            raise ValueError("A closed run cannot be changed")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", name):
            raise ValueError("Artifact name must be a single safe filename")
        if name in {"artifact_checksums.json", "COMPLETE.json"}:
            raise ValueError("Completion files are managed by sealing")
        _write(self.path / name, content)

    def write_json(self, name: str, value: Any) -> None:
        self.write_bytes(name, _encoded(value))

    def seal(self) -> Path:
        if self.closed:
            raise ValueError("A closed run cannot be sealed again")
        if not self.required_files <= _files(self.path).keys():
            raise ValueError("Cannot seal a run with missing required outputs")
        self.manifest |= {"status": "success", "ended_at": _now()}
        _replace_metadata(self.path / "manifest.json", self.manifest)
        _write(self.path / "artifact_checksums.json", _encoded(_files(self.path)))
        _write(
            self.path / "COMPLETE.json",
            _encoded(
                {
                    "run_id": self.run_id,
                    "checksums_sha256": _digest(
                        (self.path / "artifact_checksums.json").read_bytes()
                    ),
                }
            ),
        )
        verify_run(self.path)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.stat().st_dev != self.final_path.parent.stat().st_dev:
            raise ValueError("Sealing requires scratch and final output on the same filesystem")
        if self.final_path.exists():
            raise FileExistsError(self.final_path)
        self.path.rename(self.final_path)
        self.closed = True
        catalog = self.root / "experiments" / "catalog"
        catalog.mkdir(parents=True, exist_ok=True)
        _write(
            catalog / f"{self.run_id}.json",
            _encoded(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "spec_id": self.manifest["spec_id"],
                    "state": "local-sealed",
                    "run_path": f"artifacts/runs/{self.run_id}",
                    "checksums_sha256": _digest(
                        (self.final_path / "artifact_checksums.json").read_bytes()
                    ),
                }
            ),
        )
        return self.final_path

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.closed:
            for name in ("COMPLETE.json", "artifact_checksums.json"):
                (self.path / name).unlink(missing_ok=True)
            self.manifest |= {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "ended_at": _now(),
                "error_type": type(exc).__name__ if exc is not None else "UnsealedRun",
            }
            _replace_metadata(self.path / "manifest.json", self.manifest)
            self.closed = True
            if exc_type is None:
                raise ValueError("Run exited without being sealed")


@contextlib.contextmanager
def _new_directory(destination: Path) -> Iterator[Path]:
    """Publish a new directory only after validation; coordinate competing API callers."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    reservation = destination.with_name(f".{destination.name}.lock")
    reservation.mkdir()
    try:
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(
            prefix=".atlasir-transfer-", dir=destination.parent
        ) as tmp:
            work = Path(tmp) / "content"
            work.mkdir()
            yield work
            if destination.exists():
                raise FileExistsError(destination)
            work.rename(destination)
    finally:
        reservation.rmdir()


def export_run(run: Path, destination: Path) -> Path:
    """Create a portable directory containing a gzip tar archive and its external SHA-256."""
    verify_run(run)
    with _new_directory(destination) as work:
        archive = work / "run.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for path in sorted(run.iterdir()):
                stream.add(path, arcname=path.name, recursive=True)
        # Recheck the bytes actually exported, not only the source before copying.
        restored = work / "verification"
        restored.mkdir()
        _extract(archive, restored)
        verify_run(restored)
        shutil.rmtree(restored)
        _write(
            work / "archive.json",
            _encoded(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sha256": _digest(archive.read_bytes()),
                    "bytes": archive.stat().st_size,
                }
            ),
        )
    return destination


def _extract(archive: Path, destination: Path) -> None:
    """Extract regular files manually: no traversal, links, duplicate paths or special files."""
    with tarfile.open(archive, "r:gz") as stream:
        seen = set()
        total = 0
        for member in stream:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or member.name != path.as_posix()
                or member.name in seen
                or not path.parts
            ):
                raise ValueError("Archive contains an unsafe or duplicate entry")
            seen.add(member.name)
            total += member.size
            if total > MAX_ARCHIVE_BYTES or len(seen) > 10000:
                raise ValueError("Archive exceeds the fixture restore size limit")
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise ValueError("Archive entry has no content")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def restore_run(bundle: Path, destination: Path) -> dict[str, Any]:
    """Restore to a new directory, verify independently, then publish. Never run saved code."""
    metadata = json.loads((bundle / "archive.json").read_bytes())
    archive = bundle / "run.tar.gz"
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != SCHEMA_VERSION
        or archive.stat().st_size != metadata.get("bytes")
        or _digest(archive.read_bytes()) != metadata.get("sha256")
    ):
        raise ValueError("Exported archive checksum or size mismatch")
    with _new_directory(destination) as work:
        _extract(archive, work)
        manifest = verify_run(work)
    return manifest
