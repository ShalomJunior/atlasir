"""Build a private Kaggle input ZIP from current source and a verified small reference bundle."""

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from atlasir.artifacts import restore_run, source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="atlasir-handoff-") as temporary:
        manifest = restore_run(args.reference, Path(temporary) / "reference")
    if manifest["schema_version"] != 2:
        raise ValueError("Reference must be a baseline run")
    source, _ = source_snapshot(Path(__file__).resolve().parents[1])
    payloads = {
        "source.tar.gz": source,
        "reference.bundle/archive.json": (args.reference / "archive.json").read_bytes(),
        "reference.bundle/run.tar.gz": (args.reference / "run.tar.gz").read_bytes(),
        "SCIFACT-ATTRIBUTION.md": args.attribution.read_bytes(),
    }
    metadata = {
        "schema_version": 1,
        "reference_run_id": manifest["run_id"],
        "uv_version": "0.11.26",
        "python_profile": "3.11",
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    payloads["launcher-input.json"] = (
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    ).encode()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    print(
        json.dumps(
            {
                "path": str(args.destination),
                "bytes": args.destination.stat().st_size,
                "zip_sha256": hashlib.sha256(args.destination.read_bytes()).hexdigest(),
                **metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
