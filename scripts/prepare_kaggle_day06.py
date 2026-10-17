"""Create the small archive-safe source input for the synthetic E5 GPU pilot."""

import argparse
import base64
import hashlib
import json
import zipfile
from pathlib import Path

from atlasir.artifacts import source_snapshot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--destination", type=Path, required=True)
args = parser.parse_args()
source, _ = source_snapshot(Path(__file__).resolve().parents[1])
manifest = {
    "schema_version": 2,
    "job": "day06-e5-synthetic-pilot",
    "uv_version": "0.11.26",
    "python_profile": "3.11",
    "sha256": {"source.tar.gz": hashlib.sha256(source).hexdigest()},
    "base64_files": ["source.tar.gz"],
}
args.destination.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("source.tar.gz.b64", base64.b64encode(source))
    archive.writestr("launcher-input.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
print(
    json.dumps(
        {
            "path": str(args.destination),
            "bytes": args.destination.stat().st_size,
            "zip_sha256": hashlib.sha256(args.destination.read_bytes()).hexdigest(),
            **manifest,
        },
        indent=2,
    )
)
