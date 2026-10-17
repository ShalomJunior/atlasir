"""Archive-safe source and full SciFact development inputs for one substantive session."""

import argparse
import base64
import hashlib
import json
import zipfile
from pathlib import Path

from atlasir.artifacts import source_snapshot
from atlasir.comparison import _datasets

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--destination", type=Path, required=True)
parser.add_argument("--data", type=Path, default=Path(".scratch/data/scifact"))
parser.add_argument(
    "--selection", type=Path, default=Path("experiments/splits/scifact-seed17.json")
)
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
_, datasets = _datasets(args.data, "scifact", args.selection)
source, _ = source_snapshot(project)
payloads = {"source.tar.gz": source, "selection.json": args.selection.read_bytes()}
for name in ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv"):
    content = (args.data / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != datasets["train"].file_hashes[name]:
        raise ValueError("Input changed during packaging")
    payloads["data/" + name] = content
payloads["SCIFACT-ATTRIBUTION.md"] = (project / "docs/DAY08-COMPARISON.md").read_bytes()
manifest = {
    "schema_version": 2,
    "job": "scifact-fixed-comparison-v1",
    "uv_version": "0.11.26",
    "python_profile": "3.11",
    "base64_files": ["source.tar.gz"],
    "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    "datasets": {role: ds.describe() for role, ds in datasets.items()},
}
args.destination.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for name, data in payloads.items():
        archive.writestr(
            name + ".b64" if name == "source.tar.gz" else name,
            base64.b64encode(data) if name == "source.tar.gz" else data,
        )
    archive.writestr("launcher-input.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
print(
    json.dumps(
        {
            "path": str(args.destination),
            "bytes": args.destination.stat().st_size,
            "sha256": hashlib.sha256(args.destination.read_bytes()).hexdigest(),
        },
        indent=2,
    )
)
