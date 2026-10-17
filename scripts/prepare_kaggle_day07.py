"""Freeze a small source-training sample and archive-safe Day 7 source package."""

import argparse
import base64
import hashlib
import json
import zipfile
from pathlib import Path

from atlasir.artifacts import source_snapshot
from atlasir.data import load_beir
from atlasir.day07 import prepare_pilot_inputs, resolved_shard_pilot_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--destination", type=Path, required=True)
parser.add_argument("--data", type=Path, default=Path(".scratch/data/scifact"))
parser.add_argument(
    "--selection", type=Path, default=Path("experiments/splits/scifact-seed17.json")
)
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
config = resolved_shard_pilot_config(
    json.loads((project / "configs/dense/day07-pilot.json").read_bytes())
)
selection = json.loads(args.selection.read_bytes())
dataset = load_beir(args.data, name="scifact", role="train", selection=selection)
inputs = (
    json.dumps(prepare_pilot_inputs(dataset, config), sort_keys=True, indent=2) + "\n"
).encode()
source, _ = source_snapshot(project)
payloads = {"source.tar.gz.b64": base64.b64encode(source), "pilot-inputs.json": inputs}
manifest = {
    "schema_version": 2,
    "job": "day07-shards-index-pilot",
    "uv_version": "0.11.26",
    "python_profile": "3.11",
    "base64_files": ["source.tar.gz"],
    "sha256": {
        "source.tar.gz": hashlib.sha256(source).hexdigest(),
        "pilot-inputs.json": hashlib.sha256(inputs).hexdigest(),
    },
}
args.destination.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for name, data in payloads.items():
        archive.writestr(name, data)
    archive.writestr("launcher-input.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
print(
    json.dumps(
        {
            "path": str(args.destination),
            "bytes": args.destination.stat().st_size,
            "zip_sha256": hashlib.sha256(args.destination.read_bytes()).hexdigest(),
            "sample_documents": len(json.loads(inputs)["documents"]),
            "sample_queries": len(json.loads(inputs)["queries"]),
            **manifest,
        },
        indent=2,
    )
)
