"""Short locked CPU baseline reproduction; CUDA probing runs separately in the kernel."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from atlasir.artifacts import export_run, restore_run
from atlasir.baseline import reevaluate, run_baseline
from atlasir.reproduction import compare_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    metadata = json.loads((args.input / "launcher-input.json").read_bytes())
    for name, digest in metadata["sha256"].items():
        path = args.input / name
        if path.resolve().is_relative_to(args.input.resolve()) is False:
            raise ValueError("Input path escapes the input directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input hash mismatch: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    restored = args.output / "restored-reference"
    restore_run(args.input / "reference.bundle", restored)
    reevaluate(restored)
    descriptor = json.loads((restored / "dataset.json").read_bytes())
    data = args.output / "data"
    (data / "qrels").mkdir(parents=True)
    for saved, original in (
        ("data.corpus.jsonl", "corpus.jsonl"),
        ("data.queries.jsonl", "queries.jsonl"),
        ("data.qrels.tsv", f"qrels/{descriptor['official_split']}.tsv"),
    ):
        shutil.copyfile(restored / saved, data / original)
    selection = json.loads((restored / "selection.json").read_bytes())
    result = run_baseline(
        data=data,
        name=descriptor["name"],
        role=descriptor["role"],
        selection=restored / "selection.json" if selection is not None else None,
        config=json.loads((restored / "config.resolved.json").read_bytes()),
        root=args.output,
        source_root=source,
        invocation=[
            "python",
            "scripts/kaggle_day05.py",
            "--input",
            str(args.input),
            "--output",
            str(args.output),
        ],
    )
    rebuilt = Path(result["path"])
    comparison = compare_runs(restored, rebuilt)
    comparison["launcher_source_sha256"] = metadata["sha256"]["source.tar.gz"]
    comparison["scope"] = "current pinned launcher versus Day 4 reference; CPU quality only"
    exports = args.output / "exports"
    exports.mkdir()
    export_run(rebuilt, exports / f"{result['run_id']}.bundle")
    shutil.copyfile(args.input / "SCIFACT-ATTRIBUTION.md", exports / "SCIFACT-ATTRIBUTION.md")
    with (exports / "comparison.json").open("x") as stream:
        json.dump(comparison, stream, sort_keys=True, indent=2)
        stream.write("\n")
    archive = args.output / "day05-baseline-export.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(exports.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(exports))
    print(
        json.dumps(
            {
                "status": "passed",
                "run_id": result["run_id"],
                "download": str(archive),
                "comparison": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
