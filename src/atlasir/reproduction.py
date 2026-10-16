"""Rebuild a small baseline with its historical source and locked dependencies."""

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from atlasir.artifacts import _extract, verify_run


def compare_runs(original: Path, rebuilt: Path) -> dict[str, Any]:
    """Require matching inputs, metrics and ranked IDs; allow tiny score roundoff."""
    before, after = verify_run(original), verify_run(rebuilt)
    if before["schema_version"] != 2 or after["schema_version"] != 2:
        raise ValueError("Reproduction comparison requires baseline runs")
    for name in (
        "config.resolved.json",
        "dataset.json",
        "selection.json",
        "metrics.json",
        "per_query.json",
    ):
        if json.loads((original / name).read_bytes()) != json.loads((rebuilt / name).read_bytes()):
            raise ValueError(f"Reproduction differs: {name}")
    left = json.loads((original / "rankings.json").read_bytes())
    right = json.loads((rebuilt / "rankings.json").read_bytes())
    if left.keys() != right.keys():
        raise ValueError("Reproduction query coverage differs")
    exact = True
    for query, hits in left.items():
        other = right[query]
        if [h["doc_id"] for h in hits] != [h["doc_id"] for h in other]:
            raise ValueError("Reproduction ranked document IDs differ")
        for first, second in zip(hits, other, strict=True):
            if not math.isclose(first["score"], second["score"], rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError("Reproduction scores exceed roundoff tolerance")
            exact = exact and first["score"] == second["score"]
    return {
        "original_run_id": before["run_id"],
        "rebuilt_run_id": after["run_id"],
        "queries": len(left),
        "metrics_and_ranked_ids": "exact match",
        "scores_exact": exact,
        "score_rtol": 1e-6,
        "score_atol": 1e-7,
        "timings_compared": False,
    }


def reproduce_baseline(run: Path, destination: Path, *, offline: bool = False) -> dict[str, Any]:
    """Run only trusted local source; a checksum establishes identity, not trust."""
    run, destination = run.resolve(), destination.resolve()
    manifest = verify_run(run)
    if manifest["schema_version"] != 2:
        raise ValueError("Reproduction requires a baseline run")
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("Install the documented uv version before reproducing")
    descriptor = json.loads((run / "dataset.json").read_bytes())
    if descriptor["role"] not in {"train", "validation"}:
        raise ValueError("Only development baselines may be reproduced here")
    runtime = json.loads((run / "environment.json").read_bytes())
    python_version = ".".join(runtime["python"].split(".")[:2])
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / "source"
    source.mkdir()
    _extract(run / "source.tar.gz", source)
    data = destination / "data"
    (data / "qrels").mkdir(parents=True)
    for saved, original in (
        ("data.corpus.jsonl", "corpus.jsonl"),
        ("data.queries.jsonl", "queries.jsonl"),
        ("data.qrels.tsv", f"qrels/{descriptor['official_split']}.tsv"),
    ):
        shutil.copyfile(run / saved, data / original)
    config = destination / "config.json"
    shutil.copyfile(run / "config.resolved.json", config)
    selection = json.loads((run / "selection.json").read_bytes())
    command = [
        uv,
        "run",
        "--project",
        str(source),
        "--python",
        python_version,
        "--locked",
        "--no-dev",
    ]
    if offline:
        command.append("--offline")
    command += [
        "python",
        "-m",
        "atlasir",
        "baseline",
        "--data",
        str(data),
        "--name",
        descriptor["name"],
        "--role",
        descriptor["role"],
        "--config",
        str(config),
        "--source-root",
        str(source),
        "--root",
        str(destination / "output"),
    ]
    if selection is not None:
        selected = destination / "selection.json"
        shutil.copyfile(run / "selection.json", selected)
        command += ["--selection", str(selected)]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(destination / "environment")
    env["UV_PYTHON"] = python_version
    result = subprocess.run(
        command, cwd=source, env=env, check=True, capture_output=True, text=True
    )
    rebuilt = Path(json.loads(result.stdout)["path"])
    comparison = compare_runs(run, rebuilt)
    if verify_run(rebuilt)["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Rebuild source archive differs from the historical snapshot")
    comparison.update(
        {
            "historical_source_sha256": manifest["source_sha256"],
            "python_profile": python_version,
            "rebuilt_path": str(rebuilt),
            "scope": "same-machine clean-environment algorithm rebuild",
        }
    )
    with (destination / "comparison.json").open("x") as stream:
        json.dump(comparison, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return comparison
