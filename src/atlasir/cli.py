"""AtlasIR installation checks, retrieval baselines and preserved run lifecycle."""

import argparse
import json
import platform
import sys
import tarfile
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

from atlasir.artifacts import (
    RunWriter,
    export_run,
    resolved_fixture_config,
    restore_run,
    verify_run,
)
from atlasir.data import load_beir


def check_fixture(content: bytes | None = None) -> dict[str, int]:
    """Check that the installed package contains a coherent synthetic fixture.

    This is an installation check, not a general BEIR loader or a relevance evaluation.
    """
    if content is None:
        content = files("atlasir").joinpath("fixtures", "smoke.json").read_bytes()
    fixture = json.loads(content)
    documents = {document["_id"] for document in fixture["corpus"]}
    queries = {query["_id"] for query in fixture["queries"]}
    if len(documents) != len(fixture["corpus"]) or len(queries) != len(fixture["queries"]):
        raise ValueError("Synthetic fixture contains duplicate IDs")
    if not documents or not queries or not fixture["qrels"]:
        raise ValueError("Synthetic fixture must contain documents, queries and judgments")
    for judgment in fixture["qrels"]:
        if judgment["query_id"] not in queries or judgment["doc_id"] not in documents:
            raise ValueError("Synthetic judgment references an unknown query or document")
    return {"documents": len(documents), "queries": len(queries), "qrels": len(fixture["qrels"])}


def main(argv: list[str] | None = None) -> int:
    """Run installation checks or the CPU-only artifact lifecycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('atlasir')}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("smoke", help="Validate the packaged synthetic CPU fixture")
    baseline = commands.add_parser("baseline", help="Preserve a development-only CPU BM25 run")
    baseline.add_argument("--data", type=Path, required=True)
    baseline.add_argument("--name", choices=["fixture", "scifact", "nfcorpus"], required=True)
    baseline.add_argument("--role", choices=["train", "validation"], required=True)
    baseline.add_argument("--selection", type=Path)
    baseline.add_argument("--config", type=Path, default=Path("configs/baselines/bm25.yaml"))
    baseline.add_argument("--root", type=Path, default=Path.cwd())
    baseline.add_argument("--source-root", type=Path, default=Path.cwd())
    baseline.add_argument("--run-id")
    recalculate = commands.add_parser(
        "reevaluate", help="Recalculate metrics from a saved baseline"
    )
    recalculate.add_argument("run", type=Path)
    reproduce = commands.add_parser(
        "reproduce", help="Rebuild trusted historical development source"
    )
    reproduce.add_argument("run", type=Path)
    reproduce.add_argument("destination", type=Path)
    reproduce.add_argument("--offline", action="store_true")
    run = commands.add_parser("run-fixture", help="Create and seal an E00 preservation fixture")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--source-root", type=Path, default=Path.cwd())
    run.add_argument("--config", type=Path, help="JSON overrides for the E00 defaults")
    run.add_argument("--run-id", help="Optional explicit ID; an existing ID always fails")
    verify = commands.add_parser("verify", help="Verify a sealed run's complete checksum inventory")
    verify.add_argument("run", type=Path)
    export = commands.add_parser("export", help="Export a verified run into a new bundle directory")
    export.add_argument("run", type=Path)
    export.add_argument("destination", type=Path)
    restore = commands.add_parser(
        "restore", help="Restore a bundle into a new directory and verify"
    )
    restore.add_argument("bundle", type=Path)
    restore.add_argument("destination", type=Path)
    for command in ("data-check", "make-splits", "evaluate"):
        data = commands.add_parser(command, help=f"{command} on local BEIR-format files")
        data.add_argument("--data", type=Path, required=True)
        data.add_argument(
            "--name", choices=["fixture", "scifact", "nfcorpus", "fiqa", "scidocs"], required=True
        )
        data.add_argument("--role", choices=["train", "validation", "final"], required=True)
        data.add_argument("--split", choices=["train", "dev", "test"])
        data.add_argument(
            "--selection", type=Path, help="Frozen source training-pool split manifest"
        )
        data.add_argument("--allow-final", action="store_true")
        if command == "make-splits":
            data.add_argument("--seed", type=int, default=17)
            data.add_argument("--output", type=Path, required=True)
        if command == "evaluate":
            data.add_argument("--rankings", type=Path, required=True)
            data.add_argument("--purpose", choices=["development", "final"], default="development")
            data.add_argument("--self-matches", choices=["exclude", "keep"], default="exclude")
    args = parser.parse_args(argv)

    try:
        if args.command == "baseline":
            from atlasir.baseline import read_config, run_baseline

            result = run_baseline(
                data=args.data,
                name=args.name,
                role=args.role,
                selection=args.selection,
                config=read_config(args.config),
                root=args.root,
                source_root=args.source_root,
                run_id=args.run_id,
                invocation=["atlasir", *(sys.argv[1:] if argv is None else argv)],
            )
        elif args.command == "reproduce":
            from atlasir.reproduction import reproduce_baseline

            result = reproduce_baseline(args.run, args.destination, offline=args.offline)
        elif args.command == "reevaluate":
            from atlasir.baseline import reevaluate

            result = reevaluate(args.run)
        elif args.command in {"data-check", "make-splits", "evaluate"}:
            selection = json.loads(args.selection.read_bytes()) if args.selection else None
            dataset = load_beir(
                args.data,
                name=args.name,
                role=args.role,
                split=args.split,
                selection=selection,
                allow_final=args.allow_final,
            )
            if args.command == "data-check":
                result = dataset.describe()
            elif args.command == "make-splits":
                from atlasir.splits import make_selection

                result = make_selection(dataset, seed=args.seed)
                # Exclusive creation protects the original membership from accidental replacement.
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as stream:
                    json.dump(result, stream, sort_keys=True, indent=2)
                    stream.write("\n")
            else:
                from atlasir.evaluation import ScoredDocument, evaluate

                def unique_object(pairs):
                    value = {}
                    for key, item in pairs:
                        if key in value:
                            raise ValueError("Rankings JSON contains a duplicate object key")
                        value[key] = item
                    return value

                raw = json.loads(args.rankings.read_bytes(), object_pairs_hook=unique_object)
                if not isinstance(raw, dict):
                    raise ValueError("Rankings must map query IDs to lists of scored documents")
                rankings = {}
                for query_id, rows in raw.items():
                    if not isinstance(rows, list) or any(
                        not isinstance(row, dict) or set(row) != {"doc_id", "score"} for row in rows
                    ):
                        raise ValueError("Each ranking must be a list of doc_id/score objects")
                    rankings[query_id] = [ScoredDocument(**row) for row in rows]
                result = evaluate(
                    dataset, rankings, purpose=args.purpose, self_matches=args.self_matches
                )
        elif args.command == "run-fixture":
            # A source snapshot from another checkout must not masquerade as the code running now.
            installed = Path(__file__).resolve().parent
            source = args.source_root.resolve() / "src" / "atlasir"
            installed_code = {
                path.relative_to(installed): path.read_bytes() for path in installed.rglob("*.py")
            }
            source_code = {
                path.relative_to(source): path.read_bytes() for path in source.rglob("*.py")
            }
            if installed_code != source_code:
                raise ValueError("source root does not match the installed AtlasIR code")
            config = resolved_fixture_config(
                json.loads(args.config.read_bytes()) if args.config else None
            )
            fixture = files("atlasir").joinpath("fixtures", "smoke.json").read_bytes()
            invocation = ["atlasir", *(sys.argv[1:] if argv is None else argv)]
            with RunWriter(
                args.root, args.source_root, config, fixture, args.run_id, invocation
            ) as writer:
                writer.write_json(
                    "results.json",
                    {
                        "scope": "artifact-lifecycle-fixture",
                        **check_fixture(fixture),
                    },
                )
                path = writer.seal()
            result = {
                "status": "success",
                "run_id": writer.run_id,
                "path": str(path),
                "spec_id": writer.manifest["spec_id"],
                "reportable": False,
            }
        elif args.command == "verify":
            manifest = verify_run(args.run)
            result = {"status": "verified", "run_id": manifest["run_id"]}
        elif args.command == "export":
            path = export_run(args.run, args.destination)
            result = {"status": "exported", "path": str(path)}
        elif args.command == "restore":
            manifest = restore_run(args.bundle, args.destination)
            result = {
                "status": "restored",
                "run_id": manifest["run_id"],
                "path": str(args.destination),
            }
        else:
            result = {
                "status": "ok",
                "scope": "installation-only",
                "fixture": "synthetic-v1",
                "version": version("atlasir"),
                "python": platform.python_version(),
                **check_fixture(),
            }
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"atlasir: {error}\n")

    print(json.dumps(result, sort_keys=True))
    return 0
