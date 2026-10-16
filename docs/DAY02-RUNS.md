# Running and preserving the Day 2 fixture

These commands are implemented. Run them from the repository after `uv sync --locked`. They use CPU only and download no models/datasets. In the agent sandbox use `UV_CACHE_DIR=/tmp/atlasir-uv-cache` if its home cache is read-only.

## Create an experiment

```sh
uv run --locked atlasir run-fixture --config configs/fixtures/e00.json
```

The command prints JSON including the new `run_id`, `spec_id` and final directory. It snapshots the installed project's source plus lockfile, validates the packaged synthetic input and writes:

```text
artifacts/runs/<run_id>/
  manifest.json
  config.resolved.json
  environment.json
  inputs.json
  source.tar.gz
  fixture.json
  results.json
  events.jsonl
  artifact_checksums.json
  COMPLETE.json
experiments/catalog/<run_id>.json
```

Repeated configurations receive new run IDs. To test an explicit name, use `--run-id e00-my-first-check`; a second attempt with that same name fails. `--root` changes the working/output root; `--source-root` must identify source matching the installed package. This fixture is always marked non-reportable research evidence.

## Verify, export and restore

Replace `<run_id>` with the printed ID. Destinations must not already exist.

```sh
uv run --locked atlasir verify artifacts/runs/<run_id>
uv run --locked atlasir export artifacts/runs/<run_id> .scratch/<run_id>.bundle
uv run --locked atlasir restore .scratch/<run_id>.bundle .scratch/restored-<run_id>
uv run --locked atlasir verify .scratch/restored-<run_id>
```

The export contains `run.tar.gz` and `archive.json`; preserve them together. Neither export nor restore overwrites an existing destination. A failed transfer leaves no published destination under normal exception handling. Verification detects changed, missing or additional files in a completed run.

Only `.scratch` and the individual artifact directories are ignored by Git. The small catalog remains visible; future adjacent `.dvc` pointers are not ignored. No code commit or remote upload happens automatically.

## Google Drive restore drill

1. Select a private `AtlasIR/archive/` folder in an existing Drive account and check available capacity.
2. Upload the **entire** `<run_id>.bundle` directory, or both files within a uniquely named equivalent folder. Check that the upload completed.
3. Download into a different local directory; do not use the original export as the verification input.
4. Run `atlasir restore <downloaded-bundle-directory> <new-restore-directory>`, then `atlasir verify <new-restore-directory>`.
5. Record the exact run ID, destination, UTC verification time and result in a new catalog/progress event. Do not modify the sealed run to add backup status.

A local export and local restore test the format, but two directories on this laptop are not independent backups. The long-experiment gate stays closed until this independent-copy check passes. No Drive credentials should appear in project files or chat.

## Implemented versus later

Implemented: exclusive IDs/output creation, resolved fixture config, source/environment/input capture, checksums and completion, failed-attempt preservation, local catalog, portable export and verified restore.

Later: BEIR evaluation, general experiment schemas, YAML composition, large shared-object/shard management, automated cloud transfers, DVC/MLflow integration, power-loss recovery, `reproduce` and `compare` commands. `restore` preserves the historical bytes; it does not rerun the experiment.
