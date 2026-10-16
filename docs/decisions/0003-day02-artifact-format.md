# ADR 0003: Initial run bundle format

- Date: 16 October 2026
- Status: adopted for E00; research-method schemas will be introduced with their implementations
- Scope: Day 2 preservation lifecycle

## Decisions

Use a small standard-library implementation for the first preservation experiment. E00 validates installation-fixture references and preserves them; it does not retrieve documents, compute relevance metrics or train a model.

- Store validated resolved E00 configuration as JSON (`config.resolved.json`). The planned YAML configuration interface is deferred until baseline configuration needs it. JSON avoids a new parser dependency on this CPU fixture; unknown keys and invalid types fail.
- Keep the schema explicitly limited to E00 (`day02-fixture-v1`). Do not imply that general quality/timing/training-run schemas or GPU environment capture already exist.
- Use separate specification hashes and unique run IDs. Preserve timestamps, source snapshot, Git commit/dirty state, package versions, invocation and input hashes. Seed is recorded for future uniformity; this fixture performs no randomized computation.
- Reject a CLI source root whose Python package differs from the executing installed package. Include the source archive and lockfile in the bundle, because current work is uncommitted and a Git SHA alone would miss it.
- Embed the small source snapshot and synthetic input in E00 so that the export is self-contained. Shared model/data object storage remains the planned approach for large experiments.
- Write outputs exclusively and retain permanent run-ID reservations. A failed attempt keeps its scratch files with failed/interrupted status; it never gets a successful completion marker or reuses that ID.
- Seal only after required outputs exist and all file hashes are recorded. Publish on the same filesystem. The API refuses changes to a closed writer. This is application-level preservation, not write-once storage: filesystem edits remain possible and verification detects changes.
- Append a small Git-trackable catalog record only after local sealing. Its state is `local-sealed`, never `archived_and_verified` just because a local copy exists.
- Export to a new `.bundle/` directory containing `run.tar.gz` and `archive.json` (outer SHA-256 and size). Restore validates the archive, every file inventory entry and provenance before publishing a new directory.
- Restore copies data only; it never executes archived code. Historical execution and metric reevaluation remain future commands. An archive checksum detects corruption; it is not an authenticity signature against an attacker who can replace both archive and metadata.

## Limits and recovery

The initial transfer implementation is intentionally for small fixtures, with a 128 MiB uncompressed restore limit and 10,000-entry limit. Do not use it for large datasets or weights. Reject traversal paths, duplicate members, symlinks and special files. Large-object transport will require streaming hashes, shard descriptors and appropriately sized limits.

Normal Python exceptions and KeyboardInterrupt preserve failed/interrupted manifests. Abrupt process/VM loss can leave running metadata, incomplete temporary files or a transfer lock. These are not completed runs and must not be published as successes. Inspect abandoned work before removing a stale transfer lock; use a new run ID for another execution attempt. Automated shard resume is not implemented by this milestone.

No additional library, DVC initialization or Drive authentication is introduced on Day 2. Canonical small files and archives are ready for the DVC coordinator later; no automatic DVC/MLflow tracking is claimed. DVC pointer files remain eligible for Git tracking next to ignored result directories.

## Acceptance

Tests must demonstrate fail-on-overwrite (including active reservations), rejection of missing outputs, preservation of failures, source/config identity, clean restore without the original run/source directory, and rejection of modified/missing/extra files and unsafe archives. A separate local CLI drill produces the reviewable E00 record. A successful Drive upload/download is still required before claiming independent backup or allowing long experiments.
