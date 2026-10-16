# Day 2 progress

Date: 16 October 2026. Status: local artifact lifecycle implemented and verified; independent cloud-backup gate still open.

## Implemented

- E00 config resolution/validation with no new runtime dependencies.
- Separate specification hash and unique run ID; exclusive output creation and permanent ID reservations.
- Saved source snapshot (including uv.lock), Git state, installed package versions, CPU environment, invocation and input hash.
- Required-output validation, SHA-256 inventory, success marker and same-filesystem publication.
- Closed-writer protection; failed/interrupted attempts retain scratch content without successful completion.
- Small append-only local catalog records, eligible for Git tracking.
- Verified portable export and restore into fresh destinations, rejecting overwrite and unsafe archive entries.

See [commands and the Drive drill](../DAY02-RUNS.md), [format decisions and limits](../decisions/0003-day02-artifact-format.md) and the [E00 proposal](../../experiments/proposals/e00-preservation.md).

## Local experiment record

- Run: `20261016t180110z-e00-faf7a630-40b146d8942d`
- Specification: `faf7a630c05e7ef3521b659658b720a297b05946411bab61d92892f41da0bc75`
- Catalog: [run record](../../experiments/catalog/20261016t180110z-e00-faf7a630-40b146d8942d.json).
- Final run: `artifacts/runs/20261016t180110z-e00-faf7a630-40b146d8942d/`.
- Export: `.scratch/exports/20261016t180110z-e00-faf7a630-40b146d8942d.bundle/`.
- Local restore: `.scratch/restore-drill/20261016t180110z-e00-faf7a630-40b146d8942d/`.
- Export size: 59,621 bytes across `run.tar.gz` and `archive.json`.
- Observed: all ten run files matched the local restored copies byte-for-byte.
- Observed: repeating this exact ID failed and the original manifest hash was unchanged.

This is an artifact lifecycle fixture, not a retrieval experiment. It has no relevance scores, measured search latency or GPU work. Its source was dirty and is embedded explicitly; it is marked non-reportable. Later documentation changes do not modify the captured snapshot or its original specification ID.

## Verification

Automated tests cover completed/active ID collisions, attempted overwrite, unsealed and missing-output cases, failures/interruptions, input/config/source identity, changed/missing/additional files, export/restore collisions, corrupted archives, traversal/symlink/duplicate archive entries, secret/generated-file exclusion and CLI execution. The clean restore test deletes its temporary original run and source before restoring, so it cannot depend on them.

The local CLI drill additionally created the reviewable record above. Final checks passed:

| Check | Result |
| --- | --- |
| Ruff lint and formatting | Passed |
| pytest | 28 tests passed |
| Source distribution and wheel build | Passed |
| Independent wheel environment | Fixture creation, export, restore and verification passed outside the repository, with only the built package installed |
| Local CLI fixture | Ten files restored byte-for-byte; existing ID rejected without modifying original manifest |

The GitHub workflow includes these tests but has not been run remotely.

## Outstanding boundary

The local source, run, export and restored directory share a disk. This validates the format but is not independent backup. No Drive folder is selected/connected and no remote transfer or restore has occurred. Follow the Drive drill using this bundle before marking it archived-and-verified or starting long experiments. A local metadata event records that distinction.

DVC initialization, large-object sharing, MLflow import, shard resume and historical `reproduce` commands remain future work. JSON is the implemented E00 configuration format; general YAML-backed experiment configs come with the baselines. No scientific scope was dropped.

## Next step

Day 3 can implement the BEIR loader and metric fixtures using small local data. The independent-storage gate still applies before full experiment/GPU runs. Source/target split protections and hand-calculated metric cases are the next technical milestone.
