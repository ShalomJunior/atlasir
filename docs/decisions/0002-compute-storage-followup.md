# ADR 0002: Kaggle inventory and recommended artifact storage

- Date: 16 October 2026
- Status: compute inventory reported by user; storage recommendation awaiting account capacity check and restore verification
- Follows: [ADR 0001](0001-kickoff.md)

## New information

The user reports a 30-hour Kaggle GPU allowance and the dual NVIDIA T4 option. This records the available option, not a device inspection or a guarantee that every future session has that profile. Remaining quota at execution time, Python, RAM, driver/CUDA and output capacity still require session preflight.

Start with one device for the compact models. Only introduce independent embedding/pair-scoring shards across both devices after measuring throughput and quota behavior. Two visible GPUs do not automatically make the package use both; neither double useful speed nor quota accounting is assumed.

## Storage recommendation

Use an existing Google Drive account for a dedicated private `AtlasIR/archive/` folder if it has enough available space, initially targeting roughly 10 GB plus headroom. No storage subscription or new cloud infrastructure is required by this recommendation. Check actual account capacity before selecting it.

| Location | Role |
| --- | --- |
| Git repository and eventual GitHub remote | Code, lockfile, protocol, small experiment catalogs and DVC pointers |
| Kaggle versioned saved outputs | Preserved job output and an additional execution copy, with exact notebook version recorded |
| Private Google Drive archive | Durable independent copy of compressed run bundles and shared input objects, with checksum inventories |
| Local machine | Working source, small records and selectively restored artifacts; not the only full archive while disk space is constrained |

At first, use portable archives and a verified manual upload/download path. Keep shared data/model objects separate so they are uploaded once and referenced by many runs. Do not copy the full corpus/weights into every result bundle. Save completed shard exports during long sessions; only a completed, verified upload survives losing the notebook VM.

DVC remains responsible for artifact identities/pointers. Automatic remote transport can be configured after the basic restore works. The [official DVC Google Drive guide](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive) currently documents an authentication issue affecting its default Google app and a custom OAuth alternative. Direct archive upload is the simpler initial transport; a native DVC Drive remote is not considered configured or validated yet.

If Drive capacity is insufficient, use an existing external disk or another available durable store before considering paid storage. Do not treat two folders on the same local disk or `/tmp` as independent backups. Do not remove historical artifacts to accommodate a new run without a documented retention decision.

## Acceptance

Identify the actual folder/destination and available capacity, create one tiny sealed fixture bundle on Day 2, upload/copy it, download into a fresh workspace and verify checksums and metric-independent fixture content. Until this passes, the second-copy gate remains open. No account connection, upload, folder creation or external publication has been performed by this record.
