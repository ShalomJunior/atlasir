# Reproducibility and experiment preservation

Status: the [Day 2 E00 lifecycle](DAY02-RUNS.md) is implemented with JSON configuration, small self-contained bundles and local export/restore. The rest of this document describes the growing system. General research-run schemas, DVC/MLflow integration, large objects and the `reproduce`/`compare` CLI examples below are future work. Independent cloud backup remains unverified.

## What we need

This work belongs to experiment management and reproducible ML engineering, which are parts of MLOps. AtlasIR needs provenance, repeatable execution, artifact storage and recovery. It does not need a full production MLOps platform to begin.

An experiment is more than a metrics row. Preserve **code + environment + resolved configuration + input identities + outputs + execution context**. Git alone does not preserve datasets or weights; MLflow alone does not reconstruct all inputs; a notebook alone does not preserve its virtual machine.

We distinguish three promises:

| Promise | What someone can do |
| --- | --- |
| Inspect | Restore the exact saved rankings, metrics, logs and configuration |
| Recompute | Recalculate metrics and figures from the saved rankings and qrels, without neural inference |
| Rerun | Execute historical code against pinned inputs and compare new outputs |

Identical bytes are expected for archived files, not for newly measured latency. CPU/GPU kernels, library versions and hardware can affect reruns. Define ranking/metric tolerances on a reference profile during the pilot; report differences instead of silently accepting them. [PyTorch documents the limits of deterministic reproduction across releases and platforms](https://docs.pytorch.org/docs/main/notes/randomness.html).

## Tool responsibilities

| Tool | Responsibility | Canonical record? |
| --- | --- | --- |
| Git | Source code, lockfiles, protocol, config recipes, split manifests, small catalog entries and DVC pointers | Yes, for small versioned files |
| uv | Recreate the locked project environment | Lockfile plus interpreter/profile recorded |
| DVC | Version and transfer large data, model snapshots, indexes and sealed result bundles | Yes, for artifact pointers/content |
| MLflow | Search, compare and visualize run parameters, tags and metrics on a local durable host | No; rebuildable from bundles |
| Kaggle/Colab | Execute versioned jobs | No; temporary compute and one possible saved-output copy |

[DVC stores lightweight pointers in Git and larger objects in its cache/storage](https://doc.dvc.org/command-reference/add). [MLflow supports tracking parameters, metrics and artifacts](https://mlflow.org/docs/latest/ml/tracking/). We use both with distinct responsibilities: MLflow never becomes the only place an outcome exists.

Do not run a tracking server or the only SQLite database inside an ephemeral notebook. A notebook produces portable files; the durable coordinator imports them into local MLflow after validation. Tracking UI failure must not discard or block an otherwise valid experiment.

## Planned repository and storage layout

```text
atlasir/
  pyproject.toml
  uv.lock
  .python-version
  src/atlasir/
    data/ retrieval/ reranking/ routing/ evaluation/
    experiments/ artifacts/ cli.py
  configs/
    baselines/ policies/ suites/ environments/
  experiments/
    proposals/                 # questions and predeclared comparisons
    catalog/<run_id>.yaml      # small sealed-run reference, in Git
    protocols/                 # frozen protocol versions
    splits/                    # query-ID lists and hashes, in Git if small
  artifacts/
    objects/<object_id>/       # shared raw/processed data, models, indexes
    objects/<object_id>.dvc    # Git-tracked pointer
    runs/<run_id>/             # immutable sealed result bundle
    runs/<run_id>.dvc          # Git-tracked pointer
  .scratch/                    # incomplete local work, ignored
  notebooks/                  # thin Kaggle and Colab launchers
  tests/fixtures/
  reports/                    # source plus selected generated release assets
  docs/
  .github/workflows/
```

This is a target layout, not a requirement to create empty directories immediately. DVC ignores individual large targets; do **not** ignore the entire `artifacts/` parent and accidentally hide its `.dvc` pointers. Test this explicitly.

Each shared object contains a descriptor identifying its role and dependencies. Object categories include downloaded archives, prepared corpora, model/tokenizer snapshots, embeddings, indexes and trained policies. A run references these objects instead of copying a full dataset and all weights into every run directory.

## Identity: specification, run and attempt

1. A **specification ID** hashes canonical resolved configuration, source revision, dependency lock/profile, protocol version, seed and input object IDs. It represents a requested computation. For timing experiments, include hardware/session profile and benchmark settings.
2. A **run ID** is human-readable and collision resistant, for example `20261016T140000Z-e03-a1b2c3d4-7f2e91`. Use UTC, a short spec hash and random suffix, with full identifiers retained in metadata.
3. A repetition creates a new run ID with the same specification where appropriate and a `repeats` or `parent_run_id` reference. A corrected implementation has a new specification.

Identical configuration must never silently reuse a result run directory. Input caches may be reused only after dependency/hash checks. An existing final path causes a hard error. A new run is created even when a previous attempt failed. Never use names such as `results/latest.json` as scientific evidence; a convenience alias may point to an immutable run.

## Required run bundle

```text
<run_id>/
  manifest.json              # full provenance and outcome status
  config.resolved.yaml       # defaults resolved, no external config inheritance
  environment.json           # interpreter, packages, device, driver, threads
  inputs.json                # object IDs and exact source/model/data revisions
  metrics.json               # aggregate results and missing/failure counts
  per_query.parquet          # relevance metrics, selected actions, costs/status
  rankings.trec.gz           # method/query/document/rank/score
  timing.parquet             # raw query/block/stage timing samples, if timed
  events.jsonl               # structured execution/failure events
  stdout.log
  artifact_checksums.json    # hashes/sizes, excluding itself and COMPLETE.json
  COMPLETE.json              # final completion marker/checksum reference
```

Not every run needs neural timing or rankings: the manifest declares its run type and required outputs. Quality, timing, index-build, policy-training and analysis runs may be separate, linked records. A report/plot-generation run stores parent run IDs, aggregation code revision and table/figure outputs.

Minimum manifest fields:

| Category | Fields |
| --- | --- |
| Identity | schema version, run/spec ID, family E00–E12, purpose, protocol ID, UTC start/end |
| Code | full Git SHA, clean/dirty status, source archive object ID, launcher revision |
| Config | resolved config hash, original config path, full invocation and all overrides |
| Inputs | corpus/query/qrels/split hashes, model+tokenizer revisions and weight hashes, index identity |
| Environment | Python, uv/lock hash, actual installed packages, OS, PyTorch/CUDA/driver, CPU/GPU/RAM, precision/threads |
| Randomness | seed, random libraries, deterministic flags, known nondeterminism |
| Dependencies | parent run IDs and shared-object descriptors |
| Outcome | success/failed/interrupted, failure counts, output hashes, warnings |
| Compute | platform, notebook slug/version or export identity, session ID, elapsed GPU session time |

Use typed/validated metadata schemas and explicit schema versions. Future readers either support an older schema or run its historical parser. A migration creates a derived artifact with a parent reference; it never rewrites the original.

## Write and seal lifecycle

1. Validate the config and inputs before spending GPU time. Record the complete command, environment and current source revision.
2. Require a clean source revision for reportable runs. Exploratory dirty work must preserve a source snapshot/patch and is tagged exploratory. Do not rely on an unrecorded notebook edit.
3. Reserve a new ID and exclusively create `.scratch/<run_id>/`. Write the initial manifest before model loading.
4. Write shard results atomically: temporary file, flush/close, checksum, then rename. Append checkpoint metadata only after a shard is valid.
5. Validate query coverage, duplicate IDs, finite metrics, dependencies and required outputs. Recompute aggregates from per-query records.
6. Write final checksums and completion marker last. Atomically rename to `artifacts/runs/<run_id>/` on the same filesystem. A cross-filesystem transfer is separately verified, not assumed atomic.
7. On the durable coordinator, DVC-track the sealed directory and append its small catalog record. Preserve the source archive and all referenced shared objects.
8. Transfer missing content to the backup destination, verify its presence, then perform a targeted fresh restore. Only then mark the catalog record `archived_and_verified`.
9. Import metrics/tags into local MLflow with the external run ID. Import is idempotent and links to the DVC record; do not create duplicate comparisons on repeated import.

The run's internal outcome remains immutable. Later correction, invalidation and archive-verification events are separate append-only catalog/event records referencing it. A failed/interrupted attempt gets a final failure record and its valid shards preserved; it has no successful completion marker.

## Cache identity and invalidation

Shared objects are keyed by effective input/transform identity, not by a friendly dataset name alone.

| Object | Must include |
| --- | --- |
| Prepared dataset | raw hashes, preparation-code revision/hash, field composition, normalization and split rules |
| Document embeddings | ordered document-ID/text hashes, model+tokenizer revision, prefixes, truncation, pooling, dtype and relevant library versions |
| Dense index | embedding hash, exact/ANN algorithm, index parameters, ID-map hash, FAISS version |
| Lexical index | prepared corpus hash, analyzer/stopwords/stemmer, BM25 variant/parameters and implementation version |
| Pair-score cache | query/document text hashes, ordered pair IDs, model/tokenizer revision, max lengths, precision and scoring implementation |
| Trained router | training feature/label/input hashes, source IDs, split, preprocessing, loss, hyperparameters and seed |

Raw content uses SHA-256 checksums; DVC retains its own content identifiers. Do not pretend DVC checksums automatically cover unlisted external model/config dependencies. The planned object ID is a hash of the descriptor; verify its files separately. If relevant code changes, invalidate the affected derived object. Avoid a bespoke general workflow engine: small explicit stage functions and dependency descriptors suffice.

For an offline experiment, cached cross-encoder scores may be reused for identical query/document pairs. Save how the cache was constructed. A latency run disables outcome caches and actually performs query encoding, retrieval and selected reranking. Loading a warm model/index is allowed and labeled.

## DVC usage policy

Use DVC data tracking for sealed run directories and immutable shared objects initially. Example after implementation and successful sealing:

```sh
dvc add artifacts/runs/<run_id>
```

This creates the adjacent small `.dvc` pointer and ignores the large directory. Commit the pointer, small catalog record and relevant ignore rules in the normal reviewable workflow. Preserve the source Git revision referenced by the run. Transfer artifacts separately; Git push alone does not copy them.

DVC pipelines may later cache preparation/indexing stages. They must not write into old sealed run paths. In particular, [DVC can delete stage outputs before rerunning them](https://doc.dvc.org/user-guide/pipelines/running-pipelines). A mutable pipeline workspace is not an archival policy. We do not depend on DVC's hidden experiment refs for the only history; reportable runs have ordinary Git-tracked catalog entries and artifact pointers.

Never garbage-collect the only copy of a dependency. Before any storage cleanup, enumerate all retained run references, verify backup restoration and produce a removal list. The initial project does not automate deletion of historical successful runs.

## Storage and backup

Baseline with no new subscription:

- A local durable project archive is the coordinator, not notebook scratch storage.
- A second copy resides in an existing personal drive, external disk or suitable DVC remote selected during setup. Record capacity, access method and public/private status.
- Kaggle saved output is an additional execution record where available. Record the notebook version and exported files; never assume a live session directory is backed up.
- Model/data redistribution follows their terms. Restricted objects stay in an appropriately private archive; public catalogs can expose hashes and acquisition recipes without exposing restricted text.

Reserve roughly 10 GB as an initial planning envelope, then measure actual size; neither model nor notebook output limits are assumed. Budget uncompressed working copies, DVC cache and backup duplication. Archive shared objects once, not per run. Use compressed ranking files and Parquet instead of millions of small files. Keep a disk-space preflight check and stop before incomplete exports exhaust storage.

Every meaningful experiment retains its compact bundle, source recipe and dependency references, including failures. Large ephemeral scratch tensors may be discarded after sealing because they are not declared required artifacts. Official release runs keep the full required dependency closure. Any later retention change is documented explicitly; reproducibility depends on retained bytes or an available verified source.

## Restore and rerun acceptance targets

The following are planned interfaces to build incrementally:

```sh
uv sync --locked
uv run atlasir reproduce --run-id <historical_id> --mode metrics
uv run atlasir reproduce --run-id <historical_id> --mode full
uv run atlasir compare --runs <old_id> <new_id>
```

`metrics` restores rankings, qrels and historical evaluation settings, then creates a new derived run. `full` locates the historical source snapshot/commit in an isolated checkout, uses its lockfile and declared environment profile, restores exact input objects and executes its saved invocation. It must not run old configs through arbitrarily changed current code. A different device creates a new hardware profile and separate latency comparison.

Weekly restore drill: use a clean temporary checkout and empty artifact cache, restore one earlier run plus its dependency closure, validate hashes, recompute metrics and compare with the record. Disable reliance on the original working directories. At G6/G8, also rebuild one baseline and restore the report's complete evidence set. A cache hit in the original notebook is not proof of recovery.

## Tests that protect the claim

Write focused tests for fail-on-overwrite, atomic completion, interrupted shard resume, stale input/cache rejection, evaluator correctness, split disjointness, TREC round-trip ordering and artifact restoration. Use small fixed fixtures; avoid mocks that merely repeat the implementation. CI does not download full corpora or weights. A tiny integration run must exercise the whole data → retrieval → evaluation → seal → restore path.

## Daily operator checklist

Before a run: proposal/config, source identity, correct split, input hashes, estimated compute, free space and output destination.

After a run: coverage and status, hashes, sealed bundle, durable export, second-copy verification, catalog entry and short interpretation. Never repair a completed result file by hand. A wrong metric requires a derived reevaluation run; a wrong ranking requires a corrected retrieval run.
