# Kaggle and Colab execution workflow

Designed for the user's reported Kaggle allowance of about 30 GPU hours/week, with occasional Colab use. Check available quota and the assigned device each week; platform policies are not a guaranteed hardware reservation.

## Roles

First execution milestones: Day 4 runs the small SciFact BM25 baseline locally on CPU. Day 5 introduces a short Kaggle environment/device compatibility check and locked baseline reproduction. Days 6–8 start embeddings; Week 3 starts reranking. The independent backup/restore gate is still required before long GPU jobs. Actual Kaggle execution is not implied by local tests.

Use the local workspace for coding, documentation, Git, small CPU checks, durable artifact ingestion and MLflow. Kaggle supplies batch embedding/reranking and complete online benchmark execution. Colab is a fallback or a separately labeled portability experiment.

If a durable local machine is unavailable for part of the work, export to an existing persistent drive instead. The archive contract stays the same. Choose the destination and verify a restore on day 2; this is required before a long job.

The [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) describes saved notebook outputs and notebook versions. Its “Save & Run All” executes a separate run, so it must get a new run/session identity. Consult the current UI and documentation when setting up saving; do not assume saving notebook text saved all artifacts. The [Kaggle GPU usage guide](https://www.kaggle.com/docs/efficient-gpu-usage) describes quota monitoring. Colab resources and hardware vary, and its virtual machines expire; see the [official FAQ](https://research.google.com/colaboratory/faq.html).

## Thin launcher notebook

Keep one Kaggle and one Colab launcher under version control. Each contains setup, job selection, preflight, execution and export cells. Authoritative algorithms live in the package.

The first [Kaggle launcher](../notebooks/kaggle-day05.ipynb) and [Day 5 instructions](DAY05-KAGGLE.md) are implemented. It probes kernel CUDA separately from the isolated CPU package runtime; future neural-model environments need their own locked CUDA checks. The Colab launcher remains later work.

The launcher must:

1. Select an explicit Git commit or immutable source archive; never run a moving branch for reportable work.
2. Bootstrap the pinned uv tool, supported Python interpreter and isolated locked environment. Run the CLI in that environment rather than assuming the notebook kernel's global packages match it. Validate the selected torch/CUDA wheels with a small computation.
3. Resolve models/tokenizers to full revision IDs, download or mount versioned input objects, and verify file hashes. Internet-disabled jobs must have all inputs attached in advance.
4. Print a short nonsecret preflight: exact code, spec, split, device, available disk, package versions and output path.
5. Call the same package entry point used locally, with a fully resolved configuration. Never change preprocessing in an unrecorded cell.
6. Write artifacts below a dedicated output root, such as `/kaggle/working/atlasir-output/`, and temporary downloads separately. Record actual platform paths in the environment, not in portable experiment definitions.
7. Export the sealed run and new shared objects with a checksum inventory. Download/copy them to the coordinator and verify them before ending the session.

The durable coordinator DVC-tracks exported artifacts and imports MLflow records. This avoids requiring credentials for a live tracking service inside every notebook. If a persistent remote is later configured, record it as a replaceable storage transport. Never save tokens in notebooks, configs, logs or exported environment dumps; environment capture uses an allowlist.

## Session structure and interruption recovery

Target 1–3 hour jobs until measurements justify longer ones. Use corpus and pair-scoring shards sized for roughly 10–20 minutes after a pilot. Those are operational targets, not assumed platform limits.

Each shard is keyed by immutable input identities plus shard number and saves a checksum and completion marker. Embeddings include document IDs; pair scores include query/document IDs. A resume verifies descriptors and completed shards, then processes missing work. It never appends a different model/config to an old shard set.

A session checkpoint on temporary disk survives a Python restart only if the disk survives. To survive VM loss, periodically copy/export completed shards to the selected durable destination and verify transfer. If that cannot be automated reliably, shorten jobs and save output between them. State the possible unexported-work loss window.

On interrupted execution, preserve the attempt's manifest and valid exported shards. Start a new attempt ID with the parent attempt and reused-shard references. Resume quality computations across devices only with compatible settings, checking numerical differences. Do not merge timing samples from different hardware/session profiles into a single run.

Set a soft stop before the currently displayed runtime/quota limit, flush complete shards, export and validate. Handle OOM by starting a new configuration/attempt with a smaller declared batch size; do not silently change the timed benchmark settings mid-block. If batch-size adaptation is part of a policy, record and evaluate it explicitly.

## Estimated GPU allocation

These are planning envelopes to revise after day 7's pilot, not runtime predictions or required consumption.

| Week | Work | Initial GPU envelope | Human effort |
| --- | --- | --- | --- |
| 1 | CPU baseline, environment and GPU compatibility check | 0–2 h | 30 h |
| 2 | Source embeddings and dense/hybrid pilots | 3–8 h | 30 h |
| 3 | Source reranking scores and repeated cost pilots | 6–15 h | 30 h |
| 4 | Selective-policy checks; reuse offline scores | 2–6 h | 30 h |
| 5 | Full-policy validation, cost measurement and ablations | 3–10 h | 30 h |
| 6 | Final corpora embedding, all-method scoring, live timing | 10–22 h | 30 h |
| 7 | Targeted verification/recovery, no broad new sweeps | 0–4 h | 30 h |
| 8 | Reproduction and live demo checks | 0–4 h | 30 h |

Keep room below the reported weekly ceiling for interrupted sessions, duplicate Save & Run All execution and measured overruns. Check whether attached GPUs are allocated together and how Kaggle accounts quota; do not assume two devices give twice the useful free capacity. One GPU is enough for the core design. Do not require Colab to make the schedule work.

For a pilot throughput of `r` pairs/second and `P` unique query/document pairs, estimate scoring time as `P / r`, then include tokenization, I/O, initialization, exporting and a contingency margin. Estimate embeddings from actual document lengths and measured throughput. Predict full-run size from the pilot before starting it.

For offline quality sweeps, score the union of required pairs once for the same checkpoint/settings. This can derive several fixed depth choices without repeated neural scoring. Live timing must still run each selected method as actually deployed. Turn off GPU sessions when doing reading, statistics, CPU-only training or report writing.

## Canonical timing profile

Select the available Kaggle device/profile during pilots and use it consistently within comparison blocks. Do not promise a particular GPU before assignment. Save CPU model/count, RAM, GPU model/count/VRAM, driver, CUDA/runtime, package versions, precision, BLAS/OpenMP/torch threads, device placement and any concurrent workloads you control.

Reference protocol:

- One query at a time, concurrency 1, query batch size 1. Cross-encoder document-pair batching is fixed separately and identical across comparable methods.
- All required models and indexes already loaded. Clear outcome caches, not warm model/index residency. Record startup time separately.
- Use 20 deterministic warm-up source queries covering pipeline branches, excluded from measured samples. No ground-truth-based warm-up selection.
- Run five measured blocks over the full evaluation query set if the throughput pilot fits the budget. Save seeded query order; rotate method order across blocks. Use identical queries and hardware within each block.
- Use a monotonic high-resolution timer and GPU synchronization around timed stages. Exclude metric calculation/file export from query service latency and measure them separately.
- Save preprocessing, query encoding, lexical retrieval, dense retrieval, fusion, routing features, router inference, reranking, assembly and total time. Stage totals should explain the full elapsed time; asynchronous work must not disappear.
- Report mean/p50/p95 over raw query samples with block-aware uncertainty, plus per-dataset values. Use actual end-to-end times rather than sums of independently measured medians.
- For small query sets, repeated runs improve timing estimates, not statistical power of relevance comparisons. Inspect drift/contended blocks and disclose exclusions under a frozen validity rule.

If a complete five-block benchmark exceeds budget, preregister a deterministic hash-based latency subset of at most 200 queries/dataset at G5, then measure all methods on that same subset. Report sampled latency as sampled latency. Freeze the fallback rule and subset IDs before final outcomes are viewed; do not choose the fastest queries.

If the reference device becomes unavailable, run all comparators on the replacement device and publish a separate profile. A policy's source-calibrated budget may fail on it; evaluate that as transfer. Any recalibration uses only allowed source validation data and receives a new run/protocol record.

## Memory, indexing and cost reporting

Record index construction wall time, index bytes, model bytes, peak process RSS, peak CUDA allocated/reserved memory and steady loaded footprint. Record that sampling can underestimate true peaks. Count all resident indexes/models in an adaptive system even if only one branch is used for a query.

Use GPU session hours and query-processing time as the primary compute indicators. Kaggle's free allocation does not establish zero underlying economic cost or measured energy consumption. Do not publish euros/query without an explicit dated price source, hardware utilization assumptions and a separate sensitivity analysis. Keep dollars/euros and energy estimates out of the core claim initially.

## Portable bundle and end-of-session gate

A session export contains new run bundles, new shared objects, the source snapshot/lockfile references and an inventory of file sizes/hashes. Use a few reasonably sized archives rather than thousands of drive writes. Record attached Kaggle dataset/notebook versions when used as inputs.

Before stopping: verify archive integrity, confirm receipt on durable storage, check one restored artifact, register the session and stop the accelerator. A notebook remaining open is not a backup. At the weekly gate, restore an older experiment into a clean workspace with an empty local cache.
