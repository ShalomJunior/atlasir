# Hosted execution protocol

The package owns retrieval, encoding and evaluation. Notebooks are thin launchers that verify inputs, prepare an isolated environment, invoke package code and export results. The [full SciFact comparison](DAY08-COMPARISON.md) is the current end-to-end example.

## Environment and devices

Use a pinned uv version, supported Python version and committed lockfile. Neural jobs install the optional dense/search extras into an isolated environment. A notebook kernel's installed libraries do not establish the subprocess environment's compatibility.

Probe CUDA in the actual experiment environment. Select a device explicitly; neural runs reject unavailable CUDA rather than falling back silently to CPU. The encoder places both the model and input tensors on that device. Two visible GPUs do not imply that a single-device implementation uses both. Record the actual hardware, CUDA, driver, package and thread settings with each run.

Never capture credentials or unrestricted environment-variable dumps. Source archives use an allowlist and exclude local planning notes, credentials, datasets and generated outputs. Model/tokenizer revisions and input hashes must be immutable.

## Recovery and exports

Embedding jobs use immutable input/model/runtime identities and verified shards. Resume checks completed shards before reusing them. Each attempt has its own record, including failures and parent attempts. A changed model, preprocessing or execution profile requires a compatible new job identity rather than modifying old shards.

Temporary notebook storage is not a durable archive. Preserve completed session exports before ending the runtime. Keep an independent durable copy. Verification must include the complete object dependency inventory; a run bundle alone may not contain model/vector/index bytes. Perform independent restore drills at meaningful milestones.

A saved notebook's code and its saved output are distinct. Record which execution produced each export. A fresh execution receives a new run identity, even if the notebook text is unchanged. Never replace sealed results to make a reproduction appear exact.

## Measurement boundaries

Separate index construction, offline batched encoding/scoring, live single-query execution and explicitly configured concurrent throughput. Cached rankings and scores can accelerate quality comparisons but cannot establish live latency savings.

Live latency includes preprocessing, query encoding, retrieval, fusion, feature extraction, policy inference, reranking and result assembly for every stage executed. Report model/index loading and downloads separately as startup work. Warm models and indexes before repeated timing blocks; retain raw per-query observations, order, batch sizes and thread settings.

Report index size and resident/peak memory, including all models/indexes loaded for a policy. Charge feature acquisition and sunk retrieval work to the policy. Keep timing results from different hardware and sessions distinguishable. A single isolated-stage pass is exploratory evidence, not a final latency frontier or a deadline guarantee.

Cross-hardware floating-point changes can reorder near ties. Preserve original rankings, record score tolerances and disclose whether ranking IDs, metric values or only numerical scores reproduce. See the [SciFact findings](progress/scifact-comparison-findings.md) for observed limits.

Offline MLflow ingestion and DVC transport remain planned integrations. Canonical checksum-verified artifacts and catalogs currently provide the preservation contract; a tracking database is not the only copy of experimental evidence.
