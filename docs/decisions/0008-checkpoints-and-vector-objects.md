# 0008: shared objects, strict embedding resume and exact FAISS

Date: 17 October 2026. Status: accepted for the Day 7 source-training pilot.

Keep large binaries in a SHA256-addressed object store. Use atomic publication,
exclusive checkpoint metadata and a single-writer advisory lock. A completed
shard needs validated vectors plus its ID/config/input descriptor; an orphan
binary is insufficient. Every resume rechecks objects and records a new attempt.
Store original-ID mappings and source input identities with the vectors.

Bind embedding jobs to model/tokenizer revisions and file hashes, normalized
preprocessing, encoder implementation, shard size and actual runtime/profile.
Changes to these start a new job; report/demo changes do not invalidate vectors.
Reject incompatible or corrupt checkpoints rather than silently recomputing
and overwriting evidence. Resume targets compatible historical source/runtime;
there is no cross-device/version cache-mixing assumption.

Separate the model's logical revision from durable file availability. Export
model/tokenizer objects and restore them offline before resumed encoding, using
an allowlisted file set and exact byte hashes. Model weights are now covered by
the object transport, not duplicated inside every small run bundle. Imports and
exports support object subsets for future incremental transfer; automated remote
transport and DVC deduplication remain separate work.

Use CPU FAISS 1.13.2 `IndexFlatIP` with unit float32 vectors and exact search.
Expand boundary ties before score-descending/original-string-ID-ascending order.
Preserve vectors so indexes can be rebuilt when a serialized binary is not
portable. New index descriptors declare a small cross-machine score tolerance;
ranked IDs still must match. Within-session reload comparison remains exact.

Schema 4 records the capped source-training recovery/index pilot and references
its object dependencies. Preserve schemas 1–3 without modifying historical runs.
The 256-document Kaggle sample is preselected by seed-17 ID hashes, without
relevance-grade selection. It measures infrastructure costs, not retrieval
quality; Day 8 remains the full source dense-baseline comparison.
