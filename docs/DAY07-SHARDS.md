# Day 7: resumable embeddings and exact FAISS

**Execution update:** skip this standalone Kaggle pilot. Use the [full SciFact comparison](DAY08-COMPARISON.md), which incorporates recovery/index checks into real baseline work. The instructions below preserve the historical pilot procedure.

Local implementation and a real-model CPU pilot passed. The prepared Kaggle
pilot still needs to run before we have a measured GPU throughput estimate.
This is infrastructure for Day 8's full dense baseline, not a retrieval-quality
experiment.

## What happens in the pilot

The private input contains 256 documents and 16 source-training queries from
SciFact. IDs are sampled by a fixed seed-17 SHA256 recipe, then encoded in
string-ID order. Relevance grades do not choose the sample; no final-test qrels
are accessed. The frozen source descriptor, source file hashes, split identity,
selected text and data attribution travel with the input.

The notebook:

1. Encodes one 64-document passage shard and deliberately pauses.
2. Exports that partial checkpoint, model/tokenizer objects and completed vectors.
3. Imports those exports into an empty object store and restores the checkpoint.
4. Loads E5 offline from the restored model files, verifies the execution profile,
   and encodes only the remaining passage shards.
5. Runs an uninterrupted comparison and checks resumed versus clean vectors.
6. Encodes the 16 queries, builds an exact CPU FAISS `IndexFlatIP`, saves its
   row mapping/index, reloads it and checks its top-10 rankings.
7. Saves raw per-shard times and a full-SciFact encoding/storage projection.
   No nDCG or retrieval-quality claims come from this corpus subset.

The controlled pause/export/empty-store restore tests portable recovery. It is
not an assertion that Kaggle temporary disk survives VM loss. Real checkpoints
must be exported to durable storage during longer jobs, not just left on disk.

## Your next Kaggle session

Use these local files:

- `notebooks/kaggle-day07.ipynb`
- `.scratch/kaggle/atlasir-day07-input.zip`

Import the notebook and upload the ZIP as a private dataset. Attach only the Day
7 input, enable Internet and choose the available T4 option. Run cells in order.
The corrected base64 transport protects source archives during upload. Setup
installs the locked `dense` and `search` extras in isolated Python 3.11.

The runtime probe checks CUDA there; encoding uses `cuda:0`, while exact FAISS
search runs on CPU. A second GPU is not needed for this small pilot. A tokenizer
warning about long texts can appear during pre-truncation counting; the actual
model inputs are still truncated to the recorded 512-token maximum.

Download both outputs:

- **`day07-shards-export.zip`** — sealed run, checkpoint metadata and binary
  objects, including model weights. Expect roughly **80 MB**, based on the
  measured CPU export; actual size is reported.
- **`day07-runtime.zip`** — isolated runtime, source input identity and run ID.

Put them in `.scratch/kaggle/day07-downloads/`, stop the GPU session, and tell
the agent. We will import the objects, restore the run, check the vectors/index,
then perform an independent Drive download/restore. Keep both original ZIPs.

## Why the files are separate

`artifacts/runs/RUN_ID/` is an immutable execution record. It contains exact
source/config/runtime/input snapshots, dependency IDs, checkpoint metadata,
resume evidence, rankings, index descriptor and forecast.

`artifacts/objects/SHA256/` stores large binary payloads by their actual byte
hash: model files, vector shards, input tables and serialized indexes. Reusing
identical content reuses the same object. Completed objects and shard records
are not overwritten; tampering is detected on verification. Unreferenced valid
objects may remain after a failed attempt; they are not trusted as completed
shards just because a binary file exists.

An embedding checkpoint is separate from a run. Its identity binds all input
IDs/text, query/passage kind, shard size, model/tokenizer revisions and file
hashes, preprocessing, encoder implementation hash and execution profile.
Device/runtime/config/input changes require a new checkpoint. Unrelated report
or UI edits do not invalidate vectors. Every resume verifies existing shards
before computing missing ones and records a fresh attempt with its parent.
Concurrent writers to the same checkpoint are rejected with an advisory lock.
This implementation targets the project's Linux/Python profiles.

A run bundle alone is not a complete Day 7 backup: preserve its object export
as well. The object ZIP supports up to 1 GiB uncompressed per bundle; larger
collections must be split. The earlier 128 MiB run-bundle bound stays intact.
Model files alone total approximately 134.5 MB uncompressed, which is why they
belong outside the run archive. Object exports can select only new hashes for
future incremental transfers; this first export is self-contained. DVC/MLflow
integration remains later work.

## Restore commands

After safely extracting a trusted session ZIP into a new directory:

```sh
uv run --locked --extra search atlasir objects-import PATH/embedding-objects.zip --store .scratch/day07-restored-objects-RUN_ID
uv run --locked --extra search atlasir restore PATH/RUN_ID.bundle .scratch/day07-restored-RUN_ID
uv run --locked --extra search atlasir verify-shard-pilot .scratch/day07-restored-RUN_ID --store .scratch/day07-restored-objects-RUN_ID
```

Use a new empty object-store path for this drill so old cached files cannot mask missing export dependencies. After it passes, objects can also be imported into the shared coordinator store.

Verification checks object hashes, checkpoint inventories, IDs/text/config,
completed vector shapes/norms, preserved original shard, clean/resumed agreement,
and restored FAISS rankings. It does not execute the downloaded source or model.
The `EmbeddingJob`, `export_checkpoint`/`restore_checkpoint` and object-store
APIs support resume; the Day 7 CLI is a capped recovery pilot, not the full
baseline launcher.

Use the recorded FAISS version for serialized-index loading. Saved vectors can
rebuild an index if the binary is incompatible with another architecture/version.
Within-session reload rankings must match exactly. Cross-machine verification
requires identical ranked IDs; scores allow only the tolerance recorded in the
index descriptor (rtol 1e-6, atol 1e-7 for new exports). Ties at the cutoff are
expanded before applying score-descending/string-ID-ascending ordering.

## Local evidence and resource estimate

The real-model CPU pilot used 16 documents, four queries and four passage shards.
It reused exactly one shard, restored the model offline and matched clean vectors
exactly. It measured 5.879 seconds of sample encoding. Extrapolating that small
CPU sample gives about 31.7 minutes for 5,183 documents, or 63.5 minutes with a
2x margin. These figures exclude model setup, checksum/storage work, repeat
checks and export, and do not predict Kaggle GPU time.

The full SciFact float32 vector matrix is 7,961,088 bytes (about 8 MB); an exact
flat index needs approximately the same vector storage again, plus metadata.
The measured CPU pilot's model objects occupy 134,478,053 uncompressed bytes.
The GPU sample will update time/memory observations before full-corpus work.
The largest raw vectors remain manageable for this source corpus; no large
held-out-corpus cost is inferred here.

Run CPU correctness checks with:

```sh
uv sync --locked --extra search
uv run --locked --extra search pytest
```

This installs the small CPU FAISS extra, not torch/CUDA. Tensor tests are also
run separately in the optional shared environment. CI now includes real FAISS
fixture tests on Python 3.11 and 3.12. Remote CI status is separate from local
verification.

References: [FAISS getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started),
[inner product and cosine](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances),
[pinned CPU wheel](https://pypi.org/project/faiss-cpu/1.13.2/).
