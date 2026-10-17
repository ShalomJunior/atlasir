# Day 6: E5 encoding and a small GPU pilot

The local encoder, CPU pilot and actual locked Kaggle GPU pilot have passed
verification. The pilot's independent Drive download/restore also passed; Day 6 is complete. Day 5 tested Kaggle's kernel torch; this
notebook installs and probes the actual optional dense environment.

## What changed

E5 converts queries and document text into 384-dimensional float32 unit vectors.
Later, Day 7's exact vector index will search these vectors; this pilot performs
no relevance evaluation or SciFact corpus embedding.

Fixed settings: `intfloat/e5-small-v2` at model/tokenizer revision
`ffb93f3bd4047442299a41ebb6fa998a38507c52`, `query: ` / `passage: ` prefixes,
attention-mask mean pooling, L2 normalization, 512-token maximum including
special tokens, right truncation, eager attention, batch size 8, seed 17.
Title and body are joined by newline for corpus encoding. Token counts are
measured after prefixing and before truncation; padding is excluded from pooling.

The adapter uses the explicit Transformers model-card implementation. This
retains the planned E5 method; Sentence Transformers is not required for this
implementation. See [decision 0007](decisions/0007-e5-encoding.md).

The default project environment stays small. Optional `dense` dependencies are
locked to torch 2.9.1, Transformers 4.57.1 and safetensors 0.6.2, with transitive
versions in uv.lock. Do not run `uv sync --extra dense` in the shared environment.
The optional GPU dependency installation is done inside Kaggle's fresh isolated
environment; it downloads several GB of wheels and the approximately 135 MB model.
No quota or setup-duration guarantee is made.

## Your next Kaggle session

Two files are prepared on this computer:

- Notebook: `notebooks/kaggle-day06.ipynb`.
- Private input ZIP: `.scratch/kaggle/atlasir-day06-input.zip`.

1. Import the **Day 6 notebook** into Kaggle. Upload the Day 6 ZIP as a private
   dataset, attach it and remove Day 5 inputs from this notebook. There must be
   exactly one `launcher-input.json` attached.
2. Enable Internet and select the available T4 accelerator. Run the cells in
   order. The first cell uses the corrected base64 archive transport already
   tested after Day 5's upload problem.
3. Setup installs the locked `dense` extra in an isolated Python 3.11 environment.
   The third code cell checks CUDA in that same environment, separately from
   the notebook kernel. Both visible GPUs are probed; encoding uses `cuda:0`.
4. The pilot encodes **two synthetic queries and three synthetic passages**,
   including one deliberately long passage. It checks finite normalized vectors,
   stable IDs, truncation metadata and repeated encoding agreement. It creates
   a new schema-3 E02 pilot run with source/config/input/runtime and model-file
   hashes, vectors and a checksum inventory. No benchmark scores are reported.
5. Download **both** `day06-e5-pilot-export.zip` and `day06-dense-runtime.zip`.
   Save the notebook/output version as available, then stop the GPU session.
   Put both downloads in `.scratch/kaggle/day06-downloads/` and tell the agent.

Cell failures stop execution. Send the full error/output from the failed cell;
do not change package versions, precision or device silently. The job logs the
pilot output before raising a failure. Every full run creates a new workspace
and run ID, so rerunning does not overwrite earlier experiments.

The input ZIP contains source/configs only, not model weights. Model/tokenizer
files download at the pinned commit. The pilot records their byte hashes; model
weights are not included in the small run export. Day 7's shared-object/archive
work must account for model-file durability and resumable embedding shards.

## Local commands and checks

Default correctness tests do not download models or torch:

```sh
uv sync --locked
uv run --locked pytest
```

Two tensor tests skip in that lightweight profile; they were separately run
with the existing shared environment. The real model CPU pilot used that
optional profile, not the locked dense extra, and records all actual versions.
To run in a separately provisioned dense environment:

```sh
uv run --locked --extra dense atlasir encode-pilot --device cpu
```

Use `--device cuda:0` only with a usable CUDA runtime. There is no automatic CPU
fallback. `--local-files-only` forbids fetching uncached model files. Export and
restore use the same `atlasir verify/export/restore` commands as earlier runs.
`reevaluate` remains for BM25 rankings; encoding pilots are checked from their
saved vector arrays, metadata and checksum inventory.

Rebuild the input from current reviewed source using a new destination:

```sh
uv run --locked python scripts/prepare_kaggle_day06.py --destination .scratch/kaggle/atlasir-day06-input.zip
```

References: [pinned E5 files](https://huggingface.co/intfloat/e5-small-v2/tree/ffb93f3bd4047442299a41ebb6fa998a38507c52),
[E5 model-card implementation](https://huggingface.co/intfloat/e5-small-v2#usage),
[Transformers 4.57.1 installation](https://huggingface.co/docs/transformers/v4.57.1/installation).
