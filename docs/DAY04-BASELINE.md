# Day 4: preserved BM25 baseline

Use the isolated project environment (`uv sync --locked`). This CPU implementation uses BM25S, Lucene scoring/IDF, k1=1.2, b=0.75 and float32 arrays. The analyzer joins title and text with a newline, lowercases and applies `(?u)\b\w\w+\b`; it uses no stemming or stopword removal. These are explicit initial settings, not tuned or a reproduction of a published best score. See [configuration](../configs/baselines/bm25.yaml) and [BM25S implementation](https://github.com/xhluca/bm25s).

Ranking retains positive scores only, orders by descending score then ascending string document ID and excludes self-matches before taking up to 1,000 hits. Empty/out-of-vocabulary queries return no hits. Full score sorting is included in CPU query latency. Do not compare this pilot's latency to optimized library-native top-k or claim a measured online frontier from one pass.

## SciFact acquisition and attribution

The [official BEIR dataset inventory](https://github.com/beir-cellar/beir#available-datasets) publishes the download and MD5. Download separately, then extract only development inputs:

```sh
mkdir -p .scratch/downloads
curl --fail --location https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip --output .scratch/downloads/scifact.zip
uv run --locked python scripts/prepare_scifact.py .scratch/downloads/scifact.zip .scratch/data/scifact --receipt experiments/scifact-source.json
uv run --locked atlasir make-splits --data .scratch/data/scifact --name scifact --role train --output experiments/splits/scifact-seed17.json
```

The receipt and split already exist in this workspace; reuse them for reproduction and compare hashes. Creation commands deliberately refuse existing destinations. The extractor checks the published MD5 and records SHA256 for the archive and exact extracted bytes. It never extracts final-test qrels. The distributed query file contains text for multiple official splits; the loader selects only training-pool judged IDs. Access to query text does not mean final judgments were evaluated.

Attribution: Wadden et al., *Fact or Fiction: Verifying Scientific Claims* (EMNLP 2020); Lo et al., *S2ORC* (ACL 2020); Thakur et al., *BEIR* (NeurIPS 2021). The [SciFact license notice](https://github.com/allenai/scifact/blob/master/LICENSE.md) assigns claims/evidence CC BY 4.0, S2ORC abstracts ODC-By 1.0, and code Apache 2.0. Retain these notices/attribution when copying inputs or bundles; project MIT licensing does not replace data terms. Source/receipts are tracked; corpus, indexes and run payloads stay out of Git.

## Run and restore

```sh
uv run --locked atlasir baseline --data .scratch/data/scifact --name scifact --role train --selection experiments/splits/scifact-seed17.json
uv run --locked atlasir baseline --data .scratch/data/scifact --name scifact --role validation --selection experiments/splits/scifact-seed17.json
```

Each invocation creates a new E01 ID. Use its printed directory:

```sh
uv run --locked atlasir verify artifacts/runs/RUN_ID
uv run --locked atlasir reevaluate artifacts/runs/RUN_ID
uv run --locked atlasir export artifacts/runs/RUN_ID .scratch/exports/RUN_ID.bundle
uv run --locked atlasir restore .scratch/exports/RUN_ID.bundle .scratch/restore-drill/RUN_ID
uv run --locked atlasir reevaluate .scratch/restore-drill/RUN_ID
```

The schema-2 bundle preserves resolved config, exact source/lock/runtime versions, raw corpus/queries/selected official qrels, selection manifest, dataset hashes, eager sparse index/vocabulary/document map, original-score JSON rankings, rank-preserving TREC rankings, aggregate/per-query evaluation and raw pilot timings. `reevaluate` uses the snapshot without the working data directory and requires the same evaluator profile/outcomes. Index reload requires the recorded BM25S version. Rebuilding the algorithm uses the historical source archive and lock, rather than today's moving code. Schema-1 E00 bundles remain supported.

These small self-contained baseline bundles duplicate input bytes intentionally. Their export size is bounded by the existing 128 MiB archive limit. Shared content-addressed objects and tracking integration remain later work before larger experiments; do not scale this layout blindly to embeddings/large corpora.

## Kaggle handoff

Day 5 is the first planned Kaggle session: reproduce the CPU baseline from pinned source, inspect actual devices and perform a short GPU compatibility test. Dense embeddings start on Days 6–8; reranking starts in Week 3. A compatibility probe can precede long-job clearance. Before substantial GPU jobs, upload a bundle and `.scratch/exports/SCIFACT-ATTRIBUTION.md` to the independent archive (recommended private Google Drive `AtlasIR/archive/`), download it to a fresh destination and verify/reevaluate it. Local restore evidence alone does not satisfy that gate. Kaggle and independent Drive execution remain unverified until actually performed.
