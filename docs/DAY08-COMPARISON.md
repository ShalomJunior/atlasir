# Full SciFact retrieval comparison

This replaces the standalone Day 7 Kaggle pilot. Run one substantive session:
BM25, E5-small-v2 and equal-weight reciprocal-rank fusion (RRF, k=60), using
all 5,183 documents and the frozen 647 training / 162 validation queries.
No final test judgments are uploaded or evaluated. NFCorpus remains the second
source domain; this session is the first source comparison, not completion of G2.

## Run on Kaggle

1. Upload `.scratch/kaggle/atlasir-scifact-comparison-input.zip` as a **private dataset**,
   using the same upload procedure as Day 6.
2. Open `notebooks/kaggle-scifact-comparison.ipynb` on Kaggle. Remove previous AtlasIR
   input datasets and attach only the new comparison dataset.
3. Enable Internet and T4. Run all cells. The encoder uses `cuda:0`; the second T4
   stays available. FAISS and BM25 search run on CPU. Setup uses locked Python 3.11
   and dense/search dependencies in an isolated environment.
4. The last cell shows a single link, `atlasir-scifact-results-RUN_ID.zip`.
   Download it into `.scratch/kaggle/comparison-downloads/` and stop the GPU session.
   Keep a second copy of that same ZIP in your existing Drive backup folder.
   A separate Drive redownload/restore exercise is not required for every session.

The first cell verifies the upload and preserves exact source archive bytes using
base64 transport, avoiding Kaggle's nested-archive extraction issue from Day 5.
Model download/setup and full encoding make this longer than the tiny Day 6 pilot;
we do not yet have a measured full-corpus T4 duration. No GPU-hour estimate is a guarantee.

If the experiment cell fails, keep the session running and rerun **that cell**
after resolving the error. Completed shards are verified and reused if inputs,
model, code and runtime match; each attempt receives a new run ID. Rerunning the
first cell creates a new workspace. Kaggle session termination can lose its working
directory: checkpoints are not a substitute for downloading the completed export.

## What the session produces

- `results.json`: train/validation nDCG@10, Recall@100/1000 and MRR@10 for all three methods.
- `evaluation.json`: common evaluator profile, coverage and every per-query metric.
- `opportunity.json`: per-query winners, ties and a relevance-label oracle gap over
  the best fixed method **on each partition**. This is a descriptive upper bound,
  with no cost constraint. It is neither a trained policy nor evidence of adaptive gains.
- Complete compressed top-1000 rankings, resolved settings, raw development inputs,
  source snapshot, environment record and checksums in an immutable run bundle.
- Passage/query checkpoints, exact FAISS index, vectors and pinned model objects in
  a separate object ZIP within the same download. Preserve the entire download.
- Raw stage timings. Query encoding uses batch size one; searches use cached query
  vectors after five warmups. These are exploratory isolated measurements, **not**
  end-to-end latency, a hybrid budget or measurements of an adaptive controller.

The full corpus is encoded once. The first passage shard is saved and resumed
inside this real run; every saved-index query ranking is checked against a reloaded
index. Existing Day 7 tests cover independent offline model recovery. No second
full-corpus encoding is done just to repeat that pilot.

BM25 keeps the original analyzer and parameters. Dense search uses normalized E5
vectors, exact inner product and explicit original-ID tie handling. Each base
contributes its top 1000 to RRF; a missing candidate contributes zero. RRF ranks
by the sum of reciprocal ranks and returns at most 1000 distinct documents.
There is no fusion parameter search in this session. Convex fusion remains a later
predeclared comparison, alongside the NFCorpus baseline.

## After the run

Inspect the validation comparison and training query-level variation. Next build
fixed reranker outcomes, then compare never/always rerank, confidence thresholds,
matched-rate random allocation and a small learned policy. Retrieval-only oracle
headroom cannot establish reranking headroom. Final-domain tuning remains prohibited.
Offline MLflow ingestion and extra restore drills are deferred until they help
compare substantive experiments or verify a meaningful milestone.

Quality can be recalculated without model inference after restoring the inner bundle:

```bash
uv run --locked python scripts/run_comparison.py --reevaluate /path/to/restored-run
```

The schema-5 run is verified by the existing `atlasir verify`, `export` and `restore`
commands. Embedding/model objects remain separate from the small run bundle.

## Dataset attribution

SciFact: Wadden et al. (EMNLP 2020); S2ORC: Lo et al. (ACL 2020);
BEIR: Thakur et al. (NeurIPS 2021). Claims/evidence are CC BY 4.0 and S2ORC
abstracts ODC-By 1.0. See the original recorded source receipt
`experiments/scifact-source.json` and
https://github.com/allenai/scifact/blob/master/LICENSE.md.
