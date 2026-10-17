# SciFact fixed retrieval: first findings

Kaggle run: `20261017t083913z-e03-fdcb26eb-b39dca4eb46c`, 17 October 2026.
Exploratory source development results; no final-test judgments or adaptive-policy results.

| Validation method | nDCG@10 | Recall@100 | Recall@1000 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.638569 | 0.890432 | 0.962963 | 0.603775 |
| E5 dense | 0.705386 | 0.944444 | 0.993827 | 0.675965 |
| BM25 + E5 RRF | 0.709464 | 0.956790 | 0.987654 | 0.671551 |

Dense improves nDCG over BM25 by 0.066817. RRF improves nDCG over dense by only
0.004078; dense has higher MRR@10 and Recall@1000, while RRF has higher Recall@100.
There is no single winner on every metric. No statistical significance claim is
made, and the one-pass stage timings are not an end-to-end latency comparison.

## Does retrieval routing have room to help?

On the 162 validation queries, a relevance-label oracle choosing the best of these
three methods per query reaches nDCG@10 **0.798681** versus the best fixed
method's **0.709464**: a gap of **0.089216**. There are **36 queries (22.2%)**
with strictly higher nDCG from a different method than fixed RRF.

- BM25 uniquely wins 8 queries.
- Dense uniquely wins 28 queries.
- RRF uniquely wins 19 queries.
- The remaining 107 queries have a tie for best; this does not mean all three tie.

Training evidence also shows variation: oracle 0.797582 versus fixed RRF 0.723718,
gap 0.073864, and 132/647 queries with strict gain over RRF. The validation oracle
is descriptive; its label-dependent choices must not become inference features.
Neither oracle is cost constrained. A router must predict useful differences from
features available at inference and account for every stage it executes.

## Next research decision

Proceed to fixed reranker action outcomes and the first selective-reranking
comparisons: never/always rerank, confidence threshold, matched-rate random, then
a small learned policy. The observed retrieval variation supports investigating
retrieval routing too; it does not establish reranking headroom or promise gains.
Retain both dense and RRF as fixed references. NFCorpus and predeclared convex
fusion remain required source comparisons before broader conclusions.

## Reproduction and evidence

The canonical sealed run is under `artifacts/runs/20261017t083913z-e03-fdcb26eb-b39dca4eb46c`.
Raw inputs, frozen selection, source, resolved configuration, evaluator details,
full rankings and per-query outcomes are retained there. Model/vector/index
objects are under `artifacts/objects`. Preserve the original combined download,
including its object ZIP; the small run bundle alone does not contain all objects.
The import verification receipt is in
`experiments/events/20261017t083913z-e03-fdcb26eb-b39dca4eb46c-kaggle-download.json`.
An independent Drive redownload is not required for this session and has not
been verified by this import.

Verification completed: all **897 exported objects** passed integrity checks;
checkpoint vectors and restored FAISS index vectors match exactly. Historical
BM25 scores and document rankings match exactly for both source partitions.
The run's source hash also matches its uploaded launcher source hash.

The recorded encoder ran on **cuda:0, Tesla T4**. Passage model-encoding time
summed across shards was **70.11 seconds** for 5,183 documents; batch-one query
encoding summed to **16.96 seconds** for 809 queries. These exclude setup,
checkpoint IO, profiling, search and export; they are not total session duration.
The real job reused its first saved passage shard and encoded the remaining 80.
**466 documents (9.0%)** exceeded 512 tokens before truncation; maximum 1,941.
This confirms that the earlier token-length warning corresponds to tracked
preprocessing rather than an unhandled model input-length error.

Cross-machine numerical limit: local CPU FAISS replay preserves every query's
top-10 document order and top-100/top-1000 membership. All four dense evaluation
metrics therefore remain unchanged. However, 550/809 queries have some swaps
among lower-ranked near-tied documents. Maximum matched-document score difference
is **2.3842e-7**, within the index's recorded tolerance. We do not claim bitwise
identical complete dense rankings across hardware. The original Kaggle rankings
remain the canonical quality evidence; their metrics and opportunity tables
recalculate exactly. No rerun or post-hoc change of tie rules was applied.
