# Day 4 progress

Date: 16 October 2026. Status: CPU BM25 adapter and preserved runner implemented; real SciFact development baseline executed and locally restored. Kaggle and independent backup remain pending.

The [usage guide](../DAY04-BASELINE.md) records acquisition, exact analyzer/scoring settings, commands, data attribution and limitations. The [decision](../decisions/0006-bm25-baseline.md) explains schema 2 and its small-data scope. Existing E00 schema-1 runs still verify.

## Inputs and partition audit

The BEIR SciFact archive matched published MD5 `5f7d1de60b170fc8027bb7898e2efca1`. Exact archive/extracted-file SHA256 and terms are in [the source receipt](../../experiments/scifact-source.json). Extracted corpus: **5,183 documents**. The official training pool contains **809 judged queries**, frozen as **647 train / 162 validation** with seed 17 and the existing grouped recipe. The [selection manifest](../../experiments/splits/scifact-seed17.json) preserves exact membership and source hashes.

Train/validation audit passed for IDs, normalized query text and declared families. No query had a declared family ID in this release. Thus the audit does not establish independence of paraphrases or other semantically related claims; this limitation remains relevant. Final-test qrels were not extracted, loaded or evaluated. The complete distributed queries file was retained for source fidelity; only training-pool judged IDs entered development.

## Initial CPU baseline

Settings: BM25S 0.2.14; Lucene scoring/IDF; k1=1.2, b=0.75; lowercase two-or-more word-character tokens; no stemming/stopwords; positive-score top-1000; explicit tie/self-match rules. No parameter search was performed.

| Development partition | Queries | nDCG@10 | Recall@100 | Recall@1000 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 647 | 0.675063 | 0.906311 | 0.963163 | 0.641461 |
| Validation | 162 | 0.638569 | 0.890432 | 0.962963 | 0.603775 |

Run IDs:

- Train: `20261016t183351z-e01-e4a2f8f5-eeeb8f7ee4bb`
- Validation: `20261016t183413z-e01-603a3450-feab55da4672`

Both runs are exploratory (`reportable: false`); these are source-development observations, not held-out scores. The [evidence record](../../experiments/e01-day04-evidence.json) links exact metrics and local checks. Canonical artifacts are below `artifacts/runs/RUN_ID/`; catalog records are in `experiments/catalog/`.

The warm single-pass CPU pilot measured train p50/p95 of 8.00/25.52 ms and validation 7.78/42.90 ms; indexing was 2.04/1.76 seconds. This includes full candidate sorting, has no controlled repeated timing blocks, and is not comparable to Kaggle GPU or a final online frontier. Raw query timings are preserved. Final hardware/thread instrumentation and repeated comparison blocks remain Week 3 work.

## Preservation and checks

- 90 tests passed, including score calculation, stable ties, empty/OOV queries, self-filter-before-top-k, index reload and restore after deleting working inputs.
- Ruff lint/format passed; source distribution and wheel built.
- Installed wheel in a separate locked runtime environment, outside the repository: smoke, restored SciFact validation recalculation and historical E00 verification passed.
- Both real runs were sealed, verified, exported, restored locally and reevaluated with identical per-query/aggregate outputs. Reloaded indexes reproduced their first ten query rankings exactly. Full independent algorithm rebuild remains Day 5 work.
- Exported bundles are approximately 21.5 MB train and 10.1 MB validation, under `.scratch/exports/`; local restored copies are under `.scratch/restore-drill/`.

The new tests caught and fixed a missing BM25S `nonoccurrence_array` field on index reload. The shared optional `/home/masterjs/kaggle/.venv` was not modified for Day 4; use the project environment. GitHub CI, actual Kaggle execution and an independent Drive restore were not run.

## Next session

Day 5: reproduce E01 from historical source in a fresh environment, carry the attribution note and bundle to private Google Drive `AtlasIR/archive/`, download/verify/reevaluate it, and prepare/execute a short Kaggle compatibility session. The existing export is a concrete upload candidate; local copies do not constitute independent backup. Substantial GPU jobs remain behind that restore gate. Dense embeddings start in Week 2; reranking begins in Week 3.
