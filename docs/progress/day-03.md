# Day 3 progress

Date: 16 October 2026. Status: loader, partition guards and evaluator implemented; real benchmark input acquisition/manifests remain pending.

## Implemented

- Local BEIR corpus/query JSONL and qrels TSV reader; exact file hashes, original string IDs and complete corpus retention.
- Duplicate/malformed record and dangling-judgment checks; read-only dataset mappings.
- Explicit role/official-split validation; held-out/final loading safeguards and development evaluation guard.
- Deterministic source train/validation membership with normalized text/declared-family grouping, exact ID lists, source hashes and recipe validation.
- Source-partition overlap audit; NFCorpus uses official train/dev rather than a replacement random split.
- Explicit pytrec_eval-backed nDCG@10, Recall@100, Recall@1000 and MRR@10; per-query and aggregate records with fixed query coverage.
- Finite/unique scored hits, consistent tie order, explicit self-match handling, zero-scored missing/empty/no-positive queries.
- `data-check`, `make-splits` and `evaluate` commands; no dataset/model downloads during checks.

See [Day 3 usage](../DAY03-DATA.md), [contract decisions](../decisions/0004-data-evaluation-contract.md) and [reading notes](../reading/day-03.md).

## Synthetic validation evidence

The bundled BEIR fixture has four documents and four official train queries (plus separate synthetic dev/test queries). The source manifest at [fixture-seed17.json](../../experiments/splits/fixture-seed17.json) records three train IDs and one validation ID, keeping duplicate query text together. Its actual validation fraction is 25%, with a target of 20%; this is a tiny grouped correctness fixture, not an actual SciFact split.

The handwritten rankings give nDCG@10 `0.606909333640491`, Recall@100/1000 `0.75` and MRR@10 `0.625`. Tests independently calculate these expectations. One query has no submitted ranking and no positive judgment, and remains zero-scored in the denominator.

Boundary tests initially exposed that the direct installed provider call for RR@10 returned reciprocal rank beyond ten. The implementation now truncates first; tests cover ranks 10/11, 100/101 and 1000/1001. This finding is documented rather than assuming all accepted backend arguments are implemented.

Dependency versions: ir-measures 0.4.3, pytrec-eval-terrier 0.5.11, NumPy 2.4.6, SciPy 1.17.1, captured in the updated uv.lock. Runtime metrics are imported only by the evaluation command; installation and artifact-lifecycle commands retain their lightweight behavior.

## Verification

| Check | Observed result |
| --- | --- |
| Ruff lint/format checks | Passed |
| Full pytest suite | 79 tests passed |
| Source distribution and wheel | Built successfully |
| Isolated wheel with locked runtime dependencies | Packaged BEIR data inspection, expected evaluation scores, split creation and validation selection passed outside the repository |
| Historical E00 compatibility | Day 2's original sealed run verified successfully through the newly installed wheel |

The installed-wheel dependency environment was created with `uv sync --locked --no-dev --no-install-project` in a separate venv, followed by wheel installation without dependency changes. This verifies installed package behavior, not a Kaggle runtime. GitHub CI has not run remotely.

## Remaining work and handoff

No official dataset was acquired, so this day does not claim frozen real SciFact/NFCorpus IDs, official dataset counts/licenses or a benchmark result. During baseline preparation, record allowed acquisition sources/terms, download hashes and actual partition metadata, generate real manifests and audit source overlaps. Inspect available family/claim metadata; normalized-text grouping alone does not establish semantic independence.

The evaluator currently produces diagnostic JSON. Day 4's runner needs a versioned research-run schema that preserves real dataset descriptors, rankings and per-query outputs; do not force those into the unchanged E00 preservation schema. Old E00 records remain immutable and should still verify.

Independent Drive restore is still pending. Small local implementation/fixture checks can continue; do not mark the archive gate satisfied or launch long experiments based on this progress note. Next technical task: BM25 adapter, pinned analyzer/scoring settings and baseline runner integration.
