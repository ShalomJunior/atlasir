# ADR 0004: Dataset roles, split identity and metric conventions

- Date: 16 October 2026
- Status: adopted for Day 3; actual benchmark input acquisition and partition audits remain to be performed

## Data contract

Use local BEIR files directly: `corpus.jsonl`, `queries.jsonl`, and `qrels/{train,dev,test}.tsv`. No dependency on the full BEIR package or automatic downloader is needed to parse the format. Format reference: [BEIR loader](https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py).

IDs must already be nonempty strings without whitespace; do not coerce numeric values or lose leading zeros. Keep the complete corpus. Select the official judged query partition from qrels, rather than treating every query in the shared queries file as belonging to that split. A qrels query with only zero grades remains included; a query absent from that official qrels file has no inferred split membership.

Reject duplicate corpus/query IDs, duplicate qrels pairs, dangling references, invalid JSON/text fields and negative/noninteger grades. Unjudged documents are allowed in rankings and count as nonrelevant. Hash the exact bytes consumed from each file. Dataset records and nested qrels mappings are read-only through the public object.

## Partition guard

Every load requires `train`, `validation` or `final`. The expected official split is validated. NFCorpus retains train/dev/test. SciFact validation comes from its official training pool through a saved selection manifest. An unpartitioned SciFact train pool may be loaded to construct that manifest; development evaluation rejects the unpartitioned pool.

FiQA/SCIDOCS are final-only. Final loads require an explicit flag; final evaluation additionally requires final purpose. These checks prevent ordinary accidental mixing; they are not an access-control system or proof that the final protocol was frozen. No final benchmark data was downloaded or inspected for this milestone.

Partition the source pool deterministically using SHA-256 ordering with seed 17. Keep normalized duplicate query text and any declared `metadata.family_id` in the same group, using transitive grouping. Normalize with NFKC, casefold and collapsed whitespace. Choose a group-preserving validation allocation near the requested 20%, and record its actual fraction. Tiny pools or large groups cannot always yield exactly 80/20.

Saved manifests include complete source file hashes, exact train/validation IDs, grouping, seed, normalization and algorithm version. On reuse, validate source identity, coverage, disjointness and reproduction of the saved recipe. NFCorpus overlap audits compare IDs, normalized text, declared families and common file versions. Grouping does not detect every paraphrase or shared-evidence relationship; inspect actual source metadata during acquisition before claiming those were handled.

## Metrics contract

Use locked `ir-measures` 0.4.3 with the explicit `pytrec_eval` provider from `pytrec-eval-terrier` 0.5.11. The installed dependency versions are recorded in evaluation output and uv.lock.

- nDCG@10: linear relevance gains, `grade / log2(rank + 1)`, normalized against ideal ranking from the complete query judgments. This is the selected trec_eval-compatible convention, not the exponential-gain variant displayed in some textbook formulas.
- Recall@100 and Recall@1000: relevance grade >= 1.
- MRR@10: first positive judgment in the first ten returned documents; zero otherwise.
- Missing/empty rankings and no-positive queries: zero, included in the fixed partition denominator. Report coverage explicitly.
- Ties: descending original score, then ascending original string document ID. Supply strictly ordered surrogate scores to the metric backend so its internal tie order cannot change the result.
- Identical query/document IDs: exclude by default before scoring, with an explicit `keep` option and exclusion count. The original qrels denominator is retained. Freeze the appropriate policy for each actual dataset before comparison.

The installed provider's direct `RR@10` call returned `1/11` for a positive at rank 11 in our boundary fixture; inspection showed delegation to uncut `recip_rank`. Enforce the cutoff by passing only the top ten results to uncut RR. This is tested at ranks 10 and 11; no reliance on the unsupported direct cutoff remains. The [provider documentation](https://ir-measur.es/en/latest/providers.html) and [measure definitions](https://ir-measur.es/en/latest/measures.html) should be rechecked before changing provider/version.

## Scope and preservation

The bundled four-document BEIR fixture and handwritten rankings are correctness examples, not benchmark evidence. `evaluate` prints a JSON diagnostic and does not create an archived research run. Day 4's baseline runner must integrate dataset descriptors, rankings and evaluator output into a versioned run schema before reporting real experimental results. The Day 2 E00 schema stays unchanged; its old sealed run remains verifiable.
