# Day 3 reading notes

Date: 16 October 2026. These notes accompany implementation checks; the project owner's reading remains a separate task.

## Evaluation

Read: [Introduction to Information Retrieval, ranked evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html).

Recall describes how many judged relevant documents are recovered; it does not express where they appear inside the cutoff. Rank-sensitive metrics answer a different question. Queries need a fixed evaluation denominator, and graded labels allow degrees of relevance. The chapter shows an exponential-gain nDCG formula; AtlasIR explicitly selects the linear-gain trec_eval convention, so formulas must not be substituted without changing the protocol.

Owner exercise: calculate the fixture's two nontrivial query scores by hand, then explain why dropping its missing query would inflate the aggregate.

## BM25 preparation

Read: [Introduction to Information Retrieval, Okapi BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html).

BM25 combines term specificity, saturated term frequency and document-length adjustment. The analyzer determines which terms are counted; implementing the same equation with a different tokenizer can produce different rankings. The next milestone must record both analyzer and scoring parameters instead of calling every lexical setup the same baseline.

Owner exercise: explain why repeating a query term many times in one document should not increase its score linearly forever, and what changing length normalization might affect.

## Implementation references

- [BEIR data loader](https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py): local file layout and official judged-query selection.
- [ir_measures definitions](https://ir-measur.es/en/latest/measures.html) and [providers](https://ir-measur.es/en/latest/providers.html): provider-specific metric support matters. Boundary fixtures exposed an unsupported direct RR cutoff, so AtlasIR explicitly truncates before reciprocal-rank calculation.

No library metric call is accepted as correct merely because its name resembles the requested measure. The hand-calculated cases and cutoff tests are part of the experimental foundation.
