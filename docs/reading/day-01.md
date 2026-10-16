# Day 1 reading notes

Date: 16 October 2026. These are project notes from the fetched sources, not an assertion that the project owner has finished reading them. No reported paper result has been independently reproduced.

## P01: BEIR — setup and evaluation

Source read: [BEIR, arXiv v1](https://arxiv.org/html/2104.08663v1), particularly benchmark design, experimental setup, evaluation metric and limitations. Later paper versions expand the benchmark; do not mix v1 table numbers with current dataset inventories.

BEIR studies transfer across tasks and domains rather than success on a single familiar collection. It uses nDCG@10 to compare ranked outputs with binary or graded judgments. Its measurements distinguish retrieval effectiveness from runtime and index footprint. Annotation choices and text length can influence comparisons.

For AtlasIR: keep a common evaluator and full per-dataset corpora, report source and held-out results separately, save truncation settings, and establish BM25 before drawing conclusions about neural methods. Published timing is not our hardware baseline. Training an AtlasIR router on selected source datasets is not the same protocol as evaluating every model completely zero-shot.

Question to answer before Day 3: what are corpus, query and qrels files, and how does a ranked result become nDCG@10? The evaluator's precise gain/discount conventions will be pinned and tested when it is implemented.

## P09: retrieval strategy selection — initial skim

Source: [Arabzadeh, Yan and Clarke, arXiv v1](https://arxiv.org/html/2109.10739v1), method and experiment sections skimmed.

The paper studies sparse-versus-dense and sparse-versus-hybrid classifiers. Training labels depend on the rank of a relevant item under sparse retrieval. The query-only variant and a variant using an initial retrieved passage have different inference inputs and costs. Random allocation is an important comparator, and candidate recall is central to its first-stage retrieval evaluation.

For AtlasIR: generic strategy selection is already established. Track feature access and obligatory retrieval work, compare with random allocation, and preserve candidate recall alongside final top-rank quality. Substituting compact models, new datasets or new labels is an adaptation, not an exact reproduction of the paper.

Question for the deeper Day 14 reading: could a policy reduce dense calls but lose the candidates a later reranker needs?

## P10: adaptive reranking — initial skim

Source: [Adaptive Re-Ranking, arXiv v1](https://arxiv.org/html/2606.25249v1), method, utility labeling and split setup skimmed.

This work labels queries using a quality/cost utility and selects among ranking strategies with different expense. It distinguishes the potential of ground-truth-informed selection from the behavior of a learned router, and reserves test data for evaluation.

For AtlasIR: an oracle is an analytical tool, not an executable model. Derive labels only from source training outcomes, choose settings on validation, and measure actual feature/router costs. A large oracle gap may reveal a difficult prediction problem rather than a useful deployable advantage.

Question for Day 14: how much of the apparent gain survives the router's errors, feature computation and a changed domain/hardware profile?

## Owner reading task

- [ ] Read P01's benchmark/setup/metric sections and explain nDCG@10 in your own words.
- [ ] Skim P09's two classifier variants and identify which already paid for sparse retrieval.
- [ ] Skim P10's labeling/oracle discussion and distinguish training labels from inference features.
- [ ] Write a short answer to: “What result would make us reject the adaptive approach?”

Suggested answer to test rather than memorize: if the frozen learned policy fails to improve the quality–cost frontier against strong fixed/simple comparators, then the proposed practical benefit is unsupported in that setting.
