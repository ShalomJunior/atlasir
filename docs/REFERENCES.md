# Research references

Sources carried forward from the research bibliography. Inclusion does not establish independent reproduction. Paper identifiers match the research protocol.

| ID | Source | Relevance |
| --- | --- | --- |
| P01 | Thakur et al., [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663), 2021 | What zero-shot evaluation means; why domains and candidate recall matter; benchmark conventions. |
| P02 | Manning, Raghavan and Schütze, [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/), 2008 | Derive a small ranking and its metrics by hand; explain term frequency saturation, IDF and length normalization. |
| P03 | Karpukhin et al., [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), 2020 | Dual-encoder versus cross-encoder; offline document encoding versus online query encoding; candidate retrieval limitations. |
| P04 | Wang et al., [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533), initially 2022 | Why E5 is an appropriate fixed starting point; input conventions, training-source disclosures and truncation. |
| P05 | Cormack, Clarke and Büttcher, [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf), 2009 | Implement the rank-fusion formula, constant, absent-document treatment and a tiny worked example. |
| P06 | Bruch, Gai and Ingber, [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934), initially 2022 | Why RRF should have a tuned score-fusion comparison; normalization, validation labels and parameter sensitivity. |
| P07 | Nogueira and Cho, [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085), 2019 | Joint query/document scoring and why candidate depth controls inference work. |
| P08 | Cronen-Townsend, Zhou and Croft, [Predicting Query Performance](https://doi.org/10.1145/564376.564429), 2002 | Distinguish pre-retrieval signals from post-retrieval signals and predictive quality from policy utility. Full query-clarity implementation is optional. |
| P09 | Arabzadeh, Yan and Clarke, [Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection](https://arxiv.org/abs/2109.10739), 2021 | Closest retrieval-routing precedent: classifier signals, labels, tested choices, budget/cost assumptions and transfer limitations. |
| P10 | Genc, Korukluoglu and Allan, [Adaptive Re-Ranking](https://arxiv.org/abs/2606.25249), 2026 | Closest reranking precedent: utility-based labels, oracle/router gap, feature ablations and held-out domains. |
| P11 | Urbano, Lima and Hanjalic, [Statistical Significance Testing in Information Retrieval: An Empirical Analysis of Type I, Type II and Type III Errors](https://arxiv.org/abs/1905.11096), 2019 | Understand assumptions and error types before interpreting small improvements or nonsignificance. |
| P12 | Lù, [BM25S: Orders of magnitude faster lexical search via eager sparse scoring](https://arxiv.org/abs/2407.03618), 2024 | Understand the chosen lexical backend and why backend-specific timing is not a general claim about BM25. |
