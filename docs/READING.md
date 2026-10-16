# Reading and source plan

Sources checked on 16 October 2026. Source identifiers, abstracts and relevant documentation were verified for this plan; complete paper reading and reproduction are scheduled below. A link here does not mean its results have been independently replicated.

Read with a decision in mind. For each required paper, write a short note: problem, method, available signals at inference, training labels, cost definition, split policy, strongest baseline, result, limitation and implication for AtlasIR. Save the exact paper version and bibliographic entry when taking detailed notes. Distinguish arXiv submission dates from later publication dates.

## Required reading

| ID | Source | When / reading depth | What to extract |
| --- | --- | --- | --- |
| P01 | Thakur et al., [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663), 2021 | Day 1, revisit day 3; setup, datasets, tables, limitations | What zero-shot evaluation means; why domains and candidate recall matter; benchmark conventions. |
| P02 | Manning, Raghavan and Schütze, [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/), 2008 | Days 3–4; evaluation chapter and probabilistic/BM25 sections | Derive a small ranking and its metrics by hand; explain term frequency saturation, IDF and length normalization. |
| P03 | Karpukhin et al., [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), 2020 | Day 6; architecture and evaluation, skim training details | Dual-encoder versus cross-encoder; offline document encoding versus online query encoding; candidate retrieval limitations. |
| P04 | Wang et al., [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533), initially 2022 | Day 6, revisit day 8; model/evaluation sections plus current model card | Why E5 is an appropriate fixed starting point; input conventions, training-source disclosures and truncation. |
| P05 | Cormack, Clarke and Büttcher, [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf), 2009 | Day 9; full short paper | Implement the rank-fusion formula, constant, absent-document treatment and a tiny worked example. |
| P06 | Bruch, Gai and Ingber, [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934), initially 2022 | Day 9; method, comparisons and limitations | Why RRF should have a tuned score-fusion comparison; normalization, validation labels and parameter sensitivity. |
| P07 | Nogueira and Cho, [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085), 2019 | Day 11; full short paper | Joint query/document scoring and why candidate depth controls inference work. |
| P08 | Cronen-Townsend, Zhou and Croft, [Predicting Query Performance](https://doi.org/10.1145/564376.564429), 2002 | Day 15; main idea and clarity-score motivation | Distinguish pre-retrieval signals from post-retrieval signals and predictive quality from policy utility. Full query-clarity implementation is optional. |
| P09 | Arabzadeh, Yan and Clarke, [Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection](https://arxiv.org/abs/2109.10739), 2021 | Skim day 1; deep read day 14 | Closest retrieval-routing precedent: classifier signals, labels, tested choices, budget/cost assumptions and transfer limitations. |
| P10 | Genc, Korukluoglu and Allan, [Adaptive Re-Ranking](https://arxiv.org/abs/2606.25249), 2026 | Skim day 1; deep read day 14 | Closest reranking precedent: utility-based labels, oracle/router gap, feature ablations and held-out domains. |
| P11 | Urbano, Lima and Hanjalic, [Statistical Significance Testing in Information Retrieval: An Empirical Analysis of Type I, Type II and Type III Errors](https://arxiv.org/abs/1905.11096), 2019 | Day 15, revisit before day 24/29 | Understand assumptions and error types before interpreting small improvements or nonsignificance. |
| P12 | Lù, [BM25S: Orders of magnitude faster lexical search via eager sparse scoring](https://arxiv.org/abs/2407.03618), 2024 | Day 4 or day 10; implementation/limitations | Understand the chosen lexical backend and why backend-specific timing is not a general claim about BM25. |

These are learning and method-selection tasks, not a demand to reproduce twelve entire papers. Budget roughly 12–18 hours across the first three weeks, with deeper follow-up only where it affects a design choice.

## Closest-work comparison to complete by G3

| Dimension | P09 retrieval routing | P10 adaptive reranking | AtlasIR to test |
| --- | --- | --- | --- |
| Decision | Sparse versus dense/hybrid strategy selection | Choice among ranking strategies of different cost | Retrieval strategy and reranking depth, with separate ablations |
| Main adaptation | Read label/feature and classifier details | Read utility labeling and router/oracle setup | Cheap query-only versus post-BM25 features; small policy models |
| Evaluation | Verify original datasets/splits in full text | Verify source/held-out datasets and cost settings in full text | Frozen source development plus two held-out tasks |
| Efficiency | Extract measured costs and omitted work | Extract hardware, cost definition and feature overhead | Explicit actual executed pipeline costs and budget violations |
| Evidence | Record code/data availability and reproduction limits | Inspect published code/data and reproduction limits | Restorable run bundles and table lineage |

P09 studies query-dependent retrieval selection; P10 studies cost-aware ranking strategy selection. Neither is a novelty claim for AtlasIR. The table identifies what to investigate and reproduce. Do not assert the combined design is new without a broader related-work review.

By day 15, reproduce one essential labeling/selection idea from each closest paper on AtlasIR's source data where feasible. Mark this an **adaptation**, because checkpoint/data/hardware changes prevent claiming an exact paper reproduction. Keep a separate full-fidelity reproduction recipe only if all original requirements are available. Report discrepancies rather than borrowing headline numbers.

## Optional reading tied to extensions

| Source | Read only when |
| --- | --- |
| Robertson and Zaragoza, [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019), 2009 | A deeper BM25 derivation/analyzer question matters; the textbook suffices initially. |
| Chen et al., [Efficient Cost-Aware Cascade Ranking in Multi-Stage Retrieval](https://doi.org/10.1145/3077136.3080819), 2017 | Comparing routing with document-pruning cascades or expanding related work. |
| [Unsupervised Query Routing for Retrieval Augmented Generation](https://arxiv.org/abs/2501.07793), 2025 | Considering routing among separate sources; distinguish that task from methods searching the same corpus. |
| [Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672), 2024 | Starting the French extension. |
| [MIRACL project](https://project-miracl.github.io/) and [dataset paper](https://arxiv.org/abs/2210.09984) | Selecting French data; verify monolingual versus cross-language task and full corpus requirements. |

Refresh related work before G5 and report release using the references/citations of P09/P10 and targeted searches for retrieval routing, selective reranking, query performance prediction and cost-aware cascades. Save the search date and inclusion criteria. This is a focused review, not an assertion of an exhaustive systematic survey.

## Implementation documentation

| Topic | Primary documentation | When |
| --- | --- | --- |
| Data | [BEIR repository](https://github.com/beir-cellar/beir), [dataset inventory](https://github.com/beir-cellar/beir/wiki/Datasets-available) | Days 1–3; counts, formats, splits and dataset owner attribution |
| Environment | [uv projects](https://docs.astral.sh/uv/guides/projects/), [locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) | Days 1–2; lockfile in Git, validate with `uv sync --locked` |
| Artifacts | [DVC add](https://doc.dvc.org/command-reference/add), [remote storage](https://doc.dvc.org/user-guide/data-management/remote-storage) | Day 2; pointers, durable bytes and targeted restoration |
| Cache limits | [DVC pipeline execution](https://doc.dvc.org/user-guide/pipelines/running-pipelines) | Day 10; mutable outputs are not archived runs |
| Comparison | [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/), [backend stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/) | Day 10; local metadata/UI, canonical export outside the database |
| Lexical | [BM25S repository](https://github.com/xhluca/bm25s) | Day 4; variants, tokenizer/analyzer settings and serialization |
| Dense | [E5-small-v2 model card](https://huggingface.co/intfloat/e5-small-v2), [FAISS getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started) | Days 6–8; correct preprocessing, exact search, IDs |
| Reranker | [Sentence Transformers MS MARCO cross-encoders](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html) | Day 11; checkpoint usage, not imported speed claims |
| Metrics | [ir_measures](https://github.com/terrierteam/ir_measures) | Day 3; explicit metrics/providers, aggregate/per-query behavior |
| Model revisions | [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download) | Days 6/11; commit revisions and verified local snapshots |
| Randomness | [PyTorch reproducibility](https://docs.pytorch.org/docs/main/notes/randomness.html) | Days 7/18; seeds, determinism and hardware differences |
| Hosted execution | [Kaggle notebooks](https://www.kaggle.com/docs/notebooks), [GPU quota usage](https://www.kaggle.com/docs/efficient-gpu-usage), [Colab FAQ](https://research.google.com/colaboratory/faq.html) | Days 1–2, then check conditions weekly |

Pin actual software versions when installing and record the documentation/source revision used to implement APIs. Hosted platform behavior and quota may change; the user's observed account allowance determines execution planning.

## Questions to be able to answer

- Why does a dense retriever miss an exact entity that BM25 finds?
- Why can reranking improve nDCG without changing Recall@100?
- Which features require already paying for a retrieval step?
- What is the difference between a mean budget and a strict per-query deadline?
- What does an oracle know that a real router cannot know?
- Which result would falsify the proposed benefit?
- What data was used to choose the threshold and what was untouched?
- Can another reader recover the exact source and inputs behind one figure?
- Why are Kaggle and Colab latencies in separate comparisons?
- Does held-out policy evaluation rule out exposure during model pretraining?
