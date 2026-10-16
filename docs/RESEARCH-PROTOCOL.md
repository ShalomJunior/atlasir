# Research protocol: relevance versus latency

Version: planning draft 1, 16 October 2026. Freeze operational details during week 1, then freeze the final experiment matrix at G5 before viewing final-test results. This protocol defines proposed choices, not measured results.

## Research questions and hypotheses

| ID | Question | Evidence |
| --- | --- | --- |
| RQ1 | When do lexical, dense and hybrid strategies differ in quality and runtime? | Common fixed baselines, per-query gains and candidate recall |
| RQ2 | Can selective reranking improve the measured quality–latency frontier? | Learned policy versus fixed depths, matched random allocation and cheap heuristics |
| RQ3 | Does retrieval-strategy routing add value beyond selective reranking? | Full controller versus reranking-only controller and strongest feasible fixed pipelines |
| RQ4 | Does the benefit persist beyond policy-development domains? | Frozen policy on untouched FiQA and SCIDOCS queries/judgments |
| RQ5 | Are improvements robust to measurement, feature overhead and failures? | Live execution, repeated timing, feature ablations, losses and restoration audit |

Null outcome: the adaptive method has no reliable advantage over the best relevant fixed or simple policy. Publish this outcome if observed. A useful analysis of why the oracle opportunity cannot be learned is a valid release.

Adaptive sparse/dense/hybrid selection and adaptive reranking already have direct precedents. Read [Arabzadeh et al.](https://arxiv.org/abs/2109.10739) and [Genc et al.](https://arxiv.org/abs/2606.25249) before claiming novelty. The combination, domain-transfer study and artifact system are proposed contributions to investigate, not established gaps in the literature.

## Datasets and access rules

Use the complete corpus for each selected BEIR dataset. Do not improve apparent speed by removing corpus negatives. The approximate sizes below come from the [BEIR dataset inventory](https://github.com/beir-cellar/beir/wiki/Datasets-available); verify downloaded counts and checksums independently.

| Dataset | Approximate corpus / test queries | Role |
| --- | --- | --- |
| SciFact | 5,000 / 300 | Small development domain; matched-domain final test |
| NFCorpus | 3,600 / 323 | Second source domain with official train/dev; matched-domain final test |
| FiQA-2018 | 57,000 / 648 | Financial held-out domain; do not use its train/dev labels |
| SCIDOCS | 25,000 / 1,000 | Held-out citation-retrieval task; do not tune on its labels |

These datasets support a manageable cross-task study, not a claim about web-scale search. SciFact and SCIDOCS share scientific subject matter, so do not describe all four as independent unrelated domains.

Data protocol:

1. Synthetic, hand-labeled documents are the CI smoke dataset; never report them as research evidence.
2. For SciFact, split its official training queries into 80% train and 20% validation with seed 17. Group available related claim identifiers/duplicate normalized query text before splitting; record exact IDs and any grouping limitations. Do not split its official test set for development.
3. For NFCorpus, use its official train/dev partitions. Audit duplicate query text and available query-family identifiers across them. Record any removed development duplicates without changing the benchmark's final query list.
4. Train policies and choose features only from source training data. Select thresholds, budget parameters, fusion settings and model variants on source validation data. Weight source domains equally in development selection rather than letting one query count dominate.
5. Keep all four official test judgment sets outside development evaluation. FiQA/SCIDOCS query text is also excluded from feature design, tuning and manual failure analysis until G5. Their corpora may be indexed with frozen preprocessing; that is normal retrieval setup, not label access.
6. Freeze ID lists, downloaded archive hash, extracted corpus/query/qrels hashes and transformations. If official partitions differ from expectations, make a decision record before continuing.
7. Analyze overlap with pretrained-model training sources from the model cards and papers. “Held out” means unseen by AtlasIR's policy-development process; it does not guarantee never seen during pretraining.

During development, use a loader API that requires an explicit role (`train`, `validation`, `final`) and rejects final judgments in tuning functions. This is an accidental-leakage guard, not an absolute security boundary.

Check dataset/model terms before redistribution. Public download access alone does not grant permission to rehost the data. Archive allowed inputs; otherwise preserve hashes and a precise acquisition recipe and state the resulting availability dependency.

## Input and ranking conventions

- Preserve original document and query IDs as strings. Store a reversible FAISS row-to-ID mapping and its hash.
- Use `title + newline + text` for documents when both fields exist; no chunking in v1. Record UTF-8 handling and empty-field rules.
- BM25 analyzer: explicit lowercasing, tokenization, stemming/stopword choice, IDF variant, `k1` and `b`. Start with a documented fixed configuration; allow a small predeclared validation grid. BM25S being configured with a Lucene-like formula does not make its analyzer, engine or timing identical to Lucene.
- Dense default: `intfloat/e5-small-v2`, full model/tokenizer revision IDs, `query: ` / `passage: ` prefixes, normalized vectors, float32 exact inner-product FAISS search. Pin pooling and token limits. These prefixes are required by the [model card](https://huggingface.co/intfloat/e5-small-v2).
- Record dense and cross-encoder truncation separately. Report how often text is truncated. Long scientific queries/documents can behave differently from short search queries.
- Retrieve 1,000 candidates per constituent retriever, or all documents if fewer. RRF uses the union then retains 1,000. Specify absent-document handling and ties explicitly.
- Resolve score ties by ascending original document ID; test the evaluator's ranking/tie behavior. Export scores consistent with saved rank order so evaluator tie-breaking cannot silently change rankings.
- Follow dataset-specific self-match conventions, including BEIR's identical query/document ID filtering where applicable; freeze the rule before scoring and apply it to every method. Do not confuse this with deleting relevant documents generally.
- Save unique ranked IDs, with no duplicate document per query. Missing results stay missing; no silent query exclusion or successful-query-only averages.

## Fixed methods and controller action space

| Family | Required variants |
| --- | --- |
| Lexical | BM25 with documented analyzer and parameters |
| Dense | E5-small-v2 plus exact FAISS; no ANN approximation initially |
| Hybrid | RRF with initial constant 60; validation-tuned convex score fusion as a comparison |
| Fixed reranking | Each chosen BM25/dense/hybrid ranking with cross-encoder depth 20 and 100 |
| Simple selective methods | Query-length rule, post-BM25 confidence rule, matched-rate random reranking |
| Adaptive | Reranking-only router, query-only full router, staged full router |

Use one compact `cross-encoder/ms-marco-MiniLM-L6-v2` checkpoint initially, pinned to a revision. This limits model-dependent confounding and compute. RRF and convex score fusion should both be reported; choose the main hybrid definition on source validation data before training the final full policy. For convex fusion, freeze normalization, missing scores and weight grid; never tune on target-domain labels. See [fusion analysis](https://arxiv.org/abs/2210.11934).

For a reranked list, place the cross-encoder-sorted top 20 or 100 first, followed by the untouched candidate tail in its original order. Do not directly mix cross-encoder and BM25 scores. Recall@100 is unchanged by reordering within the original top 100; use this as an invariant, not evidence of a reranking improvement.

The final full controller has nine available actions: each of BM25, dense and the chosen hybrid, with reranking depth 0, 20 or 100. Constrain this catalog only through an explicit documented revision. Hyperparameter comparisons do not each become a new action.

## What the policy can observe

Query-only features: token/character counts, numeric or quoted token indicators, term-IDF summaries from indexed corpus statistics and vocabulary coverage. Feature computation is measured. No query IDs, dataset IDs, ground-truth judgments or future method outcomes as inputs.

Staged features: first run BM25, then use normalized score gaps/spread and retrieval statistics in addition to query features. The BM25 work is charged on every path, including a subsequently chosen dense path. If hybrid is selected, reuse the already computed lexical candidates. Features requiring dense retrieval cannot be called cheap pre-retrieval features.

Start with standardized logistic regression for action labels. Compare with one small tree regressor/classifier (for example HistGradientBoosting) with a small fixed validation grid. For a budget-conditioned utility model, predict per-action quality and incremental cost, then choose the highest predicted quality among actions predicted feasible under the configured budget. Use training outcomes only to fit predictors. Fit imputers/scalers on training data only. Use seeds 17, 29 and 43 for learned/randomized methods; report all, not the best seed.

For sequential policies, model shared and incremental costs once. Do not count reused lexical work twice or erase its cost. Train from measured source-domain query/action outcomes; inference observes only allowed features and configured budget.

### Concrete initial learning recipe

For training query `q` and action `a`, save `Q(q,a) = nDCG@10` and `C(q,a) = measured mean warm latency in milliseconds` from source training measurements. Establish the cost profile before creating labels. For staged policies, these costs include the obligatory BM25/features and subsequent incremental work.

Define `c_ref` once as the median training cost of the chosen fixed high-quality reference. For each penalty `lambda` in the initial grid `{0, 0.01, 0.03, 0.1, 0.3, 1}`, create the label maximizing `Q(q,a) - lambda * C(q,a)/c_ref`. Break utility ties in favor of lower cost, then a fixed action order. This label uses training judgments; the fitted policy never receives them at inference. Save class counts and check whether a constant action explains performance.

Fit multiclass logistic regression with standardized allowed features, L2 regularization `C=1` initially, and balanced source-domain sample weights. Fit the alternative small tree on the same labels with a frozen shallow-depth grid, for example `{2, 4}`. Select one model/penalty per latency budget using source validation quality and actual executed mean latency. For infeasible choices, report no feasible policy rather than relabeling the budget. When a label set contains only one action, use the corresponding constant policy and report that outcome.

This recipe is the required initial policy. The per-action quality/cost predictor described above is a follow-up model only if the development evidence justifies it; it does not replace the label-based baseline. All grids become explicit configuration files before their runs, and changes need a decision record. A budget-specific selected model need not satisfy every other budget without reselection.

Confidence heuristics use thresholds at a fixed set of source-validation score-gap quantiles, selected under the same validation-budget rule. Random routing uses action probabilities frozen from a selected validation policy, with recorded seeds and equal expected action allocation. For selective reranking, this is matched expected reranking rate; for the full controller, include a matched action-mixture random comparator. Report realized action frequencies and costs because matching rates does not guarantee identical latency.

## Objectives, budgets and oracle analysis

Primary effectiveness metric: mean nDCG@10, with the exact evaluator gain/discount convention fixed and recorded. Secondary: Recall@100, Recall@1000 and MRR@10 with a stated binary relevance threshold, normally relevance > 0. Preserve graded judgments for nDCG. Explicitly record no-relevant-document handling and evaluator provider/version.

The [Day 3 contract](decisions/0004-data-evaluation-contract.md) fixes the initial implementation to linear-gain nDCG with log2(rank+1) discount, grade >= 1 for binary metrics, explicit pytrec_eval provider, and zero scores for missing/no-positive queries in the fixed partition denominator. MRR@10 truncates the ranking before reciprocal rank. Any change to these conventions requires a new evaluator/protocol identity.

Primary comparison: quality versus measured mean end-to-end latency, with p50, p95 and violation rate reported alongside. A policy optimized for mean latency does not guarantee p95 or a per-query deadline.

Before the final test, choose three hardware-specific mean-latency budget values in milliseconds from source validation pilots spanning the fixed frontier. Store their exact values and hardware profile in the frozen protocol. Select one operating point as primary in advance. Do not choose a preferred budget after observing test results.

Operational reporting objective: at least 25% less mean latency with at most 0.01 absolute nDCG@10 loss relative to the validation-selected high-quality fixed reference. This is a proposed practical threshold, not a predicted result. Report the entire frontier even when this target is missed. A quality-preservation claim needs the paired uncertainty interval to support the tolerance; lack of significance is insufficient.

At each budget, compare with (a) the best fixed method selected as feasible on validation and (b) the frozen high-quality reference. If a comparator violates the budget on final data, show that violation. Test-wise best fixed results can appear as a descriptive envelope, clearly identified as retrospective.

An offline oracle uses true per-query quality and measured costs to estimate available headroom. For a mean budget, solve the finite query/action allocation problem under total budget, or sweep a quality-minus-cost penalty and clearly label that oracle frontier as an approximation. Independent cheapest-within-per-query-budget selection answers a different question and must not be mislabeled as a mean-budget optimum. Oracles may use ground truth only for analysis/training-label construction; they are never deployable baselines.

Per-query oracle curves ignore real prediction errors and may omit routing overhead; state these assumptions. Freeze quality/cost normalization constants from training, not final labels. A small oracle advantage means little room for a learned policy; document this before adding model complexity.

## Experiment register

| ID | Experiment | Main decision / output |
| --- | --- | --- |
| E00 | Synthetic fixture and run preservation | Validate metrics, ranking invariants and fail-on-overwrite |
| E01 | BM25 source baseline | Confirm data and analyzer correctness |
| E02 | Dense source baseline | Check prefixes, truncation, ID mapping and exact search |
| E03 | Hybrid source baseline | Compare RRF and convex fusion; freeze chosen hybrid |
| E04 | Fixed reranking depths | Measure marginal gain and actual inference work |
| E05 | Query/action oracle and feature analysis | Establish learnable opportunity and cost assumptions |
| E06 | Simple policies and random allocation | Set credible cheap comparisons |
| E07 | Selective reranking model | Test the first adaptive decision in isolation |
| E08 | Full retrieval/reranking policy | Test query-only and staged routing |
| E09 | Development ablations | Remove feature groups, vary budget and compare learning curves |
| E10 | Frozen final benchmark | All fixed comparators and selected policies, all four datasets |
| E11 | Recovery and portability | Re-evaluate archived rankings, rebuild baseline, separate hardware profiles |
| E12 | Demo and result explorer | Present pinned evidence and representative failures |

Experiment families contain multiple uniquely identified runs. A repetition, corrected run or new seed is a new run, with a parent reference. Every meaningful run uses the same archival contract from E00 onward.

Minimum final ablations: query-only versus staged features; routing-only versus reranking-only versus combined; learned versus heuristic/random; depth 20 versus 100; one small source-training learning curve. Define learning-curve query subsets and all feature groups before final evaluation. Development learning curves use source data; no repeated target-domain tuning.

## Runtime measurements

Use the [compute protocol](COMPUTE.md). Separate four things: indexing; batched offline scoring; online single-query execution; optional throughput under an explicit concurrency level. Cached rankings or pair scores can accelerate quality experiments but never establish live latency savings.

Runtime includes preprocessing, query encoding when used, retrieval, fusion, policy features/inference, reranking and result assembly. Dataset/model downloads and initial model/index loading are excluded from steady-state timing and reported separately as startup costs. Index time/size, peak RAM/VRAM and resident memory of all loaded alternatives are recorded; a router may save compute while retaining both indexes/models.

Start with concurrency 1 and warm loaded models/indexes. Randomize method order across repeated blocks. Save the exact query order and raw samples, synchronize GPU work around timers and fix thread counts. Benchmark every compared method within the same hardware/session block. Compare platforms in separate tables.

Predicted budget compliance is not a hard latency guarantee. Timeouts/failures receive an explicit status and follow a predeclared quality/latency policy; never drop them from aggregates. A fallback is evaluated as a distinct policy that includes work already spent.

## Statistical analysis and final-test discipline

- Preserve query-level scores and calculate paired differences on identical query sets. Report per-dataset results and an equal-dataset macro average; also show the two held-out datasets separately.
- Use a seeded paired bootstrap over queries within each dataset for 95% intervals (10,000 resamples). If dependent query families exist, resample groups. Do not treat repeated latency measurements as new relevance observations.
- For a formal main quality comparison, use a predeclared two-sided paired randomization test and Holm adjustment across the predefined comparisons. Predeclare the comparison family at G5. Report effects and intervals even without p-values. Read [Urbano et al.](https://arxiv.org/abs/1905.11096) before implementing statistical tests.
- Report training-seed variability separately from query-sampling uncertainty. Three seeds do not prove stability on all domains.
- Show latency uncertainty per repeated block/session and identify hardware. With only one session, acknowledge that uncertainty excludes between-session variability. Do not pool T4/P100/other devices into one p95.
- Freeze method choices, grids, seeds, budgets, split IDs, evaluator and plot recipes at G5. The final pass at E10 evaluates all predefined comparators, including failures.
- After test results are opened, explanations are allowed; further tuning becomes a new exploratory protocol version. A correctness fix gets a new run, a supersedes/invalidates link and a disclosed reason. It does not restore an untouched test set.

Do not claim “state of the art,” universal dominance, production readiness or strict deadline guarantees from these experiments. Every claim in the final report must point to archived run IDs and the relevant scope.
