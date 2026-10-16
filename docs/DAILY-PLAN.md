# Day-by-day plan: eight weeks

Assumption: 30 hours of human work/week, organized as five 6-hour workdays. Day 1 is the actual kickoff, not an assumed calendar date. Kaggle GPU hours are a separate allocation. This is the schedule baseline; actual completion and pending checks are recorded in [daily progress](progress/day-01.md).

Typical day: 45–60 minutes reading/design, about 4 hours focused implementation or analysis, and about 1 hour verification, artifact export and notes. The balance changes on benchmark/report days. Stop long jobs early enough to export their outputs. Weekends are optional catch-up or rest, not hidden required workdays.

Every experimental day ends with run IDs, a short finding and the next decision. Every Friday includes a clean restore drill. Gates require evidence, not just a completed checklist. If a gate fails, revise subsequent dates rather than pretending the dependency is satisfied.

Paper identifiers P01–P12 refer to the [reading plan](READING.md). Experiment identifiers E00–E12 refer to the [research protocol](RESEARCH-PROTOCOL.md).

Implementation records: [Day 1](progress/day-01.md), [Day 2](progress/day-02.md), [Day 3](progress/day-03.md), [Day 4](progress/day-04.md), [Day 5](progress/day-05.md). These distinguish local checks from pending Kaggle and independent-backup work. Day 5's local rebuild/handoff preparation is complete; its external checks and G1 remain open.

## Week 1: establish a baseline we cannot lose

| Day | Work and reading | Concrete definition of done |
| --- | --- | --- |
| 1 | Read P01's setup/evaluation and skim the two closest papers P09/P10. Inventory Kaggle quota/device options, storage and interpreter support. Record v1 assumptions, dataset roles and primary question. Set up the package, uv lock, Ruff/pytest and CPU fixture CI. | A fresh environment runs one package command and CI fixture. Decisions about scope, runtime profile and durable archive location are written. No claims of a novel router. |
| 2 | Read official uv/DVC guides. Implement minimal run metadata, exclusive run directories, config snapshots, checksum/seal flow and durable export. Create E00 from a tiny fixture. | Reusing a run ID fails. A sealed run is restored from the second copy with hashes intact; source/config are identifiable. A long GPU job is now allowed. |
| 3 | Read P02's evaluation/BM25 sections. Implement the BEIR-format loader, data descriptors and source train/validation IDs. Add ir_measures evaluation and hand-calculated fixture checks. | Missing/duplicate IDs and split overlap are detected. A known tiny ranking gives the expected nDCG, recall and reciprocal rank. Final-test data is excluded from the development API. |
| 4 | Implement BM25S adapter with explicit analyzer/variant. Run E01 on SciFact training/validation queries only; record corpus hash, index metadata and query outputs. | Stable IDs and valid top-k rankings; first real per-query metric table, indexing record and warm CPU latency pilot. |
| 5 | Reproduce E01 from a clean source checkout. Add end-to-end fixture CI and restore tests; record a small baseline table and limitations. Perform a short Kaggle GPU/environment compatibility check. | **G1:** BM25 baseline is repeatable, its run is backed up, metrics can be recalculated from rankings and the notebook can execute the locked package. |

If day 1 setup takes longer, retain the day 2 archive guarantee before full-data runs. A short CPU diagnostic may precede complete MLflow integration, but it still needs the minimal sealed run record.

## Week 2: dense and hybrid baselines

| Day | Work and reading | Concrete definition of done |
| --- | --- | --- |
| 6 | Read P03 and P04's encoder usage. Implement E5 query/document encoding with pinned model/tokenizer, prefixes, truncation and normalized vectors. | A small corpus has finite embeddings, explicit IDs and matching repeated encodings within the declared environment tolerance. |
| 7 | Read FAISS documentation. Pilot throughput; build exact inner-product indexes and a hashed row-ID map. Implement shard checkpoints and export/resume for embeddings. | Interrupted pilot resumes using verified shards. Full source-corpus embedding cost fits the revised GPU/storage estimate. |
| 8 | Run E02 on both source domains. Add common source query coverage, raw stage timers, memory/index measurements and BM25/dense comparison. | Both methods evaluate through the same runner. Prefixes, ID mapping and truncation have explicit checks. No target test judgments used. |
| 9 | Read P05/P06. Implement RRF and convex fusion with declared normalization/missing-score behavior. Run E03 validation comparisons using a small fixed grid. | Fusion passes ordering/deduplication checks. Tables compare lexical, dense, RRF and tuned score fusion; parameter choices trace to validation data. |
| 10 | Add offline MLflow ingestion from run bundles and a comparison query/export. Finish baseline documentation, cache-dependency checks and clean restore. | **G2:** fixed lexical/dense/hybrid quality and pilot costs are archived. MLflow can be rebuilt from canonical records. Earlier E01 still restores after new code/config changes. |

## Week 3: fixed reranking, headroom and prior work

| Day | Work and reading | Concrete definition of done |
| --- | --- | --- |
| 11 | Read P07 and reranker model documentation. Add compact cross-encoder adapter; pilot pair throughput and sequence-length behavior. | Depth 20/100 reranking works with pinned checkpoint and fixed batching. Candidate tail and Recall@100 invariants are tested. |
| 12 | Run E04 for the three retrieval bases on source train/validation data. Cache identical offline query/document pairs and export completed shards. | Source action-quality matrix exists for depths 0/20/100. Failed pairs/queries are explicit; fixed-method rankings remain accessible. |
| 13 | Implement complete online cost instrumentation and repeated timing blocks. Measure all source fixed strategies on the same device. Start index/memory table and fixed Pareto plots. | Feature/query encoding/fusion/reranking costs are distinguishable. Offline cached scoring is excluded from latency claims; actual fixed-frontier measurements are saved. |
| 14 | Read P09/P10 in depth. Run E05: query-level outcome variation, mean-budget oracle approximation, action distributions and available headroom. | A written opportunity assessment says whether adaptivity has room to help and identifies assumptions. No model is trained on final-domain data. |
| 15 | Read P08 and P11. Build the related-work matrix, identify one transferable labeling baseline and draft experiment proposals for routing. Restore older runs and review the GPU forecast. | **G3:** valid fixed frontier, documented oracle and a method-level comparison with closest prior work. If headroom is small, write a negative-result route while retaining planned controller comparisons. |

## Week 4: first adaptive result, selective reranking

| Day | Work and reading | Concrete definition of done |
| --- | --- | --- |
| 16 | Implement E06: never/always rerank, fixed depth 20, query-length/confidence heuristics and matched-rate random reranking. Choose thresholds on validation only. | Cheap comparisons have run IDs, costs and explicit feature access. Random allocations use recorded seeds and include their uncertainty. |
| 17 | Implement E07's training labels/features and logistic policy for reranking versus skipping. Keep the base retriever fixed to the validation-selected reference. | Training pipeline fits only on source train data; source validation measures actual policy quality and executed latency. |
| 18 | Compare one small tree model; add predicted gain/cost or budget-conditioned selection. Fit only a small predeclared grid and repeat learned runs with three seeds. | Training/validation records identify each feature set, label definition and seed. Router overhead and trivial-action/class-imbalance checks are present. |
| 19 | Compare E07 with E06 and fixed depth baselines at matched budgets. Run initial feature removal and error inspection on development queries. | A table establishes whether learning adds anything beyond simple reranking allocation, including cases where it loses. |
| 20 | Harden feature extraction, fallback/failure accounting and model serialization. Clean restore the old fixed baseline and the selected policy. Write an interim one-page findings note. | **G4:** selective reranking is assessed honestly and is rerunnable. This is the first controller milestone, not a replacement for full retrieval routing. |

## Week 5: full adaptive retrieval and final protocol freeze

| Day | Work and reading | Concrete definition of done |
| --- | --- | --- |
| 21 | Implement E08 query-only routing over BM25/dense/hybrid and depths 0/20/100, using source action outcomes. Compare source-only labels with the chosen P09/P10-inspired baseline adaptation. | Full nine-action catalog works. No query/dataset IDs or relevance judgments enter inference features. Adaptations from papers are explicitly documented. |
| 22 | Add staged post-BM25 routing and reuse lexical work for hybrid paths. Charge sunk lexical cost to every action. | The execution trace accounts for every executed stage exactly once; query-only and staged policies have separate results. |
| 23 | Select three absolute mean-latency budgets from source validation and fix one primary point. Evaluate predicted budget compliance versus actual cost, tail violations and out-of-budget cases. | Budget values/profile and selected policies are written before final testing. No claim that predicted costs guarantee a deadline. |
| 24 | Run E09 development ablations: retrieval-only/reranking-only/combined, feature groups, depth and small source-training learning curve. Produce preliminary paired intervals. | Planned ablations have valid input references and frozen subsets. Complexity is justified only if it improves the development evidence. |
| 25 | Freeze E10 method matrix, split IDs, seeds, checkpoints, configs, statistical comparison family, timing settings and latency fallback subset. Audit all prior final-data access. Restore an early run. | **G5:** a versioned final protocol and executable suite exist. Models and all choice rules are frozen; final test judgments have not been used to choose them. |

## Week 6: final evaluation and evidence freeze

| Day | Work and reading | Concrete definition of done |
| --- | --- | --- |
| 26 | Audit E10 using only fixture/source data. Run a fresh locked-environment check. Prepare/hash complete FiQA and SCIDOCS corpora and index them using frozen preprocessing; finalize allowed input exports. | Final suite preflight passes, quota/storage forecast fits, and every input and policy has an immutable identity. |
| 27 | Run E10 quality evaluation on the four official final query sets for all predefined fixed methods and selected policies. Export incrementally. | Full rankings and per-query outcomes are sealed; coverage/failure counts are visible. Opening final outcomes starts the no-further-tuning period. |
| 28 | Execute E10 live latency blocks on common Kaggle hardware. Use the frozen subset fallback if required. Measure warm query latency, startup/indexing and loaded memory separately. | Same-session comparisons exist with raw timings and device identity. Cached quality results were not used to simulate service latency. |
| 29 | Run E10 statistical analysis and E11 cross-environment metric/ranking checks. Colab is optional and its timing remains separate. | Paired effects/intervals, budget compliance, source/target breakdown and seed variability are generated directly from archived records. |
| 30 | Validate conclusions, freeze the evidence catalog and restore the complete report input set from backup. Rebuild one baseline from its historical source revision. | **G6:** final evidence is archived and recoverable. Bugs trigger corrected new runs and schedule revision; unfavorable findings do not trigger test-set tuning. |

Week 6 has the largest GPU allocation. If necessary, separate quality and timing jobs across sessions while retaining complete same-session method comparisons for each timed block.

## Week 7: report and review

| Day | Work and reading | Concrete definition of done |
| --- | --- | --- |
| 31 | Draft the 6–10 page report: problem, related work, method and protocol. Update bibliographic details from the primary papers. | Every method difference and evaluation limitation is understandable without reading the source code. |
| 32 | Generate final tables, Pareto plots, budget violations, action proportions and resource footprint from explicit run-ID lists. | Each table/figure records data inputs, aggregation revision and artifact hash. No manual spreadsheet metrics. |
| 33 | Analyze successes and failures using final outcomes. Show at least one win and one loss; inspect truncation, missed candidates, policy mistakes and domain shift. | Claims distinguish exploratory interpretation from preregistered comparisons, and acknowledge pretrained-data uncertainty. |
| 34 | Perform a skeptical methodology/code review, optionally using feedback from an available engineer/researcher. Prepare a concise technical-feedback request for the user to send if desired. | Review findings and dispositions are recorded. No unsolicited external messages are sent; no reviewer response is required to proceed. |
| 35 | Integrate review, archive any corrected analysis as derived runs and verify the report's references. Produce release-candidate documentation and restore it cleanly. | **G7:** report, evidence links, installation/reproduction guide and limitations are complete enough for an independent technical reader. |

## Week 8: demonstration, independent use and buffer

| Day | Work and reading | Concrete definition of done |
| --- | --- | --- |
| 36 | Design E12's small demo around pinned corpus/index/policy objects. Add result inspection and a static benchmark explorer sourced from E10 records. | Queries can expose selected strategy, ranked IDs/snippets and a stage trace. Precomputed benchmark numbers are labeled with hardware/run IDs. |
| 37 | Finish the local demo and core usability checks. Test empty/long queries, missing objects and absent GPU. Provide a recorded/screenshot fallback. | A user can start it from documented instructions. Any CPU-only timing is labeled separately; no LLM answer generator is added. |
| 38 | Perform E11's clean-room reproduction: smoke, saved-ranking metrics, historical BM25 rebuild and demo start. Use another machine/person if available. | A reproduction log records commands, observed differences, hardware and artifact access. Unavailable independent replication is disclosed. |
| 39 | Record or script a five-minute technical walkthrough; draft a short project summary and CV bullet using measured results. Package the report, plots, demo and evidence manifest. | Explain one win, one loss, one design decision and one limitation. Numerical claims have traceable sources. |
| 40 | Reserve for defects and final archive/release audit. Prepare a versioned release candidate with license/citation/model-data attribution and a clear known-issues list. | **G8:** all core deliverables are reviewable and recoverable. Publishing/tagging/remote updates occur in the subsequent authorized workflow; nothing is claimed published before it is. |

## If the schedule slips

Reduce optional work in this order: Colab comparison, extra model experiments, cosmetic demo changes, advanced benchmark interactivity and prospective submissions. Retain a minimal interactive search demo plus static result explorer. Do not remove provenance, basic tests, fixed-depth comparisons, actual runtime measurements, full routing assessment or held-out-domain evaluation without a clearly recorded scope change.

The single final buffer day and weekly review slots cannot absorb every infrastructure or research problem. If a required gate fails, use the next available workday to fix it and move subsequent dates. Thirty hours/week is a planning commitment, not a guarantee that research will resolve positively within eight weeks.

## After the core release: French extension

Allocate 5–10 further workdays: verify MIRACL-fr corpus/splits/terms and compute size; introduce a pinned multilingual model/reranker; reproduce French BM25/dense/hybrid baselines; freeze a language-transfer question; evaluate without using final labels for tuning; archive and update the report/demo. This is an explicit additional phase, not an implied result of the English benchmark.
