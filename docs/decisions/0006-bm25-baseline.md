# 0006: explicit CPU BM25 and schema-2 exploratory runs

Date: 2026-10-16. Status: accepted for the initial baseline.

Use BM25S 0.2.14 with all analyzer/scoring choices resolved in configuration. Sort original string document IDs and break score ties explicitly. Preserve positive-score candidates and filter self-matches before top-k. Store the eager sparse index and raw rankings rather than reporting metrics alone.

Use a separate versioned baseline schema while retaining the E00 fixture contract. Inputs, source, config and runtime determine the specification identity; each execution gets a new run ID. Baseline runs remain exploratory (`reportable: false`) until the research/reporting gates are satisfied. Timings are single-pass CPU pilots, not final latency evidence.

For this small SciFact pilot, include raw input bytes in each sealed bundle to make restore/re-evaluation self-contained. Preserve data attribution. Do not generalize this duplication to large corpora; the planned shared-object/shard design and tracking integration are still required. The independent archive restore gate remains open. See [Day 4 guide](../DAY04-BASELINE.md).
