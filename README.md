# AtlasIR

Adaptive information retrieval: benchmarking lexical, dense, and hybrid search under relevance, latency, and compute constraints.

Fixed SciFact BM25, E5-small-v2 and reciprocal-rank fusion comparisons have run on Kaggle using frozen development splits. Validation nDCG@10 is 0.6386, 0.7054 and 0.7095 respectively. These are exploratory source results; adaptive policies and held-out evaluation remain future work. See the [findings and numerical reproducibility limits](docs/progress/scifact-comparison-findings.md).

## Local setup

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.11.26. The project supports Python 3.11 and 3.12, with 3.11 selected by default.

```sh
uv sync --locked --extra search
uv run --locked --extra search atlasir smoke
uv run --locked --extra search pytest
uv run --locked --extra search ruff check .
uv run --locked --extra search ruff format --check .
uv build
```

The smoke command uses a bundled synthetic CPU fixture and downloads no datasets or models. The `search` extra enables FAISS tests. Neural encoding requires the optional `dense` extra; use the isolated locked environment for experiments. Avoid synchronizing unrelated shared environments. Commit `uv.lock` with dependency changes and use `--locked` during reproduction.

The [CPU workflow](.github/workflows/ci.yml) runs lint, formatting, fixture tests, package builds and an installed-wheel smoke check. It does not need a GPU or external benchmark data during tests.

## Experiments and reproduction

Runs receive unique IDs and retain exact source, input and configuration identities. Completed runs are sealed with checksum inventories. Existing IDs and destinations cannot be overwritten. Large data, models and vectors remain outside Git; catalogs and verification receipts identify the archived results.

```sh
uv run --locked atlasir run-fixture --config configs/fixtures/e00.json
uv run --locked atlasir data-check --data src/atlasir/fixtures/beir --name fixture --role train
uv run --locked atlasir evaluate --data src/atlasir/fixtures/beir --name fixture --role train --rankings src/atlasir/fixtures/beir/rankings.json
```

- [Artifact lifecycle](docs/DAY02-RUNS.md): create, verify, export and restore a run.
- [Data and evaluation](docs/DAY03-DATA.md): input hashes, frozen splits, metrics and final-data guards.
- [BM25 baseline](docs/DAY04-BASELINE.md): SciFact preparation and saved-ranking recalculation.
- [E5 encoding](docs/DAY06-ENCODING.md): pinned checkpoint and preprocessing.
- [Checkpoints and vector indexes](docs/DAY07-SHARDS.md): verified shards and separate binary objects.
- [Full SciFact comparison](docs/DAY08-COMPARISON.md): one Kaggle session for BM25, dense, RRF and descriptive query-level headroom.

The [comparison notebook](notebooks/kaggle-scifact-comparison.ipynb) delegates implementation to the package. Model encoding uses an explicitly selected GPU; BM25 and exact FAISS search use CPU. Preserve the entire session export, including the object ZIP.

## Research documentation

- [Research protocol](docs/RESEARCH-PROTOCOL.md): development/held-out datasets, fixed and adaptive comparisons, metrics and evaluation safeguards.
- [Reproducibility design](docs/REPRODUCIBILITY.md): canonical artifacts, provenance and planned tracking integrations.
- [Execution protocol](docs/EXECUTION.md): isolated hosted execution, recovery and measurement boundaries.
- [Research references](docs/REFERENCES.md): bibliography behind the method choices.
- [SciFact findings](docs/progress/scifact-comparison-findings.md): fixed-method results and descriptive routing opportunity.

Personal schedules, working notes and machine-specific setup are excluded from the public source tree and source archives. Public methodology and experimental evidence remain versioned.
