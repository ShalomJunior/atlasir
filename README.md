# atlasir
Adaptive information retrieval: benchmarking lexical, dense, and hybrid search under relevance, latency, and compute constraints.

Status: preserved SciFact BM25 runs reproduce from historical source in a fresh environment. The Day 5 Kaggle launcher is ready; independent cloud backup and actual Kaggle execution are not verified yet.

## Local setup

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.11.26 and Python 3.11. uv selects the project interpreter from `.python-version`; activating a virtual environment is optional.

```sh
uv sync --locked
uv run --locked atlasir smoke
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv build
```

The smoke command checks a bundled synthetic CPU fixture (three documents, two queries and two judgments). It downloads no data or models, writes no experiment output and does not report retrieval quality. `uv run --locked python -m atlasir smoke` is equivalent.

Commit `uv.lock` along with dependency changes; do not regenerate it during normal reproduction. Use `uv sync --locked` to reject an outdated lockfile. In a restricted environment where the default cache is read-only, set `UV_CACHE_DIR` to a writable location, such as `"$PWD/.cache/uv"`.

The [CPU workflow](.github/workflows/ci.yml) checks linting, formatting, tests, the command and a built-wheel installation. It needs package access during environment setup but no GPU, external dataset or service. GitHub execution is not verified until the workflow runs there.

The locked project environment remains the default. Python 3.12 is also supported; the user's existing `/home/masterjs/kaggle/.venv` passed the Day 3 tests as an optional profile. Day 4 adds BM25S/PyYAML, which have not been installed or tested there. See [environment decisions](docs/decisions/0005-optional-shared-environment.md). To use it explicitly, activate it and run `python -m atlasir`; do not synchronize that shared environment with `uv sync`.

See the [Day 1 progress record](docs/progress/day-01.md), [kickoff decisions](docs/decisions/0001-kickoff.md) and [reading notes](docs/reading/day-01.md). Account-specific Kaggle and second-copy storage details remain pending.

## Preserve an experiment fixture

```sh
uv run --locked atlasir run-fixture --config configs/fixtures/e00.json
```

This creates a uniquely identified run with source/config/input snapshots and verified completion files. Use the printed directory with `atlasir verify`, `atlasir export` and `atlasir restore`; see the [Day 2 guide](docs/DAY02-RUNS.md) for exact commands and the Google Drive restore drill. Old IDs and existing destinations are never reused by these commands. These are fixture lifecycle records, not retrieval benchmark results.

See [Day 2 progress](docs/progress/day-02.md) for the first E00 run and its local restore evidence.

## Load data and check evaluation

```sh
uv run --locked atlasir data-check --data src/atlasir/fixtures/beir --name fixture --role train
uv run --locked atlasir evaluate --data src/atlasir/fixtures/beir --name fixture --role train --rankings src/atlasir/fixtures/beir/rankings.json
```

These use four synthetic documents and handwritten rankings, not benchmark results. The [Day 3 guide](docs/DAY03-DATA.md) covers exact file hashes, source split manifests, final-data guards and metric conventions; [Day 3 progress](docs/progress/day-03.md) records what is implemented and what remains before real data experiments.

## Research plan

The [Day 4 baseline guide](docs/DAY04-BASELINE.md) covers SciFact acquisition, frozen development partitions, BM25 runs and saved-ranking recalculation. See [Day 4 progress](docs/progress/day-04.md) for measured results. The first Kaggle session is planned for Day 5's compatibility check, with embeddings in Week 2.

Start that session with the [Day 5 Kaggle guide](docs/DAY05-KAGGLE.md) and [launcher notebook](notebooks/kaggle-day05.ipynb). [Day 5 progress](docs/progress/day-05.md) records the exact prepared input ZIP and completed local rebuild checks.

The [execution plan](docs/PLAN.md) preserves the BM25 → dense → hybrid → adaptive retrieval and reranking roadmap. It is designed for eight weeks at about 30 hours/week, using Kaggle and occasional Colab compute.

- [Day-by-day schedule](docs/DAILY-PLAN.md): 40 workdays, deliverables and completion gates.
- [Research protocol](docs/RESEARCH-PROTOCOL.md): datasets, comparisons, metrics and held-out evaluation.
- [Reproducibility design](docs/REPRODUCIBILITY.md): preserve code, inputs and immutable experiment artifacts.
- [Kaggle/Colab workflow](docs/COMPUTE.md): checkpointing, exports, GPU budgets and runtime measurements.
- [Reading plan](docs/READING.md): papers, official documentation and reading questions.
