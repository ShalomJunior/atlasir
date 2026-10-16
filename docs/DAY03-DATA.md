# Day 3: local datasets, source splits and evaluation

The loader/evaluator are implemented and tested on a bundled synthetic BEIR dataset. Real SciFact/NFCorpus acquisition, license/source records and real ID manifests remain for baseline preparation. No full corpus, model or final benchmark partition has been downloaded here.

## Inspect the format

```sh
uv sync --locked
uv run --locked atlasir data-check --data src/atlasir/fixtures/beir --name fixture --role train
```

The complete corpus has four documents. The official synthetic training qrels identify four queries and five judgments; the shared query file also contains separate dev/test queries. Output records file hashes and the exact evaluated-query-set hash.

The same API supports names `scifact`, `nfcorpus`, `fiqa` and `scidocs` with the protocol's roles. It parses local files; it does not verify a dataset's claimed name against an authenticated registry. Document title/text composition is fixed, and invalid IDs, duplicate records or dangling judgments fail.

## Save a source-pool split

```sh
uv run --locked atlasir make-splits --data src/atlasir/fixtures/beir --name fixture --role train --seed 17 --output experiments/splits/my-fixture-seed17.json
uv run --locked atlasir data-check --data src/atlasir/fixtures/beir --name fixture --role train --selection experiments/splits/my-fixture-seed17.json
uv run --locked atlasir data-check --data src/atlasir/fixtures/beir --name fixture --role validation --selection experiments/splits/my-fixture-seed17.json
```

The output file must not already exist. The checked-in example is [fixture-seed17.json](../experiments/splits/fixture-seed17.json). It keeps the two normalized duplicate queries together. The target validation fraction is 20%; this four-query fixture yields three train queries and one validation query (25%).

For SciFact, load its official train pool, save this manifest with `--name scifact`, then provide it for both train and validation. Raw source-pool loading is permitted to build the manifest, but the evaluator rejects using an unpartitioned SciFact pool for development. NFCorpus uses its official train/dev directly; use `audit_disjoint` to check IDs, normalized query text, declared families and source versions. It fails rather than silently deleting overlap.

Available grouping is normalized text plus an optional `metadata.family_id`. Exact ID membership is frozen. This is not a guarantee against every semantic duplicate; actual claim metadata needs review when real data is acquired.

## Evaluate fixed example rankings

```sh
uv run --locked atlasir evaluate --data src/atlasir/fixtures/beir --name fixture --role train --rankings src/atlasir/fixtures/beir/rankings.json
```

Rankings JSON maps query IDs to lists of `{"doc_id": "a", "score": 1.0}`. The evaluator sorts by score and ID, rejects duplicate hits, wrong-partition queries, unknown documents and nonfinite scores. Missing queries remain zero-scored. The output includes per-query metrics, aggregate metrics, coverage, exact dataset hashes and evaluator conventions/versions.

Expected synthetic-only aggregate:

| Metric | Expected |
| --- | --- |
| nDCG@10 | 0.606909333640491 |
| Recall@100 | 0.75 |
| Recall@1000 | 0.75 |
| MRR@10 | 0.625 |

For the graded query, the returned grades are `[1, 3]`, while the ideal grades are `[3, 1]`. Its linear-gain nDCG is `(1 + 3/log2(3)) / (3 + 1/log2(3))`. Another query returns a relevant document at rank two, giving nDCG `1/log2(3)` and RR `1/2`. One missing zero-positive query stays in the denominator, and the fourth query is perfect. These independent calculations are checked against the backend in tests.

Final data access is deliberate: `--role final --allow-final` for loading and additionally `--purpose final` for evaluation. FiQA/SCIDOCS refuse development roles even if `--allow-final` is supplied. These flags are accidental-leakage guards, not a substitute for the G5 freeze. Do not exercise them on real held-out data during development.

## Python interface

```python
from pathlib import Path
from atlasir.data import load_beir
from atlasir.splits import make_selection, audit_disjoint
from atlasir.evaluation import ScoredDocument, evaluate

pool = load_beir(Path("src/atlasir/fixtures/beir"), name="fixture", role="train")
selection = make_selection(pool, seed=17)
train = load_beir(
    Path("src/atlasir/fixtures/beir"), name="fixture", role="train", selection=selection
)
validation = load_beir(
    Path("src/atlasir/fixtures/beir"), name="fixture", role="validation", selection=selection
)
audit_disjoint(train, validation)
```

See [ADR 0004](decisions/0004-data-evaluation-contract.md) for gain, relevance, self-match and missing-query rules. Day 4 integrates this into BM25 and archived experiment outputs. `evaluate` currently writes diagnostics only to stdout; it is not a replacement for a preserved run.
