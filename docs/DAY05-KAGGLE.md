# Day 5: historical reproduction and first Kaggle session

The local historical rebuild is complete. Actual Kaggle execution and independent Drive restore still require the user's account/session. Neither is inferred from the local checks.

## Historical algorithm rebuild

For a **trusted** local bundle, use its restored run directory:

```sh
uv run --locked atlasir reproduce artifacts/runs/20261016t183413z-e01-603a3450-feab55da4672 .scratch/my-historical-rebuild
```

Choose a new destination each time. `--offline` may be added when the Python/runtime/build dependencies are already cached. This executes the saved historical source, not current algorithms: only reproduce source you trust. It verifies the original run, extracts its source/data/config/selection, creates a fresh locked environment using the historical Python minor profile, runs the historical baseline and compares outcomes. It requires the historical source archive identity, identical metric/per-query records and ranked IDs, and scores within rtol 1e-6 / atol 1e-7. It also reports whether scores were exactly equal. Timings are not compared. Exact interpreter patch/platform reproduction is a separate profile, not claimed by a same-minor environment.

The Day 4 validation rebuild on this machine matched all scores exactly. Its receipt is [the reproduction event](../experiments/events/day05-historical-rebuild.json).

## Prepare the Kaggle input

Run from the project, with new output destinations:

```sh
uv run --locked python scripts/prepare_kaggle_day05.py --reference .scratch/exports/20261016t183413z-e01-603a3450-feab55da4672.bundle --attribution .scratch/exports/SCIFACT-ATTRIBUTION.md --destination .scratch/kaggle/atlasir-day05-input.zip
```

This creates a roughly 10 MB ZIP containing a deterministic source archive, hash manifest, verified validation reference bundle and attribution notice. It includes source/lock/config/docs/notebook, not the virtual environment or credentials. The reference bundle supplies the frozen selection and raw development inputs. Keep it private; retain the data terms/attribution with the inputs. This is a snapshot of current reviewed files, independent of a moving Git branch. Git commit identity will be an additional checkpoint once committed.

## Run on Kaggle

1. Upload [kaggle-day05.ipynb](../notebooks/kaggle-day05.ipynb) as a notebook. Attach the prepared ZIP as a **private** Kaggle input. Ensure it is extracted so `launcher-input.json` is visible under `/kaggle/input/`. Attach exactly one AtlasIR launcher input.
2. Enable Internet for the pinned uv tool, Python and locked CPU dependencies. Choose the available T4 accelerator option for the brief compatibility check. Check the displayed quota before starting. The notebook discovers the actual devices rather than assuming dual T4 availability.
3. Run the cells in order. Each full run creates a fresh work directory. Setup installs uv 0.11.26 in a tool directory and runs the package in an isolated Python 3.11 locked environment. Algorithms are called through the package, not copied into notebook cells.
4. The kernel CUDA probe checks a small matrix multiplication on every visible GPU. It records kernel Python/torch/CUDA/device/driver details separately from the isolated project's runtime report. This does **not** certify the future pinned E5/reranker CUDA environment; that will be tested when added in Week 2.
5. The CPU job verifies/restores the reference, recalculates metrics, rebuilds the index/rankings with the new source snapshot, compares all development outcomes and exports a newly identified run. A mismatch fails the job; do not tune using final-test judgments to fix it.
6. Download **both** `day05-baseline-export.zip` and `day05-compatibility-reports.zip` using the printed links/output browser. Keep the printed SHA256 values. Save the notebook/output version as available in the UI, then stop the accelerator.

This is intended as a brief compatibility session, not a quota-consuming training run. Setup/network time is not predicted. Do reading and CPU analysis with the accelerator off. “Save & Run All” may execute a new session; its new artifacts must have their own run IDs. See [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) and [uv's Jupyter environment guide](https://docs.astral.sh/uv/guides/integration/jupyter/) for platform/environment details.

## Download and independent backup check

Extract each trusted output ZIP into a new local directory. In the baseline ZIP, preserve `SCIFACT-ATTRIBUTION.md`, `comparison.json` and the entire `RUN_ID.bundle/`. Verify with:

```sh
uv run --locked atlasir restore PATH/TO/RUN_ID.bundle .scratch/kaggle-restored-RUN_ID
uv run --locked atlasir reevaluate .scratch/kaggle-restored-RUN_ID
```

Upload the entire bundle plus attribution to a private Google Drive `AtlasIR/archive/` folder. Download that copy to a new location and repeat restore/re-evaluation. Record the run ID, UTC time, independent location and verification result in a new event; leave sealed runs unchanged. Kaggle saved outputs plus files on this laptop do not replace evidence of the independent restore. Do not start long embedding/reranking jobs until this succeeds.

## GitHub checkpoint

The configured remote is `https://github.com/ShalomJunior/atlasir.git`. This foundation is ready for a first substantial checkpoint after review/checks: package, lockfile, plans, tests, CI, thin launcher and small provenance records. `.scratch/`, run directories, datasets, indexes, environments and credentials stay out of Git. GitHub stores the source checkpoint; the independent artifact archive stores experiment payloads. Future logical changes should be committed/pushed regularly, rather than waiting until the project is finished.

Remote publication is a separate action from preparing and checking the files. Actual branch/commit/push status is recorded in [Day 5 progress](progress/day-05.md).
