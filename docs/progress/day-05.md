# Day 5 progress

Date: 16 October 2026. Status: local historical rebuild and Kaggle handoff checks passed; actual Kaggle and independent Drive restore remain pending. G1 is not yet satisfied.

## Reproduction evidence

`atlasir reproduce` extracts a verified historical source/input/config snapshot and runs it in a fresh locked environment. The saved Day 4 validation source rebuilt all **162 queries** with identical metrics, ranked IDs and scores. The rebuilt source archive SHA256 matched the original. This is a clean environment on the same machine, not independent-machine replication; CPU timings were not compared.

See [the historical event](../../experiments/events/day05-historical-rebuild.json) for original/rebuilt IDs and tolerances. The rebuilt run is under `.scratch/day05-historical-rebuild/output/`; its comparison receipt is `.scratch/day05-historical-rebuild/comparison.json`. Original sealed runs were not changed.

## Kaggle handoff

Implemented [thin notebook](../../notebooks/kaggle-day05.ipynb), pinned input packager, CPU reproduction/export script and separate runtime/CUDA probe. The [Day 5 guide](../DAY05-KAGGLE.md) gives upload, execution, download and restore instructions.

Prepared `.scratch/kaggle/atlasir-day05-input.zip` (10,219,068 bytes), SHA256 `ff58acfbd5e56701471cca7c6cf340fe01f97132efd729acec86b51051ec435d`. Its source snapshot SHA256 is `781727ad7fa03125c5cc7c53afc42926bc3994c1c30bc1ed81b66e02b85182f4`. The ZIP includes the validation reference bundle and attribution, and uses an immutable source hash independent of a moving branch. It was packaged before this progress note was added; its manifest is authoritative.

The exact input ZIP was unpacked locally and executed with its extracted source in another fresh isolated locked environment. All 162 validation query rankings/scores/metrics matched again, and a new bundle/export ZIP was produced. This verifies the CPU handoff path, not Kaggle/CUDA. The uv 0.11.26 `pip --target` bootstrap/executable layout was tested locally. The notebook has no saved outputs; all code cells compile.

The local project runtime probe found no usable GPU. The actual Kaggle notebook must run its kernel CUDA matrix check on the assigned devices. That kernel check is separate from the locked CPU runtime and does not certify future E5/reranker dependencies.

## Checks

- 92 tests passed, including end-to-end handoff/export with a synthetic reference, damaged-transfer rejection and a comparison configuration guard.
- Ruff lint/format passed; distributable build and installed-wheel smoke/reproduction comparison checks passed.
- CPU CI covers the handoff fixture through pytest; remote GitHub CI has not been run yet.

## GitHub checkpoint

Remote: `https://github.com/ShalomJunior/atlasir.git`; branch: `main`; repository visibility verified as **public**. The first substantial checkpoint contains the Day 1–5 code, plans, lockfile, fixture tests/CI, launcher and small experiment receipts. Generated data/run/index/environment files and credential locations are ignored. The local source checkpoint is prepared separately from publishing it to the public remote; push remains pending user authorization.

## Required next actions

Upload the prepared private input and notebook to Kaggle, run the short session, download outputs and verify them locally. Then carry the bundle/attribution to private Google Drive and verify a freshly downloaded copy. Record both external checks before marking G1 complete or starting long GPU jobs. Dense encoding implementation is the next research stage once those handoffs are addressed.
