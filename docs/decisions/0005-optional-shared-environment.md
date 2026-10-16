# ADR 0005: Keep the isolated default; support the existing Kaggle environment

- Date: 16 October 2026
- Status: optional shared profile validated; project-local locked environment remains the default
- Supersedes: ADR 0001's Python 3.11-only compatibility restriction, not its reproducibility requirements

## Evidence

The user supplied `/home/masterjs/kaggle/.venv`, which uses Python 3.12.3 and already includes torch, Sentence Transformers, Transformers, FAISS and scientific libraries. The selected version inventory is [existing-kaggle-local.json](../../configs/environments/existing-kaggle-local.json). Its library versions differ from the project lockfile; it is not a locked AtlasIR environment and its path/name does not establish actual Kaggle GPU availability.

The additive compatibility check installed only AtlasIR (editable) and ir-measures 0.4.3 there, using `--no-deps`. Existing torch/NumPy/SciPy/pytrec_eval and other libraries were neither upgraded nor removed. BM25S is still absent. Python support was broadened to 3.11/3.12, and CPU CI now has both versions. The default `.python-version` stays 3.11.

Observed checks: all 79 tests passed under the shared Python 3.12.3 environment, with unrelated pytest plugin autoload disabled; smoke and verification of the original E00 bundle passed. All 79 tests and Ruff lint also passed in the project-local locked Python 3.11.6 environment. GitHub matrix execution has not run remotely.

## Recommendation

Keep `uv sync --locked` and project `.venv` as the reproducible development/reference workflow. Use the existing environment optionally for later GPU exploration to avoid downloading large libraries again. Do not point `uv sync` at the shared environment: exact synchronization may remove unrelated packages. Do not mix results/timing from the two profiles without recording their different environments and evaluator versions.

For the optional shared profile, activate it and use `python -m atlasir` / `python -m pytest` explicitly; plain `uv run` follows the project's interpreter/environment settings. New dependencies must be added deliberately without broad upgrades. Before promoting that profile to a research reference, define a locked GPU environment and capture hardware/CUDA details separately.

No migration of existing run artifacts occurred. Supporting Python 3.12 does not make old results invalid; each saved run retains its historical environment and source identity.
