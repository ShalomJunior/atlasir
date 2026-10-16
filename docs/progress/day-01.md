# Day 1 progress

Date: 16 October 2026. Status: local technical setup verified; account-specific runtime/storage inventory pending.

## Scope

Set up a lightweight installable package, locked Python environment, local checks and CPU CI fixture. Preserve the eight-week research scope in [ADR 0001](../decisions/0001-kickoff.md). Reading notes are in [Day 1 reading](../reading/day-01.md).

The synthetic fixture checks package installation and internal document/query references. It is not a search baseline, relevance evaluation, E00 archived experiment or performance result. Immutable experimental run writing belongs to Day 2.

## Local setup

- uv executable: 0.11.26.
- System Python selected for the project: CPython 3.11.6.
- Default `python3` is 3.12.3; use the project through uv.
- Available local disk at kickoff: approximately 4.1 GB; temporary and workspace storage share a filesystem.
- Agent dependency cache: `/tmp/atlasir-uv-cache`; optional `UV_CACHE_DIR` override only.

## Pending account inventory

The user has now reported a 30-hour GPU allowance and Kaggle's dual T4 option; see [ADR 0002](../decisions/0002-compute-storage-followup.md). Actual remaining quota at execution, Python/runtime environment and output capacity still need session preflight. A private existing Google Drive folder is the recommended second-copy destination, subject to available capacity and a successful restore test. No remote storage has been connected. These facts must not be inferred from local CPU checks.

## Verification evidence

| Check | Observed result |
| --- | --- |
| `uv lock` | Eight packages resolved, lockfile created with public PyPI sources and distribution hashes |
| `uv sync --locked` | Fresh Python 3.11.6 environment; atlasir 0.1.0, pytest 8.4.2, Ruff 0.17.1 installed |
| `ruff check .` | Passed |
| `ruff format --check .` | Passed |
| `pytest` | Three tests passed: installed CLI fixture, module fixture and invalid command |
| `atlasir smoke` | Success: three documents, two queries, two judgments; installation-only scope |
| `uv build` | Wheel and source distribution built; wheel built from source distribution |
| Fresh source environment | Copied source/lock into a temporary directory without the existing venv; locked offline sync, three tests and command passed using the downloaded dependency cache |
| Independent wheel environment | Installed only the built wheel into a separate venv; module smoke passed outside source checkout and packaged fixture was present |
| Hosted GitHub workflow | Authored, including independent wheel check; not run remotely |

The fresh-source check proves installation into a new environment, not a download from an empty network cache or a verified cloud runtime. No research metrics, GPU timing or archival restoration results are claimed.

The scaffold remains uncommitted in the working tree, alongside the earlier planning files; no commit, push or remote job was performed. Before a reportable experiment, preserve a source revision/snapshot as required by the reproducibility protocol. Git ignores the venv, build products, tool caches and local credentials, while leaving future adjacent `.dvc` pointers trackable.

## Owner handoff

- [ ] Review the primary question and dataset roles in ADR 0001.
- [ ] Complete the short reading checklist in the linked Day 1 notes.
- [ ] Supply Kaggle account inventory and select a second-copy destination with adequate space.

Local setup is ready for Day 2 implementation. Overall Day 1 inventory is not marked complete until the account details are recorded; the longer-run archive/restore gate remains unsatisfied.

## Day 2 handoff

Implement minimal run metadata, exclusive IDs/directories, resolved config snapshots, output checksums, sealing and a verified export/restore. Use the synthetic fixture for E00 only after that lifecycle exists. Keep GPU experiments blocked behind the archive/restore gate.
