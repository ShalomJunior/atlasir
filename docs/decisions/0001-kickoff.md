# ADR 0001: Day 1 scope and execution assumptions

- Date: 16 October 2026
- Status: adopted for the local scaffold; hosted profile and backup destination pending verification
- Scope: Day 1; no experimental dataset/model downloads or GPU jobs

## Research question

Can query-dependent selection of lexical, dense and hybrid retrieval, with optional neural reranking, improve relevance under a measured latency budget compared with strong fixed pipelines, including on held-out tasks?

The roadmap remains BM25 → dense → hybrid → fixed reranking → selective reranking → full routing. Adaptive retrieval is an existing research direction; AtlasIR makes no novelty claim at kickoff.

## Decisions

| Topic | Decision | Evidence / follow-up |
| --- | --- | --- |
| Schedule | Eight weeks, about 30 human hours/week | User-provided availability |
| GPU capacity | Plan below the user's reported 30 Kaggle GPU hours/week | Actual remaining quota/device options require the account UI |
| Interpreter | Python 3.11 series, initially tested with system CPython 3.11.6 | `/usr/bin/python3.11` exists; global `python3` is 3.12.3, so use uv commands |
| Python range | `>=3.11,<3.12` until a deliberate compatibility expansion | Avoid accidentally testing a different minor version; record actual patch versions in later run manifests |
| Package | `src/atlasir`, installed CLI and module entry point | Keeps notebook launchers thin and exercises installed imports |
| Build | Hatchling 1.27.0; uv 0.11.26 for setup/CI | Build backend pinned explicitly; development resolution saved in uv.lock |
| Runtime dependencies | None on Day 1 | No reason to download torch, models, BEIR corpora or trackers before they are needed |
| CPU checks | Ruff, pytest, packaged synthetic fixture, wheel/sdist build | No GPU, service or external dataset needed during tests |
| Research sources | SciFact and NFCorpus for policy development | Official final test partitions stay outside development selection |
| Held-out tasks | FiQA-2018 and SCIDOCS | No target query/judgment tuning; later frozen corpus indexing permitted |
| Main effectiveness / cost | nDCG@10 versus actual warm end-to-end mean latency | Tail latency and budget violations reported separately |
| First archival location | Local `artifacts/` layout in this project, managed by DVC from Day 2 | Provisional until free space and dependency closure are checked |
| Second copy | Unselected; must be an existing durable drive/disk/remote | No long experiment before Day 2's verified restore |

The local filesystem had approximately 4.1 GB available at kickoff; `/tmp` shares that filesystem. That is below the plan's initial 10 GB artifact envelope. Use lightweight local development now. Decide a larger archive location or a storage budget before full input snapshots; do not delete existing user files to make space.

The implementation-time uv cache is redirected to `/tmp/atlasir-uv-cache` because the default home cache is read-only in the agent's sandbox. This is disposable dependency cache, not a research archive. It does not alter the dependency lock or experimental settings.

## Hosted runtime inventory still required

- Kaggle: available accelerators, remaining quota, runtime Python version, disk/output allowance and Internet setting.
- Storage: named second-copy destination and available capacity; verify upload/download before Day 2 is complete.
- Later: exact GPU/CPU/RAM, CUDA and driver profile from an actual session. Do not assume a T4 or any other specific device now.

The local environment cannot establish those account-specific facts. No Kaggle/Colab job, hosted notebook or storage connection has been created by this setup.

## Acceptance and change policy

A fresh locked local environment must run the installed smoke command, CPU tests, lint/format checks and build. The wheel must contain its fixture and run outside the repository. GitHub execution remains unverified until the workflow is actually pushed and runs.

Day 1 is locally complete only after those checks pass. Account inventory remains explicitly pending if not provided; Day 2's archive/restore gate cannot be claimed until the second copy is verified. Changes to datasets, interpreter profile, storage or research objectives get a new decision record rather than replacing this history.
