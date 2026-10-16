# E00: Preserve and recover an experiment fixture

- Date: 16 October 2026
- Status: proposed; outcome recorded in the Day 2 progress log
- Protocol: `day02-fixture-v1`
- Question: can the exact source, resolved config, input and output of a completed small run be restored without relying on the original working run directory?
- Hypothesis: a sealed bundle round-trips byte-for-byte, and duplicate IDs or corrupted inputs cannot replace a previously completed run.
- Falsification: a missing/corrupted file is accepted, a repeated ID overwrites old evidence, or restore requires the old run/source directory.
- Input: bundled `synthetic-v1`, three documents, two queries and two judgments; no benchmark/test-set data.
- Configuration: `configs/fixtures/e00.json`, seed 17, no randomized computation.
- Execution: Python 3.11 CPU, source snapshot and installed package versions saved per run; dirty source allowed and explicitly non-reportable.
- Outputs: source/fixture snapshots, reference counts, resolved config, provenance, event record and checksum/completion files.
- Cost: seconds of CPU time; no GPU allocation or paid service.
- Checks: active/completed ID collision, unfinished/failing runs, content corruption, independent local restore and safe extraction.
- External archive: recommended private Google Drive folder; actual account destination and restore still pending.
- Completion: local technical lifecycle and external independent-backup evidence reported separately; no retrieval quality claim.
