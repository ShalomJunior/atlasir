"""Extract only development inputs from the pinned, separately downloaded BEIR archive.

Download URL and attribution are in docs/DAY04-BASELINE.md. This script never
extracts final-test judgments. Existing outputs are deliberately not overwritten.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

EXPECTED_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    content = args.archive.read_bytes()
    if hashlib.md5(content, usedforsecurity=False).hexdigest() != EXPECTED_MD5:
        raise ValueError("Archive differs from the BEIR-published SciFact MD5")
    with zipfile.ZipFile(args.archive) as archive:
        payloads = {
            name: archive.read("scifact/" + name)
            for name in ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv")
        }
    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination / "qrels").mkdir()
    for name, payload in payloads.items():
        (args.destination / name).write_bytes(payload)
    receipt = {
        "dataset": "BEIR SciFact",
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "archive_md5": EXPECTED_MD5,
        "archive_sha256": hashlib.sha256(content).hexdigest(),
        "extracted_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()
        },
        "final_qrels_extracted": False,
        "license_notice": "https://github.com/allenai/scifact/blob/master/LICENSE.md",
        "terms": "Claims/evidence: CC BY 4.0; S2ORC abstracts: ODC-By 1.0; code: Apache 2.0",
        "attribution": "Wadden et al., Fact or Fiction: Verifying Scientific Claims (EMNLP 2020); "
        "Lo et al., S2ORC (ACL 2020); Thakur et al., BEIR (NeurIPS 2021)",
    }
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
