"""Run the full frozen SciFact comparison, or recalculate a saved result without a GPU."""

import argparse
import json
from pathlib import Path

from atlasir.comparison import (
    reevaluate_comparison,
    resolved_comparison_config,
    run_comparison,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path)
parser.add_argument("--selection", type=Path)
parser.add_argument("--root", type=Path)
parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--device", choices=["cpu", "cuda:0", "cuda:1"], default="cuda:0")
parser.add_argument("--reevaluate", type=Path)
parser.add_argument("--result-json", type=Path)
args = parser.parse_args()
if args.reevaluate:
    result = reevaluate_comparison(args.reevaluate)
else:
    if not all((args.data, args.selection, args.root)):
        parser.error("--data, --selection and --root are required to run")
    result = run_comparison(
        data=args.data,
        selection=args.selection,
        name="scifact",
        root=args.root,
        source_root=args.source_root,
        config=resolved_comparison_config({"encoder": {"device": args.device}}),
    )
print(json.dumps(result, sort_keys=True, indent=2))

if args.result_json:
    with args.result_json.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
