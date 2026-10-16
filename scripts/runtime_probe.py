"""Nonsecret CPU/disk and optional CUDA probe; run in the intended interpreter."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def probe(*, require_gpu=False):
    result = {
        "utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "scope": (
            "compatibility only; notebook-kernel torch is separate from locked AtlasIR runtime"
        ),
        "gpu_status": "not available",
        "gpus": [],
    }
    try:
        command = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        result["nvidia_smi"] = command.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        result["nvidia_smi"] = "unavailable"
    try:
        import torch
    except ImportError:
        if require_gpu:
            raise RuntimeError("GPU probe requires torch in this interpreter") from None
        return result
    result.update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_threads": torch.get_num_threads(),
        }
    )
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            device = f"cuda:{index}"
            tensor = torch.ones((16, 16), device=device)
            product = tensor @ tensor
            torch.cuda.synchronize(index)
            if not torch.equal(product, torch.full_like(product, 16)):
                raise RuntimeError("CUDA matrix multiplication check failed")
            properties = torch.cuda.get_device_properties(index)
            result["gpus"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "matrix_multiply": "passed",
                }
            )
        result["gpu_status"] = "passed"
    elif require_gpu:
        raise RuntimeError("No usable CUDA device; check the notebook accelerator setting")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    result = probe(require_gpu=args.require_gpu)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
