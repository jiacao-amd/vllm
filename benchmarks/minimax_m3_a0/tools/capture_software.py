"""Capture installed versions without touching a GPU or printing credentials."""

import importlib.metadata
import json
import platform
from pathlib import Path


def main() -> None:
    packages = {}
    for name in (
        "vllm",
        "torch",
        "triton",
        "pytorch-triton-rocm",
        "aiter",
        "amd-aiter",
        "flydsl",
        "transformers",
        "numpy",
        "aiohttp",
    ):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        packages[name] = {"version": distribution.version}
        direct = distribution.read_text("direct_url.json")
        if direct:
            packages[name]["direct_url"] = json.loads(direct)
    import torch

    versions = {}
    for path in Path("/opt/rocm/.info").glob("version*"):
        if path.is_file():
            versions[path.name] = path.read_text().strip()
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "architecture": platform.machine(),
                "packages": packages,
                "rocm_versions": versions,
                "torch_hip": torch.version.hip,
                "torch_cxx11_abi": torch.compiled_with_cxx11_abi(),
                "gpu_initialized": torch.cuda.is_initialized(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
