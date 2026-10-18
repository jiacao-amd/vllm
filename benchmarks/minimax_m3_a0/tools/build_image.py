"""Build a source-verified arm from the exact local A0 image; no GPU devices."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--base-image", default="local/mm3-vllm-2a02f6e:latest")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    catalog = json.loads((ROOT / "catalog.json").read_text())
    if args.arm not in {"a0", *catalog["arms"]}:
        raise ValueError("Unknown arm")
    info = json.loads(
        subprocess.check_output(
            ["docker", "image", "inspect", args.base_image], text=True
        )
    )[0]
    if info["Id"] != catalog["base_image_id"]:
        raise ValueError("Not the saved A0 image; do not substitute a newer vLLM image")
    subprocess.run(
        [
            "docker",
            "build",
            "--build-arg",
            f"BASE_IMAGE={args.base_image}",
            "--build-arg",
            f"ARM={args.arm}",
            "-t",
            args.tag,
            str(ROOT),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
