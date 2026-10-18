"""CPU-only build and installed-byte verification for A0 and all eight arms."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/opt/mm3/.venv/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    catalog = json.loads((ROOT / "catalog.json").read_text())
    reports = {}
    for arm in ("a0", *catalog["arms"]):
        tag = f"local/mm3-handoff-check-{arm}:20261018"
        with (args.output_dir / f"{arm}.build.log").open("x") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "tools/build_image.py"),
                    "--arm",
                    arm,
                    "--tag",
                    tag,
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", tag], text=True
        ).strip()
        report = json.loads(
            subprocess.check_output(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "-e",
                    "HOME=/tmp/mm3-user",
                    "-e",
                    "USER=mm3-user",
                    "-e",
                    "LOGNAME=mm3-user",
                    "-e",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "--entrypoint",
                    PYTHON,
                    image_id,
                    "-B",
                    "/opt/mm3-ablation/tools/verify_package.py",
                    "--arm",
                    arm,
                ],
                text=True,
            )
        )
        report["image_id"] = image_id
        reports[arm] = report
        print(
            json.dumps(
                {
                    "arm": arm,
                    "passed": report["passed"],
                    "engine_files_checked": report["engine_files_checked"],
                }
            ),
            flush=True,
        )
    with (args.output_dir / "image_validation.json").open("x") as stream:
        json.dump({"gpu_devices_exposed": False, "reports": reports}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
