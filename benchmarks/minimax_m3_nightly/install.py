# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install only the byte-verified accepted N18 overlay; no GPU imports."""

import argparse
import hashlib
import json
import platform
import shutil
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = Path("/usr/local/lib/python3.12/dist-packages")


def load(path: Path) -> dict:
    """Read a JSON manifest."""
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """Hash file contents without loading model or GPU modules."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(root: Path, expected: dict) -> None:
    """Fail before installation when any pinned input differs."""
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {name}")
        if sha(root / relative) != digest:
            raise ValueError(f"Source hash differs: {root / relative}")


def package_check(repository: Path | None = None) -> dict:
    """Verify frozen test inputs and, optionally, the accepted source tree."""
    verify_files(HERE, load(HERE / "frozen_manifest.json"))
    accepted = load(HERE / "accepted_manifest.json")
    if len(accepted["payload_sha256"]) != 28:
        raise ValueError("Expected the complete 28-file accepted overlay")
    if repository is not None:
        verify_files(repository, accepted["payload_sha256"])
    return accepted


def expected_sources(arm: str) -> dict:
    """Construct N0 or final N18 identities, without intermediate candidates."""
    expected = dict(load(HERE / "native_manifest.json")["sources"])
    if arm == "n18":
        expected.update(load(HERE / "accepted_manifest.json")["payload_sha256"])
    elif arm != "n0":
        raise ValueError(f"Unsupported arm: {arm}")
    return expected


def audit(arm: str, site: Path = SITE) -> dict:
    """Verify the installed pinned runtime and the relevant source inventory."""
    accepted = package_check()
    native = load(HERE / "native_manifest.json")
    if platform.python_version() != native["python"]:
        raise ValueError("Python version differs from the measured image")
    for name, version in native["packages"].items():
        if metadata.version(name) != version:
            raise ValueError(f"Package version differs: {name}")
    expected = expected_sources(arm)
    verify_files(site, expected)
    if arm == "n0":
        for name in accepted["payload_sha256"].keys() - native["sources"].keys():
            if (site / name).exists():
                raise ValueError(f"N0 contains an added optimization file: {name}")
    return {
        "passed": True,
        "arm": arm,
        "source_files_checked": len(expected),
        "packages": native["packages"],
        "gpu_executed": False,
        "aiter_sources_modified": False,
    }


def main() -> None:
    """Check a checkout, audit an image, or install onto the pinned official base."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--site", type=Path, default=SITE)
    parser.add_argument("--arm", choices=("n0", "n18"), default="n18")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    accepted = package_check(args.repository)
    if args.package_only:
        if args.apply:
            parser.error("--package-only cannot install")
        print(json.dumps({"passed": True, "accepted_files": 28, "gpu_executed": False}))
        return
    if args.apply:
        if args.repository is None or args.arm != "n18":
            parser.error("--apply requires --repository and --arm n18")
        audit("n0", args.site)
        for name in accepted["payload_sha256"]:
            target = args.site / name
            if target.is_symlink() or any(p.is_symlink() for p in target.parents):
                raise ValueError(f"Refuse a symlink installation target: {target}")
        for name in accepted["payload_sha256"]:
            target = args.site / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.repository / name, target)
    print(json.dumps(audit(args.arm, args.site)))


if __name__ == "__main__":
    main()
