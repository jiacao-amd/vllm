"""Verify A0 source hashes, then install exactly one independent overlay."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(target: Path, arm: str, source_only: bool = False) -> dict:
    catalog = json.loads((ROOT / "catalog.json").read_text())
    base = json.loads((ROOT / "baseline_manifest.json").read_text())
    if arm != "a0" and arm not in catalog["arms"]:
        raise ValueError(f"Unknown arm: {arm}")
    record = None
    if arm != "a0":
        directory = ROOT / "arms" / arm
        manifest = directory / "manifest.json"
        if sha(manifest) != catalog["arms"][arm]["manifest_sha256"]:
            raise ValueError("Arm manifest changed")
        record = json.loads(manifest.read_text())
        if sha(directory / "vllm.patch") != record["patch_sha256"]:
            raise ValueError("Standalone patch changed")
        for name, info in record["files"].items():
            if sha(directory / "payload" / name) != info["after_sha256"]:
                raise ValueError(f"Payload changed: {name}")
            if info["before_sha256"] is None and (target / name).exists():
                raise ValueError(f"Non-A0 module already present: {name}")
    checked = 0
    for name, digest in base["sources"].items():
        if source_only and not name.startswith("vllm/"):
            continue
        path = target / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"A0 source/dependency mismatch: {name}")
        checked += 1
    return {"arm": arm, "base_files_checked": checked, "record": record}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Parent of vllm/ (checkout or installed site-packages)",
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="For source checkout only; does not validate AITER",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    report = verify(args.target, args.arm, args.source_only)
    record = report.pop("record")
    if not args.check_only and record is not None:
        # All inputs and baseline files were checked before the first write.
        for name, info in record["files"].items():
            destination = args.target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(
                (ROOT / "arms" / args.arm / "payload" / name).read_bytes()
            )
            if sha(destination) != info["after_sha256"]:
                raise ValueError(f"Installed payload mismatch: {name}")
    report.update(
        passed=True,
        source_only=args.source_only,
        applied=not args.check_only and record is not None,
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
