"""Read installed client and engine bytes, without importing any GPU library."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--site", type=Path, default=Path("/usr/local/lib/python3.12/dist-packages")
    )
    args = parser.parse_args()
    catalog = json.loads((ROOT / "catalog.json").read_text())
    if args.arm not in {"a0", *catalog["arms"]}:
        raise ValueError("Unknown arm")
    expected = dict(
        json.loads((ROOT / "baseline_manifest.json").read_text())["sources"]
    )
    if args.arm != "a0":
        manifest_path = ROOT / "arms" / args.arm / "manifest.json"
        if sha(manifest_path) != catalog["arms"][args.arm]["manifest_sha256"]:
            raise ValueError("Arm manifest changed")
        manifest = json.loads(manifest_path.read_text())
        expected.update(
            {name: row["after_sha256"] for name, row in manifest["files"].items()}
        )
    for name, digest in expected.items():
        if sha(args.site / name) != digest:
            raise ValueError(f"Installed engine/dependency mismatch: {name}")
    client = {name: sha(ROOT / "client" / name) for name in catalog["client_sha256"]}
    if client != catalog["client_sha256"] or len(client) != 4:
        raise ValueError("Installed benchmark client mismatch")
    print(
        json.dumps(
            {
                "passed": True,
                "arm": args.arm,
                "engine_files_checked": len(expected),
                "client_sha256": client,
                "catalog_sha256": sha(ROOT / "catalog.json"),
                "protocol_sha256": sha(ROOT / "protocol.json"),
                "gpu_library_imported": False,
            }
        )
    )


if __name__ == "__main__":
    main()
