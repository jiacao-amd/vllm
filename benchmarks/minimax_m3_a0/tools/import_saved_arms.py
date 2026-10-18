"""Mechanically package pinned, independent A0 overlays; never run GPU work."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "r01_compile_prepare",
    "r02_router",
    "r03_draft_metadata",
    "r04_sampling",
    "r05_sparse_pa",
    "r06_score_merge",
    "r07_gemm1",
    "r08_gemm2_tp",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def make_patch(before: dict, after: dict) -> bytes:
    """Produce a git-apply-compatible patch, including the two native binaries."""
    with tempfile.TemporaryDirectory(prefix="mm3-patch-") as temporary:
        root = Path(temporary)
        for name in after:
            for side, content in (("a", before.get(name)), ("b", after[name])):
                if content is not None:
                    path = root / side / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
        result = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--binary",
                "--src-prefix=",
                "--dst-prefix=",
                "a",
                "b",
            ],
            cwd=root,
            capture_output=True,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode())
        return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    args = parser.parse_args()
    base = args.suite / "sources/a0"
    manifest = json.loads((base / "manifest.json").read_text())
    before = {}
    for name, digest in manifest["sources"].items():
        before[name] = (base / name).read_bytes()
        if sha(before[name]) != digest:
            raise ValueError(f"Changed A0 source: {name}")
    save_json(ROOT / "baseline_manifest.json", manifest)
    catalog = {}
    for arm in ARMS:
        original = args.suite / arm
        destination = ROOT / "arms" / arm
        destination.mkdir(parents=True, exist_ok=False)
        plan = json.loads((original / "plan.json").read_text())
        image_audit = json.loads((original / "image_audit.json").read_text())
        if image_audit["passed"] is not True:
            raise ValueError(f"Source audit missing: {arm}")
        payload = {
            str(path.relative_to(original / "payload")): path.read_bytes()
            for path in sorted((original / "payload").rglob("*"))
            if path.is_file()
        }
        for name, data in payload.items():
            if not name.startswith("vllm/"):
                raise ValueError(f"Unexpected non-vLLM payload: {name}")
            target = destination / "payload" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        patch = make_patch(before, payload)
        (destination / "vllm.patch").write_bytes(patch)
        environment = plan.get("env") or plan.get("environment") or {}
        environment = {
            k: str(v) for k, v in environment.items() if k.startswith("MM3_")
        }
        audit_environment = plan.get("audit_environment", {})
        records = {
            name: {
                "before_sha256": sha(before[name]) if name in before else None,
                "after_sha256": sha(data),
                "bytes": len(data),
            }
            for name, data in payload.items()
        }
        record = {
            "arm": arm,
            "base_image_id": manifest["image_id"],
            "independent_on_a0": True,
            "patch_sha256": sha(patch),
            "files": records,
            "environment": environment,
            "audit_environment": audit_environment,
            "gpu_validation_of_packaged_version": False,
        }
        save_json(destination / "manifest.json", record)
        save_json(destination / "archived_plan.json", plan)
        shutil.copyfile(
            original / "image_audit.json", destination / "archived_image_audit.json"
        )
        if (original / "patch_manifest.json").exists():
            shutil.copyfile(
                original / "patch_manifest.json", destination / "patch_manifest.json"
            )
        catalog[arm] = {
            "manifest_sha256": sha((destination / "manifest.json").read_bytes()),
            "patch_sha256": sha(patch),
            "files": len(payload),
        }
    client = ROOT / "client"
    client.mkdir(exist_ok=False)
    client_hashes = {}
    for source in sorted(args.client.glob("*.py")):
        shutil.copyfile(source, client / source.name)
        client_hashes[source.name] = sha(source.read_bytes())
    shutil.copyfile(args.smoke, ROOT / "tools/long_smoke_requests.py")
    shutil.copyfile(
        args.suite / "a0_runtime_config.json", ROOT / "a0_runtime_config.json"
    )
    shutil.copyfile(
        args.suite / "verify_runtime.py", ROOT / "tools/archived_verify_runtime.py"
    )
    save_json(
        ROOT / "catalog.json",
        {
            "base_image_id": manifest["image_id"],
            "source_vllm_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "arms": catalog,
            "client_sha256": client_hashes,
            "note": (
                "Archived plans are provenance, not the current zero-warmup protocol."
            ),
        },
    )
    print(json.dumps({"arms": len(catalog), "client_files": len(client_hashes)}))


if __name__ == "__main__":
    main()
