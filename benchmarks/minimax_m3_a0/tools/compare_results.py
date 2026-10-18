"""Compare one arm only to the destination's shared zero-warmup A0."""

import argparse
import json
from pathlib import Path

from run_arm import KERNEL_ENV

ROOT = Path(__file__).resolve().parents[1]


def load(folder: Path, name: str) -> dict:
    return json.loads((folder / name).read_text())


def compare(baseline: Path, candidate: Path) -> dict:
    contracts = [load(folder, "contract.json") for folder in (baseline, candidate)]
    expected_client = json.loads((ROOT / "catalog.json").read_text())["client_sha256"]
    if contracts[0]["arm"] != "a0" or contracts[1]["arm"] == "a0":
        raise ValueError("Compare an independent arm to fresh A0, not a previous arm")
    for contract in contracts:
        if contract["client_sha256"] != expected_client or len(expected_client) != 4:
            raise ValueError("Unverified/missing benchmark client provenance")
        if not contract["model_hashes"].get("main/tokenizer.json"):
            raise ValueError("Tokenizer payload was not hashed")
        if (
            contract["phase"] != "e2e"
            or contract["http_warmup_requests"] != 0
            or not contract["fresh_process"]
            or not contract["fresh_writable_cache"]
        ):
            raise ValueError("Not the explicit zero-warmup E2E protocol")
        for key, value in KERNEL_ENV.items():
            if contract["environment"].get(key) != value:
                raise ValueError(f"Changed common runtime environment: {key}")
    for key in (
        "base_image_id",
        "host",
        "hip_visible_devices",
        "model_hashes",
        "protocol_sha256",
        "client_sha256",
        "server_arguments",
    ):
        if contracts[0][key] != contracts[1][key]:
            raise ValueError(f"Reference mismatch: {key}")
    results = []
    for folder in (baseline, candidate):
        if not load(folder, "status.json")["completed"]:
            raise ValueError("Run did not finish")
        actual = load(folder, "image_package_audit.json")
        contract = load(folder, "contract.json")
        if (
            actual.get("passed") is not True
            or actual.get("client_sha256") != expected_client
            or actual.get("arm") != contract["arm"]
            or actual.get("protocol_sha256") != contract["protocol_sha256"]
        ):
            raise ValueError("Installed image/client bytes were not verified")
        if not load(folder, "runtime_audit.json")["passed"]:
            raise ValueError("Effective runtime was not verified")
        result = load(folder, "e2e_result.json")
        for key, value in {
            "completed": 240,
            "num_prompts": 240,
            "max_concurrency": 24,
            "total_input_tokens": 1966080,
            "total_output_tokens": 245760,
        }.items():
            if result.get(key) != value:
                raise ValueError(f"Workload/count mismatch: {key}")
        if (
            result.get("failed", 0)
            or result["input_lens"] != [8192] * 240
            or result["output_lens"] != [1024] * 240
        ):
            raise ValueError("Failures or changed request lengths")
        results.append(result)
    for key in (
        "backend",
        "model_id",
        "tokenizer_id",
        "best_of",
        "request_rate",
        "burstiness",
    ):
        if results[0][key] != results[1][key]:
            raise ValueError(f"Client identity mismatch: {key}")
    metrics = {}
    for key in (
        "output_throughput",
        "duration",
        "mean_tpot_ms",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "p99_itl_ms",
        "p99_e2el_ms",
    ):
        old, new = (result[key] for result in results)
        metrics[key] = {
            "a0": old,
            "candidate": new,
            "change_pct": (new / old - 1) * 100,
        }
    return {
        "arm": contracts[1]["arm"],
        "baseline": str(baseline),
        "candidate": str(candidate),
        "protocol_matched": True,
        "metrics": metrics,
        "correctness_and_actual_dispatch_require_separate_review": True,
        "single_run_not_statistical_significance": True,
        "accepted_into_cumulative_baseline": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"arm": report["arm"], "metrics": report["metrics"]}))


if __name__ == "__main__":
    main()
