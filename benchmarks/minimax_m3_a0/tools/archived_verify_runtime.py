"""Fail before benchmark requests if the live effective configuration differs from A0."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent


def validate(log: str, cmd: list[str], phase: str, expected: dict) -> dict:
    sizes = expected["compilation_config"]["cudagraph_capture_sizes"]
    declarations = [line for line in log.splitlines()
                    if "non-default args:" in line or "Initializing a V1 LLM engine" in line]
    observed = [ast.literal_eval(m.group(1)) for line in declarations
                for m in re.finditer(r"'cudagraph_capture_sizes': (\[[^\]]+\])", line)]
    effective = [ast.literal_eval(m.group(1)) for line in declarations
                 if "Initializing a V1 LLM engine" in line
                 for m in re.finditer(r"'cudagraph_capture_sizes': (\[[^\]]+\])", line)]
    if not effective or any(row != sizes for row in observed):
        raise ValueError(f"Live CLI/effective graph sizes differ from A0: {observed}")
    parsed = json.loads(cmd[cmd.index("--compilation-config") + 1])
    if parsed != expected["compilation_config"]:
        raise ValueError("Compilation CLI differs from recovered A0 configuration")
    for flag, value in {
        "--tensor-parallel-size": "4", "--block-size": "128", "--kv-cache-dtype": "fp8",
        "--max-num-batched-tokens": "32768", "--max-num-seqs": "48", "--stream-interval": "20",
        "--attention-backend": "ROCM_AITER_UNIFIED_ATTN", "--moe-backend": "aiter",
        "--gpu-memory-utilization": "0.9",
    }.items():
        if flag not in cmd or cmd[cmd.index(flag) + 1] != value:
            raise ValueError(f"Wrong runtime setting: {flag}")
    for flag in ("--enable-prefix-caching", "--enable-chunked-prefill", "--language-model-only"):
        if flag not in cmd:
            raise ValueError(f"Missing A0 setting: {flag}")
    spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
    if spec["method"] != "eagle3" or spec["num_speculative_tokens"] != 3:
        raise ValueError("Speculative configuration changed")
    if phase == "e2e":
        if (spec.get("rejection_sample_method") != "synthetic"
                or spec.get("synthetic_acceptance_length") != 2.78
                or "--profiler-config" in cmd):
            raise ValueError("Wrong synthetic E2E/profiler configuration")
    elif "rejection_sample_method" in spec:
        raise ValueError("Smoke must use real Eagle3 acceptance")
    if not any("dtype=torch.bfloat16" in line and "max_seq_len=1048576" in line for line in declarations):
        raise ValueError("Model dtype/max length differs")
    return {"passed": True, "phase": phase, "capture_sizes": sizes,
            "effective_log_declarations_checked": len(observed), "before_benchmark_requests": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("container")
    parser.add_argument("arm")
    parser.add_argument("phase", choices=("smoke", "e2e"))
    parser.add_argument("--run-label", default="a0_matched")
    args = parser.parse_args()
    expected = json.loads((ROOT / "a0_runtime_config.json").read_text())
    info = json.loads(subprocess.check_output(["docker", "inspect", args.container], text=True))[0]
    if not info["State"]["Running"]:
        raise ValueError("Server stopped before runtime verification")
    logs = subprocess.run(["docker", "logs", args.container], check=True, capture_output=True, text=True)
    log = logs.stdout + logs.stderr
    report = validate(log, info["Config"]["Cmd"], args.phase, expected)
    report.update(image_id=info["Image"],
                  recovered_a0_config_sha256=hashlib.sha256((ROOT / "a0_runtime_config.json").read_bytes()).hexdigest())
    output = ROOT / args.arm / "runs" / args.run_label / f"{args.phase}_runtime_audit.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
