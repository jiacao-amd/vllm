# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable container wrapper around the frozen local measurement functions."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import install
import protocol
import reference_run_e2e as reference

OUT = Path("/results")
CACHE = Path("/cache")
KERNELS = (
    "n18_bf16x3_partials",
    "n18_reduce_terms",
    "gemm1_a16w4_port_a16w4_h6144_i768_ne129_bm16_tn64_swiglu_xcd1_w2_kw2",
    "mm3_n06_bpack_",
    "_sharded_greedy_target_stats",
    "_greedy_verify_and_count",
    "_local_max_pack",
    "_merge_max",
    "mm3_n05_paged_attention_head1_hybrid",
)
COMPILER_DIRS = (
    "vllm",
    "triton",
    "inductor",
    "flydsl",
    "home/.aiter/jit",
    "home/.aiter/build",
)


def cache_fingerprint(root: Path) -> dict:
    """Bind cache bytes; transient compiler lock files are deliberately excluded."""
    return {
        str(path.relative_to(root)): install.sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "lock" and not path.name.endswith(".lock")
    }


def require_empty_compiler_cache(root: Path) -> None:
    """Permit uv bootstrap metadata, but never an existing engine/AITER artifact."""
    if any(
        path.is_file() for name in COMPILER_DIRS for path in (root / name).rglob("*")
    ):
        raise ValueError("Compiler preparation contains prior kernel artifacts")


def environment(arm: str) -> dict:
    """Allow only machine identity/device placement to differ from the old host."""
    local = {"HIP_VISIBLE_DEVICES", "USER", "LOGNAME"}
    for key, expected in protocol.ENVIRONMENT.items():
        if key not in local:
            os.environ.setdefault(key, expected)
            if os.environ[key] != expected:
                raise ValueError(f"Protocol environment differs: {key}")
    devices = os.environ.get("HIP_VISIBLE_DEVICES", "").split(",")
    if len(set(devices)) != 4 or not all(devices):
        raise ValueError("Set HIP_VISIBLE_DEVICES to four distinct reserved GPUs")
    if os.geteuid() == 0 or not os.environ.get("MM3_HANDOFF_IMAGE_ID"):
        raise ValueError("Use a non-root UID and record the docker-inspected image ID")
    flags = install.package_check()["required_additional_environment"]
    for key, expected in flags.items():
        if arm == "n18" and os.environ.get(key) != expected:
            raise ValueError(f"N18 optimization flag missing/different: {key}")
        if arm == "n0" and os.environ.get(key, "0") != "0":
            raise ValueError(f"Optimization flag must not be active in N0: {key}")
    return {key: os.environ.get(key) for key in protocol.ENVIRONMENT}


def check_preparation(prepared: Path, contract: dict) -> None:
    """Never seed a formal cache from correctness, profiling, or an earlier E2E."""
    previous = install.load(prepared / "run_contract.json")
    ready = install.load(prepared / "compile_ready.json")
    status = install.load(prepared / "status.json")
    for key in (
        "arm",
        "image_id",
        "environment",
        "frozen_source_sha256",
        "wrapper_sha256",
    ):
        if previous[key] != contract[key]:
            raise ValueError(f"Compiler preparation differs: {key}")
    if (
        previous["stage"] != "prepare"
        or status["completed"] is not True
        or ready["passed"] is not True
        or ready["protocol"] != protocol.PROTOCOL_ID
        or ready["http_completion_requests"] != 0
        or ready["llm_generate_calls"] != 0
    ):
        raise ValueError("Require a completed separate-process dummy preparation")
    for name, digest in ready["source_sha256"].items():
        if install.sha(install.HERE / name) != digest:
            raise ValueError(f"Compiler source changed: {name}")
    if cache_fingerprint(CACHE) != install.load(prepared / "cache_fingerprint.json"):
        raise ValueError("Formal /cache is not a fresh clone of the prepared template")


def validate_result(result: dict) -> None:
    """Require the exact fixed workload, including every request's token lengths."""
    expected = {
        "completed": 240,
        "num_prompts": 240,
        "max_concurrency": 24,
        "total_input_tokens": 1966080,
        "total_output_tokens": 245760,
        "model_id": protocol.SERVED,
        "tokenizer_id": protocol.MAIN,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Formal workload or model identity differs")
    if (
        result.get("failed", 0)
        or any(result.get("errors", []))
        or result["input_lens"] != [8192] * 240
        or result["output_lens"] != [1024] * 240
    ):
        raise ValueError("Failed requests or changed lengths")


def e2e(port: int) -> None:
    """Run exactly one client with no completion-request warmup."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    server = None
    try:
        with (OUT / "server.log").open("x") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "vllm.entrypoints.cli.main",
                    *protocol.server_arguments(port),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Owned server exited during startup")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=3
                    ) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Server startup timed out") from None
                    time.sleep(2)
            reference.dump(
                OUT / "runtime_audit.json",
                reference.verify_runtime((OUT / "server.log").read_text()),
            )
            before = reference.cache_index(CACHE)
            reference.dump(OUT / "cache_before.json", before)
            with (OUT / "client.log").open("x") as client_log:
                subprocess.run(
                    reference.client_arguments(port),
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
            result = install.load(OUT / "e2e_result.json")
            validate_result(result)
            after = reference.cache_index(CACHE)
            text = (OUT / "server.log").read_text()
            reference.dump(
                OUT / "runtime_cache_audit.json",
                {
                    "new_cache_files": sorted(after.keys() - before.keys()),
                    "updated_cache_files": sorted(
                        key
                        for key in before.keys() & after.keys()
                        if before[key] != after[key]
                    ),
                    "runtime_jit_log_lines": [
                        line
                        for line in text.splitlines()
                        if "JIT compilation during inference" in line
                    ],
                    "no_elapsed_time_subtracted": True,
                },
            )
            print(json.dumps({"output_throughput": result["output_throughput"]}))
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)


def main() -> None:
    """Execute one explicitly requested stage in an isolated non-root container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "smoke", "e2e"), required=True)
    parser.add_argument("--arm", choices=("n0", "n18"), required=True)
    parser.add_argument("--port", type=int, default=6101)
    parser.add_argument("--prepared", type=Path, default=Path("/prepared"))
    args = parser.parse_args()
    if not OUT.is_dir() or any(OUT.iterdir()):
        raise ValueError(
            "Use a fresh empty /results; never overwrite or selectively retry"
        )
    actual_environment = environment(args.arm)
    source_audit = install.audit(args.arm)
    verify_clients = {
        name.removeprefix("client/"): digest
        for name, digest in install.load(install.HERE / "frozen_manifest.json").items()
        if name.startswith("client/")
    }
    install.verify_files(Path("/client"), verify_clients)
    contract = {
        "stage": args.stage,
        "arm": args.arm,
        "image_id": os.environ["MM3_HANDOFF_IMAGE_ID"],
        "protocol": protocol.PROTOCOL_ID,
        "environment": actual_environment,
        "frozen_source_sha256": install.load(install.HERE / "frozen_manifest.json"),
        "server_arguments": protocol.server_arguments(args.port),
        "client_arguments": reference.client_arguments(args.port),
        "wrapper_sha256": install.sha(Path(__file__)),
        "http_warmup_requests": 0,
        "source_audit": source_audit,
        "formal_e2e": args.stage == "e2e",
        "performance_claim": (
            "Cross-machine replication, not a new local accepted result"
        ),
    }
    reference.dump(OUT / "run_contract.json", contract)
    success = False
    try:
        if args.stage == "prepare":
            # The outer offline uv bootstrap may create HOME cache metadata.
            # The host must allocate a fresh directory; no compiler subtree may
            # already contain artifacts when the native preparation begins.
            require_empty_compiler_cache(CACHE)
            import prepare_cache

            prepare_cache.main()
            reference.dump(OUT / "cache_fingerprint.json", cache_fingerprint(CACHE))
        elif args.stage == "smoke":
            import model_check

            sys.argv = [sys.argv[0]]
            if args.arm == "n18":
                for kernel in KERNELS:
                    sys.argv.extend(("--required-kernel", kernel))
            model_check.main()
        else:
            check_preparation(args.prepared, contract)
            e2e(args.port)
        success = True
    finally:
        reference.dump(
            OUT / "status.json",
            {
                "completed": success,
                "stage": args.stage,
                "arm": args.arm,
                "http_warmup_requests": 0,
            },
        )


if __name__ == "__main__":
    main()
