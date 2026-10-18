"""One fresh, zero-HTTP-warmup E2E or a separate real-Eagle3 smoke."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from archived_verify_runtime import validate

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/opt/mm3/.venv/bin/python"
MAIN = "/models/main/snapshots/229447a2d64149ffb45e454ebeb74b7a9a45ce7d"
DRAFT = "/models/draft/snapshots/96692486b5fd38ebf8fd2a5f6bb53427d30819a8"
KERNEL_ENV = {
    "OMP_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_MOE": "1",
    "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1",
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
    "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "0",
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB": "256",
}


def server_args(port: int, phase: str) -> list[str]:
    config = json.loads((ROOT / "a0_runtime_config.json").read_text())
    spec = {
        "method": "eagle3",
        "model": DRAFT,
        "num_speculative_tokens": 3,
        "attention_backend": "ROCM_AITER_UNIFIED_ATTN",
    }
    if phase == "e2e":
        spec.update(
            rejection_sample_method="synthetic", synthetic_acceptance_length=2.78
        )
    args = [
        "serve",
        MAIN,
        "--served-model-name",
        "amd/MiniMax-M3-MXFP4",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--trust-remote-code",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "128",
        "--gpu-memory-utilization",
        "0.9",
        "--kv-cache-dtype",
        "fp8",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--language-model-only",
        "--max-num-batched-tokens",
        "32768",
        "--max-num-seqs",
        "48",
        "--stream-interval",
        "20",
        "--attention-backend",
        "ROCM_AITER_UNIFIED_ATTN",
        "--moe-backend",
        "aiter",
        "--attention-config",
        '{"indexer_kv_dtype":"fp8"}',
        "--compilation-config",
        json.dumps(config["compilation_config"]),
        "--speculative-config",
        json.dumps(spec),
        "--tool-call-parser",
        "minimax_m3",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "minimax_m3",
        "--default-chat-template-kwargs",
        '{"thinking_mode":"enabled"}',
    ]
    if phase == "smoke":
        args += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": "/results/smoke_traces",
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": False,
                    "torch_profiler_use_gzip": True,
                }
            ),
        ]
    return args


def client_args(port: int) -> list[str]:
    return [
        PYTHON,
        "-B",
        "/opt/mm3-ablation/client/benchmark_serving.py",
        "--backend",
        "vllm",
        "--model",
        "amd/MiniMax-M3-MXFP4",
        "--tokenizer",
        MAIN,
        "--served-model-name",
        "amd/MiniMax-M3-MXFP4",
        "--port",
        str(port),
        "--dataset-name",
        "random",
        "--random-input-len",
        "8192",
        "--random-output-len",
        "1024",
        "--random-range-ratio",
        "1.0",
        "--num-prompts",
        "240",
        "--max-concurrency",
        "24",
        "--request-rate",
        "inf",
        "--seed",
        "0",
        "--num-warmups",
        "0",
        "--ignore-eos",
        "--trust-remote-code",
        "--use-chat-template",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        "/results",
        "--result-filename",
        "e2e_result.json",
    ]


def dump(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def logs(name: str) -> str:
    result = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    return result.stdout + result.stderr


def normalize_gpus(value: str) -> str:
    parts = value.split(",")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        raise ValueError("Exactly four distinct numeric HIP indices required")
    indices = [int(part) for part in parts]
    if len(set(indices)) != 4:
        raise ValueError("Exactly four distinct numeric HIP indices required")
    return ",".join(map(str, indices))


def cleanup_owned(name: str, log_path: Path, lease: Path) -> None:
    """Always attempt teardown; retain the lease when stop cannot be confirmed."""
    failures = []
    try:
        log_path.write_text(logs(name))
    except Exception as error:
        failures.append(f"log capture failed: {error}")

    def state() -> str:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip() in ("true", "false"):
            return result.stdout.strip()
        if "no such" in result.stderr.lower():
            return "absent"
        return "unknown"

    try:
        subprocess.run(
            ["docker", "stop", "--timeout", "20", name],
            capture_output=True,
            text=True,
            check=False,
        )
        observed = state()
        if observed == "true":
            subprocess.run(
                ["docker", "kill", name], capture_output=True, text=True, check=False
            )
            observed = state()
        if observed not in ("false", "absent"):
            raise RuntimeError(f"Cannot confirm {name} is stopped; inspect {lease}")
        if observed == "false":
            removed = subprocess.run(
                ["docker", "rm", name], capture_output=True, text=True, check=False
            )
            if removed.returncode != 0:
                raise RuntimeError(f"Cannot remove stopped container {name}")
        lease.unlink()
    except Exception as error:
        failures.append(str(error))
    if failures:
        raise RuntimeError("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--draft-root", type=Path, required=True)
    parser.add_argument("--draft-blobs", type=Path)
    parser.add_argument(
        "--gpus", required=True, help="Four exclusively allocated HIP indices"
    )
    parser.add_argument("--device-group", type=int, required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        required=True,
        help="Shared lock used by ALL jobs on the allocated GPU group",
    )
    parser.add_argument("--port", type=int, default=6101)
    parser.add_argument("--phase", choices=("smoke", "e2e"), default="e2e")
    args = parser.parse_args()
    if os.getuid() == 0:
        raise ValueError("Run as the destination non-root user")
    args.gpus = normalize_gpus(args.gpus)
    catalog = json.loads((ROOT / "catalog.json").read_text())
    if args.arm not in {"a0", *catalog["arms"]}:
        raise ValueError("Unknown arm")
    info = json.loads(
        subprocess.check_output(["docker", "image", "inspect", args.image], text=True)
    )[0]
    labels = info["Config"].get("Labels") or {}
    if (
        labels.get("mm3.ablation.arm") != args.arm
        or labels.get("mm3.ablation.base") != catalog["base_image_id"]
    ):
        raise ValueError("Build this arm with tools/build_image.py first")
    protocol = json.loads((ROOT / "protocol.json").read_text())
    model_hashes = {}
    for role, root, revision in (
        ("main", args.model_root, protocol["main_revision"]),
        ("draft", args.draft_root, protocol["draft_revision"]),
    ):
        for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            path = root / "snapshots" / revision / filename
            if filename != "config.json" and role == "draft":
                continue
            model_hashes[f"{role}/{filename}"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite/retry an existing run")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock = args.lock.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease = args.lock.with_name(args.lock.name + ".owner.json")
    if lease.exists():
        raise RuntimeError(f"Previous ownership is unresolved; inspect {lease}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "cache").mkdir()
    if args.phase == "smoke":
        (output / "smoke_traces").mkdir()
    name = f"mm3-handoff-{args.arm}-{uuid.uuid4().hex[:10]}"
    environment = dict(KERNEL_ENV)
    environment.update(
        {
            "HOME": "/tmp/mm3-user",
            "USER": "mm3-user",
            "LOGNAME": "mm3-user",
            "HIP_VISIBLE_DEVICES": args.gpus,
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "TRITON_CACHE_DIR": "/cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
            "FLYDSL_RUNTIME_CACHE_DIR": "/cache/flydsl",
        }
    )
    if args.arm != "a0":
        arm = json.loads((ROOT / "arms" / args.arm / "manifest.json").read_text())
        environment.update(arm["environment"])
        if args.phase == "smoke":
            environment.update(arm["audit_environment"])
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--group-add",
        str(args.device_group),
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--ipc",
        "host",
        "--network",
        "host",
        "--workdir",
        "/tmp",
    ]
    for key, value in environment.items():
        command += ["-e", f"{key}={value}"]
    for host, guest, mode in (
        (output, "/results", "rw"),
        (output / "cache", "/cache", "rw"),
        (args.model_root.resolve(), "/models/main", "ro"),
        (args.draft_root.resolve(), "/models/draft", "ro"),
    ):
        command += ["-v", f"{host}:{guest}:{mode}"]
    if args.draft_blobs:
        command += ["-v", f"{args.draft_blobs.resolve()}:/models/blobs:ro"]
    server = server_args(args.port, args.phase)
    command += [info["Id"], *server]
    contract = {
        "arm": args.arm,
        "phase": args.phase,
        "image_id": info["Id"],
        "base_image_id": catalog["base_image_id"],
        "host": platform.node(),
        "hip_visible_devices": args.gpus,
        "model_hashes": model_hashes,
        "protocol_sha256": hashlib.sha256(
            (ROOT / "protocol.json").read_bytes()
        ).hexdigest(),
        "client_sha256": catalog["client_sha256"],
        "http_warmup_requests": 0,
        "server_arguments": server,
        "environment": environment,
        "fresh_process": True,
        "fresh_writable_cache": True,
    }
    dump(output / "contract.json", contract)
    dump(lease, {"container": name, "output": str(output), "host": platform.node()})
    success = False
    try:
        subprocess.run(command, check=True)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            running = subprocess.check_output(
                ["docker", "inspect", "-f", "{{.State.Running}}", name], text=True
            ).strip()
            if running != "true":
                raise RuntimeError("Server exited before readiness")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{args.port}/health",
                    timeout=3,
                ):
                    break
            except (OSError, TimeoutError):
                time.sleep(3)
        else:
            raise TimeoutError("Server startup timed out")
        expected = json.loads((ROOT / "a0_runtime_config.json").read_text())
        dump(
            output / "runtime_audit.json",
            validate(logs(name), server, args.phase, expected),
        )
        actual = json.loads(
            subprocess.check_output(
                [
                    "docker",
                    "exec",
                    name,
                    PYTHON,
                    "-B",
                    "/opt/mm3-ablation/tools/verify_package.py",
                    "--arm",
                    args.arm,
                ],
                text=True,
            )
        )
        dump(output / "image_package_audit.json", actual)
        if (
            actual.get("passed") is not True
            or actual["client_sha256"] != catalog["client_sha256"]
            or actual["protocol_sha256"] != contract["protocol_sha256"]
            or actual["catalog_sha256"]
            != hashlib.sha256((ROOT / "catalog.json").read_bytes()).hexdigest()
        ):
            raise ValueError("Selected image contains a stale/different test package")
        if args.phase == "e2e":
            with (output / "client.log").open("x") as stream:
                subprocess.run(
                    ["docker", "exec", name, *client_args(args.port)],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        else:

            def profile(endpoint: str, timeout: int) -> None:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/{endpoint}", data=b"", method="POST"
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    response.read()

            profile("start_profile", 30)
            for count in (24, 4):
                extra = ["--count-to", "128"] if count == 24 else []
                with (output / f"smoke_c{count}.log").open("x") as stream:
                    subprocess.run(
                        [
                            "docker",
                            "exec",
                            name,
                            PYTHON,
                            "-B",
                            "/opt/mm3-ablation/tools/long_smoke_requests.py",
                            "--port",
                            str(args.port),
                            "--num-requests",
                            str(count),
                            "--output",
                            f"/results/smoke_c{count}.json",
                            *extra,
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            profile("stop_profile", 180)
        success = True
    finally:
        cleanup_ok = False
        try:
            cleanup_owned(name, output / "server.log", lease)
            cleanup_ok = True
        finally:
            try:
                dump(
                    output / "status.json",
                    {
                        "completed": success and cleanup_ok,
                        "phase": args.phase,
                        "cleanup_confirmed": cleanup_ok,
                    },
                )
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()


if __name__ == "__main__":
    main()
