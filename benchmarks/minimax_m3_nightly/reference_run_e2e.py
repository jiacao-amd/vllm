"""One local fixed E2E using qualified compiler caches and no HTTP warmups."""

import argparse
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import urllib.request

from protocol import BASE_IMAGE, COMPILATION, ENVIRONMENT, MAIN, PROTOCOL_ID, SERVED
from protocol import server_arguments

ROOT = Path(__file__).resolve().parent
CLIENT = Path("/home/jiacao/InferenceX/utils/bench_serving")
LOCK = Path("/home/jiacao/.cache/mm3-fixed-gpus4-7.lock")
PYTHON = "/tmp/mm3-venv/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def cache_index(root):
    """Metadata-only cache inventory; no timing claim or cold-stall subtraction."""
    result = {}
    for path in root.rglob("*"):
        name = str(path.relative_to(root))
        if path.is_symlink():
            result[name] = {"symlink": str(path.readlink())}
        elif path.is_file():
            stat = path.stat()
            result[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def client_arguments(port):
    return [
        PYTHON, "-B", "/client/benchmark_serving.py",
        "--backend", "vllm", "--model", SERVED, "--tokenizer", MAIN,
        "--served-model-name", SERVED, "--port", str(port),
        "--dataset-name", "random", "--random-input-len", "8192",
        "--random-output-len", "1024", "--random-range-ratio", "1.0",
        "--num-prompts", "240", "--max-concurrency", "24",
        "--request-rate", "inf", "--seed", "0", "--num-warmups", "0",
        "--ignore-eos", "--trust-remote-code", "--use-chat-template",
        "--save-result", "--save-detailed", "--result-dir", "/results",
        "--result-filename", "e2e_result.json",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "90,99,99.9",
    ]


def logs(name):
    result = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Cannot read owned server log")
    return result.stdout + result.stderr


def verify_runtime(text):
    declarations = [
        line for line in text.splitlines()
        if "Initializing a V1 LLM engine" in line or "non-default args:" in line
    ]
    sizes = [
        ast.literal_eval(match.group(1))
        for line in declarations
        for match in re.finditer(
            r"""['"]cudagraph_capture_sizes['"]\s*:\s*(\[[^\]]+\])""", line)
    ]
    expected = COMPILATION["cudagraph_capture_sizes"]
    if not sizes or any(sorted(row) != sorted(expected) for row in sizes):
        raise ValueError(f"Effective graph list differs: {sizes}")
    if not any("max_seq_len=1048576" in line and "dtype=torch.bfloat16" in line
               for line in declarations):
        raise ValueError("Model dtype or maximum model length differs")
    return {"passed": True, "capture_sizes": expected,
            "matched_log_declarations": len(sizes), "before_formal_requests": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--prepared-by", required=True)
    parser.add_argument("--image", default=BASE_IMAGE)
    parser.add_argument("--port", type=int, default=6101)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--predecessor")
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args()
    for value in (args.name, args.prepared_by):
        if not value.replace("_", "").isalnum():
            raise ValueError("Unsafe experiment name")
    info = json.loads(subprocess.check_output(
        ["docker", "image", "inspect", args.image], text=True
    ))[0]
    admission = None
    if args.baseline:
        if info["Id"] != BASE_IMAGE.split("@")[1]:
            raise ValueError("N0 must be the exact unmodified official nightly")
        if (ROOT / "original_baseline.json").exists():
            raise ValueError("A local N0 already exists; reuse it")
    else:
        if args.admission is None:
            raise ValueError("Candidates require numerical/micro/model admission")
        admission = json.loads(args.admission.read_text())
        if (admission.get("image_id") != info["Id"]
                or not all(admission.get(key) is True for key in (
                    "correctness_passed", "micro_faster",
                    "model_smoke_passed", "model_dispatch_passed"))):
            raise ValueError("Candidate admission is incomplete or for another image")
        if not args.predecessor:
            raise ValueError("Cumulative candidates require their measured predecessor")
    preparation = ROOT / "runs" / args.prepared_by
    prepared = json.loads((preparation / "compile/compile_ready.json").read_text())
    prep_contract = json.loads((preparation / "compile/contract.json").read_text())
    exited = json.loads((preparation / "compile/exit_code.json").read_text())
    if (prepared.get("passed") is not True
            or prep_contract["image_id"] != info["Id"]
            or prepared["protocol"] != PROTOCOL_ID
            or exited != {"returncode": 0, "container_stopped": True}):
        raise ValueError("Compiler preparation is not qualified for this image")
    for name, digest in prepared["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Compiler protocol source changed: {name}")
    out = ROOT / "runs" / args.name / "e2e"
    cache = ROOT / "runs" / args.name / "cache_e2e"
    if out.exists() or cache.exists():
        raise FileExistsError("Do not overwrite or retry a completed/failed E2E")
    name = f"mm3-nightly-{args.name}-e2e"
    if subprocess.run(["docker", "inspect", name], capture_output=True).returncode == 0:
        raise RuntimeError("Inspect existing owned server before proceeding")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = LOCK.with_name(LOCK.name + ".nightly-owner.json")
        if lease.exists():
            raise RuntimeError(f"Unresolved owned job: {lease}")
        out.mkdir(parents=True)
        # No active compiler process exists now. Preserve symlinks and cached
        # binary bytes, but never propagate transient compiler lock files.
        shutil.copytree(
            preparation / "cache_template", cache, symlinks=True,
            ignore=shutil.ignore_patterns("*.lock", "lock"),
        )
        clients = {path.name: sha(path) for path in sorted(CLIENT.glob("*.py"))}
        predecessor = None
        if not args.baseline:
            if not args.predecessor.replace("_", "").isalnum():
                raise ValueError("Unsafe predecessor name")
            previous = ROOT / "runs" / args.predecessor / "e2e"
            previous_contract = json.loads((previous / "contract.json").read_text())
            previous_status = json.loads((previous / "status.json").read_text())
            if (previous_status.get("completed") is not True
                    or previous_contract["protocol"] != PROTOCOL_ID
                    or previous_contract["protocol_sha256"] != sha(ROOT / "protocol.py")
                    or previous_contract["client_sha256"] != clients
                    or previous_contract["environment"] != ENVIRONMENT
                    or previous_contract["client_arguments"] != client_arguments(args.port)
                    or previous_contract["server_arguments"] != server_arguments(args.port)):
                raise ValueError("Predecessor workload/client/cache protocol differs")
            predecessor = {
                "name": args.predecessor, "image_id": previous_contract["image_id"],
                "result_sha256": sha(previous / "e2e_result.json"),
            }
        command = [
            "docker", "run", "-d", "--name", name, "--user", "13807:13807",
            "--group-add", "110", "--device", "/dev/kfd", "--device", "/dev/dri",
            "--ipc", "host", "--network", "host", "--workdir", "/tmp",
        ]
        for key, value in ENVIRONMENT.items():
            command += ["-e", f"{key}={value}"]
        for host, guest, mode in (
            (ROOT, "/suite", "ro"), (out, "/results", "rw"), (cache, "/cache", "rw"),
            (CLIENT, "/client", "ro"),
            (Path("/data/models/models--amd--MiniMax-M3-MXFP4"), "/models/main", "ro"),
            (Path("/home/jiacao/.cache/mm3-eagle3/models--Inferact--MiniMax-M3-EAGLE3-GQA"), "/models/draft", "ro"),
            (Path("/home/jiacao/.cache/mm3-eagle3/blobs"), "/models/blobs", "ro"),
        ):
            command += ["-v", f"{host}:{guest}:{mode}"]
        server = server_arguments(args.port)
        command += [
            "--entrypoint", "sh", info["Id"], "-c",
            "uv venv --offline --python /usr/bin/python3 --system-site-packages /tmp/mm3-venv"
            ' && exec /tmp/mm3-venv/bin/python -B -m vllm.entrypoints.cli.main "$@"',
            "mm3-server", *server,
        ]
        contract = {
            "name": args.name, "image_id": info["Id"], "image_reference": args.image,
            "protocol": PROTOCOL_ID, "http_warmup_requests": 0,
            "server_arguments": server, "client_arguments": client_arguments(args.port),
            "client_sha256": clients, "environment": ENVIRONMENT,
            "preparation": args.prepared_by,
            "preparation_sha256": sha(preparation / "compile/compile_ready.json"),
            "protocol_sha256": sha(ROOT / "protocol.py"),
            "runner_sha256": sha(Path(__file__)),
            "formal_run_count": 1,
            "is_original_baseline": args.baseline,
            "admission_sha256": None if admission is None else sha(args.admission),
            "predecessor": predecessor,
            "image_environment": info["Config"]["Env"],
        }
        dump(out / "contract.json", contract)
        dump(lease, {"container": name, "output": str(out)})
        success = False
        failures = []
        baseline_record = None
        try:
            subprocess.run(command, check=True)
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                running = subprocess.check_output(
                    ["docker", "inspect", "-f", "{{.State.Running}}", name], text=True
                ).strip()
                if running != "true":
                    raise RuntimeError("Server exited during startup")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/health", timeout=3
                    ):
                        break
                except (OSError, TimeoutError):
                    time.sleep(3)
            else:
                raise TimeoutError("Server startup timed out")
            before_log = logs(name)
            dump(out / "runtime_audit.json", verify_runtime(before_log))
            before_cache = cache_index(cache)
            dump(out / "cache_before.json", before_cache)
            before_time = time.time()
            with (out / "client.log").open("x") as stream:
                subprocess.run(
                    ["docker", "exec", name, *client_arguments(args.port)],
                    stdout=stream, stderr=subprocess.STDOUT, check=True,
                )
            result = json.loads((out / "e2e_result.json").read_text())
            for key, expected in {
                "completed": 240, "num_prompts": 240, "max_concurrency": 24,
                "total_input_tokens": 1966080, "total_output_tokens": 245760,
            }.items():
                if result.get(key) != expected:
                    raise ValueError(f"Workload mismatch: {key}")
            if result.get("failed", 0):
                raise ValueError("Failed requests")
            if (result["input_lens"] != [8192] * 240
                    or result["output_lens"] != [1024] * 240
                    or result["model_id"] != SERVED
                    or result["tokenizer_id"] != MAIN):
                raise ValueError("Changed request lengths or client model identity")
            after_cache = cache_index(cache)
            new_files = sorted(set(after_cache) - set(before_cache))
            updated_files = sorted(
                key for key in set(after_cache) & set(before_cache)
                if after_cache[key] != before_cache[key]
            )
            after_log = logs(name)
            # This is an audit signal, not permission to subtract elapsed time.
            new_log = after_log[len(before_log):] if after_log.startswith(before_log) else after_log
            jit_lines = [line for line in new_log.splitlines()
                         if "JIT compilation during inference" in line]
            dump(out / "runtime_cache_audit.json", {
                "before_client_unix": before_time,
                "new_cache_files": new_files, "updated_cache_files": updated_files,
                "runtime_jit_log_lines": jit_lines,
                "no_elapsed_time_subtracted": True,
                "requires_review": bool(jit_lines or new_files),
            })
            success = True
            if args.baseline:
                baseline_record = {
                    "name": args.name, "image_id": info["Id"],
                    "protocol": PROTOCOL_ID,
                    "result": str((out / "e2e_result.json").relative_to(ROOT)),
                    "result_sha256": sha(out / "e2e_result.json"),
                    "output_throughput": result["output_throughput"],
                    "target_output_throughput_9pct": result["output_throughput"] * 1.09,
                    "runtime_cache_review_required": bool(jit_lines or new_files),
                    "performance_claim": "One local baseline, not an optimization result",
                }
            print(json.dumps({
                "name": args.name, "output_throughput": result["output_throughput"],
                "mean_tpot_ms": result["mean_tpot_ms"],
                "p99_itl_ms": result["p99_itl_ms"],
                "cache_review_required": bool(jit_lines or new_files),
            }), flush=True)
        finally:
            try:
                (out / "server.log").write_text(logs(name))
            except Exception as error:
                failures.append(f"log capture failed: {error}")
            subprocess.run(["docker", "stop", "--timeout", "30", name],
                           capture_output=True)
            state = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", name],
                capture_output=True, text=True,
            )
            if state.returncode == 0 and state.stdout.strip() == "true":
                subprocess.run(["docker", "kill", name], capture_output=True)
                state = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", name],
                    capture_output=True, text=True,
                )
            stopped = (
                state.returncode == 0 and state.stdout.strip() == "false"
            ) or "no such" in state.stderr.lower()
            if stopped:
                subprocess.run(["docker", "rm", name], capture_output=True)
                lease.unlink()
            else:
                failures.append("Container stop not confirmed; ownership lease retained")
            dump(out / "status.json", {
                "completed": success and not failures,
                "container_stopped": stopped, "cleanup_errors": failures,
            })
            if failures:
                raise RuntimeError("; ".join(failures))
        # Publish only after measurement, log preservation and owned-container
        # teardown have all succeeded. Failed attempts remain in their run dir.
        if baseline_record is not None:
            dump(ROOT / "original_baseline.json", baseline_record)


if __name__ == "__main__":
    main()
