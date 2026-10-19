"""Populate compiler caches in a disposable process; no formal requests."""

import hashlib
import json
from pathlib import Path
import time

from protocol import COMPILE_CASES, ENGINE, PROTOCOL_ID

ROOT = Path("/suite")
OUT = Path("/results")


def main():
    from vllm import LLM

    target = OUT / "compile_ready.json"
    if target.exists():
        raise FileExistsError("Compilation evidence already exists")
    started = time.time()
    llm = LLM(
        **ENGINE,
        worker_extension_cls="compile_extension.CompileCoverageExtension",
        disable_log_stats=True,
    )
    try:
        records = llm.collective_rpc(
            "mm3_compile_coverage", kwargs={"cases": COMPILE_CASES}, timeout=1800
        )
        if sorted(record["rank"] for record in records) != [0, 1, 2, 3]:
            raise ValueError("Not all TP ranks completed compiler coverage")
        if any(len(record["cases"]) != len(COMPILE_CASES) for record in records):
            raise ValueError("Incomplete compiler coverage")
        report = {
            "passed": True, "protocol": PROTOCOL_ID, "engine_kwargs": ENGINE,
            "cases": COMPILE_CASES, "ranks": records,
            "http_completion_requests": 0, "llm_generate_calls": 0,
            "formal_e2e_measurement": False, "started_unix": started,
            "finished_unix": time.time(),
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("prepare_cache.py", "compile_extension.py", "protocol.py")
            },
        }
        with target.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"compiler_coverage_passed": True, "cases": len(COMPILE_CASES)}))
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
