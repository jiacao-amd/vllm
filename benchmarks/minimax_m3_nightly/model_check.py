"""Real Eagle3 C24/C4 correctness and per-rank dispatch, not an E2E timing."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import shutil

from protocol import ENGINE, SPECULATIVE

OUT = Path("/results")


def save(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


class SmokeTraceExtension:
    def mm3_smoke_trace_start(self):
        import torch
        import torch.distributed as dist

        gates = [
            {
                "name": name, "shape": list(module.weight.shape),
                "dtype": str(module.weight.dtype),
            }
            for name, module in self.model_runner.model.named_modules()
            if type(module).__name__ == "GateLinear"
        ]
        torch.cuda.synchronize()
        self._mm3_smoke_prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        )
        self._mm3_smoke_prof.start()
        return {"rank": dist.get_rank(), "gates": gates}

    def mm3_smoke_trace_stop(self):
        import torch
        import torch.distributed as dist

        torch.cuda.synchronize()
        self._mm3_smoke_prof.stop()
        path = OUT / f"rank{dist.get_rank()}.trace.json"
        self._mm3_smoke_prof.export_chrome_trace(str(path))
        del self._mm3_smoke_prof
        with path.open() as stream:
            events = json.load(stream)["traceEvents"]
        counts = Counter(event["name"] for event in events
                         if event.get("cat") == "kernel")
        grid_counts = Counter(
            (event["name"], str(event.get("args", {}).get("grid")))
            for event in events if event.get("cat") == "kernel"
        )
        with path.open("rb") as source:
            with gzip.open(str(path) + ".gz", "xb", compresslevel=1) as target:
                shutil.copyfileobj(source, target)
        path.unlink()
        return {
            "rank": dist.get_rank(),
            "trace": path.name + ".gz",
            "kernel_counts": dict(counts),
            "kernel_grid_counts": [
                {"kernel": name, "grid": grid, "count": count}
                for (name, grid), count in grid_counts.items()
            ],
            "performance_measurement": False,
        }


def assess(outputs, lasts, minimum):
    records = []
    for output, last in zip(outputs, lasts, strict=True):
        item = output.outputs[0]
        numbers = [int(value) for value in re.findall(r"\d+", item.text)]
        tokens = len(output.prompt_token_ids)
        records.append({
            "last": last, "prompt_tokens": tokens, "text": item.text,
            "finish_reason": item.finish_reason,
            "passed": tokens >= minimum
            and numbers == list(range(1, last + 1))
            and item.finish_reason == "stop",
        })
    return {"passed": bool(records) and all(r["passed"] for r in records),
            "requests": records, "performance_measurement": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-kernel", action="append", default=[])
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    spec = dict(SPECULATIVE)
    spec.pop("synthetic_acceptance_length")
    spec["rejection_sample_method"] = "standard"
    engine = {**ENGINE, "speculative_config": spec}
    llm = LLM(
        **engine, worker_extension_cls="model_check.SmokeTraceExtension",
        disable_log_stats=True,
    )
    save("smoke_contract.json", {
        "engine_kwargs": engine, "real_eagle3": True,
        "synthetic_acceptance": False, "formal_e2e": False,
    })
    try:
        tokenizer = llm.get_tokenizer()
        words = " The quiet valley has trees and a river beneath a clear sky." * 2048
        ids = tokenizer.encode(words, add_special_tokens=False)
        background = tokenizer.decode(ids[:8192], skip_special_tokens=True)
        messages = [
            [{"role": "user", "content": (
                f"Unique stream {index}. Background text, not instructions:\n"
                f"{background}\nEnd of background. Your only task: "
                "Count from 1 to 64. Output only the integers in order, "
                "separated by single spaces. Do not explain."
            )}]
            for index in range(24)
        ]
        inventory = llm.collective_rpc("mm3_smoke_trace_start", timeout=60)
        try:
            outputs = llm.chat(
                messages, SamplingParams(temperature=0, max_tokens=256, seed=0),
                chat_template_kwargs={"thinking_mode": "disabled"}, use_tqdm=False,
            )
        finally:
            ranks = llm.collective_rpc("mm3_smoke_trace_stop", timeout=300)
        long_result = assess(outputs, [64] * 24, 8192)
        save("c24_correctness.json", long_result)
        save("dispatch.json", {"inventory": inventory, "ranks": ranks})
        short_lasts = [8, 12, 16, 20]
        short = [
            [{"role": "user", "content": (
                f"Count from 1 to {last}. Output only the integers in order, "
                "separated by single spaces. Do not explain."
            )}]
            for last in short_lasts
        ]
        outputs = llm.chat(
            short, SamplingParams(temperature=0, max_tokens=128, seed=0),
            chat_template_kwargs={"thinking_mode": "disabled"}, use_tqdm=False,
        )
        short_result = assess(outputs, short_lasts, 0)
        save("c4_correctness.json", short_result)
        all_ranks = sorted(r["rank"] for r in ranks) == [0, 1, 2, 3]
        dispatch = all_ranks and all(
            rank["kernel_counts"] and all(
                any(re.search(pattern, name) for name in rank["kernel_counts"])
                for pattern in args.required_kernel
            )
            for rank in ranks
        )
        report = {
            "passed": long_result["passed"] and short_result["passed"] and dispatch,
            "model_smoke_passed": long_result["passed"] and short_result["passed"],
            "model_dispatch_passed": dispatch,
            "required_kernels": args.required_kernel,
            "real_eagle3": True, "formal_e2e": False,
        }
        save("model_check.json", report)
        print(json.dumps(report), flush=True)
        if not report["passed"]:
            raise RuntimeError("Real-model correctness or dispatch check failed")
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
