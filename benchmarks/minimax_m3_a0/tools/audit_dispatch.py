"""Audit four-rank optimized execution in the separate real-Eagle3 smoke."""

import argparse
import bisect
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

TARGET_KERNELS = {
    "r01_compile_prepare": ("fusedQKNormIdxrQKNormKernel",),
    "r02_router": ("_rocm_fp32_router_gemm_kernel",),
    "r03_draft_metadata": (),
    "r04_sampling": (),
    "r05_sparse_pa": ("mm3_a38_paged_attention_head1_hybrid",),
    "r06_score_merge": ("mm3_grouped_pa_merge_g2",),
    "r07_gemm1": ("r07_gemm1_slot_a16w4_h6144_i768_ne129_bm16_tn64",),
    "r08_gemm2_tp": ("mm3_r08_atomic_system_tp_v1_m96_cm1",),
}
SAMPLING_KERNELS = (
    "_sharded_greedy_target_stats",
    "_greedy_verify_and_count",
    "_local_max_pack",
    "_merge_max",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=TARGET_KERNELS, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir
    c24, c4 = (
        json.loads((directory / f"smoke_c{n}.json").read_text()) for n in (24, 4)
    )
    if not (
        c24["passed"]
        and c4["passed"]
        and len(c24["requests"]) == 24
        and len(c4["requests"]) == 4
        and all(
            row["passed"] and row["prompt_tokens"] >= 8192 for row in c24["requests"]
        )
        and all(row["passed"] for row in c4["requests"])
    ):
        raise ValueError("Real C24/C4 checks incomplete")
    paths = sorted((directory / "smoke_traces").glob("dp*_rank*.pt.trace.json.gz"))
    ranks = [int(re.search(r"_rank(\d+)\.", path.name).group(1)) for path in paths]
    if sorted(ranks) != [0, 1, 2, 3]:
        raise ValueError("Exactly four TP rank traces required")
    report = {"arm": args.arm, "passed": False, "ranks": []}
    for path, rank in zip(paths, ranks, strict=True):
        with gzip.open(path, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        kernels = sorted(
            (e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"]
        )
        starts = [e["ts"] for e in kernels]
        all_names = Counter(e["name"] for e in kernels)
        scopes = []
        for event in events:
            name = event.get("name", "")
            match = re.search(r"_generation_\d+\((\d+)\)", name)
            if (
                event.get("cat") != "gpu_user_annotation"
                or not name.startswith("execute_context")
                or not match
                or not 89 <= int(match.group(1)) <= 96
                or "dur" not in event
            ):
                continue
            lo = bisect.bisect_left(starts, event["ts"])
            hi = bisect.bisect_left(starts, event["ts"] + event["dur"])
            counts = Counter(e["name"] for e in kernels[lo:hi])
            selected = {
                fragment: sum(n for name, n in counts.items() if fragment in name)
                for fragment in TARGET_KERNELS[args.arm]
            }
            scopes.append(
                {
                    "name": name,
                    "selected_counts": selected,
                    "passed": all(n == 57 for n in selected.values()),
                }
            )
        sampling = (
            {
                fragment: sum(n for name, n in all_names.items() if fragment in name)
                for fragment in SAMPLING_KERNELS
            }
            if args.arm == "r04_sampling"
            else {}
        )
        passed = bool(scopes) and all(row["passed"] for row in scopes)
        if args.arm == "r04_sampling":
            passed &= all(n > 0 for n in sampling.values())
        report["ranks"].append(
            {
                "rank": rank,
                "passed": passed,
                "m96_scopes": scopes,
                "sampling": sampling,
                "trace": str(path),
                "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    server = (directory / "server.log").read_text()
    extra = True
    if args.arm == "r01_compile_prepare":
        extra = (
            "using fused AITER QK norm" in server
            and "does not support it" not in server
        )
    elif args.arm == "r03_draft_metadata":
        extra = (
            "MM3_ABLATION_R03_FUSED_DRAFT_ENABLED steps=3" in server
            and "falling back to rebuilding attention metadata" not in server
            and "does not support it" in server
        )
    elif args.arm == "r06_score_merge":
        records = [
            json.loads(line.split("MM3_A41_LAUNCH ", 1)[1])
            for line in server.splitlines()
            if "MM3_A41_LAUNCH " in line
        ]
        selected = {
            row["rank"]
            for row in records
            if row["grid"] == [1, 24, 18]
            and row["cap"] == 432
            and row["query_len"] == 4
            and row["block_threads"] == 256
        }
        report["score_launch_records"] = records
        extra = selected == {0, 1, 2, 3}
    report["extra_runtime_checks_passed"] = extra
    report["passed"] = extra and all(row["passed"] for row in report["ranks"])
    with (directory / "dispatch_audit.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"arm": args.arm, "passed": report["passed"]}))
    if not report["passed"]:
        raise ValueError("Do not attribute fallback performance to the optimization")


if __name__ == "__main__":
    main()
