# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alternate complete captured MLA calls against the AITER CPRR wrapper."""

import argparse
import functools
import hashlib
import json
import statistics
from pathlib import Path

import aiter
import flydsl
import torch
from baseline import make_aiter
from cases import make_case


def capture(fn, output, repeats=40):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    output.fill_(float("nan"))
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.isfinite(output).all(), "graph must write all output elements"
    return graph


def measure(graph, repeats=40, replays=20):
    for _ in range(3):
        graph.replay()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (repeats * replays)


def main(args):
    config = json.loads(Path(args.config).read_text())
    environment = dict(
        torch=torch.__version__,
        hip=torch.version.hip,
        flydsl=flydsl.__version__,
        aiter_path=aiter.__file__,
        device=str(torch.cuda.get_device_properties(0)),
        kernel_sha256="",
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono import (
        mla_decode as mla_decode_kimi,
    )

    environment["kernel_sha256"] = hashlib.sha256(
        Path(mla_decode_kimi.__file__).read_bytes()
    ).hexdigest()
    environment["kernel"] = str(Path(mla_decode_kimi.__file__).relative_to(Path.cwd()))
    results = []
    print("ENV", environment, flush=True)
    for layout in args.layouts:
        for length, cfg in config.items():
            length = int(length)
            data = make_case(
                length,
                96,
                8,
                8,
                args.rank,
                layout=layout,
                seed=args.seed,
                capacity=args.capacity,
            )
            q, kv, idx, seq, qs, ks, ref, ref_lse = data
            out = torch.empty_like(ref, dtype=torch.bfloat16)
            lse = torch.empty_like(ref_lse)
            plan = mla_decode_kimi.KimiMLADecodePlan(
                max_local_tokens=idx.numel(),
                cache_slots=kv.shape[0],
                cp_rank=args.rank,
                device=q.device,
            )

            candidate = functools.partial(
                plan, q, kv, idx, seq, qs, ks, output=out, lse=lse
            )

            candidate()
            torch.testing.assert_close(out.float(), ref, atol=0.003, rtol=0.03)
            torch.testing.assert_close(lse, ref_lse, atol=0.0002, rtol=2e-05)
            graphs = {"flydsl": capture(candidate, out)}
            errors = {
                "flydsl": dict(
                    max_abs=(out.float() - ref).abs().max().item(),
                    rms=(out.float() - ref).square().mean().sqrt().item(),
                )
            }
            keep = []
            for cap in args.asm_splits:
                run, asm_out = make_aiter(data, rank=args.rank, splits=cap)
                _, asm_lse = run()
                torch.testing.assert_close(
                    asm_out[:, :96].float(), ref, atol=0.003, rtol=0.03
                )
                torch.testing.assert_close(
                    asm_lse.view(8, 128)[:, :96], ref_lse, atol=0.0002, rtol=2e-05
                )
                label = f"asm_{cap}"
                graphs[label] = capture(run, asm_out)
                errors[label] = dict(
                    max_abs=(asm_out[:, :96].float() - ref).abs().max().item(),
                    rms=(asm_out[:, :96].float() - ref).square().mean().sqrt().item(),
                )
                keep.append((run, asm_out, asm_lse))
            samples = {name: [] for name in graphs}
            names = list(graphs)
            for sample in range(args.samples):
                order = names[sample % len(names) :] + names[: sample % len(names)]
                if sample % 2:
                    order.reverse()
                for name in order:
                    samples[name].append(measure(graphs[name]))
            medians = {name: statistics.median(us) for name, us in samples.items()}
            result = dict(
                length=length,
                layout=layout,
                rank=args.rank,
                seed=args.seed,
                config=cfg,
                median_us=medians,
                samples_us=samples,
                errors=errors,
                speedup_vs_deployed=medians["asm_256"] / medians["flydsl"],
                speedup_vs_best_asm=min(
                    (us for name, us in medians.items() if name.startswith("asm"))
                )
                / medians["flydsl"],
            )
            results.append(result)
            print("RESULT", json.dumps(result), flush=True)
            Path(args.output).write_text(
                json.dumps(dict(environment=environment, results=results), indent=2)
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--layouts", nargs="+", default=["paged", "random"])
    p.add_argument("--asm-splits", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--capacity", type=int, default=32768)
    p.add_argument("--output", default="/tmp/kimi-comparison.json")
    main(p.parse_args())
