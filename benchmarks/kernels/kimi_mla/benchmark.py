# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: INP001 - standalone three-file benchmark bundle
"""Alternate complete captured MLA calls against the AITER CPRR wrapper."""

import argparse
import functools
import hashlib
import json
import statistics
from pathlib import Path

import aiter
import flydsl
import mla_decode as mla_decode_kimi
import torch
from aiter.mla import mla_decode_fwd


def reference(q, kv, indices, qs, ks, length, world, rank):
    qlen, heads, _ = q.shape
    local = max(0, (length + world - 1 - rank) // world)
    logical = kv.float()[indices[:local].long()]
    score = (q.float().reshape(qlen * heads, 576) * qs) @ (logical * ks).T * (192**-0.5)
    limit = torch.tensor(
        [max(0, (length - qlen + t + world - rank) // world) for t in range(qlen)],
        device="cuda",
    ).repeat_interleave(heads)
    score.masked_fill_(
        torch.arange(local, device="cuda")[None, :] >= limit[:, None], -float("inf")
    )
    lse = score.logsumexp(-1).reshape(qlen, heads)
    prob = score.softmax(-1).nan_to_num()
    ref = (prob @ (logical[:, :512] * ks)).reshape(qlen, heads, 512)
    return ref, lse


def make_case(
    length,
    heads,
    qlen,
    world,
    rank,
    q_fp8=True,
    kv_fp8=True,
    seed=1234,
    layout="random",
    magnitude=0.5,
    capacity=None,
):
    torch.manual_seed(seed + length)
    local = max(0, (length + world - 1 - rank) // world)
    capacity = max(capacity or 0, 64, (local + 63) // 64 * 64)
    q = (torch.randn((qlen, heads, 576), device="cuda") * magnitude).to(
        torch.float8_e4m3fn if q_fp8 else torch.bfloat16
    )
    logical = (torch.randn((capacity, 576), device="cuda") * magnitude).to(
        torch.float8_e4m3fn if kv_fp8 else torch.bfloat16
    )
    # NoPE skips the rotation, but all 576 learned channels still contribute.
    if layout == "random":
        indices = torch.randperm(capacity, device="cuda").int()
    elif layout == "paged":
        indices = (
            (
                torch.randperm(capacity // 64, device="cuda")[:, None] * 64
                + torch.arange(64, device="cuda")
            )
            .flatten()
            .int()
        )
    else:
        indices = torch.arange(capacity, device="cuda", dtype=torch.int32)
    kv = torch.empty_like(logical)
    kv.view(torch.uint8)[indices.long()] = logical.view(torch.uint8)
    seqlen = torch.tensor([length], dtype=torch.int32, device="cuda")
    qs = torch.tensor([0.75 if q_fp8 else 1.0], device="cuda")
    ks = torch.tensor([1.2 if kv_fp8 else 1.0], device="cuda")
    ref, lse = reference(q, kv, indices, qs, ks, length, world, rank)
    return q, kv, indices, seqlen, qs, ks, ref, lse


def make_aiter(data, *, world=8, rank=0, splits=16):
    q, kv, idx, seq, qs, ks, ref, ref_lse = data
    qlen, heads = q.shape[:2]
    padded_heads = 128 if heads > 64 else 16
    padded = torch.empty((qlen, padded_heads, 576), dtype=q.dtype, device=q.device)
    padded.view(torch.uint8)[:, :heads] = q.view(torch.uint8)
    padded.view(torch.uint8)[:, heads:] = q.view(torch.uint8)[:, : padded_heads - heads]
    length = seq.item()
    local = max(0, (length + world - 1 - rank) // world)
    qi = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
    ki = torch.tensor([0, local], dtype=torch.int32, device="cuda")
    gi = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    last = torch.ones(1, dtype=torch.int32, device="cuda")
    info = aiter.get_mla_metadata_info_v1(
        1,
        qlen,
        padded_heads,
        q.dtype,
        kv.dtype,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=splits,
        intra_batch_mode=False,
    )
    meta = [torch.empty(size, dtype=dtype, device="cuda") for size, dtype in info]
    wm, wi, ws, ri, rf, rp = meta
    aiter.get_mla_metadata_v1(
        qi,
        ki,
        last,
        padded_heads,
        1,
        False,
        wm,
        ws,
        wi,
        ri,
        rf,
        rp,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=qlen,
        uni_seqlen_qo=qlen,
        fast_mode=True,
        max_split_per_batch=splits,
        intra_batch_mode=False,
        dtype_q_nope=q.dtype,
        dtype_kv_nope=kv.dtype,
        is_cp_round_robin=world > 1,
    )
    out = torch.zeros((qlen, padded_heads, 512), device="cuda", dtype=torch.bfloat16)

    def run():
        return mla_decode_fwd(
            padded,
            kv.view(-1, 1, 1, 576),
            out,
            qi,
            ki,
            idx[:local],
            last,
            max_seqlen_q=qlen,
            sm_scale=192**-0.5,
            num_kv_splits=splits,
            q_scale=qs,
            kv_scale=ks,
            work_meta_data=wm,
            work_indptr=wi,
            work_info_set=ws,
            reduce_indptr=ri,
            reduce_final_map=rf,
            reduce_partial_map=rp,
            g_kv_indptr=gi,
            cp_world_size=world,
            cp_rank=rank,
            return_lse=True,
        )

    return run, out


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
    environment = dict(
        torch=torch.__version__,
        hip=torch.version.hip,
        flydsl=flydsl.__version__,
        aiter_path=aiter.__file__,
        device=str(torch.cuda.get_device_properties(0)),
        kernel_sha256="",
    )
    environment["kernel_sha256"] = hashlib.sha256(
        Path(mla_decode_kimi.__file__).read_bytes()
    ).hexdigest()
    environment["kernel"] = str(Path(mla_decode_kimi.__file__).resolve())
    results = []
    print("ENV", environment, flush=True)
    for layout in args.layouts:
        for length in args.lengths:
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
                median_us=medians,
                samples_us=samples,
                errors=errors,
                speedup_vs_best_asm=min(
                    (us for name, us in medians.items() if name.startswith("asm"))
                )
                / medians["flydsl"],
            )
            if "asm_256" in medians:
                result["speedup_vs_deployed"] = medians["asm_256"] / medians["flydsl"]
            results.append(result)
            print("RESULT", json.dumps(result), flush=True)
            if args.output is not None:
                Path(args.output).write_text(
                    json.dumps(dict(environment=environment, results=results), indent=2)
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[8192, 8200, 9216, 65536, 65544, 262144],
    )
    p.add_argument(
        "--layouts", nargs="+", default=["paged", "random"], choices=["paged", "random"]
    )
    p.add_argument("--asm-splits", nargs="+", type=int, default=[64, 128, 256, 512])
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--rank", type=int, default=0, choices=range(8))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--capacity", type=int, default=32768)
    p.add_argument("--output", type=Path, help="Optional JSON result file")
    args = p.parse_args()
    if any(length < 8192 for length in args.lengths):
        p.error("--lengths must be >= 8192 for this benchmark's error tolerance")
    if args.samples < 1 or any(cap < 1 for cap in args.asm_splits):
        p.error("--samples and --asm-splits must be positive")
    if args.capacity < 64 or args.capacity % 64:
        p.error("--capacity must be a positive multiple of 64")
    main(args)
