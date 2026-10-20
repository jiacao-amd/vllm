# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: INP001 - standalone three-file benchmark bundle
"""Compare K3 MLA decode and prefill against AITER and PR194309 forward."""

import argparse
import functools
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
import types
import urllib.request
from pathlib import Path

import aiter
import flydsl
import mla as kimi_mla
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
        Path(kimi_mla.__file__).read_bytes()
    ).hexdigest()
    environment["kernel"] = str(Path(kimi_mla.__file__).resolve())
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
            plan = kimi_mla.KimiMLADecodePlan(
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


_FLEX_COMMIT = "5bc1e0fd3e4e89f762a5b6cc805814ff8b64f156"
_FLEX_HASHES = {
    "flex_attn_fwd_gfx950.py": (
        "090357ecd2f9bd48388deef452b420ad1cfa565d7cf05cebcbc441b4f95b086b"
    ),
    "flex_attn_utils.py": (
        "4e76f965e19ba28a7586c6ec1d17a2b0704f55c09cc0334a1fa210e8ae096700"
    ),
}


def load_original_flex(source):
    """Load the unchanged PR194309 forward; download pinned files if needed."""
    download = source is None
    source = (
        Path(source)
        if source is not None
        else Path.home() / ".cache" / "kimi_mla" / _FLEX_COMMIT
    )
    if download:
        source.mkdir(parents=True, exist_ok=True)
    for name, expected in _FLEX_HASHES.items():
        path = source / name
        if not path.exists() and download:
            url = (
                f"https://raw.githubusercontent.com/pytorch/pytorch/{_FLEX_COMMIT}/"
                f"torch/_inductor/kernel/vendored_templates/flydsl/kernels/{name}"
            )
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Downloaded Flex baseline hash mismatch: {name}")
            temporary = path.with_suffix(f".tmp-{os.getpid()}")
            temporary.write_bytes(data)
            temporary.replace(path)
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Expected unchanged PR194309 baseline file: {path}")
    package = types.ModuleType("_kimi_original_flex")
    package.__path__ = [str(source.resolve())]
    sys.modules[package.__name__] = package
    print("FLEX_SOURCE", str(source.resolve()), flush=True)
    return importlib.import_module("_kimi_original_flex.flex_attn_fwd_gfx950")


def make_prefill_flex(q, k, v, module=None):
    s, h, d = q.shape
    sk = k.shape[0]
    qb, kb, vb = [t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)]
    out = torch.empty_like(v[:s])
    ob = out.transpose(0, 1).unsqueeze(0)
    lse = torch.empty((1, h, s), device=q.device, dtype=torch.float32)
    mx = torch.empty_like(lse)
    blocks, kblocks = (s // 128, sk // 128)
    prefix = sk - s
    partial_count = torch.ones(blocks, device=q.device, dtype=torch.int32)
    partial_indices = (
        torch.arange(blocks, device=q.device, dtype=torch.int32) + prefix // 128
    ).reshape(blocks, 1)
    full_count = partial_indices[:, 0].contiguous()
    full_indices = torch.arange(kblocks, device=q.device, dtype=torch.int32).repeat(
        blocks, 1
    )
    if prefix:
        mask, slot = ((("const_i32", prefix), ("add", 2, 4), ("ge", 5, 3)), 6)
    else:
        mask, slot = ((("ge", 2, 3),), 4)
    fn = module.build_flex_attn_fwd_module(
        batch_size=1,
        num_q_heads=h,
        num_kv_heads=h,
        seq_q=s,
        seq_kv=sk,
        qk_head_dim=d,
        v_head_dim=128,
        block_mask_batch=1,
        block_mask_heads=1,
        num_q_blocks=blocks,
        max_partial_blocks=1,
        max_full_blocks=kblocks,
        sparse_q_block_size=128,
        sparse_kv_block_size=128,
        scale=d ** (-0.5),
        mask_program=mask,
        mask_program_output=slot,
        q_stride=qb.stride(),
        k_stride=kb.stride(),
        v_stride=vb.stride(),
        o_stride=ob.stride(),
        write_max_scores=False,
    )

    def run():
        fn(
            qb,
            kb,
            vb,
            lse,
            mx,
            partial_count,
            partial_indices,
            full_count,
            full_indices,
            ob,
            stream=torch.cuda.current_stream(),
        )

    return (run, out, lse.squeeze(0).T)


def make_prefill_asm(q, k, v):
    s, h, d = q.shape
    sk = k.shape[0]
    nh = (h + 15) // 16 * 16

    def pad_fp8(t, actual=False):
        if h != nh:
            if actual:
                t = t.repeat(1, (nh + h - 1) // h, 1)[:, :nh].contiguous()
            else:
                t = torch.cat((t, t[:, : nh - h]), dim=1)
        return t.to(torch.float8_e4m3fn)

    q8, k8, v8 = [pad_fp8(t) for t in (q, k, v)]
    cpu_qi = torch.tensor([0, s], dtype=torch.int32)
    cpu_ki = torch.tensor([0, sk], dtype=torch.int32)
    cpu_seq = torch.tensor([sk], dtype=torch.int32)
    qi, ki = (cpu_qi.cuda(), cpu_ki.cuda())
    idx = torch.arange(sk, dtype=torch.int32, device=q.device)
    info = aiter.get_ps_metadata_info_v1(
        batch_size=1, num_head_k=nh, max_qlen=s, qlen_granularity=256
    )
    meta = [torch.empty(shape, dtype=dtype, device=q.device) for shape, dtype in info]
    wm, wi, ws, ri, rf, rp = meta
    aiter.get_ps_metadata_v1(
        cpu_qi,
        cpu_ki,
        cpu_seq,
        1,
        nh,
        wm,
        wi,
        ws,
        ri,
        rf,
        rp,
        qhead_granularity=1,
        qlen_granularity=256,
        kvlen_granularity=128,
        block_size=1,
        is_causal=True,
        need_lse=True,
    )
    partial = torch.empty(
        (rp.numel() * 256, nh, 128), device=q.device, dtype=torch.float32
    )
    stats = torch.empty((rp.numel() * 256, nh), device=q.device, dtype=torch.float32)
    out = torch.empty((s, nh, 128), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((s, nh), device=q.device, dtype=torch.float32)
    one = torch.ones(1, device=q.device, dtype=torch.float32)
    cropped = torch.empty((s, h, 128), device=q.device, dtype=torch.bfloat16)

    def kernel(qa=q8, ka=k8, va=v8, scale=one):
        aiter.mla_prefill_ps_asm_fwd(
            qa,
            ka,
            va,
            qi,
            ki,
            idx,
            wi,
            ws,
            s,
            d ** (-0.5),
            True,
            partial,
            stats,
            out,
            scale,
            scale,
            scale,
        )
        aiter.mla_reduce_v1(partial, stats, ri, rf, rp, 256, 0, out, lse)

    def full_call():
        kernel(
            *[pad_fp8(t, actual=True) for t in (q, k, v)],
            torch.ones((), dtype=torch.float32, device=q.device),
        )
        cropped.copy_(out[:, :h])

    def cat_call():
        kernel(*[pad_fp8(t) for t in (q, k, v)])
        cropped.copy_(out[:, :h])

    return (
        (kernel, out[:, :h], lse[:, :h]),
        (full_call, cropped, lse[:, :h]),
        (cat_call, cropped, lse[:, :h]),
    )


def prefill_reference(q, k, v, rows, quantized=False, flex_prescale=False):
    if quantized:
        q, k, v = [t.to(torch.float8_e4m3fn).float() for t in (q, k, v)]
    else:
        q, k, v = [t.float() for t in (q, k, v)]
    if flex_prescale:
        scale_log2 = math.log2(math.e) * q.shape[-1] ** (-0.5)
        q = (q * scale_log2).to(torch.bfloat16).float() / scale_log2
    qs = q[rows].transpose(0, 1)
    scores = qs @ k.transpose(0, 1).transpose(1, 2) * q.shape[-1] ** (-0.5)
    limit = rows + k.shape[0] - q.shape[0]
    scores.masked_fill_(
        torch.arange(k.shape[0], device=q.device)[None, None, :] > limit[None, :, None],
        -torch.inf,
    )
    return (
        (scores.softmax(-1) @ v.transpose(0, 1)).transpose(0, 1),
        scores.logsumexp(-1).T,
    )


def capture_prefill(fn, out, lse, repeats):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    out.fill_(torch.nan)
    lse.fill_(torch.nan)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all(), "empty or incomplete output graph"
    assert torch.isfinite(lse).all(), "empty or incomplete LSE graph"
    return graph


def measure_prefill(graph, repeats, replays):
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


def main_prefill(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    flex = load_original_flex(args.flex_source) if "flex" in args.impls else None
    environment = dict(
        torch=torch.__version__,
        hip=torch.version.hip,
        flydsl=flydsl.__version__,
        device=str(torch.cuda.get_device_properties(0)),
        original_sha256=_FLEX_HASHES["flex_attn_fwd_gfx950.py"],
        kernel_sha256=hashlib.sha256(Path(kimi_mla.__file__).read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    print("ENV", json.dumps(environment), flush=True)
    results = []
    for s in args.lengths:
        sk = s
        torch.manual_seed(args.seed + s)
        q = (
            torch.randn((s, 12, 192), device="cuda", dtype=torch.bfloat16)
            * args.magnitude
        )
        k = (
            torch.randn((sk, 12, 192), device="cuda", dtype=torch.bfloat16)
            * args.magnitude
        )
        v = torch.randn((sk, 12, 128), device="cuda", dtype=torch.bfloat16)
        rows = (
            torch.arange(s, device="cuda")
            if args.full_reference
            else torch.tensor(
                sorted(
                    {
                        r
                        for r in [
                            0,
                            1,
                            31,
                            32,
                            63,
                            64,
                            127,
                            128,
                            255,
                            256,
                            s // 4,
                            s // 2,
                            s - 2,
                            s - 1,
                        ]
                        if 0 <= r < s
                    }
                ),
                device="cuda",
            )
        )

        def reference(quantized, flex_prescale=False, *, q=q, k=k, v=v, rows=rows):
            chunks = [
                prefill_reference(q, k, v, r, quantized, flex_prescale)
                for r in rows.split(128)
            ]
            return tuple(torch.cat([part[i] for part in chunks]) for i in range(2))

        ref, rl = reference(False)
        ref8, rl8 = reference(True)
        reff, rlf = reference(False, True)
        funcs = {}
        if "flex" in args.impls:
            funcs["flex"] = make_prefill_flex(q, k, v, flex)
        if "asm" in args.impls:
            funcs["asm"], funcs["asm_full"], funcs["asm_full_cat"] = make_prefill_asm(
                q, k, v
            )
        if "flydsl" in args.impls:
            plan = kimi_mla.KimiMLAPrefillPlan(seq_len=s, device=q.device)
            funcs["flydsl"] = (functools.partial(plan, q, k, v), plan.output, plan.lse)
            q8, k8, v8 = [t.to(torch.float8_e4m3fn) for t in (q, k, v)]
            fp8_plan = kimi_mla.KimiMLAPrefillPlan(
                seq_len=s, device=q.device, input_dtype=torch.float8_e4m3fn
            )
            funcs["flydsl_fp8"] = (
                functools.partial(fp8_plan, q8, k8, v8),
                fp8_plan.output,
                fp8_plan.lse,
            )
        errors, graphs = ({}, {})
        repeats = max(1, min(20, 131072 // max(s, sk)))
        replays = 5
        for name, (fn, out, lse) in funcs.items():
            print("WARM", s, sk, name, flush=True)
            fn()
            torch.accelerator.synchronize()
            ro, rls = (reff, rlf) if name == "flex" else (ref8, rl8)
            error = (out[rows].float() - ro).abs()
            dense_error = out[rows].float() - ref
            row_relative_l2 = error.norm(dim=-1) / ro.norm(dim=-1).clamp_min(1e-06)
            errors[name] = dict(
                max_abs=error.max().item(),
                rms=error.square().mean().sqrt().item(),
                input_bf16_rms=dense_error.square().mean().sqrt().item(),
                lse_max=(lse[rows] - rls).abs().max().item(),
                input_bf16_lse_max=(lse[rows] - rl).abs().max().item(),
                row_relative_l2_max=row_relative_l2.max().item(),
                normalized_l2=(error.norm() / ro.norm()).item(),
                input_bf16_normalized_l2=(dense_error.norm() / ref.norm()).item(),
            )
            print("ERROR", name, errors[name], flush=True)
            if name == "flex":
                torch.testing.assert_close(out[rows].float(), ro, atol=0.025, rtol=0.03)
            else:
                assert errors[name]["normalized_l2"] < 0.04, errors[name]
                assert errors[name]["row_relative_l2_max"] < 0.08, errors[name]
            torch.testing.assert_close(lse[rows], rls, atol=0.001, rtol=0.0001)
            graphs[name] = capture_prefill(fn, out, lse, repeats)
        samples = {name: [] for name in funcs}
        names = list(funcs)
        for iteration in range(args.samples):
            order = names[iteration % len(names) :] + names[: iteration % len(names)]
            if iteration % 2:
                order.reverse()
            for name in order:
                samples[name].append(measure_prefill(graphs[name], repeats, replays))
        record = dict(
            q=s,
            kv=sk,
            seed=args.seed,
            magnitude=args.magnitude,
            errors=errors,
            median_us={
                name: statistics.median(values) for name, values in samples.items()
            },
            samples_us=samples,
            reference_rows=rows.numel(),
        )
        print("RESULT", json.dumps(record), flush=True)
        results.append(record)
        if args.output is not None:
            Path(args.output).write_text(
                json.dumps(dict(environment=environment, results=results), indent=2)
            )
        del funcs, graphs, q, k, v
    print("PASS", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    p.add_argument(
        "--impls",
        nargs="+",
        choices=["flex", "asm", "flydsl"],
        default=["flex", "asm", "flydsl"],
    )
    p.add_argument(
        "--flex-source",
        type=Path,
        help="Pinned PR194309 source directory; otherwise download to a local cache",
    )
    p.add_argument(
        "--full-reference",
        action="store_true",
        help="Prefill: compare every row (default samples causal and tile boundaries)",
    )
    p.add_argument(
        "--magnitude",
        type=float,
        default=0.5,
        help="Prefill: Q/K random input standard deviation",
    )
    p.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=None,
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
    if args.samples < 1:
        p.error("--samples must be positive")
    if args.mode == "prefill":
        args.lengths = args.lengths or [512, 2048, 8192, 16384, 65536]
        if any(s < 128 or s > 262144 or s % 128 for s in args.lengths):
            p.error("Prefill lengths must be multiples of 128 in [128, 262144]")
        if not math.isfinite(args.magnitude) or args.magnitude <= 0:
            p.error("--magnitude must be finite and positive")
        main_prefill(args)
        sys.exit(0)
    args.lengths = args.lengths or [8192, 8200, 9216, 65536, 65544, 262144]
    if any(length < 8192 for length in args.lengths):
        p.error("--lengths must be >= 8192 for this benchmark's error tolerance")
    if args.samples < 1 or any(cap < 1 for cap in args.asm_splits):
        p.error("--samples and --asm-splits must be positive")
    if args.capacity < 64 or args.capacity % 64:
        p.error("--capacity must be a positive multiple of 64")
    main(args)
