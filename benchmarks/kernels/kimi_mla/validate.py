# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU reference, DCP merge, padding and graph-replay checks for K3 MLA."""

import argparse
import json
from pathlib import Path

import torch
from baseline import make_aiter
from cases import make_case, reference


def prepare(data, rank=0, return_lse=True):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.mla_decode import (
        KimiMLADecodePlan,
    )

    q, kv, idx, seq, qs, ks, ref, ref_lse = data
    plan = KimiMLADecodePlan(
        max_local_tokens=idx.numel(),
        cache_slots=kv.shape[0],
        cp_rank=rank,
        device=q.device,
        return_lse=return_lse,
    )
    out = plan.output
    lse = plan.lse if return_lse else plan._unused_lse
    if not return_lse:
        lse.fill_(float("nan"))

    def run():
        plan(q, kv, idx, seq, qs, ks)

    return run, out, lse


def check(out, lse, ref, ref_lse, magnitude=0.5, short=False):
    # FP8 probability quantization, with a BF16 final store, as in the ASM path.
    if magnitude <= 0.5:
        torch.testing.assert_close(
            out.float(), ref, atol=0.05 if short else 0.003, rtol=0.03
        )
    torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=2e-5)
    assert torch.isfinite(out).all()
    return dict(
        max_abs=(out.float() - ref).abs().max().item(),
        rms=(out.float() - ref).square().mean().sqrt().item(),
    )


def main(args):
    results = []
    for rank in range(8):
        data = make_case(8195, 96, 8, 8, rank, capacity=32768, layout="paged")
        q, kv, idx, seq, qs, ks, _, _ = data
        run, out, lse = prepare(data, rank=rank)
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for length in (
            8,
            9,
            63,
            64,
            65,
            511,
            8191,
            8192,
            8195,
            16384,
            16385,
            65544,
            262144,
        ):
            seq.fill_(length)
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            graph.replay()
            ref, ref_lse = reference(q, kv, idx, qs, ks, length, 8, rank)
            print("CHECK", rank, length, flush=True)
            result = dict(
                rank=rank,
                length=length,
                **check(out, lse, ref, ref_lse, short=length < 8192),
            )
            if length < 8192:
                run_asm, asm_out = make_aiter(
                    (q, kv, idx, seq, qs, ks, ref, ref_lse), rank=rank, splits=64
                )
                run_asm()
                asm_rms = (asm_out[:, :96].float() - ref).square().mean().sqrt().item()
                result["asm_rms"] = asm_rms
                assert result["rms"] <= 1.3 * asm_rms + 0.001, result
            results.append(result)
        print("RANK", rank, "dynamic graph checks passed", flush=True)
    for seed in (23, 53, 97):
        for magnitude in (0.5, 1.5):
            data = make_case(8195, 96, 8, 8, 3, seed=seed, magnitude=magnitude)
            local = (8195 + 7 - 3) // 8
            invalid = data[2][local:].long()
            data[1].view(torch.uint8)[invalid] = (
                127  # FP8 NaN in unused physical slots.
            )
            run, out, lse = prepare(data, rank=3)
            run()
            result = dict(
                seed=seed,
                magnitude=magnitude,
                **check(out, lse, data[-2], data[-1], magnitude),
            )
            run_asm, asm_out = make_aiter(data, rank=3, splits=256)
            run_asm()
            asm_error = asm_out[:, :96].float() - data[-2]
            result["asm_rms"] = asm_error.square().mean().sqrt().item()
            result["asm_max_abs"] = asm_error.abs().max().item()
            assert result["rms"] <= 1.3 * result["asm_rms"] + 0.001, result
            assert result["max_abs"] <= 1.3 * result["asm_max_abs"] + 0.01, result
            results.append(result)
            print("NUMERICS", result, flush=True)
    data = make_case(8195, 96, 8, 8, 0)
    run, out, lse = prepare(data, return_lse=False)
    run()
    torch.testing.assert_close(out.float(), data[-2], atol=0.003, rtol=0.03)
    assert torch.isnan(lse).all(), "O-only specialization must not write LSE"
    print("O_ONLY passed", flush=True)

    # Simulate all eight DCP shards on one device and merge by their LSE.
    global_data = make_case(513, 96, 8, 1, 0)
    q, kv, idx, seq, qs, ks, ref, ref_lse = global_data
    logical = kv.float()[idx.long()]
    outputs, lses = [], []
    for rank in range(8):
        shard = make_case(513, 96, 8, 8, rank, capacity=128)
        sq, skv, sidx, sseq, _, _, _, _ = shard
        n = (513 + 7 - rank) // 8
        skv.view(torch.uint8)[sidx[:n].long()] = (
            logical[rank:513:8].to(q.dtype).view(torch.uint8)
        )
        shard_ref = reference(q, skv, sidx, qs, ks, 513, 8, rank)
        shard = (q, skv, sidx, seq, qs, ks, *shard_ref)
        run, out, lse = prepare(shard, rank=rank)
        run()
        outputs.append(out.float())
        lses.append(lse)
    ls = torch.stack(lses)
    merged = (torch.stack(outputs) * ls.softmax(0)[..., None]).sum(0)
    result = dict(
        test="dcp_merge", **check(merged, ls.logsumexp(0), ref, ref_lse, short=True)
    )
    results.append(result)
    print("DCP_MERGE", result, flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print("PASS", len(results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/kimi-validation.json")
    main(p.parse_args())
