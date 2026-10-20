# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the integration adapter with its real GPU plan and metadata ABI."""

from types import SimpleNamespace as NS

import torch
from cases import make_case, reference

from vllm.models.kimi_k3.amd.specialized_decode.runtime import mla as adapter

adapter.install_on_layer(NS())  # Non-MLA stubs and unsupported backends are unchanged.

fallbacks = []


def original(*args):
    fallbacks.append(args)
    return "fallback", None


def make_layer():
    impl = NS(
        dcp_world_size=8, _decode_num_heads=96, scale=192**-0.5, _forward_mqa=original
    )
    layer = NS(self_attn=NS(mla_attn=NS(mla_attn=NS(impl=impl))))
    adapter.install_on_layer(layer)
    return layer, impl


data = make_case(8195, 96, 8, 8, 3, capacity=32768)
q, kv, idx, seq, qs, ks, ref, ref_lse = data
decode = NS(
    max_qo_len=8,
    attn_out_dtype=torch.bfloat16,
    dcp_route=NS(name="CPRR"),
    cp_world_size=8,
    cp_rank=3,
    g_kv_indptr=torch.tensor([0, 8195], dtype=torch.int32, device="cuda"),
    qo_indptr=torch.tensor([0, 8], dtype=torch.int32, device="cuda"),
    paged_kv_indices=idx,
)
metadata = NS(causal=True, num_actual_tokens=8, num_decode_tokens=8, decode=decode)
attention_layer = NS(_q_scale=qs.reshape(()), _k_scale=ks.reshape(()))
cache = kv.view(-1, 1, 1, 576)
layer, impl = make_layer()
out, lse = impl._forward_mqa(q, cache, metadata, attention_layer)
torch.testing.assert_close(out.float(), ref, atol=0.003, rtol=0.03)
torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=2e-5)
same_forward = impl._forward_mqa
adapter.install_on_layer(layer)
assert impl._forward_mqa is same_forward
out2, _ = impl._forward_mqa(
    (q[..., :512], q[..., 512:]), cache, metadata, attention_layer
)
assert out2.data_ptr() == out.data_ptr()
torch.testing.assert_close(out2.float(), ref, atol=0.003, rtol=0.03)
assert impl._forward_mqa(q[:4], cache, metadata, attention_layer)[0] == "fallback"
assert (
    impl._forward_mqa(q.to(torch.bfloat16), cache, metadata, attention_layer)[0]
    == "fallback"
)
metadata.causal = False
assert impl._forward_mqa(q, cache, metadata, attention_layer)[0] == "fallback"
metadata.causal = True
decode.dcp_route = NS(name="SEGMENTED")
assert impl._forward_mqa(q, cache, metadata, attention_layer)[0] == "fallback"
decode.dcp_route = NS(name="CPRR")
assert len(fallbacks) == 4
_, other_impl = make_layer()
other_out, _ = other_impl._forward_mqa(q, cache, metadata, attention_layer)
p1 = next(iter(impl._kimi_flydsl_mla_plans.values()))
p2 = next(iter(other_impl._kimi_flydsl_mla_plans.values()))
assert p1.partial.data_ptr() == p2.partial.data_ptr()
assert out.data_ptr() != other_out.data_ptr()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    impl._forward_mqa(q, cache, metadata, attention_layer)
for length in (8200, 16385, 65544, 262144, 8195):
    decode.g_kv_indptr[1].fill_(length)
    out.fill_(float("nan"))
    graph.replay()
    ref, ref_lse = reference(q, kv, idx, qs, ks, length, 8, 3)
    torch.testing.assert_close(out.float(), ref, atol=0.003, rtol=0.03)
    torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=2e-5)
try:
    p1(q, kv, idx, seq, 0.75, ks)
except TypeError:
    pass
else:
    raise AssertionError("Scalar scales must be device tensors")
print(
    "PASS adapter: dispatch, tuple Q, fallback, workspace sharing, "
    "graph length changes, ABI guards"
)
