"""Compile the packed multi-head reducer with the exact native AITER flags."""

import hashlib
import os
from functools import cache
from pathlib import Path

import torch
from jinja2 import Template

import aiter  # noqa: F401  Initializes the installed aiter_meta/csrc import path.
from csrc.cpp_itfs.torch_utils import torch_to_c_types
from csrc.cpp_itfs.utils import AITER_CORE_DIR, compile_template_op

NATIVE_HEADER_SHA256 = (
    "d26d66a0a65e8b82054d7b5d9aa0084766dd1a36df33792b0e98786d66efdbfe"
)
VARIANT = os.getenv("MM3_A39_MERGE_VARIANT", "v1")


def source_contract():
    header = Path(AITER_CORE_DIR) / "csrc/cpp_itfs/pa/pa_ps.cuh"
    native_sha = hashlib.sha256(header.read_bytes()).hexdigest()
    if native_sha != NATIVE_HEADER_SHA256:
        raise ValueError("Native PA reduction helpers changed")
    if VARIANT not in ("v1", "v2"):
        raise ValueError("Unknown A39 merge implementation")
    name = (
        "grouped_merge_v2.cpp.jinja" if VARIANT == "v2" else "grouped_merge.cpp.jinja"
    )
    source = Path(__file__).with_name(name).read_text()
    return (
        header,
        source,
        {
            "native_header_sha256": native_sha,
            "candidate_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "compiler": "native AITER compile_template_op including its FP denormal policy",
        },
    )


@cache
def build(heads_per_block=4):
    if heads_per_block not in (2, 4, 8):
        raise ValueError("Supported groups are 2, 4 and 8 heads")
    header, source, contract = source_contract()
    return compile_template_op(
        Template(source),
        "mm3_a39_pa_merge",
        includes=[str(header)],
        heads_per_block=heads_per_block,
        revision=contract["candidate_source_sha256"],
        native_revision=contract["native_header_sha256"],
    )


def compatible(output, sums, maxima, partials):
    tensors = (output, sums, maxima, partials)
    return (
        tuple(output.shape) == (96, 1, 1, 16, 128)
        and tuple(sums.shape) == tuple(maxima.shape) == (96, 1, 6, 16)
        and tuple(partials.shape) == (96, 1, 6, 16, 128)
        and output.dtype == partials.dtype == torch.bfloat16
        and sums.dtype == maxima.dtype == torch.float32
        and all(value.is_cuda and value.is_contiguous() for value in tensors)
        and all(value.device == output.device for value in tensors)
    )


def launch(output, sums, maxima, partials, heads_per_block=4):
    if not compatible(output, sums, maxima, partials):
        raise ValueError("A39 requires contiguous M96/H16/D128/P6 BF16 outputs")
    with torch.cuda.device(output.device):
        function = build(heads_per_block)
        status = function(
            *torch_to_c_types(
                output,
                sums,
                maxima,
                partials,
                96,
                torch.cuda.current_stream(output.device),
            )
        )
    if status != 0:
        raise RuntimeError(f"Grouped PA merge launch failed: HIP status {status}")
    return output
