# SPDX-License-Identifier: Apache-2.0
"""A38's selected partition hunk only, applied privately to the pinned A0 PA."""

import hashlib
import linecache
from functools import cache
from pathlib import Path
import sys
from types import ModuleType

EXPECTED_CORE_SHA256 = "cff95e80610211a87a0849c3ed666056cb6118ec73b10bbb0840a59ee2818743"
NATIVE_KERNEL = "paged_attention_decode_sliding_window_head_1"
KERNEL = "mm3_a38_paged_attention_head1_hybrid"
OLD_PARTITION = (
    "        page_size = gl.cdiv(\n"
    "            context_length, gl.num_programs(2) // CONTEXT_PARTITION_SIZE_PER_BLOCK\n"
    "        )\n"
)
NEW_PARTITION = (
    "        original_page_size = gl.cdiv(\n"
    "            context_length, gl.num_programs(2) // CONTEXT_PARTITION_SIZE_PER_BLOCK\n"
    "        )\n"
    "        optimized_page_size = gl.cdiv(\n"
    "            gl.cdiv(context_length, CONTEXT_PARTITION_SIZE), 4\n"
    "        ) * CONTEXT_PARTITION_SIZE\n"
    "        use_aligned = ((context_length >= 1921) & (context_length <= 2048)\n"
    "            & (gl.num_programs(0) == 96) & (gl.num_programs(2) == 6))\n"
    "        page_size = gl.where(use_aligned, optimized_page_size, original_page_size)\n"
)


def patched_source(source: str) -> str:
    """Change only head-1 partition boundaries and its two kernel-name references."""
    start = source.index(f"def {NATIVE_KERNEL}(")
    end = source.index("\ndef paged_attention_decode_sliding_window(", start)
    section = source[start:end]
    if section.count(OLD_PARTITION) != 1 or source.count(NATIVE_KERNEL) != 2:
        raise ValueError("Native A0 head-1 partition/dispatch source changed")
    result = source[:start] + section.replace(OLD_PARTITION, NEW_PARTITION) + source[end:]
    return result.replace(NATIVE_KERNEL, KERNEL)


@cache
def load(*, hybrid: bool = True):
    """Retain all native functions, globals, imports and license headers in memory."""
    if hybrid is not True:
        raise ValueError("R5 includes only the selected A38 hybrid partition rule")
    import aiter.ops.triton.gluon.pa_decode_gluon as original

    source = Path(original.__file__).read_text()
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_CORE_SHA256:
        raise ValueError("Installed A0 PA differs from the required native source")
    source = patched_source(source)
    filename = "/tmp/mm3_r05_private_pa_hybrid.py"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = ModuleType("_mm3_r05_private_pa_hybrid")
    module.__file__, module.__package__ = filename, original.__package__
    sys.modules[module.__name__] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    module.a38_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return module
