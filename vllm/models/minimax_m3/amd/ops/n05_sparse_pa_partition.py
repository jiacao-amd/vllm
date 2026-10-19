# SPDX-License-Identifier: Apache-2.0
"""Private N05 partition-only copy of the exact official-nightly AITER PA."""

from __future__ import annotations

import ast
from functools import cache
import hashlib
import linecache
from pathlib import Path
import sys
from types import ModuleType

CORE_SHA = "ae24e4ad35b573d5e89c2893c43a7f193446b6544d3af30166e2fb8de598c3fb"
NATIVE_KERNEL = "paged_attention_decode_sliding_window_head_1"
KERNEL = "mm3_n05_paged_attention_head1_hybrid"
MERGE = "_paged_attention_decode_v2_reduce_kernel_wrapper"
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
    "            & (gl.num_programs(0) == 96) & (gl.num_programs(2) == 6)\n"
    "            & (CONTEXT_PARTITION_SIZE == 256)\n"
    "            & (CONTEXT_PARTITION_SIZE_PER_BLOCK == 1))\n"
    "        page_size = gl.where(use_aligned, optimized_page_size, original_page_size)\n"
)


def patched_source(source: str) -> str:
    """Prove that only head-1 partition bounds and its two name references change."""
    if hashlib.sha256(source.encode()).hexdigest() != CORE_SHA:
        raise ValueError("Not the official-nightly PA source")
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == NATIVE_KERNEL
    ]
    if len(nodes) != 1 or source.count(NATIVE_KERNEL) != 2:
        raise ValueError("Ambiguous head-1 kernel or dispatch")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    section = "".join(lines[node.lineno - 1 : node.end_lineno])
    if section.count(OLD_PARTITION) != 1:
        raise ValueError("Native head-1 partition rule changed")
    result = (
        "".join(lines[: node.lineno - 1])
        + section.replace(OLD_PARTITION, NEW_PARTITION, 1)
        + "".join(lines[node.end_lineno :])
    ).replace(NATIVE_KERNEL, KERNEL)
    restored = result.replace(KERNEL, NATIVE_KERNEL).replace(
        NEW_PARTITION, OLD_PARTITION, 1
    )
    if restored != source:
        raise ValueError("Unknown arithmetic, layout, dispatch or merge change")
    compile(result, "<n05-partition-source>", "exec")
    return result


@cache
def load():
    """Load privately, and share the native merge function rather than recompile it."""
    import aiter.ops.triton.gluon.pa_decode_gluon as native

    source = patched_source(Path(native.__file__).read_text())
    name = "_mm3_n05_private_pa_hybrid"
    filename = f"<{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = ModuleType(name)
    module.__file__, module.__package__ = filename, native.__package__
    sys.modules[name] = module
    try:
        exec(compile(source, filename, "exec"), module.__dict__)
        # Preserve the actual C++/Triton dispatch and all of its original globals.
        setattr(module, MERGE, getattr(native, MERGE))
        module.n05_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    except BaseException:
        sys.modules.pop(name, None)
        linecache.cache.pop(filename, None)
        raise
    return module
