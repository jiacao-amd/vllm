"""Keep original A0 attention compute and substitute only its merge callback."""

from functools import cache
from types import FunctionType

import hashlib
from pathlib import Path

EXPECTED_CORE_SHA256 = "cff95e80610211a87a0849c3ed666056cb6118ec73b10bbb0840a59ee2818743"

if __package__:
    from .grouped_merge import compatible, launch
else:
    from grouped_merge import compatible, launch


@cache
def load(heads_per_block=4):
    import aiter.ops.triton.gluon.pa_decode_gluon as parent

    if hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest() != EXPECTED_CORE_SHA256:
        raise ValueError("Installed A0 PA core differs from the required native source")
    original = parent._paged_attention_decode_v2_reduce_kernel_wrapper
    counters = {"selected": 0, "fallback": 0}

    def reduce(*args, **kwargs):
        if (
            len(args) >= 7
            and args[6] is None
            and kwargs.get("PS") is True
            and kwargs.get("query_seq_len") == 1
            and kwargs.get("query_group_size") == 16
            and kwargs.get("head_size") == 128
            and kwargs.get("context_partition_num") == 6
            and compatible(args[1], args[2], args[3], args[4])
        ):
            counters["selected"] += 1
            launch(args[1], args[2], args[3], args[4], heads_per_block)
            return
        counters["fallback"] += 1
        return original(*args, **kwargs)

    function = parent.pa_decode_gluon
    result = FunctionType(
        function.__code__,
        {
            **function.__globals__,
            "_paged_attention_decode_v2_reduce_kernel_wrapper": reduce,
        },
        name="mm3_a39_pa_decode",
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    result.a39_dispatch = counters
    return result
