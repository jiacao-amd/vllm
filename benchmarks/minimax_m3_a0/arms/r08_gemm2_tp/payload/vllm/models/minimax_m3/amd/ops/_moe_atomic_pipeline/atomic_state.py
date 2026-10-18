"""Test-only adapter to the audited A35 complete-MoE contract."""
from functools import cache
from pathlib import Path
import torch
import torch.distributed as dist
from aiter.ops.flydsl.moe_kernels import _run_compiled
from . import layout as cfg
from .fused_atomic import compile_atomic
from .workspace import Workspace, load_extension

@cache
def load_view():
    from . import mm3_a37_tensor_view
    return mm3_a37_tensor_view

class AtomicWorkspace(Workspace):
    """Own IPC lifetime; expose the region atomic GEMM2 must initialize."""

    def __init__(self, group=None):
        if (cfg.MERGED, cfg.M, cfg.H) != (16384, 96, 6144):
            raise ValueError('IPC tensor view layout no longer matches A36 workspace')
        if dist.get_rank(group) == 0:
            extension, view = (load_extension(), load_view())
        dist.barrier(group=group)
        if dist.get_rank(group) != 0:
            extension, view = (load_extension(), load_view())
        super().__init__(extension, group)
        self.merged = view.merged_view(self.pointer)
        self.routes = None

    def close(self):
        self.merged = None
        super().close()

class AtomicPipelineState:
    """Sorting and Stage1 unchanged; include the extra IPC clear in timing."""
    column_first = False
    clear_in_sort = False

    def __init__(self, group):
        self.collective = AtomicWorkspace(group)
        self.launcher = compile_atomic(column_first=self.column_first)
        self.zero_work = torch.zeros(2, dtype=torch.int32, device='cuda')
        pointer = self.collective.merged.data_ptr()
        _run_compiled(self.launcher, (pointer, pointer, pointer, pointer, self.zero_work.data_ptr(), pointer, pointer, pointer, self.collective.pointers.data_ptr(), self.collective.rank, 1, torch.cuda.current_stream()))
        self.collective.check(expected_epoch=0)

    def stage2(self, a, w2, tids, eids, counts, output, scale, sorted_weights):
        w2_bytes = w2.view(torch.uint8).contiguous()
        scale_bytes = scale.view(torch.uint8).contiguous().view(-1)
        counts_i32 = counts.to(torch.int32).contiguous()
        if self.clear_in_sort:
            from vllm.models.minimax_m3.amd.ops import moe_route_collective
            call = moe_route_collective._active_call.get()
            if getattr(call, 'a37_sort_destination', None) != self.collective.merged.data_ptr():
                raise RuntimeError('Native sorting did not clear the A37 IPC destination')
        else:
            self.collective.merged.zero_()
        _run_compiled(self.launcher, (a.data_ptr(), w2_bytes.data_ptr(), scale_bytes.data_ptr(), eids.data_ptr(), counts_i32.data_ptr(), tids.data_ptr(), sorted_weights.data_ptr(), output.data_ptr(), self.collective.pointers.data_ptr(), self.collective.rank, eids.numel() * 24, torch.cuda.current_stream()))
        return output
