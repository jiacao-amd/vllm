"""Owned TP4 IPC storage for the single-kernel Stage2 experiment."""
from functools import cache
from pathlib import Path
import torch
import torch.distributed as dist
from . import layout as cfg

@cache
def load_extension():
    from . import mm3_a36_workspace
    return mm3_a36_workspace

class Workspace:
    """One serialized stream owner; peers exchange only merged/reduced data."""

    def __init__(self, extension, group=None):
        if dist.get_world_size(group) != cfg.TP:
            raise ValueError('TP4 required')
        self.extension, self.group = (extension, group)
        self.rank = dist.get_rank(group)
        self.pointer, handle = extension.allocate(cfg.ALLOCATION_BYTES)
        handles = [None] * cfg.TP
        dist.all_gather_object(handles, handle, group=group)
        pointers, self.imports = ([], [])
        for rank, peer in enumerate(handles):
            if rank == self.rank:
                pointer = self.pointer
            else:
                pointer = extension.import_peer(peer)
                self.imports.append(pointer)
            pointers.append(pointer)
        self.pointers = torch.tensor(pointers, dtype=torch.int64, device='cuda')
        self.routes = torch.empty((cfg.M, cfg.TOPK, cfg.H), dtype=torch.bfloat16, device='cuda')
        self.closed = False
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def state(self):
        """Read diagnostic metadata outside all captured/timed regions."""
        torch.cuda.synchronize()
        words = self.extension.snapshot(self.pointer, cfg.META_BYTES).view(torch.int64)
        result = {'rank': self.rank, 'error': int(words[cfg.ERROR // 8]), 'counters': [int(words[(cfg.COUNTERS + col * cfg.FLAG_STRIDE) // 8]) for col in range(cfg.COLUMNS)], 'epochs': [int(words[(cfg.EPOCHS + col * cfg.FLAG_STRIDE) // 8]) for col in range(cfg.COLUMNS)]}
        for name, offset in (('service_start', cfg.SERVICE_START), ('local_merge_end', cfg.LOCAL_MERGE_END), ('merge_sync_end', cfg.MERGE_SYNC_END), ('reduce_sync_end', cfg.REDUCE_SYNC_END), ('service_end', cfg.SERVICE_END)):
            result[name] = [int(words[offset // 8 + col]) for col in range(cfg.COLUMNS)]
        return result

    def check(self, expected_epoch=None):
        local = self.state()
        states = [None] * cfg.TP
        dist.all_gather_object(states, local, group=self.group)
        valid = all((row['error'] == 0 and (not any(row['counters'])) and (expected_epoch is None or all((value == expected_epoch for value in row['epochs']))) for row in states))
        if not valid:
            raise AssertionError(states)
        return states

    def close(self):
        if self.closed:
            return
        self.check()
        dist.barrier(group=self.group)
        for pointer in self.imports:
            self.extension.close_peer(pointer)
        dist.barrier(group=self.group)
        self.extension.release(self.pointer)
        self.closed = True
