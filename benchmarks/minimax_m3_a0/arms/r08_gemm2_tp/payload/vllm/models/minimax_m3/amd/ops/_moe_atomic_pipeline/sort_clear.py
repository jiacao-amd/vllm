"""Redirect only the existing sort kernel's output-clear destination."""
from functools import wraps

def wrap_sort(original, integration):
    """Keep sorting inputs/output metadata intact; scope the clear to A37."""

    @wraps(original)
    def wrapped(*args, **kwargs):
        call = integration._active_call.get()
        state = integration._state
        if call is None or not call.enabled or state is None or (not state.clear_in_sort):
            return original(*args, **kwargs)
        if kwargs or len(args) < 9:
            raise ValueError('Unexpected AITER sort ABI; refusing to skip IPC clear')
        destination = args[6]
        merged = state.collective.merged
        if tuple(destination.shape) != (96, 6144) or destination.dtype != merged.dtype or destination.device != merged.device or (tuple(args[0].shape) != (96, 5)) or (args[7] != 129) or (int(args[8]) != 16):
            raise ValueError('Sort/IPC output contract mismatch')
        if getattr(call, 'a37_sort_destination', None) is not None:
            raise ValueError('Unexpected second sort in one A37 call')
        replacement = (*args[:6], merged, *args[7:])
        result = original(*replacement)
        call.a37_sort_destination = merged.data_ptr()
        return result
    return wrapped

def install(integration):
    """Patch only this experiment process; inactive reference calls delegate."""
    import aiter
    for name in ('moe_sorting_fwd', 'moe_sorting_opus_fwd'):
        setattr(aiter, name, wrap_sort(getattr(aiter, name), integration))
