"""Private N18 gate: all target weights verified, invocation-local workspace."""

from dataclasses import dataclass
import json
import os
import threading

EXPECTED = frozenset(f"language_model.model.layers.{i}.block_sparse_moe.gate"
                     for i in range(3, 60))
FLAG = "MM3_N18_BF16X3_ROUTER"


@dataclass
class Owner:
    prefix: str
    weight: object
    split: object

    @property
    def key(self):
        return (self.weight.data_ptr(), self.split.data_ptr())


class Registry:
    """A new repeated prefix starts a new load cycle; stale completions cannot enable."""

    def __init__(self):
        self.lock = threading.RLock()
        self.devices = {}

    def begin(self, device, prefix):
        if prefix not in EXPECTED:
            return None
        with self.lock:
            state = self.devices.setdefault(device, {"generation": 0, "seen": set(),
                                                      "owners": {}, "lookup": {},
                                                      "ready": False, "selected": set()})
            if prefix in state["seen"]:
                state.update(generation=state["generation"] + 1, seen=set(),
                             owners={}, lookup={}, ready=False, selected=set())
            state["seen"].add(prefix)
            state["ready"] = False
            return device, prefix, state["generation"]

    def finish(self, ticket, owner):
        if ticket is None:
            return False
        device, prefix, generation = ticket
        with self.lock:
            state = self.devices[device]
            if state["generation"] != generation or prefix not in state["seen"]:
                return False
            if owner is not None:
                if owner.key in state["lookup"]:
                    return False
                state["owners"][prefix] = owner
                state["lookup"][owner.key] = owner
            state["ready"] = state["seen"] == EXPECTED and state["owners"].keys() == EXPECTED
            return state["ready"]

    def find(self, device, weight, split):
        with self.lock:
            state = self.devices.get(device)
            if state is None or not state["ready"]:
                return None
            # Strong owner references prevent pointer reuse while registered.
            key = weight.data_ptr(), split.data_ptr()
            owner = state["lookup"].get(key)
            return owner if owner is not None and owner.key == key else None

    def first_selection(self, device, prefix, capture):
        with self.lock:
            selected = self.devices[device]["selected"]
            key = prefix, capture
            if key in selected:
                return False
            selected.add(key)
            return True


_REGISTRY = Registry()


def enabled():
    return os.environ.get(FLAG) == "1"


def audit(kind, **fields):
    if os.environ.get("MM3_N18_AUDIT") == "1":
        print("MM3_N18_" + kind + " " + json.dumps(fields, sort_keys=True), flush=True)


def begin_prepare(layer):
    """Called by the existing loader before native split/early-return/error paths."""
    if (not enabled() or layer.prefix not in EXPECTED
            or getattr(layer, "weight", None) is None or not layer.weight.is_cuda):
        return None
    return _REGISTRY.begin(str(layer.weight.device), layer.prefix)


def finish_prepare(ticket, layer):
    """Full-weight comparison is setup-only, never inside forward/capture/timing."""
    if ticket is None:
        return
    import torch

    weight, split = layer.weight, layer._bf16x3_weight
    valid = (
        weight.is_cuda and weight.dtype == torch.float32
        and tuple(weight.shape) == (128, 6144) and weight.is_contiguous()
        and split is not None and split.dtype == torch.bfloat16
        and tuple(split.shape) == (3, 128, 6144) and split.is_contiguous()
        and split.device == weight.device
    )
    if valid:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("N18 weight validation must precede graph capture")
        reconstructed = (split[0].float() + split[1].float()) + split[2].float()
        valid = bool(torch.equal(reconstructed.view(torch.int32), weight.view(torch.int32)))
    owner = Owner(layer.prefix, weight, split) if valid else None
    ready = _REGISTRY.finish(ticket, owner)
    audit("WEIGHT_CHECK", prefix=layer.prefix, generation=ticket[2],
          exact=valid, elements=128 * 6144, all_layers_ready=ready)
    if ready:
        audit("READY", device=ticket[0], generation=ticket[2], layers=len(EXPECTED),
              elements=len(EXPECTED) * 128 * 6144)


def maybe_run(x, weight, split):
    """Existing opaque dispatcher calls this only at M96; None means native fallback."""
    if not enabled():
        return None
    import torch

    if (tuple(x.shape) != (96, 6144) or x.dtype != torch.bfloat16
            or not x.is_cuda or not x.is_contiguous()
            or weight.dtype != torch.float32 or tuple(weight.shape) != (128, 6144)
            or not weight.is_contiguous() or x.device != weight.device
            or split.dtype != torch.bfloat16 or tuple(split.shape) != (3, 128, 6144)
            or not split.is_contiguous() or split.device != x.device):
        return None
    device = str(weight.device)
    owner = _REGISTRY.find(device, weight, split)
    if owner is None:
        return None
    from ._n18_kernel import Router

    # Never retain Router.output in a stream/owner cache. These two allocations
    # belong to this invocation/current stream; graph capture uses its private
    # allocator pool. PyTorch owns reclamation and replay lifetimes.
    op = Router(split)
    output = op(x)
    if os.environ.get("MM3_N18_AUDIT") == "1":
        capture = torch.cuda.is_current_stream_capturing()
        if _REGISTRY.first_selection(device, owner.prefix, capture):
            audit("SELECTED", prefix=owner.prefix, M=96, N=128, K=6144,
                  grid=[6, 4, 16], warps=4, stages=2, capture=capture,
                  stream=int(torch.cuda.current_stream().cuda_stream),
                  scratch=op.partials.data_ptr(), output=output.data_ptr(),
                  scratch_bytes=2359296, workspace="invocation_local")
    return output
