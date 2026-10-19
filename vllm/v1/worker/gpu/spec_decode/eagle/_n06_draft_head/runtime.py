"""Draft-only production hook; copied into a private package by generate.py."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import metadata
import os
from pathlib import Path
import platform

FLAG = "MM3_DRAFT_HEAD_PREPACK_EXPERIMENT"


def target_head(target):
    """Match the target head understood by the frozen, bounded adapter."""
    return getattr(target, "language_model", target).lm_head


def check_ownership(draft_head, target):
    """Reject shared objects before any quant-method mutation or GPU allocation."""
    other = target_head(target)
    if draft_head is other or draft_head.quant_method is other.quant_method:
        raise ValueError("N06 requires distinct draft and target head/method objects")
    return other


class Installation:
    """Keep preparation inside existing capture warmups, never request execution."""

    def __init__(self, adapter, target, torch, eligible):
        other = check_ownership(adapter.head, target)
        before = (other.weight, other.quant_method, other.quant_method.apply)
        if adapter.hook is not None or adapter.method.apply != adapter.original_apply:
            raise ValueError("Draft quant method already has an owner")
        self.adapter, self.torch, self.eligible = adapter, torch, eligible
        self.preparing = False
        # Do not replace either the ParallelLMHead or quant-method class.
        hook = self.apply
        adapter.hook = hook
        adapter.method.apply = hook
        if (
            other.weight is not before[0]
            or other.quant_method is not before[1]
            or other.quant_method.apply != before[2]
        ):
            raise AssertionError("Target head changed while installing N06")

    @contextmanager
    def prepare_capture(self):
        """The native graph manager supplies warmups on its eventual capture stream."""
        previous = self.preparing
        self.preparing = True
        try:
            yield
        finally:
            self.preparing = previous

    def apply(self, layer, hidden, bias=None):
        """All non-M24 calls stay native; no lazy initialization inside a graph."""
        adapter = self.adapter
        if layer is not adapter.head or not self.eligible(
            hidden, adapter.original, bias
        ):
            return adapter.original_apply(layer, hidden, bias=bias)
        stream = self.torch.cuda.current_stream(adapter.original.device)
        key = adapter._stream_key(stream)
        if key in adapter.streams:
            return adapter.gemm(hidden)
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("N06 capture stream was not primed by its native warmup")
        if self.preparing:
            # prime_stream includes both complete native and candidate heads once.
            # This is an existing uncaptured capture warmup, never a request or replay.
            return adapter.prime_stream(hidden, stream)[1]
        # Eager execution on an unseen stream does not pack, compile or initialize.
        return adapter.original_apply(layer, hidden, bias=bias)


def install_loaded_draft(draft, target, config):
    """Called after load_eagle_model has completed loading and head-sharing decisions."""
    value = os.getenv(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG} must be 0 or 1")
    if value == "0" or config.speculative_config.method != "eagle3":
        return None
    if (
        type(draft).__module__ != "vllm.model_executor.models.llama_eagle3"
        or type(draft).__name__ != "Eagle3LlamaForCausalLM"
    ):
        return None
    if os.getenv("MM3_SPARSE_PA_PARTITION_EXPERIMENT") != "1":
        raise ValueError("Retain the accepted N05 PA flag")
    check_ownership(draft.lm_head, target)

    from .packed_head import PackedDraftHead, eligible, sha, verify_bundle

    manifest = verify_bundle()
    site = Path(__file__).resolve().parents[7]
    for name, expected in manifest["native_sources"].items():
        if sha(site / name) != expected:
            raise ValueError(f"Installed integration source differs: {name}")
    if (
        platform.python_version() != manifest["python"]
        or {name: metadata.version(name) for name in manifest["packages"]}
        != manifest["packages"]
    ):
        raise ValueError("Integration package versions differ from the pinned image")

    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Install N06 only after weight loading, outside capture")
    with torch.inference_mode():
        adapter = PackedDraftHead(draft, target)
    return Installation(adapter, target, torch, eligible)
