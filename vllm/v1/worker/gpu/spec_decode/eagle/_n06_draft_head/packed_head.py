"""Private M24 draft-head adapter; no GPU imports or work at module import time."""

from __future__ import annotations

from contextlib import contextmanager
from functools import cache
import hashlib
import inspect
import json
import linecache
from pathlib import Path
import sys
from types import ModuleType

from . import layout

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    """Hash source bytes; never inspect tensor values in a captured call."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_version(tensor):
    """Inference tensors have no version counter; their immutability is a contract."""
    try:
        return tensor._version
    except RuntimeError:
        return None


def eligible(hidden, weight, bias=None) -> bool:
    """Only the exact original BF16 M24 head uses the packed representation."""
    return (
        bias is None
        and tuple(hidden.shape) == (layout.M, layout.K)
        and tuple(weight.shape) == (layout.N, layout.K)
        and str(hidden.dtype) == str(weight.dtype) == "torch.bfloat16"
        and hidden.device == weight.device
        and hidden.device.type == "cuda"
        and hidden.is_contiguous()
        and weight.is_contiguous()
        and hidden.data_ptr() % 16 == weight.data_ptr() % 16 == 0
    )


def verify_bundle() -> dict:
    """Verify the additive bundle without needing an external checkout at runtime."""
    if Path(layout.__file__).resolve() != (ROOT / "layout.py").resolve():
        raise ValueError("A different bundle owns the imported layout module")
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"N06 source bundle changed: {name}")
    if sha(ROOT / "generated_gemm.py") != manifest["generated_sha256"]:
        raise ValueError("Generated nightly GEMM changed")
    return manifest


@cache
def load_module():
    """Use the actual new FlyDSL helpers and share the native workspace factory."""
    manifest = verify_bundle()
    from aiter.ops.flydsl.kernels import gemm_a16w16_gfx950 as native

    site = Path(native.__file__).resolve().parents[4]
    for name, expected in manifest["native_sources"].items():
        if sha(site / name) != expected:
            raise ValueError(f"Installed nightly source differs: {name}")
    source = (ROOT / "generated_gemm.py").read_text()
    name, filename = "_mm3_n06_private_head", "/tmp/mm3_n06_private_head.py"
    if name in sys.modules:
        raise RuntimeError("Another module owns the private N06 namespace")
    module = ModuleType(name)
    module.__file__, module.__package__ = filename, native.__package__
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    sys.modules[name] = module
    try:
        # Native fx.struct fields require classes, not this caller's future strings.
        exec(compile(source, filename, "exec", dont_inherit=True), module.__dict__)
        module.get_split_k_buffers = native.get_split_k_buffers
    except BaseException:
        sys.modules.pop(name, None)
        linecache.cache.pop(filename, None)
        raise
    return module


def verify_native_call(bound, weight) -> None:
    """The executed native path, not a historical CSV, must select this exact GEMM."""
    expected = {
        **{
            key: value
            for key, value in layout.CONFIG.items()
            if key != "use_half_tile_interleaved"
        },
        "policy": "ft",
        "bias": None,
    }
    if (
        bound["b"].data_ptr() != weight.data_ptr()
        or not eligible(bound["a"], weight, bound["bias"])
        or bound["out_dtype"] not in (None, weight.dtype)
        or any(bound[name] != value for name, value in expected.items())
    ):
        raise ValueError(
            "Actual native head dispatch is not the pinned BF16 split-K2 configuration"
        )


class PackedDraftHead:
    """One immutable draft weight plus a bitwise-packed copy and per-stream state.

    The hook changes only this head's quant-method application, preserving its
    native class and every logits processor, vocabulary map and sampling caller.
    """

    def __init__(self, draft_model, target_model):
        import torch
        import triton
        from . import gpu_pack

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Pack only after final weight loading and outside capture"
            )
        if (
            type(draft_model).__module__ != "vllm.model_executor.models.llama_eagle3"
            or type(draft_model).__name__ != "Eagle3LlamaForCausalLM"
        ):
            raise ValueError("N06 binds only the actual Eagle3 draft model")
        self.head = draft_model.lm_head
        target_inner = getattr(target_model, "language_model", target_model)
        if self.head is target_inner.lm_head:
            raise ValueError(
                "A head object shared with target cannot be hooked draft-only"
            )
        self.original = self.head.weight
        processor = draft_model.logits_processor
        method = self.head.quant_method
        method_type = f"{type(method).__module__}.{type(method).__name__}"
        if (
            tuple(self.original.shape) != (layout.N, layout.K)
            or self.original.dtype != torch.bfloat16
            or not self.original.is_cuda
            or not self.original.is_contiguous()
            or self.original.data_ptr() % 16
            or torch.cuda.current_device() != self.original.device.index
            or self.head.tp_size != 4
            or self.head.num_embeddings_per_partition != layout.N
            or self.head.num_added_embeddings != 0
            or processor.org_vocab_size != 4 * layout.N
            or processor.logits_as_input
            or processor.head_dtype not in (None, torch.bfloat16)
            or self.head.bias is not None
            or method_type
            not in (
                "vllm.model_executor.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod",
                "vllm.model_executor.layers.linear.UnquantizedLinearMethod",
            )
            or getattr(method.apply, "__func__", None) is not type(method).apply
        ):
            raise ValueError(
                "Unsupported, already-hooked, padded or non-BF16 draft head"
            )
        self.module = load_module()
        if Path(gpu_pack.__file__).resolve() != (ROOT / "gpu_pack.py").resolve():
            raise ValueError("A different bundle owns the imported GPU packer")
        self.method, self.original_apply = method, method.apply
        self.storage = torch.empty(
            layout.SHAPE, dtype=torch.bfloat16, device=self.original.device
        )
        gpu_pack.pack_bf16_words[(triton.cdiv(layout.PADDED_N * layout.K, 1024),)](
            self.original.view(torch.uint16),
            self.storage.view(torch.uint16),
            layout.N,
            layout.K,
            layout.BN,
            layout.BK,
            layout.PADDED_N * layout.K,
            1024,
        )
        # This is an opaque descriptor over ALL storage, not an affine logical B.
        self.physical_b = self.storage.view(layout.PADDED_N, layout.K).t()
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(self.original.device))
        self.original_pointer, self.packed_pointer = (
            self.original.data_ptr(),
            self.storage.data_ptr(),
        )
        self.original_version, self.packed_version = (
            tensor_version(self.original),
            tensor_version(self.storage),
        )
        self.parameter = None
        self.streams = {}
        self.native_dispatch_verified = False
        self.active = False
        self.hook = None

    def validate(self):
        """Reject replacement/resizing/mutation; retain graph-owned storage for life."""
        if (
            self.head.weight is not self.original
            or self.original.data_ptr() != self.original_pointer
            or self.storage.data_ptr() != self.packed_pointer
            or self.physical_b.data_ptr() != self.packed_pointer
            or tuple(self.physical_b.shape) != (layout.K, layout.PADDED_N)
            or self.physical_b.stride() != (1, layout.K)
            or self.storage.storage_offset() != 0
            or self.storage.untyped_storage().nbytes()
            != layout.memory_cost()["packed_bytes"]
            or tensor_version(self.original) != self.original_version
            or tensor_version(self.storage) != self.packed_version
        ):
            raise ValueError(
                "Packed/original head storage changed; rebuild outside all graphs"
            )

    def _stream_key(self, stream):
        """Different stream wrappers over the same HIP stream share prepared state."""
        return self.original.device.index, int(stream.cuda_stream)

    def prime_stream(self, hidden, stream):
        """Warm the real native head and private GEMM on the exact future capture stream."""
        import torch

        from aiter.ops.flydsl import gemm_kernels

        if torch.cuda.is_current_stream_capturing() or not eligible(
            hidden, self.original
        ):
            raise ValueError("Prime an eligible head outside capture")
        if stream.device != self.original.device:
            raise ValueError("Capture stream must belong to the head device")
        self.validate()
        with torch.cuda.stream(stream):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("The requested priming stream is already capturing")
            stream.wait_event(self.ready)
            original_dispatch = gemm_kernels.flydsl_hgemm
            signature = inspect.signature(original_dispatch)
            observed = []

            def observe(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                if bound.arguments["b"].data_ptr() == self.original_pointer:
                    verify_native_call(bound.arguments, self.original)
                    observed.append(True)
                return original_dispatch(*args, **kwargs)

            gemm_kernels.flydsl_hgemm = observe
            try:
                reference = self.original_apply(self.head, hidden, bias=None)
            finally:
                gemm_kernels.flydsl_hgemm = original_dispatch
            if observed != [True]:
                raise ValueError(
                    "Native execution did not reach the audited FlyDSL head exactly once"
                )
            self.native_dispatch_verified = True
            if self.parameter is None:
                kwargs = {
                    **layout.CONFIG,
                    "a_is_transposed": False,
                    "b_is_transposed": True,
                    "in_dtype_id": self.module.GEMM_A16W16_DTYPE_BF16,
                    "out_dtype_id": self.module.GEMM_A16W16_DTYPE_BF16,
                    "has_bias": False,
                }
                self.parameter = self.module.make_gemm_a16w16_param_and_validate(
                    layout.M, layout.N, layout.K, kwargs
                )
                if self.parameter is None:
                    raise ValueError(
                        "Current native parameter factory rejects the head"
                    )
            buffers = self.module.get_split_k_buffers(stream, self.original.device)
            key = self._stream_key(stream)
            self.streams[key] = (stream, buffers)
            try:
                candidate = self.gemm(hidden)
                compiled = getattr(
                    self.module.gemm_a16w16_gfx950, "_compiled_cache", {}
                )
                if compiled.get(self.parameter.__cache_signature__()) is None:
                    raise RuntimeError(
                        "GPU priming did not materialize an executable; compile-only is not a gate"
                    )
            except BaseException:
                self.streams.pop(key, None)
                raise
        return reference, candidate

    def gemm(self, hidden, out=None):
        """Use current nine-argument ABI; never allocate/init workspaces here."""
        import torch

        if not eligible(hidden, self.original):
            raise ValueError("Packed GEMM is restricted to M24/N50016/K6144 BF16")
        if torch.cuda.current_device() != self.original.device.index:
            raise ValueError("Packed GEMM requires the original worker device")
        self.validate()
        stream = torch.cuda.current_stream(self.original.device)
        entry = self.streams.get(self._stream_key(stream))
        if entry is None or self.parameter is None:
            raise RuntimeError(
                "Prime this exact stream before capture; no lazy workspace initialization"
            )
        if out is None:
            out = torch.empty(
                (layout.M, layout.N), dtype=hidden.dtype, device=hidden.device
            )
        if (
            tuple(out.shape) != (layout.M, layout.N)
            or out.dtype != hidden.dtype
            or out.device != hidden.device
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            raise ValueError("Output must contain only the logical local vocabulary")
        semaphore, signal = entry[1]
        a_arg = self.module._dynamic_tensor_arg(hidden, 1)
        b_arg = self.module._dynamic_tensor_arg(self.physical_b, 0)
        out_arg = self.module._dynamic_tensor_arg(out, 1)
        arguments = (
            out_arg,
            a_arg,
            b_arg,
            a_arg,
            semaphore,
            signal,
            layout.SPLITS,
            self.parameter,
            stream,
        )
        self.module.run_cached(
            self.module.gemm_a16w16_gfx950,
            *arguments,
            constexpr_param=self.parameter,
            compiler=self.module.flyc.compile,
            dispatch_args=arguments,
        )
        return out

    def attach(self):
        """Preserve the native quant-method type; other heads and shapes delegate."""
        if self.hook is not None or self.method.apply != self.original_apply:
            raise RuntimeError("The quant method already has a hook")
        if not self.streams:
            raise RuntimeError("Prime before attaching the packed head")

        def apply(layer, hidden, bias=None):
            if (
                self.active
                and layer is self.head
                and eligible(hidden, self.original, bias)
            ):
                return self.gemm(hidden)
            return self.original_apply(layer, hidden, bias=bias)

        self.hook = apply
        self.method.apply = apply

    @contextmanager
    def use(self, candidate):
        """Choose the graph being captured, without changing any sampling caller."""
        previous = self.active
        self.active = bool(candidate)
        try:
            yield
        finally:
            self.active = previous

    def detach(self):
        """Restore the owned hook only; caller must retain this object until graph reset."""
        if self.hook is None:
            return
        if self.method.apply is not self.hook:
            raise RuntimeError("Another owner changed the head hook")
        del self.method.apply
        self.hook = None
