"""Portable, untimed A0-reference numerical probe for independent R7/R8.

Only the selected arm's installed hook is imported. Numerical logic is preserved
from the frozen shared driver SHA256
299fe7b479c9fbaa826823d4d7c3603c5ffb7694c6e9f8440077a8b6fa08f8b0.
Run in a fresh TP4 process, separate from any formal E2E server.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from cleanup_protocol import run_cleanup

A0_IMAGE = "sha256:2695afe0011d4a5c9ab394a8252a590fe0c0ee4497d483e50606e996762952a7"
ARMS = {"R7": "r07_gemm1", "R8": "r08_gemm2_tp"}
ORIGINAL_DRIVER_SHA256 = (
    "299fe7b479c9fbaa826823d4d7c3603c5ffb7694c6e9f8440077a8b6fa08f8b0"
)
PRE_TP_FIELDS = ("g1", "local", "tids", "eids", "counts", "sweights")
SNAPSHOT_FIELDS = (*PRE_TP_FIELDS, "global")


def sha(path):
    """Hash existing evidence without modifying it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_sequence(live, finish_global):
    """Clone pre-TP values before a collective can alias/overwrite its input.

    The check-graph capture invokes this helper synchronously, so every clone
    becomes part of that graph. Live references are retained only for poisoning.
    """
    if set(live) != set(PRE_TP_FIELDS):
        raise AssertionError("Incomplete live Stage1/Stage2 metadata")
    saved = {name: live[name].clone() for name in PRE_TP_FIELDS}
    live["global"] = finish_global()
    saved["global"] = live["global"].clone()
    return live, saved


def buffer_span(tensor):
    """Return the exact occupied byte interval of a nonempty contiguous tensor."""
    if not tensor.is_contiguous():
        raise AssertionError("Snapshot alias audit requires contiguous tensors")
    start = int(tensor.data_ptr())
    size = int(tensor.numel()) * int(tensor.element_size())
    if start <= 0 or size <= 0:
        raise AssertionError("Empty or invalid check buffer")
    return {"device": str(tensor.device), "start": start, "end": start + size}


def spans_overlap(left, right):
    """Compare occupied ranges, not just base-pointer equality or storage IDs."""
    return left["device"] == right["device"] and max(
        left["start"], right["start"]
    ) < min(left["end"], right["end"])


def audit_snapshot_buffers(live_bindings, saved_bindings):
    """Reject saved/saved and saved/live overlap, including cross-field aliases."""
    if len(live_bindings) != 2 or len(saved_bindings) != 2:
        raise AssertionError("Both reference and candidate bindings are required")
    live, saved = {}, {}
    for label, live_binding, saved_binding in zip(
        ("a0", "candidate"),
        live_bindings,
        saved_bindings,
        strict=True,
    ):
        if set(live_binding) != set(SNAPSHOT_FIELDS) or set(saved_binding) != set(
            SNAPSHOT_FIELDS
        ):
            raise AssertionError("Missing captured check snapshots")
        for name in SNAPSHOT_FIELDS:
            live[f"{label}.{name}"] = buffer_span(live_binding[name])
            saved[f"{label}.{name}"] = buffer_span(saved_binding[name])
    items = list(saved.items())
    for index, (name, span) in enumerate(items):
        for other_name, other_span in items[index + 1 :]:
            if spans_overlap(span, other_span):
                raise AssertionError(f"Saved buffers overlap: {name}, {other_name}")
        for live_name, live_span in live.items():
            if spans_overlap(span, live_span):
                raise AssertionError(
                    f"Snapshot aliases live buffer: {name}, {live_name}"
                )
    live_items = list(live.items())
    aliases = [
        [name, other_name]
        for index, (name, span) in enumerate(live_items)
        for other_name, other_span in live_items[index + 1 :]
        if spans_overlap(span, other_span)
    ]
    return {
        "saved_buffers_disjoint": True,
        "saved_live_disjoint": True,
        "saved_ranges": saved,
        "live_ranges": live,
        "permitted_live_aliases": aliases,
    }


def poison_live(binding):
    """Poison only live buffers immediately before their own graph is replayed."""
    for name in ("g1", "local", "global", "sweights"):
        binding[name].fill_(float("nan"))
    for name in ("tids", "eids", "counts"):
        binding[name].fill_(-1)


def parse_gpu_uuids(value):
    """Parse four explicit, rank-ordered destination UUIDs without device access."""
    try:
        uuids = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "Expected a JSON list of four UUIDs"
        ) from error
    if (
        not isinstance(uuids, list)
        or len(uuids) != 4
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in uuids
        )
        or len(set(uuids)) != 4
    ):
        raise argparse.ArgumentTypeError("Expected four unique, nonempty UUID strings")
    return uuids


def validate_launch(environment):
    """Require one local TP4 group with an unambiguous HIP device selection."""
    devices = environment.get("HIP_VISIBLE_DEVICES", "").split(",")
    if (
        len(devices) != 4
        or any(not re.fullmatch(r"0|[1-9][0-9]*", device) for device in devices)
        or len(set(devices)) != 4
        or environment.get("ROCR_VISIBLE_DEVICES") not in (None, "")
        or environment.get("WORLD_SIZE") != "4"
        or environment.get("LOCAL_WORLD_SIZE", "4") != "4"
        or environment.get("LOCAL_RANK") not in ("0", "1", "2", "3")
        or environment.get("RANK") != environment.get("LOCAL_RANK")
    ):
        raise ValueError(
            "Require four unique HIP_VISIBLE_DEVICES indices, unset "
            "ROCR_VISIBLE_DEVICES, and single-node WORLD_SIZE=4"
        )
    return devices


def validate_observed(uuids, architectures, device_counts, expected):
    """Reject remapped/duplicate GPUs or a non-gfx950 runtime before IPC setup."""
    if (
        uuids != expected
        or len(uuids) != 4
        or len(set(uuids)) != 4
        or len(architectures) != 4
        or not all(arch.split(":")[0] == "gfx950" for arch in architectures)
        or device_counts != [4] * 4
    ):
        raise ValueError("Observed rank-ordered GPU UUIDs/gfx950/visible count differ")


def installed_roots():
    """Resolve active packages instead of assuming a site-packages location."""
    roots = {}
    for name in ("aiter", "vllm"):
        spec = importlib.util.find_spec(name)
        locations = [] if spec is None else list(spec.submodule_search_locations or [])
        if len(locations) != 1:
            raise ValueError(f"Require one installed {name} package root")
        roots[name] = Path(locations[0]).resolve()
    return roots


def installed_path(name, roots):
    """Resolve a manifest-relative file only within an audited package root."""
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 2
        or path.parts[0] not in roots
    ):
        raise ValueError(f"Invalid installed source path: {name}")
    return roots[path.parts[0]].joinpath(*path.parts[1:])


def source_check(root, role, entry):
    """Verify packaged and installed payloads plus unchanged A0 primitives."""
    arm = root / "arms" / ARMS[role]
    manifest_path = arm / "patch_manifest.json"
    baseline_path = root / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    a0 = json.loads(baseline_path.read_text())
    if (
        manifest["arm"] != role
        or manifest["base_image"] != A0_IMAGE
        or a0["image_id"] != A0_IMAGE
    ):
        raise ValueError("Wrong independent arm/A0 export")
    roots = installed_roots()
    actual = {}
    for name, expected in manifest["payload"].items():
        actual[name] = sha(installed_path(name, roots))
        if actual[name] != expected or sha(arm / "payload" / name) != expected:
            raise ValueError(f"Packaged/installed payload mismatch: {name}")
    references = {}
    for name in (
        "aiter/fused_moe.py",
        "aiter/ops/flydsl/moe_kernels.py",
        "aiter/ops/flydsl/kernels/moe_2stage_a16wmix/gemm1.py",
        "aiter/ops/flydsl/kernels/moe_2stage_a16wmix/gemm2.py",
        "aiter/ops/flydsl/kernels/moe_2stage_a16wmix/utils.py",
        "aiter/ops/flydsl/kernels/mxfp4_gemm_common.py",
        "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py",
        "vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py",
    ):
        references[name] = sha(installed_path(name, roots))
        if references[name] != a0["sources"][name]:
            raise ValueError(f"A0 reference primitive changed: {name}")
    act = "aiter/ops/flydsl/kernels/act.py"
    references[act] = sha(installed_path(act, roots))
    if (
        references[act]
        != "0602318cdc3ee64fcef18ad79d731848f26f479c7a2badb65046e7727e10ddee"
    ):
        raise ValueError("Global A0 activation changed")
    return {
        "baseline_manifest_sha256": sha(baseline_path),
        "patch_manifest_sha256": sha(manifest_path),
        "probe_driver_sha256": sha(__file__),
        "probe_entry_sha256": sha(entry),
        "cleanup_protocol_sha256": sha(Path(__file__).with_name("cleanup_protocol.py")),
        "original_driver_sha256": ORIGINAL_DRIVER_SHA256,
        "payload": actual,
        "a0_reference_sources": references,
        "installed_package_roots": {name: str(path) for name, path in roots.items()},
    }


def run(root, role, image_id, entry, report, expected_gpu_uuids):
    """Validate complete sort/clear + G1/activation + G2 + TP; collect no timings."""
    import aiter
    import aiter.fused_moe as fm
    import torch
    import torch.distributed as dist
    from aiter import dtypes
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    def gather(value):
        rows = [None] * 4
        dist.all_gather_object(rows, value)
        return rows

    sources = source_check(root, role, entry)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuids = gather(str(props.uuid))
    architectures = gather(props.gcnArchName)
    device_counts = gather(torch.cuda.device_count())
    report.update(
        sources=sources,
        source_identity=hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()
        ).hexdigest(),
        image_id=image_id,
        gpu_uuids=uuids,
        observed_gpu_uuids=uuids,
        gpu_architectures=architectures,
        visible_device_counts=device_counts,
        torch=torch.__version__,
        hip=torch.version.hip,
    )
    validate_observed(uuids, architectures, device_counts, expected_gpu_uuids)
    identities = gather(report["source_identity"])
    if len(set(identities)) != 1:
        raise ValueError("Ranks loaded different package/source identities")
    rank = dist.get_rank()

    def weight(shape, seed):
        torch.manual_seed(seed)
        value = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
        packed, scale = quant(value, quant_dtype=dtypes.fp4x2)
        packed = packed.view(shape[0], shape[1], shape[2] // 2)
        packed = shuffle_weight_a16w4(packed, 16, False)
        packed.is_shuffled = True
        return packed, shuffle_scale_a16w4(scale, 129, False)

    w1, s1 = weight((129, 1536, 6144), 20261018 + rank * 100)
    w2, s2 = weight((129, 6144, 768), 20261019 + rank * 100)
    hidden = torch.empty((96, 6144), dtype=torch.bfloat16, device="cuda")
    ids = torch.empty((96, 5), dtype=torch.int32, device="cuda")
    route_weights = torch.empty((96, 5), dtype=torch.float32, device="cuda")
    latest = {}

    def set_frame(step):
        generator = torch.Generator(device="cuda").manual_seed(7100 + step)
        hidden.copy_(
            torch.randn(
                hidden.shape, dtype=hidden.dtype, device="cuda", generator=generator
            )
        )
        if step == 1:
            hidden.zero_()
        scores = torch.randn((96, 128), device="cuda", generator=generator)
        routed = (
            torch.tensor([0, 17, 53, 127], device="cuda").expand(96, 4)
            if step % 2
            else scores.topk(4, dim=1).indices
        )
        values = scores.gather(1, routed).sigmoid()
        ids[:, :4].copy_(routed)
        ids[:, 4].fill_(128)
        route_weights[:, :4].copy_(values / values.sum(1, keepdim=True) * 2)
        route_weights[:, 4].fill_(1)

    def original(x):
        calls = []
        fm.kernel_bench_callable = calls
        try:
            result = fm._fused_moe_impl(
                hidden_states=x,
                w1=w1,
                w2=w2,
                topk_weight=route_weights,
                topk_ids=ids,
                activation=aiter.ActivationType.Swiglu.value,
                quant_type=aiter.QuantType.per_1x32.value,
                doweight_stage1=False,
                w1_scale=s1,
                w2_scale=s2,
                block_size_M=-1,
                dtype=torch.bfloat16,
                _q_dtype_a=dtypes.bf16,
                gate_mode="separated",
                swiglu_limit=7.0,
            )
        finally:
            fm.kernel_bench_callable = None
        if [name for name, _ in calls] != ["stage1", "stage2"]:
            raise AssertionError("Expected the complete native two-stage pipeline")
        call = calls[1][1]
        if call.keywords["block_m"] != 16 or tuple(call.args[0].shape) != (96, 5, 768):
            raise AssertionError(
                "Actual A0 BM16/token-slot metadata differs from the audited contract"
            )
        latest.clear()
        latest.update(
            g1=call.args[0],
            tids=call.args[3],
            eids=call.args[4],
            counts=call.args[5],
            local=result,
            sweights=call.keywords["sorted_weights"],
        )
        return result

    set_frame(0)
    original(hidden)  # Warm A0 kernels without entering any GPU collective.
    torch.cuda.synchronize()
    dist.barrier()
    integration = workspace = communicator = None
    graphs = []
    try:
        if role == "R7":
            from vllm.models.minimax_m3.amd.ops._r07_g1 import integration

            layer = SimpleNamespace(
                moe_config=SimpleNamespace(
                    tp_size=4, ep_size=1, dp_size=1, is_sequence_parallel=False
                ),
                enable_dbo=False,
                routed_input_transform=None,
                routed_output_transform=None,
                _forward_impl=original,
            )
            config = SimpleNamespace(
                hidden_size=6144,
                intermediate_size=3072,
                num_local_experts=128,
                num_experts_per_tok=4,
                n_shared_experts=1,
                routed_scaling_factor=2,
                swiglu_limit=7,
            )
            integration.configure_for_layer(layer, config, use_aiter_fse=True)
            candidate = layer._forward_impl
        else:
            from vllm.models.minimax_m3.amd.ops import (
                moe_route_collective as integration,
            )
            from vllm.models.minimax_m3.amd.ops._moe_atomic_pipeline.sort_clear import (
                install,
            )

            integration._state = integration._RouteState(dist.group.WORLD)
            workspace = integration._state.collective
            integration._original_stage2 = fm.cktile_moe_stage2
            fm.cktile_moe_stage2 = integration._stage2
            install(integration)
            fm.get_2stage_cfgs.cache_clear()
            candidate = integration._wrap_forward(original)
        candidate(hidden)
        torch.cuda.synchronize()
        dist.barrier()
        communicator = CustomAllreduce(
            dist.group.WORLD,
            hidden.device,
            max_size=128 * 1024 * 1024,
        )
        if communicator.disabled:
            raise RuntimeError("Native A0 TP all-reduce must be available")

        def sequence(active, *, check_graph=False):
            result = candidate(hidden) if active else original(hidden)
            live = dict(latest)
            if check_graph and not torch.cuda.is_current_stream_capturing():
                raise AssertionError(
                    "Validation snapshots must be captured in the check graph"
                )
            if active and role == "R8":
                live["local"] = workspace.merged
                return snapshot_sequence(live, lambda: result)
            return snapshot_sequence(
                live, lambda: communicator.custom_all_reduce(result)
            )

        for active in (False, True):
            with communicator.capture():
                sequence(active)
            torch.cuda.synchronize()
            dist.barrier()
        live_bindings, saved_bindings = [], []
        for active in (False, True):
            graph = torch.cuda.CUDAGraph()
            with communicator.capture(), torch.cuda.graph(graph):
                live, saved = sequence(active, check_graph=True)
            graphs.append(graph)
            live_bindings.append(live)
            saved_bindings.append(saved)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
        storage_audit = audit_snapshot_buffers(live_bindings, saved_bindings)
        report["snapshot_storage"] = gather(storage_audit)
        report["validation_copies_inside_check_graph"] = True
        reference, actual = saved_bindings

        def error(value, expected):
            norm = float(torch.linalg.vector_norm(expected.float()))
            difference = float(
                torch.linalg.vector_norm(value.float() - expected.float())
            )
            relative = (
                (difference / norm if norm > 0 else (0.0 if difference == 0 else None))
                if math.isfinite(norm) and math.isfinite(difference)
                else None
            )
            finite = bool(
                torch.isfinite(value).all() and torch.isfinite(expected).all()
            )
            return {
                "finite": finite,
                "relative_l2": relative,
                "passed": finite and relative is not None and relative <= 0.005,
            }

        def bits_equal(left, right):
            return (
                left.dtype == right.dtype
                and left.shape == right.shape
                and torch.equal(
                    left.contiguous().view(torch.uint8),
                    right.contiguous().view(torch.uint8),
                )
            )

        def tp_reference(local):
            peers = gather(local.detach().cpu())
            result = torch.empty_like(local)
            for owner in range(4):
                rows = slice(owner * 24, (owner + 1) * 24)
                total = peers[owner][rows].float()
                for offset in range(1, 4):
                    total = total + peers[(owner + offset) % 4][rows].float()
                result[rows].copy_(total.bfloat16().to(local.device))
            return result

        def metadata(binding):
            used = int(binding["counts"][0].item())
            if used < 0 or used % 16 or used > binding["tids"].numel():
                return False
            packed = binding["tids"][:used]
            tokens, slots = packed & 0xFFFFFF, packed >> 24
            valid = tokens < 96
            if int(valid.sum()) != 480 or not bool(
                ((slots[valid] >= 0) & (slots[valid] < 5)).all()
            ):
                return False
            route = tokens[valid] * 5 + slots[valid]
            rows = torch.arange(used, device="cuda")[valid] // 16
            return (
                torch.equal(
                    route.sort().values,
                    torch.arange(480, device="cuda", dtype=route.dtype),
                )
                and torch.equal(binding["eids"][rows], ids.flatten()[route])
                and torch.equal(
                    binding["sweights"][:used][valid], route_weights.flatten()[route]
                )
            )

        for step in range(32):
            set_frame(step)
            saved_inputs = (hidden.clone(), ids.clone(), route_weights.clone())
            current_storage = audit_snapshot_buffers(live_bindings, saved_bindings)
            if current_storage != storage_audit:
                raise AssertionError("Captured check-buffer ranges changed")
            poison_live(live_bindings[0])
            graphs[0].replay()
            torch.cuda.synchronize()
            before = workspace.state()["epochs"] if workspace is not None else None
            dist.barrier()
            # Native communicator/live outputs may be shared between arms.
            # Poison the candidate only after the reference graph saved its data.
            poison_live(live_bindings[1])
            if rank == step % 4:
                time.sleep(
                    0.002
                )  # Untimed bounded rank-arrival skew, not a performance sample.
            graphs[1].replay()
            torch.cuda.synchronize()
            epoch_ok = True
            if workspace is not None:
                states = workspace.check()
                epoch_ok = states[rank]["epochs"] == [epoch + 1 for epoch in before]
            local = {
                "g1": error(actual["g1"], reference["g1"]),
                "g1_bitwise": bits_equal(actual["g1"], reference["g1"]),
                "local_moe": error(actual["local"], reference["local"]),
                "global_moe": error(actual["global"], reference["global"]),
                "a0_metadata": metadata(reference),
                "candidate_metadata": metadata(actual),
                "a0_tp_bitwise": bits_equal(
                    reference["global"], tp_reference(reference["local"])
                ),
                "candidate_tp_bitwise": bits_equal(
                    actual["global"], tp_reference(actual["local"])
                ),
                "epoch_increment_one": epoch_ok,
                "saved_buffers_disjoint": current_storage["saved_buffers_disjoint"],
                "saved_live_disjoint": current_storage["saved_live_disjoint"],
                "inputs_unchanged": all(
                    bits_equal(x, y)
                    for x, y in zip(
                        (hidden, ids, route_weights), saved_inputs, strict=True
                    )
                ),
            }
            local["passed"] = (
                local["g1"]["passed"]
                and local["local_moe"]["passed"]
                and local["global_moe"]["passed"]
                and (role != "R8" or local["g1_bitwise"])
                and local["a0_metadata"]
                and local["candidate_metadata"]
                and local["a0_tp_bitwise"]
                and local["candidate_tp_bitwise"]
                and epoch_ok
                and local["inputs_unchanged"]
                and local["saved_buffers_disjoint"]
                and local["saved_live_disjoint"]
            )
            ranks = gather(local)
            report["checks"].append(
                {
                    "step": step,
                    "hot": bool(step % 2),
                    "skew_rank": step % 4,
                    "ranks": ranks,
                }
            )
            if not all(row["passed"] for row in ranks):
                raise AssertionError(
                    f"{role} complete numerical comparison failed at "
                    f"step {step}: {ranks}"
                )
        for label, graph in zip(("a0", "candidate"), graphs, strict=True):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                graph.replay()
                torch.cuda.synchronize()
            names = [
                event.name
                for event in prof.events()
                if event.device_type.name == "CUDA"
            ]
            target = (
                "r07_gemm1_slot_"
                if role == "R7"
                else "mm3_r08_atomic_system_tp_v1_m96_cm1"
            )
            other = "mm3_r08_atomic" if role == "R7" else "r07_gemm1_slot_"
            if sum(target in name for name in names) != int(
                label == "candidate"
            ) or any(other in name for name in names):
                raise AssertionError("Wrong optimized/reference dispatch")
            report.setdefault("dispatch", {})[label] = gather(names)
        if source_check(root, role, entry) != sources:
            raise ValueError("Sources changed during probe")
    finally:
        actions = [("synchronize", torch.cuda.synchronize)]
        actions += [
            (f"reset_graph_{index}", graph.reset) for index, graph in enumerate(graphs)
        ]
        if workspace is not None:
            actions.append(("close_candidate_ipc", workspace.close))
        if communicator is not None:
            actions.append(("close_native_comm", communicator.close))
        run_cleanup(actions, primary_error=sys.exc_info()[1])


def output_location(package, output):
    """Require separate evidence storage; never write into the frozen package."""
    output = output.resolve()
    if output.is_relative_to(package.resolve()):
        raise ValueError("--output-dir must be outside the handoff package")
    return output


def main():
    """Run an explicitly selected arm; no benchmark baseline is measured."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuids", type=parse_gpu_uuids, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id):
        raise ValueError("Actual candidate image ID required")
    if os.environ.get("MM3_IMAGE_ID", args.image_id) != args.image_id:
        raise ValueError("Runner and probe image identities differ")
    devices = validate_launch(os.environ)
    root = args.package_root.resolve()
    entry = Path(__file__).resolve()
    output = output_location(root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("correctness_probe*.json")):
        raise FileExistsError(
            "Preserve previous numerical evidence; use a fresh output dir"
        )
    rank = int(os.environ["LOCAL_RANK"])
    report = {
        "arm": args.arm,
        "phase": "complete_numerical_probe",
        "passed": False,
        "complete": False,
        "timed": False,
        "performance_measurement": False,
        "baseline_retimed": False,
        "base_image": A0_IMAGE,
        "synthetic_weights_and_routes": True,
        "requires_model_smoke": True,
        "scope": (
            "Untimed A0-reference sort/clear + G1/activation + G2 + TP, "
            "with in-graph check snapshots"
        ),
        "relative_l2_limit": 0.005,
        "native_comm_pool_bytes": 128 * 1024 * 1024,
        "checks": [],
        "expected_gpu_uuids": args.expected_gpu_uuids,
        "hip_visible_devices": devices,
        "image_id": args.image_id,
        "local_rank": rank,
        "package_root": str(root),
        "e2e_eligible": False,
    }

    def save(name):
        with (output / name).open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")

    import torch
    import torch.distributed as dist

    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=300))
        if dist.get_world_size() != 4 or dist.get_rank() != rank:
            raise ValueError("Single-node TP4 required")
        with torch.inference_mode():
            run(root, args.arm, args.image_id, entry, report, args.expected_gpu_uuids)
    except BaseException:
        report["error"] = traceback.format_exc()
        save(f"correctness_probe.rank{rank}.failure.json")
        raise
    finally:
        if dist.is_initialized():
            try:
                run_cleanup(
                    [("destroy_group", dist.destroy_process_group)],
                    primary_error=sys.exc_info()[1],
                )
            except BaseException:
                report["error"] = traceback.format_exc()
                save(f"correctness_probe.rank{rank}.cleanup_failure.json")
                raise
    report.update(passed=True, complete=True)
    if rank == 0:
        save("correctness_probe.json")


if __name__ == "__main__":
    main()
