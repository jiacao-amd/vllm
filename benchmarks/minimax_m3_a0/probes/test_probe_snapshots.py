"""Stdlib-only snapshot, portability, and preserved-numerics regressions."""

import argparse
import ast
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import correctness_probe as probe

ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    """Small mutable-buffer stand-in; no torch or device imports."""

    next_address = 0x100000

    def __init__(self, values, *, address=None, contiguous=True):
        self.values = list(values)
        if address is None:
            address = Tensor.next_address
            Tensor.next_address += 256
        self.address = address
        self.device = "cpu"
        self.contiguous = contiguous

    def clone(self):
        return Tensor(self.values)

    def fill_(self, value):
        self.values[:] = [value] * len(self.values)

    def data_ptr(self):
        return self.address

    def numel(self):
        return len(self.values)

    def element_size(self):
        return 4

    def is_contiguous(self):
        return self.contiguous


def binding(value):
    """Provide all metadata/output fields present before the TP call."""
    return {name: Tensor([value, value + 1]) for name in probe.PRE_TP_FIELDS}


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        Tensor.next_address = 0x100000

    def captures(self):
        """Create saved sets while intentionally sharing the live comm output."""
        shared_comm = Tensor([0, 0])
        records = []
        for value in (10, 20):
            live = binding(value)

            def finish_global(value=value):
                shared_comm.fill_(value + 100)
                return shared_comm

            records.append(probe.snapshot_sequence(live, finish_global))
        return [record[0] for record in records], [record[1] for record in records]

    def test_local_and_metadata_are_cloned_before_in_place_collective(self):
        live = binding(3)
        original_local = live["local"]

        def finish_global():
            self.assertEqual(original_local.values, [3, 4])
            original_local.fill_(9)
            live["counts"].fill_(99)
            return original_local

        retained, saved = probe.snapshot_sequence(live, finish_global)
        self.assertIs(retained["local"], retained["global"])
        self.assertEqual(saved["local"].values, [3, 4])
        self.assertEqual(saved["counts"].values, [3, 4])
        self.assertEqual(saved["global"].values, [9, 9])
        self.assertIsNot(saved["local"], saved["global"])

    def test_shared_live_comm_does_not_alias_saved_comparisons(self):
        live, saved = self.captures()
        self.assertIs(live[0]["global"], live[1]["global"])
        self.assertEqual(saved[0]["global"].values, [110, 110])
        self.assertEqual(saved[1]["global"].values, [120, 120])
        audit = probe.audit_snapshot_buffers(live, saved)
        self.assertTrue(audit["saved_buffers_disjoint"])
        self.assertIn(
            ["a0.global", "candidate.global"], audit["permitted_live_aliases"]
        )

    def test_poisoning_live_buffers_preserves_both_snapshots(self):
        live, saved = self.captures()
        before = [
            {name: tensor.values[:] for name, tensor in arm.items()} for arm in saved
        ]
        for arm in live:
            probe.poison_live(arm)
        self.assertEqual(
            before,
            [{name: tensor.values[:] for name, tensor in arm.items()} for arm in saved],
        )
        self.assertTrue(
            probe.audit_snapshot_buffers(live, saved)["saved_live_disjoint"]
        )

    def test_distinct_pointers_with_overlapping_ranges_are_rejected(self):
        live, saved = self.captures()
        first = saved[0]["g1"].data_ptr()
        saved[1]["counts"] = Tensor([0, 0], address=first + 4)
        with self.assertRaisesRegex(AssertionError, "Saved buffers overlap"):
            probe.audit_snapshot_buffers(live, saved)

    def test_cross_field_snapshot_live_alias_is_rejected(self):
        live, saved = self.captures()
        saved[0]["global"] = live[1]["sweights"]
        with self.assertRaisesRegex(AssertionError, "Snapshot aliases live buffer"):
            probe.audit_snapshot_buffers(live, saved)

    def test_adjacent_ranges_are_disjoint_and_strides_fail_closed(self):
        first = Tensor([1, 2], address=0x1000)
        second = Tensor([3, 4], address=0x1008)
        self.assertFalse(
            probe.spans_overlap(probe.buffer_span(first), probe.buffer_span(second))
        )
        with self.assertRaisesRegex(AssertionError, "contiguous"):
            probe.buffer_span(Tensor([0, 0], contiguous=False))

    def test_shallow_saved_bindings_are_rejected(self):
        live, saved = self.captures()
        saved[0] = dict(live[0])
        with self.assertRaisesRegex(AssertionError, "Snapshot aliases live buffer"):
            probe.audit_snapshot_buffers(live, saved)

    def test_snapshot_sequence_is_called_inside_cuda_graph_capture(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        captures = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and ast.unparse(item.context_expr.func) == "torch.cuda.graph"
                for item in node.items
            )
        ]
        calls = [
            node
            for capture in captures
            for node in ast.walk(capture)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sequence"
        ]
        self.assertEqual(len(calls), 1)
        self.assertTrue(
            any(
                keyword.arg == "check_graph"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in calls[0].keywords
            )
        )


class PortabilityTests(unittest.TestCase):
    def test_explicit_rank_ordered_uuids_reject_duplicates_and_wrong_types(self):
        expected = ["remote-a", "remote-b", "remote-c", "remote-d"]
        self.assertEqual(probe.parse_gpu_uuids(json.dumps(expected)), expected)
        for value in ("{}", "null", "[1,2,3,4]", '["a","a","b","c"]', "[]", "bad"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                probe.parse_gpu_uuids(value)

    def test_destination_gpu_selection_is_not_source_machine_pinned(self):
        env = {
            "HIP_VISIBLE_DEVICES": "0,2,5,7",
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "4",
            "LOCAL_RANK": "2",
            "RANK": "2",
        }
        self.assertEqual(probe.validate_launch(env), ["0", "2", "5", "7"])
        for key, value in (
            ("HIP_VISIBLE_DEVICES", "0,0,2,3"),
            ("HIP_VISIBLE_DEVICES", "0,1,2"),
            ("HIP_VISIBLE_DEVICES", "-1,1,2,3"),
            ("ROCR_VISIBLE_DEVICES", "0,1,2,3"),
            ("WORLD_SIZE", "8"),
            ("RANK", "3"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                probe.validate_launch({**env, key: value})

    def test_observed_uuid_order_architecture_and_device_count_fail_closed(self):
        expected = ["a", "b", "c", "d"]
        arch = ["gfx950:sramecc+:xnack-"] * 4
        probe.validate_observed(expected, arch, [4] * 4, expected)
        for uuids, architectures, counts in (
            (list(reversed(expected)), arch, [4] * 4),
            (["a"] * 4, arch, [4] * 4),
            (expected, ["gfx942"] * 4, [4] * 4),
            (expected, arch, [8] * 4),
        ):
            with self.subTest(uuids=uuids), self.assertRaises(ValueError):
                probe.validate_observed(uuids, architectures, counts, expected)

    def test_output_must_be_separate_and_installed_paths_cannot_escape(self):
        package = Path("/portable/package")
        self.assertEqual(
            probe.output_location(package, Path("/results/r7")),
            Path("/results/r7"),
        )
        for path in (package, package / "probes" / "results"):
            with self.assertRaises(ValueError):
                probe.output_location(package, path)
        roots = {"aiter": Path("/new/site/aiter")}
        self.assertEqual(
            probe.installed_path("aiter/fused_moe.py", roots),
            Path("/new/site/aiter/fused_moe.py"),
        )
        for path in ("/aiter/fused_moe.py", "aiter/../../other", "torch/file.py"):
            with self.assertRaises(ValueError):
                probe.installed_path(path, roots)

    def test_source_check_uses_package_manifests_and_rejects_reference_mutation(self):
        """Mock only installed files; read the real portable manifests/payloads."""
        base = json.loads((ROOT / "baseline_manifest.json").read_text())
        real_sha = probe.sha
        fake_site = Path("/mock-installed")
        roots = {name: fake_site / name for name in ("aiter", "vllm")}
        for role, arm in probe.ARMS.items():
            manifest = json.loads(
                (ROOT / "arms" / arm / "patch_manifest.json").read_text()
            )
            installed = {**base["sources"], **manifest["payload"]}
            installed["aiter/ops/flydsl/kernels/act.py"] = (
                "0602318cdc3ee64fcef18ad79d731848f26f479c7a2badb65046e7727e10ddee"
            )

            def fake_sha(path, installed=installed):
                path = Path(path)
                if path.is_relative_to(fake_site):
                    return installed[str(path.relative_to(fake_site))]
                return real_sha(path)

            with (
                self.subTest(role=role),
                patch.object(probe, "installed_roots", return_value=roots),
                patch.object(probe, "sha", side_effect=fake_sha),
            ):
                report = probe.source_check(ROOT, role, Path(probe.__file__))
                self.assertEqual(report["payload"], manifest["payload"])
                self.assertEqual(
                    report["baseline_manifest_sha256"],
                    real_sha(ROOT / "baseline_manifest.json"),
                )
                installed["aiter/fused_moe.py"] = "0" * 64
                with self.assertRaisesRegex(
                    ValueError, "A0 reference primitive changed"
                ):
                    probe.source_check(ROOT, role, Path(probe.__file__))

    def test_numerical_body_matches_frozen_original_ast(self):
        """Pin all statements from fixture creation through cleanup, not just gates."""
        tree = ast.parse(Path(probe.__file__).read_text())
        run = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        first = next(
            index
            for index, node in enumerate(run.body)
            if isinstance(node, ast.FunctionDef) and node.name == "weight"
        )
        body = ast.dump(
            ast.Module(body=run.body[first:], type_ignores=[]), include_attributes=False
        )
        self.assertEqual(
            hashlib.sha256(body.encode()).hexdigest(),
            "2b563a33febf2698038a4457f52cf290f3a21353dbe8ee9cb1600523202fba5f",
        )

    def test_vendored_cleanup_and_native_sources_are_byte_exact(self):
        pins = {
            "cleanup_protocol.py": (
                "8c0afe6a8847c184efa2e9852ca245486cf0fdde304c72c1da92b6534720b53a"
            ),
            "native/workspace.cu": (
                "3e3e68d91f75a1e086049d838814af2b694cf2cf33902043c2beed8d5faaa1aa"
            ),
            "native/tensor_view.cu": (
                "2289907d7c4219fbe414f481c3e48286e5bd477ebb302883702ecbc7440e0311"
            ),
        }
        for name, expected in pins.items():
            with self.subTest(name=name):
                self.assertEqual(probe.sha(Path(__file__).parent / name), expected)


if __name__ == "__main__":
    unittest.main()
