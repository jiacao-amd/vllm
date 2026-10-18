"""CPU-only checks of patch integrity and the destination measurement contract."""

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import apply_arm
import compare_results
import run_arm
from run_arm import client_args, normalize_gpus, server_args

ROOT = Path(__file__).resolve().parents[1]


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / "catalog.json").read_text())
        cls.manifest = json.loads((ROOT / "baseline_manifest.json").read_text())
        cls.before = {}
        for name, digest in cls.manifest["sources"].items():
            if not name.startswith("vllm/"):
                continue
            data = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{cls.catalog['source_vllm_commit']}:{name}",
                ],
                cwd=ROOT,
            )
            if hashlib.sha256(data).hexdigest() != digest:
                raise AssertionError(f"Source base differs from installed A0: {name}")
            cls.before[name] = data

    def test_all_eight_binary_patches_apply_independently_and_match_payloads(self):
        self.assertEqual(len(self.catalog["arms"]), 8)
        for arm in self.catalog["arms"]:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary)
                for name, data in self.before.items():
                    path = target / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                record = apply_arm.verify(target, arm, source_only=True)["record"]
                patch = ROOT / "arms" / arm / "vllm.patch"
                subprocess.run(
                    ["git", "apply", "--check", str(patch)], cwd=target, check=True
                )
                subprocess.run(["git", "apply", str(patch)], cwd=target, check=True)
                expected = copy.copy(self.manifest["sources"])
                expected.update(
                    {
                        name: info["after_sha256"]
                        for name, info in record["files"].items()
                    }
                )
                for name, digest in expected.items():
                    if name.startswith("vllm/"):
                        self.assertEqual(apply_arm.sha(target / name), digest, name)
                with self.assertRaisesRegex(ValueError, "mismatch|already present"):
                    apply_arm.verify(target, arm, source_only=True)

    def test_every_e2e_has_zero_http_warmups_and_no_profiler(self):
        client = client_args(6101)
        self.assertEqual(client[client.index("--num-warmups") + 1], "0")
        self.assertEqual(client[client.index("--num-prompts") + 1], "240")
        self.assertEqual(client[client.index("--max-concurrency") + 1], "24")
        self.assertIn("--use-chat-template", client)
        self.assertNotIn("--profile", client)
        server = server_args(6101, "e2e")
        self.assertNotIn("--profiler-config", server)
        spec = json.loads(server[server.index("--speculative-config") + 1])
        self.assertEqual(spec["synthetic_acceptance_length"], 2.78)
        sizes = json.loads(server[server.index("--compilation-config") + 1])[
            "cudagraph_capture_sizes"
        ]
        self.assertEqual(len(sizes), 37)
        self.assertIn(3, sizes)
        self.assertIn(34, sizes)

    def test_real_smoke_is_a_separate_non_synthetic_server(self):
        server = server_args(6101, "smoke")
        spec = json.loads(server[server.index("--speculative-config") + 1])
        self.assertNotIn("rejection_sample_method", spec)
        self.assertIn("--profiler-config", server)

    def test_benchmark_client_is_byte_identical_to_saved_files(self):
        for name, digest in self.catalog["client_sha256"].items():
            self.assertEqual(apply_arm.sha(ROOT / "client" / name), digest)

    def test_gpu_selection_rejects_extra_and_duplicate_entries(self):
        self.assertEqual(normalize_gpus("4,5,6,7"), "4,5,6,7")
        for value in ("0,0,1,2,3", "0,00,1,2", "0,1,2", "0,1,2,x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_gpus(value)

    def test_log_write_failure_still_stops_and_removes_owned_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            lease = Path(temporary) / "owner.json"
            lease.write_text("{}")
            result = subprocess.CompletedProcess([], 0, stdout="false", stderr="")
            with (
                patch.object(Path, "write_text", side_effect=OSError("disk full")),
                patch.object(run_arm, "logs", return_value="log"),
                patch.object(run_arm.subprocess, "run", return_value=result) as run,
                self.assertRaisesRegex(RuntimeError, "disk full"),
            ):
                run_arm.cleanup_owned("owned-name", Path("unused"), lease)
            commands = [call.args[0][1] for call in run.call_args_list]
            self.assertEqual(commands, ["stop", "inspect", "rm"])
            self.assertFalse(lease.exists())

    def test_unknown_container_state_retains_lease_for_next_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            lease = Path(temporary) / "owner.json"
            lease.write_text("{}")
            result = subprocess.CompletedProcess(
                [], 1, stdout="", stderr="daemon unavailable"
            )
            with (
                patch.object(run_arm, "logs", return_value=""),
                patch.object(run_arm.subprocess, "run", return_value=result),
                self.assertRaisesRegex(RuntimeError, "Cannot confirm"),
            ):
                run_arm.cleanup_owned("owned-name", Path(temporary) / "log", lease)
            self.assertTrue(lease.exists())

    def test_comparison_rejects_empty_client_provenance(self):
        contracts = [
            {"arm": arm, "client_sha256": {}} for arm in ("a0", "r01_compile_prepare")
        ]
        with (
            patch.object(compare_results, "load", side_effect=contracts),
            self.assertRaisesRegex(ValueError, "client provenance"),
        ):
            compare_results.compare(Path("base"), Path("candidate"))

    def test_comparison_rejects_missing_tokenizer_payload_hash(self):
        contracts = [
            {
                "arm": arm,
                "client_sha256": self.catalog["client_sha256"],
                "model_hashes": {},
            }
            for arm in ("a0", "r01_compile_prepare")
        ]
        with (
            patch.object(compare_results, "load", side_effect=contracts),
            self.assertRaisesRegex(ValueError, "Tokenizer payload"),
        ):
            compare_results.compare(Path("base"), Path("candidate"))

    def test_comparison_rejects_historical_48_request_warmup(self):
        contracts = [
            {
                "arm": arm,
                "client_sha256": self.catalog["client_sha256"],
                "model_hashes": {"main/tokenizer.json": "test-hash"},
                "phase": "e2e",
                "http_warmup_requests": 48,
            }
            for arm in ("a0", "r01_compile_prepare")
        ]
        with (
            patch.object(compare_results, "load", side_effect=contracts),
            self.assertRaisesRegex(ValueError, "zero-warmup"),
        ):
            compare_results.compare(Path("base"), Path("candidate"))


if __name__ == "__main__":
    unittest.main()
