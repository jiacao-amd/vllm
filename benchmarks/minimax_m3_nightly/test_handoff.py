# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for payload identity and cross-machine protocol preservation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import install
import protocol
import reference_run_e2e as reference
import run_inside


class HandoffTests(unittest.TestCase):
    def test_accepted_overlay_and_frozen_client_have_exact_original_bytes(self):
        accepted = install.package_check(install.HERE.parents[1])
        self.assertEqual(len(accepted["payload_sha256"]), 28)
        self.assertFalse(accepted["goal_9pct_achieved"])
        self.assertEqual(
            set(install.expected_sources("n18")) - set(install.expected_sources("n0")),
            accepted["payload_sha256"].keys() - install.expected_sources("n0").keys(),
        )

    def test_fixed_workload_retains_zero_http_warmup_and_all_37_graph_sizes(self):
        args = reference.client_arguments(6101)
        for key, value in (
            ("--num-prompts", "240"),
            ("--max-concurrency", "24"),
            ("--random-input-len", "8192"),
            ("--random-output-len", "1024"),
            ("--num-warmups", "0"),
        ):
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn("--use-chat-template", args)
        self.assertEqual(len(protocol.COMPILATION["cudagraph_capture_sizes"]), 37)
        self.assertEqual(protocol.SPECULATIVE["synthetic_acceptance_length"], 2.78)
        self.assertEqual(len(protocol.COMPILE_CASES), 18)

    def test_result_rejects_incomplete_or_changed_work(self):
        result = {
            "completed": 240,
            "num_prompts": 240,
            "max_concurrency": 24,
            "total_input_tokens": 1966080,
            "total_output_tokens": 245760,
            "model_id": protocol.SERVED,
            "tokenizer_id": protocol.MAIN,
            "input_lens": [8192] * 240,
            "output_lens": [1024] * 240,
            "errors": [""] * 240,
        }
        run_inside.validate_result(result)
        for delta in (
            {"completed": 239},
            {"input_lens": [8191] + [8192] * 239},
            {"errors": ["request failed"]},
            {"failed": 1},
        ):
            with self.assertRaises(ValueError):
                run_inside.validate_result({**result, **delta})

    def test_environment_cannot_disable_n18_or_enable_optimizations_on_n0(self):
        flags = install.package_check()["required_additional_environment"]
        env = {
            **protocol.ENVIRONMENT,
            "HIP_VISIBLE_DEVICES": "0,1,2,3",
            "MM3_HANDOFF_IMAGE_ID": "sha256:test",
        }
        with (
            patch.dict(run_inside.os.environ, env, clear=True),
            patch.object(run_inside.os, "geteuid", return_value=1234),
        ):
            run_inside.environment("n0")
            with self.assertRaises(ValueError):
                run_inside.environment("n18")
            with patch.dict(run_inside.os.environ, flags):
                run_inside.environment("n18")
                with self.assertRaises(ValueError):
                    run_inside.environment("n0")

    def test_cache_identity_ignores_locks_but_not_kernel_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kernel.hsaco").write_bytes(b"original kernel")
            original = run_inside.cache_fingerprint(root)
            (root / "compile.lock").write_text("transient")
            (root / "lock").write_text("transient")
            self.assertEqual(original, run_inside.cache_fingerprint(root))
            (root / "kernel.hsaco").write_bytes(b"different kernel")
            self.assertNotEqual(original, run_inside.cache_fingerprint(root))

    def test_fresh_preparation_rejects_aiter_home_binaries_but_allows_uv_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "home/.cache/uv").mkdir(parents=True)
            (root / "home/.cache/uv/CACHEDIR.TAG").write_text("bootstrap")
            run_inside.require_empty_compiler_cache(root)
            for name in run_inside.COMPILER_DIRS:
                artifact = root / name / "cached_kernel.so"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b"old kernel")
                with self.assertRaises(ValueError):
                    run_inside.require_empty_compiler_cache(root)
                artifact.unlink()

    def test_preparation_rejects_a_changed_wrapper(self):
        contract = {
            "arm": "n18",
            "image_id": "sha256:test",
            "environment": {},
            "frozen_source_sha256": {},
            "wrapper_sha256": "current",
        }
        previous = {**contract, "wrapper_sha256": "different", "stage": "prepare"}
        with (
            patch.object(install, "load", side_effect=[previous, {}, {}]),
            self.assertRaisesRegex(ValueError, "wrapper_sha256"),
        ):
            run_inside.check_preparation(Path("/unused"), contract)


if __name__ == "__main__":
    unittest.main()
