"""Compiler preparation through native worker dummy-forward APIs only."""


class CompileCoverageExtension:
    def mm3_compile_coverage(self, cases):
        import torch
        import torch.distributed as dist

        runner = self.model_runner
        records = []
        for case in cases:
            hidden, sampled = runner._dummy_run(
                case["tokens"], uniform_decode=case["uniform_decode"],
                context_len=case["context_len"], randomize_inputs=False,
            )
            if sampled is not None:
                runner._dummy_sampler_run(sampled)
            torch.cuda.synchronize()
            records.append({
                **case,
                "actual_tokens": None if hidden is None else hidden.shape[0],
                "sampled_rows": None if sampled is None else sampled.shape[0],
            })
            del hidden, sampled
        return {
            "rank": dist.get_rank(), "gpu_uuid": str(
                torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
            ), "cases": records, "http_completion_requests": 0,
            "llm_generate_calls": 0,
        }
