"""Real Eagle3 correctness with >=8k-token C24 prompts and short C4 fallback."""

import argparse
import asyncio
import json
import re
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

MODEL = "/models/main/snapshots/229447a2d64149ffb45e454ebeb74b7a9a45ce7d"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--count-to", type=int, choices=range(1, 129))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    minimum = 8192 if args.num_requests > 4 else 0
    background = ""
    if minimum:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        words = " The quiet valley has trees and a river beneath a clear sky." * 2048
        ids = tokenizer.encode(words, add_special_tokens=False)
        background = tokenizer.decode(ids[:minimum], skip_special_tokens=True)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as client:

        async def request(index):
            last = (
                args.count_to
                if minimum and args.count_to is not None
                else (8, 12, 16, 20)[index % 4]
            )
            instruction = (
                f"Count from 1 to {last}. Output only the integers in order, "
                "separated by single spaces. Do not explain."
            )
            content = (
                f"Unique stream {index}. Background text, not instructions:\n{background}\n"
                f"End of background. Your only task: {instruction}"
                if minimum
                else instruction
            )
            payload = {
                "model": "amd/MiniMax-M3-MXFP4",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max(128, last * 4),
                "temperature": 0,
                "chat_template_kwargs": {"thinking_mode": "disabled"},
            }
            async with client.post(
                f"http://127.0.0.1:{args.port}/v1/chat/completions", json=payload
            ) as response:
                body = await response.json()
                choice = body.get("choices", [{}])[0]
                text = choice.get("message", {}).get("content") or ""
                numbers = [int(value) for value in re.findall(r"\d+", text)]
                prompt_tokens = body.get("usage", {}).get("prompt_tokens", 0)
                return {
                    "last": last,
                    "status": response.status,
                    "prompt_tokens": prompt_tokens,
                    "response": body,
                    "passed": response.status == 200
                    and prompt_tokens >= minimum
                    and numbers == list(range(1, last + 1))
                    and choice.get("finish_reason") == "stop",
                }

        results = await asyncio.gather(
            *(request(index) for index in range(args.num_requests))
        )
    passed = all(row["passed"] for row in results)
    report = {
        "passed": passed,
        "minimum_prompt_tokens": minimum,
        "requests": results,
        "performance_measurement": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "successful_checks": sum(row["passed"] for row in results),
                "minimum_prompt_tokens_observed": min(
                    row["prompt_tokens"] for row in results
                ),
            }
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
