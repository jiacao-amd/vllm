"""A41 changes only the number of native decode-score partitions."""

CAPS = (384, 432, 576, 768, 1536, 3072)
SOURCE_SHA256 = "34a11711c49a0724df833c447a08e3529dc64a37373d9ade8927c1fe41431dfa"


def bounded_split(native_split, max_blocks, requests, cap):
    """Keep native waves; a smaller fixed grid still covers every live block."""
    if cap not in CAPS:
        raise ValueError("Unknown, unmeasured grid cap")
    chunks, waves = native_split(max_blocks, requests)
    return min(chunks, max(1, cap // max(1, requests))), waves


def covered_blocks(blocks, chunks, waves):
    """CPU model of native decode_prologue plus its wave-strided main loop."""
    per_chunk = (blocks + chunks - 1) // chunks
    return [
        block
        for chunk in range(chunks)
        for wave in range(waves)
        for block in range(
            chunk * per_chunk + wave,
            min((chunk + 1) * per_chunk, blocks),
            waves,
        )
    ]
