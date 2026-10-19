"""CPU-only raw-BF16 layout for one bounded M24/N50016/K6144 head."""

from array import array
from collections.abc import Callable

M, N, K = 24, 50016, 6144
BN, BK, SPLITS, STAGES = 256, 64, 2, 2
NT, KT = (N + BN - 1) // BN, K // BK
PADDED_N = NT * BN
SHAPE = (NT, KT, BN, BK)
FORMAT = "n06-bf16-nt256-kt64-v1"
CONFIG = {
    "block_m": 32,
    "block_n": BN,
    "block_k": BK,
    "stages": STAGES,
    "split_k": SPLITS,
    "m_waves": 1,
    "n_waves": 4,
    "k_waves": 1,
    "group_m": 0,
    "use_half_tile_interleaved": False,
}
KERNEL = "hgemm_bf16_t32x256x64x2_ksd_w1x4x1_gm0_bias0_lnt_pft"
PREFIX = "mm3_n06_bpack_"
CSV_KERNEL = (
    "flydsl_hgemm_abf16_wbf16_bf16_t32x256x64x2_ks2_w1x4x1_bias0_ktail0_gm0_pft_gfx950"
)


def index(value: int, size: int) -> None:
    """Reject negative, wrapped, noninteger or out-of-range coordinates."""
    if type(value) is not int or not 0 <= value < size:
        raise ValueError(f"Index must be an integer in [0, {size})")


def packed_offset(n: int, k: int) -> int:
    """Map a physical coordinate, including N padding, to a uint16 word offset."""
    index(n, PADDED_N)
    index(k, K)
    return ((n // BN * KT + k // BK) * BN + n % BN) * BK + k % BK


def inverse_offset(offset: int) -> tuple[int, int]:
    """Invert the physical layout; this does not apply native row-zero clamping."""
    index(offset, PADDED_N * K)
    outer, ki = divmod(offset, BK)
    outer, ni = divmod(outer, BN)
    nt, kt = divmod(outer, KT)
    return nt * BN + ni, kt * BK + ki


def source_coordinate(n: int, k: int) -> tuple[int, int]:
    """Match the native B loader's invalid-N alias to row zero before addressing."""
    index(n, PADDED_N)
    index(k, K)
    return (n if n < N else 0), k


def pack_tile(read_word: Callable[[int, int], int], nt: int, kt: int) -> array:
    """Pack one bounded CPU reference tile with no floating-point interpretation."""
    index(nt, NT)
    index(kt, KT)
    result = array("H")
    if result.itemsize != 2:
        raise RuntimeError("uint16 words required")
    for ni in range(BN):
        for ki in range(BK):
            value = read_word(*source_coordinate(nt * BN + ni, kt * BK + ki))
            index(value, 65536)
            result.append(value)
    return result


def mfma_spans(split: int) -> tuple[tuple[int, int], ...]:
    """Two K3072 subsets, each 48 increasing K64 tiles containing two K32 MFMAs."""
    index(split, SPLITS)
    start = split * (K // SPLITS)
    return tuple(
        (start + tile * BK + half * 32, start + tile * BK + (half + 1) * 32)
        for tile in range(K // SPLITS // BK)
        for half in range(2)
    )


def memory_cost() -> dict[str, int]:
    """Both original fallback and one additional packed allocation remain resident."""
    original, packed = N * K * 2, PADDED_N * K * 2
    return {
        "original_bytes": original,
        "packed_bytes": packed,
        "extra_resident_bytes": packed,
        "combined_bytes": original + packed,
        "padding_bytes": packed - original,
    }
