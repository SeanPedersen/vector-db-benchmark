"""Parameter calculation utilities for index configuration."""


def _clamp(v, lo, hi):
    """Clamp value v to range [lo, hi]."""
    return max(lo, min(hi, v))


def recommend_ivf_params(num_vectors: int, k: int):
    """Recommend IVF parameters based on dataset size and k.

    Heuristics:
    - lists in [50, 500], target ~ N/200 (e.g., 50K -> 250)
    - For small k (<= 200): probes ~ lists/16 clamped to [8, 40]
    - For larger k (> 200): probes ~ lists/12 clamped to [10, 50]
    - binary probes ~ 3x probes (small k: [24, 120], large k: [30, 200])

    Args:
        num_vectors: Number of vectors in the dataset
        k: Number of nearest neighbors to retrieve

    Returns:
        Tuple of (lists, probes, probes_binary)
    """
    lists = _clamp(num_vectors // 200, 50, 500)
    if k <= 200:
        probes = _clamp(round(lists / 16), 8, 40)
        probes_bin = _clamp(probes * 3, 24, 120)
    else:
        probes = _clamp(round(lists / 12), 10, 50)
        probes_bin = _clamp(probes * 3, 30, 200)
    return int(lists), int(probes), int(probes_bin)
