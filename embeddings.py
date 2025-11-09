"""Embedding generation and encoding utilities."""

import numpy as np


def generate_embeddings(num_vectors: int, full_dim: int = 1024):
    """Generate random normalized embeddings.

    Args:
        num_vectors: Number of vectors to generate
        full_dim: Full dimensionality of vectors (default: 1024)

    Returns:
        Normalized numpy array of shape (num_vectors, full_dim)
    """
    np.random.seed(123)
    data = np.random.randn(num_vectors, full_dim).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    return data


def compute_dimension_means(embeddings: np.ndarray):
    """Compute mean for each dimension across all vectors."""
    return np.mean(embeddings, axis=0)


def binarize_with_means(embeddings: np.ndarray, means: np.ndarray):
    """Binarize vectors using mean-based thresholding.

    For each dimension, use its corpus-wide mean as threshold:
    - value >= mean -> 1
    - value < mean -> 0

    This ensures ~50/50 bit distribution per dimension for better
    information preservation and more discriminative binary codes.
    """
    return (embeddings >= means).astype(np.uint8)


def compute_percentile_thresholds(embeddings: np.ndarray, num_buckets: int = 8):
    """Compute percentile-based thresholds for quasi-uint8 quantization.

    For num_buckets=8, computes 7 threshold values per dimension at percentiles:
    [12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5]

    Returns: np.ndarray of shape (num_dimensions, num_buckets-1)
    """
    num_dims = embeddings.shape[1]
    num_thresholds = num_buckets - 1

    # Compute percentiles for each bucket boundary
    percentiles = np.linspace(
        100.0 / num_buckets, 100.0 * (num_buckets - 1) / num_buckets, num_thresholds
    )

    # Compute thresholds for each dimension
    thresholds = np.zeros((num_dims, num_thresholds), dtype=np.float32)
    for dim in range(num_dims):
        thresholds[dim] = np.percentile(embeddings[:, dim], percentiles)

    return thresholds


def encode_thermometer(
    embeddings: np.ndarray, thresholds: np.ndarray, num_bits_per_dim: int = None
):
    """Encode vectors using thermometer/unary code.

    For each dimension, compares value against all thresholds:
    - Creates num_bits_per_dim bits per dimension (default: num_thresholds + 1)
    - bit[i] = 1 if value >= threshold[i], else 0
    - Example (8 buckets): value between threshold[2] and threshold[3] -> 11100000
    - Example (4 buckets): value between threshold[1] and threshold[2] -> 1100

    Args:
        embeddings: (N, D) array of float vectors
        thresholds: (D, num_thresholds) array of threshold values per dimension
        num_bits_per_dim: Number of bits per dimension (default: num_thresholds + 1)

    Returns: (N, D * num_bits_per_dim) binary array where each dimension is expanded
    """
    num_vectors, num_dims = embeddings.shape
    num_thresholds = thresholds.shape[1]

    # Default: use num_thresholds + 1 bits (e.g., 7 thresholds = 8 bits, 3 thresholds = 4 bits)
    if num_bits_per_dim is None:
        num_bits_per_dim = num_thresholds + 1

    # Initialize output: N vectors x (D dimensions * num_bits_per_dim bits per dimension)
    encoded = np.zeros((num_vectors, num_dims * num_bits_per_dim), dtype=np.uint8)

    for dim in range(num_dims):
        # For this dimension, compare all vectors against all thresholds
        for bit_idx in range(num_thresholds):
            # Set bit to 1 if value >= threshold
            bit_position = dim * num_bits_per_dim + bit_idx
            encoded[:, bit_position] = (
                embeddings[:, dim] >= thresholds[dim, bit_idx]
            ).astype(np.uint8)

    # Note: bits num_thresholds to num_bits_per_dim-1 remain 0 (padding)
    # For 7 thresholds (8 buckets), bits 0-6 are used, bit 7 is always 0
    # For 3 thresholds (4 buckets), bits 0-2 are used, bit 3 is always 0

    return encoded


def encode_one_hot(
    embeddings: np.ndarray, thresholds: np.ndarray, num_bits_per_dim: int = None
):
    """Encode vectors using one-hot/categorical code.

    For each dimension, finds which bucket the value falls into and sets only that bit:
    - Creates num_bits_per_dim bits per dimension (default: num_thresholds + 1)
    - Only one bit is set to 1 per dimension (the bucket the value falls into)
    - Example (8 buckets): value between threshold[2] and threshold[3] -> 00010000
    - Example (4 buckets): value between threshold[1] and threshold[2] -> 0010

    Args:
        embeddings: (N, D) array of float vectors
        thresholds: (D, num_thresholds) array of threshold values per dimension
        num_bits_per_dim: Number of bits per dimension (default: num_thresholds + 1)

    Returns: (N, D * num_bits_per_dim) binary array where each dimension is expanded
    """
    num_vectors, num_dims = embeddings.shape
    num_thresholds = thresholds.shape[1]

    # Default: use num_thresholds + 1 bits (e.g., 7 thresholds = 8 bits, 3 thresholds = 4 bits)
    if num_bits_per_dim is None:
        num_bits_per_dim = num_thresholds + 1

    # Initialize output: N vectors x (D dimensions * num_bits_per_dim bits per dimension)
    encoded = np.zeros((num_vectors, num_dims * num_bits_per_dim), dtype=np.uint8)

    for dim in range(num_dims):
        # For each vector, find which bucket it falls into
        for vec_idx in range(num_vectors):
            value = embeddings[vec_idx, dim]

            # Find the bucket: count how many thresholds the value exceeds
            bucket_idx = 0
            for threshold_idx in range(num_thresholds):
                if value >= thresholds[dim, threshold_idx]:
                    bucket_idx = threshold_idx + 1
                else:
                    break

            # Set only the bit for this bucket
            bit_position = dim * num_bits_per_dim + bucket_idx
            encoded[vec_idx, bit_position] = 1

    return encoded


def numpy_binary_to_postgres_bit_string(binary_array: np.ndarray):
    """Convert numpy binary array to PostgreSQL bit string payload.

    Returns the raw bit digits without the B'..' wrapper so it can be safely
    bound as a parameter and cast using %s::bit(n) in SQL.
    """
    bit_string = "".join(binary_array.astype(str))
    return bit_string
