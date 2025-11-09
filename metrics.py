"""Metrics and evaluation utilities."""

import numpy as np
from config import K


def build_baseline(full_embeddings: np.ndarray, query: np.ndarray):
    """Build baseline using brute force similarity search.

    Args:
        full_embeddings: All embeddings to search through
        query: Query vector

    Returns:
        Indices of top K most similar vectors
    """
    sims = full_embeddings @ query
    top_idx = np.argsort(sims)[::-1][:K]
    return top_idx


def compute_recall(retrieved_ids: list, baseline_ids: np.ndarray):
    """Compute recall@K.

    Args:
        retrieved_ids: List of retrieved vector IDs
        baseline_ids: Baseline (ground truth) vector IDs

    Returns:
        Recall as a fraction (0.0 to 1.0)
    """
    baseline_set = set(baseline_ids)
    retrieved_set = set(retrieved_ids)
    intersection = baseline_set & retrieved_set
    return len(intersection) / len(baseline_ids)


def get_vector_storage_mb(num_vectors: int, dimensions: int, precision: str):
    """Calculate the storage size for vector data.

    Args:
        num_vectors: Number of vectors
        dimensions: Number of dimensions
        precision: "float32" or "float16"

    Returns:
        Estimated storage size in MB
    """
    bytes_per_value = (
        4 if precision == "float32" else 2
    )  # float32=4 bytes, float16=2 bytes
    total_bytes = num_vectors * dimensions * bytes_per_value
    # Add ~10% overhead for PostgreSQL storage (TOAST, alignment, etc.)
    return (total_bytes * 1.1) / (1024.0 * 1024.0)


def get_binary_storage_mb(num_vectors: int, dimensions: int, bits_per_dim: int):
    """Calculate the storage size for binary vector data.

    Args:
        num_vectors: Number of vectors
        dimensions: Number of dimensions
        bits_per_dim: Number of bits per dimension (e.g., 8 for uint8, 4 for uint4, 1 for binary)

    Returns:
        Estimated storage size in MB
    """
    total_bits = num_vectors * dimensions * bits_per_dim
    total_bytes = total_bits / 8.0  # Convert bits to bytes
    # Add ~10% overhead for PostgreSQL storage (TOAST, alignment, etc.)
    return (total_bytes * 1.1) / (1024.0 * 1024.0)
