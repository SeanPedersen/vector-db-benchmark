#!/usr/bin/env python3
"""Generate random vectors and query for ANN benchmark."""

import numpy as np

# Configuration
NUM_VECTORS = 100_000
DIMENSIONS = 512
K_NEIGHBORS = 100

def main():
    print(f"Generating {NUM_VECTORS:,} random vectors of dimension {DIMENSIONS}...")

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate 1 million random vectors (float32)
    vectors = np.random.randn(NUM_VECTORS, DIMENSIONS).astype(np.float32)

    # Normalize vectors for cosine similarity (optional but common for ANN)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Generate IDs
    ids = np.arange(NUM_VECTORS, dtype=np.int64)

    # Save vectors and IDs
    print("Saving vectors.npy...")
    np.save('vectors.npy', vectors)

    print("Saving ids.npy...")
    np.save('ids.npy', ids)

    # Generate and save query vector
    print("Generating query vector...")
    query = np.random.randn(DIMENSIONS).astype(np.float32)
    query = query / np.linalg.norm(query)  # Normalize

    print("Saving query.npy...")
    np.save('query.npy', query)

    print("\nData generation complete!")
    print(f"  - vectors.npy: {NUM_VECTORS:,} vectors of {DIMENSIONS} dimensions")
    print(f"  - ids.npy: {NUM_VECTORS:,} IDs")
    print(f"  - query.npy: 1 query vector of {DIMENSIONS} dimensions")

if __name__ == "__main__":
    main()
