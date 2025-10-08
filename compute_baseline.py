#!/usr/bin/env python3
"""Compute exact baseline using numpy."""

import numpy as np
import time

K_NEIGHBORS = 100

def main():
    print("Loading data...")
    vectors = np.load('vectors.npy')
    ids = np.load('ids.npy')
    query = np.load('query.npy')

    print(f"Loaded {len(vectors):,} vectors")
    print(f"\nComputing exact {K_NEIGHBORS} nearest neighbors using cosine similarity...")

    start_time = time.time()

    # Compute cosine similarities (dot product for normalized vectors)
    similarities = np.dot(vectors, query)

    # Get top K indices (argsort in descending order)
    top_k_indices = np.argsort(similarities)[::-1][:K_NEIGHBORS]

    # Get the corresponding IDs
    top_k_ids = ids[top_k_indices]
    top_k_similarities = similarities[top_k_indices]

    elapsed = time.time() - start_time

    print(f"Computation complete in {elapsed:.3f} seconds")
    print(f"\nTop {K_NEIGHBORS} nearest neighbors (IDs):")
    print(top_k_ids)

    # Save baseline results (for query scripts to compare)
    print("\nSaving baseline results...")
    np.save('baseline_ids.npy', top_k_ids)

    print(f"\nBaseline saved to baseline_ids.npy")
    print(f"Query latency: {elapsed:.3f} seconds ({elapsed*1000:.2f} ms)")

if __name__ == "__main__":
    main()
