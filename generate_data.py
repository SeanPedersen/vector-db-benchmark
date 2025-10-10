#!/usr/bin/env python3
"""Generate query vector for ANN benchmark."""

import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate query vector for ANN benchmark")
    parser.add_argument('--dimensions', type=int, default=512,
                        help='Vector dimensions (default: 512)')
    args = parser.parse_args()

    dimensions = args.dimensions

    print(f"Generating query vector with {dimensions} dimensions...")

    # Set seed for reproducibility
    np.random.seed(999)  # Different seed from data generation

    # Generate and normalize query vector
    query = np.random.randn(dimensions).astype(np.float32)
    query = query / np.linalg.norm(query)  # Normalize

    print("Saving query.npy...")
    np.save('query.npy', query)

    print("\nQuery generation complete!")
    print(f"  - query.npy: 1 query vector of {dimensions} dimensions")

if __name__ == "__main__":
    main()
