#!/usr/bin/env python3
"""Generate query vector for ANN benchmark."""

import numpy as np

# Configuration
DIMENSIONS = 512

def main():
    print("Generating query vector...")

    # Set seed for reproducibility
    np.random.seed(999)  # Different seed from data generation

    # Generate and normalize query vector
    query = np.random.randn(DIMENSIONS).astype(np.float32)
    query = query / np.linalg.norm(query)  # Normalize

    print("Saving query.npy...")
    np.save('query.npy', query)

    print("\nQuery generation complete!")
    print(f"  - query.npy: 1 query vector of {DIMENSIONS} dimensions")

if __name__ == "__main__":
    main()
