#!/usr/bin/env python3
"""Generate and insert vectors into database (shared for all indices)."""

import numpy as np
import psycopg2
import time
import argparse
from psycopg2.extras import execute_values
from tqdm import tqdm

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
}

BATCH_SIZE = 10_000


def generate_vectors(num_vectors, dimensions):
    """Generate normalized random vectors."""
    print(f"Generating {num_vectors:,} random vectors of dimension {dimensions}...")

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate random vectors (float32)
    vectors = np.random.randn(num_vectors, dimensions).astype(np.float32)

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    return vectors


def main():
    parser = argparse.ArgumentParser(
        description="Generate and insert vectors into database"
    )
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=100000,
        help="Number of vectors to generate (default: 100000)",
    )
    parser.add_argument(
        "--dimensions", type=int, default=512, help="Vector dimensions (default: 512)"
    )
    parser.add_argument(
        "--vectors-file",
        type=str,
        default=None,
        help="Path to .npy file containing pre-generated vectors (optional)",
    )
    args = parser.parse_args()

    num_vectors = args.num_vectors
    dimensions = args.dimensions

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Load or generate vectors first to determine dimensions
    if args.vectors_file:
        print(f"[Insert] Loading vectors from {args.vectors_file}...")
        vectors = np.load(args.vectors_file)

        # Validate shape
        if vectors.ndim != 2:
            print(
                f"Error: Expected 2D array of shape (N, D), got {vectors.ndim}D array with shape {vectors.shape}"
            )
            cursor.close()
            conn.close()
            return

        # Use actual dimensions and count from file
        num_vectors = vectors.shape[0]
        dimensions = vectors.shape[1]
        print(f"[Insert] Loaded {num_vectors:,} vectors with dimension {dimensions}")
    else:
        vectors = generate_vectors(num_vectors, dimensions)

    # Check if table exists and has the correct count
    table_exists = False
    correct_count = False
    try:
        cursor.execute("SELECT COUNT(*) FROM vectors")
        existing_count = cursor.fetchone()[0]
        table_exists = True
        if existing_count == num_vectors:
            correct_count = True
            print(f"[Insert] Table already contains {existing_count:,} vectors, skipping insertion")
    except psycopg2.Error:
        table_exists = False

    if correct_count:
        cursor.close()
        conn.close()
        return

    # Create or recreate table with correct dimensions
    print(f"[Insert] Creating table for {dimensions}-dimensional vectors...")
    cursor.execute("DROP TABLE IF EXISTS vectors")
    cursor.execute(
        f"CREATE TABLE vectors (id BIGINT PRIMARY KEY, embedding vector({dimensions}))"
    )
    conn.commit()

    ids = np.arange(num_vectors, dtype=np.int64)

    print(f"[Insert] Inserting {num_vectors:,} vectors...")
    start_time = time.time()

    # Prepare data for batch insertion
    for i in tqdm(range(0, num_vectors, BATCH_SIZE), desc="Inserting", unit="batch"):
        batch_end = min(i + BATCH_SIZE, num_vectors)
        batch_data = [(int(ids[j]), vectors[j].tolist()) for j in range(i, batch_end)]

        execute_values(
            cursor,
            "INSERT INTO vectors (id, embedding) VALUES %s",
            batch_data,
            template="(%s, %s::vector)",
        )

        if (i + BATCH_SIZE) % 100_000 == 0:
            conn.commit()

    conn.commit()
    elapsed = time.time() - start_time

    # Get count
    cursor.execute("SELECT COUNT(*) FROM vectors")
    count = cursor.fetchone()[0]

    print(
        f"[Insert] Complete! {count:,} vectors in {elapsed:.2f}s ({count / elapsed:.0f} vectors/s)"
    )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
