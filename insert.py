#!/usr/bin/env python3
"""Generate and insert vectors into database (shared for all indices)."""

import numpy as np
import psycopg2
import time
import argparse
from psycopg2.extras import execute_values
from tqdm import tqdm

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

BATCH_SIZE = 10_000
DIMENSIONS = 512

def generate_vectors(num_vectors):
    """Generate normalized random vectors."""
    print(f"Generating {num_vectors:,} random vectors of dimension {DIMENSIONS}...")

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate random vectors (float32)
    vectors = np.random.randn(num_vectors, DIMENSIONS).astype(np.float32)

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    return vectors

def main():
    parser = argparse.ArgumentParser(description="Generate and insert vectors into database")
    parser.add_argument('--num-vectors', type=int, default=100000,
                        help='Number of vectors to generate (default: 100000)')
    args = parser.parse_args()

    num_vectors = args.num_vectors

    print("\nConnecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Check if table already has the correct number of vectors
    cursor.execute("SELECT COUNT(*) FROM vectors")
    existing_count = cursor.fetchone()[0]

    if existing_count == num_vectors:
        print(f"Table already contains {existing_count:,} vectors (matches requested size). Skipping insertion.")
        cursor.close()
        conn.close()
        return

    # Clear existing data if count doesn't match
    if existing_count > 0:
        print(f"Clearing existing data ({existing_count:,} vectors)...")
        cursor.execute("TRUNCATE TABLE vectors")
        conn.commit()

    # Generate vectors
    vectors = generate_vectors(num_vectors)
    ids = np.arange(num_vectors, dtype=np.int64)

    print(f"\nInserting {num_vectors:,} vectors in batches of {BATCH_SIZE:,}...")
    start_time = time.time()

    # Prepare data for batch insertion
    for i in tqdm(range(0, num_vectors, BATCH_SIZE), desc="Inserting", unit="batch"):
        batch_end = min(i + BATCH_SIZE, num_vectors)
        batch_data = [
            (int(ids[j]), vectors[j].tolist())
            for j in range(i, batch_end)
        ]

        execute_values(
            cursor,
            "INSERT INTO vectors (id, embedding) VALUES %s",
            batch_data,
            template="(%s, %s::vector)"
        )

        if (i + BATCH_SIZE) % 100_000 == 0:
            conn.commit()

    conn.commit()
    elapsed = time.time() - start_time

    # Get count
    cursor.execute("SELECT COUNT(*) FROM vectors")
    count = cursor.fetchone()[0]

    print(f"\nInsertion complete!")
    print(f"  Total vectors: {count:,}")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Throughput: {count / elapsed:.0f} vectors/second")

    cursor.close()
    conn.close()
    print("Connection closed.")

if __name__ == "__main__":
    main()
