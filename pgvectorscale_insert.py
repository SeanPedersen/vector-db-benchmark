#!/usr/bin/env python3
"""Insert vectors into pgvectorscale database."""

import numpy as np
import psycopg2
import time
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

def main():
    print("Loading vectors and IDs...")
    vectors = np.load('vectors.npy')
    ids = np.load('ids.npy')

    print(f"Loaded {len(vectors):,} vectors")
    print("\nConnecting to pgvectorscale database...")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Clear existing data
    print("Clearing existing data...")
    cursor.execute("TRUNCATE TABLE vectors")
    conn.commit()

    print(f"\nInserting {len(vectors):,} vectors in batches of {BATCH_SIZE:,}...")
    start_time = time.time()

    # Prepare data for batch insertion
    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Inserting", unit="batch"):
        batch_end = min(i + BATCH_SIZE, len(vectors))
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

    # Create index after insertion
    print("\nCreating DiskANN index...")
    index_start = time.time()
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_vectors_embedding_diskann
        ON vectors USING diskann (embedding vector_cosine_ops)
    """)
    conn.commit()
    index_time = time.time() - index_start
    print(f"Index created in {index_time:.2f} seconds")

    cursor.close()
    conn.close()
    print("Connection closed.")

if __name__ == "__main__":
    main()
