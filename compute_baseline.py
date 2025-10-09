#!/usr/bin/env python3
"""Compute exact baseline using numpy and postgres (brute force)."""

import numpy as np
import psycopg2
import time

K_NEIGHBORS = 100

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

def compute_baseline():
    """Compute baseline performance and return timing results.

    Returns:
        tuple: (numpy_baseline_time, postgres_baseline_time) in seconds
    """
    print("Loading query vector...")
    query = np.load('query.npy')

    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Get vector count
    cursor.execute("SELECT COUNT(*) FROM vectors")
    count = cursor.fetchone()[0]
    print(f"Database contains {count:,} vectors")

    # Baseline 1: NumPy brute force (load all vectors from DB)
    print(f"\n{'='*60}")
    print("BASELINE 1: NumPy Brute Force")
    print(f"{'='*60}")
    print("Loading all vectors from database into memory...")

    load_start = time.time()
    cursor.execute("SELECT id, embedding::text FROM vectors ORDER BY id")
    rows = cursor.fetchall()
    # Convert to numpy arrays
    ids = np.array([row[0] for row in rows], dtype=np.int64)
    # Parse vector strings like '[1.0,2.0,3.0]' to numpy arrays
    vectors = np.array([
        [float(x) for x in row[1].strip('[]').split(',')]
        for row in rows
    ], dtype=np.float32)
    load_elapsed = time.time() - load_start
    print(f"Loaded {len(rows):,} vectors in {load_elapsed:.3f} seconds")

    print(f"\nComputing exact {K_NEIGHBORS} nearest neighbors using cosine similarity...")

    compute_start = time.time()
    # Compute cosine similarities (dot product for normalized vectors)
    similarities = np.dot(vectors, query)

    # Get top K indices (argsort in descending order)
    top_k_indices = np.argsort(similarities)[::-1][:K_NEIGHBORS]

    # Get the corresponding IDs
    top_k_ids = ids[top_k_indices]
    top_k_similarities = similarities[top_k_indices]

    compute_elapsed = time.time() - compute_start
    total_numpy_elapsed = load_elapsed + compute_elapsed

    print(f"Computation complete!")
    print(f"  Load time: {load_elapsed:.3f} seconds ({load_elapsed*1000:.2f} ms)")
    print(f"  Compute time: {compute_elapsed:.3f} seconds ({compute_elapsed*1000:.2f} ms)")
    print(f"  Total time: {total_numpy_elapsed:.3f} seconds ({total_numpy_elapsed*1000:.2f} ms)")
    print(f"\nTop {K_NEIGHBORS} nearest neighbors (IDs): {top_k_ids[:10]}... (showing first 10)")

    # Save baseline results (for query scripts to compare)
    print("\nSaving baseline results to baseline_ids.npy...")
    np.save('baseline_ids.npy', top_k_ids)

    # Baseline 2: Postgres brute force (no index)
    print(f"\n{'='*60}")
    print("BASELINE 2: Postgres Brute Force (No Index)")
    print(f"{'='*60}")
    print(f"Running postgres query without index for top {K_NEIGHBORS} nearest neighbors...")

    # Make it hot
    query_list = query.tolist()
    cursor.execute("""
        SELECT id, 1 - (embedding <=> %s::vector) AS similarity
        FROM vectors
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_list, query_list, K_NEIGHBORS))

    pg_results = cursor.fetchall()

    pg_start = time.time()

    # Query using cosine distance operator (without index)
    query_list = query.tolist()
    cursor.execute("""
        SELECT id, 1 - (embedding <=> %s::vector) AS similarity
        FROM vectors
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_list, query_list, K_NEIGHBORS))

    pg_results = cursor.fetchall()
    pg_elapsed = time.time() - pg_start

    pg_ids = np.array([row[0] for row in pg_results], dtype=np.int64)

    print(f"Postgres query complete!")
    print(f"  Query time: {pg_elapsed:.3f} seconds ({pg_elapsed*1000:.2f} ms)")
    print(f"\nTop {K_NEIGHBORS} nearest neighbors (IDs): {pg_ids[:10]}... (showing first 10)")

    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    # Check if results match
    matches = np.sum(np.isin(pg_ids, top_k_ids))
    recall = matches / K_NEIGHBORS

    print(f"NumPy baseline time: {compute_elapsed:.3f} seconds ({compute_elapsed*1000:.2f} ms)")
    print(f"Postgres baseline time: {pg_elapsed:.3f} seconds ({pg_elapsed*1000:.2f} ms)")
    print(f"Results match: {matches}/{K_NEIGHBORS} ({recall*100:.1f}% recall)")

    if recall < 1.0:
        print("\nWARNING: Results don't match perfectly. This may be due to:")
        print("  - Floating point precision differences")
        print("  - Ties in similarity scores handled differently")

    cursor.close()
    conn.close()
    print("\nBaseline computation complete!")

    return total_numpy_elapsed, pg_elapsed

def main():
    compute_baseline()

if __name__ == "__main__":
    main()
