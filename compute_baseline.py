#!/usr/bin/env python3
"""Compute exact baseline using numpy and postgres (brute force)."""

import numpy as np
import psycopg2
import time
from tqdm import tqdm

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
    query = np.load('query.npy')

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Get vector count
    cursor.execute("SELECT COUNT(*) FROM vectors")
    count = cursor.fetchone()[0]
    print(f"\n[Baseline] Computing brute force baseline on {count:,} vectors...")

    # Baseline 1: NumPy brute force (load all vectors from DB)
    load_start = time.time()

    # Always use batched approach for debugging
    if True:  # count > 100000:
        # Process in batches to save memory
        num_batches = 10
        batch_size = (count + num_batches - 1) // num_batches  # Ceiling division
        print(f"[Baseline] Processing {count:,} vectors in {num_batches} batches of ~{batch_size:,} vectors each")

        all_ids = []
        all_similarities = []

        compute_elapsed = 0

        for batch_idx in tqdm(range(num_batches), desc="[Baseline] Processing batches", unit="batch"):
            offset = batch_idx * batch_size
            cursor.execute(
                "SELECT id, embedding::text FROM vectors ORDER BY id LIMIT %s OFFSET %s",
                (batch_size, offset)
            )
            rows = cursor.fetchall()

            if not rows:
                break

            # Convert to numpy arrays
            batch_ids = np.array([row[0] for row in rows], dtype=np.int64)
            batch_vectors = np.array([
                [float(x) for x in row[1].strip('[]').split(',')]
                for row in rows
            ], dtype=np.float32)

            # Normalize vectors for cosine similarity (Postgres does this automatically)
            norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            batch_vectors = batch_vectors / norms

            # Compute cosine similarities for this batch
            compute_start = time.time()
            batch_similarities = np.dot(batch_vectors, query)
            compute_elapsed += time.time() - compute_start

            all_ids.append(batch_ids)
            all_similarities.append(batch_similarities)

        # Combine all batch results and find global top K
        compute_start = time.time()
        all_ids = np.concatenate(all_ids)
        all_similarities = np.concatenate(all_similarities)

        # Debug: Check if we have the expected number of vectors
        print(f"[Baseline] DEBUG: Total vectors after concatenation: {len(all_ids)}")
        print(f"[Baseline] DEBUG: Min/Max similarity: {all_similarities.min():.6f} / {all_similarities.max():.6f}")

        # Get final top K from all vectors
        final_top_k_indices = np.argsort(all_similarities)[::-1][:K_NEIGHBORS]
        top_k_ids = all_ids[final_top_k_indices]
        top_k_similarities = all_similarities[final_top_k_indices]

        # Debug: Print top 10 similarities
        print(f"[Baseline] DEBUG: Top 10 similarities: {top_k_similarities[:10]}")

        compute_elapsed += time.time() - compute_start

        load_elapsed = time.time() - load_start
        total_numpy_elapsed = load_elapsed
    else:
        # Original single-pass implementation for smaller datasets
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

        compute_start = time.time()
        # Normalize vectors for cosine similarity (Postgres does this automatically)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        # Compute cosine similarities (dot product for normalized vectors)
        similarities = np.dot(vectors, query)

        # Get top K indices (argsort in descending order)
        top_k_indices = np.argsort(similarities)[::-1][:K_NEIGHBORS]

        # Get the corresponding IDs
        top_k_ids = ids[top_k_indices]
        top_k_similarities = similarities[top_k_indices]

        compute_elapsed = time.time() - compute_start
        total_numpy_elapsed = load_elapsed + compute_elapsed

    print(f"[Baseline] NumPy brute force: {compute_elapsed*1000:.2f}ms (100% precision)")

    # Baseline 2: Postgres brute force (no index)
    # Drop all existing indexes on embedding column to ensure true brute force
    cursor.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'vectors' AND indexname LIKE '%embedding%'
    """)
    existing_indexes = cursor.fetchall()
    if existing_indexes:
        for idx in existing_indexes:
            cursor.execute(f"DROP INDEX IF EXISTS {idx[0]}")
        conn.commit()

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
    pg_similarities = np.array([row[1] for row in pg_results], dtype=np.float64)

    # Debug: Print Postgres top 10 similarities
    print(f"[Baseline] DEBUG: Postgres top 10 similarities: {pg_similarities[:10]}")

    # Compare results
    matches = np.sum(np.isin(pg_ids, top_k_ids))
    recall = matches / K_NEIGHBORS

    print(f"[Baseline] Postgres brute force: {pg_elapsed*1000:.2f}ms (100% precision)")

    if recall == 1.0:
        print(f"[Baseline] NumPy and Postgres results match perfectly ({matches}/{K_NEIGHBORS})")
    elif recall >= 0.95:
        print(f"[Baseline] NumPy and Postgres mostly match: {matches}/{K_NEIGHBORS} ({recall*100:.1f}% - likely due to floating point precision)")
    else:
        print(f"[Baseline] ERROR: Results differ significantly - {matches}/{K_NEIGHBORS} match ({recall*100:.1f}% recall)")
        print(f"[Baseline] Top 10 NumPy IDs: {top_k_ids[:10].tolist()}")
        print(f"[Baseline] Top 10 Postgres IDs: {pg_ids[:10].tolist()}")
        cursor.close()
        conn.close()
        raise ValueError(f"Baseline mismatch: NumPy and Postgres results differ ({recall*100:.1f}% recall). Cannot proceed with benchmark.")

    cursor.close()
    conn.close()

    return {
        'numpy_time': total_numpy_elapsed,
        'postgres_time': pg_elapsed,
        'baseline_ids': top_k_ids,
        'query': query
    }

def main():
    compute_baseline()

if __name__ == "__main__":
    main()
