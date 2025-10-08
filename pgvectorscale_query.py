#!/usr/bin/env python3
"""Query pgvectorscale and measure performance."""

import numpy as np
import psycopg2
import time

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

K_NEIGHBORS = 100

def compute_precision(retrieved_ids, baseline_ids):
    """Compute retrieval precision (recall@K)."""
    baseline_set = set(baseline_ids)
    retrieved_set = set(retrieved_ids)
    intersection = baseline_set & retrieved_set
    precision = len(intersection) / len(baseline_ids)
    return precision

def main():
    print("Loading query vector and baseline...")
    query = np.load('query.npy')
    baseline_ids = np.load('baseline_ids.npy')
    vectors = np.load('vectors.npy')

    # Compute baseline (brute force) query time
    print("\nComputing baseline (brute force) query time...")
    baseline_start = time.time()
    similarities = np.dot(vectors, query)
    top_k_indices = np.argsort(similarities)[::-1][:K_NEIGHBORS]
    baseline_time = time.time() - baseline_start
    print(f"Baseline query time: {baseline_time*1000:.2f} ms")

    print("\nConnecting to pgvectorscale database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Check if index exists, create if missing
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE indexname = 'idx_vectors_embedding_diskann'
        )
    """)
    if not cursor.fetchone()[0]:
        print("DiskANN index not found. Creating...")
        index_start = time.time()
        cursor.execute("""
            CREATE INDEX idx_vectors_embedding_diskann
            ON vectors USING diskann (embedding vector_cosine_ops)
        """)
        conn.commit()
        index_time = time.time() - index_start
        print(f"DiskANN index created in {index_time:.2f} seconds\n")

    # Warm-up query
    print("Running warm-up query...")
    query_str = '[' + ','.join(map(str, query)) + ']'
    cursor.execute(f"""
        SELECT id
        FROM vectors
        ORDER BY embedding <=> %s::vector
        LIMIT {K_NEIGHBORS}
    """, (query_str,))
    cursor.fetchall()

    # Actual benchmark query
    print(f"\nQuerying for {K_NEIGHBORS} nearest neighbors...")
    start_time = time.time()

    cursor.execute(f"""
        SELECT id, embedding <=> %s::vector as distance
        FROM vectors
        ORDER BY embedding <=> %s::vector
        LIMIT {K_NEIGHBORS}
    """, (query_str, query_str))

    results = cursor.fetchall()
    query_latency = time.time() - start_time

    retrieved_ids = np.array([row[0] for row in results])
    distances = np.array([row[1] for row in results])

    # Compute precision
    precision = compute_precision(retrieved_ids, baseline_ids)

    # Get database stats
    cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
    db_size = cursor.fetchone()[0]

    cursor.execute("""
        SELECT pg_size_pretty(pg_total_relation_size('vectors'))
    """)
    table_size = cursor.fetchone()[0]

    cursor.execute("""
        SELECT pg_size_pretty(pg_total_relation_size('idx_vectors_embedding_diskann'))
    """)
    index_size = cursor.fetchone()[0]

    print("\n" + "="*60)
    print("PGVECTORSCALE (DiskANN) BENCHMARK RESULTS")
    print("="*60)
    print(f"Query latency:        {query_latency*1000:.2f} ms")
    print(f"Retrieval precision:  {precision*100:.2f}% ({int(precision*K_NEIGHBORS)}/{K_NEIGHBORS} matches)")
    print(f"Database size:        {db_size}")
    print(f"Table size:           {table_size}")
    print(f"Index size:           {index_size}")
    print("="*60)

    print(f"\nTop 10 retrieved IDs: {retrieved_ids[:10].tolist()}")
    print(f"Top 10 baseline IDs:  {baseline_ids[:10].tolist()}")

    print("\n" + "="*60)
    print("BASELINE (Brute Force) PERFORMANCE")
    print("="*60)
    print(f"Query latency:        {baseline_time*1000:.2f} ms")
    print(f"Speedup:              {baseline_time/query_latency:.2f}x faster")
    print("="*60)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
