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

def benchmark_index(cursor, index_name, index_type, query_str, baseline_ids, baseline_time, db_size, table_size):
    """Run benchmark for a specific index."""
    # Warm-up query
    print("Running warm-up query...")
    cursor.execute(f"""
        SELECT id
        FROM vectors
        ORDER BY embedding <=> %s::vector
        LIMIT {K_NEIGHBORS}
    """, (query_str,))
    cursor.fetchall()

    # Actual benchmark query
    print(f"Querying for {K_NEIGHBORS} nearest neighbors...")
    start_time = time.time()

    if index_type == "DiskANN":
        print("Setting optimized DiskANN query parameters...")
        cursor.execute("BEGIN;")
        cursor.execute("SET LOCAL diskann.query_rescore = 1000;")
        cursor.execute("SET LOCAL diskann.l_value_is = 500;")

    cursor.execute(f"""
        SELECT id, embedding <=> %s::vector as distance
        FROM vectors
        ORDER BY embedding <=> %s::vector
        LIMIT {K_NEIGHBORS}
    """, (query_str, query_str))

    if index_type == "DiskANN":
        cursor.execute("COMMIT;")

    results = cursor.fetchall()
    query_latency = time.time() - start_time

    retrieved_ids = np.array([row[0] for row in results])
    distances = np.array([row[1] for row in results])

    # Compute precision
    precision = compute_precision(retrieved_ids, baseline_ids)

    # Get index size
    cursor.execute(f"""
        SELECT pg_size_pretty(pg_total_relation_size('{index_name}'))
    """)
    index_size = cursor.fetchone()[0]

    print("\n" + "="*60)
    print(f"PGVECTORSCALE ({index_type}) BENCHMARK RESULTS")
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

    query_str = '[' + ','.join(map(str, query)) + ']'

    # Get database stats (shared)
    cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
    db_size = cursor.fetchone()[0]

    cursor.execute("""
        SELECT pg_size_pretty(pg_total_relation_size('vectors'))
    """)
    table_size = cursor.fetchone()[0]

    # Test 1: DiskANN index
    print("\n" + "="*60)
    print("TESTING DISKANN INDEX")
    print("="*60)

    # Drop and recreate DiskANN index
    print("Dropping DiskANN index if it exists...")
    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_diskann")
    conn.commit()

    print("Creating DiskANN index...")
    index_start = time.time()
    cursor.execute("""
        CREATE INDEX idx_vectors_embedding_diskann
        ON vectors USING diskann (embedding vector_cosine_ops)
    """)
    conn.commit()
    index_time = time.time() - index_start
    print(f"DiskANN index created in {index_time:.2f} seconds")

    benchmark_index(cursor, 'idx_vectors_embedding_diskann', 'DiskANN',
                   query_str, baseline_ids, baseline_time, db_size, table_size)

    print("Cleaning up DiskANN index...")
    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_diskann")
    conn.commit()

    # Test 2: HNSW index
    print("\n" + "="*60)
    print("TESTING HNSW INDEX")
    print("="*60)

    # Drop and create HNSW index
    print("Dropping HNSW index if it exists...")
    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_hnsw")
    conn.commit()

    print("Creating HNSW index...")
    index_start = time.time()
    cursor.execute("""
        CREATE INDEX idx_vectors_embedding_hnsw
        ON vectors USING hnsw (embedding vector_cosine_ops)
    """)
    conn.commit()
    index_time = time.time() - index_start
    print(f"HNSW index created in {index_time:.2f} seconds")

    benchmark_index(cursor, 'idx_vectors_embedding_hnsw', 'HNSW',
                   query_str, baseline_ids, baseline_time, db_size, table_size)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
