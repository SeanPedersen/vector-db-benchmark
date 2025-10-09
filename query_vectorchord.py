#!/usr/bin/env python3
"""Query vectorchord indices and measure performance."""

import numpy as np
import psycopg2
import time
import argparse

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
    cursor.execute(f"""
        SELECT id
        FROM vectors
        ORDER BY embedding <=> %s::vector
        LIMIT {K_NEIGHBORS}
    """, (query_str,))
    cursor.fetchall()

    # Actual benchmark query
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

    # Get index size
    cursor.execute(f"""
        SELECT pg_size_pretty(pg_total_relation_size('{index_name}'))
    """)
    index_size = cursor.fetchone()[0]

    # Get index size in MB
    cursor.execute(f"""
        SELECT pg_total_relation_size('{index_name}') / (1024.0 * 1024.0)
    """)
    index_size_mb = cursor.fetchone()[0]

    print(f"✓ VectorChord ({index_type}): {query_latency*1000:.2f}ms, {precision*100:.0f}% precision")

    return {
        'query_latency': query_latency,
        'precision': precision,
        'index_size': index_size,
        'index_size_mb': index_size_mb
    }

def run_benchmark(query, baseline_ids, baseline_time):
    """Run vectorchord benchmark and return results."""
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

    # Test 1: Default index
    print("\n[VectorChord] Building vchordrq index...")
    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_vchordrq_default")
    conn.commit()

    index_start = time.time()
    cursor.execute("""
        CREATE INDEX idx_vectors_embedding_vchordrq_default
        ON vectors USING vchordrq (embedding vector_cosine_ops)
    """)
    conn.commit()
    index_time = time.time() - index_start
    print(f"[VectorChord] Index built in {index_time:.2f}s")

    result = benchmark_index(cursor, 'idx_vectors_embedding_vchordrq_default', 'vchordrq',
                   query_str, baseline_ids, baseline_time, db_size, table_size)

    cursor.close()
    conn.close()

    return {
        'method': 'vchordrq',
        'latency': result['query_latency'],
        'precision': result['precision'],
        'build_time': index_time,
        'size_mb': result['index_size_mb']
    }

def main():
    parser = argparse.ArgumentParser(description="Query vectorchord indices")
    parser.add_argument('--baseline-time', type=float, default=None,
                        help='Baseline query time in seconds (from compute_baseline.py)')
    args = parser.parse_args()

    query = np.load('query.npy')
    baseline_ids = np.load('baseline_ids.npy')

    result = run_benchmark(query, baseline_ids, args.baseline_time)

    # Output results for main script to capture (stderr so it doesn't appear in stdout)
    import sys
    print(f"RESULT:{result['method']}:{result['latency']}:{result['precision']}:{result['build_time']}:{result['size_mb']}", file=sys.stderr)

if __name__ == "__main__":
    main()
