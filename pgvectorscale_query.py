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

def benchmark_index(cursor, index_name, index_type, query_str, baseline_ids, baseline_time, db_size, table_size, probes=None, diskann_params=None, hnsw_ef=None):
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
        # Use default query parameters (no custom settings needed)
        pass
    elif index_type == "IVFFlat":
        probes_value = probes if probes is not None else 200
        print(f"Setting optimized IVFFlat query parameters (probes={probes_value})...")
        cursor.execute(f"SET ivfflat.probes = {probes_value}")
    elif index_type == "HNSW":
        ef_value = hnsw_ef if hnsw_ef is not None else 100
        print(f"Setting optimized HNSW query parameters (ef_search={ef_value})...")
        cursor.execute(f"SET hnsw.ef_search = {ef_value}")

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
    num_vectors = len(vectors)

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

    # Test 0: HNSW index (added before IVFFlat)
    print("\n" + "="*60)
    print("TESTING HNSW INDEX")
    print("="*60)

    print("Dropping HNSW index if it exists...")
    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_hnsw")
    conn.commit()

    # Choose HNSW params based on dataset size
    if num_vectors < 1_000_042:
        m = 16
        ef_construction = 200
        ef_search = 100
    elif num_vectors < 10_000_000:
        m = 24
        ef_construction = 300
        ef_search = 200
    else:
        m = 32
        ef_construction = 400
        ef_search = 300

    print(f"Creating HNSW index with parameters: m={m}, ef_construction={ef_construction}")
    index_start = time.time()
    cursor.execute(f"""
        CREATE INDEX idx_vectors_embedding_hnsw
        ON vectors USING hnsw (embedding vector_cosine_ops)
        WITH (m = {m}, ef_construction = {ef_construction})
    """)
    conn.commit()
    index_time = time.time() - index_start
    print(f"HNSW index created in {index_time:.2f} seconds")

    benchmark_index(cursor, 'idx_vectors_embedding_hnsw', 'HNSW',
                    query_str, baseline_ids, baseline_time, db_size, table_size,
                    hnsw_ef=ef_search)

    print("Cleaning up HNSW index...")
    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_hnsw")
    conn.commit()

    # Test 1: IVFFlat index (faster to build)
    print("\n" + "="*60)
    print("TESTING IVFFLAT INDEX")
    print("="*60)

    # Drop and create IVFFlat index
    print("Dropping IVFFlat index if it exists...")
    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_ivfflat")
    conn.commit()

    # Calculate optimal lists based on dataset size
    # For up to 1M rows: rows / 1000, for over 1M: sqrt(rows)
    if num_vectors <= 1_000_000:
        lists = max(50, num_vectors // 1000)  # Minimum 50 lists
    else:
        lists = int(num_vectors ** 0.5)

    # Calculate probes: recommended value is sqrt(lists)
    probes = int(lists ** 0.5)

    print(f"Creating IVFFlat index with lists={lists} (probes will be {probes})...")
    index_start = time.time()
    cursor.execute(f"""
        CREATE INDEX idx_vectors_embedding_ivfflat
        ON vectors USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {lists})
    """)
    conn.commit()
    index_time = time.time() - index_start
    print(f"IVFFlat index created in {index_time:.2f} seconds")

    benchmark_index(cursor, 'idx_vectors_embedding_ivfflat', 'IVFFlat',
                   query_str, baseline_ids, baseline_time, db_size, table_size, probes=probes)

    print("Cleaning up IVFFlat index...")
    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_ivfflat")
    conn.commit()

    # Test 2: DiskANN index (slower to build, run last)
    print("\n" + "="*60)
    print("TESTING DISKANN INDEX")
    print("="*60)

    # Drop and recreate DiskANN index
    print("Dropping DiskANN index if it exists...")
    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_diskann")
    conn.commit()

    # Calculate optimal DiskANN parameters based on dataset size
    # Note: memory_optimized uses SBQ compression for better storage efficiency and I/O performance
    if num_vectors < 1_042:
        print("Warning: DiskANN requires at least 1,000 vectors for optimal performance")
        num_neighbors = 32
        search_list_size = 50
        max_alpha = 1.0
    elif num_vectors < 1_000_042:
        # Small to Medium (1K-1M)
        num_neighbors = 50
        search_list_size = 100
        max_alpha = 1.0
    elif num_vectors < 100_000_000:
        # Large (1M-100M)
        num_neighbors = 100
        search_list_size = 200
        max_alpha = 1.2
    else:
        # Very Large (>100M)
        num_neighbors = 200
        search_list_size = 500
        max_alpha = 1.5

    # Use memory_optimized (SBQ compression) for all dataset sizes - best balance of storage & performance
    storage_layout = 'memory_optimized'

    print(f"Creating DiskANN index with optimized parameters:")
    print(f"  num_neighbors={num_neighbors}, search_list_size={search_list_size}")
    print(f"  max_alpha={max_alpha}, storage_layout='{storage_layout}'")

    index_start = time.time()
    cursor.execute(f"""
        CREATE INDEX idx_vectors_embedding_diskann
        ON vectors USING diskann (embedding vector_cosine_ops)
        WITH (
            num_neighbors = {num_neighbors},
            search_list_size = {search_list_size},
            max_alpha = {max_alpha},
            storage_layout = '{storage_layout}',
            num_dimensions = 512
        )
    """)
    conn.commit()
    index_time = time.time() - index_start
    print(f"DiskANN index created in {index_time:.2f} seconds")

    diskann_params = {
        'num_neighbors': num_neighbors,
        'search_list_size': search_list_size,
        'max_alpha': max_alpha,
        'storage_layout': storage_layout
    }

    benchmark_index(cursor, 'idx_vectors_embedding_diskann', 'DiskANN',
                   query_str, baseline_ids, baseline_time, db_size, table_size,
                   diskann_params=diskann_params)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
