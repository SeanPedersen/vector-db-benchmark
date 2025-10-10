#!/usr/bin/env python3
"""Query pgvectorscale indices and measure performance."""

import numpy as np
import psycopg2
import time
import argparse

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
}

K_NEIGHBORS = 100


def compute_recall(retrieved_ids, baseline_ids):
    """Compute retrieval recall."""
    baseline_set = set(baseline_ids)
    retrieved_set = set(retrieved_ids)
    intersection = baseline_set & retrieved_set
    recall = len(intersection) / len(baseline_ids)
    return recall


def benchmark_index(
    cursor,
    index_name,
    index_type,
    query_str,
    baseline_ids,
    baseline_time,
    db_size,
    table_size,
    probes=None,
    diskann_params=None,
    hnsw_ef=None,
):
    """Run benchmark for a specific index."""
    # Warm-up query
    cursor.execute(
        f"""
        SELECT id
        FROM vectors
        ORDER BY embedding <=> %s::vector
        LIMIT {K_NEIGHBORS}
    """,
        (query_str,),
    )
    cursor.fetchall()

    # Actual benchmark query
    start_time = time.time()

    if index_type == "DiskANN":
        pass
    elif index_type == "IVFFlat":
        probes_value = probes if probes is not None else 200
        cursor.execute(f"SET ivfflat.probes = {probes_value}")
    elif index_type == "HNSW":
        ef_value = hnsw_ef if hnsw_ef is not None else 100
        cursor.execute(f"SET hnsw.ef_search = {ef_value}")

    cursor.execute(
        f"""
        SELECT id, embedding <=> %s::vector as distance
        FROM vectors
        ORDER BY embedding <=> %s::vector
        LIMIT {K_NEIGHBORS}
    """,
        (query_str, query_str),
    )

    results = cursor.fetchall()
    query_latency = time.time() - start_time

    retrieved_ids = np.array([row[0] for row in results])
    distances = np.array([row[1] for row in results])

    # Compute recall
    recall = compute_recall(retrieved_ids, baseline_ids)

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

    print(
        f"✓ pgvectorscale ({index_type}): {query_latency * 1000:.2f}ms, {recall * 100:.0f}% recall"
    )

    return {
        "query_latency": query_latency,
        "recall": recall,
        "index_size": index_size,
        "index_size_mb": index_size_mb,
    }


def run_benchmark(query, baseline_ids, baseline_time):
    """Run pgvectorscale benchmarks and return results."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Get number of vectors from database
    cursor.execute("SELECT COUNT(*) FROM vectors")
    num_vectors = cursor.fetchone()[0]

    query_str = "[" + ",".join(map(str, query)) + "]"

    # Get database stats (shared)
    cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
    db_size = cursor.fetchone()[0]

    cursor.execute("""
        SELECT pg_size_pretty(pg_total_relation_size('vectors'))
    """)
    table_size = cursor.fetchone()[0]

    results = []

    # Test 0: HNSW index
    print("\n[pgvectorscale] Building HNSW index...")
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

    index_start = time.time()
    cursor.execute(f"""
        CREATE INDEX idx_vectors_embedding_hnsw
        ON vectors USING hnsw (embedding vector_cosine_ops)
        WITH (m = {m}, ef_construction = {ef_construction})
    """)
    conn.commit()
    hnsw_index_time = time.time() - index_start
    print(f"[pgvectorscale] HNSW index built in {hnsw_index_time:.2f}s")

    hnsw_result = benchmark_index(
        cursor,
        "idx_vectors_embedding_hnsw",
        "HNSW",
        query_str,
        baseline_ids,
        baseline_time,
        db_size,
        table_size,
        hnsw_ef=ef_search,
    )

    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_hnsw")
    conn.commit()

    results.append(("HNSW", hnsw_result, hnsw_index_time))

    # Test 1: IVFFlat index
    print("\n[pgvectorscale] Building IVFFlat index...")
    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_ivfflat")
    conn.commit()

    # Calculate optimal lists based on dataset size
    if num_vectors <= 1_000_000:
        lists = max(50, num_vectors // 1000)
    else:
        lists = int(num_vectors**0.5)

    probes = int(lists**0.5)

    index_start = time.time()
    cursor.execute(f"""
        CREATE INDEX idx_vectors_embedding_ivfflat
        ON vectors USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {lists})
    """)
    conn.commit()
    ivf_index_time = time.time() - index_start
    print(f"[pgvectorscale] IVFFlat index built in {ivf_index_time:.2f}s")

    ivf_result = benchmark_index(
        cursor,
        "idx_vectors_embedding_ivfflat",
        "IVFFlat",
        query_str,
        baseline_ids,
        baseline_time,
        db_size,
        table_size,
        probes=probes,
    )

    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_ivfflat")
    conn.commit()

    results.append(("IVFFlat", ivf_result, ivf_index_time))

    # Test 2: DiskANN index
    print("\n[pgvectorscale] Building DiskANN index...")
    cursor.execute("DROP INDEX IF EXISTS idx_vectors_embedding_diskann")
    conn.commit()

    # Calculate optimal DiskANN parameters based on dataset size
    if num_vectors < 1_042:
        num_neighbors = 32
        search_list_size = 50
        max_alpha = 1.0
    elif num_vectors < 1_000_042:
        num_neighbors = 50
        search_list_size = 100
        max_alpha = 1.0
    elif num_vectors < 100_000_000:
        num_neighbors = 100
        search_list_size = 200
        max_alpha = 1.2
    else:
        num_neighbors = 200
        search_list_size = 500
        max_alpha = 1.5

    storage_layout = "memory_optimized"

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
    diskann_index_time = time.time() - index_start
    print(f"[pgvectorscale] DiskANN index built in {diskann_index_time:.2f}s")

    diskann_params = {
        "num_neighbors": num_neighbors,
        "search_list_size": search_list_size,
        "max_alpha": max_alpha,
        "storage_layout": storage_layout,
    }

    diskann_result = benchmark_index(
        cursor,
        "idx_vectors_embedding_diskann",
        "DiskANN",
        query_str,
        baseline_ids,
        baseline_time,
        db_size,
        table_size,
        diskann_params=diskann_params,
    )

    results.append(("DiskANN", diskann_result, diskann_index_time))

    cursor.close()
    conn.close()

    # Return formatted results
    return [
        {
            "method": index_type,
            "latency": result["query_latency"],
            "recall": result["recall"],
            "build_time": index_time,
            "size_mb": result["index_size_mb"],
        }
        for index_type, result, index_time in results
    ]


def main():
    parser = argparse.ArgumentParser(description="Query pgvectorscale indices")
    parser.add_argument(
        "--baseline-time",
        type=float,
        default=None,
        help="Baseline query time in seconds (from compute_baseline.py)",
    )
    args = parser.parse_args()

    query = np.load("query.npy")
    baseline_ids = np.load("baseline_ids.npy")

    results = run_benchmark(query, baseline_ids, args.baseline_time)

    # Output results for main script to capture (stderr so it doesn't appear in stdout)
    import sys

    for result in results:
        print(
            f"RESULT:{result['method']}:{result['latency']}:{result['recall']}:{result['build_time']}:{result['size_mb']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
