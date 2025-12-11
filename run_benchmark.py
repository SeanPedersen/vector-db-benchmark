#!/usr/bin/env python3
"""
Example: Unified Vector Database Benchmark

This script demonstrates the unified benchmark approach that tests all combinations of:
- Data types: float32 (f32), float16 (f16), binary (mean, uint8, uint4)
- Matryoshka dimensions: 256, 512, 1024
- Indexes: vchordrq, ivfflat, hnsw, diskann, binary variants, exact

Key improvements over the original approach:
1. Single unified table per dimension with columns: embedding_f32, embedding_f16, embedding_bin_*
2. Eliminates data duplication between float32/float16 tables
3. Systematic testing via INDEX_CONFIGS matrix
4. DiskANN and HNSW integration
5. Cleaner, more maintainable code structure
"""

import numpy as np
import argparse
from config import INDEX_CONFIGS, DIMENSIONS, K
from database import (
    ensure_connection,
    create_and_insert_unified_table,
    build_index,
    get_index_size_mb,
    drop_all_indices_on_table,
    print_extension_versions,
)
from queries import query_index
from embeddings import (
    generate_embeddings,
    compute_dimension_means,
    compute_percentile_thresholds,
)
from metrics import build_baseline


def run_unified_benchmark(
    num_vectors=50000,
    dimensions=[256, 512, 1024],
    selected_benchmarks=None,
    vectors_file=None,
):
    """Run unified benchmark with specified parameters.

    Args:
        num_vectors: Number of vectors to test
        dimensions: List of matryoshka dimensions to test
        selected_benchmarks: List of benchmark names to run (None = all)
        vectors_file: Path to .npy file with pre-generated vectors (optional)
    """

    # Load or generate embeddings
    if vectors_file:
        print(f"[Setup] Loading vectors from {vectors_file}...")
        full_embeddings = np.load(vectors_file)

        if full_embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {full_embeddings.shape}")

        # Limit to specified size
        if full_embeddings.shape[0] > num_vectors:
            print(f"[Setup] Limiting to {num_vectors:,} vectors")
            full_embeddings = full_embeddings[:num_vectors]
        else:
            num_vectors = full_embeddings.shape[0]
            print(f"[Setup] Using all {num_vectors:,} vectors from file")

        # Ensure normalized
        norms = np.linalg.norm(full_embeddings, axis=1, keepdims=True)
        full_embeddings = full_embeddings / norms

        # Pad or truncate to 1024 dimensions
        if full_embeddings.shape[1] < 1024:
            print(f"[Setup] Padding from {full_embeddings.shape[1]} to 1024 dimensions")
            padded = np.zeros((full_embeddings.shape[0], 1024), dtype=np.float32)
            padded[:, :full_embeddings.shape[1]] = full_embeddings
            full_embeddings = padded
        elif full_embeddings.shape[1] > 1024:
            print(f"[Setup] Truncating from {full_embeddings.shape[1]} to 1024 dimensions")
            full_embeddings = full_embeddings[:, :1024]
    else:
        # Generate random embeddings
        print(f"[Setup] Generating {num_vectors:,} random vectors at 1024 dimensions...")
        full_embeddings = generate_embeddings(num_vectors, 1024)

    # Use a random vector from the dataset as the query (more realistic)
    print("[Setup] Selecting random query vector from dataset...")
    np.random.seed(999)
    query_idx = np.random.randint(0, num_vectors)
    query = full_embeddings[query_idx].copy()
    print(f"[Setup] Using vector at index {query_idx} as query")

    # Compute baseline (ground truth) using full 1024-D vectors
    print("[Baseline] Computing 1024-D ground truth...")
    baseline_ids_1024 = build_baseline(full_embeddings, query)
    print(f"[Baseline] Sanity check - top-1 is query itself: {baseline_ids_1024[0] == query_idx}")

    # Precompute thresholds for binary quantization
    dimension_means = compute_dimension_means(full_embeddings)
    uint8_thresholds = compute_percentile_thresholds(full_embeddings, num_buckets=8)
    uint4_thresholds = compute_percentile_thresholds(full_embeddings, num_buckets=4)

    # Connect to database
    conn = ensure_connection()

    # Load PostgreSQL extensions
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vectorscale")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vchord")
    conn.commit()
    cursor.close()

    # Print extension versions
    print_extension_versions(conn)

    # Results storage
    results = []

    # Use all benchmarks if none specified or if "all" is specified
    if selected_benchmarks is None:
        selected_benchmarks = list(INDEX_CONFIGS.keys())
    elif "all" in selected_benchmarks:
        selected_benchmarks = list(INDEX_CONFIGS.keys())

    print(f"\n[Setup] Running benchmarks: {', '.join(selected_benchmarks)}")
    print(f"[Setup] Testing dimensions: {dimensions}")

    # Determine data source identifier and caching strategy
    # Only cache file-based vectors since random vectors are meaningless to cache
    if vectors_file:
        import os
        data_source = os.path.basename(vectors_file)
        skip_if_exists = True
    else:
        data_source = None  # Random vectors - don't cache
        skip_if_exists = False

    # Main benchmark loop: iterate over dimensions
    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"Dimension: {dim}")
        print(f"{'='*60}")

        # Prepare truncated embeddings (Matryoshka prefix slicing)
        trunc_embeddings = full_embeddings[:, :dim]
        norms = np.linalg.norm(trunc_embeddings, axis=1, keepdims=True)
        trunc_embeddings = trunc_embeddings / norms

        # Truncate query to match dimension
        query_trunc = query[:dim] / np.linalg.norm(query[:dim])

        # Use 1024-D ground truth for all dimensions
        baseline_ids = baseline_ids_1024

        # Create unified table
        # Only cache if using file-based vectors (random vectors are fast to regenerate)
        table = f"items_{dim}"
        if skip_if_exists:
            print(f"\n[Table] Checking/creating table: {table} (source: {data_source})")
        else:
            print(f"\n[Table] Creating table: {table} (random vectors - always fresh)")

        create_and_insert_unified_table(
            conn,
            table,
            trunc_embeddings,
            use_mean_binarization=True,
            dimension_means=dimension_means,
            use_uint8_binarization=True,
            uint8_thresholds=uint8_thresholds,
            use_uint4_binarization=True,
            uint4_thresholds=uint4_thresholds,
            encoding_type="thermometer",
            data_source=data_source,
            skip_if_exists=skip_if_exists,
        )

        # Drop all existing indices to ensure clean benchmarking with current parameters
        print(f"[Index] Dropping all existing indices on {table}...")
        drop_all_indices_on_table(conn, table)

        # Test each selected index type
        for index_type in selected_benchmarks:
            if index_type not in INDEX_CONFIGS:
                available = ', '.join(INDEX_CONFIGS.keys())
                print(f"[Warning] Unknown index type: '{index_type}', skipping. Available: {available}")
                continue

            config = INDEX_CONFIGS[index_type]

            # Test full precision variants (f32, f16)
            for prec in config["precisions"]:
                print(f"\n[Index] Building {index_type} ({prec})...")
                try:
                    idx_name, build_time = build_index(
                        conn, table, prec, index_type, dim, unified_table=True
                    )
                    idx_size = get_index_size_mb(conn, idx_name)

                    print(f"[Query] Testing {index_type} ({prec})...")
                    latency, recall = query_index(
                        conn,
                        table,
                        prec,
                        index_type,
                        dim,
                        query_trunc,
                        baseline_ids,
                        unified_table=True,
                    )

                    results.append({
                        "dimension": dim,
                        "index_type": index_type,
                        "precision": prec,
                        "variant": None,
                        "build_time_s": build_time,
                        "index_size_mb": idx_size,
                        "query_latency_s": latency,
                        "recall": recall,
                    })

                    print(f"  ✓ Recall: {recall*100:.1f}%, Latency: {latency*1000:.2f}ms, Size: {idx_size:.1f}MB")

                except Exception as e:
                    print(f"  ✗ Error: {e}")

            # Test binary variants (mean, uint8, uint4)
            if config["binary"]:
                for variant in config["variants"]:
                    variant_flags = {
                        "use_mean_bin": variant == "mean",
                        "use_uint8_bin": variant == "uint8",
                        "use_uint4_bin": variant == "uint4",
                    }

                    print(f"\n[Index] Building {index_type} (binary-{variant})...")
                    try:
                        idx_name, build_time = build_index(
                            conn, table, "f32", index_type, dim,
                            unified_table=True,
                            **variant_flags
                        )
                        idx_size = get_index_size_mb(conn, idx_name)

                        print(f"[Query] Testing {index_type} (binary-{variant})...")
                        latency, recall = query_index(
                            conn,
                            table,
                            "f32",  # Rerank using f32
                            index_type,
                            dim,
                            query_trunc,
                            baseline_ids,
                            unified_table=True,
                            dimension_means=dimension_means,
                            uint8_thresholds=uint8_thresholds,
                            uint4_thresholds=uint4_thresholds,
                            **variant_flags
                        )

                        results.append({
                            "dimension": dim,
                            "index_type": index_type,
                            "precision": "binary",
                            "variant": variant,
                            "build_time_s": build_time,
                            "index_size_mb": idx_size,
                            "query_latency_s": latency,
                            "recall": recall,
                        })

                        print(f"  ✓ Recall: {recall*100:.1f}%, Latency: {latency*1000:.2f}ms, Size: {idx_size:.1f}MB")

                    except Exception as e:
                        print(f"  ✗ Error: {e}")

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dim':<6} {'Index':<15} {'Precision':<12} {'Variant':<8} {'Recall':<8} {'Latency(ms)':<12} {'Size(MB)':<10}")
    print("-" * 80)

    for r in results:
        variant_str = r["variant"] or "-"
        print(
            f"{r['dimension']:<6} "
            f"{r['index_type']:<15} "
            f"{r['precision']:<12} "
            f"{variant_str:<8} "
            f"{r['recall']*100:>6.1f}% "
            f"{r['query_latency_s']*1000:>10.2f} "
            f"{r['index_size_mb']:>10.1f}"
        )

    conn.close()
    return results


if __name__ == "__main__":
    # Build list of available indexes for help text
    available_indexes = list(INDEX_CONFIGS.keys())
    indexes_help = f"Benchmarks to run (can specify multiple, use 'all' for all indexes, default: all). Available: {', '.join(available_indexes)}"

    parser = argparse.ArgumentParser(
        description="Unified vector database benchmark example"
    )
    parser.add_argument(
        "--vectors-file",
        type=str,
        help="Path to .npy file with pre-generated vectors (e.g., instagram_vectors.npy)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50000,
        help="Number of vectors (default: 50000)",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default="256,512,1024",
        help="Comma-separated dimensions (default: 256,512,1024)",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        action="append",
        help=indexes_help,
    )

    args = parser.parse_args()

    # Parse dimensions
    dims = [int(d.strip()) for d in args.dimensions.split(",")]

    # Run unified benchmark
    results = run_unified_benchmark(
        num_vectors=args.size,
        dimensions=dims,
        selected_benchmarks=args.benchmark,
        vectors_file=args.vectors_file,
    )

    print(f"\n[Complete] Tested {len(results)} configurations")
