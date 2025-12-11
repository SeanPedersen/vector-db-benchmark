#!/usr/bin/env python3
"""Retrieval benchmark implementing TASK.md knobs.

Compares recall / latency across:
- Matryoshka storage dimensions: 128D (configurable)
- Storage float precision: float32 (vector) vs float16 (halfvec)
- Index types:
  * VectorChord vchordrq: full precision (vector, halfvec)
  * IVFFlat: full precision (vector, halfvec) + binary quantization
  * HNSW: binary quantization

Binary Quantization Strategy:
  Reduces each floating-point dimension to 1 bit (sign bit: >= 0 → 1, < 0 → 0)
  - Step 1: Fast overfetch 1000 neighbors using Hamming distance on bit vectors
  - Step 2: Fetch float32/float16 vectors for 1000 candidates (depends on table)
  - Step 3: Compute cosine distances using PostgreSQL
  - Step 4: Rerank to get top 100 by cosine similarity
  - Result: Memory-efficient index with high recall via overfetching
  - Tests both float32 and float16 reranking for binary indices

All recall measured against brute force (float32 1024-d) baseline.

NOTE: Uses random normalized embeddings as stand-in for CLIP v2 text embeddings.
"""

import numpy as np
import time
import argparse

# Import from refactored modules
import config
from params import recommend_ivf_params
from embeddings import (
    generate_embeddings,
    compute_dimension_means,
    compute_percentile_thresholds,
)
from metrics import build_baseline, get_vector_storage_mb, get_binary_storage_mb
from database import (
    ensure_connection,
    table_exists_and_populated,
    create_and_insert_table,
    build_index,
    get_index_size_mb,
)
from queries import query_index

# Import all global configuration variables
IVF_LISTS = config.IVF_LISTS
IVF_PROBES = config.IVF_PROBES
IVF_PROBES_BINARY = config.IVF_PROBES_BINARY
HNSW_EF_SEARCH = config.HNSW_EF_SEARCH
HNSW_M = config.HNSW_M
HNSW_EF_CONSTRUCTION = config.HNSW_EF_CONSTRUCTION
HNSW_EF_SEARCH_MAX = config.HNSW_EF_SEARCH_MAX
OVERFETCH_FACTOR = config.OVERFETCH_FACTOR
K = config.K


def sync_config_globals(**kwargs):
    """Sync local globals back to config module so other modules see updates."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)


def main():
    global IVF_LISTS, IVF_PROBES, IVF_PROBES_BINARY, HNSW_EF_SEARCH, OVERFETCH_FACTOR, HNSW_M, HNSW_EF_CONSTRUCTION, K, HNSW_EF_SEARCH_MAX  # NEW: include HNSW_EF_SEARCH_MAX

    parser = argparse.ArgumentParser(description="Retrieval benchmark (TASK.md)")
    parser.add_argument(
        "--vectors-file", type=str, help="Path to .npy file with pre-generated vectors"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50000,
        help="Number of vectors to use (default: 50000)",
    )
    parser.add_argument(
        "--num-vectors",
        type=int,
        help="(Deprecated, use --size) Number of vectors to generate",
    )
    # NEW: allow configuring k (neighbors to retrieve)
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of nearest neighbors to retrieve (default: 100)",
    )
    # NEW: unified table mode
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Use unified table structure with all precision columns (embedding_f32, embedding_f16)",
    )
    # NEW: make IVFFlat params optional to enable auto-tuning
    parser.add_argument(
        "--ivf-lists",
        type=int,
        default=None,
        help="IVF lists; if omitted, auto-tuned from dataset size and k",
    )
    parser.add_argument(
        "--ivf-probes",
        type=int,
        default=None,
        help="IVF probes for non-binary; if omitted, auto-tuned",
    )
    parser.add_argument(
        "--ivf-probes-binary",
        type=int,
        default=None,
        help="IVF probes for binary; if omitted, auto-tuned",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=1000,
        help="HNSW ef_search for binary index (should be >= overfetch; default: 1000)",
    )
    # NEW: cap for ef_search to satisfy server GUC range
    parser.add_argument(
        "--hnsw-ef-search-max",
        type=int,
        default=1000,
        help="Upper cap for hnsw.ef_search to satisfy server limits (default: 1000)",
    )
    # NEW: HNSW build knobs
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="HNSW M (max links per node) for binary index (default: 32)",
    )
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=300,
        help="HNSW ef_construction for binary index build (default: 300)",
    )
    parser.add_argument(
        "--overfetch",
        type=int,
        default=10,
        help="Binary overfetch factor (default: 10)",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reload data even if tables exist",
    )
    parser.add_argument(
        "--explain-analyze",
        action="store_true",
        help="Print EXPLAIN ANALYZE output for query debugging (shows only for 512-D queries)",
    )

    # NEW: Benchmark selection via positive selection (now supports comma-separated lists per -b)
    parser.add_argument(
        "--benchmark",
        "-b",
        action="append",
        help=(
            "Benchmarks to run (can be specified multiple times or as comma-separated list). "
            "Available: vchordrq, ivfflat, binary-hnsw, binary-ivf, binary-exact, exact, all. "
            "Examples: -b binary-hnsw -b exact  |  -b binary-hnsw,binary-exact,ivfflat"
        ),
    )
    parser.add_argument(
        "--enable-mean-binarization",
        action="store_true",
        help="Enable mean-based binarization (uses corpus-wide mean per dimension as threshold)",
    )
    parser.add_argument(
        "--enable-quasi-uint8",
        action="store_true",
        help="Enable quasi-uint8 binarization (8 buckets per dimension using percentile thresholds)",
    )
    parser.add_argument(
        "--enable-quasi-uint4",
        action="store_true",
        help="Enable quasi-uint4 binarization (4 buckets per dimension using percentile thresholds)",
    )
    parser.add_argument(
        "--encoding-type",
        type=str,
        choices=["thermometer", "one-hot"],
        default="thermometer",
        help="Encoding type for quasi-uint8 and quasi-uint4: 'thermometer' (unary) or 'one-hot' (categorical). Default: thermometer",
    )
    parser.add_argument(
        "--dimensions",
        "--dims",
        type=str,
        default="256,512,1024",
        help="Comma-separated list of dimensions to test (default: 256,512,1024). Must not exceed vector dimensions.",
    )

    args = parser.parse_args()

    # Expand and validate benchmark selections (supports comma-separated values per -b)
    allowed_benchmarks = {
        "vchordrq",
        "ivfflat",
        "hnsw",
        "diskann",
        "binary-hnsw",
        "binary-ivf",
        "binary-exact",
        "exact",
        "all",
    }
    raw_benchmarks = args.benchmark or []
    expanded = []
    for entry in raw_benchmarks:
        for token in entry.split(","):
            b = token.strip()
            if not b:
                continue
            if b not in allowed_benchmarks:
                # Use parser.error for consistent CLI error formatting
                parser.error(
                    f"Invalid benchmark '{b}'. Allowed: {', '.join(sorted(allowed_benchmarks))}"
                )
            expanded.append(b)

    if not expanded or "all" in expanded:
        benchmarks = allowed_benchmarks - {"all"}
    else:
        benchmarks = set(expanded) - {"all"}

    print(f"[Setup] Running benchmarks: {', '.join(sorted(benchmarks))}")

    # Handle backward compatibility for --num-vectors
    if args.num_vectors is not None:
        num_vectors = args.num_vectors
        print("[Warning] --num-vectors is deprecated, use --size instead")
    else:
        num_vectors = args.size

    # NEW: set global K from CLI
    K = args.k

    # Update global parameters from args (with auto-tuning for IVFFlat)
    if (
        args.ivf_lists is None
        or args.ivf_probes is None
        or args.ivf_probes_binary is None
    ):
        rec_lists, rec_probes, rec_probes_bin = recommend_ivf_params(num_vectors, K)
        IVF_LISTS = rec_lists if args.ivf_lists is None else args.ivf_lists
        IVF_PROBES = rec_probes if args.ivf_probes is None else args.ivf_probes
        IVF_PROBES_BINARY = (
            rec_probes_bin if args.ivf_probes_binary is None else args.ivf_probes_binary
        )
        print(
            f"[Params] IVFFlat auto: lists={IVF_LISTS}, probes={IVF_PROBES}, probes_binary={IVF_PROBES_BINARY} (N={num_vectors:,}, k={K})"
        )
    else:
        IVF_LISTS = args.ivf_lists
        IVF_PROBES = args.ivf_probes
        IVF_PROBES_BINARY = args.ivf_probes_binary

    OVERFETCH_FACTOR = args.overfetch
    HNSW_M = args.hnsw_m
    HNSW_EF_CONSTRUCTION = args.hnsw_ef_construction
    # Ensure ef_search >= overfetch (k) per HNSW guidance, but clamp to server cap
    required_ef = K * OVERFETCH_FACTOR
    desired_ef = max(args.hnsw_ef_search, required_ef)
    if desired_ef > HNSW_EF_SEARCH_MAX:
        print(
            f"[Params] Capping hnsw_ef_search at {HNSW_EF_SEARCH_MAX} (requested {desired_ef}) to satisfy server range"
        )
    HNSW_EF_SEARCH = min(desired_ef, HNSW_EF_SEARCH_MAX)

    # Sync all config changes to the config module so other modules see them
    sync_config_globals(
        K=K,
        IVF_LISTS=IVF_LISTS,
        IVF_PROBES=IVF_PROBES,
        IVF_PROBES_BINARY=IVF_PROBES_BINARY,
        OVERFETCH_FACTOR=OVERFETCH_FACTOR,
        HNSW_M=HNSW_M,
        HNSW_EF_CONSTRUCTION=HNSW_EF_CONSTRUCTION,
        HNSW_EF_SEARCH=HNSW_EF_SEARCH,
        HNSW_EF_SEARCH_MAX=HNSW_EF_SEARCH_MAX,
    )

    # Load or generate embeddings
    if args.vectors_file:
        print(f"[Setup] Loading vectors from {args.vectors_file}...")
        full_embeddings = np.load(args.vectors_file)

        if full_embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D array from .npy file, got shape {full_embeddings.shape}"
            )

        # Limit to specified size
        if full_embeddings.shape[0] > num_vectors:
            print(
                f"[Setup] Limiting vectors from {full_embeddings.shape[0]:,} to {num_vectors:,}"
            )
            full_embeddings = full_embeddings[:num_vectors]
        else:
            num_vectors = full_embeddings.shape[0]
            print(f"[Setup] Using all {num_vectors:,} vectors from file")

        # Ensure vectors are normalized
        norms = np.linalg.norm(full_embeddings, axis=1, keepdims=True)
        full_embeddings = full_embeddings / norms

        # Pad or truncate to 1024 dimensions if needed
        if full_embeddings.shape[1] < 1024:
            print(
                f"[Setup] Padding vectors from {full_embeddings.shape[1]} to 1024 dimensions"
            )
            padded = np.zeros((full_embeddings.shape[0], 1024), dtype=np.float32)
            padded[:, : full_embeddings.shape[1]] = full_embeddings
            full_embeddings = padded
        elif full_embeddings.shape[1] > 1024:
            print(
                f"[Setup] Truncating vectors from {full_embeddings.shape[1]} to 1024 dimensions"
            )
            full_embeddings = full_embeddings[:, :1024]

        print(
            f"[Setup] Loaded {num_vectors:,} vectors with {full_embeddings.shape[1]} dimensions"
        )
    else:
        print(f"[Setup] Generating {num_vectors:,} normalized 1024-d embeddings...")
        full_embeddings = generate_embeddings(num_vectors)

    # Parse and validate dimensions to test
    try:
        requested_dims = [int(d.strip()) for d in args.dimensions.split(",")]
    except ValueError:
        print(
            f"[Error] Invalid dimensions format: '{args.dimensions}'. Expected comma-separated integers."
        )
        return

    max_dim = full_embeddings.shape[1]
    invalid_dims = [d for d in requested_dims if d > max_dim or d <= 0]
    if invalid_dims:
        print(
            f"[Error] Invalid dimensions {invalid_dims}: must be positive and not exceed vector dimensions ({max_dim})"
        )
        return

    # Use requested dimensions (sorted for consistency)
    DIMENSIONS = sorted(requested_dims)
    print(f"[Setup] Testing dimensions: {DIMENSIONS}")

    # Compute dimension means for mean-based binarization if enabled
    dimension_means_1024 = None
    if args.enable_mean_binarization:
        print("[Setup] Computing dimension-wise means for mean-based binarization...")
        dimension_means_1024 = compute_dimension_means(full_embeddings)
        print(f"[Setup] Computed means for all 1024 dimensions")
        print(
            f"[Setup] Mean range: [{dimension_means_1024.min():.6f}, {dimension_means_1024.max():.6f}]"
        )

    # Compute percentile thresholds for quasi-uint8 binarization if enabled
    uint8_thresholds_1024 = None
    if args.enable_quasi_uint8:
        print("[Setup] Computing percentile thresholds for quasi-uint8 binarization...")
        uint8_thresholds_1024 = compute_percentile_thresholds(
            full_embeddings, num_buckets=8
        )
        print(
            f"[Setup] Computed 7 thresholds per dimension (8 buckets) for all 1024 dimensions"
        )
        print(
            f"[Setup] Threshold range: [{uint8_thresholds_1024.min():.6f}, {uint8_thresholds_1024.max():.6f}]"
        )

    # Compute percentile thresholds for quasi-uint4 binarization if enabled
    uint4_thresholds_1024 = None
    if args.enable_quasi_uint4:
        print("[Setup] Computing percentile thresholds for quasi-uint4 binarization...")
        uint4_thresholds_1024 = compute_percentile_thresholds(
            full_embeddings, num_buckets=4
        )
        print(
            f"[Setup] Computed 3 thresholds per dimension (4 buckets) for all 1024 dimensions"
        )
        print(
            f"[Setup] Threshold range: [{uint4_thresholds_1024.min():.6f}, {uint4_thresholds_1024.max():.6f}]"
        )

    # Use a random vector from the dataset as the query (more realistic)
    print("[Setup] Selecting random query vector from dataset...")
    np.random.seed(999)
    query_idx = np.random.randint(0, num_vectors)
    query = full_embeddings[query_idx].copy()
    print(f"[Setup] Using vector at index {query_idx} as query")

    # Compute ground truth baseline using full 1024-D embeddings
    # This will be used for ALL dimensions to measure true recall
    print("[Baseline] Computing brute force baseline (1024-D float32)...")
    baseline_ids_1024 = build_baseline(full_embeddings, query)
    print(f"[Baseline] Ground truth top-{K} computed using full 1024-D vectors")
    print(
        f"[Baseline] Top-1 should be query itself (id={query_idx}): {baseline_ids_1024[0] == query_idx}"
    )

    print(
        f"[Params] k={K} | IVF: lists={IVF_LISTS}, probes={IVF_PROBES} (binary: {IVF_PROBES_BINARY}) | "
        f"Binary/HNSW: overfetch={OVERFETCH_FACTOR}x, m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}, "
        f"ef_search={HNSW_EF_SEARCH}, ef_search_cap={HNSW_EF_SEARCH_MAX}"
    )

    # NEW: concise IVFFlat tuning guidance for ~50K vectors and k≈1000
    if num_vectors >= 50000 and K >= 1000:
        print(
            "[Tips] IVFFlat: with ~50K vectors and k≈1000, try lists in [50, 500] (e.g., 200–300) and probes=10–20;"
        )
        print(
            "[Tips] increasing probes improves recall but slows queries; probes≈lists approaches exact search."
        )
    # NEW: concise IVFFlat tuning guidance for ~50K vectors and k≈100
    elif num_vectors >= 50000 and K <= 200:
        print(
            "[Tips] IVFFlat: with ~50K vectors and k≈100, try lists in [200, 300] and probes≈12–20; "
        )
        print(
            "[Tips] binary IVFFlat: set probes ~3x non-binary probes (e.g., 36–60). HNSW ef_search≈1000 works well with 10x overfetch."
        )

    conn = ensure_connection()
    # Ensure required extensions; handle failures with rollback to avoid aborted transaction
    cur_ext = conn.cursor()
    try:
        cur_ext.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception:
        conn.rollback()
    try:
        cur_ext.execute("CREATE EXTENSION IF NOT EXISTS vectorscale")
    except Exception:
        conn.rollback()
    try:
        cur_ext.execute("CREATE EXTENSION IF NOT EXISTS vchord")
    except Exception:
        conn.rollback()
    conn.commit()
    cur_ext.close()

    results = []

    for dim in DIMENSIONS:
        print(f"\n=== Dimension {dim} ===")
        # Prepare truncated embeddings (prefix)
        trunc_embeddings = full_embeddings[:, :dim]
        # Normalize again (Matryoshka often already approx normalized, enforce)
        norms = np.linalg.norm(trunc_embeddings, axis=1, keepdims=True)
        trunc_embeddings = trunc_embeddings / norms

        # Truncate query to match dimension (for querying the index)
        query_trunc = query[:dim] / np.linalg.norm(
            query[:dim]
        )  # Truncate and renormalize

        # Use 1024-D ground truth baseline for ALL dimensions
        # This measures how well lower dimensions approximate the full 1024-D space
        baseline_ids = baseline_ids_1024

        # Create tables & insert
        tbl_vector = f"items_vec_{dim}"
        tbl_half = f"items_half_{dim}"

        # Check if tables already exist and are populated
        vec_exists = table_exists_and_populated(
            conn,
            tbl_vector,
            num_vectors,
            check_mean_bin=args.enable_mean_binarization,
            check_uint8_bin=args.enable_quasi_uint8,
            check_uint4_bin=args.enable_quasi_uint4,
        )
        half_exists = table_exists_and_populated(
            conn,
            tbl_half,
            num_vectors,
            check_mean_bin=args.enable_mean_binarization,
            check_uint8_bin=args.enable_quasi_uint8,
            check_uint4_bin=args.enable_quasi_uint4,
        )

        if args.force_reload or not vec_exists:
            print(f"[Storage] Creating + inserting float32 table {tbl_vector} ...")
            create_and_insert_table(
                conn,
                tbl_vector,
                trunc_embeddings,
                "vector",
                use_mean_binarization=args.enable_mean_binarization,
                dimension_means=dimension_means_1024,
                use_uint8_binarization=args.enable_quasi_uint8,
                uint8_thresholds=uint8_thresholds_1024,
                use_uint4_binarization=args.enable_quasi_uint4,
                uint4_thresholds=uint4_thresholds_1024,
                encoding_type=args.encoding_type,
            )
        else:
            print(
                f"[Storage] Skipping {tbl_vector} (already populated with {num_vectors:,} rows)"
            )

        if args.force_reload or not half_exists:
            print(f"[Storage] Creating + inserting float16 table {tbl_half} ...")
            create_and_insert_table(
                conn,
                tbl_half,
                trunc_embeddings,
                "halfvec",
                use_mean_binarization=args.enable_mean_binarization,
                dimension_means=dimension_means_1024,
                use_uint8_binarization=args.enable_quasi_uint8,
                uint8_thresholds=uint8_thresholds_1024,
                use_uint4_binarization=args.enable_quasi_uint4,
                uint4_thresholds=uint4_thresholds_1024,
                encoding_type=args.encoding_type,
            )
        else:
            print(
                f"[Storage] Skipping {tbl_half} (already populated with {num_vectors:,} rows)"
            )

        # Build indices (conditionally based on selected benchmarks)
        # VectorChord indices
        if "vchordrq" in benchmarks:
            print("[Index] Building VectorChord (vector)...")
            idx_vchord_vec, t_vchord_vec = build_index(
                conn, tbl_vector, "vector", "full", dim
            )
            print("[Index] Building VectorChord (halfvec)...")
            idx_vchord_half, t_vchord_half = build_index(
                conn, tbl_half, "halfvec", "half", dim
            )
        else:
            idx_vchord_vec, t_vchord_vec = None, 0.0
            idx_vchord_half, t_vchord_half = None, 0.0

        # IVFFlat indices
        if "ivfflat" in benchmarks:
            print(f"[Index] Building IVFFlat (vector, lists={IVF_LISTS})...")
            idx_ivf_vec, t_ivf_vec = build_index(conn, tbl_vector, "vector", "ivf", dim)
            print(f"[Index] Building IVFFlat (halfvec, lists={IVF_LISTS})...")
            idx_ivf_half, t_ivf_half = build_index(
                conn, tbl_half, "halfvec", "ivf", dim
            )
        else:
            idx_ivf_vec, t_ivf_vec = None, 0.0
            idx_ivf_half, t_ivf_half = None, 0.0

        # Binary HNSW indices
        if "binary-hnsw" in benchmarks:
            print("[Index] Building HNSW binary (bit, float32 rerank) ...")
            idx_hnsw_bin, t_hnsw_bin = build_index(
                conn, tbl_vector, "vector", "binary_hnsw_rerank", dim
            )
            print("[Index] Building HNSW binary (bit, float16 rerank) ...")
            idx_hnsw_bin_half, t_hnsw_bin_half = build_index(
                conn, tbl_half, "halfvec", "binary_hnsw_rerank", dim
            )
            if args.enable_mean_binarization:
                print("[Index] Building HNSW binary MEAN (bit, float32 rerank) ...")
                idx_hnsw_bin_mean, t_hnsw_bin_mean = build_index(
                    conn, tbl_vector, "vector", "binary_hnsw_rerank", dim, use_mean_bin=True
                )
                print("[Index] Building HNSW binary MEAN (bit, float16 rerank) ...")
                idx_hnsw_bin_half_mean, t_hnsw_bin_half_mean = build_index(
                    conn, tbl_half, "halfvec", "binary_hnsw_rerank", dim, use_mean_bin=True
                )
            else:
                idx_hnsw_bin_mean, t_hnsw_bin_mean = None, 0.0
                idx_hnsw_bin_half_mean, t_hnsw_bin_half_mean = None, 0.0
            if args.enable_quasi_uint8:
                print("[Index] Building HNSW binary UINT8 (bit, float32 rerank) ...")
                idx_hnsw_bin_uint8, t_hnsw_bin_uint8 = build_index(
                    conn, tbl_vector, "vector", "binary_hnsw_rerank", dim, use_uint8_bin=True
                )
                print("[Index] Building HNSW binary UINT8 (bit, float16 rerank) ...")
                idx_hnsw_bin_half_uint8, t_hnsw_bin_half_uint8 = build_index(
                    conn, tbl_half, "halfvec", "binary_hnsw_rerank", dim, use_uint8_bin=True
                )
            else:
                idx_hnsw_bin_uint8, t_hnsw_bin_uint8 = None, 0.0
                idx_hnsw_bin_half_uint8, t_hnsw_bin_half_uint8 = None, 0.0
            if args.enable_quasi_uint4:
                print("[Index] Building HNSW binary UINT4 (bit, float32 rerank) ...")
                idx_hnsw_bin_uint4, t_hnsw_bin_uint4 = build_index(
                    conn, tbl_vector, "vector", "binary_hnsw_rerank", dim, use_uint4_bin=True
                )
                print("[Index] Building HNSW binary UINT4 (bit, float16 rerank) ...")
                idx_hnsw_bin_half_uint4, t_hnsw_bin_half_uint4 = build_index(
                    conn, tbl_half, "halfvec", "binary_hnsw_rerank", dim, use_uint4_bin=True
                )
            else:
                idx_hnsw_bin_uint4, t_hnsw_bin_uint4 = None, 0.0
                idx_hnsw_bin_half_uint4, t_hnsw_bin_half_uint4 = None, 0.0
        else:
            idx_hnsw_bin, t_hnsw_bin = None, 0.0
            idx_hnsw_bin_half, t_hnsw_bin_half = None, 0.0
            idx_hnsw_bin_mean, t_hnsw_bin_mean = None, 0.0
            idx_hnsw_bin_half_mean, t_hnsw_bin_half_mean = None, 0.0
            idx_hnsw_bin_uint8, t_hnsw_bin_uint8 = None, 0.0
            idx_hnsw_bin_half_uint8, t_hnsw_bin_half_uint8 = None, 0.0
            idx_hnsw_bin_uint4, t_hnsw_bin_uint4 = None, 0.0
            idx_hnsw_bin_half_uint4, t_hnsw_bin_half_uint4 = None, 0.0

        # Binary IVFFlat indices
        if "binary-ivf" in benchmarks:
            print(
                f"[Index] Building IVFFlat binary (bit, float32 rerank, lists={IVF_LISTS}) ..."
            )
            idx_ivf_bin, t_ivf_bin = build_index(
                conn, tbl_vector, "vector", "binary_ivf_rerank", dim
            )
            print(
                f"[Index] Building IVFFlat binary (bit, float16 rerank, lists={IVF_LISTS}) ..."
            )
            idx_ivf_bin_half, t_ivf_bin_half = build_index(
                conn, tbl_half, "halfvec", "binary_ivf_rerank", dim
            )
            if args.enable_mean_binarization:
                print(
                    f"[Index] Building IVFFlat binary MEAN (bit, float32 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_mean, t_ivf_bin_mean = build_index(
                    conn, tbl_vector, "vector", "binary_ivf_rerank", dim, use_mean_bin=True
                )
                print(
                    f"[Index] Building IVFFlat binary MEAN (bit, float16 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_half_mean, t_ivf_bin_half_mean = build_index(
                    conn, tbl_half, "halfvec", "binary_ivf_rerank", dim, use_mean_bin=True
                )
            else:
                idx_ivf_bin_mean, t_ivf_bin_mean = None, 0.0
                idx_ivf_bin_half_mean, t_ivf_bin_half_mean = None, 0.0
            if args.enable_quasi_uint8:
                print(
                    f"[Index] Building IVFFlat binary UINT8 (bit, float32 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_uint8, t_ivf_bin_uint8 = build_index(
                    conn, tbl_vector, "vector", "binary_ivf_rerank", dim, use_uint8_bin=True
                )
                print(
                    f"[Index] Building IVFFlat binary UINT8 (bit, float16 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_half_uint8, t_ivf_bin_half_uint8 = build_index(
                    conn, tbl_half, "halfvec", "binary_ivf_rerank", dim, use_uint8_bin=True
                )
            else:
                idx_ivf_bin_uint8, t_ivf_bin_uint8 = None, 0.0
                idx_ivf_bin_half_uint8, t_ivf_bin_half_uint8 = None, 0.0
            if args.enable_quasi_uint4:
                print(
                    f"[Index] Building IVFFlat binary UINT4 (bit, float32 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_uint4, t_ivf_bin_uint4 = build_index(
                    conn, tbl_vector, "vector", "binary_ivf_rerank", dim, use_uint4_bin=True
                )
                print(
                    f"[Index] Building IVFFlat binary UINT4 (bit, float16 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_half_uint4, t_ivf_bin_half_uint4 = build_index(
                    conn, tbl_half, "halfvec", "binary_ivf_rerank", dim, use_uint4_bin=True
                )
            else:
                idx_ivf_bin_uint4, t_ivf_bin_uint4 = None, 0.0
                idx_ivf_bin_half_uint4, t_ivf_bin_half_uint4 = None, 0.0
        else:
            idx_ivf_bin, t_ivf_bin = None, 0.0
            idx_ivf_bin_half, t_ivf_bin_half = None, 0.0
            idx_ivf_bin_mean, t_ivf_bin_mean = None, 0.0
            idx_ivf_bin_half_mean, t_ivf_bin_half_mean = None, 0.0
            idx_ivf_bin_uint8, t_ivf_bin_uint8 = None, 0.0
            idx_ivf_bin_half_uint8, t_ivf_bin_half_uint8 = None, 0.0
            idx_ivf_bin_uint4, t_ivf_bin_uint4 = None, 0.0
            idx_ivf_bin_half_uint4, t_ivf_bin_half_uint4 = None, 0.0

        # Query each (conditionally based on selected benchmarks)
        # Enable explain_analyze only for 512-D queries
        do_explain = args.explain_analyze and dim == 512

        # VectorChord queries
        if "vchordrq" in benchmarks:
            print("[Query] VectorChord full precision...")
            lat_vchord_vec, rec_vchord_vec = query_index(
                conn, tbl_vector, "vector", "full", dim, query_trunc, baseline_ids
            )
            size_vchord_vec = get_index_size_mb(conn, idx_vchord_vec)

            print("[Query] VectorChord half precision...")
            lat_vchord_half, rec_vchord_half = query_index(
                conn, tbl_half, "halfvec", "half", dim, query_trunc, baseline_ids
            )
            size_vchord_half = get_index_size_mb(conn, idx_vchord_half)

        # IVFFlat queries
        if "ivfflat" in benchmarks:
            print(f"[Query] IVFFlat full precision (probes={IVF_PROBES})...")
            lat_ivf_vec, rec_ivf_vec = query_index(
                conn, tbl_vector, "vector", "ivf", dim, query_trunc, baseline_ids
            )
            size_ivf_vec = get_index_size_mb(conn, idx_ivf_vec)

            print(f"[Query] IVFFlat half precision (probes={IVF_PROBES})...")
            lat_ivf_half, rec_ivf_half = query_index(
                conn, tbl_half, "halfvec", "ivf", dim, query_trunc, baseline_ids
            )
            size_ivf_half = get_index_size_mb(conn, idx_ivf_half)

        # Binary HNSW queries
        if "binary-hnsw" in benchmarks:
            print(
                f"[Query] HNSW binary float32 rerank (ef={HNSW_EF_SEARCH}, {OVERFETCH_FACTOR}x overfetch)..."
            )
            lat_hnsw_bin, rec_hnsw_bin = query_index(
                conn,
                tbl_vector,
                "vector",
                "binary_hnsw_rerank",
                dim,
                query_trunc,
                baseline_ids,
                explain_analyze=do_explain,
            )
            size_hnsw_bin = get_index_size_mb(conn, idx_hnsw_bin)

            print(
                f"[Query] HNSW binary float16 rerank (ef={HNSW_EF_SEARCH}, {OVERFETCH_FACTOR}x overfetch)..."
            )
            lat_hnsw_bin_half, rec_hnsw_bin_half = query_index(
                conn,
                tbl_half,
                "halfvec",
                "binary_hnsw_rerank",
                dim,
                query_trunc,
                baseline_ids,
                explain_analyze=do_explain,
            )
            size_hnsw_bin_half = get_index_size_mb(conn, idx_hnsw_bin_half)

            if args.enable_mean_binarization:
                print(
                    f"[Query] HNSW binary MEAN float32 rerank (ef={HNSW_EF_SEARCH}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_hnsw_bin_mean, rec_hnsw_bin_mean = query_index(
                    conn,
                    tbl_vector,
                    "vector",
                    "binary_hnsw_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_mean_bin=True,
                    dimension_means=dimension_means_1024,
                )
                size_hnsw_bin_mean = get_index_size_mb(conn, idx_hnsw_bin_mean)

                print(
                    f"[Query] HNSW binary MEAN float16 rerank (ef={HNSW_EF_SEARCH}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_hnsw_bin_half_mean, rec_hnsw_bin_half_mean = query_index(
                    conn,
                    tbl_half,
                    "halfvec",
                    "binary_hnsw_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_mean_bin=True,
                    dimension_means=dimension_means_1024,
                )
                size_hnsw_bin_half_mean = get_index_size_mb(
                    conn, idx_hnsw_bin_half_mean
                )

            if args.enable_quasi_uint8:
                print(
                    f"[Query] HNSW binary UINT8 float32 rerank (ef={HNSW_EF_SEARCH}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_hnsw_bin_uint8, rec_hnsw_bin_uint8 = query_index(
                    conn,
                    tbl_vector,
                    "vector",
                    "binary_hnsw_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_uint8_bin=True,
                    uint8_thresholds=uint8_thresholds_1024,
                )
                size_hnsw_bin_uint8 = get_index_size_mb(conn, idx_hnsw_bin_uint8)

                print(
                    f"[Query] HNSW binary UINT8 float16 rerank (ef={HNSW_EF_SEARCH}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_hnsw_bin_half_uint8, rec_hnsw_bin_half_uint8 = query_index(
                    conn,
                    tbl_half,
                    "halfvec",
                    "binary_hnsw_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_uint8_bin=True,
                    uint8_thresholds=uint8_thresholds_1024,
                )
                size_hnsw_bin_half_uint8 = get_index_size_mb(
                    conn, idx_hnsw_bin_half_uint8
                )

            if args.enable_quasi_uint4:
                print(
                    f"[Query] HNSW binary UINT4 float32 rerank (ef={HNSW_EF_SEARCH}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_hnsw_bin_uint4, rec_hnsw_bin_uint4 = query_index(
                    conn,
                    tbl_vector,
                    "vector",
                    "binary_hnsw_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_uint4_bin=True,
                    uint4_thresholds=uint4_thresholds_1024,
                )
                size_hnsw_bin_uint4 = get_index_size_mb(conn, idx_hnsw_bin_uint4)

                print(
                    f"[Query] HNSW binary UINT4 float16 rerank (ef={HNSW_EF_SEARCH}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_hnsw_bin_half_uint4, rec_hnsw_bin_half_uint4 = query_index(
                    conn,
                    tbl_half,
                    "halfvec",
                    "binary_hnsw_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_uint4_bin=True,
                    uint4_thresholds=uint4_thresholds_1024,
                )
                size_hnsw_bin_half_uint4 = get_index_size_mb(
                    conn, idx_hnsw_bin_half_uint4
                )

        # Binary IVFFlat queries
        if "binary-ivf" in benchmarks:
            print(
                f"[Query] IVFFlat binary float32 rerank (probes={IVF_PROBES_BINARY}, {OVERFETCH_FACTOR}x overfetch)..."
            )
            lat_ivf_bin, rec_ivf_bin = query_index(
                conn,
                tbl_vector,
                "vector",
                "binary_ivf_rerank",
                dim,
                query_trunc,
                baseline_ids,
                explain_analyze=do_explain,
            )
            size_ivf_bin = get_index_size_mb(conn, idx_ivf_bin)

            print(
                f"[Query] IVFFlat binary float16 rerank (probes={IVF_PROBES_BINARY}, {OVERFETCH_FACTOR}x overfetch)..."
            )
            lat_ivf_bin_half, rec_ivf_bin_half = query_index(
                conn,
                tbl_half,
                "halfvec",
                "binary_ivf_rerank",
                dim,
                query_trunc,
                baseline_ids,
                explain_analyze=do_explain,
            )
            size_ivf_bin_half = get_index_size_mb(conn, idx_ivf_bin_half)

            if args.enable_mean_binarization:
                print(
                    f"[Query] IVFFlat binary MEAN float32 rerank (probes={IVF_PROBES_BINARY}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_ivf_bin_mean, rec_ivf_bin_mean = query_index(
                    conn,
                    tbl_vector,
                    "vector",
                    "binary_ivf_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_mean_bin=True,
                    dimension_means=dimension_means_1024,
                )
                size_ivf_bin_mean = get_index_size_mb(conn, idx_ivf_bin_mean)

                print(
                    f"[Query] IVFFlat binary MEAN float16 rerank (probes={IVF_PROBES_BINARY}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_ivf_bin_half_mean, rec_ivf_bin_half_mean = query_index(
                    conn,
                    tbl_half,
                    "halfvec",
                    "binary_ivf_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_mean_bin=True,
                    dimension_means=dimension_means_1024,
                )
                size_ivf_bin_half_mean = get_index_size_mb(conn, idx_ivf_bin_half_mean)

            if args.enable_quasi_uint8:
                print(
                    f"[Query] IVFFlat binary UINT8 float32 rerank (probes={IVF_PROBES_BINARY}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_ivf_bin_uint8, rec_ivf_bin_uint8 = query_index(
                    conn,
                    tbl_vector,
                    "vector",
                    "binary_ivf_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_uint8_bin=True,
                    uint8_thresholds=uint8_thresholds_1024,
                )
                size_ivf_bin_uint8 = get_index_size_mb(conn, idx_ivf_bin_uint8)

                print(
                    f"[Query] IVFFlat binary UINT8 float16 rerank (probes={IVF_PROBES_BINARY}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_ivf_bin_half_uint8, rec_ivf_bin_half_uint8 = query_index(
                    conn,
                    tbl_half,
                    "halfvec",
                    "binary_ivf_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_uint8_bin=True,
                    uint8_thresholds=uint8_thresholds_1024,
                )
                size_ivf_bin_half_uint8 = get_index_size_mb(
                    conn, idx_ivf_bin_half_uint8
                )

            if args.enable_quasi_uint4:
                print(
                    f"[Query] IVFFlat binary UINT4 float32 rerank (probes={IVF_PROBES_BINARY}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_ivf_bin_uint4, rec_ivf_bin_uint4 = query_index(
                    conn,
                    tbl_vector,
                    "vector",
                    "binary_ivf_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_uint4_bin=True,
                    uint4_thresholds=uint4_thresholds_1024,
                )
                size_ivf_bin_uint4 = get_index_size_mb(conn, idx_ivf_bin_uint4)

                print(
                    f"[Query] IVFFlat binary UINT4 float16 rerank (probes={IVF_PROBES_BINARY}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_ivf_bin_half_uint4, rec_ivf_bin_half_uint4 = query_index(
                    conn,
                    tbl_half,
                    "halfvec",
                    "binary_ivf_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_uint4_bin=True,
                    uint4_thresholds=uint4_thresholds_1024,
                )
                size_ivf_bin_half_uint4 = get_index_size_mb(
                    conn, idx_ivf_bin_half_uint4
                )

        # Binary exact queries
        if "binary-exact" in benchmarks:
            # Exact binary (no binary index) + rerank
            print(
                f"[Query] Exact binary float32 rerank ({OVERFETCH_FACTOR}x overfetch)..."
            )
            lat_bin_exact_vec, rec_bin_exact_vec = query_index(
                conn,
                tbl_vector,
                "vector",
                "binary_exact_rerank",
                dim,
                query_trunc,
                baseline_ids,
            )
            print(
                f"[Query] Exact binary float16 rerank ({OVERFETCH_FACTOR}x overfetch)..."
            )
            lat_bin_exact_half, rec_bin_exact_half = query_index(
                conn,
                tbl_half,
                "halfvec",
                "binary_exact_rerank",
                dim,
                query_trunc,
                baseline_ids,
            )

            # Keep exact-binary (1x) for reference (no rerank; pure Hamming)
            print(f"[Query] Exact binary float32 (1x, no rerank)...")
            lat_bin_exact_k_vec, rec_bin_exact_k_vec = query_index(
                conn,
                tbl_vector,
                "vector",
                "binary_exact_k",
                dim,
                query_trunc,
                baseline_ids,
            )
            print(f"[Query] Exact binary float16 (1x, no rerank)...")
            lat_bin_exact_k_half, rec_bin_exact_k_half = query_index(
                conn,
                tbl_half,
                "halfvec",
                "binary_exact_k",
                dim,
                query_trunc,
                baseline_ids,
            )

            if args.enable_mean_binarization:
                # Mean-based binary exact queries
                print(
                    f"[Query] Exact binary MEAN float32 rerank ({OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_bin_exact_vec_mean, rec_bin_exact_vec_mean = query_index(
                    conn,
                    tbl_vector,
                    "vector",
                    "binary_exact_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    use_mean_bin=True,
                    dimension_means=dimension_means_1024,
                )
                print(
                    f"[Query] Exact binary MEAN float16 rerank ({OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_bin_exact_half_mean, rec_bin_exact_half_mean = query_index(
                    conn,
                    tbl_half,
                    "halfvec",
                    "binary_exact_rerank",
                    dim,
                    query_trunc,
                    baseline_ids,
                    use_mean_bin=True,
                    dimension_means=dimension_means_1024,
                )

            if args.enable_quasi_uint8:
                # Quasi-uint8 binary exact queries (no rerank - pure Hamming on 8xN bits)
                print(f"[Query] Exact binary UINT8 float32 (1x, no rerank)...")
                lat_bin_exact_uint8_vec, rec_bin_exact_uint8_vec = query_index(
                    conn,
                    tbl_vector,
                    "vector",
                    "binary_exact_k",
                    dim,
                    query_trunc,
                    baseline_ids,
                    use_uint8_bin=True,
                    uint8_thresholds=uint8_thresholds_1024,
                )
                print(f"[Query] Exact binary UINT8 float16 (1x, no rerank)...")
                lat_bin_exact_uint8_half, rec_bin_exact_uint8_half = query_index(
                    conn,
                    tbl_half,
                    "halfvec",
                    "binary_exact_k",
                    dim,
                    query_trunc,
                    baseline_ids,
                    use_uint8_bin=True,
                    uint8_thresholds=uint8_thresholds_1024,
                )

            if args.enable_quasi_uint4:
                # Quasi-uint4 binary exact queries (no rerank - pure Hamming on 4xN bits)
                print(f"[Query] Exact binary UINT4 float32 (1x, no rerank)...")
                lat_bin_exact_uint4_vec, rec_bin_exact_uint4_vec = query_index(
                    conn,
                    tbl_vector,
                    "vector",
                    "binary_exact_k",
                    dim,
                    query_trunc,
                    baseline_ids,
                    use_uint4_bin=True,
                    uint4_thresholds=uint4_thresholds_1024,
                )
                print(f"[Query] Exact binary UINT4 float16 (1x, no rerank)...")
                lat_bin_exact_uint4_half, rec_bin_exact_uint4_half = query_index(
                    conn,
                    tbl_half,
                    "halfvec",
                    "binary_exact_k",
                    dim,
                    query_trunc,
                    baseline_ids,
                    use_uint4_bin=True,
                    uint4_thresholds=uint4_thresholds_1024,
                )

        # Exact (sequential) queries
        if "exact" in benchmarks:
            # Add back exact retrieval (sequential, no index)
            print("[Query] Exact float32 (sequential, no index)...")
            lat_exact_vec, rec_exact_vec = query_index(
                conn,
                tbl_vector,
                "vector",
                "exact",
                dim,
                query_trunc,
                baseline_ids,
                explain_analyze=do_explain,
            )
            print("[Query] Exact float16 (sequential, no index)...")
            lat_exact_half, rec_exact_half = query_index(
                conn,
                tbl_half,
                "halfvec",
                "exact",
                dim,
                query_trunc,
                baseline_ids,
                explain_analyze=do_explain,
            )

        # Calculate storage sizes
        storage_vec_mb = get_vector_storage_mb(num_vectors, dim, "float32")
        storage_half_mb = get_vector_storage_mb(num_vectors, dim, "float16")

        # Calculate binary-only storage sizes (for exact queries that don't use float columns)
        storage_uint8_mb = get_binary_storage_mb(
            num_vectors, dim, bits_per_dim=8
        )  # uint8: 8 bits per dimension
        storage_uint4_mb = get_binary_storage_mb(
            num_vectors, dim, bits_per_dim=4
        )  # uint4: 4 bits per dimension

        # Collect results (conditionally based on which benchmarks ran)
        if "vchordrq" in benchmarks:
            results.append(
                {
                    "dim": dim,
                    "storage": "float32",
                    "index": "vchordrq",
                    "lat_ms": lat_vchord_vec * 1000,
                    "recall": rec_vchord_vec,
                    "build_s": t_vchord_vec,
                    "index_mb": size_vchord_vec,
                    "storage_mb": storage_vec_mb,
                }
            )
            results.append(
                {
                    "dim": dim,
                    "storage": "float16",
                    "index": "vchordrq",
                    "lat_ms": lat_vchord_half * 1000,
                    "recall": rec_vchord_half,
                    "build_s": t_vchord_half,
                    "index_mb": size_vchord_half,
                    "storage_mb": storage_half_mb,
                }
            )

        if "ivfflat" in benchmarks:
            results.append(
                {
                    "dim": dim,
                    "storage": "float32",
                    "index": f"ivfflat(L{IVF_LISTS},P{IVF_PROBES})",
                    "lat_ms": lat_ivf_vec * 1000,
                    "recall": rec_ivf_vec,
                    "build_s": t_ivf_vec,
                    "index_mb": size_ivf_vec,
                    "storage_mb": storage_vec_mb,
                }
            )
            results.append(
                {
                    "dim": dim,
                    "storage": "float16",
                    "index": f"ivfflat(L{IVF_LISTS},P{IVF_PROBES})",
                    "lat_ms": lat_ivf_half * 1000,
                    "recall": rec_ivf_half,
                    "build_s": t_ivf_half,
                    "index_mb": size_ivf_half,
                    "storage_mb": storage_half_mb,
                }
            )

        if "binary-hnsw" in benchmarks:
            results.append(
                {
                    "dim": dim,
                    "storage": "float32",
                    "index": f"hnsw+binary(ef{HNSW_EF_SEARCH},{OVERFETCH_FACTOR}x)",
                    "lat_ms": lat_hnsw_bin * 1000,
                    "recall": rec_hnsw_bin,
                    "build_s": t_hnsw_bin,
                    "index_mb": size_hnsw_bin,
                    "storage_mb": storage_vec_mb,
                }
            )
            results.append(
                {
                    "dim": dim,
                    "storage": "float16",
                    "index": f"hnsw+binary(ef{HNSW_EF_SEARCH},{OVERFETCH_FACTOR}x)",
                    "lat_ms": lat_hnsw_bin_half * 1000,
                    "recall": rec_hnsw_bin_half,
                    "build_s": t_hnsw_bin_half,
                    "index_mb": size_hnsw_bin_half,
                    "storage_mb": storage_half_mb,
                }
            )

            if args.enable_mean_binarization:
                results.append(
                    {
                        "dim": dim,
                        "storage": "float32",
                        "index": f"hnsw+binary-mean(ef{HNSW_EF_SEARCH},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_hnsw_bin_mean * 1000,
                        "recall": rec_hnsw_bin_mean,
                        "build_s": t_hnsw_bin_mean,
                        "index_mb": size_hnsw_bin_mean,
                        "storage_mb": storage_vec_mb,
                    }
                )
                results.append(
                    {
                        "dim": dim,
                        "storage": "float16",
                        "index": f"hnsw+binary-mean(ef{HNSW_EF_SEARCH},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_hnsw_bin_half_mean * 1000,
                        "recall": rec_hnsw_bin_half_mean,
                        "build_s": t_hnsw_bin_half_mean,
                        "index_mb": size_hnsw_bin_half_mean,
                        "storage_mb": storage_half_mb,
                    }
                )

            if args.enable_quasi_uint8:
                results.append(
                    {
                        "dim": dim,
                        "storage": "float32",
                        "index": f"hnsw+binary-uint8(ef{HNSW_EF_SEARCH},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_hnsw_bin_uint8 * 1000,
                        "recall": rec_hnsw_bin_uint8,
                        "build_s": t_hnsw_bin_uint8,
                        "index_mb": size_hnsw_bin_uint8,
                        "storage_mb": storage_vec_mb,
                    }
                )
                results.append(
                    {
                        "dim": dim,
                        "storage": "float16",
                        "index": f"hnsw+binary-uint8(ef{HNSW_EF_SEARCH},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_hnsw_bin_half_uint8 * 1000,
                        "recall": rec_hnsw_bin_half_uint8,
                        "build_s": t_hnsw_bin_half_uint8,
                        "index_mb": size_hnsw_bin_half_uint8,
                        "storage_mb": storage_half_mb,
                    }
                )

            if args.enable_quasi_uint4:
                results.append(
                    {
                        "dim": dim,
                        "storage": "float32",
                        "index": f"hnsw+binary-uint4(ef{HNSW_EF_SEARCH},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_hnsw_bin_uint4 * 1000,
                        "recall": rec_hnsw_bin_uint4,
                        "build_s": t_hnsw_bin_uint4,
                        "index_mb": size_hnsw_bin_uint4,
                        "storage_mb": storage_vec_mb,
                    }
                )
                results.append(
                    {
                        "dim": dim,
                        "storage": "float16",
                        "index": f"hnsw+binary-uint4(ef{HNSW_EF_SEARCH},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_hnsw_bin_half_uint4 * 1000,
                        "recall": rec_hnsw_bin_half_uint4,
                        "build_s": t_hnsw_bin_half_uint4,
                        "index_mb": size_hnsw_bin_half_uint4,
                        "storage_mb": storage_half_mb,
                    }
                )

        if "binary-ivf" in benchmarks:
            results.append(
                {
                    "dim": dim,
                    "storage": "float32",
                    "index": f"ivf+binary(L{IVF_LISTS},P{IVF_PROBES_BINARY},{OVERFETCH_FACTOR}x)",
                    "lat_ms": lat_ivf_bin * 1000,
                    "recall": rec_ivf_bin,
                    "build_s": t_ivf_bin,
                    "index_mb": size_ivf_bin,
                    "storage_mb": storage_vec_mb,
                }
            )
            results.append(
                {
                    "dim": dim,
                    "storage": "float16",
                    "index": f"ivf+binary(L{IVF_LISTS},P{IVF_PROBES_BINARY},{OVERFETCH_FACTOR}x)",
                    "lat_ms": lat_ivf_bin_half * 1000,
                    "recall": rec_ivf_bin_half,
                    "build_s": t_ivf_bin_half,
                    "index_mb": size_ivf_bin_half,
                    "storage_mb": storage_half_mb,
                }
            )

            if args.enable_mean_binarization:
                results.append(
                    {
                        "dim": dim,
                        "storage": "float32",
                        "index": f"ivf+binary-mean(L{IVF_LISTS},P{IVF_PROBES_BINARY},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_ivf_bin_mean * 1000,
                        "recall": rec_ivf_bin_mean,
                        "build_s": t_ivf_bin_mean,
                        "index_mb": size_ivf_bin_mean,
                        "storage_mb": storage_vec_mb,
                    }
                )
                results.append(
                    {
                        "dim": dim,
                        "storage": "float16",
                        "index": f"ivf+binary-mean(L{IVF_LISTS},P{IVF_PROBES_BINARY},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_ivf_bin_half_mean * 1000,
                        "recall": rec_ivf_bin_half_mean,
                        "build_s": t_ivf_bin_half_mean,
                        "index_mb": size_ivf_bin_half_mean,
                        "storage_mb": storage_half_mb,
                    }
                )

            if args.enable_quasi_uint8:
                results.append(
                    {
                        "dim": dim,
                        "storage": "float32",
                        "index": f"ivf+binary-uint8(L{IVF_LISTS},P{IVF_PROBES_BINARY},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_ivf_bin_uint8 * 1000,
                        "recall": rec_ivf_bin_uint8,
                        "build_s": t_ivf_bin_uint8,
                        "index_mb": size_ivf_bin_uint8,
                        "storage_mb": storage_vec_mb,
                    }
                )
                results.append(
                    {
                        "dim": dim,
                        "storage": "float16",
                        "index": f"ivf+binary-uint8(L{IVF_LISTS},P{IVF_PROBES_BINARY},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_ivf_bin_half_uint8 * 1000,
                        "recall": rec_ivf_bin_half_uint8,
                        "build_s": t_ivf_bin_half_uint8,
                        "index_mb": size_ivf_bin_half_uint8,
                        "storage_mb": storage_half_mb,
                    }
                )

            if args.enable_quasi_uint4:
                results.append(
                    {
                        "dim": dim,
                        "storage": "float32",
                        "index": f"ivf+binary-uint4(L{IVF_LISTS},P{IVF_PROBES_BINARY},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_ivf_bin_uint4 * 1000,
                        "recall": rec_ivf_bin_uint4,
                        "build_s": t_ivf_bin_uint4,
                        "index_mb": size_ivf_bin_uint4,
                        "storage_mb": storage_vec_mb,
                    }
                )
                results.append(
                    {
                        "dim": dim,
                        "storage": "float16",
                        "index": f"ivf+binary-uint4(L{IVF_LISTS},P{IVF_PROBES_BINARY},{OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_ivf_bin_half_uint4 * 1000,
                        "recall": rec_ivf_bin_half_uint4,
                        "build_s": t_ivf_bin_half_uint4,
                        "index_mb": size_ivf_bin_half_uint4,
                        "storage_mb": storage_half_mb,
                    }
                )

        if "binary-exact" in benchmarks:
            results.append(
                {
                    "dim": dim,
                    "storage": "float32",
                    "index": f"exact-binary({OVERFETCH_FACTOR}x)",
                    "lat_ms": lat_bin_exact_vec * 1000,
                    "recall": rec_bin_exact_vec,
                    "build_s": 0.0,
                    "index_mb": 0.0,
                    "storage_mb": storage_vec_mb,
                }
            )
            results.append(
                {
                    "dim": dim,
                    "storage": "float16",
                    "index": f"exact-binary({OVERFETCH_FACTOR}x)",
                    "lat_ms": lat_bin_exact_half * 1000,
                    "recall": rec_bin_exact_half,
                    "build_s": 0.0,
                    "index_mb": 0.0,
                    "storage_mb": storage_half_mb,
                }
            )
            results.append(
                {
                    "dim": dim,
                    "storage": "float32",
                    "index": "exact-binary(1x)",
                    "lat_ms": lat_bin_exact_k_vec * 1000,
                    "recall": rec_bin_exact_k_vec,
                    "build_s": 0.0,
                    "index_mb": 0.0,
                    "storage_mb": storage_vec_mb,
                }
            )
            results.append(
                {
                    "dim": dim,
                    "storage": "float16",
                    "index": "exact-binary(1x)",
                    "lat_ms": lat_bin_exact_k_half * 1000,
                    "recall": rec_bin_exact_k_half,
                    "build_s": 0.0,
                    "index_mb": 0.0,
                    "storage_mb": storage_half_mb,
                }
            )

            if args.enable_mean_binarization:
                results.append(
                    {
                        "dim": dim,
                        "storage": "float32",
                        "index": f"exact-binary-mean({OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_bin_exact_vec_mean * 1000,
                        "recall": rec_bin_exact_vec_mean,
                        "build_s": 0.0,
                        "index_mb": 0.0,
                        "storage_mb": storage_vec_mb,
                    }
                )
                results.append(
                    {
                        "dim": dim,
                        "storage": "float16",
                        "index": f"exact-binary-mean({OVERFETCH_FACTOR}x)",
                        "lat_ms": lat_bin_exact_half_mean * 1000,
                        "recall": rec_bin_exact_half_mean,
                        "build_s": 0.0,
                        "index_mb": 0.0,
                        "storage_mb": storage_half_mb,
                    }
                )

            if args.enable_quasi_uint8:
                # Note: exact-binary-uint8(1x) only uses the binary column, not float columns
                # So we use storage_uint8_mb which reflects only the binary column size
                results.append(
                    {
                        "dim": dim,
                        "storage": "binary-uint8",
                        "index": "exact-binary-uint8(1x)",
                        "lat_ms": lat_bin_exact_uint8_vec * 1000,
                        "recall": rec_bin_exact_uint8_vec,
                        "build_s": 0.0,
                        "index_mb": 0.0,
                        "storage_mb": storage_uint8_mb,
                    }
                )
                results.append(
                    {
                        "dim": dim,
                        "storage": "binary-uint8",
                        "index": "exact-binary-uint8(1x)",
                        "lat_ms": lat_bin_exact_uint8_half * 1000,
                        "recall": rec_bin_exact_uint8_half,
                        "build_s": 0.0,
                        "index_mb": 0.0,
                        "storage_mb": storage_uint8_mb,
                    }
                )

            if args.enable_quasi_uint4:
                # Note: exact-binary-uint4(1x) only uses the binary column, not float columns
                # So we use storage_uint4_mb which reflects only the binary column size
                results.append(
                    {
                        "dim": dim,
                        "storage": "binary-uint4",
                        "index": "exact-binary-uint4(1x)",
                        "lat_ms": lat_bin_exact_uint4_vec * 1000,
                        "recall": rec_bin_exact_uint4_vec,
                        "build_s": 0.0,
                        "index_mb": 0.0,
                        "storage_mb": storage_uint4_mb,
                    }
                )
                results.append(
                    {
                        "dim": dim,
                        "storage": "binary-uint4",
                        "index": "exact-binary-uint4(1x)",
                        "lat_ms": lat_bin_exact_uint4_half * 1000,
                        "recall": rec_bin_exact_uint4_half,
                        "build_s": 0.0,
                        "index_mb": 0.0,
                        "storage_mb": storage_uint4_mb,
                    }
                )

        if "exact" in benchmarks:
            results.append(
                {
                    "dim": dim,
                    "storage": "float32",
                    "index": "exact",
                    "lat_ms": lat_exact_vec * 1000,
                    "recall": rec_exact_vec,
                    "build_s": 0.0,
                    "index_mb": 0.0,
                    "storage_mb": storage_vec_mb,
                }
            )
            results.append(
                {
                    "dim": dim,
                    "storage": "float16",
                    "index": "exact",
                    "lat_ms": lat_exact_half * 1000,
                    "recall": rec_exact_half,
                    "build_s": 0.0,
                    "index_mb": 0.0,
                    "storage_mb": storage_half_mb,
                }
            )

    conn.close()

    # Print summary with pretty formatting
    print("\n" + "=" * 140)
    print("RETRIEVAL BENCHMARK RESULTS".center(140))
    print("=" * 140)

    # Column headers
    headers = [
        "Dim",
        "Storage",
        "Index",
        "Latency",
        "Recall",
        "Build",
        "Storage",
        "Index",
    ]
    units = ["", "", "", "(ms)", "(%)", "(s)", "(MB)", "(MB)"]

    # Calculate column widths
    col_widths = {
        "dim": 5,
        "storage": 8,
        "index": 35,
        "lat_ms": 10,
        "recall": 8,
        "build_s": 8,
        "storage_mb": 10,
        "index_mb": 10,
    }

    # Print header
    print(
        f"| {'Dim':<{col_widths['dim']}} "
        f"| {'Storage':<{col_widths['storage']}} "
        f"| {'Index':<{col_widths['index']}} "
        f"| {'Latency (ms)':<{col_widths['lat_ms']}} "
        f"| {'Recall (%)':<{col_widths['recall']}} "
        f"| {'Build (s)':<{col_widths['build_s']}} "
        f"| {'Storage (MB)':<{col_widths['storage_mb']}} "
        f"| {'Index (MB)':<{col_widths['index_mb']}} |"
    )

    print("||")

    # Print rows grouped by dimension
    current_dim = None
    for r in results:
        if current_dim != r["dim"]:
            if current_dim is not None:
                print("||")
            current_dim = r["dim"]

        print(
            f"| {r['dim']:<{col_widths['dim']}} "
            f"| {r['storage']:<{col_widths['storage']}} "
            f"| {r['index']:<{col_widths['index']}} "
            f"| {r['lat_ms']:>{col_widths['lat_ms']}.2f} "
            f"| {r['recall'] * 100:>{col_widths['recall']}.1f} "
            f"| {r['build_s']:>{col_widths['build_s']}.2f} "
            f"| {r['storage_mb']:>{col_widths['storage_mb']}.1f} "
            f"| {r['index_mb']:>{col_widths['index_mb']}.1f} |"
        )

    print("=" * 140)


if __name__ == "__main__":
    main()
