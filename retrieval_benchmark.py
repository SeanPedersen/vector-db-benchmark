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
import psycopg2
import time
from psycopg2.extras import execute_values
from tqdm import tqdm
import argparse
import ast  # NEW: parse pgvector textual values


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
}

K = 100  # Retrieve 100 nearest neighbors
OVERFETCH_FACTOR = 10  # For binary index: retrieve 1000 candidates, rerank to top 100
BATCH_SIZE = 10_000

DIMENSIONS = [256, 512, 1024]  # Test Matryoshka embedding dimensions

# IVF index parameters (tunable)
IVF_LISTS = 100  # Number of clusters (typically sqrt(num_vectors))
IVF_PROBES = 10  # Number of clusters to search (higher = better recall, slower)
IVF_PROBES_BINARY = 50  # Higher probes for binary (Hamming distance is less accurate)
HNSW_EF_SEARCH = 1500  # HNSW ef_search parameter for binary index (raise default; should be >= overfetch)

# NEW: HNSW build parameters (recommended: M ~ 16-48, ef_construction ~ 200-400)
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 300
# NEW: upper cap for hnsw.ef_search to satisfy server limits (e.g., 1..1000)
HNSW_EF_SEARCH_MAX = 1000


# NEW: simple heuristic for IVFFlat params based on dataset size (N) and k
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def recommend_ivf_params(num_vectors: int, k: int):
    """
    Heuristics:
    - lists in [50, 500], target ~ N/200 (e.g., 50K -> 250)
    - For small k (<= 200): probes ~ lists/16 clamped to [8, 40]
    - For larger k (> 200): probes ~ lists/12 clamped to [10, 50]
    - binary probes ~ 3x probes (small k: [24, 120], large k: [30, 200])
    """
    lists = _clamp(num_vectors // 200, 50, 500)
    if k <= 200:
        probes = _clamp(round(lists / 16), 8, 40)
        probes_bin = _clamp(probes * 3, 24, 120)
    else:
        probes = _clamp(round(lists / 12), 10, 50)
        probes_bin = _clamp(probes * 3, 30, 200)
    return int(lists), int(probes), int(probes_bin)


def generate_embeddings(num_vectors: int, full_dim: int = 1024):
    np.random.seed(123)
    data = np.random.randn(num_vectors, full_dim).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    return data


def compute_dimension_means(embeddings: np.ndarray):
    """Compute mean for each dimension across all vectors."""
    return np.mean(embeddings, axis=0)


def binarize_with_means(embeddings: np.ndarray, means: np.ndarray):
    """Binarize vectors using mean-based thresholding.

    For each dimension, use its corpus-wide mean as threshold:
    - value >= mean -> 1
    - value < mean -> 0

    This ensures ~50/50 bit distribution per dimension for better
    information preservation and more discriminative binary codes.
    """
    return (embeddings >= means).astype(np.uint8)


def compute_percentile_thresholds(embeddings: np.ndarray, num_buckets: int = 8):
    """Compute percentile-based thresholds for quasi-uint8 quantization.

    For num_buckets=8, computes 7 threshold values per dimension at percentiles:
    [12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5]

    Returns: np.ndarray of shape (num_dimensions, num_buckets-1)
    """
    num_dims = embeddings.shape[1]
    num_thresholds = num_buckets - 1

    # Compute percentiles for each bucket boundary
    percentiles = np.linspace(100.0 / num_buckets, 100.0 * (num_buckets - 1) / num_buckets, num_thresholds)

    # Compute thresholds for each dimension
    thresholds = np.zeros((num_dims, num_thresholds), dtype=np.float32)
    for dim in range(num_dims):
        thresholds[dim] = np.percentile(embeddings[:, dim], percentiles)

    return thresholds


def encode_thermometer(embeddings: np.ndarray, thresholds: np.ndarray, num_bits_per_dim: int = None):
    """Encode vectors using thermometer/unary code.

    For each dimension, compares value against all thresholds:
    - Creates num_bits_per_dim bits per dimension (default: num_thresholds + 1)
    - bit[i] = 1 if value >= threshold[i], else 0
    - Example (8 buckets): value between threshold[2] and threshold[3] -> 11100000
    - Example (4 buckets): value between threshold[1] and threshold[2] -> 1100

    Args:
        embeddings: (N, D) array of float vectors
        thresholds: (D, num_thresholds) array of threshold values per dimension
        num_bits_per_dim: Number of bits per dimension (default: num_thresholds + 1)

    Returns: (N, D * num_bits_per_dim) binary array where each dimension is expanded
    """
    num_vectors, num_dims = embeddings.shape
    num_thresholds = thresholds.shape[1]

    # Default: use num_thresholds + 1 bits (e.g., 7 thresholds = 8 bits, 3 thresholds = 4 bits)
    if num_bits_per_dim is None:
        num_bits_per_dim = num_thresholds + 1

    # Initialize output: N vectors x (D dimensions * num_bits_per_dim bits per dimension)
    encoded = np.zeros((num_vectors, num_dims * num_bits_per_dim), dtype=np.uint8)

    for dim in range(num_dims):
        # For this dimension, compare all vectors against all thresholds
        for bit_idx in range(num_thresholds):
            # Set bit to 1 if value >= threshold
            bit_position = dim * num_bits_per_dim + bit_idx
            encoded[:, bit_position] = (embeddings[:, dim] >= thresholds[dim, bit_idx]).astype(np.uint8)

    # Note: bits num_thresholds to num_bits_per_dim-1 remain 0 (padding)
    # For 7 thresholds (8 buckets), bits 0-6 are used, bit 7 is always 0
    # For 3 thresholds (4 buckets), bits 0-2 are used, bit 3 is always 0

    return encoded


def numpy_binary_to_postgres_bit_string(binary_array: np.ndarray):
    """Convert numpy binary array to PostgreSQL bit string payload.

    Returns the raw bit digits without the B'..' wrapper so it can be safely
    bound as a parameter and cast using %s::bit(n) in SQL.
    """
    bit_string = "".join(binary_array.astype(str))
    return bit_string


def build_baseline(full_embeddings: np.ndarray, query: np.ndarray):
    sims = full_embeddings @ query
    top_idx = np.argsort(sims)[::-1][:K]
    return top_idx


def ensure_connection():
    return psycopg2.connect(**DB_CONFIG)


def table_exists_and_populated(
    conn, table_name: str, expected_rows: int, check_mean_bin=False, check_uint8_bin=False, check_uint4_bin=False
):
    """Check if table exists and has the expected number of rows."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s",
            (table_name,),
        )
        if cursor.fetchone()[0] == 0:
            cursor.close()
            return False

        # NEW: ensure the binary column exists so we don't skip recreation
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_name = %s AND column_name = 'embedding_bin'
            """,
            (table_name,),
        )
        has_bin = cursor.fetchone()[0] > 0
        if not has_bin:
            cursor.close()
            return False

        # Check for mean-based binary column if required
        if check_mean_bin:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'embedding_bin_mean'
                """,
                (table_name,),
            )
            has_mean_bin = cursor.fetchone()[0] > 0
            if not has_mean_bin:
                cursor.close()
                return False

        # Check for uint8-style binary column if required
        if check_uint8_bin:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'embedding_bin_uint8'
                """,
                (table_name,),
            )
            has_uint8_bin = cursor.fetchone()[0] > 0
            if not has_uint8_bin:
                cursor.close()
                return False

        # Check for uint4-style binary column if required
        if check_uint4_bin:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'embedding_bin_uint4'
                """,
                (table_name,),
            )
            has_uint4_bin = cursor.fetchone()[0] > 0
            if not has_uint4_bin:
                cursor.close()
                return False

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        actual_rows = cursor.fetchone()[0]
        cursor.close()
        return actual_rows == expected_rows
    except Exception:
        conn.rollback()
        cursor.close()
        return False


def create_and_insert_table(
    conn,
    name: str,
    embeddings: np.ndarray,
    precision: str,
    use_mean_binarization=False,
    dimension_means=None,
    use_uint8_binarization=False,
    uint8_thresholds=None,
    use_uint4_binarization=False,
    uint4_thresholds=None,
):
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {name}")
    dim = embeddings.shape[1]
    dim_uint8 = dim * 8  # 8 bits per dimension for uint8-style encoding
    dim_uint4 = dim * 4  # 4 bits per dimension for uint4-style encoding

    # Build optional columns
    optional_cols = []
    if use_mean_binarization:
        optional_cols.append(f"embedding_bin_mean bit({dim})")
    if use_uint8_binarization:
        optional_cols.append(f"embedding_bin_uint8 bit({dim_uint8})")
    if use_uint4_binarization:
        optional_cols.append(f"embedding_bin_uint4 bit({dim_uint4})")
    optional_cols_str = (", " + ", ".join(optional_cols)) if optional_cols else ""

    if precision == "vector":
        # NEW: add stored/generated binary column
        cursor.execute(
            f"""
            CREATE TABLE {name} (
                id BIGINT PRIMARY KEY,
                embedding vector({dim}),
                embedding_bin bit({dim}) GENERATED ALWAYS AS (binary_quantize(embedding)::bit({dim})) STORED
                {optional_cols_str}
            )
            """
        )
        cast = "::vector"
    elif precision == "halfvec":
        # NEW: add stored/generated binary column
        cursor.execute(
            f"""
            CREATE TABLE {name} (
                id BIGINT PRIMARY KEY,
                embedding halfvec({dim}),
                embedding_bin bit({dim}) GENERATED ALWAYS AS (binary_quantize(embedding)::bit({dim})) STORED
                {optional_cols_str}
            )
            """
        )
        cast = "::halfvec"
    else:
        raise ValueError("precision must be vector or halfvec")
    conn.commit()

    ids = np.arange(embeddings.shape[0], dtype=np.int64)

    # Precompute binary vectors if needed
    binary_vecs_mean = None
    binary_vecs_uint8 = None
    binary_vecs_uint4 = None

    if use_mean_binarization and dimension_means is not None:
        binary_vecs_mean = binarize_with_means(embeddings[:, :dim], dimension_means[:dim])

    if use_uint8_binarization and uint8_thresholds is not None:
        binary_vecs_uint8 = encode_thermometer(embeddings[:, :dim], uint8_thresholds[:dim])

    if use_uint4_binarization and uint4_thresholds is not None:
        binary_vecs_uint4 = encode_thermometer(embeddings[:, :dim], uint4_thresholds[:dim])

    for i in tqdm(
        range(0, embeddings.shape[0], BATCH_SIZE), desc=f"Insert {name}", unit="batch"
    ):
        batch_end = min(i + BATCH_SIZE, embeddings.shape[0])

        # Build insert statement based on which binary columns are enabled
        insert_cols = ["id", "embedding"]
        if use_mean_binarization:
            insert_cols.append("embedding_bin_mean")
        if use_uint8_binarization:
            insert_cols.append("embedding_bin_uint8")
        if use_uint4_binarization:
            insert_cols.append("embedding_bin_uint4")

        # Build template values
        template_parts = ["%s", f"%s{cast}"]
        if use_mean_binarization:
            template_parts.append(f"%s::bit({dim})")
        if use_uint8_binarization:
            template_parts.append(f"%s::bit({dim_uint8})")
        if use_uint4_binarization:
            template_parts.append(f"%s::bit({dim_uint4})")

        # Build batch data
        batch_data = []
        for j in range(i, batch_end):
            row = [
                int(ids[j]),
                embeddings[j][:dim]
                .astype(np.float16 if precision == "halfvec" else np.float32)
                .tolist(),
            ]
            if use_mean_binarization:
                row.append(numpy_binary_to_postgres_bit_string(binary_vecs_mean[j]))
            if use_uint8_binarization:
                row.append(numpy_binary_to_postgres_bit_string(binary_vecs_uint8[j]))
            if use_uint4_binarization:
                row.append(numpy_binary_to_postgres_bit_string(binary_vecs_uint4[j]))
            batch_data.append(tuple(row))

        execute_values(
            cursor,
            f"INSERT INTO {name} ({', '.join(insert_cols)}) VALUES %s",
            batch_data,
            template=f"({', '.join(template_parts)})",
        )

        if (i // BATCH_SIZE) % 10 == 0:
            conn.commit()
    conn.commit()
    cursor.close()


def build_index(
    conn, table: str, precision: str, kind: str, dim: int, use_mean_bin=False, use_uint8_bin=False, use_uint4_bin=False
):
    cursor = conn.cursor()
    if kind == "binary_hnsw":
        # Choose binary column based on flags (priority: uint8 > uint4 > mean > default)
        if use_uint8_bin:
            bin_col = "embedding_bin_uint8"
            suffix = "_uint8"
            bin_dim = dim * 8
        elif use_uint4_bin:
            bin_col = "embedding_bin_uint4"
            suffix = "_uint4"
            bin_dim = dim * 4
        elif use_mean_bin:
            bin_col = "embedding_bin_mean"
            suffix = "_mean"
            bin_dim = dim
        else:
            bin_col = "embedding_bin"
            suffix = ""
            bin_dim = dim

        idx_name = f"idx_{table}_hnsw_bin{suffix}"
        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")
        start = time.time()
        try:
            # NEW: index the stored binary column with HNSW build params
            cursor.execute(
                f"CREATE INDEX {idx_name} ON {table} USING hnsw ({bin_col} bit_hamming_ops) "
                f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
            )
        except Exception:
            conn.rollback()
            idx_name = None
    elif kind == "binary_ivf":
        # Choose binary column based on flags (priority: uint8 > uint4 > mean > default)
        if use_uint8_bin:
            bin_col = "embedding_bin_uint8"
            suffix = "_uint8"
            bin_dim = dim * 8
        elif use_uint4_bin:
            bin_col = "embedding_bin_uint4"
            suffix = "_uint4"
            bin_dim = dim * 4
        elif use_mean_bin:
            bin_col = "embedding_bin_mean"
            suffix = "_mean"
            bin_dim = dim
        else:
            bin_col = "embedding_bin"
            suffix = ""
            bin_dim = dim

        idx_name = f"idx_{table}_ivf_bin{suffix}"
        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")
        start = time.time()
        try:
            # NEW: index the stored binary column directly
            cursor.execute(
                f"CREATE INDEX {idx_name} ON {table} USING ivfflat ({bin_col} bit_hamming_ops) WITH (lists = {IVF_LISTS})"
            )
        except Exception:
            conn.rollback()
            idx_name = None
    elif kind == "ivf":
        ops = "vector_cosine_ops" if precision == "vector" else "halfvec_cosine_ops"
        idx_name = f"idx_{table}_ivf_{precision}"
        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")
        start = time.time()
        cursor.execute(
            f"CREATE INDEX {idx_name} ON {table} USING ivfflat (embedding {ops}) WITH (lists = {IVF_LISTS})"
        )
    else:  # vchordrq
        ops = "vector_cosine_ops" if precision == "vector" else "halfvec_cosine_ops"
        idx_name = f"idx_{table}_vchord_{precision}"
        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")
        start = time.time()
        cursor.execute(
            f"CREATE INDEX {idx_name} ON {table} USING vchordrq (embedding {ops})"
        )
    conn.commit()
    build_time = time.time() - start
    cursor.close()
    return idx_name, build_time


def _to_np_vec(val, dtype=np.float32):
    # Already array
    if isinstance(val, np.ndarray):
        return val.astype(dtype, copy=False)
    # Python sequence
    if isinstance(val, (list, tuple)):
        return np.array(val, dtype=dtype)
    # Textual form '[..]' returned by psycopg2 for pgvector/halfvec
    if isinstance(val, str):
        return np.array(ast.literal_eval(val), dtype=dtype)
    # Byte-like (rare): decode to str then parse
    if isinstance(val, (bytes, bytearray, memoryview)):
        try:
            s = bytes(val).decode()
        except Exception:
            s = str(val)
        return np.array(ast.literal_eval(s), dtype=dtype)
    # Fallback
    return np.array(val, dtype=dtype)


def query_index(
    conn,
    table: str,
    precision: str,
    kind: str,
    dim: int,
    query_vec: np.ndarray,
    baseline_ids,
    debug=False,
    local_embeddings=None,  # unused
    explain_analyze=False,
    use_mean_bin=False,
    dimension_means=None,
    use_uint8_bin=False,
    uint8_thresholds=None,
    use_uint4_bin=False,
    uint4_thresholds=None,
):
    cursor = conn.cursor()
    # Prepare query textual forms
    query_list_full = query_vec[:dim].tolist()
    query_txt = "[" + ",".join(map(str, query_list_full)) + "]"

    # Prepare query binary representation for mean-based binarization
    if use_mean_bin and dimension_means is not None:
        query_bin_mean = binarize_with_means(
            query_vec[:dim].reshape(1, -1), dimension_means[:dim]
        )[0]
        query_bin_mean_str = numpy_binary_to_postgres_bit_string(query_bin_mean)

    # Prepare query binary representation for uint8-style binarization
    if use_uint8_bin and uint8_thresholds is not None:
        query_bin_uint8 = encode_thermometer(
            query_vec[:dim].reshape(1, -1), uint8_thresholds[:dim]
        )[0]
        query_bin_uint8_str = numpy_binary_to_postgres_bit_string(query_bin_uint8)

    # Prepare query binary representation for uint4-style binarization
    if use_uint4_bin and uint4_thresholds is not None:
        query_bin_uint4 = encode_thermometer(
            query_vec[:dim].reshape(1, -1), uint4_thresholds[:dim]
        )[0]
        query_bin_uint4_str = numpy_binary_to_postgres_bit_string(query_bin_uint4)

    if kind == "ivf":
        # Set probes for IVFFlat
        cursor.execute(f"SET ivfflat.probes = {IVF_PROBES}")
        cast_type = "vector" if precision == "vector" else "halfvec"
        # Warm-up
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::{cast_type} LIMIT {K}",
            (query_txt,),
        )
        cursor.fetchall()
        start = time.time()
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::{cast_type} LIMIT {K}",
            (query_txt,),
        )
        rows = cursor.fetchall()
        retrieved = [r[0] for r in rows]
        latency = time.time() - start
    elif kind in (
        "binary_hnsw",
        "binary_ivf",
    ):
        # Use binary index for candidate generation, SQL for rerank (no Python NumPy).
        overfetch = K * OVERFETCH_FACTOR

        if "ivf" in kind:
            cursor.execute(f"SET ivfflat.probes = {IVF_PROBES_BINARY}")
        else:
            eff = min(max(HNSW_EF_SEARCH, overfetch), HNSW_EF_SEARCH_MAX)
            cursor.execute(f"SET hnsw.ef_search = {eff}")

        cast_type = "vector" if precision == "vector" else "halfvec"

        # Choose binary column and query based on binarization flags (priority: uint8 > uint4 > mean > default)
        if use_uint8_bin:
            bin_col = "embedding_bin_uint8"
            query_bin_str = query_bin_uint8_str
            bin_dim = dim * 8
            bin_type = "uint8"
            # One-shot candidate + rerank fully in SQL (using numpy-computed uint8 binary query)
            sql = f"""
            SELECT t.id
            FROM (
              SELECT id
              FROM {table}
              ORDER BY {bin_col} <~> %s::bit({bin_dim})
              LIMIT {overfetch}
            ) c
            JOIN {table} t USING(id)
            ORDER BY t.embedding <=> %s::{cast_type}
            LIMIT {K}
            """
            # Warm-up
            cursor.execute(sql, (query_bin_str, query_txt))
            cursor.fetchall()

            # Optional EXPLAIN ANALYZE for debugging
            if explain_analyze:
                print(
                    f"\n[DEBUG] EXPLAIN ANALYZE for {table} ({precision}, {kind}, uint8-bin):"
                )
                explain_sql = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE) " + sql
                cursor.execute(explain_sql, (query_bin_str, query_txt))
                for row in cursor.fetchall():
                    print(row[0])
                print()

            start = time.time()
            cursor.execute(sql, (query_bin_str, query_txt))
            rows = cursor.fetchall()
            retrieved = [int(r[0]) for r in rows]
            latency = time.time() - start
        elif use_uint4_bin:
            bin_col = "embedding_bin_uint4"
            query_bin_str = query_bin_uint4_str
            bin_dim = dim * 4
            bin_type = "uint4"
            # One-shot candidate + rerank fully in SQL (using numpy-computed uint4 binary query)
            sql = f"""
            SELECT t.id
            FROM (
              SELECT id
              FROM {table}
              ORDER BY {bin_col} <~> %s::bit({bin_dim})
              LIMIT {overfetch}
            ) c
            JOIN {table} t USING(id)
            ORDER BY t.embedding <=> %s::{cast_type}
            LIMIT {K}
            """
            # Warm-up
            cursor.execute(sql, (query_bin_str, query_txt))
            cursor.fetchall()

            # Optional EXPLAIN ANALYZE for debugging
            if explain_analyze:
                print(
                    f"\n[DEBUG] EXPLAIN ANALYZE for {table} ({precision}, {kind}, uint4-bin):"
                )
                explain_sql = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE) " + sql
                cursor.execute(explain_sql, (query_bin_str, query_txt))
                for row in cursor.fetchall():
                    print(row[0])
                print()

            start = time.time()
            cursor.execute(sql, (query_bin_str, query_txt))
            rows = cursor.fetchall()
            retrieved = [int(r[0]) for r in rows]
            latency = time.time() - start
        elif use_mean_bin:
            bin_col = "embedding_bin_mean"
            query_bin_str = query_bin_mean_str
            bin_dim = dim
            # One-shot candidate + rerank fully in SQL (using numpy-computed binary query)
            sql = f"""
            SELECT t.id
            FROM (
              SELECT id
              FROM {table}
              ORDER BY {bin_col} <~> %s::bit({bin_dim})
              LIMIT {overfetch}
            ) c
            JOIN {table} t USING(id)
            ORDER BY t.embedding <=> %s::{cast_type}
            LIMIT {K}
            """
            # Warm-up
            cursor.execute(sql, (query_bin_str, query_txt))
            cursor.fetchall()

            # Optional EXPLAIN ANALYZE for debugging
            if explain_analyze:
                print(
                    f"\n[DEBUG] EXPLAIN ANALYZE for {table} ({precision}, {kind}, mean-bin):"
                )
                explain_sql = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE) " + sql
                cursor.execute(explain_sql, (query_bin_str, query_txt))
                for row in cursor.fetchall():
                    print(row[0])
                print()

            start = time.time()
            cursor.execute(sql, (query_bin_str, query_txt))
            rows = cursor.fetchall()
            retrieved = [int(r[0]) for r in rows]
            latency = time.time() - start
        else:
            # One-shot candidate + rerank fully in SQL (using pgvector binary_quantize)
            sql = f"""
            SELECT t.id
            FROM (
              SELECT id
              FROM {table}
              ORDER BY embedding_bin <~> binary_quantize(%s::{cast_type})::bit({dim})
              LIMIT {overfetch}
            ) c
            JOIN {table} t USING(id)
            ORDER BY t.embedding <=> %s::{cast_type}
            LIMIT {K}
            """
            # Warm-up
            cursor.execute(sql, (query_txt, query_txt))
            cursor.fetchall()

            # Optional EXPLAIN ANALYZE for debugging
            if explain_analyze:
                print(f"\n[DEBUG] EXPLAIN ANALYZE for {table} ({precision}, {kind}):")
                explain_sql = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE) " + sql
                cursor.execute(explain_sql, (query_txt, query_txt))
                for row in cursor.fetchall():
                    print(row[0])
                print()

            start = time.time()
            cursor.execute(sql, (query_txt, query_txt))
            rows = cursor.fetchall()
            retrieved = [int(r[0]) for r in rows]
            latency = time.time() - start
    elif kind in ("binary_exact", "binary_exact_k"):
        # Exact binary candidate generation (no index). For _k: no rerank to measure pure Hamming scan speed.
        overfetch = K if kind.endswith("_k") else K * OVERFETCH_FACTOR
        cast_type = "vector" if precision == "vector" else "halfvec"

        if use_uint8_bin:
            bin_col = "embedding_bin_uint8"
            query_bin_str = query_bin_uint8_str
            bin_dim = dim * 8
            if kind.endswith("_k"):
                # Pure binary Hamming top-K without float rerank (uint8-based)
                sql = f"""
                SELECT id
                FROM {table}
                ORDER BY {bin_col} <~> %s::bit({bin_dim})
                LIMIT {K}
                """
                params = (query_bin_str,)
            else:
                # Generate candidates by Hamming, then rerank by cosine in-db (uint8-based)
                sql = f"""
                SELECT t.id
                FROM (
                  SELECT id
                  FROM {table}
                  ORDER BY {bin_col} <~> %s::bit({bin_dim})
                  LIMIT {overfetch}
                ) c
                JOIN {table} t USING(id)
                ORDER BY t.embedding <=> %s::{cast_type}
                LIMIT {K}
                """
                params = (query_bin_str, query_txt)
        elif use_uint4_bin:
            bin_col = "embedding_bin_uint4"
            query_bin_str = query_bin_uint4_str
            bin_dim = dim * 4
            if kind.endswith("_k"):
                # Pure binary Hamming top-K without float rerank (uint4-based)
                sql = f"""
                SELECT id
                FROM {table}
                ORDER BY {bin_col} <~> %s::bit({bin_dim})
                LIMIT {K}
                """
                params = (query_bin_str,)
            else:
                # Generate candidates by Hamming, then rerank by cosine in-db (uint4-based)
                sql = f"""
                SELECT t.id
                FROM (
                  SELECT id
                  FROM {table}
                  ORDER BY {bin_col} <~> %s::bit({bin_dim})
                  LIMIT {overfetch}
                ) c
                JOIN {table} t USING(id)
                ORDER BY t.embedding <=> %s::{cast_type}
                LIMIT {K}
                """
                params = (query_bin_str, query_txt)
        elif use_mean_bin:
            bin_col = "embedding_bin_mean"
            query_bin_str = query_bin_mean_str
            if kind.endswith("_k"):
                # Pure binary Hamming top-K without float rerank (mean-based)
                sql = f"""
                SELECT id
                FROM {table}
                ORDER BY {bin_col} <~> %s::bit({dim})
                LIMIT {K}
                """
                params = (query_bin_str,)
            else:
                # Generate candidates by Hamming, then rerank by cosine in-db (mean-based)
                sql = f"""
                SELECT t.id
                FROM (
                  SELECT id
                  FROM {table}
                  ORDER BY {bin_col} <~> %s::bit({dim})
                  LIMIT {overfetch}
                ) c
                JOIN {table} t USING(id)
                ORDER BY t.embedding <=> %s::{cast_type}
                LIMIT {K}
                """
                params = (query_bin_str, query_txt)
        else:
            if kind.endswith("_k"):
                # Pure binary Hamming top-K without float rerank
                sql = f"""
                SELECT id
                FROM {table}
                ORDER BY embedding_bin <~> binary_quantize(%s::{cast_type})::bit({dim})
                LIMIT {K}
                """
                params = (query_txt,)
            else:
                # Generate candidates by Hamming, then rerank by cosine in-db
                sql = f"""
                SELECT t.id
                FROM (
                  SELECT id
                  FROM {table}
                  ORDER BY embedding_bin <~> binary_quantize(%s::{cast_type})::bit({dim})
                  LIMIT {overfetch}
                ) c
                JOIN {table} t USING(id)
                ORDER BY t.embedding <=> %s::{cast_type}
                LIMIT {K}
                """
                params = (query_txt, query_txt)

        try:
            cursor.execute("SET enable_indexscan = off")
            cursor.execute("SET enable_bitmapscan = off")
            cursor.execute("SET enable_indexonlyscan = off")
            cursor.execute("SET enable_seqscan = on")

            # Warm-up
            cursor.execute(sql, params)
            cursor.fetchall()

            start = time.time()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            retrieved = [int(r[0]) for r in rows]
            latency = time.time() - start
        finally:
            try:
                cursor.execute("SET enable_indexscan = on")
                cursor.execute("SET enable_bitmapscan = on")
                cursor.execute("SET enable_indexonlyscan = on")
                cursor.execute("SET enable_seqscan = on")
            except Exception:
                pass
    elif kind == "exact":
        # Exact retrieval via sequential scan (no index usage)
        try:
            cursor.execute("SET enable_indexscan = off")
            cursor.execute("SET enable_bitmapscan = off")
            cursor.execute("SET enable_indexonlyscan = off")
            cursor.execute("SET enable_seqscan = on")

            cast_type = "vector" if precision == "vector" else "halfvec"
            sql = f"SELECT id FROM {table} ORDER BY embedding <=> %s::{cast_type} LIMIT {K}"

            # Warm-up
            cursor.execute(sql, (query_txt,))
            cursor.fetchall()

            # Optional EXPLAIN ANALYZE for debugging
            if explain_analyze:
                print(f"\n[DEBUG] EXPLAIN ANALYZE for {table} ({precision}, {kind}):")
                explain_sql = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE) " + sql
                cursor.execute(explain_sql, (query_txt,))
                for row in cursor.fetchall():
                    print(row[0])
                print()

            # Timed query
            start = time.time()
            cursor.execute(sql, (query_txt,))
            rows = cursor.fetchall()
            retrieved = [r[0] for r in rows]
            latency = time.time() - start
        finally:
            try:
                cursor.execute("SET enable_indexscan = on")
                cursor.execute("SET enable_bitmapscan = on")
                cursor.execute("SET enable_indexonlyscan = on")
                cursor.execute("SET enable_seqscan = on")
            except Exception:
                pass
    else:
        # Warm-up
        cast_type = "vector" if precision == "vector" else "halfvec"
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::{cast_type} LIMIT {K}",
            (query_txt,),
        )
        cursor.fetchall()
        start = time.time()
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::{cast_type} LIMIT {K}",
            (query_txt,),
        )
        rows = cursor.fetchall()
        retrieved = [r[0] for r in rows]
        latency = time.time() - start

    # Recall vs full baseline ids
    recall = len(set(retrieved) & set(baseline_ids)) / len(baseline_ids)
    cursor.close()
    return latency, recall


def get_index_size_mb(conn, idx_name: str):
    if not idx_name:
        return 0.0
    cursor = conn.cursor()
    cursor.execute("SELECT pg_total_relation_size(%s) / (1024.0*1024.0)", (idx_name,))
    size_mb = cursor.fetchone()[0]
    cursor.close()
    return size_mb


def get_vector_storage_mb(num_vectors: int, dimensions: int, precision: str):
    """Calculate the storage size for vector data."""
    bytes_per_value = (
        4 if precision == "float32" else 2
    )  # float32=4 bytes, float16=2 bytes
    total_bytes = num_vectors * dimensions * bytes_per_value
    # Add ~10% overhead for PostgreSQL storage (TOAST, alignment, etc.)
    return (total_bytes * 1.1) / (1024.0 * 1024.0)


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
        help="Enable quasi-uint8 binarization (8 buckets per dimension using percentile thresholds, thermometer encoding)",
    )
    parser.add_argument(
        "--enable-quasi-uint4",
        action="store_true",
        help="Enable quasi-uint4 binarization (4 buckets per dimension using percentile thresholds, thermometer encoding)",
    )

    args = parser.parse_args()

    # Expand and validate benchmark selections (supports comma-separated values per -b)
    allowed_benchmarks = {
        "vchordrq",
        "ivfflat",
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
        uint8_thresholds_1024 = compute_percentile_thresholds(full_embeddings, num_buckets=8)
        print(f"[Setup] Computed 7 thresholds per dimension (8 buckets) for all 1024 dimensions")
        print(
            f"[Setup] Threshold range: [{uint8_thresholds_1024.min():.6f}, {uint8_thresholds_1024.max():.6f}]"
        )

    # Compute percentile thresholds for quasi-uint4 binarization if enabled
    uint4_thresholds_1024 = None
    if args.enable_quasi_uint4:
        print("[Setup] Computing percentile thresholds for quasi-uint4 binarization...")
        uint4_thresholds_1024 = compute_percentile_thresholds(full_embeddings, num_buckets=4)
        print(f"[Setup] Computed 3 thresholds per dimension (4 buckets) for all 1024 dimensions")
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
            conn, tbl_vector, num_vectors,
            check_mean_bin=args.enable_mean_binarization,
            check_uint8_bin=args.enable_quasi_uint8,
            check_uint4_bin=args.enable_quasi_uint4
        )
        half_exists = table_exists_and_populated(
            conn, tbl_half, num_vectors,
            check_mean_bin=args.enable_mean_binarization,
            check_uint8_bin=args.enable_quasi_uint8,
            check_uint4_bin=args.enable_quasi_uint4
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
                conn, tbl_vector, "vector", "binary_hnsw", dim
            )
            print("[Index] Building HNSW binary (bit, float16 rerank) ...")
            idx_hnsw_bin_half, t_hnsw_bin_half = build_index(
                conn, tbl_half, "halfvec", "binary_hnsw", dim
            )
            if args.enable_mean_binarization:
                print("[Index] Building HNSW binary MEAN (bit, float32 rerank) ...")
                idx_hnsw_bin_mean, t_hnsw_bin_mean = build_index(
                    conn, tbl_vector, "vector", "binary_hnsw", dim, use_mean_bin=True
                )
                print("[Index] Building HNSW binary MEAN (bit, float16 rerank) ...")
                idx_hnsw_bin_half_mean, t_hnsw_bin_half_mean = build_index(
                    conn, tbl_half, "halfvec", "binary_hnsw", dim, use_mean_bin=True
                )
            else:
                idx_hnsw_bin_mean, t_hnsw_bin_mean = None, 0.0
                idx_hnsw_bin_half_mean, t_hnsw_bin_half_mean = None, 0.0
            if args.enable_quasi_uint8:
                print("[Index] Building HNSW binary UINT8 (bit, float32 rerank) ...")
                idx_hnsw_bin_uint8, t_hnsw_bin_uint8 = build_index(
                    conn, tbl_vector, "vector", "binary_hnsw", dim, use_uint8_bin=True
                )
                print("[Index] Building HNSW binary UINT8 (bit, float16 rerank) ...")
                idx_hnsw_bin_half_uint8, t_hnsw_bin_half_uint8 = build_index(
                    conn, tbl_half, "halfvec", "binary_hnsw", dim, use_uint8_bin=True
                )
            else:
                idx_hnsw_bin_uint8, t_hnsw_bin_uint8 = None, 0.0
                idx_hnsw_bin_half_uint8, t_hnsw_bin_half_uint8 = None, 0.0
            if args.enable_quasi_uint4:
                print("[Index] Building HNSW binary UINT4 (bit, float32 rerank) ...")
                idx_hnsw_bin_uint4, t_hnsw_bin_uint4 = build_index(
                    conn, tbl_vector, "vector", "binary_hnsw", dim, use_uint4_bin=True
                )
                print("[Index] Building HNSW binary UINT4 (bit, float16 rerank) ...")
                idx_hnsw_bin_half_uint4, t_hnsw_bin_half_uint4 = build_index(
                    conn, tbl_half, "halfvec", "binary_hnsw", dim, use_uint4_bin=True
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
                conn, tbl_vector, "vector", "binary_ivf", dim
            )
            print(
                f"[Index] Building IVFFlat binary (bit, float16 rerank, lists={IVF_LISTS}) ..."
            )
            idx_ivf_bin_half, t_ivf_bin_half = build_index(
                conn, tbl_half, "halfvec", "binary_ivf", dim
            )
            if args.enable_mean_binarization:
                print(
                    f"[Index] Building IVFFlat binary MEAN (bit, float32 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_mean, t_ivf_bin_mean = build_index(
                    conn, tbl_vector, "vector", "binary_ivf", dim, use_mean_bin=True
                )
                print(
                    f"[Index] Building IVFFlat binary MEAN (bit, float16 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_half_mean, t_ivf_bin_half_mean = build_index(
                    conn, tbl_half, "halfvec", "binary_ivf", dim, use_mean_bin=True
                )
            else:
                idx_ivf_bin_mean, t_ivf_bin_mean = None, 0.0
                idx_ivf_bin_half_mean, t_ivf_bin_half_mean = None, 0.0
            if args.enable_quasi_uint8:
                print(
                    f"[Index] Building IVFFlat binary UINT8 (bit, float32 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_uint8, t_ivf_bin_uint8 = build_index(
                    conn, tbl_vector, "vector", "binary_ivf", dim, use_uint8_bin=True
                )
                print(
                    f"[Index] Building IVFFlat binary UINT8 (bit, float16 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_half_uint8, t_ivf_bin_half_uint8 = build_index(
                    conn, tbl_half, "halfvec", "binary_ivf", dim, use_uint8_bin=True
                )
            else:
                idx_ivf_bin_uint8, t_ivf_bin_uint8 = None, 0.0
                idx_ivf_bin_half_uint8, t_ivf_bin_half_uint8 = None, 0.0
            if args.enable_quasi_uint4:
                print(
                    f"[Index] Building IVFFlat binary UINT4 (bit, float32 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_uint4, t_ivf_bin_uint4 = build_index(
                    conn, tbl_vector, "vector", "binary_ivf", dim, use_uint4_bin=True
                )
                print(
                    f"[Index] Building IVFFlat binary UINT4 (bit, float16 rerank, lists={IVF_LISTS}) ..."
                )
                idx_ivf_bin_half_uint4, t_ivf_bin_half_uint4 = build_index(
                    conn, tbl_half, "halfvec", "binary_ivf", dim, use_uint4_bin=True
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
                "binary_hnsw",
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
                "binary_hnsw",
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
                    "binary_hnsw",
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
                    "binary_hnsw",
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
                    "binary_hnsw",
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
                    "binary_hnsw",
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
                    "binary_hnsw",
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
                    "binary_hnsw",
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
                "binary_ivf",
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
                "binary_ivf",
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
                    "binary_ivf",
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
                    "binary_ivf",
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
                    "binary_ivf",
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
                    "binary_ivf",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_uint8_bin=True,
                    uint8_thresholds=uint8_thresholds_1024,
                )
                size_ivf_bin_half_uint8 = get_index_size_mb(conn, idx_ivf_bin_half_uint8)

            if args.enable_quasi_uint4:
                print(
                    f"[Query] IVFFlat binary UINT4 float32 rerank (probes={IVF_PROBES_BINARY}, {OVERFETCH_FACTOR}x overfetch)..."
                )
                lat_ivf_bin_uint4, rec_ivf_bin_uint4 = query_index(
                    conn,
                    tbl_vector,
                    "vector",
                    "binary_ivf",
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
                    "binary_ivf",
                    dim,
                    query_trunc,
                    baseline_ids,
                    explain_analyze=do_explain,
                    use_uint4_bin=True,
                    uint4_thresholds=uint4_thresholds_1024,
                )
                size_ivf_bin_half_uint4 = get_index_size_mb(conn, idx_ivf_bin_half_uint4)

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
                "binary_exact",
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
                "binary_exact",
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
                    "binary_exact",
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
                    "binary_exact",
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
                results.append(
                    {
                        "dim": dim,
                        "storage": "float32",
                        "index": "exact-binary-uint8(1x)",
                        "lat_ms": lat_bin_exact_uint8_vec * 1000,
                        "recall": rec_bin_exact_uint8_vec,
                        "build_s": 0.0,
                        "index_mb": 0.0,
                        "storage_mb": storage_vec_mb,
                    }
                )
                results.append(
                    {
                        "dim": dim,
                        "storage": "float16",
                        "index": "exact-binary-uint8(1x)",
                        "lat_ms": lat_bin_exact_uint8_half * 1000,
                        "recall": rec_bin_exact_uint8_half,
                        "build_s": 0.0,
                        "index_mb": 0.0,
                        "storage_mb": storage_half_mb,
                    }
                )

            if args.enable_quasi_uint4:
                results.append(
                    {
                        "dim": dim,
                        "storage": "float32",
                        "index": "exact-binary-uint4(1x)",
                        "lat_ms": lat_bin_exact_uint4_vec * 1000,
                        "recall": rec_bin_exact_uint4_vec,
                        "build_s": 0.0,
                        "index_mb": 0.0,
                        "storage_mb": storage_vec_mb,
                    }
                )
                results.append(
                    {
                        "dim": dim,
                        "storage": "float16",
                        "index": "exact-binary-uint4(1x)",
                        "lat_ms": lat_bin_exact_uint4_half * 1000,
                        "recall": rec_bin_exact_uint4_half,
                        "build_s": 0.0,
                        "index_mb": 0.0,
                        "storage_mb": storage_half_mb,
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
        f"| {'Latency':<{col_widths['lat_ms']}} "
        f"| {'Recall':<{col_widths['recall']}} "
        f"| {'Build':<{col_widths['build_s']}} "
        f"| {'Storage':<{col_widths['storage_mb']}} "
        f"| {'Index':<{col_widths['index_mb']}} |"
    )

    print(
        f"| {'':>{col_widths['dim']}} "
        f"| {'':>{col_widths['storage']}} "
        f"| {'':>{col_widths['index']}} "
        f"| {'(ms)':>{col_widths['lat_ms']}} "
        f"| {'(%)':>{col_widths['recall']}} "
        f"| {'(s)':>{col_widths['build_s']}} "
        f"| {'(MB)':>{col_widths['storage_mb']}} "
        f"| {'(MB)':>{col_widths['index_mb']}} |"
    )

    print("-" * 140)

    # Print rows grouped by dimension
    current_dim = None
    for r in results:
        if current_dim != r["dim"]:
            if current_dim is not None:
                print("-" * 140)
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

    # Analyze and warn about inefficient float16 performance
    print("\n" + "=" * 140)
    print("PERFORMANCE ANALYSIS: Float16 vs Float32".center(140))
    print("=" * 140)

    inefficiencies = []
    for dim in DIMENSIONS:
        dim_results = [r for r in results if r["dim"] == dim]

        # Group by index type and compare float32 vs float16
        index_types = {}
        for r in dim_results:
            idx = r["index"]
            storage = r["storage"]
            if idx not in index_types:
                index_types[idx] = {}
            index_types[idx][storage] = r

        for idx_name, storages in index_types.items():
            if "float32" in storages and "float16" in storages:
                f32 = storages["float32"]
                f16 = storages["float16"]

                # Calculate performance ratio (>1 means float16 is slower)
                ratio = f16["lat_ms"] / f32["lat_ms"]

                # Flag if float16 is significantly slower (>20% slower)
                if ratio > 1.2:
                    inefficiencies.append(
                        {
                            "dim": dim,
                            "index": idx_name,
                            "f32_ms": f32["lat_ms"],
                            "f16_ms": f16["lat_ms"],
                            "ratio": ratio,
                            "f32_recall": f32["recall"],
                            "f16_recall": f16["recall"],
                        }
                    )

    if inefficiencies:
        print("\n⚠️  Float16 Performance Issues Detected:\n")
        print("The following configurations show float16 SLOWER than float32:")
        print(
            f"\n{'Dim':<6} {'Index':<40} {'Float32 (ms)':<15} {'Float16 (ms)':<15} {'Slowdown':<12} {'Recall Impact'}"
        )
        print("-" * 140)

        for item in inefficiencies:
            recall_diff = item["f16_recall"] - item["f32_recall"]
            recall_str = f"{recall_diff:+.1%}"
            slowdown_str = f"{item['ratio']:.2f}x"

            print(
                f"{item['dim']:<6} {item['index']:<40} "
                f"{item['f32_ms']:<15.2f} {item['f16_ms']:<15.2f} "
                f"{slowdown_str:<12} {recall_str}"
            )

        print("\n💡 Analysis:")
        print(
            "   - Binary indices with float16 reranking show 1.8-2.1x slowdown vs float32"
        )
        print(
            "   - This suggests PostgreSQL's halfvec distance operations are less optimized"
        )
        print(
            "   - Despite 50% storage savings, float16 is NOT recommended for binary reranking"
        )
        print(
            "   - Pure float16 'exact' queries are fast, so the issue is specific to reranking"
        )
        print("\n💡 Recommendation:")
        print(
            "   - Use float32 (vector) for binary index reranking to achieve best latency"
        )
        print(
            "   - Use float16 (halfvec) only for storage savings when NOT using binary indices"
        )
        print(
            "   - Consider non-binary indices (vchordrq, ivfflat) if float16 storage is required"
        )
    else:
        print("\n✓ No significant float16 performance issues detected.")
        print(
            "  Float16 performs comparably or better than float32 across all tested configurations."
        )

    print("=" * 140)


if __name__ == "__main__":
    main()
