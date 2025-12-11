"""Database operations for retrieval benchmark."""

import ast
import time
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

from config import (
    DB_CONFIG,
    BATCH_SIZE,
    IVF_LISTS,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    VCHORDRQ_LISTS,
    VCHORDRQ_SPHERICAL_CENTROIDS,
    VCHORDRQ_BUILD_THREADS,
)
from embeddings import (
    binarize_with_means,
    encode_thermometer,
    encode_one_hot,
    numpy_binary_to_postgres_bit_string,
)


def ensure_connection():
    """Create and return a database connection."""
    return psycopg2.connect(**DB_CONFIG)


def print_extension_versions(conn):
    """Print versions of installed vector extensions."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT extname, extversion
        FROM pg_extension
        WHERE extname IN ('vector', 'vectorscale', 'vchord')
        ORDER BY extname
    """)
    extensions = cursor.fetchall()
    cursor.close()

    if extensions:
        print("\n[Extensions] Installed versions:")
        for ext_name, ext_version in extensions:
            print(f"  - {ext_name}: {ext_version}")
    else:
        print("\n[Extensions] No vector extensions found")


def table_exists_and_populated(
    conn,
    table_name: str,
    expected_rows: int,
    check_mean_bin=False,
    check_uint8_bin=False,
    check_uint4_bin=False,
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

        # Ensure the binary column exists so we don't skip recreation
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
    encoding_type="thermometer",
):
    """Create a table and insert embeddings.

    DEPRECATED: Use create_and_insert_unified_table() for new unified table structure.
    This function is kept for backward compatibility with existing code.
    """
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
        binary_vecs_mean = binarize_with_means(
            embeddings[:, :dim], dimension_means[:dim]
        )

    # Select encoding function based on encoding_type parameter
    encode_fn = encode_thermometer if encoding_type == "thermometer" else encode_one_hot

    if use_uint8_binarization and uint8_thresholds is not None:
        binary_vecs_uint8 = encode_fn(embeddings[:, :dim], uint8_thresholds[:dim])

    if use_uint4_binarization and uint4_thresholds is not None:
        binary_vecs_uint4 = encode_fn(embeddings[:, :dim], uint4_thresholds[:dim])

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


def check_unified_table_exists(
    conn,
    name: str,
    num_vectors: int,
    dimension: int,
    data_source: str = None,
    use_mean_binarization=False,
    use_uint8_binarization=False,
    use_uint4_binarization=False,
):
    """Check if a unified table exists with the correct metadata.

    Args:
        data_source: Filename or None for random vectors.
                    If None, always returns False (don't cache random vectors).
    """
    cursor = conn.cursor()
    try:
        # Don't cache random vectors (data_source=None)
        if data_source is None:
            cursor.close()
            return False

        # Check if metadata table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'benchmark_metadata'
            )
        """)
        if not cursor.fetchone()[0]:
            cursor.close()
            return False

        # Check if this specific table's metadata exists
        cursor.execute("""
            SELECT data_source, num_vectors, dimension,
                   has_mean_bin, has_uint8_bin, has_uint4_bin
            FROM benchmark_metadata
            WHERE table_name = %s
        """, (name,))

        result = cursor.fetchone()
        if not result:
            cursor.close()
            return False

        # Verify metadata matches
        stored_source, stored_num, stored_dim, has_mean, has_uint8, has_uint4 = result

        matches = (
            stored_source == data_source and
            stored_num == num_vectors and
            stored_dim == dimension and
            has_mean == use_mean_binarization and
            has_uint8 == use_uint8_binarization and
            has_uint4 == use_uint4_binarization
        )

        cursor.close()
        return matches
    except Exception:
        conn.rollback()
        cursor.close()
        return False


def create_and_insert_unified_table(
    conn,
    name: str,
    embeddings: np.ndarray,
    use_mean_binarization=False,
    dimension_means=None,
    use_uint8_binarization=False,
    uint8_thresholds=None,
    use_uint4_binarization=False,
    uint4_thresholds=None,
    encoding_type="thermometer",
    data_source="random",
    skip_if_exists=True,
):
    """Create a unified table with all precision columns and insert embeddings.

    This creates a single table with columns for:
    - embedding_f32 (vector)
    - embedding_f16 (halfvec)
    - embedding_bin (generated from f32)
    - embedding_bin_mean (optional)
    - embedding_bin_uint8 (optional)
    - embedding_bin_uint4 (optional)

    Args:
        skip_if_exists: If True, skip creation if table exists with same metadata
        data_source: Identifier for data source (filename or "random")
    """
    dim = embeddings.shape[1]
    num_vectors = embeddings.shape[0]

    # Check if table already exists with same metadata
    if skip_if_exists and check_unified_table_exists(
        conn, name, num_vectors, dim, data_source,
        use_mean_binarization, use_uint8_binarization, use_uint4_binarization
    ):
        print(f"[Table] Skipping {name} (already exists with matching data)")
        return

    cursor = conn.cursor()

    # Create metadata table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_metadata (
            table_name TEXT PRIMARY KEY,
            data_source TEXT,
            num_vectors INTEGER,
            dimension INTEGER,
            has_mean_bin BOOLEAN,
            has_uint8_bin BOOLEAN,
            has_uint4_bin BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    cursor.execute(f"DROP TABLE IF EXISTS {name}")
    dim_uint8 = dim * 8  # 8 bits per dimension for uint8-style encoding
    dim_uint4 = dim * 4  # 4 bits per dimension for uint4-style encoding

    # Build optional binary columns
    optional_cols = []
    if use_mean_binarization:
        optional_cols.append(f"embedding_bin_mean bit({dim})")
    if use_uint8_binarization:
        optional_cols.append(f"embedding_bin_uint8 bit({dim_uint8})")
    if use_uint4_binarization:
        optional_cols.append(f"embedding_bin_uint4 bit({dim_uint4})")
    optional_cols_str = (", " + ", ".join(optional_cols)) if optional_cols else ""

    # Create unified table with all precision columns
    cursor.execute(
        f"""
        CREATE TABLE {name} (
            id BIGINT PRIMARY KEY,
            embedding_f32 vector({dim}),
            embedding_f16 halfvec({dim}),
            embedding_bin bit({dim}) GENERATED ALWAYS AS (binary_quantize(embedding_f32)::bit({dim})) STORED
            {optional_cols_str}
        )
        """
    )
    conn.commit()

    ids = np.arange(embeddings.shape[0], dtype=np.int64)

    # Precompute binary vectors if needed
    binary_vecs_mean = None
    binary_vecs_uint8 = None
    binary_vecs_uint4 = None

    if use_mean_binarization and dimension_means is not None:
        binary_vecs_mean = binarize_with_means(
            embeddings[:, :dim], dimension_means[:dim]
        )

    # Select encoding function based on encoding_type parameter
    encode_fn = encode_thermometer if encoding_type == "thermometer" else encode_one_hot

    if use_uint8_binarization and uint8_thresholds is not None:
        binary_vecs_uint8 = encode_fn(embeddings[:, :dim], uint8_thresholds[:dim])

    if use_uint4_binarization and uint4_thresholds is not None:
        binary_vecs_uint4 = encode_fn(embeddings[:, :dim], uint4_thresholds[:dim])

    # Insert data in batches
    for i in tqdm(
        range(0, embeddings.shape[0], BATCH_SIZE), desc=f"Insert {name}", unit="batch"
    ):
        batch_end = min(i + BATCH_SIZE, embeddings.shape[0])

        # Build insert statement based on which binary columns are enabled
        insert_cols = ["id", "embedding_f32", "embedding_f16"]
        if use_mean_binarization:
            insert_cols.append("embedding_bin_mean")
        if use_uint8_binarization:
            insert_cols.append("embedding_bin_uint8")
        if use_uint4_binarization:
            insert_cols.append("embedding_bin_uint4")

        # Build template values
        template_parts = ["%s", "%s::vector", "%s::halfvec"]
        if use_mean_binarization:
            template_parts.append(f"%s::bit({dim})")
        if use_uint8_binarization:
            template_parts.append(f"%s::bit({dim_uint8})")
        if use_uint4_binarization:
            template_parts.append(f"%s::bit({dim_uint4})")

        # Build batch data
        batch_data = []
        for j in range(i, batch_end):
            vec_f32 = embeddings[j][:dim].astype(np.float32).tolist()
            vec_f16 = embeddings[j][:dim].astype(np.float16).tolist()

            row = [int(ids[j]), vec_f32, vec_f16]

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

    # Store metadata
    cursor.execute("""
        INSERT INTO benchmark_metadata
            (table_name, data_source, num_vectors, dimension,
             has_mean_bin, has_uint8_bin, has_uint4_bin)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (table_name) DO UPDATE SET
            data_source = EXCLUDED.data_source,
            num_vectors = EXCLUDED.num_vectors,
            dimension = EXCLUDED.dimension,
            has_mean_bin = EXCLUDED.has_mean_bin,
            has_uint8_bin = EXCLUDED.has_uint8_bin,
            has_uint4_bin = EXCLUDED.has_uint4_bin,
            created_at = CURRENT_TIMESTAMP
    """, (name, data_source, num_vectors, dim,
          use_mean_binarization, use_uint8_binarization, use_uint4_binarization))
    conn.commit()

    cursor.close()


def drop_all_indices_on_table(conn, table: str):
    """Drop all indices on the specified table.

    This ensures clean slate for benchmarking - only table data is kept,
    all indices are rebuilt fresh with current parameters.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT indexname
        FROM pg_indexes
        WHERE tablename = %s AND schemaname = 'public'
    """, (table,))
    indices = cursor.fetchall()

    for (idx_name,) in indices:
        # Skip primary key index
        if idx_name.endswith('_pkey'):
            continue
        print(f"[Index] Dropping existing index: {idx_name}")
        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")

    conn.commit()
    cursor.close()


def build_index(
    conn,
    table: str,
    precision: str,
    kind: str,
    dim: int,
    use_mean_bin=False,
    use_uint8_bin=False,
    use_uint4_bin=False,
    unified_table=False,
):
    """Build an index on the specified table.

    Args:
        conn: Database connection
        table: Table name
        precision: Precision type ("vector"/"halfvec" for old tables, "f32"/"f16" for unified)
        kind: Index kind (vchordrq, ivf, hnsw, diskann, binary_hnsw_rerank, binary_ivf_rerank)
        dim: Dimension
        use_mean_bin: Use mean-based binary quantization
        use_uint8_bin: Use quasi-uint8 binary quantization
        use_uint4_bin: Use quasi-uint4 binary quantization
        unified_table: Whether this is a unified table with embedding_f32/embedding_f16 columns
    """
    cursor = conn.cursor()

    # Map precision identifiers for unified tables
    if unified_table:
        if precision == "f32":
            pg_type = "vector"
            col_name = "embedding_f32"
            prec_suffix = "f32"
        elif precision == "f16":
            pg_type = "halfvec"
            col_name = "embedding_f16"
            prec_suffix = "f16"
        else:
            raise ValueError(f"Unknown precision for unified table: {precision}")
    else:
        # Old-style table with single "embedding" column
        col_name = "embedding"
        pg_type = precision
        prec_suffix = precision

    if kind == "binary_hnsw_rerank":
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
            cursor.execute(
                f"CREATE INDEX {idx_name} ON {table} USING hnsw ({bin_col} bit_hamming_ops) "
                f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
            )
        except Exception:
            conn.rollback()
            idx_name = None
    elif kind == "binary_ivf_rerank":
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
            cursor.execute(
                f"CREATE INDEX {idx_name} ON {table} USING ivfflat ({bin_col} bit_hamming_ops) WITH (lists = {IVF_LISTS})"
            )
        except Exception:
            conn.rollback()
            idx_name = None
    elif kind == "ivf":
        ops = "vector_cosine_ops" if pg_type == "vector" else "halfvec_cosine_ops"
        idx_name = f"idx_{table}_ivf_{prec_suffix}"
        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")
        start = time.time()
        cursor.execute(
            f"CREATE INDEX {idx_name} ON {table} USING ivfflat ({col_name} {ops}) WITH (lists = {IVF_LISTS})"
        )
    elif kind == "hnsw":
        ops = "vector_cosine_ops" if pg_type == "vector" else "halfvec_cosine_ops"
        idx_name = f"idx_{table}_hnsw_{prec_suffix}"
        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")
        start = time.time()
        cursor.execute(
            f"CREATE INDEX {idx_name} ON {table} USING hnsw ({col_name} {ops}) "
            f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
        )
    elif kind == "diskann":
        ops = "vector_cosine_ops" if pg_type == "vector" else "halfvec_cosine_ops"
        idx_name = f"idx_{table}_diskann_{prec_suffix}"
        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")
        start = time.time()
        cursor.execute(
            f"CREATE INDEX {idx_name} ON {table} USING diskann ({col_name} {ops})"
        )
    else:  # vchordrq (default)
        ops = "vector_cosine_ops" if pg_type == "vector" else "halfvec_cosine_ops"
        idx_name = f"idx_{table}_vchord_{prec_suffix}"
        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")
        start = time.time()
        cursor.execute(
            f"CREATE INDEX {idx_name} ON {table} USING vchordrq ({col_name} {ops}) WITH (options = $$\n"
            f"rerank_in_table = true\n"
            f"[build.internal]\n"
            f"lists = [{VCHORDRQ_LISTS}]\n"
            f"spherical_centroids = {str(VCHORDRQ_SPHERICAL_CENTROIDS).lower()}\n"
            f"build_threads = {VCHORDRQ_BUILD_THREADS}\n"
            f"$$)"
        )
    conn.commit()
    build_time = time.time() - start
    cursor.close()
    return idx_name, build_time


def get_index_size_mb(conn, idx_name: str):
    """Get the size of an index in MB."""
    if not idx_name:
        return 0.0
    cursor = conn.cursor()
    cursor.execute("SELECT pg_total_relation_size(%s) / (1024.0*1024.0)", (idx_name,))
    size_mb = cursor.fetchone()[0]
    cursor.close()
    return size_mb


def _to_np_vec(val, dtype=np.float32):
    """Convert various vector representations to numpy array."""
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
