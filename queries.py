"""Query execution logic for retrieval benchmark."""

import time
import numpy as np

from config import (
    K,
    OVERFETCH_FACTOR,
    IVF_PROBES,
    IVF_PROBES_BINARY,
    HNSW_EF_SEARCH,
    HNSW_EF_SEARCH_MAX,
    VCHORDRQ_EPSILON,
    VCHORDRQ_PROBES,
)
from embeddings import (
    binarize_with_means,
    encode_thermometer,
    numpy_binary_to_postgres_bit_string,
)


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
    unified_table=False,
):
    """Execute a query against the specified index and return latency and recall.

    Args:
        conn: Database connection
        table: Table name
        precision: "vector"/"halfvec" for old tables, "f32"/"f16" for unified tables
        kind: Index type ("ivf", "hnsw", "diskann", "binary_hnsw_rerank", "binary_ivf_rerank", "binary_exact_rerank", "exact", "vchordrq")
        dim: Number of dimensions
        query_vec: Query vector
        baseline_ids: Baseline (ground truth) IDs for recall calculation
        debug: Enable debug output
        local_embeddings: Unused parameter (kept for compatibility)
        explain_analyze: Print EXPLAIN ANALYZE output
        use_mean_bin: Use mean-based binarization
        dimension_means: Dimension means for binarization
        use_uint8_bin: Use uint8-style binarization
        uint8_thresholds: Thresholds for uint8 encoding
        use_uint4_bin: Use uint4-style binarization
        uint4_thresholds: Thresholds for uint4 encoding
        unified_table: Whether this is a unified table with embedding_f32/embedding_f16 columns

    Returns:
        Tuple of (latency, recall)
    """
    cursor = conn.cursor()

    # Map precision identifiers for unified tables
    if unified_table:
        if precision == "f32":
            cast_type = "vector"
            emb_col = "embedding_f32"
        elif precision == "f16":
            cast_type = "halfvec"
            emb_col = "embedding_f16"
        else:
            raise ValueError(f"Unknown precision for unified table: {precision}")
    else:
        # Old-style table with single "embedding" column
        emb_col = "embedding"
        cast_type = precision
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
        # Warm-up
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY {emb_col} <=> %s::{cast_type} LIMIT {K}",
            (query_txt,),
        )
        cursor.fetchall()
        start = time.time()
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY {emb_col} <=> %s::{cast_type} LIMIT {K}",
            (query_txt,),
        )
        rows = cursor.fetchall()
        retrieved = [r[0] for r in rows]
        latency = time.time() - start
    elif kind in (
        "binary_hnsw_rerank",
        "binary_ivf_rerank",
    ):
        # Use binary index for candidate generation, SQL for rerank (no Python NumPy).
        overfetch = K * OVERFETCH_FACTOR

        if "ivf" in kind:
            cursor.execute(f"SET ivfflat.probes = {IVF_PROBES_BINARY}")
        else:
            eff = min(max(HNSW_EF_SEARCH, overfetch), HNSW_EF_SEARCH_MAX)
            cursor.execute(f"SET hnsw.ef_search = {eff}")

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
            ORDER BY t.{emb_col} <=> %s::{cast_type}
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
            ORDER BY t.{emb_col} <=> %s::{cast_type}
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
            ORDER BY t.{emb_col} <=> %s::{cast_type}
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
            ORDER BY t.{emb_col} <=> %s::{cast_type}
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
    elif kind in ("binary_exact_rerank", "binary_exact_k"):
        # Exact binary candidate generation (no index). For _k: no rerank to measure pure Hamming scan speed.
        overfetch = K if kind.endswith("_k") else K * OVERFETCH_FACTOR

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
                ORDER BY t.{emb_col} <=> %s::{cast_type}
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
                ORDER BY t.{emb_col} <=> %s::{cast_type}
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
                ORDER BY t.{emb_col} <=> %s::{cast_type}
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
                ORDER BY t.{emb_col} <=> %s::{cast_type}
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

            sql = f"SELECT id FROM {table} ORDER BY {emb_col} <=> %s::{cast_type} LIMIT {K}"

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
    elif kind in ("hnsw", "diskann"):
        # HNSW or DiskANN index query
        if kind == "hnsw":
            eff = min(max(HNSW_EF_SEARCH, K), HNSW_EF_SEARCH_MAX)
            cursor.execute(f"SET hnsw.ef_search = {eff}")

        # Warm-up
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY {emb_col} <=> %s::{cast_type} LIMIT {K}",
            (query_txt,),
        )
        cursor.fetchall()
        start = time.time()
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY {emb_col} <=> %s::{cast_type} LIMIT {K}",
            (query_txt,),
        )
        rows = cursor.fetchall()
        retrieved = [r[0] for r in rows]
        latency = time.time() - start
    else:
        # VectorChord vchordrq - set RaBitQ epsilon and probes
        cursor.execute(f"SET vchordrq.epsilon = {VCHORDRQ_EPSILON}")
        cursor.execute(f"SET vchordrq.probes = {VCHORDRQ_PROBES}")
        # Warm-up
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY {emb_col} <=> %s::{cast_type} LIMIT {K}",
            (query_txt,),
        )
        cursor.fetchall()
        start = time.time()
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY {emb_col} <=> %s::{cast_type} LIMIT {K}",
            (query_txt,),
        )
        rows = cursor.fetchall()
        retrieved = [r[0] for r in rows]
        latency = time.time() - start

    # Recall vs full baseline ids
    recall = len(set(retrieved) & set(baseline_ids)) / len(baseline_ids)
    cursor.close()
    return latency, recall
