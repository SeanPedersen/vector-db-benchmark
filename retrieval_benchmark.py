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
  - Step 2: Fetch float32 vectors for 1000 candidates
  - Step 3: Compute cosine distances using numpy
  - Step 4: Rerank to get top 100 by cosine similarity
  - Result: Memory-efficient index with high recall via overfetching

All recall measured against brute force (float32 1024-d) baseline.

NOTE: Uses random normalized embeddings as stand-in for CLIP v2 text embeddings.
"""

import numpy as np
import psycopg2
import time
from psycopg2.extras import execute_values
from tqdm import tqdm
import argparse

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

DIMENSIONS = [128]  # Focus on 128D for tuning

# IVF index parameters (tunable)
IVF_LISTS = 100  # Number of clusters (typically sqrt(num_vectors))
IVF_PROBES = 10  # Number of clusters to search (higher = better recall, slower)


def generate_embeddings(num_vectors: int, full_dim: int = 1024):
    np.random.seed(123)
    data = np.random.randn(num_vectors, full_dim).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    return data


def build_baseline(full_embeddings: np.ndarray, query: np.ndarray):
    sims = full_embeddings @ query
    top_idx = np.argsort(sims)[::-1][:K]
    return top_idx


def ensure_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_and_insert_table(conn, name: str, embeddings: np.ndarray, precision: str):
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {name}")
    dim = embeddings.shape[1]
    if precision == "vector":
        cursor.execute(f"CREATE TABLE {name} (id BIGINT PRIMARY KEY, embedding vector({dim}))")
        cast = "::vector"
    elif precision == "halfvec":
        cursor.execute(f"CREATE TABLE {name} (id BIGINT PRIMARY KEY, embedding halfvec({dim}))")
        cast = "::halfvec"
    else:
        raise ValueError("precision must be vector or halfvec")
    conn.commit()

    ids = np.arange(embeddings.shape[0], dtype=np.int64)

    for i in tqdm(range(0, embeddings.shape[0], BATCH_SIZE), desc=f"Insert {name}", unit="batch"):
        batch_end = min(i + BATCH_SIZE, embeddings.shape[0])
        batch_data = [
            (int(ids[j]), embeddings[j][:dim].astype(np.float16 if precision == "halfvec" else np.float32).tolist())
            for j in range(i, batch_end)
        ]
        execute_values(
            cursor,
            f"INSERT INTO {name} (id, embedding) VALUES %s",
            batch_data,
            template=f"(%s, %s{cast})",
        )
        if (i // BATCH_SIZE) % 10 == 0:
            conn.commit()
    conn.commit()
    cursor.close()


def build_index(conn, table: str, precision: str, kind: str, dim: int):
    cursor = conn.cursor()
    if kind == "binary_hnsw":
        idx_name = f"idx_{table}_hnsw_bin"
        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")
        start = time.time()
        try:
            # Use pgvector's HNSW with binary_quantize for binary quantization
            # Use bit_hamming_ops for Hamming distance
            cursor.execute(
                f"CREATE INDEX {idx_name} ON {table} USING hnsw ((binary_quantize(embedding)::bit({dim})) bit_hamming_ops)"
            )
        except Exception:
            conn.rollback()
            idx_name = None
    elif kind == "binary_ivf":
        idx_name = f"idx_{table}_ivf_bin"
        cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")
        start = time.time()
        try:
            # Use pgvector's IVFFlat with binary_quantize for binary quantization
            # Use bit_hamming_ops for Hamming distance
            cursor.execute(
                f"CREATE INDEX {idx_name} ON {table} USING ivfflat ((binary_quantize(embedding)::bit({dim})) bit_hamming_ops) WITH (lists = {IVF_LISTS})"
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


def query_index(conn, table: str, precision: str, kind: str, dim: int, query_vec: np.ndarray, baseline_ids):
    cursor = conn.cursor()
    # Prepare query textual forms
    query_list_full = query_vec[:dim].tolist()
    query_txt = "[" + ",".join(map(str, query_list_full)) + "]"

    if kind == "ivf":
        # Set probes for IVFFlat
        cursor.execute(f"SET ivfflat.probes = {IVF_PROBES}")
        cast_type = "vector" if precision == "vector" else "halfvec"
        # Warm-up
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::{cast_type} LIMIT {K}", (query_txt,)
        )
        cursor.fetchall()
        start = time.time()
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::{cast_type} LIMIT {K}", (query_txt,)
        )
        rows = cursor.fetchall()
        retrieved = [r[0] for r in rows]
        latency = time.time() - start
    elif kind == "binary_hnsw" or kind == "binary_ivf":
        # Binary quantization: 1 bit per dimension (sign bit: >= 0 → 1, < 0 → 0)
        # Step 1: Fast overfetch (1000 neighbors) using Hamming distance on bit vectors
        # Step 2: Fetch float32 vectors and rerank with numpy cosine distance to get top 100
        overfetch = K * OVERFETCH_FACTOR  # 100 * 10 = 1000

        # Set probes for IVFFlat binary
        if kind == "binary_ivf":
            cursor.execute(f"SET ivfflat.probes = {IVF_PROBES}")

        # Warm-up
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY binary_quantize(embedding)::bit({dim}) <~> binary_quantize(%s::{precision})::bit({dim}) LIMIT {K}",
            (query_txt,)
        )
        cursor.fetchall()
        start = time.time()
        # Step 1: Overfetch 1000 candidates using Hamming distance on bit vectors
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY binary_quantize(embedding)::bit({dim}) <~> binary_quantize(%s::{precision})::bit({dim}) LIMIT {overfetch}",
            (query_txt,)
        )
        cand_ids = [r[0] for r in cursor.fetchall()]

        # Step 2: Fetch float32 vectors for the 1000 candidates
        cursor.execute(
            f"SELECT id, embedding::text FROM {table} WHERE id = ANY(%s)", (cand_ids,)
        )
        rows = cursor.fetchall()

        # Build candidate matrix for vectorized numpy computation
        id_to_idx = {cand_id: i for i, cand_id in enumerate(cand_ids)}
        candidate_vecs = np.zeros((len(cand_ids), dim), dtype=np.float32)

        for rid, etxt in rows:
            vals = [float(x) for x in etxt.strip("[]").split(",")]
            v = np.array(vals[:dim], dtype=np.float32)
            candidate_vecs[id_to_idx[rid]] = v

        # Normalize candidate vectors
        norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
        candidate_vecs = candidate_vecs / norms

        # Step 3: Compute cosine similarities using numpy (dot product of normalized vectors)
        query_normalized = query_vec[:dim] / np.linalg.norm(query_vec[:dim])
        cosine_similarities = candidate_vecs @ query_normalized

        # Step 4: Get top 100 by cosine similarity
        top_indices = np.argsort(cosine_similarities)[::-1][:K]
        retrieved = [cand_ids[i] for i in top_indices]
        latency = time.time() - start
    else:
        # Warm-up
        cast_type = "vector" if precision == "vector" else "halfvec"
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::{cast_type} LIMIT {K}", (query_txt,)
        )
        cursor.fetchall()
        start = time.time()
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY embedding <=> %s::{cast_type} LIMIT {K}", (query_txt,)
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
    cursor.execute(
        "SELECT pg_total_relation_size(%s) / (1024.0*1024.0)", (idx_name,)
    )
    size_mb = cursor.fetchone()[0]
    cursor.close()
    return size_mb


def main():
    global IVF_LISTS, IVF_PROBES, OVERFETCH_FACTOR

    parser = argparse.ArgumentParser(description="Retrieval benchmark (TASK.md)")
    parser.add_argument("--num-vectors", type=int, default=50000, help="Number of vectors to generate")
    parser.add_argument("--ivf-lists", type=int, default=100, help="IVF lists parameter (default: 100)")
    parser.add_argument("--ivf-probes", type=int, default=10, help="IVF probes parameter (default: 10)")
    parser.add_argument("--overfetch", type=int, default=10, help="Binary overfetch factor (default: 10)")
    args = parser.parse_args()

    num_vectors = args.num_vectors
    # Update global parameters from args
    IVF_LISTS = args.ivf_lists
    IVF_PROBES = args.ivf_probes
    OVERFETCH_FACTOR = args.overfetch

    print(f"[Setup] Generating {num_vectors:,} normalized 1024-d embeddings...")
    full_embeddings = generate_embeddings(num_vectors)

    print("[Setup] Generating query vector...")
    np.random.seed(999)
    query = np.random.randn(1024).astype(np.float32)
    query /= np.linalg.norm(query)

    print("[Baseline] Computing brute force baseline (1024-d float32)...")
    baseline_ids = build_baseline(full_embeddings, query)
    print(f"[Baseline] Top-K baseline computed (K={K})")
    print(f"[Params] IVF: lists={IVF_LISTS}, probes={IVF_PROBES} | Binary overfetch: {OVERFETCH_FACTOR}x")

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

        # Create tables & insert
        tbl_vector = f"items_vec_{dim}"
        tbl_half = f"items_half_{dim}"
        print(f"[Storage] Creating + inserting float32 table {tbl_vector} ...")
        create_and_insert_table(conn, tbl_vector, trunc_embeddings, "vector")
        print(f"[Storage] Creating + inserting float16 table {tbl_half} ...")
        create_and_insert_table(conn, tbl_half, trunc_embeddings, "halfvec")

        # Build indices
        print("[Index] Building VectorChord (vector)...")
        idx_vchord_vec, t_vchord_vec = build_index(conn, tbl_vector, "vector", "full", dim)
        print("[Index] Building VectorChord (halfvec)...")
        idx_vchord_half, t_vchord_half = build_index(conn, tbl_half, "halfvec", "half", dim)
        print(f"[Index] Building IVFFlat (vector, lists={IVF_LISTS})...")
        idx_ivf_vec, t_ivf_vec = build_index(conn, tbl_vector, "vector", "ivf", dim)
        print(f"[Index] Building IVFFlat (halfvec, lists={IVF_LISTS})...")
        idx_ivf_half, t_ivf_half = build_index(conn, tbl_half, "halfvec", "ivf", dim)
        print("[Index] Building HNSW binary (bit) ...")
        idx_hnsw_bin, t_hnsw_bin = build_index(conn, tbl_vector, "vector", "binary_hnsw", dim)
        print(f"[Index] Building IVFFlat binary (bit, lists={IVF_LISTS}) ...")
        idx_ivf_bin, t_ivf_bin = build_index(conn, tbl_vector, "vector", "binary_ivf", dim)

        # Query each
        print("[Query] VectorChord full precision...")
        lat_vchord_vec, rec_vchord_vec = query_index(conn, tbl_vector, "vector", "full", dim, query, baseline_ids)
        size_vchord_vec = get_index_size_mb(conn, idx_vchord_vec)

        print("[Query] VectorChord half precision...")
        lat_vchord_half, rec_vchord_half = query_index(conn, tbl_half, "halfvec", "half", dim, query, baseline_ids)
        size_vchord_half = get_index_size_mb(conn, idx_vchord_half)

        print(f"[Query] IVFFlat full precision (probes={IVF_PROBES})...")
        lat_ivf_vec, rec_ivf_vec = query_index(conn, tbl_vector, "vector", "ivf", dim, query, baseline_ids)
        size_ivf_vec = get_index_size_mb(conn, idx_ivf_vec)

        print(f"[Query] IVFFlat half precision (probes={IVF_PROBES})...")
        lat_ivf_half, rec_ivf_half = query_index(conn, tbl_half, "halfvec", "ivf", dim, query, baseline_ids)
        size_ivf_half = get_index_size_mb(conn, idx_ivf_half)

        print(f"[Query] HNSW binary ({OVERFETCH_FACTOR}x overfetch + rerank)...")
        lat_hnsw_bin, rec_hnsw_bin = query_index(conn, tbl_vector, "vector", "binary_hnsw", dim, query, baseline_ids)
        size_hnsw_bin = get_index_size_mb(conn, idx_hnsw_bin)

        print(f"[Query] IVFFlat binary (probes={IVF_PROBES}, {OVERFETCH_FACTOR}x overfetch + rerank)...")
        lat_ivf_bin, rec_ivf_bin = query_index(conn, tbl_vector, "vector", "binary_ivf", dim, query, baseline_ids)
        size_ivf_bin = get_index_size_mb(conn, idx_ivf_bin)

        results.extend([
            {
                "dim": dim,
                "storage": "float32",
                "index": "vchordrq",
                "lat_ms": lat_vchord_vec * 1000,
                "recall": rec_vchord_vec,
                "build_s": t_vchord_vec,
                "size_mb": size_vchord_vec,
            },
            {
                "dim": dim,
                "storage": "float16",
                "index": "vchordrq",
                "lat_ms": lat_vchord_half * 1000,
                "recall": rec_vchord_half,
                "build_s": t_vchord_half,
                "size_mb": size_vchord_half,
            },
            {
                "dim": dim,
                "storage": "float32",
                "index": f"ivfflat(L{IVF_LISTS},P{IVF_PROBES})",
                "lat_ms": lat_ivf_vec * 1000,
                "recall": rec_ivf_vec,
                "build_s": t_ivf_vec,
                "size_mb": size_ivf_vec,
            },
            {
                "dim": dim,
                "storage": "float16",
                "index": f"ivfflat(L{IVF_LISTS},P{IVF_PROBES})",
                "lat_ms": lat_ivf_half * 1000,
                "recall": rec_ivf_half,
                "build_s": t_ivf_half,
                "size_mb": size_ivf_half,
            },
            {
                "dim": dim,
                "storage": "float32",
                "index": f"hnsw+binary({OVERFETCH_FACTOR}x)",
                "lat_ms": lat_hnsw_bin * 1000,
                "recall": rec_hnsw_bin,
                "build_s": t_hnsw_bin,
                "size_mb": size_hnsw_bin,
            },
            {
                "dim": dim,
                "storage": "float32",
                "index": f"ivf+binary(L{IVF_LISTS},P{IVF_PROBES},{OVERFETCH_FACTOR}x)",
                "lat_ms": lat_ivf_bin * 1000,
                "recall": rec_ivf_bin,
                "build_s": t_ivf_bin,
                "size_mb": size_ivf_bin,
            },
        ])

    conn.close()

    # Print summary
    print("\n=== Retrieval Benchmark Results ===")
    print(
        "| Dim | Storage | Index | Latency ms | Recall | Build s | Index MB |"\
    )
    print(
        "|-----|---------|-------|-----------:|-------:|--------:|---------:|"\
    )
    for r in results:
        print(
            f"| {r['dim']} | {r['storage']} | {r['index']} | {r['lat_ms']:.2f} | {r['recall']*100:.2f}% | {r['build_s']:.2f} | {r['size_mb']:.1f} |"
        )


if __name__ == "__main__":
    main()
