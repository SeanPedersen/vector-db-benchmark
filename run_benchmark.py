#!/usr/bin/env python3
"""Main benchmark orchestration script."""

import subprocess
import sys
import os
import time
import argparse
import psycopg2
from compute_baseline import compute_baseline
from query_vectorchord import run_benchmark as run_vectorchord_benchmark
from query_pgvectorscale import run_benchmark as run_pgvectorscale_benchmark


def run_command(cmd, description):
    """Run a command and capture output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    return result


def wait_for_db(max_attempts=30, delay=2):
    """Wait for database to be ready."""
    db_config = {
        "host": "localhost",
        "port": 5432,
        "dbname": "postgres",
        "user": "postgres",
        "password": "postgres",
    }

    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**db_config)
            conn.close()
            return True
        except psycopg2.OperationalError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
            else:
                print(f"Error: Database not ready after {max_attempts * delay} seconds")
                return False
    return False


def check_table_count(expected_count):
    """Check if the vectors table has the expected number of rows."""
    db_config = {
        "host": "localhost",
        "port": 5432,
        "dbname": "postgres",
        "user": "postgres",
        "password": "postgres",
    }

    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM vectors")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count == expected_count
    except Exception as e:
        print(f"Error checking table count: {e}")
        return False


def print_results_table(baseline_time, num_vectors, results):
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print(f"# ANN Benchmark Results ({num_vectors // 1000}K vectors)")
    print("=" * 100)
    print()
    print(
        "| Method | Query Latency (ms) | Retrieval recall | Speedup vs Baseline | Index Build Time (s) | Index Size (MB) |"
    )
    print(
        "|--------|-------------------|---------------------|---------------------|---------------------|-----------------|"
    )

    # Baseline row
    print(
        f"| Baseline (Brute Force) | {baseline_time * 1000:.2f} | 100.00% | 1.00x | - | - |"
    )

    # Sort results by method name
    for result in sorted(results, key=lambda x: x["method"]):
        method = result["method"]
        if method == "vchordrq":
            display_name = "**VectorChord (vchordrq)**"
        elif method == "HNSW":
            display_name = "**pgvector (HNSW)**"
        elif method == "IVFFlat":
            display_name = "**pgvector (IVFFlat)**"
        elif method == "DiskANN":
            display_name = "**pgvectorscale (DiskANN)**"
        else:
            display_name = f"**{method}**"

        latency_ms = result["latency"] * 1000
        recall_pct = result["recall"] * 100
        speedup = baseline_time / result["latency"]
        build_time = result["build_time"]
        size_mb = int(result["size_mb"])

        print(
            f"| {display_name} | {latency_ms:.2f} | {recall_pct:.2f}% | {speedup:.2f}x | {build_time:.2f} | {size_mb} |"
        )


def main():
    parser = argparse.ArgumentParser(
        description="ANN Benchmark - pgvectorscale vs vectorchord"
    )
    parser.add_argument(
        "--skip-pgvectorscale", action="store_true", help="Skip pgvectorscale benchmark"
    )
    parser.add_argument(
        "--skip-vectorchord", action="store_true", help="Skip vectorchord benchmark"
    )
    parser.add_argument(
        "--skip-insertion",
        action="store_true",
        help="Skip data insertion (run queries only)",
    )
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=100000,
        help="Number of vectors to generate and benchmark (default: 100000)",
    )
    parser.add_argument(
        "--vectors-file",
        type=str,
        default=None,
        help="Path to .npy file containing pre-generated vectors (optional)",
    )
    args = parser.parse_args()

    print("\n=== ANN Benchmark: pgvectorscale vs vectorchord ===")

    # Determine dimensions from vectors file if provided
    dimensions = 512  # default
    if args.vectors_file:
        import numpy as np

        print(f"[Setup] Loading {args.vectors_file} to determine dimensions...")
        vectors = np.load(args.vectors_file)
        if vectors.ndim == 2:
            dimensions = vectors.shape[1]
            print(f"[Setup] Detected {dimensions} dimensions from vectors file")
        else:
            print(f"Error: Invalid vectors file shape {vectors.shape}")
            sys.exit(1)

    # Step 1: Generate query vector
    if not os.path.exists("query.npy"):
        print(f"[Setup] Generating query vector with {dimensions} dimensions...")
        run_command(
            f"python3 generate_data.py --dimensions {dimensions}", "Query generation"
        )
    else:
        print("[Setup] Using existing query.npy")

    # Step 2: Start unified database
    print("[Setup] Starting database...")
    run_command("docker compose up -d", "Start database")

    if not wait_for_db():
        sys.exit(1)

    # Step 3: Insert data (once for all benchmarks)
    if not args.skip_insertion:
        # Check if table already has the correct number of vectors
        if check_table_count(args.num_vectors):
            print(f"[Setup] Table already contains {args.num_vectors:,} vectors")
        else:
            # Run insertion with live output (no capture) so user can see progress
            cmd = f"python3 insert.py --num-vectors {args.num_vectors} --dimensions {dimensions}"
            if args.vectors_file:
                cmd += f" --vectors-file {args.vectors_file}"
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(
                    f"Error: Data insertion failed with exit code {result.returncode}"
                )
                sys.exit(1)
    else:
        print("[Setup] Skipping insertion (--skip-insertion)")

    # Step 4: Compute baseline (numpy + postgres brute force)
    baseline = compute_baseline()

    results = []

    # Step 5: Test vectorchord
    if not args.skip_vectorchord:
        result = run_vectorchord_benchmark(
            baseline["query"], baseline["baseline_ids"], baseline["postgres_time"]
        )
        results.append(result)

    # Step 6: Test pgvectorscale
    if not args.skip_pgvectorscale:
        pgvectorscale_results = run_pgvectorscale_benchmark(
            baseline["query"], baseline["baseline_ids"], baseline["postgres_time"]
        )
        results.extend(pgvectorscale_results)

    # Print final results table
    if results:
        print_results_table(baseline["postgres_time"], args.num_vectors, results)

    print("Database container is still running. Stop with: docker compose stop")


if __name__ == "__main__":
    main()
