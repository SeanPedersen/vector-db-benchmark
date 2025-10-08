#!/usr/bin/env python3
"""Main benchmark orchestration script."""

import subprocess
import sys
import os
import time
import argparse
import psycopg2

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    return result

def wait_for_db(max_attempts=30, delay=2):
    """Wait for database to be ready."""
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'postgres'
    }

    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**db_config)
            conn.close()
            print(f"Database ready after {attempt + 1} attempts ({(attempt + 1) * delay} seconds)")
            return True
        except psycopg2.OperationalError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
            else:
                print(f"Database not ready after {max_attempts * delay} seconds")
                return False
    return False

def main():
    parser = argparse.ArgumentParser(description="ANN Benchmark - pgvectorscale vs vectorchord")
    parser.add_argument('--skip-pgvectorscale', action='store_true',
                        help='Skip pgvectorscale benchmark')
    parser.add_argument('--skip-vectorchord', action='store_true',
                        help='Skip vectorchord benchmark')
    parser.add_argument('--skip-insertion', action='store_true',
                        help='Skip data insertion (run queries only)')
    parser.add_argument('--num-vectors', type=int, default=100000,
                        help='Number of vectors to generate and benchmark (default: 100000)')
    args = parser.parse_args()

    print("ANN BENCHMARK - pgvectorscale vs vectorchord")
    print("="*60)

    # Step 1: Generate data
    regenerate_data = False
    if not os.path.exists('vectors.npy'):
        regenerate_data = True
    else:
        # Check if existing vectors.npy has the correct size
        import numpy as np
        existing_vectors = np.load('vectors.npy')
        if len(existing_vectors) != args.num_vectors:
            print(f"\nStep 1: Existing vectors.npy has {len(existing_vectors):,} vectors, but {args.num_vectors:,} requested.")
            print("Regenerating data...")
            regenerate_data = True
        else:
            print("\nStep 1: Using existing vectors.npy")

    if regenerate_data:
        print(f"\nStep 1: Generating {args.num_vectors:,} random vectors...")
        run_command(f"python3 generate_data.py --num-vectors {args.num_vectors}", "Data generation")

    # Step 2: Compute baseline
    print("\nStep 2: Computing exact baseline...")
    run_command("python3 compute_baseline.py", "Baseline computation")

    # Step 3: Test pgvectorscale
    if not args.skip_pgvectorscale:
        print("\n" + "="*60)
        print("TESTING PGVECTORSCALE (DiskANN)")
        print("="*60)

        print("\nStarting pgvectorscale container...")
        run_command("docker compose -f docker-compose-pgvectorscale.yml up -d",
                    "Start pgvectorscale")

        print("\nWaiting for database to be ready...")
        if not wait_for_db():
            print("Error: Database failed to start")
            sys.exit(1)

        if not args.skip_insertion:
            print("\nInserting vectors into pgvectorscale...")
            run_command("python3 pgvectorscale_insert.py", "pgvectorscale insertion")
        else:
            print("\nSkipping insertion (--skip-insertion)")

        print("\nQuerying pgvectorscale...")
        run_command("python3 pgvectorscale_query.py", "pgvectorscale query")

        print("\nStopping pgvectorscale container...")
        run_command("docker compose -f docker-compose-pgvectorscale.yml down",
                    "Stop pgvectorscale")
    else:
        print("\n" + "="*60)
        print("SKIPPING PGVECTORSCALE (--skip-pgvectorscale)")
        print("="*60)

    # Step 4: Test vectorchord
    if not args.skip_vectorchord:
        print("\n" + "="*60)
        print("TESTING VECTORCHORD (vchordrq)")
        print("="*60)

        print("\nStarting vectorchord container...")
        run_command("docker compose -f docker-compose-vectorchord.yml up -d",
                    "Start vectorchord")

        print("\nWaiting for database to be ready...")
        if not wait_for_db():
            print("Error: Database failed to start")
            sys.exit(1)

        if not args.skip_insertion:
            print("\nInserting vectors into vectorchord...")
            run_command("python3 vectorchord_insert.py", "vectorchord insertion")
        else:
            print("\nSkipping insertion (--skip-insertion)")

        print("\nQuerying vectorchord...")
        run_command("python3 vectorchord_query.py", "vectorchord query")

        print("\nStopping vectorchord container...")
        run_command("docker compose -f docker-compose-vectorchord.yml down",
                    "Stop vectorchord")
    else:
        print("\n" + "="*60)
        print("SKIPPING VECTORCHORD (--skip-vectorchord)")
        print("="*60)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE!")
    print("="*60)

    tested_systems = []
    if not args.skip_pgvectorscale:
        tested_systems.append("pgvectorscale")
    if not args.skip_vectorchord:
        tested_systems.append("vectorchord")

    if tested_systems:
        print(f"\nTested systems: {', '.join(tested_systems)}. Check the output above for results.")
    else:
        print("\nNo systems were tested (both were skipped).")

if __name__ == "__main__":
    main()
