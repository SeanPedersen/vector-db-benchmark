#!/bin/bash
# Reset database by removing Docker volume

echo "Stopping container..."
docker compose down

echo "Removing Docker volume..."
docker volume rm ann-benchmark_postgres-data 2>/dev/null

echo "Database volume removed. Database will be recreated on next startup."
