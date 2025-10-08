#!/bin/bash
# Reset databases by removing all Docker volumes

echo "Stopping all containers..."
docker compose -f docker-compose-pgvectorscale.yml down 2>/dev/null
docker compose -f docker-compose-vectorchord.yml down 2>/dev/null

echo "Removing Docker volumes..."
docker volume rm ann-benchmark_pgvectorscale-data 2>/dev/null
docker volume rm ann-benchmark_vectorchord-data 2>/dev/null

echo "Database volumes removed. Databases will be recreated on next startup."
