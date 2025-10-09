-- Enable extensions for all index types
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;

-- Create single table for all benchmark indices
CREATE TABLE IF NOT EXISTS vectors (
    id BIGINT PRIMARY KEY,
    embedding vector(512)
);
