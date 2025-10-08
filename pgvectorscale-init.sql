-- Enable vectorscale extension
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

-- Create table for benchmark
CREATE TABLE IF NOT EXISTS vectors (
    id BIGINT PRIMARY KEY,
    embedding vector(512)
);
