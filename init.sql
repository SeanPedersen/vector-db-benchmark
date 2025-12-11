-- Enable all vector search extensions
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;  -- Includes pgvector + DiskANN
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;       -- VectorChord vchordrq

-- Table will be created by benchmark scripts with appropriate dimensions
