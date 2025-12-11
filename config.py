"""Configuration constants for retrieval benchmark."""

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
}

# Retrieval parameters
K = 100  # Retrieve 100 nearest neighbors
OVERFETCH_FACTOR = 10  # For binary index: retrieve 1000 candidates, rerank to top 100
BATCH_SIZE = 10_000

# Test dimensions
DIMENSIONS = [256, 512, 1024]  # Test Matryoshka embedding dimensions

# IVF index parameters (tunable)
IVF_LISTS = 100  # Number of clusters (typically sqrt(num_vectors))
IVF_PROBES = 10  # Number of clusters to search (higher = better recall, slower)
IVF_PROBES_BINARY = 50  # Higher probes for binary (Hamming distance is less accurate)

# HNSW parameters
HNSW_EF_SEARCH = 1500  # HNSW ef_search parameter for binary index (raise default; should be >= overfetch)
HNSW_M = 32  # HNSW build parameter (recommended: M ~ 16-48)
HNSW_EF_CONSTRUCTION = 300  # HNSW ef_construction (recommended: 200-400)
HNSW_EF_SEARCH_MAX = 1000  # Upper cap for hnsw.ef_search to satisfy server limits (e.g., 1..1000)

# VectorChord vchordrq parameters
VCHORDRQ_EPSILON = 1.9  # RaBitQ epsilon (range: 0.0-4.0; higher = more accurate, slower)
VCHORDRQ_BUILD_THREADS = 8  # Build threads for K-means clustering (range: 1-255)
VCHORDRQ_LISTS = 1000  # Number of clusters for vchordrq index
VCHORDRQ_PROBES = 100  # Number of clusters to search (higher = better recall, slower)
VCHORDRQ_SPHERICAL_CENTROIDS = True  # Enable spherical centroids (recommended with cosine similarity)

# Index configuration matrix - defines which data types each index supports
INDEX_CONFIGS = {
    "vchordrq": {
        "precisions": ["f32", "f16"],
        "binary": False,
    },
    "ivfflat": {
        "precisions": ["f32", "f16"],
        "binary": False,
    },
    "diskann": {
        "precisions": ["f32", "f16"],
        "binary": False,
    },
    "hnsw": {
        "precisions": ["f32", "f16"],
        "binary": False,
    },
    "binary_hnsw_rerank": {
        "precisions": [],
        "binary": True,
        "variants": ["mean", "uint8", "uint4"],
    },
    "binary_ivf_rerank": {
        "precisions": [],
        "binary": True,
        "variants": ["mean", "uint8", "uint4"],
    },
    "exact": {
        "precisions": ["f32", "f16"],
        "binary": False,
    },
    "binary_exact_rerank": {
        "precisions": [],
        "binary": True,
        "variants": ["mean", "uint8", "uint4"],
    },
}

# Precision mappings
PRECISION_TYPES = {
    "f32": "vector",
    "f16": "halfvec",
}
