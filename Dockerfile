# Custom PostgreSQL image with both vchord and pgvectorscale (DiskANN)
FROM timescale/timescaledb-ha:pg17

USER root

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Detect architecture and install pgvectorscale 0.9.0 pre-built binary
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then \
        PGVS_ARCH="arm64"; \
    else \
        PGVS_ARCH="amd64"; \
    fi && \
    curl -L https://github.com/timescale/pgvectorscale/releases/download/0.9.0/pgvectorscale-0.9.0-pg17-${PGVS_ARCH}.zip -o /tmp/pgvectorscale.zip && \
    unzip /tmp/pgvectorscale.zip -d /tmp/pgvectorscale && \
    dpkg -i /tmp/pgvectorscale/pgvectorscale-postgresql-17_0.9.0-Linux_${PGVS_ARCH}.deb && \
    rm -rf /tmp/pgvectorscale*

# Install vchord from tensorchord's vchord-suite
# Copy vchord binaries from the official image
COPY --from=tensorchord/vchord-suite:pg17-latest /usr/lib/postgresql/17/lib/vchord.so /usr/lib/postgresql/17/lib/
COPY --from=tensorchord/vchord-suite:pg17-latest /usr/share/postgresql/17/extension/vchord* /usr/share/postgresql/17/extension/

USER postgres
WORKDIR /
