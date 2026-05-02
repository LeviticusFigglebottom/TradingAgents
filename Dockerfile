FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY . .
RUN pip install --no-cache-dir .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADINGAGENTS_RESULTS_DIR=/data/logs \
    TRADINGAGENTS_CACHE_DIR=/data/cache \
    TRADINGAGENTS_MEMORY_LOG_PATH=/data/memory/trading_memory.md

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# NOTE: we deliberately run as root.
# Railway mounts persistent volumes as root-owned at /data, and there is no
# way to chown them before mount. Running as root avoids a permission-denied
# failure when the runner tries to write trace.jsonl / dashboard.html into
# the mounted volume. This container has no inbound network and only reads
# secrets from env, so root is acceptable for this workload.
WORKDIR /app
COPY --from=builder /build /app

ENTRYPOINT ["tradingagents"]
