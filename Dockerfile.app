# CapRover passes this arg so webhook deploys can pin an immutable GHCR image.
ARG BROWSER_USE_BASE_IMAGE=browseruse/browseruse:latest
FROM ${BROWSER_USE_BASE_IMAGE}

ARG BROWSER_USE_BASE_IMAGE
ARG CACHE_BUSTER=1

USER root

RUN echo "base-image=${BROWSER_USE_BASE_IMAGE} cache-buster=${CACHE_BUSTER}" > /tmp/.caprover-cache-buster

# Install web server components and ensure the core library is up-to-date
RUN uv pip install --upgrade browser-use uvicorn gradio

WORKDIR /app
COPY app.py .

# --- NEW: Fix permissions for uv cache ---
# We point the cache to /app/.cache and make sure the user owns it
ENV UV_CACHE_DIR=/app/.cache/uv
RUN mkdir -p /app/.cache/uv && chown -R browseruse:browseruse /app

# Ensure /data is also owned by the user
RUN mkdir -p /data/profiles && chown -R browseruse:browseruse /data

ENV BROWSER_USE_HEADLESS=true \
    PYTHONUNBUFFERED=1

USER browseruse

EXPOSE 80

# Start the Hub
ENTRYPOINT ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
