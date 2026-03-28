# Use the official pre-built image
FROM browseruse/browseruse:latest

# Add a build argument. Change its value in captain-definition to force a fresh build.
ARG CACHE_BUSTER=1

USER root

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
