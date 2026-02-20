# Use the official pre-built image
FROM browseruse/browseruse:latest

USER root

# Install web server components
RUN uv pip install uvicorn gradio

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
