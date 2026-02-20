# Use the official pre-built image
FROM browseruse/browseruse:latest

# 1. Switch to root for setup and installation
USER root

# 2. Install web server components
RUN uv pip install uvicorn gradio

# 3. Prepare the workspace
WORKDIR /app
COPY app.py .

# 4. Ensure the non-root user (browseruse) owns the app and data directories
# The base image already creates the 'browseruse' user and /data volume
RUN chown -R browseruse:browseruse /app /data

# 5. Environment Overrides
ENV BROWSER_USE_HEADLESS=true \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

# 6. SWITCH TO NON-ROOT USER
USER browseruse

# CapRover usually likes port 80 or 3000
EXPOSE 80

# 7. Start the Hub
# uv run ensures we use the correct environment created by the base image
ENTRYPOINT ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
