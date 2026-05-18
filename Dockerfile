FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-inference.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -q torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -q -r requirements-inference.txt

# Copy application code
COPY omnimodel/ ./omnimodel/
COPY cli.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_NUM_THREADS=1

# Memory limit: 900MB
ENV MEMORY_LIMIT=900

# Create non-root user
RUN useradd -m -u 1000 omni && chown -R omni:omni /app
USER omni

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "cli.py", "--help"]
