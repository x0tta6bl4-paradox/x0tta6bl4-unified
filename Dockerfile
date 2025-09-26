# Multi-stage build for x0tta6bl4-unified platform

# Build stage
FROM python:3.12-slim as builder

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install setuptools for building packages
RUN pip install --no-cache-dir setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements_min.txt .
RUN pip install --no-cache-dir --user -r requirements_min.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Add Python packages to PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs && chmod 755 /app/logs

# Create non-root user
RUN useradd -m -u 1000 x0tta6bl4 && chown -R x0tta6bl4:x0tta6bl4 /app
USER x0tta6bl4

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "-c", "import sys; sys.path.insert(0, '/root/.local/lib/python3.12/site-packages'); import uvicorn; uvicorn.run('main:app', host='0.0.0.0', port=8000)"]
