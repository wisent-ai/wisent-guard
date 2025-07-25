# Multi-stage Dockerfile for Level 1 optimization
# Reduces image size and improves startup time

# Build stage - includes build tools and dependencies
FROM python:3.12-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3-dev \
    linux-headers \
    git

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages in virtual environment (minimal set)
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir \
        timeout-decorator==0.5.0 \
    && pip cache purge

# Runtime stage - minimal image without build tools
FROM python:3.12-alpine AS runtime

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Security: Create non-root user
RUN addgroup -g 1001 coderunner && \
    adduser -u 1001 -G coderunner -s /bin/sh -D coderunner

# Security: Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HOME=/home/coderunner

# Create secure workspace
RUN mkdir -p /home/coderunner/workspace && \
    chmod 750 /home/coderunner/workspace && \
    chown -R coderunner:coderunner /home/coderunner

# Set working directory
WORKDIR /home/coderunner/workspace

# Security: Minimal healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=2 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Switch to non-root user
USER 1001:1001

# Default command
CMD ["python", "-c", "print('Optimized secure code execution container ready')"]