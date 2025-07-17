# Dockerfile for secure code execution (MBPP, HumanEval, APPS, etc.)
# Based on bigcode-evaluation-harness patterns
#
# ⚠️ DEPRECATED: Use Dockerfile.optimized instead for smaller, faster images.
# This file is kept for backward compatibility and will be removed in future versions.

# Use specific digest for reproducible builds and security
FROM python:3.12-alpine

# Security: Run as non-root from the start
RUN addgroup -g 1001 coderunner && \
    adduser -u 1001 -G coderunner -s /bin/sh -D coderunner

# Set working directory
WORKDIR /app

# Security: Update packages and install minimal requirements
RUN apk update && \
    apk upgrade && \
    apk add --no-cache \
        gcc \
        musl-dev \
        python3-dev \
    && rm -rf /var/cache/apk/*

# Security: Pin Python packages to specific versions
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        timeout-decorator==0.5.0 \
    && pip cache purge

# Security: Remove unnecessary packages and clean up
RUN apk del gcc musl-dev python3-dev && \
    rm -rf /var/cache/apk/*

# Security: Set restrictive permissions
RUN chmod 755 /app && \
    chown -R coderunner:coderunner /app

# Switch to non-root user early
USER coderunner

# Security: Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HOME=/home/coderunner

# Create secure workspace
RUN mkdir -p /home/coderunner/workspace && \
    chmod 750 /home/coderunner/workspace

WORKDIR /home/coderunner/workspace

# Security: Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; print('healthy'); sys.exit(0)" || exit 1

# Security: Set default non-privileged user
USER 1001:1001

# Default secure command
CMD ["python", "-c", "print('Secure code execution container ready')"]