FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install PyTorch with CUDA support first (CUDA 12.1 compatible versions)
RUN pip3 install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Copy wisent-guard source code from build context
COPY . /wisent-guard

# Install wisent-guard (this installs datasets>=2.0.0 and other dependencies)
RUN cd /wisent-guard && pip3 install -e .

# Force upgrade datasets to 2.17.0 to fix pyarrow compatibility and install accelerate
RUN pip3 install --force-reinstall datasets==2.17.0 accelerate


# Copy the benchmark runner application
COPY tests/benchmark_runner /app

# Set the entrypoint
ENTRYPOINT ["python3", "main.py"]