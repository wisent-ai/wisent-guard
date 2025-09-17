#!/bin/bash

# LiveCodeBench benchmark runner script
# This script provides easy commands to run the benchmark with different configurations

set -e

echo "🚀 LiveCodeBench Benchmark Runner"
echo "=================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed or not in PATH"
    exit 1
fi

# Default parameters
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-1B"}
RELEASE_VERSION=${RELEASE_VERSION:-"release_v1"}
LIMIT=${LIMIT:-10}
MAX_TOKENS=${MAX_TOKENS:-512}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --release-version)
            RELEASE_VERSION="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model-name      HuggingFace model name (default: meta-llama/Llama-3.2-1B)"
            echo "  --release-version LiveCodeBench release version (default: release_v1)"
            echo "  --limit          Number of problems to process (default: 10)"
            echo "  --max-tokens     Maximum tokens per generation (default: 512)"
            echo "  --build          Force rebuild of Docker image"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  MODEL_NAME, RELEASE_VERSION, LIMIT, MAX_TOKENS can be set instead of flags"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with defaults"
            echo "  $0 --limit 20 --build               # Process 20 problems and rebuild"
            echo "  MODEL_NAME=meta-llama/Llama-3.2-3B $0  # Use different model"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Release Version: $RELEASE_VERSION"
echo "  Problem Limit: $LIMIT"
echo "  Max Tokens: $MAX_TOKENS"
echo ""

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    echo "⚠️ No NVIDIA GPU detected. Will run on CPU (slower)."
fi
echo ""

# Create results directory
mkdir -p results

# Run the benchmark
echo "🔄 Starting benchmark runner..."
docker-compose run ${BUILD_FLAG} benchmark-runner \
    --model-name "$MODEL_NAME" \
    --release-version "$RELEASE_VERSION" \
    --limit "$LIMIT" \
    --max-tokens "$MAX_TOKENS"

echo ""
echo "✅ Benchmark completed! Check the results/ directory for output files."