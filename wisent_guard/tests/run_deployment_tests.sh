#!/bin/bash

# Deployment readiness test runner for wisent-guard
# This script runs comprehensive tests to ensure the system is ready for deployment

set -e  # Exit on any error

echo "🚀 Starting Wisent-Guard Deployment Readiness Tests..."
echo ""

# Change to the tests directory
cd "$(dirname "$0")"

# Run the deployment tests
echo "Running deployment readiness tests..."
python test_deployment_readiness.py

echo ""
echo "🎯 Running existing test suite for additional validation..."

# Run the existing tests as well
python -m unittest test_vectors.py -v

echo ""
echo "✨ All tests completed successfully!"
echo "The wisent-guard system is ready for deployment." 