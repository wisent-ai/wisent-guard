# Steering Methods Validation Tests

This directory contains validation tests for steering method implementations in wisent-guard.

## Running Tests

1. Generate test data first with `python generate_data_with_original_{method_implementation}.py`
2. Run all steering validation tests: `pytest tests/steering_validation/ -m "slow and heavy"`

**Note**: These tests require GPU availability and are resource-intensive.

## How to Implement a New Steering Method - Advice from Previous Implementations

1. Clone the original implementation repository.
2. Create a generation script that will:
   - Generate steering vectors
   - Generate steering examples
   - Produce other tests if needed
3. Create a steering vector/tensor test - compare the vector/tensor created with the original implementation against the wisent_guard implementation.
4. Create a steering generation test - compare generation made with the original repository against the wisent_guard implementation.

I strongly advise splitting the implementation into these phases.

### Key Insights from Previous Implementations

1. **Start with Real Models**: Mock tests are insufficient - use actual transformer models from the beginning.
2. **Reference Comparison is Ultimate Validation**: Achieving exact token match proves implementation correctness.
3. **Tokenization Matters**: Exact format matching with reference implementation is crucial for behavioral equivalence.
