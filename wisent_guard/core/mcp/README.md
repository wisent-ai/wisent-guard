# Wisent-Guard MCP Server

A Model Control Protocol (MCP) server that enables models to perform self-reflection and behavior editing using wisent-guard capabilities.

## Overview

The Wisent-Guard MCP server provides tools for models to:

1. **Self-Reflection Analysis** - Analyze their own responses for hallucinations, quality issues, and problematic behaviors
2. **Hallucination Detection** - Specialized detection of factual errors and knowledge inconsistencies  
3. **Behavior Editing** - Use steering methods (CAA, K-Steering) to improve response quality
4. **Comprehensive Analysis** - Multi-dimensional quality assessment with improvement suggestions
5. **Performance Tracking** - Monitor self-reflection operations and effectiveness

## Architecture

```
wisent_guard/core/mcp/
├── __init__.py          # Module exports
├── server.py            # Main MCP server implementation
├── client.py            # Client for interacting with server
├── tools.py             # Individual tool implementations
├── examples.py          # Usage examples and demos
└── README.md           # This documentation
```

## Installation

The MCP server requires the `mcp` package:

```bash
pip install mcp
```

## Quick Start

### Running the Server

```python
# Start the MCP server
python -m wisent_guard.core.mcp.server
```

### Using the Client

```python
import asyncio
from wisent_guard.core.mcp.client import WisentGuardMCPClient, SelfReflectionRequest

async def example():
    async with WisentGuardMCPClient() as client:
        # Quick hallucination check
        request = SelfReflectionRequest(
            response_text="The moon is made of cheese and was discovered in 1969.",
            original_prompt="Tell me about the moon.",
            analysis_depth="standard"
        )
        
        result = await client.perform_self_reflection(request)
        print(f"Hallucinations detected: {result['hallucination_analysis']['is_hallucinating']}")

asyncio.run(example())
```

## Available Tools

### 1. Self-Reflection Analysis (`perform_self_reflection`)

Comprehensive analysis of model responses including hallucination detection, behavior analysis, and quality assessment.

**Parameters:**
- `response_text` (required): The model response to analyze
- `original_prompt` (optional): The original prompt that generated the response
- `analysis_depth`: "quick", "standard", or "comprehensive"
- `focus_areas`: List of areas to focus on (hallucinations, coherence, accuracy, bias)

**Returns:**
- Hallucination analysis results
- Quality assessment scores
- Behavior analysis findings
- Overall reflection score with recommendations

### 2. Hallucination Detection (`detect_hallucinations`)

Specialized detection of hallucinations, factual errors, and knowledge inconsistencies.

**Parameters:**
- `response_text` (required): The response to check for hallucinations
- `knowledge_domain`: Domain for specialized checking ("general", "science", "history", etc.)
- `fact_check_level`: "basic", "intermediate", or "advanced"
- `reference_context`: Reference context for comparison

**Returns:**
- Hallucination detection results
- Enhanced fact-checking analysis
- Domain-specific issue detection
- Confidence scores

### 3. Behavior Editing (`edit_behavior`)

Edit model behavior using steering methods to improve response quality.

**Parameters:**
- `original_prompt` (required): The original prompt
- `current_response` (required): The current problematic response
- `desired_changes` (required): List of desired changes
- `steering_method`: "CAA", "KSteering", or "auto"
- `steering_strength`: Strength of intervention (0.1-3.0)
- `max_attempts`: Maximum editing attempts (1-5)

**Returns:**
- Original and edited responses
- Improvement scores
- Steering method used
- Success status

### 4. Comprehensive Analysis (`analyze_response_comprehensive`)

Multi-dimensional quality assessment with improvement suggestions.

**Parameters:**
- `response_text` (required): The response to analyze
- `original_prompt`: The original prompt
- `analysis_categories`: Categories to analyze (quality, safety, accuracy, helpfulness)
- `include_suggestions`: Whether to include improvement suggestions
- `compare_to_baseline`: Whether to compare against baseline performance

**Returns:**
- Overall quality metrics
- Category-specific scores
- Improvement suggestions
- Baseline comparisons

### 5. Reflection History (`get_reflection_history`)

Access history of self-reflection analyses.

**Parameters:**
- `limit`: Maximum number of entries to return
- `filter_by`: Filter by type ("all", "hallucinations", "behavior_issues", "high_confidence")

**Returns:**
- Historical reflection results
- Entry counts and filters applied

### 6. Performance Metrics (`get_performance_metrics`)

Get performance metrics for self-reflection operations.

**Parameters:**
- `include_detailed`: Whether to include detailed timing and memory metrics

**Returns:**
- Operation counts and success rates
- Detailed performance metrics (if requested)

## Usage Examples

### Basic Self-Reflection

```python
async def basic_example():
    async with WisentGuardMCPClient() as client:
        result = await client.perform_self_reflection(
            SelfReflectionRequest(
                response_text="Your response here",
                analysis_depth="standard",
                focus_areas=["hallucinations", "accuracy"]
            )
        )
        
        if result["overall_reflection_score"]["needs_improvement"]:
            print("Response needs improvement!")
```

### Hallucination Detection

```python
async def hallucination_example():
    async with WisentGuardMCPClient() as client:
        result = await client.detect_hallucinations(
            response_text="The Great Wall of China was built in 1969.",
            knowledge_domain="history",
            fact_check_level="advanced"
        )
        
        if result["hallucination_detection"]["is_hallucinating"]:
            print("Hallucinations detected!")
            print(f"Issues: {result['hallucination_detection']['issues_detected']}")
```

### Behavior Editing

```python
async def editing_example():
    async with WisentGuardMCPClient() as client:
        result = await client.edit_behavior(
            BehaviorEditRequest(
                original_prompt="Explain vaccines",
                current_response="Vaccines are dangerous and cause autism",
                desired_changes=["more factual", "evidence-based"],
                steering_method="CAA"
            )
        )
        
        if result["behavior_editing"]["editing_successful"]:
            print("Successfully improved response!")
            print(f"New response: {result['behavior_editing']['best_result']['edited_response']}")
```

### Auto-Improvement Workflow

```python
async def auto_improvement_example():
    async with WisentGuardMCPClient() as client:
        result = await client.auto_improve_response(
            original_prompt="Tell me about climate change",
            current_response="Climate change is fake news made up by scientists",
            max_iterations=3
        )
        
        improvement = result["auto_improvement"]
        print(f"Original: {improvement['original_response']}")
        print(f"Improved: {improvement['final_response']}")
        print(f"Success: {improvement['successful']}")
```

### Quick Convenience Functions

```python
from wisent_guard.core.mcp.client import quick_hallucination_check, auto_fix_response

# Quick hallucination check
is_hallucinating = await quick_hallucination_check("The moon is made of cheese")

# Auto-fix problematic response
fixed_response = await auto_fix_response(
    "Tell me about vaccines", 
    "Vaccines are bad and cause autism"
)
```

## Integration with Other Models

The MCP server can be used by any model that supports MCP protocol:

### Claude with MCP

```json
{
  "mcpServers": {
    "wisent-guard": {
      "command": "python",
      "args": ["-m", "wisent_guard.core.mcp.server"]
    }
  }
}
```

### Custom Integration

```python
# Custom model integration
class SelfReflectiveModel:
    def __init__(self):
        self.mcp_client = WisentGuardMCPClient()
    
    async def generate_with_reflection(self, prompt):
        # Generate initial response
        response = self.generate(prompt)
        
        # Self-reflect on response
        quick_check = await self.mcp_client.quick_self_check(response, prompt)
        
        if quick_check["needs_regeneration"]:
            # Auto-improve the response
            improved = await self.mcp_client.auto_improve_response(prompt, response)
            return improved["auto_improvement"]["final_response"]
        
        return response
```

## Configuration

### Server Configuration

```python
# Custom server configuration
server = WisentGuardMCPServer(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    default_layer=15,
    enable_tracking=True
)
```

### Client Configuration

```python
# Custom client configuration
client = WisentGuardMCPClient(
    server_command=["python", "-m", "wisent_guard.core.mcp.server", "--custom-args"]
)
```

## Performance Considerations

### Memory Usage
- The server loads a full language model for steering operations
- Memory usage scales with model size and batch size
- Consider using smaller models for real-time applications

### Latency
- Self-reflection analysis: ~100-500ms per response
- Behavior editing: ~1-5 seconds depending on steering method
- Hallucination detection: ~50-200ms per response

### Optimization Tips
1. Use "quick" analysis depth for real-time applications
2. Cache reflection results for similar responses
3. Use batch processing for multiple responses
4. Enable performance tracking to identify bottlenecks

## Error Handling

The MCP server includes comprehensive error handling:

```python
try:
    result = await client.perform_self_reflection(request)
except RuntimeError as e:
    print(f"Client not connected: {e}")
except Exception as e:
    print(f"Analysis failed: {e}")
```

Common error scenarios:
- Model loading failures
- Memory limitations
- Steering method errors
- Network connectivity issues

## Monitoring and Debugging

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

```python
# Get detailed performance metrics
metrics = await client.get_performance_metrics(include_detailed=True)
print(f"Average analysis time: {metrics['detailed_timing']['mean_latency']}")
```

### Reflection History Analysis

```python
# Analyze reflection patterns
history = await client.get_reflection_history(limit=100, filter_by="hallucinations")
hallucination_rate = len(history["reflection_history"]) / history["total_entries"]
print(f"Hallucination rate: {hallucination_rate:.2%}")
```

## Advanced Features

### Custom Steering Methods

```python
# Extend with custom steering methods
class CustomSteeringMethod:
    def train(self, pairs, layer):
        # Custom training logic
        pass
    
    def apply_steering(self, activations, strength):
        # Custom steering logic
        pass

# Register custom method
server.register_steering_method("custom", CustomSteeringMethod)
```

### Domain-Specific Analysis

```python
# Domain-specific hallucination detection
result = await client.detect_hallucinations(
    response_text="Medical advice here",
    knowledge_domain="medicine",
    fact_check_level="advanced"
)
```

### Baseline Comparison

```python
# Compare against established baselines
result = await client.analyze_response_comprehensive(
    response_text="Your response",
    compare_to_baseline=True
)

baseline_comparison = result["baseline_comparison"]
for metric, comparison in baseline_comparison.items():
    if comparison["improvement"]:
        print(f"{metric}: Improved by {comparison['difference']:.2f}")
```

## Troubleshooting

### Common Issues

1. **MCP Package Not Found**
   ```bash
   pip install mcp
   ```

2. **Model Loading Errors**
   - Check available GPU memory
   - Try smaller model variants
   - Verify model name is correct

3. **Steering Method Failures**
   - Ensure sufficient training data
   - Check layer compatibility
   - Verify steering strength is reasonable

4. **Connection Issues**
   - Verify server is running
   - Check server command configuration
   - Review firewall/network settings

### Debug Mode

```python
# Enable debug mode for detailed output
async with WisentGuardMCPClient() as client:
    client.debug_mode = True
    result = await client.perform_self_reflection(request)
```

## Contributing

To extend the MCP server:

1. Add new tools in `tools.py`
2. Register tools in `server.py`
3. Update client methods in `client.py`
4. Add examples in `examples.py`
5. Update documentation

## License

This MCP server is part of the Wisent-Guard project and follows the same license terms. 