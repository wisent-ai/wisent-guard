# Wisent-Guard

<p align="center">
  <a href="https://github.com/wisent-ai/wisent-guard/stargazers">
    <img src="https://img.shields.io/github/stars/wisent-ai/wisent-guard" alt="stars" />
  </a>
  <a href="https://pypi.org/project/wisent-guard">
    <img src="https://static.pepy.tech/badge/wisent-guard" alt="PyPI - Downloads" />
  </a>
  <br />
</p>

<p align="center">
  <img src="wisent-guard-logo.png" alt="Wisent Guard" width="200">
</p>

A Python package for latent space monitoring and guardrails. Delivered to you by the [Wisent](https://wisent.ai) team led by [Lukasz Bartoszcze](https://lukaszbartoszcze.com).

## Overview

Wisent-Guard allows you to control your AI by identifying brain patterns corresponding to responses you don't like, like hallucinations or harmful outputs. We use contrastive pairs of representations to detect when a model might be generating harmful content or hallucinating. Learn more at https://www.wisent.ai/wisent-guard.

## ✨ New: MCP Server for Self-Reflection

🚀 **Wisent-Guard now includes an MCP (Model Control Protocol) server that enables models to perform self-reflection and behavior editing on their own outputs!**

### Key Features:
- 🔍 **Hallucination Detection**: Automatically detect factual errors, impossible dates, and anachronisms
- 🛡️ **Behavior Analysis**: Identify bias, toxicity, repetition, and nonsense in real-time
- ✏️ **Auto-Improvement**: Use steering methods (CAA, K-Steering) to automatically fix problematic responses
- 📊 **Performance Monitoring**: Track reflection history and improvement success rates

### Quick Start:
```bash
# Run the demo
python -m wisent_guard.core.mcp.demo

# Start MCP server
python -m wisent_guard.core.mcp.server

# Interactive testing
python -m wisent_guard.core.mcp.demo interactive
```

See `wisent_guard/core/mcp/README.md` for full documentation.  


## License

This project is licensed under the MIT License - see the LICENSE file for details. 
