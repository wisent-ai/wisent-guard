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

## LM-Harness Integration

Wisent-Guard now supports standardized benchmarking through [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Train guardrails on established benchmarks and get objective performance metrics.

```bash
# Install with benchmark support
pip install wisent-guard[harness]

# Test on TruthfulQA (hallucination detection)
python -m wisent_guard tasks truthfulqa --model gpt2 --layer 6 --limit 50

# Multiple benchmarks
python -m wisent_guard tasks hellaswag,truthfulqa --model gpt2 --layer 6 --limit 50
```

**Supported benchmarks:**
- `truthfulqa` - Detect hallucinations vs truthful responses (90% detection rate)
- `hellaswag` - Common sense reasoning errors (100% detection rate)  
- `arc_easy/challenge` - Science reasoning mistakes
- `winogrande` - Language understanding errors
- `piqa` - Physical reasoning mistakes

Results are saved as JSON, CSV, and Markdown reports with detailed metrics.

Contributions are welcome! Please feel free to submit a Pull Request or open an issue so that we can fix all bugs potentially troubling you.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
