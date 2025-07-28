# %% [markdown]
# # 🚀 Comprehensive Evaluation Framework for Wisent Guard
# 
# This notebook provides an interactive interface for running comprehensive evaluations that properly separate:
# 
# 1. **🎯 Benchmark Performance**: How well the model solves mathematical problems
# 2. **🔍 Probe Performance**: How well probes detect correctness from model activations
# 3. **⚙️ DAC Hyperparameter Optimization**: Grid search to find optimal DAC configurations
# 
# ## Key Features:
# - **Real Data Integration**: Uses GSM8KExtractor to get contrastive pairs from training data
# - **DAC Hyperparameter Grid Search**: Systematic optimization of entropy_threshold, ptop, and max_alpha
# - **Real-time Progress**: Live updates during evaluation with tqdm
# - **Rich Visualizations**: Comprehensive plots and analysis
# - **Modular Design**: Clean separation of concerns
# - **Export Results**: Save results and generate reports
# 
# ## DAC Hyperparameters:
# - **entropy_threshold**: Controls dynamic steering based on entropy (default: 1.0)
# - **ptop**: Probability threshold for KL-based dynamic control (default: 0.4)
# - **max_alpha**: Maximum steering intensity (default: 2.0)

# %% [markdown]
# ## 📋 Setup and Imports

# %%
# Core imports
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Set HuggingFace cache to permanent directory
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/transformers'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

# Create cache directories if they don't exist
os.makedirs('/workspace/.cache/huggingface/transformers', exist_ok=True)
os.makedirs('/workspace/.cache/huggingface/datasets', exist_ok=True)

# Add project root to path
project_root = '/workspace/wisent-guard'
if project_root not in sys.path:
    sys.path.append(project_root)

# Fix wandb connection issues
def fix_wandb_connection():
    """Fix wandb connection issues by properly initializing or disabling it."""
    try:
        import wandb
        
        # Check if wandb is already initialized
        if wandb.run is not None:
            print("⚠️ Cleaning up existing wandb run...")
            wandb.finish()
        
        # Clear any broken connections
        import subprocess
        import signal
        try:
            # Kill any hanging wandb processes
            subprocess.run(['pkill', '-f', 'wandb'], capture_output=True)
        except:
            pass
        
        print("✅ Wandb connection cleaned up")
        return True
    except Exception as e:
        print(f"⚠️ Wandb cleanup warning: {e}")
        return False

# Check HuggingFace authentication
def check_hf_auth():
    """Check if user is logged into HuggingFace and show login instructions if needed."""
    try:
        import subprocess
        result = subprocess.run(['huggingface-cli', 'whoami'], capture_output=True, text=True)
        if result.returncode == 0:
            username = result.stdout.strip()
            print(f"✅ Logged into HuggingFace as: {username}")
            return True
        else:
            print("⚠️ Not logged into HuggingFace!")
            print("🔐 Please run: huggingface-cli login")
            print("   This is required to access datasets like AIME 2024/2025")
            return False
    except Exception as e:
        print(f"⚠️ Could not check HuggingFace authentication: {e}")
        print("🔐 If you encounter dataset loading issues, try: huggingface-cli login")
        return False

# Clean up wandb first
wandb_ok = fix_wandb_connection()

# Import comprehensive evaluation framework
from wisent_guard.core.evaluation.comprehensive import (
    ComprehensiveEvaluationConfig,
    ComprehensiveEvaluationPipeline,
    plot_evaluation_results,
    create_results_dashboard,
    generate_summary_report,
    calculate_comprehensive_metrics,
    generate_performance_summary
)

# Visualization and interactivity
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from IPython.display import display, HTML, Markdown

# Data manipulation
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Utilities
from tqdm.notebook import tqdm
import logging

print("✅ All imports successful!")
print(f"📍 Working directory: {os.getcwd()}")
print(f"🐍 Python version: {sys.version}")
print(f"💾 HuggingFace cache: {os.environ['HF_HOME']}")
print(f"🔗 Wandb status: {'✅ Ready' if wandb_ok else '⚠️ May have issues'}")
print()

# Check authentication
hf_authenticated = check_hf_auth()

# %% [markdown]
# ## ⚙️ Configuration
# 
# Edit the constants in the next cell to customize your evaluation. All parameters are clearly documented with examples.

# %%
# Configuration Constants - Edit these values to customize your evaluation

# Import math tasks from our task configuration
import sys
sys.path.append('/workspace/wisent-guard')
from wisent_guard.parameters.task_config import MATH_TASKS

# Import DAC steering method
from wisent_guard.core.steering_methods.dac import DAC

# Convert MATH_TASKS set to sorted list for easier selection
MATH_TASKS_LIST = sorted(list(MATH_TASKS))
print(f"📚 Available math tasks: {MATH_TASKS_LIST}")

# ============================================================================
# MAIN CONFIGURATION - Edit these constants to customize your evaluation
# ============================================================================

# Model configuration
MODEL_NAME = 'distilbert/distilgpt2'  # Using smaller model for quick testing - Examples: 'distilbert/distilgpt2', 'gpt2', '/workspace/models/llama31-8b-instruct-hf', 'Qwen/Qwen3-8B'
MODEL_NAME = "/workspace/models/llama31-8b-instruct-hf"

# Dataset configuration - Choose from MATH_TASKS
TRAIN_DATASET = 'gsm8k'     # Training dataset - Change to any task from MATH_TASKS_LIST
VAL_DATASET = 'gsm8k'       # Validation dataset - Change to any task from MATH_TASKS_LIST  
TEST_DATASET = 'gsm8k'      # Test dataset - Change to any task from MATH_TASKS_LIST

# Validate dataset choices
for dataset, name in [(TRAIN_DATASET, 'TRAIN'), (VAL_DATASET, 'VAL'), (TEST_DATASET, 'TEST')]:
    if dataset not in MATH_TASKS:
        raise ValueError(f"{name}_DATASET '{dataset}' not in MATH_TASKS. Choose from: {MATH_TASKS_LIST}")

# Sample limits (small for quick testing)
TRAIN_LIMIT = 5   # Number of training samples
VAL_LIMIT = 5    # Number of validation samples  
TEST_LIMIT = 20     # Number of test samples

# Layer configuration - specify which layers to search during optimization
PROBE_LAYERS = [15]     # Examples: [2, 3, 4, 5], [8, 16, 24, 32], [5, 6, 7, 8] 
STEERING_LAYERS = [15]  # Same as probe layers for now - Examples: [3, 4, 5], [16, 24, 32], [6, 8, 10]

# DAC Hyperparameters - specify arrays of values to search
ENTROPY_THRESHOLDS = [1.0]    # Examples: [0.5, 1.0, 1.5], [1.0, 2.0], [0.8, 1.2]
PTOP_VALUES = [1.0]            # Examples: [0.3, 0.4, 0.5], [0.4], [0.2, 0.6]  
MAX_ALPHA_VALUES = [0.5]       # Examples: [1.5, 2.0, 2.5], [2.0], [1.0, 3.0]

# Options
ENABLE_WANDB = False                        # Disable for quick testing
EXPERIMENT_NAME = 'dac_hyperparameter_search'  # Experiment name for logging
BATCH_SIZE = 4

# ============================================================================
# DATASET SIZE MAPPING (for validation - don't edit unless adding new datasets)
# ============================================================================



# ============================================================================
# AUTO-VALIDATION AND INFO
# ============================================================================

def detect_model_layers(model_name):
    """Detect number of layers in a model without loading it fully."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        
        # Different models store layer count differently
        if hasattr(config, 'n_layer'):
            return config.n_layer
        elif hasattr(config, 'num_hidden_layers'):
            return config.num_hidden_layers
        elif hasattr(config, 'num_layers'):
            return config.num_layers
        else:
            return "Unknown"
    except Exception as e:
        return f"Error: {str(e)}"

# Validate configuration
print("📋 CONFIGURATION SUMMARY")
print("="*50)
print(f"🤖 Model: {MODEL_NAME}")
print(f"📊 Datasets: {TRAIN_DATASET} → {VAL_DATASET} → {TEST_DATASET}")
print(f"🔢 Samples: {TRAIN_LIMIT} + {VAL_LIMIT} + {TEST_LIMIT} = {TRAIN_LIMIT + VAL_LIMIT + TEST_LIMIT} total")
print(f"🎯 Probe layers: {PROBE_LAYERS}")
print(f"⚙️ Steering layers: {STEERING_LAYERS}")
print(f"🎛️ Steering method: DAC (Dynamic Activation Composition)")
print(f"📊 DAC Hyperparameters:")
print(f"   • Entropy thresholds: {ENTROPY_THRESHOLDS}")  
print(f"   • Ptop values: {PTOP_VALUES}")
print(f"   • Max alpha values: {MAX_ALPHA_VALUES}")
print(f"📚 Using tasks from MATH_TASKS: ✓")

# Calculate total combinations
total_combinations = (len(STEERING_LAYERS) * 
                     len(ENTROPY_THRESHOLDS) *
                     len(PTOP_VALUES) *
                     len(MAX_ALPHA_VALUES) *
                     len(PROBE_LAYERS) * 
                     3)  # Assuming 3 probe C values

print(f"🧪 Total hyperparameter combinations: {total_combinations}")
print(f"📈 Wandb enabled: {ENABLE_WANDB}")

# Model info
try:
    num_layers = detect_model_layers(MODEL_NAME)
    print(f"🏗️ Model layers: {num_layers}")
    if isinstance(num_layers, int):
        max_probe_layer = max(PROBE_LAYERS) if PROBE_LAYERS else 0
        max_steering_layer = max(STEERING_LAYERS) if STEERING_LAYERS else 0
        if max_probe_layer >= num_layers or max_steering_layer >= num_layers:
            print("⚠️ WARNING: Some configured layers exceed model size!")
except Exception as e:
    print(f"⚠️ Could not detect model layers: {e}")

print("\n✅ Configuration validated successfully!")
print("💡 This evaluation now uses REAL mathematical training data instead of synthetic generation!")
print("🎯 DAC will be trained on actual math questions from your training dataset.")
print("💡 Configuration updated for quick testing with distilgpt2 and GSM8K dataset.")
print(f"🎯 Currently configured for: {TRAIN_DATASET}")
print(f"📚 Available math tasks: {len(MATH_TASKS_LIST)} tasks including GSM8K, MATH-500, AIME, etc.")
print("💡 To change datasets, edit TRAIN_DATASET, VAL_DATASET, TEST_DATASET above.")

# %% [markdown]
# ## 🛠️ Create Configuration
# 
# Configuration is automatically created from the constants defined above.

# %%
# Create configuration from constants

config = ComprehensiveEvaluationConfig(
    model_name=MODEL_NAME,
    train_dataset=TRAIN_DATASET,
    val_dataset=VAL_DATASET,
    test_dataset=TEST_DATASET,
    train_limit=TRAIN_LIMIT,
    val_limit=VAL_LIMIT,
    test_limit=TEST_LIMIT,
    probe_layers=PROBE_LAYERS,
    steering_layers=STEERING_LAYERS,
    steering_methods=["dac"],  # Fixed to DAC only
    # DAC hyperparameters
    dac_entropy_thresholds=ENTROPY_THRESHOLDS,
    dac_ptop_values=PTOP_VALUES,
    dac_max_alpha_values=MAX_ALPHA_VALUES,
    enable_wandb=ENABLE_WANDB,
    experiment_name=EXPERIMENT_NAME,
    batch_size=BATCH_SIZE,
    max_length=512,
    max_new_tokens=256
)

print("✅ Configuration object created successfully!")
print("🚀 Ready to run comprehensive evaluation!")
print("\n💡 All configuration is now controlled by the constants in the previous cell.")

# %% [markdown]
# ## 🚀 Run Comprehensive Evaluation
# 
# This is the main evaluation cell. It will:
# 
# 1. **🎯 Train Probes**: Train correctness classifiers on all specified layers
# 2. **⚙️ Optimize Hyperparameters**: Grid search for best steering + probe combinations
# 3. **🏆 Final Evaluation**: Test optimized configuration on held-out test set
# 
# **Note**: This may take several minutes depending on your configuration.

# %%
print("\n" + "="*80)
print("🚀 STARTING COMPREHENSIVE EVALUATION WITH REAL MATHEMATICAL DATA")
print("="*80)
print(f"✅ DAC will be trained on actual {TRAIN_DATASET.upper()} mathematical questions from training data")
print(f"🎯 Using task extractors for proper format handling across datasets")
print("="*80)

# Initialize pipeline
pipeline = ComprehensiveEvaluationPipeline(config)

# Run evaluation with progress tracking
try:
    results = pipeline.run_comprehensive_evaluation()
    print("\n" + "="*80)
    print("✅ Evaluation completed successfully!")
    
    # Store results for analysis
    evaluation_results = results
    
except Exception as e:
    print(f"\n❌ Evaluation failed: {str(e)}")
    print("Check the logs above for more details.")
    raise

# %% [markdown]
# ## 📊 Results Analysis
# 
# Now let's analyze the results with comprehensive metrics and visualizations.

# %%
# Calculate comprehensive metrics
comprehensive_metrics = calculate_comprehensive_metrics(evaluation_results)

# Generate performance summary
performance_summary = generate_performance_summary(comprehensive_metrics)
print(performance_summary)

