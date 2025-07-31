import os
from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read version from __init__.py
with open(os.path.join("wisent_guard", "__init__.py"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="wisent-guard",
    version=version,
    author="Wisent Team",
    author_email="your.email@example.com",  # Replace with your email
    description="Monitor and guard against harmful content in language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wisent-activation-guardrails",  # Replace with your GitHub repo
    packages=find_packages(exclude=["patches", "patches.*"]),  # Exclude patches directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "tqdm>=4.50.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0",
        "numpy>=1.21.0",
        "datasets>=2.0.0",
        "sentence-transformers>=2.0.0",
        "faiss-cpu>=1.7.0",
    ],
    extras_require={
        "harness": [
            "lm-eval==0.4.8",
        ],
    },
    keywords="nlp, machine learning, language models, safety, guardrails, lm-evaluation-harness",
) 