from setuptools import setup, find_packages

setup(
    name="wisent-guard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.5.0",
    ],
    author="Wisent Team",
    author_email="info@wisentai.com",
    description="A package for guarding against harmful content in language models using activation monitoring",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wisent/wisent-guard",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 