#!/bin/bash
# Script to build and upload wisent-guard to PyPI

# Ensure the latest build tools are installed
echo "Installing/upgrading build tools..."
pip install --upgrade setuptools wheel twine build

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Build the package
echo "Building the package..."
python -m build

# Display the files to be uploaded
echo "Files to be uploaded:"
ls -l dist/

# Check if any of the files already exist on PyPI
echo "Checking if wisent-guard already exists on PyPI..."
pip_output=$(pip install wisent-guard 2>&1)
if [[ $pip_output != *"No matching distribution found"* ]]; then
  echo "WARNING: wisent-guard already exists on PyPI. This will update the package."
  echo "Version in __init__.py: $(grep -o "__version__ = \".*\"" wisent_guard/__init__.py)"
  echo "Make sure you've updated the version number to avoid conflicts."
  
  echo -n "Continue with upload? (y/n): "
  read answer
  if [[ "$answer" != "y" ]]; then
    echo "Upload aborted."
    exit 1
  fi
fi

# Upload to PyPI
echo "Uploading to PyPI..."
echo "NOTE: You will need to enter your PyPI credentials."
python -m twine upload dist/*

echo "Upload process completed. Check the output for any errors." 