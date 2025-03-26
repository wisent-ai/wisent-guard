#!/usr/bin/env python3
"""
Script to split datasets into training and evaluation sets with a fixed proportion.
"""

import argparse
import os
import random
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_dataset(file_path):
    """
    Load dataset from file based on its extension.
    Supports CSV, JSON, and JSONL formats.
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        data: Loaded dataset
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_ext in ['.jsonl', '.jsonlines']:
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    elif file_ext in ['.txt', '.text']:
        with open(file_path, 'r') as f:
            return f.read().splitlines()
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def save_dataset(data, output_path, file_format):
    """
    Save dataset to file in the specified format.
    
    Args:
        data: Dataset to save
        output_path (str): Path to save the dataset
        file_format (str): Format to save the dataset (csv, json, jsonl)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if file_format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        else:
            pd.DataFrame(data).to_csv(output_path, index=False)
    elif file_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif file_format == 'jsonl':
        with open(output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    elif file_format == 'txt':
        with open(output_path, 'w') as f:
            for item in data:
                f.write(f"{item}\n")
    else:
        raise ValueError(f"Unsupported output format: {file_format}")


def split_dataset(data, eval_ratio=0.2, random_state=42):
    """
    Split dataset into training and evaluation sets.
    
    Args:
        data: Dataset to split
        eval_ratio (float): Proportion of the dataset to use for evaluation
        random_state (int): Random seed for reproducibility
        
    Returns:
        train_data, eval_data: Split datasets
    """
    if isinstance(data, pd.DataFrame):
        train_data, eval_data = train_test_split(
            data, test_size=eval_ratio, random_state=random_state
        )
    else:
        # For lists (JSON, JSONL, TXT)
        indices = list(range(len(data)))
        train_indices, eval_indices = train_test_split(
            indices, test_size=eval_ratio, random_state=random_state
        )
        
        train_data = [data[i] for i in train_indices]
        eval_data = [data[i] for i in eval_indices]
    
    return train_data, eval_data


def main():
    parser = argparse.ArgumentParser(description='Split dataset into training and evaluation sets')
    parser.add_argument('--input', '-i', required=True, help='Input dataset file path')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for split datasets')
    parser.add_argument('--eval-ratio', '-e', type=float, default=0.2, 
                        help='Proportion of data to use for evaluation (default: 0.2)')
    parser.add_argument('--output-format', '-f', choices=['csv', 'json', 'jsonl', 'txt'], 
                        help='Output format (default: same as input)')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Determine output format if not specified
    if args.output_format is None:
        args.output_format = os.path.splitext(args.input)[1].lower().lstrip('.')
        if args.output_format in ['jsonlines']:
            args.output_format = 'jsonl'
        elif args.output_format not in ['csv', 'json', 'jsonl', 'txt']:
            args.output_format = 'json'  # Default to JSON if unknown format
    
    # Load dataset
    print(f"Loading dataset from {args.input}...")
    data = load_dataset(args.input)
    
    # Split dataset
    print(f"Splitting dataset with {args.eval_ratio:.0%} for evaluation...")
    train_data, eval_data = split_dataset(data, args.eval_ratio, args.seed)
    
    # Create output file paths
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    train_output = os.path.join(args.output_dir, f"{input_name}_train.{args.output_format}")
    eval_output = os.path.join(args.output_dir, f"{input_name}_eval.{args.output_format}")
    
    # Save split datasets
    print(f"Saving training set ({len(train_data)} samples) to {train_output}...")
    save_dataset(train_data, train_output, args.output_format)
    
    print(f"Saving evaluation set ({len(eval_data)} samples) to {eval_output}...")
    save_dataset(eval_data, eval_output, args.output_format)
    
    print("Dataset splitting completed successfully.")


if __name__ == "__main__":
    main()
