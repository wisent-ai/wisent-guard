#!/usr/bin/env python
"""
Update all example files to use meta-llama/Llama-3.1-8B-Instruct, layer 15, and MPS.

This script applies consistent changes to all example files to ensure they work
with the Llama-3.1-8B-Instruct model, monitor layer 15, and properly utilize
MPS for Apple Silicon GPUs.
"""

import os
import re
import glob
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Update all examples to use Llama-3.1 and layer 15")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying them")
    parser.add_argument("--backup", action="store_true", help="Create backups of original files")
    parser.add_argument("--fix", action="store_true", help="Fix previously updated files")
    return parser.parse_args()

def update_examples(dry_run=False, backup=False, fix=False):
    """Update all example files with the new model and layer configuration."""
    example_files = glob.glob("examples/*.py")
    
    # Skip files that are already updated or don't need updating
    skip_files = [
        "examples/mps_complete_example.py",
        "examples/standalone_test.py",
        "examples/mps_layer15_example.py",
        "examples/mps_simplified_example.py",
        "examples/mps_patch_example.py"
    ]
    
    if not fix:
        example_files = [f for f in example_files if f not in skip_files]
    else:
        # If fixing, just process files that might have been incorrectly updated
        example_files = [f for f in example_files if f in ["examples/basic_usage.py"]]
    
    print(f"Found {len(example_files)} example files to update")
    
    for file_path in example_files:
        print(f"\nProcessing {file_path}...")
        
        # Create backup if requested
        if backup and not dry_run and not fix:
            backup_path = f"{file_path}.backup"
            shutil.copy2(file_path, backup_path)
            print(f"  Created backup at {backup_path}")
        
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply the changes
        updated_content = content
        
        if fix:
            # Fix incorrect model name replacements within strings
            updated_content = re.sub(
                r'torch\.mps, \'is_available"meta-llama/Llama-3\.1-8B-Instruct"',
                "torch.backends, 'mps') and torch.backends.mps.is_available(",
                updated_content
            )
            
            # Fix other string replacements
            updated_content = re.sub(
                r'"meta-llama/Llama-3\.1-8B-Instruct"s wifi\?"',
                '"What\'s wifi?"',
                updated_content
            )
        else:
            # 1. Update the model name only in specific contexts
            # Look for model loading patterns
            model_patterns = [
                r'(model_name\s*=\s*)["\']([^"\']*llama[^"\']*)["\']',  # model_name = "..."
                r'(from_pretrained\s*\(\s*)["\']([^"\']*llama[^"\']*)["\']',  # from_pretrained("...")
                r'(--model-name[^"\']*default=)["\']([^"\']*llama[^"\']*)["\']',  # --model-name default="..."
                r'(print\s*\(\s*f\s*["\'][^"\']*model[^"\']*:\s*)[{][^{}]*[}]'  # print(f"...model...: {...")
            ]
            
            for pattern in model_patterns:
                def model_replace(match):
                    prefix = match.group(1)
                    if len(match.groups()) > 1:
                        # Not capturing the model name in the last pattern
                        return f'{prefix}"meta-llama/Llama-3.1-8B-Instruct"'
                    return f'{prefix}"meta-llama/Llama-3.1-8B-Instruct"'
                
                updated_content = re.sub(pattern, model_replace, updated_content, flags=re.IGNORECASE)
            
            # 2. Set the layer to 15
            # Method A: Update direct layer number assignments
            layer_pattern = r'(layers?\s*=\s*\[?)(\d+)(\]?)'
            
            def layer_replace(match):
                prefix, number, suffix = match.groups()
                if number != "15":
                    return f"{prefix}15{suffix}"
                return match.group(0)
            
            updated_content = re.sub(layer_pattern, layer_replace, updated_content)
            
            # Method B: Update layer_number variable assignments
            layer_var_pattern = r'(layer_number\s*=\s*)(\d+)'
            
            def layer_var_replace(match):
                prefix, number = match.groups()
                if number != "15":
                    return f"{prefix}15"
                return match.group(0)
            
            updated_content = re.sub(layer_var_pattern, layer_var_replace, updated_content)
            
            # 3. Add MPS support if not present
            if "mps" not in content.lower():
                # Add import for torch if not present
                if "import torch" not in content:
                    updated_content = "import torch\n" + updated_content
                
                # Add MPS device detection
                mps_detection = (
                    "# Check for MPS availability\n"
                    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
                    "if device == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n"
                    "    device = 'mps'\n"
                    "    print(f\"Using MPS device: {device}\")\n"
                    "\n"
                    "# Apply MPS patches if available\n"
                    "if device == 'mps':\n"
                    "    try:\n"
                    "        from patches.mps_compatibility import apply_mps_patches\n"
                    "        apply_mps_patches()\n"
                    "        print(\"Applied MPS compatibility patches\")\n"
                    "    except ImportError:\n"
                    "        print(\"Warning: MPS patches not found\")\n"
                )
                
                # Find appropriate insertion point - after imports but before main code
                if "def main" in updated_content:
                    # Insert before main function
                    updated_content = re.sub(
                        r'(\s*def\s+main\s*\([^)]*\)\s*:)',
                        f"\n{mps_detection}\n\\1",
                        updated_content
                    )
                else:
                    # Insert after imports
                    updated_content = re.sub(
                        r'((?:^import|^from).+\n)+',
                        f'\\g<0>\n{mps_detection}',
                        updated_content,
                        count=1
                    )
            
            # 4. Add half-precision support
            if "--half-precision" not in content:
                # Add argparse for half-precision
                if "argparse" in content:
                    # Add to existing parser
                    updated_content = re.sub(
                        r'(ArgumentParser\([^)]*\))',
                        '\\1\n    parser.add_argument("--half-precision", action="store_true", help="Use half precision (float16)")',
                        updated_content
                    )
                
                # Add model loading with half-precision
                if "AutoModelForCausalLM.from_pretrained" in content:
                    model_loading_pattern = r'(model\s*=\s*AutoModelForCausalLM\.from_pretrained\([^)]*\))'
                    
                    half_precision_code = (
                        'if args.half_precision:\n'
                        '    print("Using half precision (float16)")\n'
                        '    # For MPS, load on CPU first then move\n'
                        '    model = AutoModelForCausalLM.from_pretrained(\n'
                        '        model_name,\n'
                        '        torch_dtype=torch.float16,\n'
                        '        device_map="cpu"\n'
                        '    )\n'
                        '    model = model.to(device)\n'
                        'else:\n'
                        '    model = AutoModelForCausalLM.from_pretrained(model_name)\n'
                        '    model = model.to(device)'
                    )
                    
                    updated_content = re.sub(model_loading_pattern, half_precision_code, updated_content)
        
        # Check if any changes were made
        if content == updated_content:
            print("  No changes needed")
            continue
        
        # Print the changes in dry-run mode
        if dry_run:
            print("  Changes to be applied:")
            # Show diff (simplified)
            for i, (old_line, new_line) in enumerate(zip(content.splitlines(), updated_content.splitlines())):
                if old_line != new_line:
                    print(f"  Line {i+1}:")
                    print(f"    - {old_line}")
                    print(f"    + {new_line}")
        else:
            # Write the updated content
            with open(file_path, 'w') as f:
                f.write(updated_content)
            print("  Updated successfully")
    
    print("\nCompleted updating examples")

if __name__ == "__main__":
    args = parse_args()
    update_examples(dry_run=args.dry_run, backup=args.backup, fix=args.fix) 