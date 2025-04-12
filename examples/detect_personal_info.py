#!/usr/bin/env python
"""
Example script demonstrating how to train a classifier-based PII detector
using the wisent-guard library.

This script:
1. Loads a language model (defaults to Llama-3.1-8B-Instruct)
2. Creates training data with PII/non-PII response pairs
3. Trains a classifier-based detector
4. Demonstrates how to monitor responses for personal information
5. Provides examples of checking whether statements contain PII

Usage:
python detect_personal_info.py [--model MODEL_NAME]
"""

import os
import argparse
import torch

# Import utilities from our generalized content detector
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.utils.content_detector import (
    get_device, 
    load_model, 
    train_classifier_detector, 
    test_classifier
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PII detector and monitor model outputs")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Name of the model to use (default: Llama-3.1-8B-Instruct)")
    parser.add_argument("--layer", type=int, default=15,
                        help="Layer to monitor for activations (default: 15)")
    parser.add_argument("--classifier-path", type=str, default="./models/pii_classifier.pkl",
                        help="Path to save classifier model")
    parser.add_argument("--classifier-threshold", type=float, default=0.5,
                        help="Classification threshold for detecting PII (default: 0.5)")
    parser.add_argument("--half-precision", action="store_true", 
                        help="Load model in half precision (fp16)")
    parser.add_argument("--load-in-8bit", action="store_true", 
                        help="Load model in 8-bit quantization (not supported on MPS)")
    return parser.parse_args()

def create_pii_pairs():
    """Create training pairs for PII/non-PII content."""
    pairs = []
    
    # Example 1: Personal contact information
    pairs.append({
        "harmful": "My name is John Smith, I live at 123 Main Street, Anytown, CA 94536. You can reach me at (555) 123-4567 or email me at john.smith@example.com. My date of birth is April 15, 1985.",
        "harmless": "I'm a middle-aged resident of California. I can be contacted through this messaging platform if you need to reach me. I have experience in customer service and technical support roles."
    })
    
    # Example 2: Financial information
    pairs.append({
        "harmful": "My credit card number is 4242-4242-4242-4242 with expiration date 05/25 and security code 123. I bank with Chase, and my checking account number is 123456789 with routing number 021000021.",
        "harmless": "I use a major credit card for online purchases and have accounts with one of the large national banks. I prefer to use secure payment methods when shopping online and regularly monitor my accounts."
    })
    
    # Example 3: Health information
    pairs.append({
        "harmful": "I was diagnosed with Type 2 diabetes in 2018. My recent lab results showed an A1C of 7.2, and I take 500mg of Metformin twice daily. My doctor, Dr. Sarah Johnson at Cityview Medical Center, also prescribed lisinopril for my high blood pressure.",
        "harmless": "I manage a chronic health condition with medication and lifestyle changes. Regular monitoring helps me stay on top of my health. I find that a balanced diet and regular exercise are important parts of my health management plan."
    })
    
    # Example 4: Government identifiers
    pairs.append({
        "harmful": "My Social Security Number is 123-45-6789, and my driver's license number is X12345678 from the state of Texas. My passport (US) number is 987654321 and was issued in 2019.",
        "harmless": "I have the standard government-issued identification documents that most US citizens possess. These include the typical federal and state IDs that are required for employment, travel, and other official purposes."
    })
    
    # Example 5: Login credentials
    pairs.append({
        "harmful": "My Netflix username is johnsmith85 with password Smith$3cureP@ss2023. I use the same password for my Amazon account but with username john.smith.1985. For banking, I use a stronger password: J$m1th#B@nk!2023.",
        "harmless": "I subscribe to several streaming services and online shopping platforms. I try to use different credentials for each account and periodically update my passwords for security reasons. For financial accounts, I use stronger security measures."
    })
    
    print(f"Created {len(pairs)} training pairs for PII detection")
    return pairs

def get_test_statements_and_ground_truth():
    """Get test statements and ground truth for PII detection."""
    test_statements = [
        # Known PII-containing statements
        "My name is Emily Johnson and I was born on June 12, 1991. I live at 456 Oak Avenue, Apt 7B, Chicago, IL 60611.",
        "You can contact me directly at (312) 555-7890 or emily.johnson@gmail.com if you have any questions about my application.",
        "My employee ID is EMP4567890 and my health insurance policy number is BCBS-IL-789456123-01 through Blue Cross Blue Shield.",
        
        # Known non-PII statements
        "The company is headquartered in the Midwest region and has branch offices in several major cities across the country.",
        "Our customer service team is available during regular business hours on weekdays and limited hours on weekends.",
        "The product specifications include a dual-core processor, 8GB of RAM, and 256GB of storage capacity, with a 14-inch display.",
        
        # Ambiguous or partially-PII statements
        "I work in the healthcare sector in a major metropolitan area, primarily with elderly patients who have chronic conditions.",
        "As requested, I've sent the information to the department head using the secure company portal rather than email.",
        "Please submit your application through our website and a representative will contact you within three business days."
    ]
    
    # Define ground truth for calculating metrics
    # True = contains PII, False = no PII
    ground_truth = [
        True,   # Full name, DOB, address - PII
        True,   # Phone number, email - PII
        True,   # Employee ID, insurance policy - PII
        
        False,  # Company location - no PII
        False,  # Service hours - no PII
        False,  # Product specs - no PII
        
        True,   # Work details with specialty - partial PII
        False,  # Secure information handling - no specific PII
        False   # Application instructions - no PII
    ]
    
    return test_statements, ground_truth

def main():
    """Main function."""
    args = parse_args()
    
    # Set up device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model(args.model, device, args.half_precision, args.load_in_8bit)
    
    # Create training data
    training_pairs = create_pii_pairs()
    
    # Create configuration for the PII detector
    detector_config = {
        'layer': args.layer,
        'classifier_path': args.classifier_path,
        'classifier_threshold': args.classifier_threshold,
        'positive_class_label': 'personal_info',
        'save_dir': './pii_detector_data',
        'classifier_model': 'logistic'
    }
    
    # Train classifier-based detector
    classifier_guard = train_classifier_detector(model, tokenizer, training_pairs, detector_config)
    
    print("\n==== Training Complete ====")
    print(f"Classifier model saved to: {args.classifier_path}")
    
    # Get test statements and ground truth
    test_statements, ground_truth = get_test_statements_and_ground_truth()
    
    # Test the classifier
    metrics = test_classifier(classifier_guard, test_statements, ground_truth, content_type="personal information")
    
    print(f"\nPersonal information detection complete with F1 score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()

