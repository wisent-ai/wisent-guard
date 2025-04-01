#!/usr/bin/env python
"""
Script to display a hallucinated item from guard_results.csv with claude_score=0
showing token scores in a nice format
"""

import argparse
import csv

def parse_token_scores(token_scores_str):
    """Parse the token_scores string into a structured format"""
    if not token_scores_str:
        return []
    
    tokens = []
    # Split the string by the pipe character to get individual token entries
    token_entries = token_scores_str.split('|')
    
    for entry in token_entries:
        # Split each entry by colon to get the components
        parts = entry.split(':')
        if len(parts) >= 6:
            tokens.append({
                'position': parts[0],
                'token_id': parts[1],
                'token_text': parts[2],
                'score': float(parts[3]) if parts[3] != 'None' else 0.0,
                'category': parts[4],
                'is_harmful': parts[5] == 'True'
            })
    
    return tokens

def display_token_scores(csv_path, item_index=0):
    """
    Display token scores for a single item with claude_score=0
    """
    zero_score_items = []
    
    print(f"Reading from: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                claude_score = float(row.get('claude_score', '-1'))
                if claude_score == 0:
                    zero_score_items.append(row)
            except (ValueError, TypeError):
                continue
    
    print(f"\nFound {len(zero_score_items)} items with Claude score 0")
    
    if not zero_score_items:
        print("No items with Claude score 0 found")
        return
    
    # Select the item based on the index
    if item_index >= len(zero_score_items):
        print(f"Warning: Item index {item_index} is out of range. Using the first item.")
        item_index = 0
        
    item = zero_score_items[item_index]
    
    # Display the item
    print(f"\n==== Item {item_index+1}/{len(zero_score_items)} ====")
    
    # Display query/question if available
    if 'question' in item:
        print(f"\nQuestion: {item['question']}")
    
    # Display category if available
    if 'category' in item:
        print(f"Category: {item['category']}")
    
    # Display a preview of the response
    if 'response' in item and item['response']:
        print(f"\nResponse: {item['response'][:100]}...")
    
    # Display scores
    print("\nScores:")
    for key, value in item.items():
        if key in ['claude_score', 'wisent_score', 'is_harmful']:
            print(f"  {key}: {value}")
    
    # Parse and display token scores
    if 'token_scores' in item and item['token_scores']:
        tokens = parse_token_scores(item['token_scores'])
        
        if tokens:
            print("\n==== Token-by-Token Analysis ====")
            print("{:<5} {:<20} {:<10} {:<15} {:<10}".format(
                "Pos", "Token", "Score", "Category", "Harmful"
            ))
            print("-" * 65)
            
            for token in tokens:
                # Format token for display
                token_text = token['token_text'].replace('\n', '\\n')
                if len(token_text) > 20:
                    token_text = token_text[:17] + "..."
                
                # Format score with color or highlighting for high scores
                score = token['score']
                score_str = f"{score:.6f}"
                
                print("{:<5} {:<20} {:<10} {:<15} {:<10}".format(
                    token['position'],
                    token_text,
                    score_str,
                    token['category'] if token['category'] != 'None' else '-',
                    "Yes" if token['is_harmful'] else "No"
                ))
        else:
            print("\nNo token scores to display")
    else:
        print("\nNo token scores available for this item")

def main():
    parser = argparse.ArgumentParser(description="Display token scores for a hallucinated item")
    
    # Input/output options
    parser.add_argument("--csv-file", type=str, default="evaluation/results/guard_results.csv",
                        help="Path to the CSV file with guard results")
    parser.add_argument("--item-index", type=int, default=0,
                        help="Index of the item to display (0-based)")
    
    args = parser.parse_args()
    display_token_scores(args.csv_file, args.item_index)

if __name__ == "__main__":
    main()
