import os
import pandas as pd
from typing import Optional
from tqdm import tqdm

from universal_representations.evaluation.evaluate_string import call_claude_and_evaluate

def evaluate_truthfulqa_csv(
    input_csv_path: str,
    output_csv_path: Optional[str] = None,
    actual_responses_column: str = "Actual Response",
    good_responses_column: str = "Best Answer",
    bad_responses_column: str = "Incorrect Answers"
) -> pd.DataFrame:
    """Evaluate a TruthfulQA dataset CSV and add a score column.
    
    Args:
        input_csv_path: Path to input CSV file with TruthfulQA questions and responses
        output_csv_path: Path to save the output CSV with scores
        actual_responses_column: Column name containing actual responses to evaluate
        good_responses_column: Column name containing good/correct responses
        bad_responses_column: Column name containing bad/incorrect responses
        
    Returns:
        DataFrame with added 'score' column
    """
    # Load the CSV
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Loaded CSV with {len(df)} entries.")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")
    
    # Check for required columns
    required_columns = [actual_responses_column, good_responses_column, bad_responses_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV file missing required column(s): {missing_columns}. Found: {df.columns}")
    
    # Evaluate each row
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating responses"):
        actual_response = row[actual_responses_column]
        good_response = row[good_responses_column]
        bad_response = row[bad_responses_column]
        
        # Handle cases where any of the responses are missing
        if pd.isna(actual_response) or pd.isna(good_response) or pd.isna(bad_response):
            scores.append(-1)
            continue
        
        # For cases where bad responses are provided as a list, take the first one
        if isinstance(bad_response, str) and '[' in bad_response:
            try:
                import ast
                bad_responses_list = ast.literal_eval(bad_response)
                if isinstance(bad_responses_list, list) and len(bad_responses_list) > 0:
                    bad_response = bad_responses_list[0]
            except:
                pass  # If parsing fails, use the string as is
        
        # Call Claude to evaluate
        score = call_claude_and_evaluate(
            actual_response=str(actual_response),
            good_response=str(good_response),
            bad_response=str(bad_response)
        )
        scores.append(score)
    
    # Add scores to dataframe
    df["score"] = scores
    
    # Save if output path is provided
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"Evaluation results saved to {output_csv_path}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate responses in a CSV by comparing to good and bad examples")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save output CSV file with scores")
    parser.add_argument("--actual-col", type=str, default="Actual Response",
                        help="Column name containing actual responses to evaluate")
    parser.add_argument("--good-col", type=str, default="Best Answer",
                        help="Column name containing good/correct responses")
    parser.add_argument("--bad-col", type=str, default="Incorrect Answers",
                        help="Column name containing bad/incorrect responses")
    
    args = parser.parse_args()
    
    evaluate_truthfulqa_csv(
        input_csv_path=args.input,
        output_csv_path=args.output,
        actual_responses_column=args.actual_col,
        good_responses_column=args.good_col,
        bad_responses_column=args.bad_col
    )
