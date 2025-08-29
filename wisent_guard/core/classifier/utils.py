from typing import List


def calculate_roc_auc(y_true: List[float], y_scores: List[float]) -> float:
    """
    Calculate the ROC AUC score without using scikit-learn.

    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted scores

    Returns:
        ROC AUC score
    """
    if len(y_true) != len(y_scores):
        raise ValueError("Length of y_true and y_scores must match")

    if len(set(y_true)) != 2:
        # Not a binary classification problem or only one class in the data
        return 0.5

    # Pair the scores with their true labels and sort by score in descending order
    pair_list = sorted(zip(y_scores, y_true), reverse=True)

    # Count the number of positive and negative samples
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        # Only one class present, ROC AUC is not defined
        return 0.5

    # Count the number of correctly ranked pairs
    auc = 0.0
    pos_so_far = 0

    # Iterate through pairs
    for i, (_, label) in enumerate(pair_list):
        if label == 1:
            # This is a positive example
            pos_so_far += 1
        else:
            # This is a negative example, add the number of positive examples seen so far
            auc += pos_so_far

    # Normalize by the number of positive-negative pairs
    auc /= n_pos * n_neg

    return auc
