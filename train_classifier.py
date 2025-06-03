# Integrated with Wisent-Guard v1

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_classifier_from_data(data_file="final_training_data.json", save_path="classifier_pipeline.joblib"):
    """
    Train a logistic regression classifier on labeled text data.

    Args:
        data_file (str): Path to JSON file with "text" and "label" fields
        save_path (str): Path to save the trained pipeline using joblib

    Returns:
        pipeline (Pipeline): Trained scikit-learn pipeline
    """
    with open(data_file) as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, save_path)
    print(f"✅ Classifier pipeline saved to {save_path}")

    return pipeline

# Runable directly from CLI
if __name__ == "__main__":
    train_classifier_from_data()