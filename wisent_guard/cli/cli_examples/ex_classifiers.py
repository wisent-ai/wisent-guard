# Example usage of ClassifierRotator

from wisent_guard.cli.classifiers.classifier_rotator import ClassifierRotator
import time

# Initialize the rotator, auto-discovering classifiers
rotator = ClassifierRotator(autoload=True)

print("Available classifiers:")

for clf_info in rotator.list_classifiers():
    print(f"- {clf_info['name']}: {clf_info['description']}")

# Use a specific classifier by name
rotator.use("mlp", hidden_dim=64)

# Create some dummy data
import torch
X = torch.randn(100, 20)  # 100 samples, 20 features
y = torch.randint(0, 2, (100,)).float().unsqueeze(1)  # Binary targets

# Fit the classifier
report = rotator.fit(X, y, epochs=5, batch_size=16)

print("Training report:", report)