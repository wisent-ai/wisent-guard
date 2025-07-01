import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from wisent_guard.core.model import Model
from wisent_guard.core.agent.diagnose.synthetic_classifier_option import (
    create_classifiers_for_prompt,
    apply_classifiers_to_response
)
from wisent_guard.core.agent.budget import set_time_budget

def main():
    # Set budget first
    set_time_budget(0.5)  # 30 seconds
    
    model = Model(name="meta-llama/Llama-3.1-8B-Instruct")
    prompt = "What is the capital of France?"
    
    # Test the system
    classifiers, trait_discovery = create_classifiers_for_prompt(model, prompt)
    print(f"Created {len(classifiers)} classifiers for {len(trait_discovery.traits_discovered)} traits")

if __name__ == "__main__":
    main()
