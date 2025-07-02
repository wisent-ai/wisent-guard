import sys
import time
import signal
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

class TimeoutError(Exception):
    """Raised when test exceeds time budget."""
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded time budget!")

def main():
    # Set budget and timeout separately
    budget_minutes = 1.0  # 1 minute - internal budget for classifier creation
    timeout_seconds = 120  # 2 minutes - hard timeout for the test process
    set_time_budget(budget_minutes)
    
    print(f"⏱️ Starting synthetic classifier test with {timeout_seconds}s timeout and {budget_minutes*60}s budget...")
    
    # Set up timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    start_time = time.time()
    
    try:
        model = Model(name="meta-llama/Llama-3.1-8B-Instruct")
        prompt = "What is the capital of France?"
        
        # Test the system
        classifiers, trait_discovery = create_classifiers_for_prompt(model, prompt)
        
        # Clear the alarm since we completed successfully
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        print(f"✅ SUCCESS: Created {len(classifiers)} classifiers for {len(trait_discovery.traits_discovered)} traits")
        print(f"⏱️ Total time: {elapsed_time:.1f}s (timeout: {timeout_seconds}s, budget: {budget_minutes*60}s)")
        
        if elapsed_time > timeout_seconds:
            print(f"⚠️ WARNING: Test completed but exceeded timeout by {elapsed_time - timeout_seconds:.1f}s")
        else:
            print(f"🎉 Test completed within timeout with {timeout_seconds - elapsed_time:.1f}s to spare!")
            
    except TimeoutError as e:
        elapsed_time = time.time() - start_time
        print(f"❌ ERROR: {e}")
        print(f"❌ Test failed after {elapsed_time:.1f}s (timeout: {timeout_seconds}s, budget: {budget_minutes*60}s)")
        print("❌ This indicates a performance issue that needs investigation.")
        sys.exit(1)
    except Exception as e:
        signal.alarm(0)  # Clear timeout
        elapsed_time = time.time() - start_time
        print(f"❌ ERROR: Test failed with exception: {e}")
        print(f"❌ Time elapsed: {elapsed_time:.1f}s")
        sys.exit(1)

if __name__ == "__main__":
    main()
