import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent_guard.benchmark_loader import BenchmarkLoader
from wisent_guard.monitor import ActivationMonitor
from wisent_guard.vectors import ContrastiveVectors
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    model_name = "meta-llama/Llama-3.1-8B"
    benchmark_name = "hellaswag"
    layer = 15
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize vectors and monitor
    logger.info("Initializing activation monitor...")
    vectors = ContrastiveVectors()
    layers = [layer]
    monitor = ActivationMonitor(
        model=model,
        layers=layers,
        vectors=vectors,
        token_strategy="last"
    )
    
    # Load benchmark
    logger.info("Loading benchmark...")
    loader = BenchmarkLoader(benchmark_name)
    train_data, test_data = loader.get_train_test_split()
    
    # Show first example
    logger.info("Showing example prompt and response...")
    example = train_data[0]
    prompt = loader.task.doc_to_text(example)
    logger.info(f"\nExample prompt: {prompt}")
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"\nGenerated response: {response}")
    
    # Get activations
    logger.info("\nGetting activations...")
    activations = monitor.get_activations(prompt)
    logger.info(f"\nActivations shape: {activations.shape}")

if __name__ == "__main__":
    main()
