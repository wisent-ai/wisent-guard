from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator
from wisent_guard.cli.steering_methods.steering_rotator import SteeringMethodRotator
from wisent_guard.core.models.wisent_model import WisentModel
from wisent_guard.core.trainers.steering_trainer import WisentSteeringTrainer
from wisent_guard.cli.evaluators.evaluator_rotator import EvaluatorRotator
import time
import json
import torch
from pathlib import Path
import os
from itertools import product
import gc

def run_training(model, model_name, data_limit, layer, scale, normalize):

    rot_data = DataLoaderRotator()
    rot_data.use("lm_eval")
    data = rot_data.load(task="prost", limit=data_limit)

    rot_steer = SteeringMethodRotator()
    method_name = "caa"
    rot_steer.use(method_name)
    caa_method = rot_steer._method 

    training_data = data['train_qa_pairs']
    trainer = WisentSteeringTrainer(model=model, pair_set=training_data, steering_method=caa_method)

    TEST_PROMPTS = [
        "Q: A person is trying to roll an orange, a tennis racket, a chair, and a wheel. Question: Which is the hardest to roll? Answer: A. chair B. orange",
        "Q: A person is trying to roll a pencil, a plate, a sofa, and a lemon. Question: Which is the hardest to roll? Answer: A. sofa B. pencil",
        "Q: A person is trying to roll a mug, a skateboard, a coin, and a refrigerator. Question: Which is the hardest to roll? Answer: A. refrigerator B. coin",
        "Q: A person is trying to roll a tire, a pillow, a broom, and a laptop. Question: Which is the hardest to roll? Answer: A. laptop B. tire",
        "Q: A person is trying to roll a suitcase, a globe, a spoon, and a chair. Question: Which is the hardest to roll? Answer: A. chair B. globe",
        "Q: A person is trying to roll an umbrella, a cup, a stool, and a pan. Question: Which is the hardest to roll? Answer: A. stool B. cup",
        "Q: A person is trying to roll a remote control, a paint roller, a banana, and a table. Question: Which is the hardest to roll? Answer: A. table B. banana",
        "Q: A person is trying to roll a watermelon, a pen, a tablet, and a lamp. Question: Which is the hardest to roll? Answer: A. lamp B. pen",
        "Q: A person is trying to roll a printer, a marker, an eggplant, and a wheelbarrow. Question: Which is the hardest to roll? Answer: A. printer B. marker",
        "Q: A person is trying to roll a guitar, a jar, a skateboard, and a pillow. Question: Which is the hardest to roll? Answer: A. guitar B. jar"
    ]

    CORRECT_ANSWERS = [
        "chair",
        "sofa",
        "refrigerator",
        "laptop",
        "chair",
        "stool",
        "table",
        "lamp",
        "printer",
        "guitar"
    ]

    normalize_str = "norm" if normalize else "no_norm"
    save_dir = f"./tests/prost/steering_output_{model_name}/best_steering_output_layer{layer}_scale{abs(scale)}_{normalize_str}"

    print(f"\n{'='*80}")
    print(f"Training: Layer={layer}, Scale={scale}, Normalize={normalize}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*80}\n")

    training_result = trainer.run(
        layers_spec=layer,  
        aggregation="continuation_token", 
        return_full_sequence=False,  
        normalize_layers=normalize,  
        save_dir=save_dir  
    )

    print(f"Training completed for Layer={layer}, Scale={scale} Normalize={normalize}")

    print("\n" + "="*80)
    print(f"INFERENCE: Layer={layer}, Scale={scale}, Normalize={normalize}")
    print("="*80 + "\n")

    steering_vectors = training_result.steered_vectors.to_dict()
    print(f"Loaded steering vectors for layers: {list(steering_vectors.keys())}")

    model.set_steering_from_raw(steering_vectors, scale=scale, normalize=False)

    results = []
    for prompt, correct_answer in zip(TEST_PROMPTS, CORRECT_ANSWERS):
        print(f"Testing: {prompt[:50]}...")

        with model.detached():
            messages_unsteered = [[{"role": "user", "content": prompt}]]
            unsteered = model.generate(messages_unsteered, max_new_tokens=250, use_steering=False)[0]

        messages_steered = [[{"role": "user", "content": prompt}]]
        steered = model.generate(messages_steered, max_new_tokens=250, use_steering=True)[0]

        model.set_steering_from_raw(steering_vectors, scale=-scale, normalize=False)
        messages_steered_pos = [[{"role": "user", "content": prompt}]]
        steered_pos = model.generate(messages_steered_pos, max_new_tokens=250, use_steering=True)[0]
        model.set_steering_from_raw(steering_vectors, scale=scale, normalize=False)
        
        result = {
            "layer": layer,
            "scale": scale,
            "normalize": normalize,
            "prompt": prompt,
            "unsteered_response": unsteered,
            "steered_response": steered,
            "steered_response_positive_scale": steered_pos,
            "correct_answer": correct_answer
        }
        results.append(result)
    print(f"Completed {len(TEST_PROMPTS)} prompts for Layer={layer}, Scale={scale}, Normalize={normalize}")
    return results

configs = {
    # 16 layers:
    "meta-llama/Llama-3.2-1B-Instruct": {
        "model_name": "Llama-3.2-1B-Instruct",
        "data_limit": 30,
        "best_l_s_n": [(6, -1, True)] 
    },
    # 36 layers
    "unsloth/Qwen3-4B-bnb-4bit": {
        "model_name": "Qwen3-4B-bnb-4bit",
        "data_limit": 200,
        "best_l_s_n": [(14, -3, True)] 
    },
    # 36 layers:
    "unsloth/Qwen2.5-3B-Instruct": {
        "model_name": "Qwen2.5-3B-Instruct",
        "data_limit": 200,
        "best_l_s_n": [(14, -3, True)] 
    }
}

for model_path, cfg in configs.items():
    model = WisentModel(model_name=model_path, layers={}, device="cuda")
    all_results = []
    for l, s, n in cfg["best_l_s_n"]:
        results = run_training(model=model, model_name=cfg["model_name"], data_limit=cfg["data_limit"], scale=s, layer=l, normalize=n)
        all_results.extend(results)

    output_file = f"./tests/prost/best_results_{cfg['model_name']}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(10) 

print(f"\n{'='*80}")
print(f"✓ Grid search completed!")

