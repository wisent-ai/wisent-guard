# cli_train.py
import os
from typing import Dict, List, Optional

from wisent_guard.core import SteeringMethod, SteeringType

def _auto_discover(load_classifier: str, model_name: str, task_name: str, layers: List[int]) -> str:
    if load_classifier:
        return load_classifier
    safe_model = model_name.replace("/", "_").replace(":", "_")
    base = f"./optimized_classifiers/{safe_model}/{task_name}_classifier"
    # verify presence
    if len(layers) == 1:
        path = f"{base}_layer_{layers[0]}.pkl"
        return base if os.path.exists(path) else ""
    # multi
    ok = True
    for l in layers:
        if not os.path.exists(f"{base}_layer_{l}.pkl"):
            ok = False; break
    return base if ok else ""

def train_or_load_classifiers(
    is_multi_layer: bool,
    processed_pairs,
    layers: List[int],
    classifier_type: str,
    load_classifier: str,
    classifier_dir: str,
    auto_discover_model_name: str,
    task_name: str,
    detection_threshold: float,
    train_only: bool,
    verbose: bool,
):
    steering_methods: Dict[int, SteeringMethod] = {}
    layer_training_results: Dict[int, Dict] = {}

    # Try auto-discover unless explicitly set
    auto_path = _auto_discover(load_classifier, auto_discover_model_name, task_name, layers)
    if not load_classifier and auto_path:
        load_classifier = auto_path
        if verbose:
            print("\n📦 Found pre-trained classifiers, loading automatically...")
            print(f"   • Loading from: {load_classifier}")

    if load_classifier:
        from wisent_guard.core.model_persistence import ModelPersistence
        if is_multi_layer:
            data = ModelPersistence.load_multi_layer_classifiers(load_classifier, layers)
            for layer_idx, (clf, meta) in data.items():
                stype = SteeringType.LOGISTIC if meta.get("classifier_type", "logistic") == "logistic" else SteeringType.MLP
                steering_methods[layer_idx] = SteeringMethod(method_type=stype, threshold=meta.get("detection_threshold", detection_threshold), classifier=clf)
                layer_training_results[layer_idx] = {"loaded": True, "metadata": meta}
        else:
            clf, meta = ModelPersistence.load_classifier(load_classifier, layers[0])
            stype = SteeringType.LOGISTIC if meta.get("classifier_type", "logistic") == "logistic" else SteeringType.MLP
            steering_methods[layers[0]] = SteeringMethod(method_type=stype, threshold=meta.get("detection_threshold", detection_threshold), classifier=clf)
            layer_training_results[layers[0]] = {"loaded": True, "metadata": meta}
        return steering_methods, layer_training_results, True

    # Train
    from wisent_guard.core.classifiers import train_classifier_for_layer
    for li in layers:
        res = train_classifier_for_layer(processed_pairs, layer_index=li, classifier_type=classifier_type, verbose=verbose)
        steering_methods[li] = res["steering_method"]
        layer_training_results[li] = res["metrics"]
    return steering_methods, layer_training_results, False

def save_trained_artifacts(
    *,
    steering_methods: Dict[int, SteeringMethod],
    layers: List[int],
    is_multi_layer: bool,
    save_classifier: Optional[str],
    train_only: bool,
    task_name: str,
    model_name: str,
    classifier_dir: str,
    classifier_type: str,
    training_results: Dict[int, Dict],      # per-layer training metrics (e.g., {"accuracy": ...})
    contrastive_pairs_count: int,
    token_aggregation: str,
    detection_threshold: float,
    verbose: bool,
) -> Dict[int, str]:
    """
    Saves trained classifier(s) with per-layer metadata.
    - If `save_classifier` is provided, use it as the base path.
    - Else if `train_only` is True, save under a sensible default in `classifier_dir`.
    - Else (no explicit request), do nothing.
    Returns: {layer_idx: saved_path}
    """
    saved_paths: Dict[int, str] = {}

    # Decide whether we should save at all
    if not save_classifier and not train_only:
        return saved_paths

    # Determine base save path
    if save_classifier:
        base_path = save_classifier
    else:
        # Default path for train-only mode
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        base_path = os.path.join(classifier_dir, f"{task_name}_{safe_model_name}_classifier")

    if verbose:
        print("\n💾 SAVING TRAINED CLASSIFIERS:")
        print(f"   • Save path: {base_path}")

    try:
        from wisent_guard.core.model_persistence import ModelPersistence, create_classifier_metadata
        # Save multiple classifiers or single
        if is_multi_layer:
            for layer_idx in layers:
                if layer_idx not in steering_methods:
                    continue
                classifier = steering_methods[layer_idx].classifier
                layer_metrics = training_results.get(layer_idx, {})
                metadata = create_classifier_metadata(
                    model_name=model_name,
                    task_name=task_name,
                    layer=layer_idx,
                    classifier_type=classifier_type,
                    training_accuracy=layer_metrics.get("accuracy", 0.0),
                    training_samples=contrastive_pairs_count,
                    token_aggregation=token_aggregation,
                    detection_threshold=detection_threshold,
                )
                # Keep the old ordering: (classifier, layer_idx, base_path, metadata)
                path = ModelPersistence.save_classifier(classifier, layer_idx, base_path, metadata)
                saved_paths[layer_idx] = path
                if verbose:
                    print(f"     ✅ Layer {layer_idx}: {path}")
        else:
            # Single layer
            only_layer = layers[0]
            classifier = steering_methods[only_layer].classifier
            layer_metrics = training_results.get(only_layer, {})
            metadata = create_classifier_metadata(
                model_name=model_name,
                task_name=task_name,
                layer=only_layer,
                classifier_type=classifier_type,
                training_accuracy=layer_metrics.get("accuracy", 0.0),
                training_samples=contrastive_pairs_count,
                token_aggregation=token_aggregation,
                detection_threshold=detection_threshold,
            )
            path = ModelPersistence.save_classifier(classifier, only_layer, base_path, metadata)
            saved_paths[only_layer] = path
            if verbose:
                print(f"     ✅ Saved: {path}")

    except Exception as e:
        if verbose:
            print(f"     ❌ Error saving classifiers: {e}")

    return saved_paths
