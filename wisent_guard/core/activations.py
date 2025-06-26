import torch
import torch.nn.functional as F
from .layer import Layer
from enum import Enum
from typing import Optional, Dict, Any, Union, List, Tuple
try:
    from .contrastive_pairs import ContrastivePair
except ImportError:
    from contrastive_pairs import ContrastivePair
from .response import PositiveResponse, NegativeResponse

class ActivationAggregationMethod(Enum):
    LAST_TOKEN = "last_token"
    MEAN = "mean"
    MAX = "max"

class Activations:
    def __init__(self, tensor, layer, aggregation_method=None):
        self.tensor = tensor
        self.layer = layer
        self.aggregation_method = aggregation_method or ActivationAggregationMethod.LAST_TOKEN
    
    def get_aggregated(self):
        """
        Get aggregated activations based on the aggregation method.
        
        Returns:
            torch.Tensor: Aggregated activation tensor
        """
        if self.aggregation_method == ActivationAggregationMethod.LAST_TOKEN:
            # Return the last token's activations
            if len(self.tensor.shape) >= 3:  # [batch, seq_len, hidden_dim]
                return self.tensor[0, -1, :]  # Take last token of first batch
            elif len(self.tensor.shape) == 2:  # [seq_len, hidden_dim]
                return self.tensor[-1, :]  # Take last token
            else:
                return self.tensor  # Already aggregated
                
        elif self.aggregation_method == ActivationAggregationMethod.MEAN:
            # Return mean across sequence dimension
            if len(self.tensor.shape) >= 3:  # [batch, seq_len, hidden_dim]
                return self.tensor[0].mean(dim=0)  # Mean across sequence length
            elif len(self.tensor.shape) == 2:  # [seq_len, hidden_dim]
                return self.tensor.mean(dim=0)  # Mean across sequence length
            else:
                return self.tensor
                
        elif self.aggregation_method == ActivationAggregationMethod.MAX:
            # Return max across sequence dimension
            if len(self.tensor.shape) >= 3:  # [batch, seq_len, hidden_dim]
                return self.tensor[0].max(dim=0)[0]  # Max across sequence length
            elif len(self.tensor.shape) == 2:  # [seq_len, hidden_dim]
                return self.tensor.max(dim=0)[0]  # Max across sequence length
            else:
                return self.tensor
        else:
            # Default to last token
            if len(self.tensor.shape) >= 3:  # [batch, seq_len, hidden_dim]
                return self.tensor[0, -1, :]  # Take last token of first batch
            elif len(self.tensor.shape) == 2:  # [seq_len, hidden_dim]
                return self.tensor[-1, :]  # Take last token
            else:
                return self.tensor  # Already aggregated

    def calculate_similarity(self, other_tensor: torch.Tensor, method: str = "cosine") -> float:
        """
        Calculate similarity between this activation and another tensor.
        
        Args:
            other_tensor: Tensor to compare against (e.g., contrastive vector)
            method: Similarity method ("cosine", "dot", "euclidean")
            
        Returns:
            Similarity score
        """
        # Get aggregated activation
        activation = self.get_aggregated()
        
        # Ensure tensors are on the same device and have compatible shapes
        if activation.device != other_tensor.device:
            other_tensor = other_tensor.to(activation.device)
        
        # Flatten tensors if needed
        if len(activation.shape) > 1:
            activation = activation.flatten()
        if len(other_tensor.shape) > 1:
            other_tensor = other_tensor.flatten()
        
        # Handle dimension mismatch
        if activation.shape[0] != other_tensor.shape[0]:
            min_dim = min(activation.shape[0], other_tensor.shape[0])
            activation = activation[:min_dim]
            other_tensor = other_tensor[:min_dim]
        
        try:
            if method == "cosine":
                # Cosine similarity
                cos_sim = F.cosine_similarity(
                    activation.unsqueeze(0), 
                    other_tensor.unsqueeze(0), 
                    dim=1
                )
                # Convert to similarity score (0 to 1)
                similarity = (cos_sim.item() + 1.0) / 2.0
                return max(0.0, min(1.0, similarity))
                
            elif method == "dot":
                # Dot product similarity
                dot_product = torch.dot(activation, other_tensor)
                return dot_product.item()
                
            elif method == "euclidean":
                # Negative euclidean distance (higher = more similar)
                distance = torch.norm(activation - other_tensor)
                return -distance.item()
                
            else:
                raise ValueError(f"Unknown similarity method: {method}")
                
        except Exception as e:
            # Return 0 similarity on error
            return 0.0

    def compare_with_vectors(self, vector_dict: Dict[str, torch.Tensor], threshold: float = 0.7) -> Dict[str, Any]:
        """
        Compare this activation with multiple contrastive vectors.
        
        Args:
            vector_dict: Dictionary mapping category names to contrastive vectors
            threshold: Threshold for determining harmful content
            
        Returns:
            Dictionary with comparison results for each category
        """
        results = {}
        
        for category, vector in vector_dict.items():
            similarity = self.calculate_similarity(vector)
            is_harmful = similarity >= threshold
            
            results[category] = {
                "similarity": similarity,
                "is_harmful": is_harmful,
                "threshold": threshold
            }
        
        return results

    def extract_features_for_classifier(self) -> torch.Tensor:
        """
        Extract features suitable for classifier input.
        
        Returns:
            Flattened tensor ready for classification
        """
        features = self.get_aggregated()
        
        # Ensure it's a PyTorch tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Flatten if needed
        if len(features.shape) > 1:
            features = features.flatten()
        
        return features

    def to_device(self, device: str) -> 'Activations':
        """
        Move activations to a specific device.
        
        Args:
            device: Target device (e.g., 'cuda', 'cpu', 'mps')
            
        Returns:
            New Activations object on the target device
        """
        new_tensor = self.tensor.to(device)
        return Activations(
            tensor=new_tensor,
            layer=self.layer,
            aggregation_method=self.aggregation_method
        )

    def normalize(self) -> 'Activations':
        """
        Normalize the activation tensor.
        
        Returns:
            New Activations object with normalized tensor
        """
        aggregated = self.get_aggregated()
        
        # L2 normalization
        norm = torch.norm(aggregated, p=2, dim=-1, keepdim=True)
        normalized = aggregated / (norm + 1e-8)  # Add small epsilon to avoid division by zero
        
        return Activations(
            tensor=normalized,
            layer=self.layer,
            aggregation_method=self.aggregation_method
        )

    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistical information about the activations.
        
        Returns:
            Dictionary with statistics
        """
        aggregated = self.get_aggregated()
        
        return {
            "mean": float(aggregated.mean()),
            "std": float(aggregated.std()),
            "min": float(aggregated.min()),
            "max": float(aggregated.max()),
            "norm": float(torch.norm(aggregated)),
            "shape": list(aggregated.shape),
            "device": str(aggregated.device),
            "dtype": str(aggregated.dtype)
        }

    @classmethod
    def from_model_output(cls, model_outputs, layer: Layer, aggregation_method=None) -> 'Activations':
        """
        Create Activations object from model forward pass outputs.
        
        Args:
            model_outputs: Output from model forward pass
            layer: Layer object specifying which layer to extract
            aggregation_method: How to aggregate the activations
            
        Returns:
            Activations object
        """
        if hasattr(model_outputs, 'hidden_states') and model_outputs.hidden_states is not None:
            hidden_states = model_outputs.hidden_states
            
            # Get the hidden state for the specified layer (add 1 because hidden_states[0] is embeddings)
            if layer.index + 1 < len(hidden_states):
                layer_hidden_state = hidden_states[layer.index + 1]
                
                return cls(
                    tensor=layer_hidden_state,
                    layer=layer,
                    aggregation_method=aggregation_method
                )
            else:
                raise ValueError(f"Layer {layer.index} not found in model with {len(hidden_states)} layers")
        else:
            raise ValueError("Model outputs don't contain hidden_states")

    @classmethod
    def from_tensor_dict(cls, tensor_dict: Dict[str, torch.Tensor], layer: Layer, aggregation_method=None) -> 'Activations':
        """
        Create Activations object from a dictionary of tensors (legacy compatibility).
        
        Args:
            tensor_dict: Dictionary containing activation tensors
            layer: Layer object
            aggregation_method: How to aggregate the activations
            
        Returns:
            Activations object
        """
        # Try to find the activation tensor in the dictionary
        if 'activations' in tensor_dict:
            tensor = tensor_dict['activations']
        elif str(layer.index) in tensor_dict:
            tensor = tensor_dict[str(layer.index)]
        elif layer.index in tensor_dict:
            tensor = tensor_dict[layer.index]
        else:
            raise ValueError(f"No activation tensor found for layer {layer.index} in dictionary")
        
        return cls(
            tensor=tensor,
            layer=layer,
            aggregation_method=aggregation_method
        )

    def __repr__(self) -> str:
        """String representation of the Activations object."""
        aggregated = self.get_aggregated()
        return f"Activations(layer={self.layer.index}, shape={list(aggregated.shape)}, method={self.aggregation_method.value})"

    # Monitoring functionality
    def cosine_similarity(self, other: 'Activations') -> float:
        """Calculate cosine similarity with another activation."""
        return self.calculate_similarity(other.get_aggregated(), method="cosine")
    
    def dot_product_similarity(self, other: 'Activations') -> float:
        """Calculate dot product similarity with another activation."""
        return self.calculate_similarity(other.get_aggregated(), method="dot")
    
    def euclidean_distance(self, other: 'Activations') -> float:
        """Calculate euclidean distance with another activation."""
        return -self.calculate_similarity(other.get_aggregated(), method="euclidean")

class ActivationMonitor:
    """
    Monitor for tracking and analyzing model activations.
    Integrated into the activations primitive.
    """
    
    def __init__(self):
        """Initialize activation monitor."""
        self.current_activations = {}
        self.activation_history = []
    
    def store_activations(
        self,
        activations: Dict[int, Activations],
        text: Optional[str] = None
    ) -> None:
        """Store activations with optional text context."""
        self.current_activations = activations
        self.activation_history.append({
            "text": text,
            "activations": activations
        })
    
    def analyze_activations(
        self,
        activations: Optional[Dict[int, Activations]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Analyze activations for patterns and statistics."""
        if activations is None:
            activations = self.current_activations
        
        analysis = {}
        for layer_idx, activation in activations.items():
            stats = activation.get_statistics()
            features = activation.extract_features_for_classifier()
            
            analysis[layer_idx] = {
                "statistics": stats,
                "feature_vector": features,
                "tensor_shape": activation.tensor.shape,
                "device": str(activation.tensor.device)
            }
        
        return analysis
    
    def compare_with_baseline(
        self,
        baseline_activations: Dict[int, Activations],
        current_activations: Optional[Dict[int, Activations]] = None
    ) -> Dict[int, Dict[str, float]]:
        """Compare current activations with baseline."""
        if current_activations is None:
            current_activations = self.current_activations
        
        comparisons = {}
        for layer_idx in current_activations:
            if layer_idx in baseline_activations:
                current = current_activations[layer_idx]
                baseline = baseline_activations[layer_idx]
                
                comparisons[layer_idx] = {
                    "cosine_similarity": current.cosine_similarity(baseline),
                    "dot_product": current.dot_product_similarity(baseline),
                    "euclidean_distance": current.euclidean_distance(baseline)
                }
        
        return comparisons
    
    def detect_anomalies(
        self,
        threshold: float = 0.8,
        activations: Optional[Dict[int, Activations]] = None
    ) -> Dict[int, bool]:
        """Detect anomalies in activations based on historical patterns."""
        if activations is None:
            activations = self.current_activations
        
        if len(self.activation_history) < 2:
            return {layer_idx: False for layer_idx in activations}
        
        anomalies = {}
        for layer_idx in activations:
            historical_activations = []
            for entry in self.activation_history[:-1]:
                if layer_idx in entry["activations"]:
                    historical_activations.append(entry["activations"][layer_idx])
            
            if not historical_activations:
                anomalies[layer_idx] = False
                continue
            
            current = activations[layer_idx]
            similarities = [current.cosine_similarity(hist) for hist in historical_activations]
            avg_similarity = sum(similarities) / len(similarities)
            anomalies[layer_idx] = avg_similarity < threshold
        
        return anomalies
    
    def clear_history(self) -> None:
        """Clear activation history."""
        self.activation_history.clear()
    
    def save_activations(self, filepath: str) -> None:
        """Save current activations to file."""
        import torch
        
        if not self.current_activations:
            raise ValueError("No activations to save")
        
        save_data = {}
        for layer_idx, activation in self.current_activations.items():
            save_data[layer_idx] = activation.tensor
        
        torch.save(save_data, filepath)
    
    def load_activations(self, filepath: str) -> Dict[int, Activations]:
        """Load activations from file."""
        import torch
        from .layer import Layer
        
        loaded_data = torch.load(filepath)
        activations = {}
        for layer_idx, tensor in loaded_data.items():
            layer_obj = Layer(index=layer_idx, type="transformer")
            activations[layer_idx] = Activations(tensor=tensor, layer=layer_obj)
        
        self.current_activations = activations
        return activations

class TestActivationCache:
    """
    Cache for test-time activations with associated questions and responses.
    Allows saving activations from one layer and reusing them for different layer classifiers.
    """
    
    def __init__(self):
        self.activations = []  # List of dicts with 'question', 'response', 'activations', 'layer'
    
    def add_activation(self, question: str, response: str, activations: 'Activations', layer: int):
        """Add a test activation to the cache."""
        self.activations.append({
            'question': question,
            'response': response,  
            'activations': activations,
            'layer': layer
        })
    
    def save_to_file(self, filepath: str) -> None:
        """Save cached activations to file."""
        import torch
        import json
        
        if not self.activations:
            raise ValueError("No activations to save")
        
        # Prepare data for saving
        save_data = {
            'metadata': {
                'num_samples': len(self.activations),
                'layers': list(set(item['layer'] for item in self.activations)),
                'created_at': __import__('datetime').datetime.now().isoformat()
            },
            'activations': []
        }
        
        for i, item in enumerate(self.activations):
            activation_data = {
                'index': i,
                'question': item['question'],
                'response': item['response'],
                'layer': item['layer'],
                'activation_tensor': item['activations'].tensor,
                'activation_shape': list(item['activations'].tensor.shape),
                'aggregation_method': item['activations'].aggregation_method.value if hasattr(item['activations'].aggregation_method, 'value') else str(item['activations'].aggregation_method)
            }
            save_data['activations'].append(activation_data)
        
        # Save using torch
        torch.save(save_data, filepath)
        print(f"💾 Saved {len(self.activations)} test activations to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TestActivationCache':
        """Load cached activations from file."""
        import torch
        from .layer import Layer
        
        loaded_data = torch.load(filepath, map_location='cpu')
        cache = cls()
        
        metadata = loaded_data.get('metadata', {})
        print(f"📁 Loading {metadata.get('num_samples', 0)} test activations from {filepath}")
        
        for item in loaded_data['activations']:
            # Reconstruct Activations object
            layer_obj = Layer(index=item['layer'], type="transformer")
            
            # Reconstruct aggregation method
            from .activations import ActivationAggregationMethod
            agg_method = ActivationAggregationMethod.LAST_TOKEN  # default
            try:
                if item.get('aggregation_method'):
                    agg_method = ActivationAggregationMethod(item['aggregation_method'])
            except:
                pass
            
            activations = Activations(
                tensor=item['activation_tensor'],
                layer=layer_obj,
                aggregation_method=agg_method
            )
            
            cache.add_activation(
                question=item['question'],
                response=item['response'],
                activations=activations,
                layer=item['layer']
            )
        
        return cache
    
    def get_activations_for_layer(self, target_layer: int) -> List[Dict]:
        """Get all activations for a specific layer."""
        return [item for item in self.activations if item['layer'] == target_layer]
    
    def __len__(self):
        return len(self.activations)
    
    def __repr__(self):
        layers = list(set(item['layer'] for item in self.activations))
        return f"TestActivationCache({len(self.activations)} samples, layers: {layers})"


