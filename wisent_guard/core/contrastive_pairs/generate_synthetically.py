import json
import torch
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

from .contrastive_pair import ContrastivePair
from .contrastive_pair_set import ContrastivePairSet
from ..response import PositiveResponse, NegativeResponse


class SyntheticContrastivePairGenerator:
    """Generate contrastive pairs synthetically from natural language trait descriptions."""
    
    def __init__(self, model, similarity_threshold: float = 0.8):
        """
        Initialize the synthetic pair generator.
        
        Args:
            model: The language model to use for generation
            similarity_threshold: Threshold for deduplication (0-1, higher = more strict)
        """
        self.model = model
        self.similarity_threshold = similarity_threshold
        
        # Load sentence transformer for similarity checking
        try:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load similarity model: {e}")
            self.similarity_model = None
    
    def generate_scenarios(self, trait_description: str, num_scenarios: int) -> List[str]:
        """
        Generate diverse scenarios where the trait would be relevant.
        
        Args:
            trait_description: Natural language description of desired trait
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of scenario descriptions
        """
        # Overgenerate to ensure diversity
        target_scenarios = num_scenarios * 3
        all_scenarios = []
        
        # Different prompt strategies to ensure diversity
        prompt_templates = [
            f"Generate specific scenarios where this trait is important: '{trait_description}'. List {target_scenarios//4} different situations:",
            f"Think of edge cases and challenging situations for: '{trait_description}'. Provide {target_scenarios//4} examples:",
            f"Consider various contexts where '{trait_description}' would apply. Give {target_scenarios//4} diverse scenarios:",
            f"What are subtle or complex situations where '{trait_description}' matters? List {target_scenarios//4} cases:"
        ]
        
        for template in prompt_templates:
            try:
                response, _ = self.model.generate(
                    template,
                    layer_index=15,  # Use middle layer
                    max_new_tokens=500,
                    temperature=0.9,  # Higher temperature for creativity
                    do_sample=True
                )
                
                # Parse scenarios from response
                scenarios = self._parse_scenarios_from_response(response)
                all_scenarios.extend(scenarios)
                
            except Exception as e:
                print(f"Warning: Error generating scenarios with template: {e}")
                continue
        
        # Deduplicate and select most diverse scenarios
        unique_scenarios = self._deduplicate_scenarios(all_scenarios)
        
        # Select the best diverse scenarios
        selected_scenarios = self._select_diverse_scenarios(unique_scenarios, num_scenarios)
        
        return selected_scenarios
    
    def _parse_scenarios_from_response(self, response: str) -> List[str]:
        """Parse individual scenarios from model response."""
        scenarios = []
        
        # Split by common delimiters
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or len(line) < 10:
                continue
            
            # Remove numbering and bullet points
            cleaned = line
            for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', 
                          '-', '*', '•', 'a)', 'b)', 'c)', 'd)', 'e)']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            # Only keep substantial scenarios
            if len(cleaned) > 15 and len(cleaned) < 200:
                scenarios.append(cleaned)
        
        return scenarios[:10]  # Limit per response
    
    def _deduplicate_scenarios(self, scenarios: List[str]) -> List[str]:
        """Remove duplicate or very similar scenarios."""
        if not self.similarity_model or len(scenarios) <= 1:
            # Fallback to simple text-based deduplication
            return list(set(scenarios))
        
        unique_scenarios = []
        
        for scenario in scenarios:
            is_duplicate = False
            
            if unique_scenarios:
                # Check similarity with existing scenarios
                scenario_embedding = self.similarity_model.encode([scenario])
                existing_embeddings = self.similarity_model.encode(unique_scenarios)
                
                # Calculate cosine similarities
                similarities = np.dot(scenario_embedding, existing_embeddings.T)[0]
                
                if np.max(similarities) > self.similarity_threshold:
                    is_duplicate = True
            
            if not is_duplicate:
                unique_scenarios.append(scenario)
        
        return unique_scenarios
    
    def _select_diverse_scenarios(self, scenarios: List[str], target_count: int) -> List[str]:
        """Select the most diverse scenarios up to target count."""
        if len(scenarios) <= target_count:
            return scenarios
        
        if not self.similarity_model:
            # Random selection fallback
            return random.sample(scenarios, target_count)
        
        # Use embeddings to select diverse scenarios
        embeddings = self.similarity_model.encode(scenarios)
        
        selected_indices = [0]  # Start with first scenario
        
        for _ in range(target_count - 1):
            remaining_indices = [i for i in range(len(scenarios)) if i not in selected_indices]
            
            if not remaining_indices:
                break
            
            # Find scenario most different from already selected ones
            max_min_distance = -1
            best_idx = remaining_indices[0]
            
            for idx in remaining_indices:
                # Calculate minimum distance to any selected scenario
                distances = []
                for selected_idx in selected_indices:
                    distance = 1 - np.dot(embeddings[idx], embeddings[selected_idx])
                    distances.append(distance)
                
                min_distance = min(distances)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx
            
            selected_indices.append(best_idx)
        
        return [scenarios[i] for i in selected_indices]
    
    def generate_contrastive_pair(self, scenario: str, trait_description: str) -> ContrastivePair:
        """
        Generate a contrastive pair for a specific scenario.
        
        Args:
            scenario: The scenario to generate responses for
            trait_description: The trait description for context
            
        Returns:
            ContrastivePair object
        """
        # Generate positive response (demonstrates good trait)
        positive_prompt = f"""Scenario: {scenario}

Show how a model should respond to demonstrate this trait: "{trait_description}"

Response:"""
        
        positive_response, _ = self.model.generate(
            positive_prompt,
            layer_index=15,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        
        # Generate negative response (opposite of trait)
        negative_prompt = f"""Scenario: {scenario}

Show how a model should NOT respond (demonstrate the opposite of this trait): "{trait_description}"

Response:"""
        
        negative_response, _ = self.model.generate(
            negative_prompt,
            layer_index=15,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        
        # Create contrastive pair
        prompt = f"Scenario: {scenario}"
        
        pair = ContrastivePair(
            prompt=prompt,
            positive_response=PositiveResponse(text=positive_response.strip()),
            negative_response=NegativeResponse(text=negative_response.strip())
        )
        
        # Store metadata
        pair.scenario = scenario
        pair.trait_description = trait_description
        
        return pair
    
    def generate_contrastive_pair_set(
        self,
        trait_description: str,
        num_pairs: int = 30,
        name: Optional[str] = None
    ) -> ContrastivePairSet:
        """
        Generate a complete contrastive pair set from a trait description.
        
        Args:
            trait_description: Natural language description of desired trait
            num_pairs: Number of contrastive pairs to generate
            name: Optional name for the pair set
            
        Returns:
            ContrastivePairSet with generated pairs
        """
        print(f"🎯 Generating {num_pairs} contrastive pairs for trait: '{trait_description}'")
        
        # Generate diverse scenarios
        print("📝 Generating diverse scenarios...")
        scenarios = self.generate_scenarios(trait_description, num_pairs)
        print(f"✅ Generated {len(scenarios)} unique scenarios")
        
        # Generate contrastive pairs for each scenario
        print("🔄 Generating contrastive pairs...")
        pair_set = ContrastivePairSet(
            name=name or f"synthetic_{trait_description[:30]}",
            task_type="synthetic"
        )
        
        for i, scenario in enumerate(scenarios):
            try:
                print(f"   Generating pair {i+1}/{len(scenarios)}: {scenario[:50]}...")
                pair = self.generate_contrastive_pair(scenario, trait_description)
                pair_set.pairs.append(pair)
            except Exception as e:
                print(f"   ⚠️ Error generating pair for scenario '{scenario[:50]}': {e}")
                continue
        
        print(f"✅ Successfully generated {len(pair_set.pairs)} contrastive pairs")
        
        return pair_set
    
    def save_to_json(self, pair_set: ContrastivePairSet, filepath: str) -> None:
        """Save contrastive pair set to JSON file."""
        data = {
            "name": pair_set.name,
            "task_type": pair_set.task_type,
            "pairs": []
        }
        
        for pair in pair_set.pairs:
            pair_data = {
                "prompt": pair.prompt,
                "positive_response": pair.positive_response.text,
                "negative_response": pair.negative_response.text
            }
            data["pairs"].append(pair_data)
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(pair_set.pairs)} pairs to {filepath}")
    
    def load_from_json(self, filepath: str) -> ContrastivePairSet:
        """Load contrastive pair set from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pair_set = ContrastivePairSet(
            name=data.get("name", "loaded_synthetic"),
            task_type=data.get("task_type", "synthetic")
        )
        
        for pair_data in data["pairs"]:
            pair = ContrastivePair(
                prompt=pair_data["prompt"],
                positive_response=PositiveResponse(text=pair_data["positive_response"]),
                negative_response=NegativeResponse(text=pair_data["negative_response"])
            )
            
            pair_set.pairs.append(pair)
        
        print(f"📂 Loaded {len(pair_set.pairs)} pairs from {filepath}")
        
        return pair_set


def generate_synthetic_pairs_cli(
    trait_description: str,
    num_pairs: int = 30,
    output_file: Optional[str] = None,
    model=None
) -> ContrastivePairSet:
    """
    CLI function to generate synthetic contrastive pairs.
    
    Args:
        trait_description: Natural language description of desired trait
        num_pairs: Number of pairs to generate
        output_file: Optional file to save pairs to
        model: Model instance to use
        
    Returns:
        Generated ContrastivePairSet
    """
    if model is None:
        raise ValueError("Model must be provided")
    
    generator = SyntheticContrastivePairGenerator(model)
    
    pair_set = generator.generate_contrastive_pair_set(
        trait_description=trait_description,
        num_pairs=num_pairs,
        name=f"synthetic_{trait_description.replace(' ', '_')[:20]}"
    )
    
    if output_file:
        generator.save_to_json(pair_set, output_file)
    
    return pair_set


def load_synthetic_pairs_cli(filepath: str, model=None) -> ContrastivePairSet:
    """
    CLI function to load synthetic contrastive pairs from JSON.
    
    Args:
        filepath: Path to JSON file
        model: Model instance (for compatibility)
        
    Returns:
        Loaded ContrastivePairSet
    """
    generator = SyntheticContrastivePairGenerator(model)
    return generator.load_from_json(filepath)
