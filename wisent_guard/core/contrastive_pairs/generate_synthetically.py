import json
import torch
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

from .contrastive_pair import ContrastivePair
from .contrastive_pair_set import ContrastivePairSet
from .quality_check import quality_check_synthetic_pairs
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
        print(f"🎯 DEBUG: Generating scenarios for trait: '{trait_description}'")
        print(f"🎯 DEBUG: Target number of scenarios: {num_scenarios}")
        
        # Overgenerate to ensure diversity
        target_scenarios = num_scenarios * 5  # Increased from 3x to 5x
        all_scenarios = []
        
        print(f"🎯 DEBUG: Will generate {target_scenarios} total scenarios to select {num_scenarios} best ones")
        
        # Different prompt strategies to ensure diversity  
        prompt_templates = [
            f"List {target_scenarios//4} very short, simple questions (maximum 10 words) about everyday life:\n1.",
            f"Generate {target_scenarios//4} brief open-ended questions about common situations:\n1.",
            f"Create {target_scenarios//4} short questions people might ask in casual conversation:\n1.",
            f"Write {target_scenarios//4} concise questions about opinions, decisions, or advice:\n1."
        ]
        
        for i, template in enumerate(prompt_templates):
            print(f"🎯 DEBUG: Using prompt template {i+1}/{len(prompt_templates)}")
            print(f"🎯 DEBUG: Template: {template[:100]}...")
            try:
                response, _ = self.model.generate(
                    template,
                    layer_index=15,  # Use middle layer
                    max_new_tokens=500,
                    do_sample=True
                )
                
                print(f"🎯 DEBUG: Generated response length: {len(response)} chars")
                print(f"🎯 DEBUG: Response preview: {response[:200]}...")
                
                # Parse scenarios from response
                scenarios = self._parse_scenarios_from_response(response)
                print(f"🎯 DEBUG: Parsed {len(scenarios)} scenarios from this template")
                for j, scenario in enumerate(scenarios):
                    print(f"🎯 DEBUG:   Scenario {j+1}: {scenario[:100]}...")
                all_scenarios.extend(scenarios)
                
            except Exception as e:
                print(f"🎯 DEBUG: Error generating scenarios with template: {e}")
                continue
        
        print(f"🎯 DEBUG: Total scenarios before deduplication: {len(all_scenarios)}")
        
        # Deduplicate and select most diverse scenarios
        unique_scenarios = self._deduplicate_scenarios(all_scenarios)
        print(f"🎯 DEBUG: Unique scenarios after deduplication: {len(unique_scenarios)}")
        
        # Select the best diverse scenarios
        selected_scenarios = self._select_diverse_scenarios(unique_scenarios, num_scenarios)
        print(f"🎯 DEBUG: Final selected scenarios: {len(selected_scenarios)}")
        
        for i, scenario in enumerate(selected_scenarios):
            print(f"🎯 DEBUG: Final scenario {i+1}: {scenario}")
        
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
            
            # Remove markdown formatting
            cleaned = cleaned.replace('**', '').replace('*', '')
            
            # Filter out meta-descriptions and keep only actual questions/scenarios
            skip_phrases = [
                'here are', 'here is', 'these are', 'this is', 'the following',
                'examples of', 'scenarios where', 'situations where', 'cases where',
                'questions that', 'prompts that', 'list of', 'different situations',
                'advice-seeking:', 'opinion:', 'conversation starter:', 'prompt:', 'scenario:'
            ]
            
            cleaned_lower = cleaned.lower()
            is_meta = any(phrase in cleaned_lower for phrase in skip_phrases)
            
            # Keep only short questions and prompts
            if (len(cleaned) > 10 and len(cleaned) < 80 and not is_meta):
                # Reject model refusals
                refusal_phrases = ['i cannot', "i can't", 'i am not able', 'i cannot provide', 'i cannot create', 
                                 'i cannot generate', 'harmful', 'evil behavior', 'glorify']
                if any(phrase in cleaned_lower for phrase in refusal_phrases):
                    continue
                
                # Prefer short questions (under 15 words)
                word_count = len(cleaned.split())
                if '?' in cleaned and word_count <= 15:
                    scenarios.append(cleaned)
                # Accept very short imperative statements
                elif word_count <= 10 and any(word in cleaned_lower for word in ['tell me', 'explain', 'describe', 'what', 'how', 'why']):
                    scenarios.append(cleaned)
        
        return scenarios[:25]  # Increased limit per response to ensure we get enough
    
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
        print(f"🔄 DEBUG: Generating contrastive pair for scenario: {scenario[:100]}...")
        print(f"🔄 DEBUG: Trait: {trait_description}")
        
        # Generate positive response (demonstrates the trait)
        positive_prompt = f"""{scenario}

Respond directly with {trait_description} (no preamble):"""
        
        print(f"🔄 DEBUG: Positive prompt: {positive_prompt}")
        
        positive_response, _ = self.model.generate(
            positive_prompt,
            layer_index=15,
            max_new_tokens=50,
            do_sample=True
        )
        
        print(f"🔄 DEBUG: Positive response: {positive_response[:100]}...")
        
        # Generate negative response (opposite of trait)
        negative_prompt = f"""{scenario}

Respond directly with the opposite of {trait_description} (no preamble):"""
        
        print(f"🔄 DEBUG: Negative prompt: {negative_prompt}")
        
        negative_response, _ = self.model.generate(
            negative_prompt,
            layer_index=15,
            max_new_tokens=50,
            do_sample=True
        )
        
        print(f"🔄 DEBUG: Negative response: {negative_response[:100]}...")
        
        # Create contrastive pair - always use the question directly
        prompt = scenario.strip()
        print(f"🔄 DEBUG: Using question as direct prompt: {prompt}")
        
        pair = ContrastivePair(
            prompt=prompt,
            positive_response=PositiveResponse(text=positive_response.strip()),
            negative_response=NegativeResponse(text=negative_response.strip())
        )
        
        # Store metadata
        pair.scenario = scenario
        pair.trait_description = trait_description
        
        print(f"🔄 DEBUG: Created contrastive pair successfully")
        
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
        
        # Apply quality check to filter out low-quality pairs
        print("🔍 Applying quality check to filter pairs...")
        filtered_pair_set = quality_check_synthetic_pairs(
            pair_set, 
            trait_description, 
            strict_mode=True
        )
        
        return filtered_pair_set
    
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
