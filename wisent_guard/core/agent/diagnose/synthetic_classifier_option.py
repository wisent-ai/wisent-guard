"""
Synthetic Classifier Option System

Creates custom classifiers from automatically discovered traits using synthetic contrastive pairs.
The model analyzes prompts to determine relevant traits for responses, then creates classifiers for those traits.
The actual response is NEVER analyzed as text - only its activations are classified.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ....core.contrastive_pairs.generate_synthetically import SyntheticContrastivePairGenerator
from ....core.classifier import ActivationClassifier
from ....core.agent.budget import calculate_max_tasks_for_time_budget, get_budget_manager, ResourceType


@dataclass
class TraitDiscoveryResult:
    """Result of automatic trait discovery."""
    traits_discovered: List[str]


@dataclass
class SyntheticClassifierResult:
    """Result of synthetic classifier creation and diagnosis."""
    trait_description: str
    classifier_confidence: float
    prediction: int
    confidence_score: float
    training_pairs_count: int
    generation_time: float


class AutomaticTraitDiscovery:
    """Automatically discovers relevant traits for prompt response analysis."""
    
    def __init__(self, model):
        self.model = model
    
    def discover_relevant_traits(self, prompt: str, time_budget_minutes: float) -> TraitDiscoveryResult:
        """
        Analyze a prompt to automatically discover relevant quality traits for responses.
        
        Args:
            prompt: The prompt/question to analyze for trait discovery
            time_budget_minutes: Time budget for classifier creation in minutes
            
        Returns:
            TraitDiscoveryResult with discovered traits
        """
        # Calculate max traits based on time budget
        max_traits = calculate_max_tasks_for_time_budget("classifier_training", time_budget_minutes)
        max_traits = max(1, min(max_traits, 5))  # Cap between 1-5 traits
        print(f"   💰 Budget system: {time_budget_minutes:.1f} min budget → max {max_traits} traits")
        
        # Generate dynamic trait prompt based on budget
        trait_lines = "\n".join([f"TRAIT_{i+1}:" for i in range(max_traits)])
        
        discovery_prompt = f"""USER PROMPT: {prompt}

List {max_traits} quality traits for responses:
{trait_lines}"""

        try:
            analysis, _ = self.model.generate(
                discovery_prompt,
                layer_index=15,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )
            
            print(f"   🔍 Model generated analysis: {analysis[:200]}...")
            return self._parse_discovery_result(analysis)
            
        except Exception as e:
            print(f"   ⚠️ Error in trait discovery: {e}")
            # Fallback to general traits
            return TraitDiscoveryResult(
                traits_discovered=["accuracy and truthfulness", "helpfulness", "safety"]
            )
    
    def _parse_discovery_result(self, analysis: str) -> TraitDiscoveryResult:
        """Parse the model's trait discovery response."""
        traits = []
        
        lines = analysis.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('TRAIT_'):
                # Extract trait description
                if ':' in line:
                    trait = line.split(':', 1)[1].strip()
                    if len(trait) > 3:
                        traits.append(trait)
        
        return TraitDiscoveryResult(traits_discovered=traits)


class SyntheticClassifierFactory:
    """Creates custom classifiers from trait descriptions using synthetic contrastive pairs."""
    
    def __init__(self, model):
        self.model = model
        self.pair_generator = SyntheticContrastivePairGenerator(model)
    
    def create_classifier_from_trait(
        self,
        trait_description: str,
        num_pairs: int = 15
    ) -> Tuple[ActivationClassifier, int]:
        """
        Create a classifier for a specific trait using synthetic contrastive pairs.
        
        Args:
            trait_description: Natural language description of the trait
            num_pairs: Number of contrastive pairs to generate
            
        Returns:
            Tuple of (trained classifier, number of training pairs)
        """
        try:
            # Generate synthetic contrastive pairs for this trait
            pair_set = self.pair_generator.generate_contrastive_pair_set(
                trait_description=trait_description,
                num_pairs=num_pairs,
                name=f"synthetic_{trait_description[:20].replace(' ', '_')}"
            )
            
            if len(pair_set.pairs) < 3:
                raise ValueError(f"Insufficient training pairs generated: {len(pair_set.pairs)}")
            
            # Extract activations for training
            positive_activations = []
            negative_activations = []
            
            print(f"      🧠 Extracting activations from {len(pair_set.pairs)} pairs...")
            
            # Create Layer object for activation extraction
            from wisent_guard.core.layer import Layer
            layer_obj = Layer(index=15, type="transformer")
            print(f"      🔧 Created Layer object: index={layer_obj.index}, type={layer_obj.type}")
            
            for i, pair in enumerate(pair_set.pairs):
                print(f"      🔍 Processing pair {i+1}/{len(pair_set.pairs)}...")
                try:
                    # Get activations for positive response
                    print(f"         📊 Extracting positive activations for: {repr(pair.positive_response.text[:100])}")
                    pos_activations = self.model.extract_activations(pair.positive_response.text, layer_obj)
                    print(f"         ✅ Positive activations shape: {pos_activations.shape if hasattr(pos_activations, 'shape') else 'N/A'}")
                    positive_activations.append(pos_activations)
                    
                    # Get activations for negative response
                    print(f"         📊 Extracting negative activations for: {repr(pair.negative_response.text[:100])}")
                    neg_activations = self.model.extract_activations(pair.negative_response.text, layer_obj)
                    print(f"         ✅ Negative activations shape: {neg_activations.shape if hasattr(neg_activations, 'shape') else 'N/A'}")
                    negative_activations.append(neg_activations)
                    
                    print(f"         ✅ Successfully processed pair {i+1}")
                    
                except Exception as e:
                    print(f"         ⚠️ Error extracting activations for pair {i+1}: {e}")
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"         📜 Full error traceback:\n{error_details}")
                    continue
            
            print(f"      📊 ACTIVATION EXTRACTION SUMMARY:")
            print(f"         Positive activations collected: {len(positive_activations)}")
            print(f"         Negative activations collected: {len(negative_activations)}")
            print(f"         Total pairs processed: {len(pair_set.pairs)}")
            print(f"         Success rate: {(len(positive_activations) / len(pair_set.pairs) * 100):.1f}%")
            
            if len(positive_activations) < 2 or len(negative_activations) < 2:
                error_msg = f"Insufficient activation data for training: {len(positive_activations)} positive, {len(negative_activations)} negative"
                print(f"      ❌ ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Train classifier on activations
            print(f"      🏋️ Training classifier on {len(positive_activations)} positive, {len(negative_activations)} negative activations...")
            
            print(f"      🔧 Creating ActivationClassifier instance...")
            classifier = ActivationClassifier()
            print(f"      ✅ ActivationClassifier created")
            
            print(f"      🎯 Starting classifier training...")
            try:
                classifier.fit(negative_activations, positive_activations)
                print(f"      ✅ Classifier training completed successfully!")
            except Exception as e:
                print(f"      ❌ ERROR during classifier training: {e}")
                import traceback
                error_details = traceback.format_exc()
                print(f"      📜 Full training error traceback:\n{error_details}")
                raise
            
            return classifier, len(pair_set.pairs)
            
        except Exception as e:
            print(f"   ❌ Error creating classifier for trait '{trait_description}': {e}")
            raise


class SyntheticClassifierSystem:
    """
    Creates synthetic classifiers based on prompt analysis and applies them to response activations.
    
    Analyzes prompts to discover relevant traits, creates classifiers using synthetic 
    contrastive pairs, and applies them to response activations only.
    """
    
    def __init__(self, model):
        self.model = model
        self.trait_discovery = AutomaticTraitDiscovery(model)
        self.classifier_factory = SyntheticClassifierFactory(model)
    
    def create_classifiers_for_prompt(
        self,
        prompt: str,
        time_budget_minutes: float,
        pairs_per_trait: int = 12
    ) -> Tuple[List[ActivationClassifier], TraitDiscoveryResult]:
        """
        Create synthetic classifiers for a prompt by discovering relevant traits.
        
        Args:
            prompt: The prompt to analyze and create classifiers for
            time_budget_minutes: Time budget for classifier creation in minutes
            pairs_per_trait: Number of contrastive pairs per trait
            
        Returns:
            Tuple of (list of trained classifiers, trait discovery result)
        """
        print(f"🔍 Creating synthetic classifiers for prompt (budget: {time_budget_minutes:.1f} minutes)...")
        
        # Step 1: Discover relevant traits for this prompt
        print("   🎯 Discovering relevant traits for this prompt...")
        discovery_result = self.trait_discovery.discover_relevant_traits(prompt, time_budget_minutes)
        
        if not discovery_result.traits_discovered:
            print("   ⚠️ No traits discovered, cannot create classifiers")
            return [], discovery_result
        
        print(f"   ✅ Discovered {len(discovery_result.traits_discovered)} traits: {discovery_result.traits_discovered}")
        
        # Step 2: Create classifiers for discovered traits
        classifiers = []
        
        for i, trait_description in enumerate(discovery_result.traits_discovered):
            print(f"   🎯 Creating classifier {i+1}/{len(discovery_result.traits_discovered)}: {trait_description}")
            
            try:
                # Create classifier for this trait
                print(f"      🏗️ Creating classifier...")
                classifier, pairs_count = self.classifier_factory.create_classifier_from_trait(
                    trait_description=trait_description,
                    num_pairs=pairs_per_trait
                )
                
                # Store trait info in classifier for later reference
                classifier._trait_description = trait_description
                classifier._pairs_count = pairs_count
                
                classifiers.append(classifier)
                
                print(f"      ✅ Classifier created with {pairs_count} training pairs")
                
            except Exception as e:
                print(f"      ❌ Error creating classifier for trait '{trait_description}': {e}")
                continue
        
        print(f"   🎉 Created {len(classifiers)} synthetic classifiers")
        return classifiers, discovery_result
    
    def apply_classifiers_to_response(
        self,
        response_text: str,
        classifiers: List[ActivationClassifier],
        trait_discovery: TraitDiscoveryResult
    ) -> List[SyntheticClassifierResult]:
        """
        Apply pre-trained synthetic classifiers to a response.
        
        Args:
            response_text: The response to analyze (only used for activation extraction)
            classifiers: List of trained classifiers to apply
            trait_discovery: Original trait discovery result for context
            
        Returns:
            List of classification results
        """
        print(f"🔬 Applying {len(classifiers)} synthetic classifiers to response...")
        
        # Extract activations from the response ONCE
        print("   🧠 Extracting activations from response...")
        try:
            response_activations, _ = self.model.extract_activations(response_text, layer=15)
        except Exception as e:
            print(f"   ❌ Error extracting response activations: {e}")
            return []
        
        results = []
        
        for i, classifier in enumerate(classifiers):
            trait_description = getattr(classifier, '_trait_description', f'trait_{i}')
            pairs_count = getattr(classifier, '_pairs_count', 0)
            
            print(f"   🔬 Applying classifier {i+1}/{len(classifiers)}: {trait_description}")
            
            try:
                start_time = time.time()
                
                # Apply classifier to response activations
                prediction = classifier.predict(response_activations)
                confidence = classifier.predict_proba(response_activations)
                
                # Handle confidence score (could be array or scalar)
                if hasattr(confidence, '__iter__') and len(confidence) > 1:
                    confidence_score = float(max(confidence))
                else:
                    confidence_score = float(confidence)
                
                generation_time = time.time() - start_time
                
                result = SyntheticClassifierResult(
                    trait_description=trait_description,
                    classifier_confidence=confidence_score,
                    prediction=int(prediction),
                    confidence_score=confidence_score,
                    training_pairs_count=pairs_count,
                    generation_time=generation_time
                )
                
                results.append(result)
                
                print(f"      ✅ Result: prediction={prediction}, confidence={confidence_score:.3f}")
                
            except Exception as e:
                print(f"      ❌ Error applying classifier for trait '{trait_description}': {e}")
                continue
        
        print(f"   🎉 Applied {len(results)} classifiers successfully")
        return results


def get_time_budget_from_manager() -> float:
    """Get time budget from the global budget manager."""
    budget_manager = get_budget_manager()
    time_budget = budget_manager.get_budget(ResourceType.TIME)
    if not time_budget:
        raise ValueError("No time budget set in budget manager. Call set_time_budget(minutes) first.")
    return time_budget.remaining_budget / 60.0  # Convert to minutes


# Main interface functions
def create_synthetic_classifier_system(model) -> SyntheticClassifierSystem:
    """Create a synthetic classifier system instance."""
    return SyntheticClassifierSystem(model)


def create_classifiers_for_prompt(
    model,
    prompt: str,
    pairs_per_trait: int = 12
) -> Tuple[List[ActivationClassifier], TraitDiscoveryResult]:
    """
    Convenience function to create synthetic classifiers for a prompt.
    
    Args:
        model: The language model instance
        prompt: Prompt to analyze and create classifiers for
        pairs_per_trait: Number of contrastive pairs per trait
        
    Returns:
        Tuple of (trained classifiers, trait discovery result)
    """
    time_budget_minutes = get_time_budget_from_manager()
    system = create_synthetic_classifier_system(model)
    return system.create_classifiers_for_prompt(prompt, time_budget_minutes, pairs_per_trait)


def apply_classifiers_to_response(
    model,
    response_text: str,
    classifiers: List[ActivationClassifier],
    trait_discovery: TraitDiscoveryResult
) -> List[SyntheticClassifierResult]:
    """
    Convenience function to apply classifiers to a response.
    
    Args:
        model: The language model instance
        response_text: Response to analyze
        classifiers: Pre-trained classifiers
        trait_discovery: Original trait discovery result
        
    Returns:
        List of classification results
    """
    system = create_synthetic_classifier_system(model)
    return system.apply_classifiers_to_response(response_text, classifiers, trait_discovery)


def create_classifier_from_trait_description(
    model,
    trait_description: str,
    num_pairs: int = 15
) -> ActivationClassifier:
    """
    Direct function to create a classifier from a trait description.
    
    Args:
        model: The language model instance
        trait_description: Natural language description of the trait (e.g., "accuracy and truthfulness")
        num_pairs: Number of contrastive pairs to generate for training
        
    Returns:
        Trained ActivationClassifier
    """
    import datetime
    
    # Setup logging to file
    log_file = f"synthetic_classifier_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log_and_print(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()}: {message}\n")
    
    log_and_print(f"🎯 Creating classifier for trait: '{trait_description}'")
    log_and_print(f"📋 Parameters: num_pairs={num_pairs}")
    
    # Create synthetic contrastive pair generator
    log_and_print("🏭 Creating SyntheticContrastivePairGenerator...")
    pair_generator = SyntheticContrastivePairGenerator(model)
    log_and_print("✅ SyntheticContrastivePairGenerator created successfully")
    
    # Generate contrastive pairs for this trait
    log_and_print(f"📝 Generating {num_pairs} contrastive pairs...")
    pair_set = pair_generator.generate_contrastive_pair_set(
        trait_description=trait_description,
        num_pairs=num_pairs,
        name=f"synthetic_{trait_description[:20].replace(' ', '_')}"
    )
    
    log_and_print(f"✅ Generated {len(pair_set.pairs)} pairs total")
    
    # Log all generated pairs in detail
    log_and_print("=" * 80)
    log_and_print("DETAILED PAIR ANALYSIS:")
    log_and_print("=" * 80)
    
    for i, pair in enumerate(pair_set.pairs):
        log_and_print(f"\n--- PAIR {i+1}/{len(pair_set.pairs)} ---")
        log_and_print(f"Prompt: {repr(pair.prompt)}")
        log_and_print(f"Positive Response: {repr(pair.positive_response.text)}")
        log_and_print(f"Negative Response: {repr(pair.negative_response.text)}")
        log_and_print(f"Positive Response Type: {type(pair.positive_response)}")
        log_and_print(f"Negative Response Type: {type(pair.negative_response)}")
        log_and_print(f"Positive Response Length: {len(pair.positive_response.text) if hasattr(pair.positive_response, 'text') else 'N/A'}")
        log_and_print(f"Negative Response Length: {len(pair.negative_response.text) if hasattr(pair.negative_response, 'text') else 'N/A'}")
        
        # Check for any special attributes
        if hasattr(pair, '_prompt_pair'):
            log_and_print(f"Has _prompt_pair: {pair._prompt_pair}")
        if hasattr(pair, '_prompt_strategy'):
            log_and_print(f"Has _prompt_strategy: {pair._prompt_strategy}")
    
    log_and_print("=" * 80)
    
    if len(pair_set.pairs) < 3:
        error_msg = f"Insufficient training pairs generated: {len(pair_set.pairs)}"
        log_and_print(f"❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Extract activations for training
    positive_activations = []
    negative_activations = []
    
    log_and_print(f"🧠 Extracting activations from {len(pair_set.pairs)} pairs...")
    
    # Create Layer object for activation extraction
    from wisent_guard.core.layer import Layer
    layer_obj = Layer(index=15, type="transformer")
    log_and_print(f"🔧 Created Layer object: index={layer_obj.index}, type={layer_obj.type}")
    
    for i, pair in enumerate(pair_set.pairs):
        log_and_print(f"\n🔍 Processing pair {i+1}/{len(pair_set.pairs)}...")
        try:
            # Get activations for positive response
            log_and_print(f"   📊 Extracting positive activations for: {repr(pair.positive_response.text[:100])}")
            pos_activations = model.extract_activations(pair.positive_response.text, layer_obj)
            log_and_print(f"   ✅ Positive activations shape: {pos_activations.shape if hasattr(pos_activations, 'shape') else 'N/A'}")
            positive_activations.append(pos_activations)
            
            # Get activations for negative response
            log_and_print(f"   📊 Extracting negative activations for: {repr(pair.negative_response.text[:100])}")
            neg_activations = model.extract_activations(pair.negative_response.text, layer_obj)
            log_and_print(f"   ✅ Negative activations shape: {neg_activations.shape if hasattr(neg_activations, 'shape') else 'N/A'}")
            negative_activations.append(neg_activations)
            
            log_and_print(f"   ✅ Successfully processed pair {i+1}")
            
        except Exception as e:
            log_and_print(f"   ⚠️ Error extracting activations for pair {i+1}: {e}")
            import traceback
            error_details = traceback.format_exc()
            log_and_print(f"   📜 Full error traceback:\n{error_details}")
            continue
    
    log_and_print(f"\n📊 ACTIVATION EXTRACTION SUMMARY:")
    log_and_print(f"   Positive activations collected: {len(positive_activations)}")
    log_and_print(f"   Negative activations collected: {len(negative_activations)}")
    log_and_print(f"   Total pairs processed: {len(pair_set.pairs)}")
    log_and_print(f"   Success rate: {(len(positive_activations) / len(pair_set.pairs) * 100):.1f}%")
    
    if len(positive_activations) < 2 or len(negative_activations) < 2:
        error_msg = f"Insufficient activation data for training: {len(positive_activations)} positive, {len(negative_activations)} negative"
        log_and_print(f"❌ ERROR: {error_msg}")
        raise ValueError(error_msg)

    # Train classifier on activations
    log_and_print(f"🏋️ Training classifier on {len(positive_activations)} positive, {len(negative_activations)} negative activations...")
    
    log_and_print("🔧 Creating ActivationClassifier instance...")
    classifier = ActivationClassifier()
    log_and_print("✅ ActivationClassifier created")
    
    log_and_print("🎯 Starting classifier training...")
    try:
        classifier.fit(negative_activations, positive_activations)
        log_and_print("✅ Classifier training completed successfully!")
    except Exception as e:
        log_and_print(f"❌ ERROR during classifier training: {e}")
        import traceback
        error_details = traceback.format_exc()
        log_and_print(f"📜 Full training error traceback:\n{error_details}")
        raise
    
    # Store metadata
    classifier._trait_description = trait_description
    classifier._pairs_count = len(pair_set.pairs)
    log_and_print(f"📝 Stored metadata: trait='{trait_description}', pairs_count={len(pair_set.pairs)}")
    
    log_and_print(f"🎉 Classifier creation completed successfully!")
    log_and_print(f"📁 Debug log saved to: {log_file}")
    
    return classifier


def evaluate_response_with_trait_classifier(
    model,
    response_text: str,
    trait_classifier: ActivationClassifier
) -> SyntheticClassifierResult:
    """
    Evaluate a response using a trait-specific classifier.
    
    Args:
        model: The language model instance
        response_text: Response to analyze
        trait_classifier: Pre-trained classifier for a specific trait
        
    Returns:
        Classification result
    """
    trait_description = getattr(trait_classifier, '_trait_description', 'unknown_trait')
    pairs_count = getattr(trait_classifier, '_pairs_count', 0)
    
    print(f"🔬 Evaluating response with '{trait_description}' classifier...")
    
    # Extract activations from response
    try:
        response_activations, _ = model.extract_activations(response_text, layer=15)
    except Exception as e:
        raise ValueError(f"Error extracting response activations: {e}")
    
    # Apply classifier
    start_time = time.time()
    prediction = trait_classifier.predict(response_activations)
    confidence = trait_classifier.predict_proba(response_activations)
    
    # Handle confidence score
    if hasattr(confidence, '__iter__') and len(confidence) > 1:
        confidence_score = float(max(confidence))
    else:
        confidence_score = float(confidence)
    
    generation_time = time.time() - start_time
    
    result = SyntheticClassifierResult(
        trait_description=trait_description,
        classifier_confidence=confidence_score,
        prediction=int(prediction),
        confidence_score=confidence_score,
        training_pairs_count=pairs_count,
        generation_time=generation_time
    )
    
    print(f"✅ Result: prediction={prediction}, confidence={confidence_score:.3f}")
    return result 