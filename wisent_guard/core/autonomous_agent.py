#!/usr/bin/env python3
"""
Autonomous Wisent-Guard Agent

A model that can autonomously use wisent-guard capabilities on itself:
- Generate responses
- Analyze its own outputs for issues
- Auto-discover or create classifiers on demand
- Apply corrections to improve future responses
"""

import asyncio
from typing import Any, Dict, List, Optional

from wisent_guard.core.activations import ActivationAggregationStrategy, Activations

from .agent.diagnose import AgentClassifierDecisionSystem, AnalysisResult, ClassifierMarketplace, ResponseDiagnostics
from .agent.steer import ImprovementResult, ResponseSteering
from .model import Model


class AutonomousAgent:
    """
    An autonomous agent that can generate responses, analyze them for issues,
    and improve them using activation-based steering and correction techniques.

    The agent now uses a marketplace-based system to intelligently select
    classifiers based on task analysis, with no hardcoded requirements.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        layer_override: int = None,
        enable_tracking: bool = True,
        steering_method: str = "CAA",
        steering_strength: float = 1.0,
        steering_mode: bool = False,
        normalization_method: str = "none",
        target_norm: Optional[float] = None,
        hpr_beta: float = 1.0,
        dac_dynamic_control: bool = False,
        dac_entropy_threshold: float = 1.0,
        bipo_beta: float = 0.1,
        bipo_learning_rate: float = 5e-4,
        bipo_epochs: int = 100,
        ksteering_num_labels: int = 6,
        ksteering_hidden_dim: int = 512,
        ksteering_learning_rate: float = 1e-3,
        ksteering_classifier_epochs: int = 100,
        ksteering_target_labels: str = "0",
        ksteering_avoid_labels: str = "",
        ksteering_alpha: float = 50.0,
        # Priority-aware benchmark selection parameters
        priority: str = "all",
        fast_only: bool = False,
        time_budget_minutes: float = None,
        max_benchmarks: int = None,
        smart_selection: bool = False,
    ):
        """
        Initialize the autonomous agent.

        Args:
            model_name: Name of the model to use
            layer_override: Layer override from CLI (None to use parameter file)
            enable_tracking: Whether to track improvement history
            steering_method: Steering method to use (CAA, HPR, DAC, BiPO, KSteering)
            steering_strength: Strength of steering application
            steering_mode: Whether to enable steering mode
            priority: Priority level for benchmark selection ("all", "high", "medium", "low")
            fast_only: Only use fast benchmarks (high priority)
            time_budget_minutes: Time budget in minutes for benchmark selection
            max_benchmarks: Maximum number of benchmarks to select
            smart_selection: Use smart benchmark selection based on relevance and priority
            prefer_fast: Prefer fast benchmarks in selection
            (... other steering parameters ...)
        """
        self.model_name = model_name
        self.model: Optional[Model] = None
        self.layer_override = layer_override
        self.enable_tracking = enable_tracking

        # Load model parameters first
        from .parameters import load_model_parameters

        self.params = load_model_parameters(model_name, layer_override)

        # Store steering parameters and load method-specific configs from parameter file
        self.steering_method = steering_method
        self.steering_strength = steering_strength
        self.steering_mode = steering_mode
        self.normalization_method = normalization_method
        self.target_norm = target_norm

        # Load method-specific parameters from parameter file, with CLI overrides
        steering_config = self.params.get_steering_config(steering_method)

        self.hpr_beta = hpr_beta if hpr_beta != 1.0 else steering_config.get("beta", 1.0)
        self.dac_dynamic_control = (
            dac_dynamic_control if dac_dynamic_control else steering_config.get("dynamic_control", False)
        )
        self.dac_entropy_threshold = (
            dac_entropy_threshold if dac_entropy_threshold != 1.0 else steering_config.get("entropy_threshold", 1.0)
        )
        self.bipo_beta = bipo_beta if bipo_beta != 0.1 else steering_config.get("beta", 0.1)
        self.bipo_learning_rate = (
            bipo_learning_rate if bipo_learning_rate != 5e-4 else steering_config.get("learning_rate", 5e-4)
        )
        self.bipo_epochs = bipo_epochs if bipo_epochs != 100 else steering_config.get("num_epochs", 100)
        self.ksteering_num_labels = (
            ksteering_num_labels if ksteering_num_labels != 6 else steering_config.get("num_labels", 6)
        )
        self.ksteering_hidden_dim = (
            ksteering_hidden_dim if ksteering_hidden_dim != 512 else steering_config.get("hidden_dim", 512)
        )
        self.ksteering_learning_rate = (
            ksteering_learning_rate if ksteering_learning_rate != 1e-3 else steering_config.get("learning_rate", 1e-3)
        )
        self.ksteering_classifier_epochs = (
            ksteering_classifier_epochs
            if ksteering_classifier_epochs != 100
            else steering_config.get("classifier_epochs", 100)
        )
        self.ksteering_target_labels = (
            ksteering_target_labels
            if ksteering_target_labels != "0"
            else ",".join(map(str, steering_config.get("target_labels", [0])))
        )
        self.ksteering_avoid_labels = (
            ksteering_avoid_labels
            if ksteering_avoid_labels != ""
            else ",".join(map(str, steering_config.get("avoid_labels", [])))
        )
        self.ksteering_alpha = ksteering_alpha if ksteering_alpha != 50.0 else steering_config.get("alpha", 50.0)

        # Priority-aware benchmark selection parameters
        self.priority = priority
        self.fast_only = fast_only
        self.time_budget_minutes = time_budget_minutes
        self.max_benchmarks = max_benchmarks
        self.smart_selection = smart_selection

        # New marketplace-based system
        self.marketplace: Optional[ClassifierMarketplace] = None
        self.decision_system: Optional[AgentClassifierDecisionSystem] = None
        self.diagnostics: Optional[ResponseDiagnostics] = None
        self.steering: Optional[ResponseSteering] = None

        # Tracking
        self.improvement_history: List[ImprovementResult] = []
        self.analysis_history: List[AnalysisResult] = []

        print(f"🤖 Autonomous Agent initialized with {model_name}")
        print("   🎯 Using marketplace-based classifier selection")
        print(f"   🎛️ Steering: {steering_method} (strength: {steering_strength})")
        if steering_mode:
            print(f"   🔧 Steering mode enabled with {normalization_method} normalization")
        print(self.params.get_summary())

    async def initialize(
        self,
        classifier_search_paths: Optional[List[str]] = None,
        quality_threshold: float = 0.3,
        default_time_budget_minutes: float = 10.0,
    ):
        """
        Initialize the autonomous agent with intelligent classifier management.

        Args:
            classifier_search_paths: Paths to search for existing classifiers
            quality_threshold: Minimum quality threshold for existing classifiers
            default_time_budget_minutes: Default time budget for creating new classifiers
        """
        print("🚀 Initializing Autonomous Agent...")

        # Load model
        print("   📦 Loading model...")
        self.model = Model(self.model_name)

        # Initialize marketplace
        print("   🏪 Setting up classifier marketplace...")
        self.marketplace = ClassifierMarketplace(model=self.model, search_paths=classifier_search_paths)

        # Initialize decision system
        print("   🧠 Setting up intelligent decision system...")
        self.decision_system = AgentClassifierDecisionSystem(self.marketplace)

        # Store configuration
        self.quality_threshold = quality_threshold
        self.default_time_budget_minutes = default_time_budget_minutes

        # Show marketplace summary
        summary = self.marketplace.get_marketplace_summary()
        print(summary)

        print("   ✅ Autonomous Agent ready!")

    async def respond_autonomously(
        self,
        prompt: str,
        max_attempts: int = 3,
        quality_threshold: float = None,
        time_budget_minutes: float = None,
        max_classifiers: int = None,
    ) -> Dict[str, Any]:
        """
        Generate a response and autonomously improve it if needed.
        The agent will intelligently select classifiers based on the prompt.

        Args:
            prompt: The prompt to respond to
            max_attempts: Maximum improvement attempts
            quality_threshold: Quality threshold for classifiers (uses default if None)
            time_budget_minutes: Time budget for creating classifiers (uses default if None)
            max_classifiers: Maximum classifiers to use (None = no limit)

        Returns:
            Dictionary with response and improvement details
        """
        print(f"\n🎯 AUTONOMOUS RESPONSE TO: {prompt[:100]}...")

        # Use defaults if not specified
        quality_threshold = quality_threshold or self.quality_threshold
        time_budget_minutes = time_budget_minutes or self.default_time_budget_minutes

        # Step 1: Intelligent classifier selection based on the prompt
        print("\n🧠 Analyzing task and selecting classifiers...")
        classifier_configs = await self.decision_system.smart_classifier_selection(
            prompt=prompt,
            quality_threshold=quality_threshold,
            time_budget_minutes=time_budget_minutes,
            max_classifiers=max_classifiers,
        )

        # Step 2: Initialize diagnostics and steering with selected classifiers
        if classifier_configs:
            print(f"   📊 Initializing diagnostics with {len(classifier_configs)} classifiers")
            self.diagnostics = ResponseDiagnostics(model=self.model, classifier_configs=classifier_configs)

            self.steering = ResponseSteering(
                generate_response_func=self._generate_response, analyze_response_func=self.diagnostics.analyze_response
            )
        else:
            print("   ⚠️ No classifiers selected - proceeding without advanced diagnostics")
            # Could fall back to basic text analysis or skip diagnostics
            return {
                "final_response": await self._generate_response(prompt),
                "attempts": 1,
                "improvement_chain": [],
                "classifier_info": "No classifiers used",
            }

        # Step 3: Generate and improve response
        attempt = 0
        current_response = None
        improvement_chain = []

        while attempt < max_attempts:
            attempt += 1
            print(f"\n--- Attempt {attempt} ---")

            # Generate response
            if current_response is None:
                print("💭 Generating initial response...")
                current_response = await self._generate_response(prompt)
                print(f"   Response: {current_response[:100]}...")

            # Analyze response using selected classifiers
            print("🔍 Analyzing response...")
            analysis = await self.diagnostics.analyze_response(current_response, prompt)

            print(f"   Issues found: {analysis.issues_found}")
            print(f"   Quality score: {analysis.quality_score:.2f}")
            print(f"   Confidence: {analysis.confidence:.2f}")

            # Track analysis
            if self.enable_tracking:
                self.analysis_history.append(analysis)

            # Decide if improvement is needed
            needs_improvement = self._decide_if_improvement_needed(analysis)

            if not needs_improvement:
                print("✅ Response quality acceptable, no improvement needed")
                break

            # Attempt improvement
            print("🛠️ Attempting to improve response...")
            improvement = await self.steering.improve_response(prompt, current_response, analysis)

            if improvement.success:
                print(f"   Improvement successful! Score: {improvement.improvement_score:.2f}")
                current_response = improvement.improved_response
                improvement_chain.append(improvement)

                if self.enable_tracking:
                    self.improvement_history.append(improvement)
            else:
                print("   Improvement failed, keeping original response")
                break

        return {
            "final_response": current_response,
            "attempts": attempt,
            "improvement_chain": improvement_chain,
            "final_analysis": analysis,
            "classifier_info": {
                "count": len(classifier_configs),
                "types": [c.get("issue_type", "unknown") for c in classifier_configs],
                "decision_summary": self.decision_system.get_decision_summary(),
            },
        }

    async def _generate_response(self, prompt: str) -> str:
        """Generate a response to the prompt with optional steering."""
        if self.steering_mode:
            # Use actual activation steering
            print(f"   🎛️ Applying {self.steering_method} steering...")
            try:
                # Use actual steering methods from steering_methods folder
                from ..inference import generate_with_classification_and_handling

                # Create steering method object based on configuration
                steering_method = self._create_steering_method()

                response, _, _, _ = generate_with_classification_and_handling(
                    self.model,
                    prompt,
                    self.params.layer,
                    max_new_tokens=200,
                    steering_method=steering_method,
                    token_aggregation="average",
                    threshold=0.6,
                    verbose=False,
                    detection_handler=None,
                )
                return response

            except Exception as e:
                print(f"   ⚠️ Steering failed, falling back to basic generation: {e}")
                # Fall through to basic generation

        # Basic generation without steering
        result = self.model.generate(prompt, self.params.layer, max_new_tokens=200)
        # Handle both 2 and 3 return values
        if isinstance(result, tuple) and len(result) == 3:
            response, _, _ = result
        elif isinstance(result, tuple) and len(result) == 2:
            response, _ = result
        else:
            response = result
        return response

    def _create_steering_method(self):
        """Create a steering method object based on configuration."""
        # Import actual steering methods
        from .steering_methods import CAA, DAC, HPR, BiPO, KSteering

        # Create the appropriate steering method with parameters
        if self.steering_method == "CAA":
            steering_method = CAA(device=None)
        elif self.steering_method == "HPR":
            steering_method = HPR(device=None, beta=self.hpr_beta)
        elif self.steering_method == "DAC":
            steering_method = DAC(
                device=None, dynamic_control=self.dac_dynamic_control, entropy_threshold=self.dac_entropy_threshold
            )
        elif self.steering_method == "BiPO":
            steering_method = BiPO(
                device=None, beta=self.bipo_beta, learning_rate=self.bipo_learning_rate, num_epochs=self.bipo_epochs
            )
        elif self.steering_method == "KSteering":
            # Parse target and avoid labels
            target_labels = [int(x.strip()) for x in self.ksteering_target_labels.split(",") if x.strip()]
            avoid_labels = [int(x.strip()) for x in self.ksteering_avoid_labels.split(",") if x.strip()]

            steering_method = KSteering(
                device=None,
                num_labels=self.ksteering_num_labels,
                hidden_dim=self.ksteering_hidden_dim,
                learning_rate=self.ksteering_learning_rate,
                classifier_epochs=self.ksteering_classifier_epochs,
                target_labels=target_labels,
                avoid_labels=avoid_labels,
                alpha=self.ksteering_alpha,
            )
        else:
            # Default to CAA
            steering_method = CAA(device=None)

        return steering_method

    async def evaluate_response_quality(
        self, response: str, classifier, classifier_params: "ClassifierParams"
    ) -> "QualityResult":
        """
        Evaluate response quality using classifier + model judgment for threshold determination.

        Args:
            response: The response to evaluate
            classifier: The trained classifier to use
            classifier_params: Parameters used for classifier training

        Returns:
            QualityResult with score and acceptability judgment
        """
        from .agent.diagnose.agent_classifier_decision import QualityResult

        # Extract activations and get classifier score
        activations_tensor = self.model.extract_activations(response, classifier_params.optimal_layer)
        from .layer import Layer

        activations = Activations(
            tensor=activations_tensor,
            layer=Layer(index=classifier_params.optimal_layer, type="transformer"),
            aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
        )

        features = activations.extract_features_for_classifier()
        features_cpu = features.cpu() if hasattr(features, "cpu") else features

        # Get classifier prediction
        prob_result = classifier.predict_proba([features_cpu.numpy()])
        score = prob_result[0] if isinstance(prob_result, (list, tuple)) else prob_result

        # Use model to determine if this score represents acceptable quality
        acceptable = await self._model_determines_quality_threshold(response, float(score))

        return QualityResult(
            score=float(score),
            acceptable=acceptable,
            reasoning=f"Model judged score {score:.3f} as {'acceptable' if acceptable else 'unacceptable'}",
        )

    async def _model_determines_quality_threshold(self, response: str, classifier_score: float) -> bool:
        """
        Use model to determine if classifier score represents acceptable quality.

        Args:
            response: The response to evaluate
            classifier_score: Score from the classifier (0-1, where 1 typically means more problematic)

        Returns:
            Boolean indicating if quality is acceptable
        """
        threshold_prompt = f"""
        Evaluate if this response quality is acceptable for a user:
        
        Response: "{response}"
        
        Quality Classifier Score: {classifier_score:.3f}
        (Note: Score ranges 0-1, where 0 = high quality, 1 = low quality/problematic)
        
        Consider:
        - Is the response helpful and informative?
        - Are there factual errors or concerning content?
        - Is the response appropriate length and detail?
        - Would a typical user be satisfied with this response?
        
        Respond ONLY with:
        "ACCEPTABLE" if the response quality is good enough for the user
        "UNACCEPTABLE" if the response needs improvement
        """

        # Generate model judgment
        result = self.model.generate(threshold_prompt, layer_index=15, max_new_tokens=20)
        judgment = result[0] if isinstance(result, tuple) else result
        judgment = judgment.strip().upper()

        return "ACCEPTABLE" in judgment

    async def _determine_classifier_parameters(self, prompt: str, benchmark_names: List[str]) -> "ClassifierParams":
        """
        Use model to determine optimal classifier parameters based on prompt analysis.

        Args:
            prompt: The user prompt to analyze
            benchmark_names: Selected benchmarks for training

        Returns:
            ClassifierParams with model-determined parameters
        """

        parameter_prompt = f"""
        Analyze this prompt and determine optimal classifier parameters:
        
        Prompt: "{prompt}"
        Selected Benchmarks: {benchmark_names}
        
        Consider:
        - Prompt complexity (simple conversational vs complex technical)
        - Domain type (technical/casual/creative/factual)
        - Expected response length and detail needs
        - Quality requirements and safety considerations
        
        Determine optimal parameters:
        1. Optimal Layer (8-20): What layer captures the right semantic complexity?
           - Simple prompts: layers 8-12
           - Medium complexity: layers 12-16  
           - Complex technical: layers 16-20
        
        2. Classification Threshold (0.1-0.9): How strict should quality detection be?
           - Lenient (casual conversation): 0.1-0.3
           - Moderate (general use): 0.4-0.6
           - Strict (important/technical): 0.7-0.9
        
        3. Training Samples (10-50): How many samples needed for good training?
           - Simple patterns: 10-20 samples
           - Medium complexity: 20-35 samples  
           - Complex patterns: 35-50 samples
        
        4. Classifier Type: What classifier works best for this data?
           - logistic: Simple patterns, fast training
           - svm: Medium complexity, robust
           - neural: Complex patterns, more data needed
        
        Format your response as:
        LAYER: [number]
        THRESHOLD: [number]
        SAMPLES: [number]
        TYPE: [logistic/svm/neural]
        REASONING: [one sentence explanation]
        """

        # Generate model response
        result = self.model.generate(parameter_prompt, layer_index=15, max_new_tokens=150)
        response = result[0] if isinstance(result, tuple) else result

        # Parse the response
        return self._parse_classifier_params(response)

    def _parse_classifier_params(self, response: str) -> "ClassifierParams":
        """Parse model response to extract classifier parameters."""
        from .agent.diagnose.agent_classifier_decision import ClassifierParams

        # Default values in case parsing fails
        layer = 15
        threshold = 0.5
        samples = 25
        classifier_type = "logistic"
        reasoning = "Using default parameters due to parsing failure"

        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("LAYER:"):
                    layer = int(line.split(":")[1].strip())
                elif line.startswith("THRESHOLD:"):
                    threshold = float(line.split(":")[1].strip())
                elif line.startswith("SAMPLES:"):
                    samples = int(line.split(":")[1].strip())
                elif line.startswith("TYPE:"):
                    classifier_type = line.split(":")[1].strip().lower()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
        except Exception as e:
            print(f"   ⚠️ Failed to parse classifier parameters: {e}")
            print(
                f"   📋 Using defaults: layer={layer}, threshold={threshold}, samples={samples}, type={classifier_type}"
            )

        # Validate ranges
        layer = max(8, min(20, layer))
        threshold = max(0.1, min(0.9, threshold))
        samples = max(10, min(50, samples))
        if classifier_type not in ["logistic", "svm", "neural"]:
            classifier_type = "logistic"

        return ClassifierParams(
            optimal_layer=layer,
            classification_threshold=threshold,
            training_samples=samples,
            classifier_type=classifier_type,
            reasoning=reasoning,
            model_name=self.model_name,
            aggregation_method="last_token",  # Default for model-determined params
            token_aggregation="average",  # Default for model-determined params
            num_epochs=50,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=10,
            hidden_dim=128,
        )

    async def _determine_steering_parameters(
        self, prompt: str, current_quality: float, attempt_number: int
    ) -> "SteeringParams":
        """
        Use model to determine optimal steering parameters based on current quality and prompt.

        Args:
            prompt: The original user prompt
            current_quality: Current quality score from classifier
            attempt_number: Which attempt this is (1, 2, 3...)

        Returns:
            SteeringParams with model-determined parameters
        """

        steering_prompt = f"""
        Determine optimal steering parameters for improving this response:
        
        Original Prompt: "{prompt}"
        Current Quality Score: {current_quality:.3f} (0=good, 1=bad)
        Attempt Number: {attempt_number}
        
        Available Steering Methods:
        - CAA: Gentle activation steering, good for general improvements
        - HPR: Precise harmfulness reduction, good for safety issues
        - DAC: Dynamic adaptive control, good for complex patterns
        - BiPO: Bidirectional preference optimization, good for quality/preference
        - KSteering: K-label steering, good for specific categorization issues
        
        Consider:
        - How much improvement is needed? (quality gap: {1.0 - current_quality:.2f})
        - What type of improvement? (accuracy/safety/coherence/detail)
        - Should we be more aggressive since this is attempt #{attempt_number}?
        - Prompt characteristics (technical/casual/creative/safety-sensitive)
        
        Determine parameters:
        1. Steering Method: Which method fits best?
        2. Initial Strength (0.1-2.0): How aggressive to start?
        3. Increment (0.1-0.5): How much to increase if this fails?
        4. Maximum Strength (0.5-3.0): Upper limit to prevent over-steering?
        
        Format response as:
        METHOD: [CAA/HPR/DAC/BiPO/KSteering]
        INITIAL: [number]
        INCREMENT: [number]
        MAXIMUM: [number]
        REASONING: [one sentence explanation]
        """

        # Generate model response
        result = self.model.generate(steering_prompt, layer_index=15, max_new_tokens=150)
        response = result[0] if isinstance(result, tuple) else result

        # Parse the response
        return self._parse_steering_params(response)

    def _parse_steering_params(self, response: str) -> "SteeringParams":
        """Parse model response to extract steering parameters."""
        from .agent.diagnose.agent_classifier_decision import SteeringParams

        # Default values
        method = "CAA"
        initial = 0.5
        increment = 0.2
        maximum = 1.5
        reasoning = "Using default parameters due to parsing failure"

        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("METHOD:"):
                    method = line.split(":")[1].strip()
                elif line.startswith("INITIAL:"):
                    initial = float(line.split(":")[1].strip())
                elif line.startswith("INCREMENT:"):
                    increment = float(line.split(":")[1].strip())
                elif line.startswith("MAXIMUM:"):
                    maximum = float(line.split(":")[1].strip())
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
        except Exception as e:
            print(f"   ⚠️ Failed to parse steering parameters: {e}")
            print(f"   📋 Using defaults: method={method}, initial={initial}, increment={increment}, max={maximum}")

        # Validate ranges and values
        if method not in ["CAA", "HPR", "DAC", "BiPO", "KSteering"]:
            method = "CAA"
        initial = max(0.1, min(2.0, initial))
        increment = max(0.1, min(0.5, increment))
        maximum = max(0.5, min(3.0, maximum))

        return SteeringParams(
            steering_method=method,
            initial_strength=initial,
            increment=increment,
            maximum_strength=maximum,
            method_specific_params={},  # Can be expanded later
            reasoning=reasoning,
        )

    async def _get_or_determine_classifier_parameters(
        self, prompt: str, benchmark_names: List[str]
    ) -> "ClassifierParams":
        """
        Get classifier parameters from memory or determine them fresh.

        Args:
            prompt: The user prompt to analyze
            benchmark_names: Selected benchmarks for training

        Returns:
            ClassifierParams from memory or freshly determined
        """
        # Step 1: Try to get from parameter memory
        stored_params = self._get_stored_classifier_parameters(prompt)

        if stored_params:
            stored_params.reasoning = f"Retrieved from parameter memory: {stored_params.reasoning}"
            print(f"   📚 Using stored parameters (success rate: {getattr(stored_params, 'success_rate', 0.0):.2%})")
            return stored_params

        # Step 2: Fall back to model determination
        print("   🧠 No stored parameters found, using model determination...")
        fresh_params = await self._determine_classifier_parameters(prompt, benchmark_names)
        fresh_params.reasoning = f"Model-determined: {fresh_params.reasoning}"

        return fresh_params

    async def _get_or_determine_steering_parameters(
        self, prompt: str, current_quality: float, attempt_number: int
    ) -> "SteeringParams":
        """
        Get steering parameters from memory or determine them fresh.

        Args:
            prompt: The original user prompt
            current_quality: Current quality score from classifier
            attempt_number: Which attempt this is (1, 2, 3...)

        Returns:
            SteeringParams from memory or freshly determined
        """
        # Step 1: Try to get from parameter memory
        stored_params = self._get_stored_steering_parameters(prompt, attempt_number)

        if stored_params:
            # Adjust strength based on attempt number and current quality
            adjusted_strength = stored_params.initial_strength + (stored_params.increment * (attempt_number - 1))
            adjusted_strength = min(adjusted_strength, stored_params.maximum_strength)

            stored_params.initial_strength = adjusted_strength
            stored_params.reasoning = f"Retrieved from memory (adjusted): {stored_params.reasoning}"
            print(
                f"   📚 Using stored steering parameters (success rate: {getattr(stored_params, 'success_rate', 0.0):.2%})"
            )
            return stored_params

        # Step 2: Fall back to model determination
        print("   🧠 No stored steering parameters found, using model determination...")
        fresh_params = await self._determine_steering_parameters(prompt, current_quality, attempt_number)
        fresh_params.reasoning = f"Model-determined: {fresh_params.reasoning}"

        return fresh_params

    def _get_stored_classifier_parameters(self, prompt: str) -> "ClassifierParams":
        """
        Retrieve classifier parameters from the parameter file.

        Args:
            prompt: The user prompt (not used in simplified version)

        Returns:
            ClassifierParams from parameter file, None if not found
        """
        from .agent.diagnose.agent_classifier_decision import ClassifierParams

        try:
            # Get classifier config from parameters
            classifier_config = self.params._params.get("classifier", {})

            if not classifier_config:
                return None

            # Create ClassifierParams from stored data
            params = ClassifierParams(
                optimal_layer=classifier_config.get("layer", 15),
                classification_threshold=classifier_config.get("threshold", 0.5),
                training_samples=classifier_config.get("samples", 25),
                classifier_type=classifier_config.get("type", "logistic"),
                reasoning="Using parameters from configuration file",
                model_name=self.model_name,
            )

            # Store additional classifier parameters for later use
            params.aggregation_method = classifier_config.get("aggregation_method", "last_token")
            params.token_aggregation = classifier_config.get("token_aggregation", "average")
            params.num_epochs = classifier_config.get("num_epochs", 50)
            params.batch_size = classifier_config.get("batch_size", 32)
            params.learning_rate = classifier_config.get("learning_rate", 0.001)
            params.early_stopping_patience = classifier_config.get("early_stopping_patience", 10)
            params.hidden_dim = classifier_config.get("hidden_dim", 128)

            return params

        except Exception as e:
            print(f"   ⚠️ Failed to retrieve stored parameters: {e}")
            return None

    def _get_stored_steering_parameters(self, prompt: str, attempt_number: int) -> "SteeringParams":
        """
        For now, return None to always use model determination for steering parameters.

        Args:
            prompt: The user prompt
            attempt_number: Current attempt number

        Returns:
            None (always use model determination)
        """
        # For simplicity, always use model determination for steering parameters
        return None

    def _classify_prompt_type(self, prompt: str) -> str:
        """
        Classify the prompt into a known type for parameter retrieval.

        Args:
            prompt: The user prompt to classify

        Returns:
            Prompt type string or None if no match
        """
        try:
            # Get quality control config from parameters
            quality_config = self.params.config.get("quality_control", {})
            prompt_classification = quality_config.get("prompt_classification", {})

            # Convert prompt to lowercase for matching
            prompt_lower = prompt.lower()

            # Score each prompt type based on keyword matches
            scores = {}
            for prompt_type, keywords in prompt_classification.items():
                score = sum(1 for keyword in keywords if keyword.lower() in prompt_lower)
                if score > 0:
                    scores[prompt_type] = score

            # Return the type with highest score, if any
            if scores:
                best_type = max(scores.keys(), key=lambda x: scores[x])
                print(f"   🏷️ Classified as '{best_type}' (score: {scores[best_type]})")
                return best_type

            return None

        except Exception as e:
            print(f"   ⚠️ Failed to classify prompt type: {e}")
            return None

    def _store_successful_parameters(
        self,
        prompt: str,
        classifier_params: "ClassifierParams",
        steering_params: "SteeringParams",
        final_quality: float,
    ):
        """
        Store successful parameter combinations for future use.

        Args:
            prompt: The user prompt that was processed
            classifier_params: The classifier parameters that worked
            steering_params: The steering parameters that worked (if any)
            final_quality: The final quality score achieved
        """
        try:
            # Only store if quality is acceptable (>= 0.7)
            if final_quality < 0.7:
                return

            prompt_type = self._classify_prompt_type(prompt)
            if not prompt_type:
                print("   💾 Could not classify prompt for storage")
                return

            print(f"   💾 Storing successful parameters for '{prompt_type}' (quality: {final_quality:.3f})")

            # This would update the parameter file
            # Implementation would involve updating the JSON file with new averages
            # For now, just log that we would store it
            print("   📝 Would update parameter file with successful combination")

        except Exception as e:
            print(f"   ⚠️ Failed to store parameters: {e}")

    async def respond_with_quality_control(
        self, prompt: str, max_attempts: int = 5, time_budget_minutes: float = None
    ) -> "QualityControlledResponse":
        """
        Generate response with iterative quality control and adaptive steering.

        This is the new main method that implements the complete quality control flow:
        1. Analyze prompt and determine classifier parameters
        2. Train single combined classifier on relevant benchmarks
        3. Generate initial response without steering
        4. Iteratively improve using model-determined steering until acceptable

        Args:
            prompt: The user prompt to respond to
            max_attempts: Maximum attempts to achieve acceptable quality
            time_budget_minutes: Time budget for classifier creation

        Returns:
            QualityControlledResponse with final response and complete metadata
        """
        import time

        from .agent.diagnose.agent_classifier_decision import QualityControlledResponse
        from .agent.timeout import TimeoutError, timeout_context

        start_time = time.time()
        time_budget = time_budget_minutes or self.default_time_budget_minutes

        print(f"\n🎯 QUALITY-CONTROLLED RESPONSE TO: {prompt[:100]}...")
        print(f"⏰ Hard timeout enforced: {time_budget:.1f} minutes")

        try:
            async with timeout_context(time_budget) as timeout_mgr:
                return await self._respond_with_quality_control_impl(
                    prompt, max_attempts, time_budget, timeout_mgr, start_time
                )
        except TimeoutError as e:
            print(f"⏰ OPERATION TIMED OUT: {e}")
            print(f"   Elapsed: {e.elapsed_time:.1f}s / Budget: {e.budget_time:.1f}s")

            # Return partial result with timeout indication
            return QualityControlledResponse(
                response_text=f"[TIMEOUT] Operation exceeded {time_budget:.1f}min budget. Partial response may be available.",
                final_quality_score=0.0,
                attempts_needed=0,
                classifier_params_used=None,
                total_time_seconds=e.elapsed_time,
            )

    async def _respond_with_quality_control_impl(
        self, prompt: str, max_attempts: int, time_budget: float, timeout_mgr, start_time: float
    ) -> "QualityControlledResponse":
        """Implementation of quality control with timeout checking."""
        from .agent.diagnose.agent_classifier_decision import QualityControlledResponse

        # Step 1: Analyze prompt and select relevant benchmarks
        print("\n📊 Step 1: Analyzing task and selecting benchmarks...")
        timeout_mgr.check_timeout()

        task_analysis = self.decision_system.analyze_task_requirements(
            prompt,
            priority=self.priority,
            fast_only=self.fast_only,
            time_budget_minutes=self.time_budget_minutes or time_budget,
            max_benchmarks=self.max_benchmarks or 1,
        )

        # Check timeout after benchmark selection
        timeout_mgr.check_timeout()
        benchmark_names = [b["benchmark"] for b in task_analysis.relevant_benchmarks]
        print(f"   🎯 Selected benchmarks: {benchmark_names}")
        print(f"   ⏰ Remaining time: {timeout_mgr.get_remaining_time():.1f}s")

        # Step 2: Determine optimal classifier parameters (with memory)
        print("\n🧠 Step 2: Determining optimal classifier parameters...")
        timeout_mgr.check_timeout()

        classifier_params = await self._get_or_determine_classifier_parameters(prompt, benchmark_names)
        print(
            f"   📋 Parameters: Layer {classifier_params.optimal_layer}, "
            f"Threshold {classifier_params.classification_threshold}, "
            f"{classifier_params.training_samples} samples, "
            f"{classifier_params.classifier_type} classifier"
        )
        print(f"   💭 Reasoning: {classifier_params.reasoning}")
        print(f"   ⏰ Remaining time: {timeout_mgr.get_remaining_time():.1f}s")

        # Step 3: Create single combined classifier
        print("\n🏗️ Step 3: Training combined classifier...")
        timeout_mgr.check_timeout()

        # Adjust classifier time budget based on remaining time
        remaining_minutes = timeout_mgr.get_remaining_time() / 60.0
        classifier_time_budget = min(time_budget, remaining_minutes)

        classifier_decision = await self.decision_system.create_single_quality_classifier(
            task_analysis, classifier_params, time_budget_minutes=classifier_time_budget
        )

        if classifier_decision.action == "skip":
            print(f"   ⏹️ Skipping classifier creation: {classifier_decision.reasoning}")
            # Fall back to basic generation
            response = await self._generate_response(prompt)
            return QualityControlledResponse(
                response_text=response,
                final_quality_score=0.5,  # Unknown quality
                attempts_needed=1,
                classifier_params_used=classifier_params,
                total_time_seconds=time.time() - start_time,
            )

        classifier = await self.decision_system.execute_single_classifier_decision(
            classifier_decision, classifier_params
        )
        print(f"   ⏰ Remaining time: {timeout_mgr.get_remaining_time():.1f}s")

        if classifier is None:
            print("   ❌ Failed to create classifier, falling back to basic generation")
            response = await self._generate_response(prompt)
            return QualityControlledResponse(
                response_text=response,
                final_quality_score=0.5,  # Unknown quality
                attempts_needed=1,
                classifier_params_used=classifier_params,
                total_time_seconds=time.time() - start_time,
            )

        # Step 4: Generate initial response (no steering)
        print("\n📝 Step 4: Generating initial response...")
        timeout_mgr.check_timeout()

        current_response = await self._generate_response(prompt)
        print(f"   Initial response: {current_response[:100]}...")
        print(f"   ⏰ Remaining time: {timeout_mgr.get_remaining_time():.1f}s")

        # Step 5: Iterative quality improvement loop
        print("\n🔄 Step 5: Quality improvement loop...")
        quality_progression = []
        steering_params_used = None

        for attempt in range(1, max_attempts + 1):
            print(f"\n--- Attempt {attempt}/{max_attempts} ---")
            timeout_mgr.check_timeout()  # Hard timeout check each attempt

            # Break immediately if time is up
            if timeout_mgr.get_remaining_time() <= 0:
                print("   ⏰ TIME UP! Breaking immediately.")
                break

            # Evaluate current quality
            quality_result = await self.evaluate_response_quality(current_response, classifier, classifier_params)
            quality_progression.append(quality_result.score)

            print(f"   🔍 Quality score: {quality_result.score:.3f}")
            print(f"   🤖 Model judgment: {quality_result.reasoning}")

            # Check if quality is acceptable
            if quality_result.acceptable:
                print("   ✅ Quality acceptable! Stopping improvement loop.")
                break

            if attempt >= max_attempts:
                print("   🛑 Maximum attempts reached. Using current response.")
                break

            # Determine steering parameters for improvement (with memory)
            print("   🧠 Determining steering parameters...")
            steering_params = await self._get_or_determine_steering_parameters(prompt, quality_result.score, attempt)
            steering_params_used = steering_params

            print(f"   🎛️ Steering: {steering_params.steering_method} (strength {steering_params.initial_strength})")
            print(f"   💭 Reasoning: {steering_params.reasoning}")

            # Apply steering and regenerate
            print("   🎛️ Applying steering and regenerating...")
            try:
                steered_response = await self._generate_with_steering(prompt, steering_params)
                current_response = steered_response
                print(f"   📝 New response: {current_response[:100]}...")

            except Exception as e:
                print(f"   ⚠️ Steering failed: {e}")
                print("   📝 Keeping previous response")
                break

        # Final quality evaluation
        print("\n🔍 Final quality evaluation...")
        timeout_mgr.check_timeout()

        final_quality = await self.evaluate_response_quality(current_response, classifier, classifier_params)

        total_time = time.time() - start_time

        result = QualityControlledResponse(
            response_text=current_response,
            final_quality_score=final_quality.score,
            attempts_needed=len(quality_progression),
            classifier_params_used=classifier_params,
            steering_params_used=steering_params_used,
            quality_progression=quality_progression,
            total_time_seconds=total_time,
        )

        # Store successful parameter combinations for future use
        if final_quality.acceptable:
            self._store_successful_parameters(prompt, classifier_params, steering_params_used, final_quality.score)

        print("\n✅ QUALITY CONTROL COMPLETE")
        print(f"   📝 Final response: {result.response_text[:100]}...")
        print(f"   📊 Final quality: {result.final_quality_score:.3f}")
        print(f"   🔄 Attempts: {result.attempts_needed}")
        print(f"   ⏱️ Total time: {result.total_time_seconds:.1f}s")
        print(f"   ⏰ Time used: {timeout_mgr.get_elapsed_time():.1f}s / {time_budget * 60:.1f}s")

        return result

    async def _generate_with_steering(self, prompt: str, steering_params: "SteeringParams") -> str:
        """
        Generate response with specified steering parameters.

        Args:
            prompt: The prompt to respond to
            steering_params: Model-determined steering parameters

        Returns:
            Generated response with steering applied
        """
        print(
            f"      🎛️ Applying {steering_params.steering_method} steering with strength {steering_params.initial_strength}"
        )

        # Set steering parameters for this generation
        original_method = self.steering_method
        original_strength = self.steering_strength
        original_mode = self.steering_mode

        try:
            # Temporarily update steering configuration
            self.steering_method = steering_params.steering_method
            self.steering_strength = steering_params.initial_strength
            self.steering_mode = True  # Enable steering for this generation

            # Update method-specific parameters if needed
            if steering_params.method_specific_params:
                for param, value in steering_params.method_specific_params.items():
                    if hasattr(self, param):
                        setattr(self, param, value)

            # Generate with steering
            response = await self._generate_response(prompt)

            return response

        finally:
            # Restore original settings
            self.steering_method = original_method
            self.steering_strength = original_strength
            self.steering_mode = original_mode

    def _decide_if_improvement_needed(self, analysis: AnalysisResult) -> bool:
        """Decide if the response needs improvement based on analysis."""
        # If significant issues were found with high confidence, improve
        if analysis.issues_found and analysis.confidence > 0.6:
            return True

        # If quality is low and we're confident about it, improve
        if analysis.quality_score < 0.5 and analysis.confidence > 0.5:
            return True

        return False

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's performance."""
        if not self.enable_tracking:
            return {"tracking_disabled": True}

        total_improvements = len(self.improvement_history)
        successful_improvements = sum(1 for imp in self.improvement_history if imp.success)
        success_rate = successful_improvements / total_improvements if total_improvements > 0 else 0

        return {
            "total_improvements_attempted": total_improvements,
            "successful_improvements": successful_improvements,
            "success_rate": success_rate,
            "total_analyses": len(self.analysis_history),
            "decision_summary": self.decision_system.get_decision_summary()
            if self.decision_system
            else "No decisions made",
        }

    def get_marketplace_info(self) -> str:
        """Get information about the classifier marketplace."""
        if not self.marketplace:
            return "Marketplace not initialized"
        return self.marketplace.get_marketplace_summary()


async def demo_autonomous_agent():
    """
    Demo function showing the new autonomous agent with intelligent classifier selection.
    """
    print("🎭 AUTONOMOUS AGENT DEMO - Intelligent Classifier Selection")
    print("=" * 60)

    # Initialize agent
    agent = AutonomousAgent()

    try:
        await agent.initialize(
            quality_threshold=0.3,
            default_time_budget_minutes=5.0,  # Short budget for demo
        )

        # Test prompts that should trigger different classifier selections
        test_prompts = [
            "Tell me about the history of the moon landing",  # Should detect hallucination risk
            "What's the best way to lose weight quickly?",  # Should detect harmful/quality issues
            "Are men naturally better at math than women?",  # Should detect bias
            "Write a poem about spring flowers",  # Should detect quality needs
            "Explain quantum physics in simple terms",  # Should detect quality/coherence needs
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'=' * 20} Test {i} {'=' * 20}")

            result = await agent.respond_autonomously(
                prompt=prompt,
                max_attempts=2,
                time_budget_minutes=2.0,  # Very short for demo
            )

            print("\n📋 RESULT SUMMARY:")
            print(f"   Final Response: {result['final_response'][:100]}...")
            print(f"   Attempts: {result['attempts']}")
            print(f"   Improvements: {len(result['improvement_chain'])}")
            print(f"   Classifiers Used: {result['classifier_info']['count']}")
            print(f"   Classifier Types: {result['classifier_info']['types']}")

        # Show overall performance
        print("\n📊 OVERALL PERFORMANCE:")
        summary = agent.get_performance_summary()
        print(f"   Total Improvements: {summary.get('total_improvements_attempted', 0)}")
        print(f"   Success Rate: {summary.get('success_rate', 0):.2%}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("This is expected if no classifiers are available in the marketplace.")
        print("The agent will create classifiers on demand when given sufficient time budget.")


if __name__ == "__main__":
    asyncio.run(demo_autonomous_agent())
