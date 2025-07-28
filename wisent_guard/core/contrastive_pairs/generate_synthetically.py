import json
import random
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from wisent_guard.core.model import Model

from ..response import NegativeResponse, PositiveResponse
from .contrastive_database import ContrastivePairDatabase
from .contrastive_pair import ContrastivePair
from .contrastive_pair_set import ContrastivePairSet
from .contrastive_generation_conf import CONTRASTIVE_GEN, SCENARIO_GEN, SCENARIO_PARSE
from .quality_check import quality_check_synthetic_pairs


class SyntheticContrastivePairGenerator:
    """Generate contrastive pairs synthetically from natural language trait descriptions."""

    def __init__(
        self,
        model: Model,
        similarity_threshold: float = 0.8,
        db_path: Optional[str] = None,
        db_similarity_threshold: float = 0.95,
    ):
        """
        Initialize the synthetic pair generator.

        Args:
            model: The language model to use for generation
            similarity_threshold: Threshold for deduplication of scenarios/pairs (0-1, higher = more strict)
            db_path: Optional path to the contrastive pair database. If None, caching is disabled.
            db_similarity_threshold: Threshold for retrieving a set from the database (0-1, higher = more strict).
        """
        self.model: Model = model
        self.similarity_threshold: float = similarity_threshold
        self.db_similarity_threshold: float = db_similarity_threshold

        self.similarity_model: Optional[SentenceTransformer] = (
            self._initialize_similarity_model()
        )
        self.database: Optional[ContrastivePairDatabase] = self._initialize_database(
            db_path
        )

    def _initialize_similarity_model(self) -> Optional[SentenceTransformer]:
        """Loads the SentenceTransformer model for similarity calculations."""
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Similarity model loaded successfully.")
            return model
        except Exception as e:
            print(
                f"⚠️ WARNING: Could not load similarity model ('all-MiniLM-L6-v2'). VDB caching and diversity selection will be disabled. Error: {e}"
            )
            return None

    def _initialize_database(
        self, db_path: Optional[str]
    ) -> Optional[ContrastivePairDatabase]:
        """Initializes the contrastive pair database if a path is provided.
        Args:
            db_path: The path to the database directory. If None, caching is disabled.
        Returns:
            An optional ContrastivePairDatabase instance
        """
        if not db_path:
            print("ℹ️ No database path provided. Contrastive pair caching is disabled.")
            return None

        try:
            database = ContrastivePairDatabase(db_path)
            print(
                f"🗂️  Initialized contrastive pair database at: {Path(db_path).resolve()}"
            )
            return database
        except (IOError, OSError, json.JSONDecodeError, RuntimeError) as e:
            print(
                f"⚠️ WARNING: Failed to initialize contrastive pair database at '{db_path}'. Caching is disabled. Error: {e}"
            )
            return None

    def generate_scenarios(
        self, trait_description: str, num_scenarios: int
    ) -> list[str]:
        """
        Generate diverse scenarios where the trait would be relevant.

        Args:
            trait_description: Natural language description of desired trait
            num_scenarios: Number of scenarios to generate

        Returns:
            list of scenario descriptions
        """
        print(f"🎯 DEBUG: Generating scenarios for trait: '{trait_description}'")
        print(f"🎯 DEBUG: Target number of scenarios: {num_scenarios}")

        target_scenarios: int = num_scenarios
        all_scenarios: list[str] = []

        print(
            f"🎯 DEBUG: Will generate {target_scenarios} total scenarios to select {num_scenarios} best ones"
        )

        num_prompts_per_template: int = target_scenarios // len(
            SCENARIO_GEN.PROMPT_TEMPLATES
        )

        for i, template in enumerate(SCENARIO_GEN.PROMPT_TEMPLATES):
            prompt: str = template.format(num_prompts=num_prompts_per_template)
            print(
                f"🎯 DEBUG: Using prompt template {i+1}/{len(SCENARIO_GEN.PROMPT_TEMPLATES)}"
            )
            print(f"🎯 DEBUG: Template: {prompt[:100]}...")
            try:
                response: str
                response, _ = self.model.generate(prompt, **SCENARIO_GEN.CONFIG)

                print(f"🎯 DEBUG: Generated response length: {len(response)} chars")
                print(f"🎯 DEBUG: Response preview: {response[:200]}...")

                # Parse scenarios from response
                scenarios: list[str] = self._parse_scenarios_from_response(response)
                print(f"🎯 DEBUG: Parsed {len(scenarios)} scenarios from this template")
                for j, scenario in enumerate(scenarios):
                    print(f"🎯 DEBUG:   Scenario {j+1}: {scenario[:100]}...")
                all_scenarios.extend(scenarios)

            except Exception as e:
                print(f"🎯 DEBUG: Error generating scenarios with template: {e}")
                continue

        print(f"🎯 DEBUG: Total scenarios before deduplication: {len(all_scenarios)}")

        # Deduplicate and select most diverse scenarios
        unique_scenarios: list[str] = self._deduplicate_scenarios(all_scenarios)
        print(
            f"🎯 DEBUG: Unique scenarios after deduplication: {len(unique_scenarios)}"
        )

        # Select the best diverse scenarios
        selected_scenarios: list[str] = self._select_diverse_scenarios(
            unique_scenarios, num_scenarios
        )
        print(f"🎯 DEBUG: Final selected scenarios: {len(selected_scenarios)}")

        for i, scenario in enumerate(selected_scenarios):
            print(f"🎯 DEBUG: Final scenario {i+1}: {scenario}")

        return selected_scenarios

    def _parse_scenarios_from_response(self, response: str) -> list[str]:
        """Parse individual scenarios from model response using regex and filters.
        Args:
            response: The raw response text from the model
        Returns:
            A list of parsed scenario strings
        """
        scenarios: list[str] = []

        # Regex to remove common list prefixes (e.g., "1.", "-", "* ") and markdown
        prefix_re: re.Pattern = re.compile(r"^\s*(?:\d+\.|\-|\*|•|[a-e]\))\s*")
        markdown_re: re.Pattern = re.compile(r"(\*\*|\*)")

        lines: list[str] = response.split("\n")

        for line in lines:
            # 1. Clean the line
            cleaned: str = markdown_re.sub("", prefix_re.sub("", line.strip()))

            # 2. Basic filtering
            if not cleaned or not (
                SCENARIO_PARSE.MIN_SCENARIO_LENGTH
                < len(cleaned)
                < SCENARIO_PARSE.MAX_SCENARIO_LENGTH
            ):
                continue

            cleaned_lower: str = cleaned.lower()

            # 3. Filter based on content
            if any(
                phrase in cleaned_lower for phrase in SCENARIO_PARSE.SKIP_PHRASES
            ) or any(
                phrase in cleaned_lower for phrase in SCENARIO_PARSE.REFUSAL_PHRASES
            ):
                continue

            # 4. Validate based on structure (question or imperative)
            word_count: int = len(cleaned.split())
            is_question: bool = (
                "?" in cleaned and word_count <= SCENARIO_PARSE.MAX_QUESTION_WORDS
            )
            is_imperative: bool = (
                any(kw in cleaned_lower for kw in SCENARIO_PARSE.IMPERATIVE_KEYWORDS)
                and word_count <= SCENARIO_PARSE.MAX_IMPERATIVE_WORDS
            )

            if is_question or is_imperative:
                scenarios.append(cleaned)

        return scenarios[: SCENARIO_PARSE.CANDIDATE_LIMIT]

    def _deduplicate_scenarios(self, scenarios: list[str]) -> list[str]:
        """Remove duplicate or very similar scenarios.
        Args:
            scenarios: The list of scenario strings to deduplicate
        Returns:
            A list of unique scenario strings
        """
        if not self.similarity_model or len(scenarios) <= 1:
            # Fallback to simple text-based deduplication
            return list(set(scenarios))

        unique_scenarios: list[str] = []

        for scenario in scenarios:
            is_duplicate: bool = False

            if unique_scenarios:
                # Check similarity with existing scenarios
                scenario_embedding: np.ndarray = self.similarity_model.encode(
                    [scenario]
                )
                existing_embeddings: np.ndarray = self.similarity_model.encode(
                    unique_scenarios
                )

                # Calculate cosine similarities
                similarities: np.ndarray = np.dot(
                    scenario_embedding, existing_embeddings.T
                )[0]

                if np.max(similarities) > self.similarity_threshold:
                    is_duplicate = True

            if not is_duplicate:
                unique_scenarios.append(scenario)

        return unique_scenarios

    def _select_diverse_scenarios(
        self, scenarios: list[str], target_count: int
    ) -> list[str]:
        """Select the most diverse scenarios up to target count.
        Args:
            scenarios: The list of scenario strings to select from
            target_count: The number of diverse scenarios to select
        Returns:
            A list of diverse scenario strings
        """
        if len(scenarios) <= target_count:
            return scenarios

        if not self.similarity_model:
            # Random selection fallback
            return random.sample(scenarios, target_count)

        # Use embeddings to select diverse scenarios
        embeddings: np.ndarray = self.similarity_model.encode(scenarios)

        selected_indices: list[int] = [0]  # Start with first scenario

        for _ in range(target_count - 1):
            remaining_indices: list[int] = [
                i for i in range(len(scenarios)) if i not in selected_indices
            ]

            if not remaining_indices:
                break

            # Find scenario most different from already selected ones
            max_min_distance: float = -1.0
            best_idx: int = remaining_indices[0]

            for idx in remaining_indices:
                # Calculate minimum distance to any selected scenario
                distances: list[float] = []
                for selected_idx in selected_indices:
                    distance: float = 1 - np.dot(
                        embeddings[idx], embeddings[selected_idx]
                    )
                    distances.append(distance)

                min_distance: float = min(distances)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx

            selected_indices.append(best_idx)

        return [scenarios[i] for i in selected_indices]

    def _get_pair_embedding(self, pair: ContrastivePair) -> np.ndarray:
        """Computes a single embedding for a contrastive pair.
        Args:
            pair: The ContrastivePair object to compute embedding for
        Returns:
            A numpy array representing the embedding of the contrastive pair
        """
        if not self.similarity_model:
            return np.array([])

        # Embed all components of the pair
        prompt_emb: np.ndarray = self.similarity_model.encode([pair.prompt])[0]
        pos_emb: np.ndarray = self.similarity_model.encode(
            [pair.positive_response.text]
        )[0]
        neg_emb: np.ndarray = self.similarity_model.encode(
            [pair.negative_response.text]
        )[0]

        # Concatenate and normalize to get a single representative vector
        combined_emb: np.ndarray = np.concatenate([prompt_emb, pos_emb, neg_emb])
        norm = np.linalg.norm(combined_emb)
        return combined_emb / norm if norm != 0 else combined_emb

    def _select_diverse_pairs(
        self, pairs: list[ContrastivePair], target_count: int
    ) -> list[ContrastivePair]:
        """Selects a diverse subset of contrastive pairs using their embeddings.
        Args:
            pairs: The list of contrastive pairs to select from
            target_count: The number of diverse pairs to select
        Returns:
            A list of diverse contrastive pairs
        """
        if len(pairs) <= target_count:
            return pairs

        if not self.similarity_model:
            print("⚠️ DEBUG: No similarity model found. Selecting pairs randomly.")
            return random.sample(pairs, target_count)

        print(
            f"🔎 Selecting {target_count} diverse pairs from {len(pairs)} candidates..."
        )

        embeddings: np.ndarray = np.array([self._get_pair_embedding(p) for p in pairs])

        selected_indices: list[int] = [0]
        for _ in range(target_count - 1):
            remaining_indices: list[int] = [
                i for i in range(len(pairs)) if i not in selected_indices
            ]
            if not remaining_indices:
                break

            # Use vectorized operations for faster distance calculation
            selected_embeddings: np.ndarray = embeddings[selected_indices]
            remaining_embeddings: np.ndarray = embeddings[remaining_indices]

            # Calculate cosine similarity, then convert to distance
            dist_matrix: np.ndarray = 1 - np.dot(
                remaining_embeddings, selected_embeddings.T
            )

            # Find the minimum distance from each remaining point to any selected point
            min_distances: np.ndarray = np.min(dist_matrix, axis=1)

            # Find the point that has the maximum minimum-distance
            max_dist_idx: int = np.argmax(min_distances)

            selected_indices.append(remaining_indices[max_dist_idx])

        return [pairs[i] for i in selected_indices]

    def _generate_response(self, prompt: str, config: dict[str, Any]) -> str:
        """Generates a response from the model using the provided configuration.
        Args:
            prompt: The input prompt for the model
            config: Configuration parameters for the model generation
        Returns
            response: The generated response text
        """
        response: str
        response, _ = self.model.generate(prompt, **config)
        return response.strip()

    def generate_contrastive_pair(
        self, scenario: str, trait_description: str
    ) -> ContrastivePair:
        """
        Generate a contrastive pair for a specific scenario.

        Args:
            scenario: The scenario to generate responses for
            trait_description: The trait description for context

        Returns:
            ContrastivePair object
        """
        print(
            f"🔄 DEBUG: Generating contrastive pair for scenario: {scenario[:100]}..."
        )
        print(f"🔄 DEBUG: Trait: {trait_description}")

        # Generate positive response (demonstrates the trait)
        positive_prompt: str = CONTRASTIVE_GEN.POSITIVE_PROMPT_TEMPLATE.format(
            scenario=scenario, trait_description=trait_description
        )
        print(f"🔄 DEBUG: Positive prompt: {positive_prompt}")
        positive_response: str = self._generate_response(
            positive_prompt, CONTRASTIVE_GEN.CONFIG
        )
        print(f"🔄 DEBUG: Positive response: {positive_response[:100]}...")

        # Generate negative response (opposite of trait)
        negative_prompt: str = CONTRASTIVE_GEN.NEGATIVE_PROMPT_TEMPLATE.format(
            scenario=scenario, trait_description=trait_description
        )
        print(f"🔄 DEBUG: Negative prompt: {negative_prompt}")
        negative_response: str = self._generate_response(
            negative_prompt, CONTRASTIVE_GEN.CONFIG
        )
        print(f"🔄 DEBUG: Negative response: {negative_response[:100]}...")

        # Create contrastive pair - always use the question directly
        prompt: str = scenario.strip()
        print(f"🔄 DEBUG: Using question as direct prompt: {prompt}")

        pair: ContrastivePair = ContrastivePair(
            prompt=prompt,
            positive_response=PositiveResponse(text=positive_response),
            negative_response=NegativeResponse(text=negative_response),
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
        pair_overgeneration_factor: float = 1.5,
        force_regenerate: bool = False,
    ) -> ContrastivePairSet:
        """
        Generate a complete contrastive pair set from a trait description.
        Checks a vector database first to avoid re-generation if available.

        Args:
            trait_description: Natural language description of desired trait
            num_pairs: Number of contrastive pairs to generate
            pair_overgeneration_factor: Factor to overgenerate pairs for diversity selection
            force_regenerate: If True, bypasses the cache and generates a new set.

        Returns:
            ContrastivePairSet with generated pairs
        """
        use_database: bool = self.database is not None and not force_regenerate

        if not self.similarity_model and use_database:
            print("⚠️ WARNING: Similarity model not loaded. Cannot use database cache.")
            use_database = False

        if use_database:
            print(f"🔍 Checking database for similar trait: '{trait_description}'")
            trait_embedding: np.ndarray = self.similarity_model.encode(
                [trait_description]
            )

            found_set_info: Optional[dict] = self.database.search_for_trait(
                trait_embedding, self.db_similarity_threshold
            )

            if found_set_info is not None:
                cached_pair_count: int = found_set_info.get("pair_count", 0)
                print(
                    f"✅ Found sufficiently similar pair set in cache (ID: {found_set_info['set_id']}) with {cached_pair_count} pairs."
                )

                if num_pairs <= cached_pair_count:
                    print("Sufficient pairs in cache. Retrieving...")
                    return self.database.get_set_by_id(found_set_info["set_id"])
                else:
                    print(
                        f"⚠️ Cache has {cached_pair_count} pairs, but {num_pairs} are requested. Augmenting the set..."
                    )
                    # Load existing set and generate the missing pairs
                    existing_set: ContrastivePairSet = self.database.get_set_by_id(
                        found_set_info["set_id"]
                    )
                    additional_pairs_needed: int = num_pairs - cached_pair_count

                    # We need to generate more scenarios and pairs, avoiding duplicates from the existing set
                    # For simplicity, we'll generate a new batch and combine, then re-filter for diversity.
                    # A more advanced implementation could try to find scenarios dissimilar to existing ones.

                    print(
                        f"🔄 Generating {additional_pairs_needed} additional pairs..."
                    )
                    newly_generated_set: ContrastivePairSet = self._generate_new_pairs(
                        trait_description,
                        additional_pairs_needed,
                        pair_overgeneration_factor,
                    )

                    # Combine, ensure diversity, and update the cache
                    combined_set = self._combine_and_update_set(
                        existing_set,
                        newly_generated_set,
                        num_pairs,
                        trait_description,
                        trait_embedding,
                    )
                    return combined_set

        print(
            f"ℹ️ No suitable pair set found in cache or caching is disabled. Generating a new one for trait: '{trait_description}'"
        )

        # If we are generating and have a database, we need the embedding for caching later.
        trait_embedding: Optional[np.ndarray] = None
        if use_database:
            trait_embedding = self.similarity_model.encode([trait_description])

        # Generate the full set of new pairs
        new_pair_set = self._generate_new_pairs(
            trait_description, num_pairs, pair_overgeneration_factor
        )

        # Add the newly generated set to the database
        if use_database and trait_embedding is not None:
            print(f"💾 Caching new pair set to database...")
            self.database.add_set(new_pair_set, trait_embedding)
            print(f"✅ Cached new set.")

        return new_pair_set

    def _generate_new_pairs(
        self, trait_description: str, num_pairs: int, pair_overgeneration_factor: float
    ) -> ContrastivePairSet:
        """Generate a new set of contrastive pairs.
        Args:
            trait_description: Natural language description of desired trait
            num_pairs: Number of contrastive pairs to generate
            pair_overgeneration_factor: Factor to overgenerate pairs for diversity selection
        Returns:
            ContrastivePairSet with generated pairs
        """
        num_scenarios_to_generate: int = int(num_pairs * pair_overgeneration_factor)
        print(f"📝 Generating {num_scenarios_to_generate} diverse scenarios...")
        scenarios: list[str] = self.generate_scenarios(
            trait_description, num_scenarios_to_generate
        )
        print(f"✅ Generated {len(scenarios)} unique scenarios")

        print("🔄 Generating contrastive pairs...")
        all_pairs: list[ContrastivePair] = []
        for i, scenario in enumerate(scenarios):
            try:
                print(f"   Generating pair {i+1}/{len(scenarios)}: {scenario[:50]}...")
                pair: ContrastivePair = self.generate_contrastive_pair(
                    scenario, trait_description
                )
                all_pairs.append(pair)
            except Exception as e:
                print(f"   ⚠️ Error generating pair for scenario '{scenario[:50]}': {e}")
                continue

        print(f"✅ Successfully generated {len(all_pairs)} raw contrastive pairs")
        diverse_pairs: list[ContrastivePair] = self._select_diverse_pairs(
            all_pairs, num_pairs
        )
        print(f"✅ Selected {len(diverse_pairs)} diverse pairs")

        pair_set: ContrastivePairSet = ContrastivePairSet(
            name=f"synthetic_{trait_description[:30]}",
            task_type="synthetic",
            pairs=diverse_pairs,
        )

        print("🔍 Applying quality check to filter pairs...")
        return quality_check_synthetic_pairs(
            pair_set, trait_description, strict_mode=True
        )

    def _combine_and_update_set(
        self,
        existing_set: ContrastivePairSet,
        new_set: ContrastivePairSet,
        target_pair_count: int,
        trait_embedding: np.ndarray,
    ) -> ContrastivePairSet:
        """Combines an existing set with a new one, ensures diversity, and updates the database.
        Args:
            existing_set: The existing ContrastivePairSet from the database
            new_set: The newly generated ContrastivePairSet
            target_pair_count: The desired number of pairs in the final set
            trait_embedding: The embedding of the trait description for caching purposes
        Returns:
            ContrastivePairSet with combined and deduplicated pairs"""

        # Combine pairs and deduplicate based on prompt
        combined_pairs_map: dict[str, ContrastivePair] = {
            p.prompt: p for p in existing_set.pairs
        }
        combined_pairs_map.update({p.prompt: p for p in new_set.pairs})
        all_unique_pairs: list[ContrastivePair] = list(combined_pairs_map.values())

        print(f"🤝 Combined sets, resulting in {len(all_unique_pairs)} unique pairs.")

        # Select the most diverse subset
        final_diverse_pairs: list[ContrastivePair] = self._select_diverse_pairs(
            all_unique_pairs, target_pair_count
        )

        # Create the final, augmented pair set
        augmented_set = ContrastivePairSet(
            name=existing_set.name,
            task_type=existing_set.task_type,
            pairs=final_diverse_pairs,
        )

        # Add the new, larger set to the database. This creates a new entry.
        print(
            f"💾 Caching augmented pair set with {len(final_diverse_pairs)} pairs to database..."
        )
        self.database.add_set(augmented_set, trait_embedding)
        print(f"✅ Cached augmented set.")

        return augmented_set

    def save_to_json(self, pair_set: ContrastivePairSet, filepath: str) -> None:
        """Save contrastive pair set to JSON file.
        Args:
            pair_set: ContrastivePairSet to save
            filepath: Path to save the JSON file
        Returns:
            None
        """
        data: dict[str, Any] = {
            "name": pair_set.name,
            "task_type": pair_set.task_type,
            "pairs": [],
        }

        for pair in pair_set.pairs:
            pair_data: dict[str, Any] = {
                "prompt": pair.prompt,
                "positive_response": pair.positive_response.text,
                "negative_response": pair.negative_response.text,
            }
            data["pairs"].append(pair_data)

        filepath: Path = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(pair_set.pairs)} pairs to {filepath}")

    def load_from_json(self, filepath: str) -> ContrastivePairSet:
        """Load contrastive pair set from JSON file.
        Args:
            filepath: Path to JSON file

        Returns:
            ContrastivePairSet with loaded pairs
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        pair_set: ContrastivePairSet = ContrastivePairSet(
            name=data.get("name", "loaded_synthetic"),
            task_type=data.get("task_type", "synthetic"),
        )

        for pair_data in data["pairs"]:
            pair = ContrastivePair(
                prompt=pair_data["prompt"],
                positive_response=PositiveResponse(text=pair_data["positive_response"]),
                negative_response=NegativeResponse(text=pair_data["negative_response"]),
            )

            pair_set.pairs.append(pair)

        print(f"📂 Loaded {len(pair_set.pairs)} pairs from {filepath}")

        return pair_set


def generate_synthetic_pairs_cli(
    trait_description: str,
    num_pairs: int = 30,
    output_file: Optional[str] = None,
    model=None,
    force_regenerate: bool = False,
) -> ContrastivePairSet:
    """
    CLI function to generate synthetic contrastive pairs.

    Args:
        trait_description: Natural language description of desired trait
        num_pairs: Number of pairs to generate
        output_file: Optional file to save pairs to
        model: Model instance to use
        force_regenerate: If True, bypasses the cache and generates a new set.

    Returns:
        Generated ContrastivePairSet
    """
    if model is None:
        raise ValueError("Model must be provided")

    generator: SyntheticContrastivePairGenerator = SyntheticContrastivePairGenerator(
        model
    )

    pair_set: ContrastivePairSet = generator.generate_contrastive_pair_set(
        trait_description=trait_description,
        num_pairs=num_pairs,
        name=f"synthetic_{trait_description.replace(' ', '_')[:20]}",
        force_regenerate=force_regenerate,
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
    generator: SyntheticContrastivePairGenerator = SyntheticContrastivePairGenerator(
        model
    )
    return generator.load_from_json(filepath)
