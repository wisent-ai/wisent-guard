"""
Quality check system for synthetically generated contrastive pairs.
Filters out nonsensical, irrelevant, or low-quality pairs before training.
"""

import re
from typing import Any, Dict, List, Tuple

from .contrastive_pair import ContrastivePair
from .contrastive_pair_set import ContrastivePairSet


class ContrastivePairQualityChecker:
    """Quality checker for synthetically generated contrastive pairs."""

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the quality checker.

        Args:
            strict_mode: If True, applies stricter filtering criteria
        """
        self.strict_mode = strict_mode

        # Common signs of poor quality questions
        self.bad_patterns = [
            r"користувач",  # Strange Unicode characters
            r"елек",  # Garbled text
            r"assistant",  # Model talking about itself
            r"here are",  # Meta-commentary
            r"here is",  # Meta-commentary
            r"these are",  # Meta-commentary
            r"list of",  # Meta-commentary
            r"examples of",  # Meta-commentary
            r"questions where",  # Meta-commentary
            r"situations where",  # Meta-commentary
        ]

        # Signs of good questions
        self.good_indicators = [
            r"\?$",  # Ends with question mark
            r"^(what|how|why|when|where|who|which)",  # Question words
            r"explain",  # Request for explanation
            r"tell me",  # Direct request
            r"describe",  # Request for description
            r"can you",  # Polite request
        ]

    def check_question_quality(
        self, question: str, trait_description: str
    ) -> Dict[str, Any]:
        """
        Check the quality of a question.

        Args:
            question: The question text
            trait_description: The trait being tested

        Returns:
            Dictionary with quality assessment
        """
        issues = []
        score = 100  # Start with perfect score

        # Check for bad patterns
        for pattern in self.bad_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                issues.append(f"Contains bad pattern: {pattern}")
                score -= 30

        # Check question length - prefer shorter questions
        if len(question) < 10:
            issues.append("Too short")
            score -= 40
        elif len(question) > 150:
            issues.append("Too long")
            score -= 30
        elif len(question) > 100:
            issues.append("Quite long")
            score -= 10

        # Check if it's actually a question or prompt
        has_good_indicator = any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in self.good_indicators
        )

        if not has_good_indicator:
            issues.append("Doesn't look like a proper question or prompt")
            score -= 25

        # Check for relevance to trait (basic heuristics)
        trait_keywords = trait_description.lower().split()
        question_lower = question.lower()

        # Simple relevance check - question should relate to the trait somehow
        if self.strict_mode:
            # In strict mode, require some connection to the trait
            trait_related = any(keyword in question_lower for keyword in trait_keywords)
            if not trait_related and len(trait_keywords) > 1:
                # Check for semantic relatedness (basic)
                semantic_keywords = self._get_semantic_keywords(trait_description)
                trait_related = any(
                    keyword in question_lower for keyword in semantic_keywords
                )

            if not trait_related:
                issues.append(f"May not be relevant to trait '{trait_description}'")
                score -= 15

        # Check for repetitive content
        words = question.split()
        if len(words) > 5:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio < 0.7:
                issues.append("Too repetitive")
                score -= 20

        return {
            "score": max(0, score),
            "issues": issues,
            "is_quality": score >= 50,
            "question": question,
        }

    def check_response_quality(
        self, response_text: str, trait_description: str, is_positive: bool
    ) -> Dict[str, Any]:
        """
        Check the quality of a response.

        Args:
            response_text: The response text
            trait_description: The trait being tested
            is_positive: Whether this is the positive (good) response

        Returns:
            Dictionary with quality assessment
        """
        issues = []
        score = 100

        # Check for bad patterns
        for pattern in self.bad_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                issues.append(f"Contains bad pattern: {pattern}")
                score -= 30

        # Check response length - prefer shorter responses
        if len(response_text) < 5:
            issues.append("Too short")
            score -= 40
        elif len(response_text) > 200:
            issues.append("Too long")
            score -= 20
        elif len(response_text) > 100:
            issues.append("Quite long")
            score -= 5

        # Check for meta-commentary about being a model
        meta_patterns = [
            r"as an ai",
            r"i am a",
            r"as a language model",
            r"i cannot",
            r"i should not",
            r"show how a model",
            r"demonstrate.*trait",
        ]

        for pattern in meta_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                issues.append("Contains meta-commentary")
                score -= 25
                break

        # Check for actual content vs instructions
        if "[" in response_text and "]" in response_text:
            issues.append("Contains instruction brackets")
            score -= 30

        return {
            "score": max(0, score),
            "issues": issues,
            "is_quality": score >= 50,
            "response": response_text,
        }

    def check_pair_quality(
        self, pair: ContrastivePair, trait_description: str
    ) -> Dict[str, Any]:
        """
        Check the overall quality of a contrastive pair.

        Args:
            pair: The contrastive pair to check
            trait_description: The trait being tested

        Returns:
            Dictionary with comprehensive quality assessment
        """
        # Check question quality
        question_check = self.check_question_quality(pair.prompt, trait_description)

        # Check positive response quality
        pos_check = self.check_response_quality(
            pair.positive_response.text, trait_description, is_positive=True
        )

        # Check negative response quality
        neg_check = self.check_response_quality(
            pair.negative_response.text, trait_description, is_positive=False
        )

        # Calculate overall score
        overall_score = (
            question_check["score"] * 0.4
            + pos_check["score"] * 0.3
            + neg_check["score"] * 0.3
        )

        # Aggregate issues
        all_issues = []
        if question_check["issues"]:
            all_issues.extend(
                [f"Question: {issue}" for issue in question_check["issues"]]
            )
        if pos_check["issues"]:
            all_issues.extend([f"Positive: {issue}" for issue in pos_check["issues"]])
        if neg_check["issues"]:
            all_issues.extend([f"Negative: {issue}" for issue in neg_check["issues"]])

        # Check for contrast between positive and negative
        pos_text = pair.positive_response.text.lower()
        neg_text = pair.negative_response.text.lower()

        # Simple similarity check
        pos_words = set(pos_text.split())
        neg_words = set(neg_text.split())

        if len(pos_words) > 0 and len(neg_words) > 0:
            common_words = pos_words.intersection(neg_words)
            similarity = len(common_words) / max(len(pos_words), len(neg_words))

            if similarity > 0.8:
                all_issues.append("Positive and negative responses are too similar")
                overall_score -= 20

        return {
            "overall_score": max(0, overall_score),
            "question_score": question_check["score"],
            "positive_score": pos_check["score"],
            "negative_score": neg_check["score"],
            "issues": all_issues,
            "is_quality": overall_score >= 60,
            "question_check": question_check,
            "positive_check": pos_check,
            "negative_check": neg_check,
        }

    def filter_pair_set(
        self, pair_set: ContrastivePairSet, trait_description: str
    ) -> ContrastivePairSet:
        """
        Filter a contrastive pair set, keeping only high-quality pairs.

        Args:
            pair_set: The pair set to filter
            trait_description: The trait being tested

        Returns:
            Filtered ContrastivePairSet with only quality pairs
        """
        print(f"🔍 Quality checking {len(pair_set.pairs)} contrastive pairs...")

        quality_pairs = []
        rejected_pairs = []

        for i, pair in enumerate(pair_set.pairs):
            print(f"   Checking pair {i+1}/{len(pair_set.pairs)}...")

            quality_check = self.check_pair_quality(pair, trait_description)

            if quality_check["is_quality"]:
                quality_pairs.append(pair)
                print(f"   ✅ PASS (score: {quality_check['overall_score']:.1f})")
            else:
                rejected_pairs.append((pair, quality_check))
                print(f"   ❌ REJECT (score: {quality_check['overall_score']:.1f})")
                for issue in quality_check["issues"][:2]:  # Show first 2 issues
                    print(f"      Issue: {issue}")

        # Create filtered pair set
        filtered_set = ContrastivePairSet(
            name=f"{pair_set.name}_filtered", task_type=pair_set.task_type
        )
        filtered_set.pairs = quality_pairs

        print(f"✅ Quality check complete:")
        print(f"   • Kept: {len(quality_pairs)} pairs")
        print(f"   • Rejected: {len(rejected_pairs)} pairs")
        if len(pair_set.pairs) > 0:
            print(f"   • Quality rate: {len(quality_pairs)/len(pair_set.pairs)*100:.1f}%")
        else:
            print(f"   • Quality rate: N/A (no pairs to check)")
        # Show examples of rejected pairs
        if rejected_pairs and len(rejected_pairs) <= 3:
            print(f"\n📋 Examples of rejected pairs:")
            for pair, check in rejected_pairs[:3]:
                print(f"   Question: {pair.prompt[:100]}...")
                print(f"   Issues: {', '.join(check['issues'][:2])}")

        return filtered_set

    def _get_semantic_keywords(self, trait_description: str) -> List[str]:
        """Get semantically related keywords for a trait."""
        # Basic semantic expansion
        semantic_map = {
            "accuracy": ["correct", "precise", "accurate", "factual", "true", "exact"],
            "truthfulness": ["honest", "factual", "reliable", "authentic", "genuine"],
            "helpfulness": ["useful", "assistance", "support", "guidance", "aid"],
            "safety": ["secure", "safe", "protection", "harm", "risk"],
            "clarity": ["clear", "understandable", "simple", "explain", "plain"],
            "politeness": ["polite", "respectful", "courteous", "kind", "nice"],
        }

        keywords = []
        for base_word, related in semantic_map.items():
            if base_word in trait_description.lower():
                keywords.extend(related)

        return keywords


def quality_check_synthetic_pairs(
    pair_set: ContrastivePairSet, trait_description: str, strict_mode: bool = True
) -> ContrastivePairSet:
    """
    Convenience function to quality check synthetic contrastive pairs.

    Args:
        pair_set: The pair set to check
        trait_description: The trait being tested
        strict_mode: Whether to apply strict filtering

    Returns:
        Filtered ContrastivePairSet
    """
    checker = ContrastivePairQualityChecker(strict_mode=strict_mode)
    return checker.filter_pair_set(pair_set, trait_description)
