"""
Nonsense detection system to stop lobotomized model responses.

This module detects and prevents various forms of model degradation including:
- Non-existent words (gibberish)
- Highly repetitive content
- Unreasonably long words
- Incoherent responses
"""

import re
import string
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import nltk
from nltk.corpus import words as nltk_words
from nltk.tokenize import word_tokenize, sent_tokenize


class NonsenseDetector:
    """Detects various forms of nonsensical or lobotomized model responses."""
    
    def __init__(self, 
                 max_word_length: int = 20,
                 repetition_threshold: float = 0.7,
                 gibberish_threshold: float = 0.3,
                 min_response_length: int = 10,
                 enable_dictionary_check: bool = True):
        """
        Initialize the nonsense detector.
        
        Args:
            max_word_length: Maximum reasonable word length
            repetition_threshold: Threshold for repetitive content (0-1)
            gibberish_threshold: Threshold for gibberish words (0-1)
            min_response_length: Minimum response length to evaluate
            enable_dictionary_check: Whether to use dictionary for word validation
        """
        self.max_word_length = max_word_length
        self.repetition_threshold = repetition_threshold
        self.gibberish_threshold = gibberish_threshold
        self.min_response_length = min_response_length
        self.enable_dictionary_check = enable_dictionary_check
        
        # Download NLTK data if needed
        self._ensure_nltk_data()
        
        # Load English word dictionary if enabled
        if self.enable_dictionary_check:
            try:
                self.english_words = set(nltk_words.words())
                # Add common contractions and variations
                self.english_words.update([
                    "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
                    "won't", "can't", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't",
                    "haven't", "hasn't", "hadn't", "wouldn't", "shouldn't", "couldn't",
                    "i'll", "you'll", "he'll", "she'll", "it'll", "we'll", "they'll",
                    "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd"
                ])
            except Exception:
                print("Warning: Could not load NLTK words corpus. Dictionary check disabled.")
                self.enable_dictionary_check = False
                self.english_words = set()
        else:
            self.english_words = set()
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception:
                pass
        
        # Also try punkt_tab for newer NLTK versions
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception:
                pass
        
        if self.enable_dictionary_check:
            try:
                nltk.data.find('corpora/words')
            except LookupError:
                try:
                    nltk.download('words', quiet=True)
                except Exception:
                    pass
    
    def detect_nonsense(self, response: str) -> Dict[str, Any]:
        """
        Detect various forms of nonsense in a response.
        
        Args:
            response: The model response to analyze
            
        Returns:
            Dictionary with detection results and scores
        """
        if len(response.strip()) < self.min_response_length:
            return {
                "is_nonsense": False,
                "reason": "too_short",
                "details": {"length": len(response.strip())},
                "confidence": 0.0
            }
        
        # Run all detection methods
        long_words = self._detect_long_words(response)
        repetition = self._detect_repetition(response)
        gibberish = self._detect_gibberish(response)
        incoherence = self._detect_incoherence(response)
        
        # Combine results
        issues = []
        total_score = 0.0
        
        if long_words["has_long_words"]:
            issues.append("long_words")
            total_score += 0.3
        
        if repetition["is_repetitive"]:
            issues.append("repetitive")
            total_score += repetition["score"] * 0.4
        
        if gibberish["has_gibberish"]:
            issues.append("gibberish")
            total_score += gibberish["score"] * 0.5
        
        if incoherence["is_incoherent"]:
            issues.append("incoherent")
            total_score += incoherence["score"] * 0.3
        
        # Determine if response is nonsense
        is_nonsense = total_score > 0.5 or len(issues) >= 2
        
        return {
            "is_nonsense": is_nonsense,
            "issues": issues,
            "confidence": min(total_score, 1.0),
            "details": {
                "long_words": long_words,
                "repetition": repetition,
                "gibberish": gibberish,
                "incoherence": incoherence
            }
        }
    
    def _detect_long_words(self, response: str) -> Dict[str, Any]:
        """Detect unreasonably long words."""
        words = word_tokenize(response.lower())
        long_words = [w for w in words if len(w) > self.max_word_length and w.isalpha()]
        
        return {
            "has_long_words": len(long_words) > 0,
            "long_words": long_words,
            "count": len(long_words),
            "max_length": max([len(w) for w in long_words], default=0)
        }
    
    def _detect_repetition(self, response: str) -> Dict[str, Any]:
        """Detect highly repetitive content."""
        # Tokenize into words
        words = word_tokenize(response.lower())
        words = [w for w in words if w.isalpha()]  # Only alphabetic words
        
        if len(words) < 5:
            return {"is_repetitive": False, "score": 0.0, "details": {}}
        
        # Check for repeated phrases (2-4 words)
        phrase_repetition = self._check_phrase_repetition(words)
        
        # Check for repeated individual words
        word_counts = Counter(words)
        total_words = len(words)
        unique_words = len(word_counts)
        
        # Calculate repetition metrics
        word_repetition_score = 1 - (unique_words / total_words)
        phrase_repetition_score = phrase_repetition["score"]
        
        # Overall repetition score
        repetition_score = max(word_repetition_score, phrase_repetition_score)
        
        return {
            "is_repetitive": repetition_score > self.repetition_threshold,
            "score": repetition_score,
            "details": {
                "word_repetition": word_repetition_score,
                "phrase_repetition": phrase_repetition_score,
                "unique_words": unique_words,
                "total_words": total_words,
                "repeated_phrases": phrase_repetition["repeated_phrases"]
            }
        }
    
    def _check_phrase_repetition(self, words: List[str]) -> Dict[str, Any]:
        """Check for repeated phrases in the word list."""
        repeated_phrases = []
        max_score = 0.0
        
        # Check for 2-4 word phrases
        for phrase_length in [2, 3, 4]:
            if len(words) < phrase_length * 2:
                continue
            
            phrases = []
            for i in range(len(words) - phrase_length + 1):
                phrase = " ".join(words[i:i + phrase_length])
                phrases.append(phrase)
            
            phrase_counts = Counter(phrases)
            repeated = [(phrase, count) for phrase, count in phrase_counts.items() if count > 1]
            
            if repeated:
                repeated_phrases.extend(repeated)
                # Calculate score based on repetition
                total_phrase_instances = sum(count for _, count in repeated)
                score = total_phrase_instances / len(phrases)
                max_score = max(max_score, score)
        
        return {
            "score": max_score,
            "repeated_phrases": repeated_phrases
        }
    
    def _detect_gibberish(self, response: str) -> Dict[str, Any]:
        """Detect gibberish/non-existent words."""
        if not self.enable_dictionary_check:
            return {"has_gibberish": False, "score": 0.0, "gibberish_words": []}
        
        words = word_tokenize(response.lower())
        # Filter to alphabetic words longer than 2 characters
        words = [w for w in words if w.isalpha() and len(w) > 2]
        
        if not words:
            return {"has_gibberish": False, "score": 0.0, "gibberish_words": []}
        
        # Check against dictionary
        gibberish_words = []
        for word in words:
            if word not in self.english_words and not self._is_likely_valid_word(word):
                gibberish_words.append(word)
        
        gibberish_ratio = len(gibberish_words) / len(words)
        
        return {
            "has_gibberish": gibberish_ratio > self.gibberish_threshold,
            "score": gibberish_ratio,
            "gibberish_words": gibberish_words,
            "total_words": len(words)
        }
    
    def _is_likely_valid_word(self, word: str) -> bool:
        """Check if a word is likely valid even if not in dictionary."""
        # Common patterns that might not be in dictionary
        valid_patterns = [
            r'^[a-z]+ing$',      # -ing words
            r'^[a-z]+ed$',       # -ed words  
            r'^[a-z]+er$',       # -er words
            r'^[a-z]+est$',      # -est words
            r'^[a-z]+ly$',       # -ly words
            r'^un[a-z]+$',       # un- prefix
            r'^re[a-z]+$',       # re- prefix
            r'^[a-z]+s$',        # plural -s
        ]
        
        for pattern in valid_patterns:
            if re.match(pattern, word):
                return True
        
        # Check if it's a proper noun (capitalized)
        if word[0].isupper():
            return True
        
        # Check if it's a number written as word
        number_words = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}
        if word.lower() in number_words:
            return True
        
        return False
    
    def _detect_incoherence(self, response: str) -> Dict[str, Any]:
        """Detect incoherent responses."""
        sentences = sent_tokenize(response)
        
        if len(sentences) < 2:
            return {"is_incoherent": False, "score": 0.0, "details": {}}
        
        # Check for very short sentences (might indicate fragmentation)
        short_sentences = [s for s in sentences if len(s.split()) < 3]
        short_ratio = len(short_sentences) / len(sentences)
        
        # Check for sentences without proper punctuation
        no_punct_sentences = [s for s in sentences if not any(p in s for p in '.!?')]
        no_punct_ratio = len(no_punct_sentences) / len(sentences)
        
        # Simple coherence score
        incoherence_score = (short_ratio * 0.5) + (no_punct_ratio * 0.3)
        
        return {
            "is_incoherent": incoherence_score > 0.4,
            "score": incoherence_score,
            "details": {
                "short_sentences": len(short_sentences),
                "no_punct_sentences": len(no_punct_sentences),
                "total_sentences": len(sentences)
            }
        }
    
    def should_stop_generation(self, partial_response: str) -> Tuple[bool, str]:
        """
        Check if generation should be stopped based on partial response.
        
        Args:
            partial_response: The response generated so far
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if len(partial_response.strip()) < self.min_response_length:
            return False, ""
        
        result = self.detect_nonsense(partial_response)
        
        if result["is_nonsense"]:
            issues = result["issues"]
            if "gibberish" in issues:
                return True, "gibberish_detected"
            elif "repetitive" in issues and result["confidence"] > 0.8:
                return True, "excessive_repetition"
            elif len(issues) >= 2:
                return True, f"multiple_issues: {', '.join(issues)}"
        
        return False, ""


def create_nonsense_detector(
    max_word_length: int = 20,
    repetition_threshold: float = 0.7,
    gibberish_threshold: float = 0.3,
    enable_dictionary_check: bool = True
) -> NonsenseDetector:
    """
    Factory function to create a nonsense detector with custom settings.
    
    Args:
        max_word_length: Maximum reasonable word length
        repetition_threshold: Threshold for repetitive content (0-1)
        gibberish_threshold: Threshold for gibberish words (0-1)
        enable_dictionary_check: Whether to use dictionary for word validation
        
    Returns:
        Configured NonsenseDetector instance
    """
    return NonsenseDetector(
        max_word_length=max_word_length,
        repetition_threshold=repetition_threshold,
        gibberish_threshold=gibberish_threshold,
        enable_dictionary_check=enable_dictionary_check
    )


def evaluate_response_quality(response: str, detector: Optional[NonsenseDetector] = None) -> Dict[str, Any]:
    """
    Evaluate the quality of a response using nonsense detection.
    
    Args:
        response: The response to evaluate
        detector: Optional custom detector, creates default if None
        
    Returns:
        Quality evaluation results
    """
    if detector is None:
        detector = create_nonsense_detector()
    
    result = detector.detect_nonsense(response)
    
    # Add quality score (inverse of nonsense confidence)
    quality_score = 1.0 - result["confidence"]
    
    return {
        "quality_score": quality_score,
        "is_high_quality": quality_score > 0.7,
        "nonsense_detection": result
    }
