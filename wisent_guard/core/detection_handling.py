"""
Detection handling module for wisent-guard.

This module provides different strategies for handling responses that have been
detected as problematic (hallucinations, harmful content, bias, etc.).
"""

from enum import Enum
from typing import Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DetectionAction(Enum):
    """Actions to take when problematic content is detected."""
    PASS_THROUGH = "pass_through"  # Output the response as-is
    REPLACE_WITH_PLACEHOLDER = "replace_with_placeholder"  # Replace with safe message
    REGENERATE_UNTIL_SAFE = "regenerate_until_safe"  # Keep regenerating until safe


class DetectionHandler:
    """
    Handles responses when problematic content is detected.
    
    Provides configurable strategies for dealing with detected issues like
    hallucinations, harmful content, bias, etc.
    """
    
    def __init__(
        self,
        action: DetectionAction = DetectionAction.REPLACE_WITH_PLACEHOLDER,
        placeholder_message: Optional[str] = None,
        max_regeneration_attempts: int = 3,
        custom_placeholder_generator: Optional[Callable[[str, str], str]] = None,
        log_detections: bool = True
    ):
        """
        Initialize the detection handler.
        
        Args:
            action: What action to take when detection occurs
            placeholder_message: Custom placeholder message (if None, uses default)
            max_regeneration_attempts: Maximum times to regenerate before giving up
            custom_placeholder_generator: Function to generate custom placeholders
            log_detections: Whether to log detection events
        """
        self.action = action
        self.placeholder_message = placeholder_message
        self.max_regeneration_attempts = max_regeneration_attempts
        self.custom_placeholder_generator = custom_placeholder_generator
        self.log_detections = log_detections
        
        # Default placeholder messages for different detection types
        self.default_placeholders = {
            "hallucination": "I apologize, but I may not have accurate information about this topic. Please verify any factual claims from reliable sources.",
            "harmful_content": "I cannot provide information that could be harmful or dangerous. Please ask about something else I can help with safely.",
            "bias": "I want to avoid potentially biased responses. Let me try to provide a more balanced perspective on this topic.",
            "personal_info": "I cannot generate or discuss personal information. Please ask about general topics instead.",
            "scheming": "I cannot provide advice on deceptive or manipulative behavior. Let me help you with ethical approaches instead.",
            "bad_code": "I cannot provide code examples that may contain security vulnerabilities. Let me suggest secure coding practices instead.",
            "default": "I apologize, but I cannot provide an appropriate response to this request. Please try rephrasing your question."
        }
    
    def handle_detection(
        self,
        original_response: str,
        detection_type: str,
        confidence_score: float,
        original_prompt: str,
        regenerate_function: Optional[Callable[[], str]] = None
    ) -> str:
        """
        Handle a detected problematic response based on the configured action.
        
        Args:
            original_response: The response that was flagged
            detection_type: Type of issue detected (e.g., "hallucination", "bias")
            confidence_score: Confidence score of the detection (0.0 to 1.0)
            original_prompt: The original prompt that generated the response
            regenerate_function: Function to call for regeneration (if needed)
            
        Returns:
            The final response to return to the user
        """
        if self.log_detections:
            logger.warning(
                f"Detected {detection_type} with confidence {confidence_score:.3f} "
                f"in response: {original_response[:100]}..."
            )
        
        if self.action == DetectionAction.PASS_THROUGH:
            return self._handle_pass_through(original_response, detection_type, confidence_score)
            
        elif self.action == DetectionAction.REPLACE_WITH_PLACEHOLDER:
            return self._handle_replacement(original_response, detection_type, original_prompt)
            
        elif self.action == DetectionAction.REGENERATE_UNTIL_SAFE:
            return self._handle_regeneration(
                original_response, detection_type, original_prompt, regenerate_function
            )
        
        else:
            raise ValueError(f"Unknown detection action: {self.action}")
    
    def _handle_pass_through(
        self, 
        original_response: str, 
        detection_type: str, 
        confidence_score: float
    ) -> str:
        """Handle pass-through action - return response as-is with optional warning."""
        if self.log_detections:
            logger.info(f"Passing through response despite {detection_type} detection")
        
        # Optionally add a warning prefix (can be configured)
        return original_response
    
    def _handle_replacement(
        self, 
        original_response: str, 
        detection_type: str, 
        original_prompt: str
    ) -> str:
        """Handle replacement action - return placeholder message."""
        if self.custom_placeholder_generator:
            return self.custom_placeholder_generator(detection_type, original_prompt)
        
        if self.placeholder_message:
            return self.placeholder_message
        
        # Use default placeholder for the detection type
        return self.default_placeholders.get(detection_type, self.default_placeholders["default"])
    
    def _handle_regeneration(
        self,
        original_response: str,
        detection_type: str,
        original_prompt: str,
        regenerate_function: Optional[Callable[[], str]]
    ) -> str:
        """Handle regeneration action - keep generating until safe response."""
        if not regenerate_function:
            logger.warning("No regeneration function provided, falling back to placeholder")
            return self._handle_replacement(original_response, detection_type, original_prompt)
        
        attempts = 0
        current_response = original_response
        
        while attempts < self.max_regeneration_attempts:
            attempts += 1
            
            if self.log_detections:
                logger.info(f"Regeneration attempt {attempts}/{self.max_regeneration_attempts}")
            
            try:
                # Generate a new response
                new_response = regenerate_function()
                
                # Note: In a real implementation, you would re-run the detection here
                # For now, we'll assume the regeneration function handles this
                return new_response
                
            except Exception as e:
                logger.error(f"Error during regeneration attempt {attempts}: {e}")
                continue
        
        # If we've exhausted attempts, fall back to placeholder
        if self.log_detections:
            logger.warning(
                f"Failed to generate safe response after {self.max_regeneration_attempts} attempts, "
                f"using placeholder"
            )
        
        return self._handle_replacement(original_response, detection_type, original_prompt)
    
    def set_custom_placeholder(self, detection_type: str, message: str):
        """Set a custom placeholder message for a specific detection type."""
        self.default_placeholders[detection_type] = message
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get statistics about detection handling (placeholder for future implementation)."""
        return {
            "action": self.action.value,
            "max_regeneration_attempts": self.max_regeneration_attempts,
            "available_placeholders": list(self.default_placeholders.keys())
        }


# Convenience functions for common use cases

def create_pass_through_handler() -> DetectionHandler:
    """Create a handler that passes through all responses unchanged."""
    return DetectionHandler(action=DetectionAction.PASS_THROUGH)


def create_placeholder_handler(custom_message: Optional[str] = None) -> DetectionHandler:
    """Create a handler that replaces detected responses with placeholders."""
    return DetectionHandler(
        action=DetectionAction.REPLACE_WITH_PLACEHOLDER,
        placeholder_message=custom_message
    )


def create_regeneration_handler(max_attempts: int = 3) -> DetectionHandler:
    """Create a handler that regenerates responses until they're safe."""
    return DetectionHandler(
        action=DetectionAction.REGENERATE_UNTIL_SAFE,
        max_regeneration_attempts=max_attempts
    )


def create_custom_handler(
    placeholder_generator: Callable[[str, str], str],
    action: DetectionAction = DetectionAction.REPLACE_WITH_PLACEHOLDER
) -> DetectionHandler:
    """Create a handler with a custom placeholder generator function."""
    return DetectionHandler(
        action=action,
        custom_placeholder_generator=placeholder_generator
    )


# Example custom placeholder generators

def educational_placeholder_generator(detection_type: str, original_prompt: str) -> str:
    """Generate educational placeholders that explain why content was flagged."""
    explanations = {
        "hallucination": f"The response to '{original_prompt}' may contain inaccurate information. "
                        "Please verify facts from reliable sources before relying on this information.",
        "harmful_content": f"I cannot provide a response to '{original_prompt}' as it may involve "
                          "harmful or dangerous content. Please ask about safer topics.",
        "bias": f"The response to '{original_prompt}' might contain biased perspectives. "
               "Consider seeking multiple viewpoints on this topic.",
        "personal_info": f"I cannot respond to '{original_prompt}' as it involves personal information. "
                        "Please ask about general topics instead."
    }
    
    return explanations.get(
        detection_type,
        f"I cannot provide an appropriate response to '{original_prompt}'. "
        "Please try rephrasing your question."
    )


def brief_placeholder_generator(detection_type: str, original_prompt: str) -> str:
    """Generate brief, minimal placeholder messages."""
    brief_messages = {
        "hallucination": "Information may be inaccurate.",
        "harmful_content": "Cannot provide harmful content.",
        "bias": "Response may be biased.",
        "personal_info": "Cannot share personal information.",
        "scheming": "Cannot provide deceptive advice.",
        "bad_code": "Cannot provide insecure code."
    }
    
    return brief_messages.get(detection_type, "Cannot provide response.")
