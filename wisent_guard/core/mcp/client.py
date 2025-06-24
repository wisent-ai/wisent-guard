"""
Wisent-Guard MCP Client

Client for interacting with the Wisent-Guard MCP server for self-reflection capabilities.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

try:
    from mcp import stdio_client, ClientSession, StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    stdio_client = None
    StdioServerParameters = None

logger = logging.getLogger(__name__)


@dataclass
class SelfReflectionRequest:
    """Request for self-reflection analysis."""
    response_text: str
    original_prompt: Optional[str] = None
    analysis_depth: str = "standard"
    focus_areas: Optional[List[str]] = None


@dataclass
class BehaviorEditRequest:
    """Request for behavior editing."""
    original_prompt: str
    current_response: str
    desired_changes: List[str]
    steering_method: str = "auto"
    steering_strength: float = 1.0
    max_attempts: int = 3


class WisentGuardMCPClient:
    """
    Client for interacting with Wisent-Guard MCP server.
    
    Provides high-level methods for model self-reflection and behavior editing.
    """
    
    def __init__(self, server_command: Optional[List[str]] = None):
        """
        Initialize the MCP client.
        
        Args:
            server_command: Command to start the MCP server. If None, assumes server is already running.
        """
        if not MCP_AVAILABLE:
            raise ImportError("MCP package not available. Install with: pip install mcp")
        
        self.server_command = server_command or ["python", "-m", "wisent_guard.core.mcp.server"]
        self.client = None
        self.client_context = None
        self.available_tools = []
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Connect to the MCP server."""
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=self.server_command[0],
                args=self.server_command[1:] if len(self.server_command) > 1 else []
            )
            
            # Start the server process and connect using context manager
            self.client_context = stdio_client(server_params)
            read_stream, write_stream = await self.client_context.__aenter__()
            
            # Create client session with the streams
            self.client = ClientSession(read_stream, write_stream)
            
            # Initialize the client
            await self.client.initialize()
            
            # Get available tools
            tools_response = await self.client.list_tools()
            self.available_tools = [tool.name for tool in tools_response.tools]
            
            logger.info(f"Connected to Wisent-Guard MCP server. Available tools: {self.available_tools}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.client:
            try:
                await self.client_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error disconnecting from MCP server: {e}")
            finally:
                self.client = None
                self.client_context = None
    
    async def perform_self_reflection(self, request: SelfReflectionRequest) -> Dict[str, Any]:
        """
        Perform comprehensive self-reflection analysis.
        
        Args:
            request: Self-reflection request parameters
            
        Returns:
            Comprehensive analysis results
        """
        if not self.client:
            raise RuntimeError("Client not connected. Use async context manager or call connect() first.")
        
        args = {
            "response_text": request.response_text,
            "analysis_depth": request.analysis_depth
        }
        
        if request.original_prompt:
            args["original_prompt"] = request.original_prompt
        
        if request.focus_areas:
            args["focus_areas"] = request.focus_areas
        
        try:
            response = await self.client.call_tool("perform_self_reflection", args)
            return json.loads(response.content[0].text)
        except Exception as e:
            logger.error(f"Self-reflection analysis failed: {e}")
            raise
    
    async def detect_hallucinations(self, 
                                  response_text: str, 
                                  knowledge_domain: str = "general",
                                  fact_check_level: str = "intermediate",
                                  reference_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect hallucinations in model response.
        
        Args:
            response_text: The response to analyze
            knowledge_domain: Domain for specialized checking
            fact_check_level: Level of fact-checking
            reference_context: Reference context for comparison
            
        Returns:
            Hallucination detection results
        """
        if not self.client:
            raise RuntimeError("Client not connected")
        
        args = {
            "response_text": response_text,
            "knowledge_domain": knowledge_domain,
            "fact_check_level": fact_check_level
        }
        
        if reference_context:
            args["reference_context"] = reference_context
        
        try:
            response = await self.client.call_tool("detect_hallucinations", args)
            return json.loads(response.content[0].text)
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            raise
    
    async def edit_behavior(self, request: BehaviorEditRequest) -> Dict[str, Any]:
        """
        Edit model behavior using steering methods.
        
        Args:
            request: Behavior editing request parameters
            
        Returns:
            Behavior editing results
        """
        if not self.client:
            raise RuntimeError("Client not connected")
        
        args = {
            "original_prompt": request.original_prompt,
            "current_response": request.current_response,
            "desired_changes": request.desired_changes,
            "steering_method": request.steering_method,
            "steering_strength": request.steering_strength,
            "max_attempts": request.max_attempts
        }
        
        try:
            response = await self.client.call_tool("edit_behavior", args)
            return json.loads(response.content[0].text)
        except Exception as e:
            logger.error(f"Behavior editing failed: {e}")
            raise
    
    async def analyze_response_comprehensive(self,
                                           response_text: str,
                                           original_prompt: Optional[str] = None,
                                           analysis_categories: Optional[List[str]] = None,
                                           include_suggestions: bool = True,
                                           compare_to_baseline: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive response analysis.
        
        Args:
            response_text: The response to analyze
            original_prompt: The original prompt
            analysis_categories: Categories of analysis to perform
            include_suggestions: Whether to include improvement suggestions
            compare_to_baseline: Whether to compare against baseline
            
        Returns:
            Comprehensive analysis results
        """
        if not self.client:
            raise RuntimeError("Client not connected")
        
        args = {
            "response_text": response_text,
            "include_suggestions": include_suggestions,
            "compare_to_baseline": compare_to_baseline
        }
        
        if original_prompt:
            args["original_prompt"] = original_prompt
        
        if analysis_categories:
            args["analysis_categories"] = analysis_categories
        
        try:
            response = await self.client.call_tool("analyze_response_comprehensive", args)
            return json.loads(response.content[0].text)
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    async def get_reflection_history(self, 
                                   limit: int = 10, 
                                   filter_by: str = "all") -> Dict[str, Any]:
        """
        Get history of self-reflection analyses.
        
        Args:
            limit: Maximum number of entries to return
            filter_by: Filter results by type
            
        Returns:
            Reflection history
        """
        if not self.client:
            raise RuntimeError("Client not connected")
        
        args = {
            "limit": limit,
            "filter_by": filter_by
        }
        
        try:
            response = await self.client.call_tool("get_reflection_history", args)
            return json.loads(response.content[0].text)
        except Exception as e:
            logger.error(f"Failed to get reflection history: {e}")
            raise
    
    async def get_performance_metrics(self, include_detailed: bool = False) -> Dict[str, Any]:
        """
        Get performance metrics for self-reflection operations.
        
        Args:
            include_detailed: Whether to include detailed metrics
            
        Returns:
            Performance metrics
        """
        if not self.client:
            raise RuntimeError("Client not connected")
        
        args = {"include_detailed": include_detailed}
        
        try:
            response = await self.client.call_tool("get_performance_metrics", args)
            return json.loads(response.content[0].text)
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            raise
    
    # Convenience methods for common workflows
    
    async def quick_self_check(self, response_text: str, original_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform a quick self-check for common issues.
        
        Args:
            response_text: The response to check
            original_prompt: The original prompt
            
        Returns:
            Quick check results with recommendations
        """
        request = SelfReflectionRequest(
            response_text=response_text,
            original_prompt=original_prompt,
            analysis_depth="quick",
            focus_areas=["hallucinations", "coherence"]
        )
        
        result = await self.perform_self_reflection(request)
        
        # Extract key insights for quick decision making
        quick_check = {
            "needs_regeneration": False,
            "issues_found": [],
            "confidence": 0.0,
            "recommendation": "accept"
        }
        
        if "hallucination_analysis" in result:
            hall_analysis = result["hallucination_analysis"]
            if hall_analysis["is_hallucinating"]:
                quick_check["needs_regeneration"] = True
                quick_check["issues_found"].extend(hall_analysis["issues_detected"])
                quick_check["confidence"] = hall_analysis["confidence"]
        
        if "overall_reflection_score" in result:
            score = result["overall_reflection_score"]
            if score["needs_improvement"]:
                quick_check["recommendation"] = score["recommendation"]
        
        quick_check["full_analysis"] = result
        return quick_check
    
    async def auto_improve_response(self, 
                                  original_prompt: str, 
                                  current_response: str,
                                  max_iterations: int = 3) -> Dict[str, Any]:
        """
        Automatically improve a response through iterative self-reflection and editing.
        
        Args:
            original_prompt: The original prompt
            current_response: The current response to improve
            max_iterations: Maximum number of improvement iterations
            
        Returns:
            Improvement process results
        """
        improvement_history = []
        best_response = current_response
        best_score = 0.0
        
        for iteration in range(max_iterations):
            # Analyze current response
            analysis = await self.analyze_response_comprehensive(
                response_text=best_response,
                original_prompt=original_prompt,
                include_suggestions=True
            )
            
            current_score = analysis.get("overall_metrics", {}).get("overall_score", 0.0)
            
            improvement_history.append({
                "iteration": iteration + 1,
                "response": best_response,
                "analysis": analysis,
                "score": current_score
            })
            
            # Check if we have a good enough response
            if current_score > 0.8:
                break
            
            # Get improvement suggestions
            suggestions = analysis.get("improvement_suggestions", {})
            steering_recs = suggestions.get("steering_recommendations", [])
            
            if not steering_recs:
                break  # No steering recommendations available
            
            # Apply behavior editing
            edit_request = BehaviorEditRequest(
                original_prompt=original_prompt,
                current_response=best_response,
                desired_changes=steering_recs,
                steering_method="auto",
                max_attempts=2
            )
            
            edit_result = await self.edit_behavior(edit_request)
            
            if edit_result.get("behavior_editing", {}).get("editing_successful", False):
                best_result = edit_result["behavior_editing"]["best_result"]
                if best_result and best_result["improvement_score"] > 0.5:
                    best_response = best_result["edited_response"]
                    best_score = current_score + best_result["improvement_score"] * 0.2
        
        return {
            "auto_improvement": {
                "original_response": current_response,
                "final_response": best_response,
                "improvement_iterations": len(improvement_history),
                "final_score": best_score,
                "improvement_history": improvement_history,
                "successful": best_score > 0.6
            }
        }


# Convenience functions for simple use cases

async def quick_hallucination_check(response_text: str, context: Optional[str] = None) -> bool:
    """
    Quick check if a response contains hallucinations.
    
    Args:
        response_text: The response to check
        context: Optional context
        
    Returns:
        True if hallucinations detected, False otherwise
    """
    async with WisentGuardMCPClient() as client:
        result = await client.detect_hallucinations(
            response_text=response_text,
            reference_context=context,
            fact_check_level="basic"
        )
        return result.get("hallucination_detection", {}).get("is_hallucinating", False)


async def auto_fix_response(prompt: str, response: str) -> str:
    """
    Automatically fix a problematic response.
    
    Args:
        prompt: The original prompt
        response: The problematic response
        
    Returns:
        Improved response
    """
    async with WisentGuardMCPClient() as client:
        result = await client.auto_improve_response(prompt, response)
        return result.get("auto_improvement", {}).get("final_response", response) 