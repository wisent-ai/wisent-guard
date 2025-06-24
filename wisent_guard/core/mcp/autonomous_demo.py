#!/usr/bin/env python3
"""
Autonomous Model Demo with MCP Tools

This demo presents problems to a model and lets it autonomously decide:
- Which MCP tools to use
- How to train classifiers/steering vectors  
- How to analyze and improve its own responses

The demo provides NO hardcoded logic - the model makes all decisions.
"""

import asyncio
from typing import Dict, Any, List

# Simple demo that just presents problems
class AutonomousDemo:
    """
    Presents problems to a model that has access to MCP tools.
    The model must decide autonomously how to solve them.
    """
    
    def __init__(self):
        self.problems = [
            {
                "id": 1,
                "description": "Tell me about Vatican City's population and when it was founded"
            },
            {
                "id": 2, 
                "description": "Write a creative story about magical talking animals"
            },
            {
                "id": 3,
                "description": "Explain the symptoms of a heart attack and what to do"
            },
            {
                "id": 4,
                "description": "Describe how different cultures approach problem-solving"
            },
            {
                "id": 5,
                "description": "Explain quantum entanglement in simple terms"
            }
        ]

    def present_problem(self, problem_id: int) -> Dict[str, Any]:
        """Present a problem to the model."""
        if problem_id < 1 or problem_id > len(self.problems):
            return {"error": "Invalid problem ID"}
        
        problem = self.problems[problem_id - 1]
        
        return {
            "problem": problem,
            "instructions": (
                "Respond to this request. "
                "You have access to an MCP server. "
                "Use it however you think is appropriate, or not at all."
            )
        }

    def list_problems(self) -> List[Dict[str, Any]]:
        """List all available problems."""
        return [
            {
                "id": p["id"],
                "description": p["description"], 
            }
            for p in self.problems
        ]

async def run_demo():
    """Run the autonomous demo."""
    demo = AutonomousDemo()
    
    print("🤖 AUTONOMOUS MODEL DEMO WITH MCP TOOLS")
    print("=" * 50)
    print("Available problems for autonomous solving:")
    print()
    
    problems = demo.list_problems()
    for problem in problems:
        print(f"Problem {problem['id']}: {problem['description']}")
        print()
    
    print("To use this demo:")
    print("1. Start the MCP server: python -m wisent_guard.core.mcp.server")
    print("2. Connect a model (like Claude) to the MCP server")
    print("3. Present problems to the model and let it decide how to solve them")
    print()
    print("Example usage:")
    for i, problem in enumerate(problems[:2], 1):
        problem_data = demo.present_problem(i)
        print(f"\n--- Problem {i} ---")
        print(f"Description: {problem_data['problem']['description']}")
        print("Instructions: Let the model decide autonomously how to use these tools")

if __name__ == "__main__":
    asyncio.run(run_demo()) 