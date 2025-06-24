#!/usr/bin/env python3
"""
Test Autonomous MCP Usage

This script demonstrates a model autonomously:
1. Connecting to the MCP server
2. Discovering available tools
3. Deciding which tools to use for each problem
4. Solving problems using MCP tools
"""

import asyncio
import json
from wisent_guard.core.mcp.client import WisentGuardMCPClient
from wisent_guard.core.mcp.autonomous_demo import AutonomousDemo
from wisent_guard.core.model import Model

class AutonomousTestModel:
    """
    A model that autonomously uses MCP tools to solve problems.
    """
    
    def __init__(self):
        self.model = None
        self.mcp_client = None
        self.available_tools = []
        
    async def initialize(self):
        """Initialize the model and MCP client."""
        print("🚀 Initializing autonomous test model...")
        
        # Load the model
        print("📥 Loading model...")
        self.model = Model("meta-llama/Llama-3.1-8B-Instruct")
        
        # Connect to MCP server
        print("🔌 Connecting to MCP server...")
        self.mcp_client = WisentGuardMCPClient()
        await self.mcp_client.connect()
        
        # Discover available tools
        print("🔍 Discovering available MCP tools...")
        self.available_tools = await self.mcp_client.list_tools()
        print(f"   Found {len(self.available_tools)} tools: {[tool['name'] for tool in self.available_tools]}")
        
    async def solve_problem_autonomously(self, problem: dict):
        """
        Autonomously solve a problem using MCP tools.
        The model decides everything - what tools to use, how to use them, etc.
        """
        print(f"\n🎯 SOLVING PROBLEM {problem['id']}")
        print(f"   Description: {problem['description']}")
        
        # Model autonomously decides what to do
        decision_prompt = f"""
        I need to solve this problem: {problem['description']}
        
        I have access to these MCP tools: {[tool['name'] for tool in self.available_tools]}
        
        What should I worry about? What tools should I use? Should I:
        - Generate a response first, then analyze it?
        - Train classifiers before responding?
        - Use steering vectors?
        - Check for specific issues?
        - Do nothing special?
        
        Decide autonomously what approach to take. Be specific about which tools to use and why.
        """
        
        print("🤔 Model deciding autonomous strategy...")
        strategy = await self.model.generate_async(decision_prompt, max_new_tokens=300)
        print(f"   Strategy: {strategy[:200]}...")
        
        # Model generates initial response
        print("💭 Generating initial response...")
        initial_response = await self.model.generate_async(problem['description'], max_new_tokens=200)
        print(f"   Response: {initial_response[:100]}...")
        
        # Model decides if it wants to analyze/improve the response
        analysis_prompt = f"""
        I just generated this response: {initial_response}
        
        For the problem: {problem['description']}
        
        Should I analyze this response? What should I check for? Available tools:
        {[tool['name'] for tool in self.available_tools]}
        
        Decide if I should use any tools and which ones. Be specific.
        """
        
        print("🔍 Model deciding whether to analyze response...")
        analysis_decision = await self.model.generate_async(analysis_prompt, max_new_tokens=200)
        print(f"   Analysis decision: {analysis_decision[:150]}...")
        
        # If model decides to use tools, let it choose which ones
        if "analyze" in analysis_decision.lower() or "tool" in analysis_decision.lower():
            print("🛠️ Model chose to use MCP tools...")
            
            # Model picks specific tools to use
            tool_selection_prompt = f"""
            I want to analyze my response: {initial_response}
            
            Available tools: {[tool['name'] for tool in self.available_tools]}
            
            Which specific tool should I call first? Just give me the tool name.
            """
            
            tool_choice = await self.model.generate_async(tool_selection_prompt, max_new_tokens=50)
            print(f"   Model chose tool: {tool_choice.strip()}")
            
            # Try to use the chosen tool
            chosen_tool_name = tool_choice.strip().split()[0]  # Get first word
            
            if any(tool['name'] == chosen_tool_name for tool in self.available_tools):
                print(f"✅ Using tool: {chosen_tool_name}")
                try:
                    # Use the tool with basic parameters
                    if chosen_tool_name == "analyze_response_for_hallucinations":
                        result = await self.mcp_client.call_tool(chosen_tool_name, {
                            "response": initial_response,
                            "original_prompt": problem['description']
                        })
                        print(f"   Tool result: {result}")
                    else:
                        print(f"   Tool {chosen_tool_name} would be called here")
                        
                except Exception as e:
                    print(f"   ⚠️ Tool call failed: {e}")
            else:
                print(f"   ❌ Tool {chosen_tool_name} not found")
        else:
            print("🎯 Model chose not to use any tools")
            
        print(f"✅ Problem {problem['id']} completed")
        
    async def run_test(self):
        """Run the full autonomous test."""
        await self.initialize()
        
        # Get problems from demo
        demo = AutonomousDemo()
        problems = demo.get_problems()
        
        print(f"\n🎮 TESTING AUTONOMOUS PROBLEM SOLVING")
        print(f"   Problems to solve: {len(problems)}")
        
        # Solve first 2 problems autonomously
        for problem in problems[:2]:
            await self.solve_problem_autonomously(problem)
            
        print(f"\n🎉 AUTONOMOUS TEST COMPLETED")
        
        # Cleanup
        await self.mcp_client.disconnect()

async def main():
    """Run the autonomous test."""
    test_model = AutonomousTestModel()
    await test_model.run_test()

if __name__ == "__main__":
    asyncio.run(main()) 