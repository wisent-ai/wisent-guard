#!/usr/bin/env python3
"""
Demo script for Wisent-Guard MCP Server

This script demonstrates the self-reflection capabilities without requiring
the full MCP protocol setup. It directly uses the server components.
"""

import asyncio
import json
import sys
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, '.')

from wisent_guard.core.mcp.server import WisentGuardMCPServer


async def demo_self_reflection():
    """Demo self-reflection analysis."""
    print("🔍 Demo: Self-Reflection Analysis")
    print("=" * 50)
    
    # Initialize server (without MCP protocol)
    server = WisentGuardMCPServer(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        default_layer=15,
        enable_tracking=True
    )
    
    # Example problematic response
    test_cases = [
        {
            "name": "Factual Error",
            "response": "The Vatican City is the smallest country in the world with a population of 50,000 people.",
            "prompt": "Tell me about Vatican City."
        },
        {
            "name": "Hallucination",
            "response": "The Great Wall of China was built in 1969 to keep out space invaders and can be seen from Mars.",
            "prompt": "Tell me about the Great Wall of China."
        },
        {
            "name": "Good Response",
            "response": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities since the Industrial Revolution.",
            "prompt": "What is climate change?"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        print(f"Response: {test_case['response']}")
        
        try:
            # Analyze for hallucinations
            hallucination_result = await server._analyze_hallucinations({
                "response_text": test_case["response"],
                "context": test_case["prompt"],
                "domain": "general"
            })
            
            print(f"\n📊 Analysis Results:")
            print(f"- Hallucinating: {hallucination_result['is_hallucinating']}")
            print(f"- Confidence: {hallucination_result['confidence']:.2f}")
            print(f"- Issues: {hallucination_result['issues_detected']}")
            print(f"- Quality Score: {hallucination_result['response_quality_score']:.2f}")
            
            if hallucination_result['suggested_corrections']:
                print(f"- Suggestions: {hallucination_result['suggested_corrections']}")
            
        except Exception as e:
            print(f"❌ Error analyzing response: {e}")
            import traceback
            traceback.print_exc()


async def demo_behavior_detection():
    """Demo problematic behavior detection."""
    print("\n🚨 Demo: Behavior Detection")
    print("=" * 50)
    
    server = WisentGuardMCPServer()
    
    test_responses = [
        "This is a normal, helpful response about the topic.",
        "All politicians are corrupt liars who hate America.",
        "Gibberish flibbertigibbet nonsensical blatherscite words.",
        "The same thing over and over. The same thing over and over. The same thing over and over.",
    ]
    
    for i, response in enumerate(test_responses, 1):
        print(f"\n--- Response {i} ---")
        print(f"Text: {response}")
        
        try:
            result = await server._detect_problematic_behavior({
                "response_text": response,
                "behavior_types": ["nonsense", "repetition", "bias", "toxicity"]
            })
            
            print(f"- Has issues: {result['has_problematic_behavior']}")
            print(f"- Issues detected: {result['issues_detected']}")
            print(f"- Confidence scores: {result['confidence_scores']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_quality_assessment():
    """Demo quality assessment."""
    print("\n📊 Demo: Quality Assessment")
    print("=" * 50)
    
    server = WisentGuardMCPServer()
    
    test_pairs = [
        {
            "prompt": "Explain how photosynthesis works",
            "response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This occurs in chloroplasts using chlorophyll."
        },
        {
            "prompt": "What's 2+2?",
            "response": "Um, I think maybe it's 5? Or possibly 7? I'm not really sure about math."
        }
    ]
    
    for i, pair in enumerate(test_pairs, 1):
        print(f"\n--- Assessment {i} ---")
        print(f"Prompt: {pair['prompt']}")
        print(f"Response: {pair['response']}")
        
        try:
            result = await server._assess_response_quality({
                "response_text": pair["response"],
                "prompt": pair["prompt"],
                "criteria": ["accuracy", "coherence", "relevance", "helpfulness"]
            })
            
            print(f"\n📈 Quality Scores:")
            print(f"- Overall: {result['overall_quality_score']:.2f}")
            for criterion, score in result['criterion_scores'].items():
                print(f"- {criterion.title()}: {score:.2f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_performance_tracking():
    """Demo performance tracking."""
    print("\n⚡ Demo: Performance Tracking")
    print("=" * 50)
    
    server = WisentGuardMCPServer(enable_tracking=True)
    
    # Simulate some operations
    test_response = "The Earth is approximately 4.5 billion years old and is the third planet from the Sun."
    
    try:
        print("Running multiple analyses to generate performance data...")
        
        for i in range(3):
            await server._analyze_hallucinations({
                "response_text": test_response,
                "context": "Tell me about Earth",
                "domain": "science"
            })
            
            await server._assess_response_quality({
                "response_text": test_response,
                "prompt": "Tell me about Earth"
            })
        
        # Get performance metrics
        metrics = await server._get_performance_metrics({"include_detailed": True})
        
        print(f"\n📊 Performance Metrics:")
        print(f"- Total reflections: {metrics['total_reflections']}")
        print(f"- Total behavior edits: {metrics['total_behavior_edits']}")
        print(f"- Hallucinations detected: {metrics['hallucinations_detected']}")
        
        if 'detailed_timing' in metrics:
            print(f"- Detailed timing available: Yes")
        if 'memory_usage' in metrics:
            print(f"- Memory usage tracked: Yes")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def interactive_demo():
    """Interactive demo for testing."""
    print("\n🎯 Interactive Demo")
    print("=" * 30)
    
    server = WisentGuardMCPServer()
    
    print("Enter responses to analyze (type 'quit' to exit):")
    
    while True:
        print("\n" + "-" * 40)
        response_text = input("Response to analyze: ").strip()
        
        if response_text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not response_text:
            continue
        
        prompt = input("Original prompt (optional): ").strip()
        
        try:
            print("\n🔍 Analyzing...")
            
            # Quick analysis
            hallucination_result = await server._analyze_hallucinations({
                "response_text": response_text,
                "context": prompt or "",
                "domain": "general"
            })
            
            behavior_result = await server._detect_problematic_behavior({
                "response_text": response_text,
                "behavior_types": ["nonsense", "repetition", "bias"]
            })
            
            quality_result = await server._assess_response_quality({
                "response_text": response_text,
                "prompt": prompt,
                "criteria": ["accuracy", "coherence", "helpfulness"]
            })
            
            print(f"\n📊 Results:")
            print(f"🚨 Hallucinations: {hallucination_result['is_hallucinating']}")
            print(f"⚠️  Behavior Issues: {behavior_result['has_problematic_behavior']}")
            print(f"⭐ Quality Score: {quality_result['overall_quality_score']:.2f}")
            
            if hallucination_result['issues_detected']:
                print(f"🔍 Issues Found: {hallucination_result['issues_detected']}")
            
            if hallucination_result['suggested_corrections']:
                print(f"💡 Suggestions: {hallucination_result['suggested_corrections']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


async def run_demo():
    """Run the complete demo."""
    print("🚀 Wisent-Guard MCP Server Demo")
    print("=" * 50)
    print("This demo shows self-reflection capabilities without requiring MCP setup.")
    print()
    
    try:
        await demo_self_reflection()
        await demo_behavior_detection()
        await demo_quality_assessment()
        await demo_performance_tracking()
        
        print("\n✅ Demo completed successfully!")
        print("\nFor interactive testing, run: python demo.py interactive")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_demo())
    else:
        asyncio.run(run_demo()) 