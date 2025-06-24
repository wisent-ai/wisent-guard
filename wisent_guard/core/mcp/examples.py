"""
Example usage of Wisent-Guard MCP server for model self-reflection.
"""

import asyncio
import json
from typing import Dict, Any

from .client import (
    WisentGuardMCPClient, 
    SelfReflectionRequest, 
    BehaviorEditRequest,
    quick_hallucination_check,
    auto_fix_response
)


async def example_basic_self_reflection():
    """Example of basic self-reflection analysis."""
    print("🔍 Example: Basic Self-Reflection Analysis")
    print("=" * 50)
    
    # Example problematic response
    response_text = """
    The Vatican City is the smallest country in the world that is at least one square mile in area. 
    It was founded in 1929 when the Pope signed the Lateran Treaty with Benito Mussolini. 
    The population is approximately 50,000 people, making it very densely populated.
    """
    
    original_prompt = "Tell me about Vatican City."
    
    async with WisentGuardMCPClient() as client:
        # Perform self-reflection
        request = SelfReflectionRequest(
            response_text=response_text,
            original_prompt=original_prompt,
            analysis_depth="standard",
            focus_areas=["hallucinations", "accuracy", "coherence"]
        )
        
        result = await client.perform_self_reflection(request)
        
        print(f"Response analyzed: {response_text[:100]}...")
        print(f"\nSelf-reflection results:")
        print(f"- Hallucinations detected: {result.get('hallucination_analysis', {}).get('is_hallucinating', False)}")
        print(f"- Issues found: {result.get('hallucination_analysis', {}).get('issues_detected', [])}")
        print(f"- Overall reflection score: {result.get('overall_reflection_score', {})}")
        
        return result


async def example_hallucination_detection():
    """Example of specialized hallucination detection."""
    print("\n🚨 Example: Hallucination Detection")
    print("=" * 50)
    
    # Example with factual errors
    response_text = """
    The Great Wall of China was built in 1969 to keep out the Mongolian space invaders. 
    It's made entirely of concrete and stretches for exactly 50,000 kilometers. 
    The wall can be seen from Mars with the naked eye.
    """
    
    async with WisentGuardMCPClient() as client:
        result = await client.detect_hallucinations(
            response_text=response_text,
            knowledge_domain="history",
            fact_check_level="advanced"
        )
        
        print(f"Response: {response_text}")
        print(f"\nHallucination detection:")
        detection = result.get("hallucination_detection", {})
        print(f"- Is hallucinating: {detection.get('is_hallucinating', False)}")
        print(f"- Confidence: {detection.get('confidence', 0.0):.2f}")
        print(f"- Issues: {detection.get('issues_detected', [])}")
        print(f"- Enhanced analysis: {json.dumps(detection.get('enhanced_analysis', {}), indent=2)}")
        
        return result


async def example_behavior_editing():
    """Example of behavior editing using steering."""
    print("\n✏️  Example: Behavior Editing")
    print("=" * 50)
    
    original_prompt = "Explain why vaccines are important."
    problematic_response = "Vaccines are completely useless and cause autism in all children."
    
    async with WisentGuardMCPClient() as client:
        request = BehaviorEditRequest(
            original_prompt=original_prompt,
            current_response=problematic_response,
            desired_changes=["more factual", "less biased", "evidence-based"],
            steering_method="CAA",
            steering_strength=1.5,
            max_attempts=3
        )
        
        result = await client.edit_behavior(request)
        
        print(f"Original response: {problematic_response}")
        print(f"\nBehavior editing results:")
        editing = result.get("behavior_editing", {})
        print(f"- Editing successful: {editing.get('editing_successful', False)}")
        print(f"- Attempts made: {editing.get('total_attempts', 0)}")
        
        best_result = editing.get("best_result", {})
        if best_result:
            print(f"- Improved response: {best_result.get('edited_response', 'N/A')}")
            print(f"- Improvement score: {best_result.get('improvement_score', 0.0):.2f}")
        
        return result


async def example_comprehensive_analysis():
    """Example of comprehensive response analysis."""
    print("\n📊 Example: Comprehensive Analysis")
    print("=" * 50)
    
    response_text = """
    Climate change is a complex phenomenon that scientists have been studying for decades. 
    While there is broad scientific consensus that human activities contribute to climate change, 
    the exact impacts and timelines remain subjects of ongoing research. 
    It's important to consider multiple perspectives and continue supporting scientific research 
    to better understand this important issue.
    """
    
    original_prompt = "What do you think about climate change?"
    
    async with WisentGuardMCPClient() as client:
        result = await client.analyze_response_comprehensive(
            response_text=response_text,
            original_prompt=original_prompt,
            analysis_categories=["quality", "safety", "accuracy", "helpfulness"],
            include_suggestions=True,
            compare_to_baseline=True
        )
        
        print(f"Response: {response_text}")
        print(f"\nComprehensive analysis:")
        
        overall = result.get("overall_metrics", {})
        print(f"- Overall score: {overall.get('overall_score', 0.0):.2f}")
        print(f"- Quality score: {overall.get('quality_score', 0.0):.2f}")
        print(f"- Safety score: {overall.get('safety_score', 0.0):.2f}")
        print(f"- Issues count: {overall.get('issues_count', 0)}")
        print(f"- Strengths: {overall.get('strengths', [])}")
        print(f"- Weaknesses: {overall.get('weaknesses', [])}")
        
        suggestions = result.get("improvement_suggestions", {})
        if suggestions:
            print(f"\nImprovement suggestions:")
            print(f"- Immediate actions: {suggestions.get('immediate_actions', [])}")
            print(f"- Steering recommendations: {suggestions.get('steering_recommendations', [])}")
        
        return result


async def example_quick_workflows():
    """Example of quick convenience workflows."""
    print("\n⚡ Example: Quick Workflows")
    print("=" * 50)
    
    # Quick hallucination check
    response = "The moon is made of cheese and was discovered by aliens in 1969."
    is_hallucinating = await quick_hallucination_check(response)
    print(f"Quick hallucination check: '{response[:50]}...' -> {is_hallucinating}")
    
    # Auto-fix response
    prompt = "Tell me about the moon."
    problematic = "The moon is made of cheese and inhabited by mice."
    fixed = await auto_fix_response(prompt, problematic)
    print(f"\nAuto-fix example:")
    print(f"Original: {problematic}")
    print(f"Fixed: {fixed}")
    
    # Quick self-check
    async with WisentGuardMCPClient() as client:
        quick_result = await client.quick_self_check(
            response_text="The Earth is flat and gravity is just a theory.",
            original_prompt="Explain the shape of the Earth."
        )
        
        print(f"\nQuick self-check:")
        print(f"- Needs regeneration: {quick_result['needs_regeneration']}")
        print(f"- Issues found: {quick_result['issues_found']}")
        print(f"- Recommendation: {quick_result['recommendation']}")


async def example_auto_improvement():
    """Example of automatic iterative improvement."""
    print("\n🔄 Example: Auto-Improvement")
    print("=" * 50)
    
    prompt = "Explain how vaccines work."
    poor_response = "Vaccines are bad. They have chemicals. Don't use them. Natural immunity is better always."
    
    async with WisentGuardMCPClient() as client:
        result = await client.auto_improve_response(
            original_prompt=prompt,
            current_response=poor_response,
            max_iterations=3
        )
        
        improvement = result.get("auto_improvement", {})
        print(f"Original response: {improvement.get('original_response', '')}")
        print(f"Final response: {improvement.get('final_response', '')}")
        print(f"Iterations: {improvement.get('improvement_iterations', 0)}")
        print(f"Final score: {improvement.get('final_score', 0.0):.2f}")
        print(f"Successful: {improvement.get('successful', False)}")
        
        return result


async def example_reflection_history():
    """Example of accessing reflection history and metrics."""
    print("\n📈 Example: Reflection History & Metrics")
    print("=" * 50)
    
    async with WisentGuardMCPClient() as client:
        # Get reflection history
        history = await client.get_reflection_history(limit=5, filter_by="all")
        print(f"Reflection history entries: {history.get('total_entries', 0)}")
        print(f"Filtered entries: {history.get('filtered_entries', 0)}")
        
        # Get performance metrics
        metrics = await client.get_performance_metrics(include_detailed=True)
        print(f"\nPerformance metrics:")
        print(f"- Total reflections: {metrics.get('total_reflections', 0)}")
        print(f"- Total behavior edits: {metrics.get('total_behavior_edits', 0)}")
        print(f"- Hallucinations detected: {metrics.get('hallucinations_detected', 0)}")
        print(f"- Successful edits: {metrics.get('successful_edits', 0)}")
        
        return history, metrics


async def run_all_examples():
    """Run all examples in sequence."""
    print("🚀 Wisent-Guard MCP Server Examples")
    print("=" * 60)
    
    try:
        await example_basic_self_reflection()
        await example_hallucination_detection()
        await example_behavior_editing()
        await example_comprehensive_analysis()
        await example_quick_workflows()
        await example_auto_improvement()
        await example_reflection_history()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


# Interactive example for testing
async def interactive_self_reflection():
    """Interactive example for testing self-reflection."""
    print("🎯 Interactive Self-Reflection Test")
    print("=" * 40)
    
    async with WisentGuardMCPClient() as client:
        while True:
            print("\nEnter a response to analyze (or 'quit' to exit):")
            response_text = input("> ")
            
            if response_text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not response_text.strip():
                continue
            
            print("\nEnter the original prompt (optional):")
            prompt = input("> ")
            
            try:
                # Quick self-check
                quick_result = await client.quick_self_check(
                    response_text=response_text,
                    original_prompt=prompt if prompt.strip() else None
                )
                
                print(f"\n📊 Quick Analysis Results:")
                print(f"- Needs regeneration: {quick_result['needs_regeneration']}")
                print(f"- Issues found: {quick_result['issues_found']}")
                print(f"- Confidence: {quick_result['confidence']:.2f}")
                print(f"- Recommendation: {quick_result['recommendation']}")
                
                # Ask if user wants detailed analysis
                if input("\nWant detailed analysis? (y/n): ").lower().startswith('y'):
                    detailed = await client.analyze_response_comprehensive(
                        response_text=response_text,
                        original_prompt=prompt if prompt.strip() else None,
                        include_suggestions=True
                    )
                    
                    overall = detailed.get("overall_metrics", {})
                    print(f"\n📈 Detailed Analysis:")
                    print(f"- Overall score: {overall.get('overall_score', 0.0):.2f}")
                    print(f"- Strengths: {overall.get('strengths', [])}")
                    print(f"- Weaknesses: {overall.get('weaknesses', [])}")
                    
                    suggestions = detailed.get("improvement_suggestions", {})
                    if suggestions.get("steering_recommendations"):
                        print(f"- Steering recommendations: {suggestions['steering_recommendations']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_self_reflection())
    else:
        asyncio.run(run_all_examples()) 