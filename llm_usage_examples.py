#!/usr/bin/env python3
"""
Multi-LLM Usage Examples

This file demonstrates how to use the multi-LLM provider system
with various features including failover, specialized routing,
health monitoring, and configuration management.
"""

import os
import sys
from typing import Optional

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llms import llm_manager, MultiLLMManager


def example_basic_usage():
    """Example 1: Basic LLM usage with automatic failover."""
    print("="*60)
    print("EXAMPLE 1: Basic LLM Usage")
    print("="*60)

    if not llm_manager:
        print("❌ LLM Manager not initialized")
        return

    # Simple prompt
    prompt = "Explain machine learning in one sentence."

    try:
        response = llm_manager.invoke_with_fallback(prompt)
        print(f"✅ Response from {response.provider} ({response.model}):")
        print(f"   {response.content}")
        print(f"   Response time: {response.response_time:.2f}s")
    except Exception as e:
        print(f"❌ Failed: {e}")


def example_specialized_routing():
    """Example 2: Specialized routing for different use cases."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Specialized Routing")
    print("="*60)

    if not llm_manager:
        return

    # Math problem - uses specialized math routing
    math_prompt = "Solve: d/dx(x² + 3x - 5)"
    try:
        response = llm_manager.invoke_with_fallback(
            math_prompt,
            use_case="math_problems"
        )
        print(f"📐 Math problem solved by {response.provider}:")
        print(f"   {response.content[:200]}...")
    except Exception as e:
        print(f"❌ Math routing failed: {e}")

    # Language analysis - uses specialized language routing
    lang_prompt = "Analyze the grammatical structure of: 'Despite the rain, they continued hiking.'"
    try:
        response = llm_manager.invoke_with_fallback(
            lang_prompt,
            use_case="language_analysis"
        )
        print(f"\n🔍 Language analysis by {response.provider}:")
        print(f"   {response.content[:200]}...")
    except Exception as e:
        print(f"❌ Language routing failed: {e}")


def example_health_monitoring():
    """Example 3: Health monitoring and status checking."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Health Monitoring")
    print("="*60)

    if not llm_manager:
        return

    # Get comprehensive health status
    health_status = llm_manager.get_health_status()

    print("Provider Health Summary:")
    for provider, status in health_status.items():
        health_icon = "🟢" if status['is_healthy'] else "🔴"
        print(f"{health_icon} {provider.upper()}:")
        print(f"   Success rate: {status['success_rate']}")
        print(f"   Total requests: {status['total_requests']}")
        print(f"   Avg response time: {status['average_response_time']}")
        print(f"   Circuit breaker: {status['circuit_breaker_state']}")


def example_configuration_management():
    """Example 4: Configuration and priority management."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Configuration Management")
    print("="*60)

    if not llm_manager:
        return

    # Show current configuration
    config = llm_manager.config
    print("Current Configuration:")
    print(f"🔧 Provider priority: {config.get('provider_priority', {})}")
    print(f"🔧 Enabled providers: {list(llm_manager.providers.keys())}")
    print(f"🔧 Failover enabled: {config.get('failover', {}).get('enabled', False)}")

    # Show priority order for different use cases
    print("\nPriority Orders:")
    print(f"📊 Default: {llm_manager.get_priority_order()}")
    print(f"📐 Math: {llm_manager.get_priority_order('math_problems')}")
    print(f"🔍 Language: {llm_manager.get_priority_order('language_analysis')}")
    print(f"✍️  Creative: {llm_manager.get_priority_order('creative_writing')}")
    print(f"💻 Code: {llm_manager.get_priority_order('code_analysis')}")


def example_error_handling():
    """Example 5: Error handling and recovery."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Error Handling & Recovery")
    print("="*60)

    if not llm_manager:
        return

    # Test with a challenging prompt that might fail
    challenging_prompt = """
    This is a very long and complex prompt that might challenge some LLM providers.
    Please solve this multi-step problem:
    1. Calculate the derivative of f(x) = x³ + 2x² - 5x + 1
    2. Find the critical points
    3. Determine local maxima and minima
    4. Sketch the function behavior
    """ * 3  # Make it even longer

    try:
        print("🧪 Testing with challenging prompt...")
        response = llm_manager.invoke_with_fallback(challenging_prompt)
        print(f"✅ Successfully handled by {response.provider}")
        print(f"   Response length: {len(response.content)} characters")
        print(f"   Response time: {response.response_time:.2f}s")
    except Exception as e:
        print(f"❌ All providers failed: {e}")


def example_custom_configuration():
    """Example 6: Using custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Configuration")
    print("="*60)

    try:
        # Create a custom LLM manager with specific configuration
        custom_manager = MultiLLMManager("llm_config.yaml")

        print(f"✅ Custom manager initialized with {len(custom_manager.providers)} providers")

        # Test with custom manager
        response = custom_manager.invoke_with_fallback(
            "What is the capital of France?",
            max_retries=5
        )
        print(f"✅ Custom manager response from {response.provider}: {response.content}")

    except Exception as e:
        print(f"❌ Custom configuration failed: {e}")


def example_backward_compatibility():
    """Example 7: Backward compatibility with existing code."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Backward Compatibility")
    print("="*60)

    # Import legacy functions
    from llms import get_available_llm, invoke_with_fallback, create_groq_llm

    try:
        # Old way of getting LLM
        llm = get_available_llm()
        if llm:
            print(f"✅ Legacy get_available_llm() works: {type(llm).__name__}")

        # Old way of invoking with fallback
        response = invoke_with_fallback("Hello, world!")
        print(f"✅ Legacy invoke_with_fallback() works: {response.content[:50]}...")

    except Exception as e:
        print(f"❌ Backward compatibility failed: {e}")


def run_all_examples():
    """Run all usage examples."""
    print("🚀 MULTI-LLM PROVIDER USAGE EXAMPLES")
    print("="*80)

    if not llm_manager:
        print("❌ LLM Manager not available. Please check your configuration and API keys.")
        return

    examples = [
        example_basic_usage,
        example_specialized_routing,
        example_health_monitoring,
        example_configuration_management,
        example_error_handling,
        example_custom_configuration,
        example_backward_compatibility,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example {example_func.__name__} failed: {e}")

    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("💡 Check the individual examples above for specific usage patterns.")


if __name__ == "__main__":
    run_all_examples()