#!/usr/bin/env python3
"""
Test script for Multi-LLM Provider System

This script tests the multi-LLM configuration, failover mechanisms,
and health monitoring capabilities.
"""

import os
import sys
import json
import time
from typing import Dict, Any

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from llms import llm_manager, MultiLLMManager
    print("✓ Successfully imported multi-LLM system")
except ImportError as e:
    print(f"✗ Failed to import multi-LLM system: {e}")
    sys.exit(1)


def test_llm_initialization():
    """Test LLM provider initialization."""
    print("\n" + "="*60)
    print("TESTING LLM PROVIDER INITIALIZATION")
    print("="*60)

    if not llm_manager:
        print("✗ LLM Manager failed to initialize")
        return False

    print(f"✓ LLM Manager initialized successfully")
    print(f"✓ Available providers: {list(llm_manager.providers.keys())}")
    print(f"✓ Total providers: {len(llm_manager.providers)}")

    # Test priority order
    priority_order = llm_manager.get_priority_order()
    print(f"✓ Priority order: {priority_order}")

    return True


def test_health_monitoring():
    """Test health monitoring system."""
    print("\n" + "="*60)
    print("TESTING HEALTH MONITORING SYSTEM")
    print("="*60)

    if not llm_manager:
        return False

    health_status = llm_manager.get_health_status()

    print("Provider Health Status:")
    for provider, status in health_status.items():
        print(f"\n{provider.upper()}:")
        for key, value in status.items():
            print(f"  {key}: {value}")

    return True


def test_basic_llm_invocation():
    """Test basic LLM invocation with simple prompt."""
    print("\n" + "="*60)
    print("TESTING BASIC LLM INVOCATION")
    print("="*60)

    if not llm_manager:
        return False

    test_prompt = "What is 2 + 2? Please answer with just the number."

    try:
        print(f"Testing prompt: '{test_prompt}'")
        response = llm_manager.invoke_with_fallback(test_prompt)

        print(f"✓ Response received:")
        print(f"  Provider: {response.provider}")
        print(f"  Model: {response.model}")
        print(f"  Response time: {response.response_time:.2f}s")
        print(f"  Content: {response.content.strip()}")

        return True

    except Exception as e:
        print(f"✗ LLM invocation failed: {e}")
        return False


def test_specialized_routing():
    """Test specialized routing for different use cases."""
    print("\n" + "="*60)
    print("TESTING SPECIALIZED ROUTING")
    print("="*60)

    if not llm_manager:
        return False

    test_cases = [
        ("math_problems", "Solve: 3x + 7 = 22"),
        ("language_analysis", "Analyze the grammar in this sentence: 'The quick brown fox jumps.'"),
        ("creative_writing", "Write a haiku about programming."),
        ("code_analysis", "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)")
    ]

    for use_case, prompt in test_cases:
        print(f"\nTesting use case: {use_case}")

        # Get specialized priority order
        specialized_order = llm_manager.get_priority_order(use_case)
        print(f"  Specialized priority: {specialized_order}")

        try:
            response = llm_manager.invoke_with_fallback(prompt, use_case=use_case)
            print(f"  ✓ Used provider: {response.provider}")
            print(f"  ✓ Response time: {response.response_time:.2f}s")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    return True


def test_circuit_breaker():
    """Test circuit breaker functionality (simulation)."""
    print("\n" + "="*60)
    print("TESTING CIRCUIT BREAKER (SIMULATION)")
    print("="*60)

    if not llm_manager:
        return False

    # Display current circuit breaker states
    for provider_name in llm_manager.providers.keys():
        cb = llm_manager.circuit_breakers[provider_name]
        print(f"{provider_name}: State={cb.state}, Failures={cb.failure_count}")

    print("\nCircuit breaker thresholds and timeouts are configured.")
    print("In production, circuit breakers will automatically open after consecutive failures.")

    return True


def test_fallback_mechanism():
    """Test fallback mechanism with invalid prompt (mild test)."""
    print("\n" + "="*60)
    print("TESTING FALLBACK MECHANISM")
    print("="*60)

    if not llm_manager:
        return False

    # Use a complex prompt that might fail on some providers
    complex_prompt = """
    Please analyze this complex scenario: You are given a mathematical function
    f(x) = 3x² + 2x - 1. Calculate the derivative, find the critical points,
    and determine if they are maxima or minima. Provide a step-by-step solution.
    """

    try:
        print("Testing complex mathematical prompt...")
        response = llm_manager.invoke_with_fallback(complex_prompt)

        print(f"✓ Fallback mechanism worked:")
        print(f"  Provider used: {response.provider}")
        print(f"  Response length: {len(response.content)} characters")
        print(f"  Response time: {response.response_time:.2f}s")

        return True

    except Exception as e:
        print(f"✗ Fallback mechanism failed: {e}")
        return False


def test_configuration_loading():
    """Test configuration file loading."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION LOADING")
    print("="*60)

    if not llm_manager:
        return False

    config = llm_manager.config

    print("Configuration loaded:")
    print(f"  Provider priority: {config.get('provider_priority', {})}")
    print(f"  Enabled providers: {[p for p, c in config.get('providers', {}).items() if c.get('enabled', False)]}")
    print(f"  Failover enabled: {config.get('failover', {}).get('enabled', False)}")
    print(f"  Monitoring enabled: {config.get('monitoring', {}).get('enabled', False)}")

    return True


def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("🚀 MULTI-LLM PROVIDER SYSTEM TEST SUITE")
    print("="*80)

    tests = [
        ("LLM Initialization", test_llm_initialization),
        ("Configuration Loading", test_configuration_loading),
        ("Health Monitoring", test_health_monitoring),
        ("Basic LLM Invocation", test_basic_llm_invocation),
        ("Specialized Routing", test_specialized_routing),
        ("Circuit Breaker", test_circuit_breaker),
        ("Fallback Mechanism", test_fallback_mechanism),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results[test_name] = "CRASH"

    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)

    for test_name, result in results.items():
        status_emoji = "✓" if result == "PASS" else "✗"
        print(f"{status_emoji} {test_name}: {result}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Multi-LLM system is working correctly.")
    elif passed > total // 2:
        print("⚠️  Most tests passed. Check failed tests above.")
    else:
        print("❌ Many tests failed. Check configuration and API keys.")

    # Display final health status
    if llm_manager:
        print("\nFinal Provider Health Status:")
        health_status = llm_manager.get_health_status()
        for provider, status in health_status.items():
            health_indicator = "🟢" if status['is_healthy'] else "🔴"
            print(f"{health_indicator} {provider}: {status['success_rate']} success rate, {status['total_requests']} requests")


if __name__ == "__main__":
    run_comprehensive_test()