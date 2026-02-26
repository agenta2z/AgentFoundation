#!/usr/bin/env python3
"""
Example usage of ai_gateway_claude_llm module.

Prerequisites:
1. atlas plugin install -n slauth
2. atlas slauth server --port 5000
3. export AI_GATEWAY_USER_ID="your_staff_id"
4. pip install atlassian-ai-gateway-sdk atlassian_ml_studio_sdk
"""

from science_modeling_tools.apis.ai_gateway.ai_gateway_claude_llm import (
    generate_text,
    AIGatewayClaudeModels
)

def example_basic():
    """Basic text generation example."""
    print("\n" + "="*60)
    print("Example 1: Basic Text Generation")
    print("="*60)
    
    response = generate_text(
        prompt_or_messages="What is the capital of France? Answer in one sentence.",
        model=AIGatewayClaudeModels.CLAUDE_45_SONNET,
        max_new_tokens=100,
        temperature=0.7,
        verbose=True
    )
    
    print(f"\nResponse: {response}")


def example_multi_turn():
    """Multi-turn conversation example."""
    print("\n" + "="*60)
    print("Example 2: Multi-turn Conversation")
    print("="*60)
    
    conversation = [
        "Hello, I need help with Python",
        "Sure! What would you like to know?",
        "How do I read a CSV file?"
    ]
    
    response = generate_text(
        prompt_or_messages=conversation,
        model=AIGatewayClaudeModels.CLAUDE_45_SONNET,
        max_new_tokens=300,
        verbose=True
    )
    
    print(f"\nResponse: {response}")


def example_with_system_prompt():
    """Example with system prompt."""
    print("\n" + "="*60)
    print("Example 3: With System Prompt")
    print("="*60)
    
    response = generate_text(
        prompt_or_messages="Write a haiku about coding",
        model=AIGatewayClaudeModels.CLAUDE_45_SONNET,
        system="You are a creative poet who loves technology.",
        temperature=0.9,
        max_new_tokens=200,
        verbose=True
    )
    
    print(f"\nResponse: {response}")


def example_different_models():
    """Compare different Claude models."""
    print("\n" + "="*60)
    print("Example 4: Different Models Comparison")
    print("="*60)
    
    prompt = "Explain what AI Gateway does in one sentence."
    
    models_to_test = [
        (AIGatewayClaudeModels.CLAUDE_45_SONNET, "Claude Sonnet 4.5"),
        (AIGatewayClaudeModels.CLAUDE_45_HAIKU, "Claude Haiku 4.5"),
    ]
    
    for model, name in models_to_test:
        print(f"\n--- {name} ---")
        response = generate_text(
            prompt_or_messages=prompt,
            model=model,
            max_new_tokens=150,
            temperature=0.7
        )
        print(f"Response: {response}")


def example_with_stop_sequences():
    """Example with stop sequences."""
    print("\n" + "="*60)
    print("Example 5: With Stop Sequences")
    print("="*60)
    
    response = generate_text(
        prompt_or_messages="List the first 5 programming languages:\n1.",
        model=AIGatewayClaudeModels.CLAUDE_45_SONNET,
        stop=["6.", "\n\n"],
        max_new_tokens=200,
        verbose=True
    )
    
    print(f"\nResponse:\n1.{response}")


if __name__ == '__main__':
    import sys
    import os
    
    # Show user_id that will be used
    user_id = os.environ.get('AI_GATEWAY_USER_ID') or os.environ.get('USER')
    print(f"Using user_id: {user_id}")
    print("(Set AI_GATEWAY_USER_ID environment variable to override)")
    print()
    
    # Check if SLAUTH server is running
    print("Make sure SLAUTH server is running:")
    print("  atlas slauth server --port 5000")
    print("\nPress Enter to continue or Ctrl+C to exit...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    
    try:
        example_basic()
        example_multi_turn()
        example_with_system_prompt()
        example_different_models()
        example_with_stop_sequences()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
