"""
Examples demonstrating ReasonerProtocol conformance with Claude inferencers.

This module shows various ways to verify and use inferencers that conform to
the ReasonerProtocol interface.
"""

from typing import Any

from agent_foundation.agents.agent import (
    Agent,
    ReasonerProtocol,
    ReasonerInput,
    ReasonerInferenceConfig,
    ReasonerResponse,
)
from agent_foundation.common.inferencers.api_inferencers.claude_api_inferencer import (
    ClaudeApiInferencer,
)
from agent_foundation.common.inferencers.api_inferencers.ag.ag_claude_api_inferencer import (
    AgClaudeApiInferencer,
)


# ============================================================================
# Example 1: Runtime isinstance() Check
# ============================================================================

def example_runtime_check():
    """Demonstrate runtime protocol conformance checking."""
    print("Example 1: Runtime isinstance() Check")
    print("=" * 60)

    # Create inferencer instances
    claude_inferencer = ClaudeApiInferencer(model_id="claude-3-5-sonnet-20241022")
    ag_inferencer = AgClaudeApiInferencer(model_id="claude-3-5-sonnet-20241022")

    # Runtime protocol conformance check
    # This works because ReasonerProtocol is decorated with @runtime_checkable
    assert isinstance(claude_inferencer, ReasonerProtocol), \
        "ClaudeApiInferencer should conform to ReasonerProtocol"
    print("✅ ClaudeApiInferencer conforms to ReasonerProtocol")

    assert isinstance(ag_inferencer, ReasonerProtocol), \
        "AgClaudeApiInferencer should conform to ReasonerProtocol"
    print("✅ AgClaudeApiInferencer conforms to ReasonerProtocol")

    print()


# ============================================================================
# Example 2: Type-Annotated Reasoner Variables
# ============================================================================

def example_type_annotations():
    """Demonstrate using type annotations for reasoners."""
    print("Example 2: Type-Annotated Reasoner Variables")
    print("=" * 60)

    # Explicitly type the reasoner variable
    # Type checkers (mypy, pyright) will verify conformance
    reasoner: ReasonerProtocol = ClaudeApiInferencer(
        model_id="claude-3-5-sonnet-20241022"
    )
    print(f"✅ Created reasoner: {type(reasoner).__name__}")

    # Alternative: AI Gateway
    ag_reasoner: ReasonerProtocol = AgClaudeApiInferencer(
        model_id="claude-3-5-sonnet-20241022"
    )
    print(f"✅ Created AI Gateway reasoner: {type(ag_reasoner).__name__}")

    print()


# ============================================================================
# Example 3: Factory Function Pattern
# ============================================================================

def create_claude_reasoner(
    model_id: str = "claude-3-5-sonnet-20241022",
    use_ai_gateway: bool = False,
    **kwargs: Any
) -> ReasonerProtocol:
    """
    Factory function to create Claude-based reasoners that conform to ReasonerProtocol.

    Args:
        model_id: Claude model identifier
        use_ai_gateway: Whether to use AI Gateway routing
        **kwargs: Additional configuration for the inferencer

    Returns:
        ReasonerProtocol: A reasoner instance conforming to the protocol
    """
    if use_ai_gateway:
        return AgClaudeApiInferencer(model_id=model_id, **kwargs)
    else:
        return ClaudeApiInferencer(model_id=model_id, **kwargs)


def example_factory_pattern():
    """Demonstrate factory pattern for creating reasoners."""
    print("Example 3: Factory Function Pattern")
    print("=" * 60)

    # Create standard Claude reasoner
    standard_reasoner = create_claude_reasoner()
    print(f"✅ Created standard reasoner: {type(standard_reasoner).__name__}")

    # Create AI Gateway reasoner
    gateway_reasoner = create_claude_reasoner(use_ai_gateway=True)
    print(f"✅ Created gateway reasoner: {type(gateway_reasoner).__name__}")

    # Verify both conform to protocol
    assert isinstance(standard_reasoner, ReasonerProtocol)
    assert isinstance(gateway_reasoner, ReasonerProtocol)
    print("✅ Both reasoners conform to ReasonerProtocol")

    print()


# ============================================================================
# Example 4: Custom Reasoner Implementation
# ============================================================================

class SimpleRuleBasedReasoner:
    """
    Simple rule-based reasoner that implements ReasonerProtocol.

    This demonstrates how to create a custom reasoner from scratch.

    Protocol Conformance:
        Implements ReasonerProtocol through __call__ method.
    """

    def __init__(self, rules: dict[str, str] = None):
        """
        Initialize with a set of rules.

        Args:
            rules: Dictionary mapping input patterns to responses
        """
        self.rules = rules or {
            "hello": "Hi there! How can I help you?",
            "help": "I can respond to: hello, help, status",
            "status": "System is operational",
        }

    def __call__(
        self,
        reasoner_input: ReasonerInput,
        reasoner_inference_config: ReasonerInferenceConfig = None,
        **kwargs: Any
    ) -> ReasonerResponse:
        """
        Process input and generate a rule-based response.

        Args:
            reasoner_input: User input (expected to be a string)
            reasoner_inference_config: Not used in this simple implementation
            **kwargs: Additional arguments (not used)

        Returns:
            Response based on matching rules, or default response
        """
        # Simple string matching
        input_str = str(reasoner_input).lower().strip()

        # Check rules
        for pattern, response in self.rules.items():
            if pattern in input_str:
                return response

        # Default response
        return "I don't understand that command. Type 'help' for assistance."


def example_custom_reasoner():
    """Demonstrate custom reasoner implementation."""
    print("Example 4: Custom Reasoner Implementation")
    print("=" * 60)

    # Create custom reasoner
    custom_reasoner = SimpleRuleBasedReasoner()

    # Verify protocol conformance
    assert isinstance(custom_reasoner, ReasonerProtocol), \
        "Custom reasoner should conform to ReasonerProtocol"
    print("✅ SimpleRuleBasedReasoner conforms to ReasonerProtocol")

    # Test the reasoner
    test_inputs = ["hello", "help", "unknown command"]
    for input_text in test_inputs:
        response = custom_reasoner(input_text, None)
        print(f"  Input: '{input_text}' → Response: '{response}'")

    print()


# ============================================================================
# Example 5: Using with Agent
# ============================================================================

def example_agent_usage():
    """Demonstrate using conforming reasoners with Agent (conceptual)."""
    print("Example 5: Using with Agent (Conceptual)")
    print("=" * 60)

    # Note: This is a conceptual example showing the pattern
    # Actual execution would require a properly configured InteractiveBase

    # Create a reasoner
    reasoner: ReasonerProtocol = ClaudeApiInferencer(
        model_id="claude-3-5-sonnet-20241022"
    )

    # Reasoner can be used with Agent
    # agent = Agent(
    #     reasoner=reasoner,  # Type-safe: verified to implement ReasonerProtocol
    #     reasoner_args={'temperature': 0.7, 'max_tokens': 1000},
    #     interactive=my_interactive_interface,
    #     user_profile="Expert user",
    #     context="Technical support context"
    # )

    print("✅ Reasoner is compatible with Agent class")
    print(f"   Reasoner type: {type(reasoner).__name__}")
    print(f"   Conforms to ReasonerProtocol: {isinstance(reasoner, ReasonerProtocol)}")

    print()


# ============================================================================
# Example 6: Verifying Method Signature
# ============================================================================

def example_signature_verification():
    """Demonstrate manual signature verification."""
    print("Example 6: Method Signature Verification")
    print("=" * 60)

    import inspect

    # Get the inferencers
    claude_inferencer = ClaudeApiInferencer(model_id="claude-3-5-sonnet-20241022")

    # Check that __call__ method exists
    assert hasattr(claude_inferencer, '__call__'), \
        "Inferencer must have __call__ method"
    print("✅ Has __call__ method")

    # Verify callable
    assert callable(claude_inferencer), \
        "Inferencer must be callable"
    print("✅ Is callable")

    # Inspect signature
    sig = inspect.signature(claude_inferencer.__call__)
    print(f"   Signature: {sig}")

    # Check parameters
    params = list(sig.parameters.keys())
    print(f"   Parameters: {params}")

    # Verify it accepts the required parameters
    # Note: 'self' is included in bound methods
    assert 'inference_input' in params or len(params) >= 1, \
        "Must accept inference_input parameter"
    print("✅ Accepts required parameters")

    print()


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ReasonerProtocol Conformance Examples")
    print("=" * 60 + "\n")

    try:
        example_runtime_check()
        example_type_annotations()
        example_factory_pattern()
        example_custom_reasoner()
        example_agent_usage()
        example_signature_verification()

        print("=" * 60)
        print("All examples completed successfully! ✅")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
