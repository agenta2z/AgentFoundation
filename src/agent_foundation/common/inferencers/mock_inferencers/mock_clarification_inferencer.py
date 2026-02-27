"""
Mock clarification inferencer for testing clarification question flows.

This module provides a mock inferencer that demonstrates how agents handle
clarification questions, using a grocery order example.
"""
from typing import Any
from agent_foundation.agents.agent_response import AgentResponse, AgentAction


class MockClarificationInferencer:
    """
    Mock inferencer that demonstrates clarification question flow.

    This inferencer simulates an agent that:
    1. On first call: Returns a clarification question asking for delivery location
       and grocery service preference
    2. On subsequent calls: Returns a completion response

    This is useful for testing and debugging how the UI handles clarification flows,
    including the display of HTML-formatted clarification questions and the
    Response Monitor's structured data display.

    Example:
        >>> inferencer = MockClarificationInferencer()
        >>>
        >>> # First call - returns clarification
        >>> response1 = inferencer("Order pasta, sauce, and garlic bread")
        >>> print(response1.instant_response)
        "I'll help you place an online grocery order..."
        >>> print(response1.next_actions[0][0].type)
        "Clarification.MissingInformation"
        >>>
        >>> # Second call - returns completion
        >>> response2 = inferencer("Seattle, WA 98121, use Instacart")
        >>> print(response2.instant_response)
        "Thank you for providing that information!..."
        >>> print(response2.next_actions)
        []
    """

    def __init__(self):
        """Initialize the mock inferencer."""
        self.call_count = 0

    def __call__(self, reasoner_input: Any, reasoner_config: Any = None) -> AgentResponse:
        """
        Process input and return appropriate response based on call count.

        Args:
            reasoner_input: The user's input/request
            reasoner_config: Optional configuration (not used in this mock)

        Returns:
            AgentResponse with instant_response and next_actions
        """
        return """<StructuredResponse>
<NewTask>true</NewTask>
<TaskStatus>Ongoing</TaskStatus>
<InstantResponse>I'll help you place an online grocery order for pasta, pasta sauce, and garlic bread. To proceed with the order, I need some information from you.</InstantResponse>
<InstantLearnings></InstantLearnings>
<ImmediateNextActions>
<Action>
<Reasoning>To place an online grocery order with delivery, I absolutely need the user's delivery location and preferred grocery service. Without this information, I cannot determine which services are available, search for the items, or complete the checkout process. This is essential missing information that prevents me from proceeding with the task.</Reasoning>
<Target>To place your grocery order, I need to know:
<ul>
<li><strong>Delivery location:</strong> What's your delivery address or zip code?</li>
<li><strong>Preferred grocery service:</strong> Do you have a preferred online grocery service? Popular options include:
  <ul>
    <li>Instacart (shops from multiple local stores)</li>
    <li>Amazon Fresh</li>
    <li>Walmart Grocery</li>
    <li>Target</li>
    <li>Local grocery chains in your area</li>
  </ul>
</li>
</ul>
If you don't have a preference, I can suggest available options once I know your location.</Target>
<Type>Clarification.MissingInformation</Type>
</Action>
</ImmediateNextActions>
<PlannedActions>Once I have the delivery location and preferred grocery service, I'll access that service's website, search for pasta, pasta sauce, and garlic bread, add them to the cart, and proceed with the order.</PlannedActions>
</StructuredResponse>"""

    def reset(self):
        """Reset the inferencer state to start fresh."""
        self.call_count = 0
