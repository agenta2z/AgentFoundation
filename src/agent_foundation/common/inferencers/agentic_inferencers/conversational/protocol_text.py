"""LLM-facing protocol text shared by the conversational orchestrators.

These strings appear verbatim in conversation history and are part of the
contract the model reads; changing them may affect prompt comprehension.
"""

WIDGET_RESPONSE_PREFIX = "[Collected from conversation widget]"
TOOL_RESULT_HEADER = "[Tool Result: {}]"  # .format(tool_name)
TOOL_RESULTS_PREFIX = "[Tool execution results]"
CONTINUE_AFTER_TOOLS = "Continue based on the tool execution results above."
