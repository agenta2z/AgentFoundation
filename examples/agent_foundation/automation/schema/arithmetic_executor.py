"""
ArithmeticExecutor - A simple executor for arithmetic operations.

This executor maintains an accumulator value that operations modify.
Used for demonstrating ActionGraph execution with verifiable results.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Result:
    """Result object returned by ArithmeticExecutor."""
    success: bool
    value: Any
    error: Optional[Exception] = None


class ArithmeticExecutor:
    """
    Executor for arithmetic operations with verifiable results.

    Maintains an accumulator value and history of operations.
    Supports: set, add, subtract, multiply, divide, power
    """

    def __init__(self):
        self.history: List[Dict] = []
        self.accumulator: float = 0

    def __call__(
        self,
        action_type: str,
        action_target: Optional[str] = None,
        action_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Result:
        action_args = action_args or {}
        value = action_args.get("value")

        try:
            if action_type == "set":
                self.accumulator = float(value)
            elif action_type == "add":
                self.accumulator += float(value)
            elif action_type == "subtract":
                self.accumulator -= float(value)
            elif action_type == "multiply":
                self.accumulator *= float(value)
            elif action_type == "divide":
                self.accumulator /= float(value)
            elif action_type == "power":
                self.accumulator = self.accumulator ** float(value)
            else:
                raise ValueError(f"Unknown action type: {action_type}")

            self.history.append({
                "action_type": action_type,
                "value": value,
                "result": self.accumulator,
            })

            return Result(success=True, value=self.accumulator)

        except Exception as e:
            return Result(success=False, value=self.accumulator, error=e)

    def reset(self):
        """Reset accumulator and history."""
        self.accumulator = 0
        self.history = []
