"""
Unit tests for callable resolution and feed conflict resolution in PromptBasedAgent.

Tests the callable resolution logic that resolves callables in
additional_reasoner_input_feed by calling them with user_input.
Callable returning string is stored under its key, callable returning dict
merges all keys into feed, and static values pass through unchanged.

Also tests FeedConflictResolution strategies (ATTRIBUTE_ONLY, FEED_ONLY, MERGE)
for resolving conflicts when knowledge dict keys overlap with prompt feed keys.

Requirements: 4.1, 4.2, 4.3
"""
import sys
from pathlib import Path

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
_spu_src = Path(__file__).resolve().parents[3] / "SciencePythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

import pytest


def resolve_callables(additional_reasoner_input_feed: dict, user_input) -> dict:
    """Extract the callable resolution logic from PromptBasedAgent._construct_reasoner_input.

    This mirrors the exact logic in _construct_reasoner_input (lines ~385-393):
        resolved_extra = {}
        for k, v in self.additional_reasoner_input_feed.items():
            if callable(v):
                result = v(user_input)
                if isinstance(result, dict):
                    resolved_extra.update(result)
                else:
                    resolved_extra[k] = result
            else:
                resolved_extra[k] = v

    Args:
        additional_reasoner_input_feed: The feed dict with possible callables.
        user_input: The user input to pass to callables.

    Returns:
        The resolved dict with all callables evaluated.
    """
    resolved_extra = {}
    for k, v in additional_reasoner_input_feed.items():
        if callable(v):
            result = v(user_input)
            if isinstance(result, dict):
                resolved_extra.update(result)
            else:
                resolved_extra[k] = result
        else:
            resolved_extra[k] = v
    return resolved_extra


class TestCallableReturningString:
    """Test that a callable returning a string is stored under its key.

    Requirements: 4.1
    """

    def test_callable_returning_string(self):
        """A callable that returns a string should be stored under its original key."""
        feed = {
            "greeting": lambda user_input: f"Hello, {user_input}!"
        }
        result = resolve_callables(feed, "Alice")
        assert result == {"greeting": "Hello, Alice!"}

    def test_callable_receives_user_input(self):
        """The callable should receive the user_input as its argument."""
        received_args = []

        def capture_input(user_input):
            received_args.append(user_input)
            return "captured"

        feed = {"key": capture_input}
        resolve_callables(feed, "test_query")
        assert received_args == ["test_query"]

    def test_callable_returning_empty_string(self):
        """A callable returning an empty string should store empty string under its key."""
        feed = {"empty": lambda user_input: ""}
        result = resolve_callables(feed, "anything")
        assert result == {"empty": ""}


class TestCallableReturningDict:
    """Test that a callable returning a dict merges all keys into the feed.

    Requirements: 4.2
    """

    def test_callable_returning_dict(self):
        """A callable that returns a dict should merge all returned keys into the feed."""
        feed = {
            "_knowledge": lambda user_input: {
                "user_profile": "Name: Alice",
                "instructions": "Be helpful",
            }
        }
        result = resolve_callables(feed, "query")
        assert "user_profile" in result
        assert result["user_profile"] == "Name: Alice"
        assert "instructions" in result
        assert result["instructions"] == "Be helpful"
        # The original key "_knowledge" should NOT be in the result
        assert "_knowledge" not in result

    def test_callable_returning_single_key_dict(self):
        """A callable returning a dict with one key should merge that key."""
        feed = {
            "provider": lambda user_input: {"context": "Some context"}
        }
        result = resolve_callables(feed, "query")
        assert result == {"context": "Some context"}
        assert "provider" not in result

    def test_callable_returning_empty_dict(self):
        """A callable returning an empty dict should not add any keys."""
        feed = {"_knowledge": lambda user_input: {}}
        result = resolve_callables(feed, "query")
        assert result == {}


class TestStaticValuesPassThrough:
    """Test that non-callable values pass through unchanged.

    Requirements: 4.3
    """

    def test_static_string_passes_through(self):
        """A static string value should pass through unchanged."""
        feed = {"role": "assistant"}
        result = resolve_callables(feed, "query")
        assert result == {"role": "assistant"}

    def test_static_none_passes_through(self):
        """A static None value should pass through unchanged."""
        feed = {"optional_field": None}
        result = resolve_callables(feed, "query")
        assert result == {"optional_field": None}

    def test_static_int_passes_through(self):
        """A static integer value should pass through unchanged."""
        feed = {"max_tokens": 1000}
        result = resolve_callables(feed, "query")
        assert result == {"max_tokens": 1000}

    def test_static_list_passes_through(self):
        """A static list value should pass through unchanged."""
        feed = {"tags": ["a", "b", "c"]}
        result = resolve_callables(feed, "query")
        assert result == {"tags": ["a", "b", "c"]}


class TestMixedCallablesAndStatic:
    """Test that a mix of callables and static values works correctly.

    Requirements: 4.1, 4.2, 4.3
    """

    def test_mixed_callables_and_static(self):
        """A feed with both callables and static values should resolve correctly."""
        feed = {
            "static_key": "static_value",
            "string_callable": lambda user_input: f"dynamic: {user_input}",
            "_knowledge": lambda user_input: {
                "user_profile": "Profile data",
                "instructions": "Do this",
            },
            "another_static": 42,
        }
        result = resolve_callables(feed, "my query")

        # Static values pass through
        assert result["static_key"] == "static_value"
        assert result["another_static"] == 42

        # String-returning callable stored under its key
        assert result["string_callable"] == "dynamic: my query"

        # Dict-returning callable merges keys (original key not present)
        assert result["user_profile"] == "Profile data"
        assert result["instructions"] == "Do this"
        assert "_knowledge" not in result

    def test_empty_feed(self):
        """An empty feed should return an empty dict."""
        result = resolve_callables({}, "query")
        assert result == {}


class TestDictCallableOverridesKey:
    """Test that dict-returning callable keys replace the original key.

    When a dict-returning callable is stored under key '_knowledge',
    the returned dict keys (e.g., 'user_profile', 'instructions') are
    what appear in the feed, NOT '_knowledge'.

    Requirements: 4.2
    """

    def test_dict_callable_overrides_key(self):
        """Dict-returning callable under '_knowledge' should produce returned dict keys, not '_knowledge'."""
        feed = {
            "_knowledge": lambda user_input: {
                "user_profile": "Tony Chen, Seattle",
                "instructions": "Follow the grocery procedure",
                "context": "Shopping session",
            }
        }
        result = resolve_callables(feed, "buy groceries at safeway")

        # The returned dict keys should be present
        assert "user_profile" in result
        assert "instructions" in result
        assert "context" in result

        # The original key should NOT be present
        assert "_knowledge" not in result

        # Values should match what the callable returned
        assert result["user_profile"] == "Tony Chen, Seattle"
        assert result["instructions"] == "Follow the grocery procedure"
        assert result["context"] == "Shopping session"

    def test_dict_callable_keys_can_overlap_with_static(self):
        """If a dict-returning callable returns a key that also exists as static, the last one wins."""
        # In the resolution loop, items are processed in insertion order.
        # If a static key comes before a dict-returning callable that returns the same key,
        # the callable's value overwrites the static one.
        feed = {
            "user_profile": "static profile",
            "_knowledge": lambda user_input: {"user_profile": "dynamic profile"},
        }
        result = resolve_callables(feed, "query")
        # The dict-returning callable processes after the static key,
        # so its value overwrites
        assert result["user_profile"] == "dynamic profile"
        assert "_knowledge" not in result


from agent_foundation.agents.prompt_based_agents.prompt_based_agent import (
    FeedConflictResolution,
)


def merge_into_feed(feed: dict, extra: dict, strategy: FeedConflictResolution) -> dict:
    """Standalone version of PromptBasedAgent._merge_into_feed for testing.

    Mirrors the exact merge logic used to combine knowledge dict / additional
    feed into the base prompt feed.
    """
    for k, v in extra.items():
        if k in feed and feed[k]:
            if strategy == FeedConflictResolution.ATTRIBUTE_ONLY:
                pass  # keep existing value
            elif strategy == FeedConflictResolution.MERGE:
                feed[k] = f"{feed[k]}\n\n{v}"
            else:  # FEED_ONLY (default)
                feed[k] = v
        else:
            feed[k] = v
    return feed


class TestFeedConflictResolutionEnum:
    """Test FeedConflictResolution enum values."""

    def test_enum_values(self):
        assert FeedConflictResolution.ATTRIBUTE_ONLY == 'attribute_only'
        assert FeedConflictResolution.FEED_ONLY == 'feed_only'
        assert FeedConflictResolution.MERGE == 'merge'

    def test_enum_is_strenum(self):
        assert isinstance(FeedConflictResolution.FEED_ONLY, str)


class TestFeedOnlyResolution:
    """Test FEED_ONLY strategy: extra value overwrites existing.

    Requirements: 4.1
    """

    def test_overwrites_existing_key(self):
        feed = {"user_profile": "static profile"}
        merge_into_feed(feed, {"user_profile": "dynamic profile"}, FeedConflictResolution.FEED_ONLY)
        assert feed["user_profile"] == "dynamic profile"

    def test_adds_new_key(self):
        feed = {"role": "assistant"}
        merge_into_feed(feed, {"instructions": "Be helpful"}, FeedConflictResolution.FEED_ONLY)
        assert feed["instructions"] == "Be helpful"
        assert feed["role"] == "assistant"

    def test_overwrites_none_value(self):
        """When existing value is None/falsy, extra always wins regardless of strategy."""
        feed = {"user_profile": None}
        merge_into_feed(feed, {"user_profile": "from knowledge"}, FeedConflictResolution.FEED_ONLY)
        assert feed["user_profile"] == "from knowledge"


class TestAttributeOnlyResolution:
    """Test ATTRIBUTE_ONLY strategy: existing value kept, extra ignored.

    Requirements: 4.1
    """

    def test_keeps_existing_key(self):
        feed = {"user_profile": "static profile"}
        merge_into_feed(feed, {"user_profile": "dynamic profile"}, FeedConflictResolution.ATTRIBUTE_ONLY)
        assert feed["user_profile"] == "static profile"

    def test_adds_new_key(self):
        """Non-conflicting keys are always added."""
        feed = {"role": "assistant"}
        merge_into_feed(feed, {"instructions": "Be helpful"}, FeedConflictResolution.ATTRIBUTE_ONLY)
        assert feed["instructions"] == "Be helpful"

    def test_adds_when_existing_is_falsy(self):
        """When existing value is None/empty, extra is added."""
        feed = {"user_profile": None}
        merge_into_feed(feed, {"user_profile": "from knowledge"}, FeedConflictResolution.ATTRIBUTE_ONLY)
        assert feed["user_profile"] == "from knowledge"


class TestMergeResolution:
    """Test MERGE strategy: values concatenated with double newline.

    Requirements: 4.1
    """

    def test_merges_conflicting_key(self):
        feed = {"user_profile": "Name: Alice"}
        merge_into_feed(feed, {"user_profile": "Membership: Gold"}, FeedConflictResolution.MERGE)
        assert feed["user_profile"] == "Name: Alice\n\nMembership: Gold"

    def test_adds_new_key(self):
        feed = {"role": "assistant"}
        merge_into_feed(feed, {"instructions": "Be helpful"}, FeedConflictResolution.MERGE)
        assert feed["instructions"] == "Be helpful"

    def test_adds_when_existing_is_falsy(self):
        feed = {"user_profile": ""}
        merge_into_feed(feed, {"user_profile": "from knowledge"}, FeedConflictResolution.MERGE)
        assert feed["user_profile"] == "from knowledge"


class TestMergeWithKnowledgeDict:
    """Test merge behavior with a knowledge dict (typical KnowledgeProvider output).

    Requirements: 4.1, 4.2
    """

    def test_knowledge_dict_merged_feed_only(self):
        feed = {"user_profile": "static", "context": "existing context"}
        knowledge = {"user_profile": "Tony Chen", "instructions": "Follow procedure"}
        merge_into_feed(feed, knowledge, FeedConflictResolution.FEED_ONLY)
        assert feed["user_profile"] == "Tony Chen"
        assert feed["context"] == "existing context"
        assert feed["instructions"] == "Follow procedure"

    def test_knowledge_dict_merged_attribute_only(self):
        feed = {"user_profile": "static", "context": "existing context"}
        knowledge = {"user_profile": "Tony Chen", "instructions": "Follow procedure"}
        merge_into_feed(feed, knowledge, FeedConflictResolution.ATTRIBUTE_ONLY)
        assert feed["user_profile"] == "static"
        assert feed["context"] == "existing context"
        assert feed["instructions"] == "Follow procedure"

    def test_knowledge_dict_merged_merge(self):
        feed = {"user_profile": "static", "context": "existing context"}
        knowledge = {"user_profile": "Tony Chen", "instructions": "Follow procedure"}
        merge_into_feed(feed, knowledge, FeedConflictResolution.MERGE)
        assert feed["user_profile"] == "static\n\nTony Chen"
        assert feed["context"] == "existing context"
        assert feed["instructions"] == "Follow procedure"
