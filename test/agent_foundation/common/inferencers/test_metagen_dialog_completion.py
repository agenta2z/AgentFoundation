

"""Integration tests for dialog_completion mode in MetaGen API.

Tests cover:
- CompletionMode routing (AUTO/DIALOG/CHAT)
- Dialog object construction from various input formats
- Dialog response text extraction
- End-to-end generate_text() with dialog mode (real API)
- End-to-end generate_text_async() with dialog mode (real API)
- MetagenApiInferencer with default Claude Opus 4.6 (real API)
- Multi-turn conversation support

Usage:
    python -m pytest test_metagen_dialog_completion.py -v -k "unit"
    python test_metagen_dialog_completion.py --mode unit
    python test_metagen_dialog_completion.py --mode e2e
    python test_metagen_dialog_completion.py --mode all
"""

import argparse
import asyncio
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Unit Tests (no API calls — test internal logic only)
# ---------------------------------------------------------------------------


def test_resolve_completion_mode():
    """Test that _resolve_completion_mode auto-selects correctly."""
    from agent_foundation.apis.metagen.metagen_llm import (
        _resolve_completion_mode,
        CompletionMode,
    )

    print("\n--- test_resolve_completion_mode ---")

    # AUTO → DIALOG for Claude models
    assert _resolve_completion_mode("claude-4-6-opus-genai", CompletionMode.AUTO) == CompletionMode.DIALOG
    assert _resolve_completion_mode("claude-4-sonnet-genai", CompletionMode.AUTO) == CompletionMode.DIALOG
    assert _resolve_completion_mode("claude-3-7-sonnet-20250219-us", CompletionMode.AUTO) == CompletionMode.DIALOG
    assert _resolve_completion_mode("CLAUDE-4-6-OPUS-GENAI", CompletionMode.AUTO) == CompletionMode.DIALOG

    # AUTO → CHAT for non-Claude models
    assert _resolve_completion_mode("llama3.1-405b-instruct", CompletionMode.AUTO) == CompletionMode.CHAT
    assert _resolve_completion_mode("gpt-5-genai", CompletionMode.AUTO) == CompletionMode.CHAT
    assert _resolve_completion_mode("gemini-2-5-pro", CompletionMode.AUTO) == CompletionMode.CHAT

    # Explicit modes always override AUTO detection
    assert _resolve_completion_mode("claude-4-6-opus-genai", CompletionMode.CHAT) == CompletionMode.CHAT
    assert _resolve_completion_mode("llama3.1-405b-instruct", CompletionMode.DIALOG) == CompletionMode.DIALOG

    print("  ✅ PASS")


def test_build_dialog_string_input():
    """Test _build_dialog with a simple string prompt."""
    from metagen import Dialog, DialogMessage, DialogSource, DialogTextContent
    from agent_foundation.apis.metagen.metagen_llm import _build_dialog

    print("\n--- test_build_dialog_string_input ---")

    dialog = _build_dialog("Hello, world!")

    assert isinstance(dialog, Dialog), f"Expected Dialog, got {type(dialog)}"
    assert len(dialog.messages) == 1, f"Expected 1 message, got {len(dialog.messages)}"

    msg = dialog.messages[0]
    assert msg.source == DialogSource.USER
    assert len(msg.contents) == 1
    assert isinstance(msg.contents[0], DialogTextContent)
    assert msg.contents[0].text == "Hello, world!"

    print("  ✅ PASS")


def test_build_dialog_dict_input():
    """Test _build_dialog with a single message dict."""
    from metagen import DialogSource
    from agent_foundation.apis.metagen.metagen_llm import _build_dialog

    print("\n--- test_build_dialog_dict_input ---")

    dialog = _build_dialog({"role": "system", "content": "You are a helpful assistant."})
    assert len(dialog.messages) == 1
    assert dialog.messages[0].source == DialogSource.SYSTEM
    assert dialog.messages[0].contents[0].text == "You are a helpful assistant."

    dialog = _build_dialog({"role": "user", "content": "Hi"})
    assert dialog.messages[0].source == DialogSource.USER

    dialog = _build_dialog({"role": "assistant", "content": "Hello!"})
    assert dialog.messages[0].source == DialogSource.ASSISTANT

    print("  ✅ PASS")


def test_build_dialog_list_of_strings():
    """Test _build_dialog with alternating user/assistant string pairs."""
    from metagen import DialogSource
    from agent_foundation.apis.metagen.metagen_llm import _build_dialog

    print("\n--- test_build_dialog_list_of_strings ---")

    # Even number: user/assistant pairs
    dialog = _build_dialog(["What is Python?", "Python is a programming language."])
    assert len(dialog.messages) == 2
    assert dialog.messages[0].source == DialogSource.USER
    assert dialog.messages[1].source == DialogSource.ASSISTANT

    # Odd number: last message is user
    dialog = _build_dialog(["Hi", "Hello!", "How are you?"])
    assert len(dialog.messages) == 3
    assert dialog.messages[0].source == DialogSource.USER
    assert dialog.messages[1].source == DialogSource.ASSISTANT
    assert dialog.messages[2].source == DialogSource.USER

    # Single string in list
    dialog = _build_dialog(["Just one message"])
    assert len(dialog.messages) == 1
    assert dialog.messages[0].source == DialogSource.USER

    print("  ✅ PASS")


def test_build_dialog_list_of_dicts():
    """Test _build_dialog with a list of role/content dicts (multi-turn)."""
    from metagen import DialogSource
    from agent_foundation.apis.metagen.metagen_llm import _build_dialog

    print("\n--- test_build_dialog_list_of_dicts ---")

    messages = [
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": "What is a decorator?"},
        {"role": "assistant", "content": "A decorator wraps a function."},
        {"role": "user", "content": "Give an example."},
    ]
    dialog = _build_dialog(messages)

    assert len(dialog.messages) == 4
    assert dialog.messages[0].source == DialogSource.SYSTEM
    assert dialog.messages[1].source == DialogSource.USER
    assert dialog.messages[2].source == DialogSource.ASSISTANT
    assert dialog.messages[3].source == DialogSource.USER
    assert dialog.messages[0].contents[0].text == "You are a Python expert."
    assert dialog.messages[3].contents[0].text == "Give an example."

    print("  ✅ PASS")


def test_build_dialog_invalid_input():
    """Test _build_dialog raises ValueError for invalid input."""
    from agent_foundation.apis.metagen.metagen_llm import _build_dialog

    print("\n--- test_build_dialog_invalid_input ---")

    try:
        _build_dialog(12345)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("  ✅ PASS")


def test_extract_dialog_response_text():
    """Test _extract_dialog_response_text handles various response shapes."""
    from agent_foundation.apis.metagen.metagen_llm import (
        _extract_dialog_response_text,
    )
    from metagen import Dialog, DialogMessage, DialogSource, DialogTextContent

    print("\n--- test_extract_dialog_response_text ---")

    # Simulate a DialogCompletionResponse-like object
    class FakeChoice:
        def __init__(self, dialog=None, text=None):
            self.dialog = dialog
            self.text = text

    class FakeResponse:
        def __init__(self, choices):
            self.choices = choices

    # Case 1: dialog response with DialogTextContent
    assistant_msg = DialogMessage(
        source=DialogSource.ASSISTANT,
        contents=[DialogTextContent(text="Python is great!")],
    )
    inner_dialog = Dialog(messages=[assistant_msg])
    resp = FakeResponse([FakeChoice(dialog=inner_dialog)])
    assert _extract_dialog_response_text(resp) == "Python is great!"

    # Case 2: fallback to .text attribute
    resp2 = FakeResponse([FakeChoice(text="Fallback text")])
    assert _extract_dialog_response_text(resp2) == "Fallback text"

    print("  ✅ PASS")


def test_completion_mode_enum():
    """Test CompletionMode enum values."""
    from agent_foundation.apis.metagen.metagen_llm import CompletionMode

    print("\n--- test_completion_mode_enum ---")

    assert CompletionMode.CHAT == "chat"
    assert CompletionMode.DIALOG == "dialog"
    assert CompletionMode.AUTO == "auto"
    assert CompletionMode("chat") == CompletionMode.CHAT
    assert CompletionMode("dialog") == CompletionMode.DIALOG
    assert CompletionMode("auto") == CompletionMode.AUTO

    print("  ✅ PASS")


def test_metagen_models_enum():
    """Test MetaGenModels includes Claude Opus 4.6."""
    from agent_foundation.apis.metagen.metagen_llm import MetaGenModels

    print("\n--- test_metagen_models_enum ---")

    assert MetaGenModels.CLAUDE_4_6_OPUS == "claude-4-6-opus-genai"
    assert MetaGenModels.CLAUDE_4_SONNET == "claude-4-sonnet-genai"
    assert "claude" in MetaGenModels.CLAUDE_4_6_OPUS.lower()

    print("  ✅ PASS")


def test_default_max_tokens():
    """Test DEFAULT_MAX_TOKENS has correct value for Claude Opus 4.6."""
    from agent_foundation.apis.metagen.metagen_llm import (
        DEFAULT_MAX_TOKENS,
        MetaGenModels,
    )

    print("\n--- test_default_max_tokens ---")

    assert DEFAULT_MAX_TOKENS[str(MetaGenModels.CLAUDE_4_6_OPUS)] == 128000
    assert DEFAULT_MAX_TOKENS[str(MetaGenModels.CLAUDE_4_SONNET)] == 10240

    print("  ✅ PASS")


# ---------------------------------------------------------------------------
# E2E Tests (real MetaGen API calls)
# ---------------------------------------------------------------------------


def test_e2e_generate_text_dialog_mode():
    """E2E: generate_text() with Claude Opus 4.6 in dialog mode."""
    from agent_foundation.apis.metagen import (
        CompletionMode,
        generate_text,
        MetaGenModels,
    )

    print("\n--- test_e2e_generate_text_dialog_mode ---")

    t0 = time.time()
    result = generate_text(
        prompt_or_messages="What is Python? Answer in one sentence.",
        model=MetaGenModels.CLAUDE_4_6_OPUS,
        max_new_tokens=256,
        temperature=0.7,
        completion_mode=CompletionMode.DIALOG,
    )
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.1f}s): {result[:200]}")
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) > 10, f"Response too short: {result!r}"
    assert "python" in result.lower() or "programming" in result.lower(), (
        f"Response doesn't mention Python: {result!r}"
    )
    print("  ✅ PASS")


def test_e2e_generate_text_auto_mode_claude():
    """E2E: generate_text() with AUTO mode auto-selects dialog for Claude."""
    from agent_foundation.apis.metagen import (
        CompletionMode,
        generate_text,
        MetaGenModels,
    )

    print("\n--- test_e2e_generate_text_auto_mode_claude ---")

    t0 = time.time()
    result = generate_text(
        prompt_or_messages="What is 2+2? Just the number.",
        model=MetaGenModels.CLAUDE_4_6_OPUS,
        max_new_tokens=64,
        temperature=0.0,
        completion_mode=CompletionMode.AUTO,
    )
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.1f}s): {result[:100]}")
    assert isinstance(result, str) and len(result) > 0
    assert "4" in result, f"Expected '4' in response: {result!r}"
    print("  ✅ PASS")


def test_e2e_generate_text_multi_turn():
    """E2E: generate_text() with multi-turn conversation in dialog mode."""
    from agent_foundation.apis.metagen import (
        CompletionMode,
        generate_text,
        MetaGenModels,
    )

    print("\n--- test_e2e_generate_text_multi_turn ---")

    messages = [
        {"role": "user", "content": "Remember the number 42."},
        {"role": "assistant", "content": "I'll remember the number 42."},
        {"role": "user", "content": "What number did I ask you to remember? Just the number."},
    ]

    t0 = time.time()
    result = generate_text(
        prompt_or_messages=messages,
        model=MetaGenModels.CLAUDE_4_6_OPUS,
        max_new_tokens=64,
        temperature=0.0,
        completion_mode=CompletionMode.DIALOG,
    )
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.1f}s): {result[:100]}")
    assert isinstance(result, str) and len(result) > 0
    assert "42" in result, f"Expected '42' in response: {result!r}"
    print("  ✅ PASS")


async def test_e2e_generate_text_async_dialog():
    """E2E: generate_text_async() with Claude Opus 4.6 dialog mode."""
    from agent_foundation.apis.metagen import (
        CompletionMode,
        generate_text_async,
        MetaGenModels,
    )

    print("\n--- test_e2e_generate_text_async_dialog ---")

    t0 = time.time()
    result = await generate_text_async(
        prompt_or_messages="What is Rust? Answer in one sentence.",
        model=MetaGenModels.CLAUDE_4_6_OPUS,
        max_new_tokens=256,
        temperature=0.7,
        completion_mode=CompletionMode.DIALOG,
    )
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.1f}s): {result[:200]}")
    assert isinstance(result, str) and len(result) > 10
    print("  ✅ PASS")


async def test_e2e_parallel_async():
    """E2E: Parallel async calls with dialog mode."""
    from agent_foundation.apis.metagen import (
        CompletionMode,
        generate_text_async,
        MetaGenModels,
    )

    print("\n--- test_e2e_parallel_async ---")

    queries = [
        "What is Python? One sentence.",
        "What is Java? One sentence.",
        "What is Rust? One sentence.",
    ]

    t0 = time.time()
    results = await asyncio.gather(*[
        generate_text_async(
            q,
            model=MetaGenModels.CLAUDE_4_6_OPUS,
            max_new_tokens=128,
            temperature=0.7,
            completion_mode=CompletionMode.DIALOG,
        )
        for q in queries
    ])
    elapsed = time.time() - t0

    for i, (q, r) in enumerate(zip(queries, results)):
        print(f"  [{i}] Q: {q}")
        print(f"      A: {r[:120]}")
        assert isinstance(r, str) and len(r) > 5

    print(f"  Total: {elapsed:.1f}s (avg {elapsed/len(queries):.1f}s/query)")
    print("  ✅ PASS")


def test_e2e_inferencer_defaults():
    """E2E: MetagenApiInferencer with default model (Claude Opus 4.6)."""
    from agent_foundation.common.inferencers.api_inferencers.metagen import (
        MetagenApiInferencer,
    )
    from agent_foundation.apis.metagen import MetaGenModels

    print("\n--- test_e2e_inferencer_defaults ---")

    inferencer = MetagenApiInferencer()
    assert str(inferencer.model_id) == str(MetaGenModels.CLAUDE_4_6_OPUS), (
        f"Expected {MetaGenModels.CLAUDE_4_6_OPUS}, got {inferencer.model_id}"
    )
    assert inferencer.secret_key.startswith("mg-api-"), (
        f"Key doesn't start with mg-api-: {inferencer.secret_key}"
    )

    t0 = time.time()
    result = inferencer.infer("What is 3+3? Just the number.")
    elapsed = time.time() - t0

    print(f"  Model:  {inferencer.model_id}")
    print(f"  Key:    {inferencer.secret_key[:20]}...")
    print(f"  Response ({elapsed:.1f}s): {result[:100]}")
    assert isinstance(result, str) and len(result) > 0
    assert "6" in result, f"Expected '6' in response: {result!r}"
    print("  ✅ PASS")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


UNIT_TESTS = [
    test_resolve_completion_mode,
    test_build_dialog_string_input,
    test_build_dialog_dict_input,
    test_build_dialog_list_of_strings,
    test_build_dialog_list_of_dicts,
    test_build_dialog_invalid_input,
    test_extract_dialog_response_text,
    test_completion_mode_enum,
    test_metagen_models_enum,
    test_default_max_tokens,
]

E2E_SYNC_TESTS = [
    test_e2e_generate_text_dialog_mode,
    test_e2e_generate_text_auto_mode_claude,
    test_e2e_generate_text_multi_turn,
    test_e2e_inferencer_defaults,
]

E2E_ASYNC_TESTS = [
    test_e2e_generate_text_async_dialog,
    test_e2e_parallel_async,
]


def run_tests(tests, label):
    print(f"\n{'='*60}")
    print(f"  {label} ({len(tests)} tests)")
    print(f"{'='*60}")

    passed, failed = 0, 0
    for test_fn in tests:
        try:
            if asyncio.iscoroutinefunction(test_fn):
                asyncio.run(test_fn())
            else:
                test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAIL: {test_fn.__name__}")
            traceback.print_exc()

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed


def main():
    parser = argparse.ArgumentParser(description="MetaGen Dialog Completion Tests")
    parser.add_argument(
        "--mode",
        choices=["unit", "e2e", "all"],
        default="all",
        help="Which tests to run (default: all)",
    )
    args = parser.parse_args()

    total_failures = 0

    if args.mode in ("unit", "all"):
        total_failures += run_tests(UNIT_TESTS, "Unit Tests")

    if args.mode in ("e2e", "all"):
        total_failures += run_tests(E2E_SYNC_TESTS, "E2E Sync Tests")
        total_failures += run_tests(E2E_ASYNC_TESTS, "E2E Async Tests")

    print(f"\n{'='*60}")
    if total_failures == 0:
        print("ALL TESTS PASSED ✅")
    else:
        print(f"FAILURES: {total_failures} ❌")
    print(f"{'='*60}")

    sys.exit(1 if total_failures > 0 else 0)


if __name__ == "__main__":
    main()
