# Actor Contract

This document defines the contract that actors must satisfy when used with the Agent framework.

## Overview

An **actor** is a callable that the Agent invokes to execute actions and return results. Actors are registered via `MultiActionExecutor`, which maps action types to specific actor implementations (e.g., `'default'` → WebDriver).

The Agent framework defines a set of **standard action types** that actors should support. Actors may support additional action types beyond this contract.

## Standard Action Types

### `no_op`

**Purpose**: No actual operation; may capture latest state in action results.

**Contract**:
- The actor MUST NOT perform any side effects (no clicks, navigation, input, waiting, etc.)
- The actor SHOULD return a result that captures the current state, using the same result pipeline as regular actions
- The actor MUST NOT require `action_target` or `action_args`

**When it's used**:
- After user copilot interaction (`needs_user_copilot=True`), the Agent calls `no_op` on the default actor to capture fresh state. This replaces stale action results that were captured before the user interacted directly with the system (e.g., manually completing authentication in a browser).

**Implementation example (WebDriver)**:
- Falls through to the HTML capture pipeline — calls `add_unique_index_to_elements()`, `get_body_html()`, and `_clean_html_for_action_results()`
- Returns a `WebDriverActionResult` with fresh `cleaned_body_html_after_last_action`
- No `sleep()`, no element lookup, no browser action

## Result Contract

Actor results are consumed by the Agent's prompt construction pipeline:
- `_get_action_result_string()` calls `str()` on each result item
- The string representation is embedded in the reasoner's prompt as `<ActionResult>` XML

This means `__str__()` on the result object determines what the reasoner sees. For WebDriver, `WebDriverActionResult.__str__()` returns `cleaned_body_html_after_last_action` — the page HTML with `__id__` attributes that the reasoner uses to reference elements.

## MultiActionExecutor Resolution

When the Agent needs to call a specific action type:
1. `self.actor.resolve(action_type)` looks up the actor in `callable_mapping`
2. If `action_type` is not found, falls back to `self.actor.default_key` (typically `'default'`)
3. The resolved actor is called with `action_type=...` and other kwargs

For `no_op`, the resolution typically falls back to the default actor since `no_op` is not explicitly mapped — the default actor (e.g., WebDriver) handles it.
