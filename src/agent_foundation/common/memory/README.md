# Content Memory Module

A generic, action-agnostic system for tracking and memorizing web page content across state changes (scrolling, clicking, navigation, etc.).

## Overview

The ContentMemory module solves the problem of losing page content during dynamic actions:
- **Scroll**: Elements disappear when scrolled out of view (virtual scrolling, lazy loading)
- **Click**: Page state changes, previous content may be lost
- **Navigate**: Moving between pages loses previous page content
- **Modals/Popups**: Content before/after modal opens

This module captures snapshots of page state, tracks elements across changes, and maintains a cumulative record of all seen content.

## Architecture

```
┌─────────────────────────────────┐
│      ContentMemory              │  Main coordinator
│  - capture_snapshot()           │
│  - merge_snapshots()            │
│  - get_cumulative_html()        │
└─────────────────────────────────┘
         │
         ├──> SnapshotCapturer     (Extracts elements from HTML)
         ├──> SnapshotMerger       (Merges snapshots, tracks changes)
         └──> HTMLGenerator        (Generates cumulative HTML)
```

## Key Features

✅ **Action-Agnostic**: Works with any action (scroll, click, navigate, etc.)
✅ **Automatic Tracking**: Identifies new, removed, and persistent elements
✅ **Cumulative HTML**: Generates HTML containing all seen elements
✅ **Visibility Markers**: Marks elements as visible/removed in output
✅ **Statistics**: Tracks element counts, view counts, timestamps
✅ **Memory Management**: Prune old elements to prevent memory bloat

## Quick Start

### Basic Usage

```python
from webaxon.automation.web_driver import WebDriver

# Create WebDriver with memory enabled
driver = WebDriver(enable_content_memory=True)

# Navigate and add element IDs
driver.open_url('https://example.com')
driver.add_unique_index_to_elements()

# Scroll with memory
result = driver.execute_action_with_memory(
    action_func=lambda: driver.execute_single_action(
        element=driver.find_element_by_xpath(tag_name='body'),
        action_type='scroll',
        action_args={'direction': 'Down', 'distance': 'Full'}
    ),
    action_context='scroll'
)

# Access results
print(f"New elements: {len(result['new_elements'])}")
print(f"Removed elements: {len(result['removed_elements'])}")

# Get cumulative HTML
all_content = result['cumulative_html']  # All elements ever seen
visible_content = result['visible_html']  # Only currently visible
```

### Standalone Usage (Without WebDriver)

```python
from webaxon.automation.content_memory import ContentMemory

memory = ContentMemory()

# Capture snapshot before action
snapshot_before = memory.capture_snapshot(
    body_html=html_before,
    scroll_position={'x': 0, 'y': 0},
    viewport_size={'width': 1920, 'height': 1080},
    action_context='before_scroll'
)

# Perform action (scroll, click, etc.)
# ...

# Capture snapshot after action
snapshot_after = memory.capture_snapshot(
    body_html=html_after,
    scroll_position={'x': 0, 'y': 500},
    viewport_size={'width': 1920, 'height': 1080},
    action_context='after_scroll'
)

# Merge and analyze
result = memory.merge_snapshots(snapshot_before, snapshot_after)
print(result.summary())  # "Merge: 10 new, 5 removed, 20 persistent"

# Get cumulative HTML
all_html = memory.get_cumulative_html()
```

## API Reference

### ContentMemory

Main class for managing content memorization.

**Methods:**

- `capture_snapshot(body_html, scroll_position, viewport_size, action_context=None, url=None)` - Capture page state
- `merge_snapshots(snapshot_before, snapshot_after)` - Merge two snapshots
- `get_cumulative_html(include_removed=True, ...)` - Get HTML of all tracked elements
- `get_visible_html()` - Get HTML of only visible elements
- `get_statistics()` - Get memory statistics
- `reset()` - Clear all memorized content
- `prune_old_elements(max_age_seconds=300)` - Remove old elements

### ElementData

Represents a single HTML element with metadata.

**Attributes:**
- `id` - Element ID
- `tag` - HTML tag name
- `text` - Text content
- `html` - Raw HTML
- `visibility_state` - VISIBLE, REMOVED, or HIDDEN
- `first_seen` - Timestamp when first seen
- `last_seen` - Timestamp when last seen
- `view_count` - Number of times seen

### ContentSnapshot

A snapshot of page content at a specific moment.

**Attributes:**
- `timestamp` - When snapshot was taken
- `elements` - Dict of ElementData objects
- `scroll_position` - Scroll position
- `viewport_size` - Viewport dimensions
- `action_context` - Context string (e.g., 'before_scroll')

### MergeResult

Result of merging two snapshots.

**Attributes:**
- `new_elements` - Elements only in after snapshot
- `removed_elements` - Elements only in before snapshot
- `persistent_elements` - Elements in both snapshots
- `summary()` - Human-readable summary

## Examples

### Example 1: Track Scroll Session

```python
driver = WebDriver(enable_content_memory=True)
driver.open_url('https://news.ycombinator.com/')
driver.add_unique_index_to_elements()

# Scroll multiple times
for i in range(5):
    result = driver.execute_action_with_memory(
        action_func=lambda: driver.execute_single_action(
            element=driver.find_element_by_xpath(tag_name='body'),
            action_type='scroll',
            action_args={'direction': 'Down', 'distance': 'Full'}
        ),
        action_context=f'scroll_{i}'
    )
    print(f"Scroll {i+1}: {len(result['new_elements'])} new elements")

# Get all content seen during session
all_content = driver.content_memory.get_cumulative_html()
```

### Example 2: Track Click Navigation

```python
driver = WebDriver(enable_content_memory=True)

# Capture before click
result = driver.execute_action_with_memory(
    action_func=lambda: driver.execute_single_action(
        element=link_element,
        action_type='click'
    ),
    action_context='navigate'
)

# See what changed
print(f"Page changed: {len(result['removed_elements'])} elements removed")
print(f"New page: {len(result['new_elements'])} elements loaded")
```

### Example 3: Memory Management

```python
# Get statistics
stats = driver.content_memory.get_statistics()
print(f"Tracking {stats['total_elements']} elements")
print(f"Took {stats['snapshots_taken']} snapshots")

# Prune old elements (removed more than 5 minutes ago)
removed_count = driver.content_memory.prune_old_elements(max_age_seconds=300)
print(f"Pruned {removed_count} old elements")

# Reset entirely
driver.content_memory.reset()
```

## Implementation Details

### Element Identification

Elements are identified using the `__incremental_id__` attribute added by `add_unique_index_to_elements()`. This ensures consistent tracking across snapshots.

### Deduplication

Elements are deduplicated using:
1. Primary: `__incremental_id__` attribute
2. Fallback: Content hash (tag + text + key attributes)

### Memory Usage

- Each ElementData object stores ~500-1000 bytes
- 1000 elements ≈ 500KB-1MB memory
- Use `prune_old_elements()` for long sessions

### Performance

- Snapshot capture: ~50-200ms (depends on page size)
- Merge operation: ~10-50ms (depends on element count)
- HTML generation: ~20-100ms (depends on element count)

## Best Practices

1. **Enable selectively**: Only enable memory when needed (scrolling long pages, tracking navigation)
2. **Prune regularly**: Call `prune_old_elements()` periodically in long sessions
3. **Reset between pages**: Call `reset()` when navigating to new websites
4. **Save cumulative HTML**: Store results for later analysis rather than keeping in memory

## Troubleshooting

### "Content memory not enabled" error

Enable content memory in constructor:
```python
driver = WebDriver(enable_content_memory=True)
```

### Memory growing too large

Prune old elements:
```python
driver.content_memory.prune_old_elements(max_age_seconds=300)
```

### Elements not being tracked

Ensure `add_unique_index_to_elements()` is called before actions:
```python
driver.add_unique_index_to_elements()
```

## See Also

- Example script: `examples/content_memory_example.py`
- WebDriver integration: `webagent/automation/web_driver.py`
- Module source: `webagent/automation/content_memory/`
