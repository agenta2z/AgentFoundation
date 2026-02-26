"""
Content Memory Module

A generic memory system for capturing and storing child elements from any structure.

Works with HTML, JSON, objects, or any structure with children.
Uses get_() function for flexible child extraction.
Stores elements in a dict keyed by signature for efficient deduplication.

Usage:
    from agent_foundation.common.memory import ContentMemory

    memory = ContentMemory(
        accumulate=True,
        auto_merge_memory=True,
        default_get_children=lambda x: x.find_all('div'),
        default_get_signature=lambda elem: elem.get('id')
    )
    memory.capture_snapshot(content=html_content)

    # Access results
    count = memory.count
    elements = memory.memory
"""

from .content_memory import ContentMemory


__all__ = ['ContentMemory']
