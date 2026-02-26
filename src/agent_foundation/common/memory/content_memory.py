"""
Main ContentMemory class - simplified generic memory for capturing child elements.
"""

from typing import List, Dict, Optional, Callable, Any
from attr import attrs, attrib
from rich_python_utils.common_utils import get_


@attrs
class ContentMemory:
    """
    Generic memory for capturing and storing child elements from any structure.

    Works with HTML, JSON, objects, or any structure with children.
    Uses get_() function for flexible child extraction.
    Stores elements in a dict keyed by fingerprint for efficient deduplication.

    Attributes:
        accumulate: If True, keep adding elements. If False, replace on each capture.
        auto_merge_memory: If True and accumulate=True, skip elements with existing fingerprints.
                          If False, overwrite elements with same fingerprint (update to latest).
        memory_accumulator: Optional custom function(base_memory, memory_list) -> combined_content
                           to control how base_memory and memory elements are combined.
                           If None, uses default accumulator based on base_memory type.
        elements: Dict mapping fingerprints to captured elements.

    Example:
        # HTML parsing with auto-deduplication
        memory = ContentMemory(accumulate=True, auto_merge_memory=True)
        memory.capture_snapshot(
            content=body_html,
            get_children=lambda x: BeautifulSoup(x).find_all('div'),
            get_fingerprint=lambda elem: elem.get('id')  # Use element ID as fingerprint
        )

        # JSON data using default hash() for fingerprinting
        memory.capture_snapshot(
            content={'data': {'items': [1, 2, 3]}},
            get_children='data.items'
        )

        # Custom accumulator for special combining logic
        def custom_accumulator(base_memory, memory_list):
            return f"<wrapper>{base_memory}{''.join(str(e) for e in memory_list)}</wrapper>"

        memory = ContentMemory(memory_accumulator=custom_accumulator)

        # Access results
        count = memory.count
        elements_list = memory.memory  # Get elements as list
        combined = memory.accumulated_content  # Uses custom accumulator
    """

    # Configuration
    accumulate: bool = attrib(default=True)
    auto_merge_memory: bool = attrib(default=True)
    exclude_last_entry_from_memory: bool = attrib(default=True)

    default_get_children: Any = attrib(default='children')  # Default strategy for extracting children
    default_get_signature: Optional[Callable[[Any], Any]] = attrib(default=None)  # Default signature function
    use_base_memory_for_merge: bool = attrib(default=False)  # If True, deduplicate against base_memory
    get_base_signatures: Optional[Callable[[Any], set]] = attrib(default=None)  # Extract signatures from base_memory
    memory_accumulator: Optional[Callable[[Optional[Any], List[Any]], Any]] = attrib(
        default=None)  # Custom accumulator(base_memory, memory_list) -> combined

    associated_attributes: Dict[str, Any] = attrib(factory=dict)  # Metadata (e.g., last_action_type)

    # Base memory and metadata (private backing fields)
    _base_memory: Optional[Any] = attrib(default=None)  # Readonly baseline content (e.g., original HTML)
    _base_memory_for_comparison: Optional[Any] = attrib(
        default=None)  # Baseline used for deduplication (e.g., target element for incremental tracking)
    _base_memory_signatures: Optional[set] = attrib(init=False, default=None)  # Cached signatures from base_memory
    _memory: Dict[Any, Any] = attrib(init=False, factory=dict)  # State - Dict[signature, element] (private)
    _last_added_memory_signatures: set = attrib(init=False,
                                                factory=set)  # Signatures of elements added in the last capture_snapshot

    @property
    def base_memory(self) -> Optional[Any]:
        """Get the baseline content."""
        return self._base_memory

    @property
    def base_memory_for_comparison(self) -> Optional[Any]:
        """Get the baseline used for deduplication."""
        return self._base_memory_for_comparison

    def set_base_memory(
            self,
            base_memory: Optional[Any] = None,
            base_memory_for_comparison: Optional[Any] = None
    ) -> None:
        """
        Set base memory fields together and clear cached signatures if values change.

        This method should be used instead of setting base_memory and base_memory_for_comparison
        separately to ensure the signature cache is properly invalidated.

        Args:
            base_memory: Baseline content (e.g., original HTML)
            base_memory_for_comparison: Baseline used for deduplication (e.g., target element).
                                       If None, falls back to base_memory for comparison.
        """
        # Check if effective comparison base will change
        old_comparison_base = self._base_memory_for_comparison if self._base_memory_for_comparison is not None else self._base_memory
        new_comparison_base = base_memory_for_comparison if base_memory_for_comparison is not None else base_memory

        # Update base memory fields
        self._base_memory = base_memory
        self._base_memory_for_comparison = base_memory_for_comparison

        # Clear cached signatures if comparison base changed
        if old_comparison_base != new_comparison_base:
            self._base_memory_signatures = None

        # Whenever the base memory is set, the incremental memory resets
        self.reset_incremental()

    def capture_snapshot(
            self,
            content: Any,
            get_children: Optional[Any] = None,
            get_signature: Optional[Callable[[Any], Any]] = None
    ) -> None:
        """
        Capture children from content structure.

        Args:
            content: Source content (HTML, dict, object, WebDriver, etc.)
            get_children: Path/key to extract children via get_() function, or callable.
                         If None, uses self.default_get_children.
            get_signature: Function to compute signature (hash) for deduplication.
                          Must return a hashable value. If None, uses self.default_get_signature
                          (which defaults to hash()).

        Examples:
            # HTML with custom signature (element ID)
            memory.capture_snapshot(
                content=html_string,
                get_children=lambda x: BeautifulSoup(x).find_all('div'),
                get_signature=lambda elem: elem.get('id')
            )

            # JSON data with default hash()
            memory.capture_snapshot(
                content={'data': {'items': [1, 2, 3]}},
                get_children='data.items'
            )

            # WebDriver elements with text as signature
            memory.capture_snapshot(
                content=driver,
                get_children=lambda x: x.find_elements_by_tag_name('div'),
                get_signature=lambda elem: elem.text
            )
        """
        # Use defaults if not specified
        if get_children is None:
            get_children = self.default_get_children
        if get_signature is None:
            get_signature = self.default_get_signature or hash

        # Compute base signatures if needed (lazy computation)
        # Use base_memory_for_comparison if set, otherwise fall back to base_memory
        comparison_base = self.base_memory_for_comparison if self.base_memory_for_comparison is not None else self.base_memory
        if self.use_base_memory_for_merge and self._base_memory_signatures is None and comparison_base is not None:
            if self.get_base_signatures is not None:
                self._base_memory_signatures = self.get_base_signatures(comparison_base)
            else:
                self._base_memory_signatures = set()  # No extractor, treat as empty

        # Extract children using get_()
        new_elements = get_(content, get_children, default=[])

        # Ensure it's a list
        if not isinstance(new_elements, list):
            new_elements = [new_elements] if new_elements is not None else []

        # Clear last added signatures from previous capture
        self._last_added_memory_signatures.clear()

        if not self.accumulate:
            # Replace entire dict
            self._memory = {}
            for elem in new_elements:
                signature = get_signature(elem)
                # Skip if signature exists in base_memory (when use_base_memory_for_merge=True)
                if self.use_base_memory_for_merge and self._base_memory_signatures and signature in self._base_memory_signatures:
                    continue
                self._memory[signature] = elem
                self._last_added_memory_signatures.add(signature)
        else:
            # Accumulate
            for elem in new_elements:
                signature = get_signature(elem)

                # Skip if signature exists in base_memory (when use_base_memory_for_merge=True)
                if self.use_base_memory_for_merge and self._base_memory_signatures and signature in self._base_memory_signatures:
                    continue

                if self.auto_merge_memory:
                    # Skip if signature already exists in memory (deduplicate)
                    if signature not in self._memory:
                        self._memory[signature] = elem
                        self._last_added_memory_signatures.add(signature)
                else:
                    # Overwrite if signature exists (update to latest)
                    self._memory[signature] = elem
                    self._last_added_memory_signatures.add(signature)

    @property
    def memory(self):
        """Get memory elements as a list (in insertion order for Python 3.7+)."""
        accumulator = self.memory_accumulator or self._default_memory_accumulator
        memory_list = self._get_memory_list(exclude_last=self.exclude_last_entry_from_memory)
        return accumulator(self.base_memory, memory_list)

    def _default_memory_accumulator(self, base_memory: Optional[Any], memory_list: List[Any]) -> Any:
        """
        Default memory accumulator implementation.

        Combines base_memory with memory_list based on the type of base_memory
        and the use_base_memory_for_merge flag.

        Args:
            base_memory: The baseline content (e.g., initial HTML state).
            memory_list: List of accumulated memory elements.

        Returns:
            Combined content based on base_memory type and merge strategy.
        """
        if not self.use_base_memory_for_merge or base_memory is None:
            # base_memory is independent, just return memory elements
            return memory_list

        # Combine base_memory with memory elements
        if isinstance(base_memory, str):
            return base_memory + '\n' + '\n'.join(memory_list)
        elif isinstance(base_memory, list):
            # List content: extend
            return base_memory + memory_list
        else:
            # Other types: return memory elements only (caller handles combination)
            return memory_list

    def _get_memory_list(self, exclude_last: bool = False) -> List[Any]:
        """
        Get memory values as a list, optionally excluding last added items.

        Args:
            exclude_last: If True, exclude elements added in the last capture_snapshot call.

        Returns:
            List of memory elements in insertion order.
        """
        if exclude_last and self._last_added_memory_signatures:
            # Filter out items whose signatures are in the last added set
            return [elem for sig, elem in self._memory.items() if sig not in self._last_added_memory_signatures]
        return list(self._memory.values())

    def reset_incremental(self):
        """Clear all memory elements."""
        if self._memory:
            self._memory.clear()
        if self._last_added_memory_signatures:
            self._last_added_memory_signatures.clear()

    def clear(self):
        self._base_memory = None
        self._base_memory_for_comparison = None
        if self._base_memory_signatures:
            self._base_memory_signatures.clear()
        self.reset_incremental()