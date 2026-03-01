"""DualInferencer — propose-review-fix consensus loop as a composable inferencer.

Models a single consensus phase: base_inferencer proposes, review_inferencer
reviews, fixer_inferencer addresses issues, and the loop repeats until consensus
or max_iterations. Two DualInferencer instances can be chained with swapped roles
for a full dual-phase workflow (e.g., planning → execution).
"""

import json
import logging
import re
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Union

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusAttemptRecord,
    ConsensusConfig,
    ConsensusIterationRecord,
    DualInferencerResponse,
    InferencerResponse,
    ReflectionStyles,
    ResponseSelectors,
    Severity,
    severity_at_most,
)
from agent_foundation.common.inferencers.agentic_inferencers.constants import (
    DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE,
    DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE,
    DEFAULT_PLACEHOLDER_DUAL_COUNTER_FEEDBACK,
    DEFAULT_PLACEHOLDER_DUAL_INPUT,
    DEFAULT_PLACEHOLDER_DUAL_ISSUES,
    DEFAULT_PLACEHOLDER_DUAL_PROPOSAL,
    DEFAULT_PLACEHOLDER_DUAL_REASONING,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from attr import attrib, attrs
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.input_and_response import InputAndResponse
from rich_python_utils.string_utils.formatting.template_manager import (
    TemplateManager,
)
from rich_python_utils.string_utils.xml_helpers import unescape_xml

logger = logging.getLogger(__name__)


@attrs
class DualInferencer(InferencerBase):
    """Propose-review-fix consensus loop as a first-class inferencer.

    Implements a multi-round consensus workflow:
    1. base_inferencer generates an initial proposal
    2. review_inferencer reviews the proposal
    3. If consensus not reached, fixer_inferencer addresses issues
    4. The improved proposal is re-reviewed (step 2)
    5. Loop continues until consensus or max_iterations

    The fixer can optionally produce counter-feedback rejecting invalid
    review issues, which is passed to the reviewer in the next iteration.

    Usage:
        # Simple 2-agent mode (base_inferencer also fixes):
        dual = DualInferencer(
            base_inferencer=proposer,
            review_inferencer=reviewer,
        )
        result = dual("Design a REST API for user management")

        # 3-agent mode:
        dual = DualInferencer(
            base_inferencer=proposer,
            review_inferencer=reviewer,
            fixer_inferencer=dedicated_fixer,
        )

        # Async (recommended):
        async with DualInferencer(...) as inf:
            result = await inf.ainfer("Design a REST API")

    Attributes:
        base_inferencer: Proposer/planner inferencer.
        review_inferencer: Reviewer/reflector inferencer.
        fixer_inferencer: Fixer inferencer (defaults to base_inferencer).
        consensus_config: Loop configuration (max iterations, threshold, etc.).
        prompt_formatter: Shared TemplateManager for all prompts. When set,
            initial_prompt/review_prompt/followup_prompt are used as template_key
            names. When None, those strings are treated as raw Jinja2 templates.
        initial_prompt: Template key (when prompt_formatter is set) or raw Jinja2
            template string for the initial prompt. None means passthrough.
        review_prompt: Template key or raw template for review prompts.
        followup_prompt: Template key or raw template for followup/fix prompts.
        review_parser: Callable to parse raw review output into structured dict.
        followup_response_parser: Callable to parse counter-feedback from fixer output.
        response_parser: Callable to extract/clean raw output from any sub-inferencer.
        consensus_checker: Callable (parsed_review, threshold) → bool.
        response_selector: How to select the final output from the response object.
        issue_id_format: Format string for issue IDs.
        phase: Label for this consensus phase (for logging/metadata).
    """

    base_inferencer: InferencerBase = attrib(default=None)
    review_inferencer: InferencerBase = attrib(default=None)
    fixer_inferencer: Optional[InferencerBase] = attrib(default=None)

    consensus_config: ConsensusConfig = attrib(factory=ConsensusConfig)

    prompt_formatter: Callable = attrib(default=None)
    initial_prompt: Optional[str] = attrib(default=None)
    review_prompt: str = attrib(default=DEFAULT_DUAL_REVIEW_PROMPT_TEMPLATE)
    followup_prompt: str = attrib(default=DEFAULT_DUAL_FOLLOWUP_PROMPT_TEMPLATE)

    review_parser: Callable = attrib(default=None)
    followup_response_parser: Callable = attrib(default=None)
    response_parser: Callable = attrib(default=None)
    consensus_checker: Callable = attrib(default=None)

    response_selector: Union[
        Callable[["InferencerResponse"], Any], ResponseSelectors
    ] = attrib(default=ResponseSelectors.BaseResponse)

    issue_id_format: str = attrib(default="ISS-{iteration:02d}-{index:03d}")
    phase: str = attrib(default="")

    # Placeholder keys for template variables
    placeholder_input: str = attrib(default=DEFAULT_PLACEHOLDER_DUAL_INPUT)
    placeholder_proposal: str = attrib(default=DEFAULT_PLACEHOLDER_DUAL_PROPOSAL)
    placeholder_issues: str = attrib(default=DEFAULT_PLACEHOLDER_DUAL_ISSUES)
    placeholder_reasoning: str = attrib(default=DEFAULT_PLACEHOLDER_DUAL_REASONING)
    placeholder_counter_feedback: str = attrib(
        default=DEFAULT_PLACEHOLDER_DUAL_COUNTER_FEEDBACK
    )

    def __attrs_post_init__(self):
        super(DualInferencer, self).__attrs_post_init__()

        if (not self.response_types) or self.response_types == (str,):
            self.response_types = (str, InferencerResponse, DualInferencerResponse)

        # Default fixer to base_inferencer (2-agent mode)
        if self.fixer_inferencer is None:
            self.fixer_inferencer = self.base_inferencer

        # Build prompt rendering infrastructure
        if isinstance(self.prompt_formatter, TemplateManager):
            # Shared TemplateManager — initial/review/followup are template_key names
            self._prompt_tms = None
        else:
            # No shared TemplateManager — wrap each raw template string individually
            custom_formatter = self.prompt_formatter
            self._prompt_tms = {}
            for role, prompt_str in [
                ("initial", self.initial_prompt),
                ("review", self.review_prompt),
                ("followup", self.followup_prompt),
            ]:
                if prompt_str is not None:
                    self._prompt_tms[role] = TemplateManager(
                        templates=prompt_str,
                        template_formatter=custom_formatter,
                        enable_templated_feed=True,
                    )

        # Default parsers
        if self.review_parser is None:
            self.review_parser = DualInferencer._default_parse_review
        if self.followup_response_parser is None:
            self.followup_response_parser = (
                DualInferencer._default_parse_counter_feedback
            )
        if self.response_parser is None:
            self.response_parser = DualInferencer._default_response_parser
        if self.consensus_checker is None:
            self.consensus_checker = DualInferencer._default_check_consensus

        # Set parent debuggable for nested inferencers (deduplicate by identity)
        seen_ids = set()
        for inf in (
            self.base_inferencer,
            self.review_inferencer,
            self.fixer_inferencer,
        ):
            if (
                inf is not None
                and isinstance(inf, Debuggable)
                and id(inf) not in seen_ids
            ):
                seen_ids.add(id(inf))
                inf.set_parent_debuggable(self)

    # region Sync/Async Bridge

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        """Sync bridge — delegates to _ainfer() via _run_async().

        For multi-call usage, prefer the async interface:
            async with DualInferencer(...) as inf:
                result = await inf.ainfer("task")
        """
        from rich_python_utils.common_utils.async_function_helper import _run_async

        return _run_async(
            self._ainfer(inference_input, inference_config, **_inference_args)
        )

    # endregion

    # region Core Consensus Loop

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        """Async inference — core consensus loop.

        Args:
            inference_input: The original task/request prompt.
            inference_config: Optional dict with overrides:
                - "consensus_config": ConsensusConfig override.
                - "phase": Phase label override.
            **_inference_args: Additional args passed to sub-inferencers.

        Returns:
            DualInferencerResponse with consensus history and final proposal.
        """
        if inference_config is None:
            inference_config = {}
        elif not isinstance(inference_config, Mapping):
            raise ValueError("'inference_config' must be a mapping")

        config = inference_config.get("consensus_config", self.consensus_config)
        phase = inference_config.get("phase", self.phase)

        all_attempt_records: List[ConsensusAttemptRecord] = []
        final_output = None
        final_review = None
        consensus_achieved = False
        total_iterations = 0

        for attempt in range(1, config.max_consensus_attempts + 1):
            if attempt > 1:
                await self._areset_sub_inferencers()

            logger.info(
                "[%s] Starting consensus attempt %d/%d",
                phase or "DualInferencer",
                attempt,
                config.max_consensus_attempts,
            )
            attempt_record = ConsensusAttemptRecord(attempt=attempt)

            # Step 1: Build initial prompt (if initial_prompt is set)
            if self.initial_prompt is not None:
                initial_prompt = self._build_initial_prompt(
                    inference_input,
                    inference_config,
                    attempt=attempt,
                )
            else:
                initial_prompt = inference_input  # backward compatible passthrough

            self.log_info(
                initial_prompt,
                "InitialPrompt",
                is_artifact=True,
                parts_min_size=0,
                parts_subfolder=f"Round{total_iterations + 1:02d}",
            )

            # Step 2: Initial proposal from base_inferencer
            _raw_base = str(
                await self.base_inferencer.ainfer(initial_prompt, **_inference_args)
            )
            self.log_debug(
                _raw_base,
                "RawBaseResponse",
                is_artifact=True,
                parts_min_size=0,
                parts_subfolder=f"Round{total_iterations + 1:02d}",
            )
            base_output_str = self.response_parser(_raw_base)
            self.log_info(
                base_output_str,
                "InitialResponse",
                is_artifact=True,
                parts_min_size=0,
                parts_subfolder=f"Round{total_iterations + 1:02d}",
            )
            counter_feedback_str = None

            for iteration in range(1, config.max_iterations + 1):
                total_iterations += 1

                # Step 3: Build review prompt
                review_prompt = self._build_review_prompt(
                    inference_input,
                    base_output_str,
                    counter_feedback_str,
                    inference_config,
                    iteration=iteration,
                    attempt=attempt,
                )
                self.log_info(
                    review_prompt,
                    "ReviewPrompt",
                    is_artifact=True,
                    parts_min_size=0,
                    parts_subfolder=f"Round{total_iterations:02d}",
                )

                # Step 4: Run reviewer
                _raw_review = str(
                    await self.review_inferencer.ainfer(
                        review_prompt, **_inference_args
                    )
                )
                self.log_debug(
                    _raw_review,
                    "RawReviewResponse",
                    is_artifact=True,
                    parts_min_size=0,
                    parts_subfolder=f"Round{total_iterations:02d}",
                )
                self.log_info(
                    _raw_review,
                    "ReviewResponse",
                    is_artifact=True,
                    parts_min_size=0,
                    parts_subfolder=f"Round{total_iterations:02d}",
                )

                # Step 5: Parse review
                review_output_str = self.response_parser(_raw_review)
                parsed_review = self.review_parser(review_output_str)
                parsed_review = self._assign_issue_ids(parsed_review, iteration)

                # Step 6: Check consensus
                reached = self.consensus_checker(
                    parsed_review, config.consensus_threshold
                )

                logger.info(
                    "[%s] Iteration %d: severity=%s, consensus_reached=%s",
                    phase or "DualInferencer",
                    iteration,
                    parsed_review.get("severity", "UNKNOWN"),
                    reached,
                )

                iteration_record = ConsensusIterationRecord(
                    iteration=iteration,
                    base_output=base_output_str,
                    review_input=review_prompt,
                    review_output=review_output_str,
                    review_feedback=parsed_review,
                    consensus_reached=reached,
                )

                if reached:
                    attempt_record.iterations.append(iteration_record)
                    attempt_record.consensus_reached = True
                    attempt_record.final_output = base_output_str
                    attempt_record.final_feedback = parsed_review
                    final_output = base_output_str
                    final_review = InputAndResponse(
                        input=review_prompt, response=review_output_str
                    )
                    consensus_achieved = True
                    break

                # Step 7: Build followup prompt
                followup_prompt = self._build_followup_prompt(
                    inference_input,
                    base_output_str,
                    parsed_review,
                    inference_config,
                    iteration=iteration,
                    attempt=attempt,
                    review_output=review_output_str,
                )
                self.log_info(
                    followup_prompt,
                    "FollowupPrompt",
                    is_artifact=True,
                    parts_min_size=0,
                    parts_subfolder=f"Round{total_iterations:02d}",
                )

                # Step 8: Run fixer
                _raw_fix = str(
                    await self.fixer_inferencer.ainfer(
                        followup_prompt, **_inference_args
                    )
                )
                self.log_debug(
                    _raw_fix,
                    "RawFixResponse",
                    is_artifact=True,
                    parts_min_size=0,
                    parts_subfolder=f"Round{total_iterations:02d}",
                )
                self.log_info(
                    _raw_fix,
                    "FollowupResponse",
                    is_artifact=True,
                    parts_min_size=0,
                    parts_subfolder=f"Round{total_iterations:02d}",
                )

                # Step 9: Parse counter-feedback and extract improved proposal
                fix_output_str = self.response_parser(_raw_fix)
                parsed_counter = self.followup_response_parser(fix_output_str)
                counter_feedback_str = (
                    json.dumps(parsed_counter, indent=2)
                    if parsed_counter.get("items")
                    else None
                )
                improved_proposal = fix_output_str

                iteration_record.counter_feedback = parsed_counter
                attempt_record.iterations.append(iteration_record)

                # Step 10: Update base_output for next iteration
                base_output_str = improved_proposal

            # End of inner loop
            if not attempt_record.consensus_reached:
                attempt_record.final_output = base_output_str
                final_output = base_output_str

            all_attempt_records.append(attempt_record)
            if consensus_achieved:
                break

        # Build final review if none captured (no consensus case)
        if (
            final_review is None
            and all_attempt_records
            and all_attempt_records[-1].iterations
        ):
            last_iter = all_attempt_records[-1].iterations[-1]
            final_review = InputAndResponse(
                input=last_iter.review_input, response=last_iter.review_output
            )

        logger.info(
            "[%s] Consensus loop complete: achieved=%s, total_iterations=%d, attempts=%d",
            phase or "DualInferencer",
            consensus_achieved,
            total_iterations,
            len(all_attempt_records),
        )

        return DualInferencerResponse(
            base_response=final_output or "",
            reflection_response=final_review,
            reflection_style=ReflectionStyles.Sequential,
            response_selector=self.response_selector,
            consensus_history=all_attempt_records,
            total_iterations=total_iterations,
            consensus_achieved=consensus_achieved,
            phase=phase,
        )

    # endregion

    # region Prompt Builders

    def _render_prompt(self, role: str, feed: dict, inference_config: dict) -> str:
        """Render a prompt template by role name.

        When prompt_formatter is a TemplateManager, uses the role's prompt attr
        as template_key. Otherwise uses the per-role wrapped TemplateManager.

        Args:
            role: One of "initial", "review", "followup".
            feed: Template variables dict.
            inference_config: Passed through to TemplateManager as **kwargs.

        Returns:
            Rendered prompt string.
        """
        post_process = partial(unescape_xml, unescape_for_html=True)

        if self._prompt_tms is None:
            # Shared TemplateManager — use prompt attr value as template_key
            key = getattr(self, f"{role}_prompt")
            return self.prompt_formatter(
                template_key=key,
                feed=feed,
                post_process=post_process,
                **inference_config,
            )
        else:
            # Per-role TemplateManagers (backward compatible)
            return self._prompt_tms[role](
                feed=feed,
                post_process=post_process,
                **inference_config,
            )

    def _build_initial_prompt(
        self,
        inference_input,
        inference_config: dict,
        attempt: int = 1,
    ) -> str:
        """Build the initial prompt from template.

        Args:
            inference_input: Raw user request text.
            inference_config: Config dict passed through to TemplateManager.
            attempt: Current consensus attempt (1-based).

        Returns:
            Rendered initial prompt string.
        """
        feed = {
            self.placeholder_input: inference_input,
            "iteration": 0,
            "attempt": attempt,
            "round_index": 0,
        }

        return self._render_prompt("initial", feed, inference_config)

    def _build_review_prompt(
        self,
        inference_input,
        proposal: str,
        counter_feedback: Optional[str],
        inference_config: dict,
        iteration: int = 1,
        attempt: int = 1,
    ) -> str:
        """Build the review prompt from template.

        Args:
            inference_input: Original user request.
            proposal: Current proposal text.
            counter_feedback: Serialized counter-feedback JSON, or None.
            inference_config: Config dict passed through to TemplateManager.
            iteration: Current iteration within the attempt (1-based).
            attempt: Current consensus attempt (1-based).

        Returns:
            Rendered review prompt string.
        """
        feed = {
            self.placeholder_input: inference_input,
            self.placeholder_proposal: proposal,
            "iteration": iteration,
            "attempt": attempt,
            "round_index": iteration - 1,
        }
        if counter_feedback is not None:
            feed[self.placeholder_counter_feedback] = counter_feedback

        return self._render_prompt("review", feed, inference_config)

    def _build_followup_prompt(
        self,
        inference_input,
        proposal: str,
        parsed_review: dict,
        inference_config: dict,
        iteration: int = 1,
        attempt: int = 1,
        review_output: Optional[str] = None,
    ) -> str:
        """Build the followup prompt from template.

        Args:
            inference_input: Original user request.
            proposal: Current proposal text.
            parsed_review: Parsed review dict with 'issues' and 'reasoning'.
            inference_config: Config dict passed through to TemplateManager.
            iteration: Current iteration within the attempt (1-based).
            attempt: Current consensus attempt (1-based).
            review_output: Full text of the reviewer's response (optional).

        Returns:
            Rendered followup prompt string.
        """
        issues = parsed_review.get("issues", [])
        reasoning = parsed_review.get("reasoning", "")
        config = inference_config.get("consensus_config", self.consensus_config)

        feed = {
            self.placeholder_input: inference_input,
            self.placeholder_proposal: proposal,
            self.placeholder_issues: self._serialize_issues(issues),
            self.placeholder_reasoning: reasoning,
            "enable_counter_feedback": config.enable_counter_feedback,
            "iteration": iteration,
            "attempt": attempt,
            "round_index": iteration,
        }
        if review_output is not None:
            feed["reviewer_response"] = review_output

        return self._render_prompt("followup", feed, inference_config)

    # endregion

    # region Default Parsers

    @staticmethod
    def _default_response_parser(raw: str) -> str:
        """Extract response content from delimiter tags.

        Checks ``<Response>`` and ``<ImprovedProposal>`` tags (in that order).
        Falls back to the raw string if no tags are found.

        This replaces the old ``_default_extract_proposal`` as the default for
        ``response_parser``, applied to ALL sub-inferencer outputs.

        Args:
            raw: Raw output string from any sub-inferencer.

        Returns:
            Extracted content (stripped), or the original string.
        """
        for tag in ("Response", "ImprovedProposal"):
            match = re.search(rf"<{tag}>([\s\S]*?)</{tag}>", raw)
            if match:
                return match.group(1).strip()
        return raw

    @staticmethod
    def _default_parse_review(raw: str) -> dict:
        """Parse structured review JSON from raw reviewer output.

        Extracts JSON from ```json ... ``` code blocks. Falls back to a
        MAJOR-severity error if parsing fails.

        Handles both naming conventions:
        - ``approved`` / ``approve`` → normalised to ``approved``
        - ``severity`` / ``overall_severity`` → normalised to ``severity``

        Args:
            raw: Raw reviewer output string.

        Returns:
            Dict with keys: approved, severity, issues, reasoning.
        """
        match = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
        if match:
            try:
                parsed = json.loads(match.group(1))
                # Support both "approved" and "approve"
                approved = parsed.get("approved", parsed.get("approve", False))
                # Support both "severity" and "overall_severity"
                severity = parsed.get(
                    "severity", parsed.get("overall_severity", "MAJOR")
                )
                return {
                    "approved": approved,
                    "severity": severity,
                    "issues": parsed.get("issues", []),
                    "reasoning": parsed.get("reasoning", ""),
                }
            except json.JSONDecodeError:
                pass

        return {
            "approved": False,
            "severity": "MAJOR",
            "issues": [
                {
                    "severity": "MAJOR",
                    "category": "parsing_error",
                    "description": "Failed to parse structured review from reviewer output.",
                    "location": "N/A",
                    "suggestion": "Ensure reviewer produces valid JSON in ```json blocks.",
                }
            ],
            "reasoning": "Review parsing failed — treating as non-consensus.",
        }

    @staticmethod
    def _default_parse_counter_feedback(raw: str) -> dict:
        """Parse counter-feedback JSON from fixer output.

        Extracts JSON from ```json ... ``` code blocks. Falls back to
        empty items if parsing fails.

        Args:
            raw: Raw fixer output string.

        Returns:
            Dict with keys: items, summary.
        """
        match = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return {
                    "items": parsed.get("items", []),
                    "summary": parsed.get("summary", ""),
                }
            except json.JSONDecodeError:
                pass

        return {"items": [], "summary": ""}

    @staticmethod
    def _default_extract_proposal(raw: str) -> str:
        """Extract improved proposal from fixer output.

        Looks for content inside <ImprovedProposal>...</ImprovedProposal> tags.
        Falls back to the full raw output (minus any ```json blocks) if tags
        are not found.

        Args:
            raw: Raw fixer output string.

        Returns:
            Extracted or cleaned proposal text.
        """
        match = re.search(r"<ImprovedProposal>([\s\S]*?)</ImprovedProposal>", raw)
        if match:
            return match.group(1).strip()

        # Fallback: remove ```json blocks and return the rest
        cleaned = re.sub(r"```json\s*[\s\S]*?\s*```", "", raw).strip()
        return cleaned if cleaned else raw

    @staticmethod
    def _default_check_consensus(parsed_review: dict, threshold: Severity) -> bool:
        """Check if consensus is reached based on review feedback.

        Consensus is reached if the reviewer approved, OR if the highest
        severity is at or below the threshold.

        Args:
            parsed_review: Parsed review dict with 'approved' and 'severity' keys.
            threshold: Maximum acceptable severity level.

        Returns:
            True if consensus is reached.
        """
        if parsed_review.get("approved", False):
            return True

        severity_str = parsed_review.get("severity", "MAJOR")
        try:
            review_severity = Severity(severity_str)
        except ValueError:
            return False

        return severity_at_most(review_severity, threshold)

    # endregion

    # region Utilities

    def _assign_issue_ids(self, parsed_review: dict, iteration: int) -> dict:
        """Assign unique IDs to each issue in the parsed review.

        Args:
            parsed_review: Parsed review dict (modified in place).
            iteration: Current iteration number.

        Returns:
            The same parsed_review dict with issue IDs assigned.
        """
        for index, issue in enumerate(parsed_review.get("issues", []), start=1):
            issue["id"] = self.issue_id_format.format(iteration=iteration, index=index)
        return parsed_review

    @staticmethod
    def _serialize_issues(issues: list) -> str:
        """Serialize issues list to JSON string for template rendering.

        Args:
            issues: List of issue dicts.

        Returns:
            JSON string representation.
        """
        return json.dumps(issues, indent=2)

    # endregion

    # region Lifecycle

    async def _areset_sub_inferencers(self):
        """Reset all sub-inferencers by disconnecting and reconnecting.

        Used between consensus attempts to get fresh state.
        Deduplicates sub-inferencers by identity to avoid double-reset.
        """
        seen_ids = set()
        for inf in (
            self.base_inferencer,
            self.review_inferencer,
            self.fixer_inferencer,
        ):
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                await inf.adisconnect()
                await inf.aconnect()

    async def aconnect(self, **kwargs):
        """Establish connections for all sub-inferencers."""
        seen_ids = set()
        for inf in (
            self.base_inferencer,
            self.review_inferencer,
            self.fixer_inferencer,
        ):
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                await inf.aconnect(**kwargs)

    async def adisconnect(self):
        """Disconnect all sub-inferencers."""
        seen_ids = set()
        for inf in (
            self.base_inferencer,
            self.review_inferencer,
            self.fixer_inferencer,
        ):
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                await inf.adisconnect()

    # endregion
