from collections.abc import Callable
from copy import copy
from enum import StrEnum
from functools import partial
from typing import Tuple, Any, Union, Mapping, Dict, Sequence, Iterable, Optional, TypeAlias, Protocol, \
    runtime_checkable

from attr import attrs, attrib

from agent_foundation.agents.agent_attachment import AgentAttachment
from agent_foundation.agents.agent_response import AgentResponse, AgentAction
from agent_foundation.agents.agent_state import AgentTaskStatusFlags, AgentStateItem, AgentStates
from agent_foundation.automation.schema.action_executor import MultiActionExecutor
from agent_foundation.agents.constants import DEFAULT_AGENT_TASK_INPUT_FIELD_USER_INPUT, \
    DEFAULT_AGENT_TASK_INPUT_FIELD_USER_PROFILE, DEFAULT_AGENT_TASK_INPUT_FIELD_CONTEXT, \
    DEFAULT_AGENT_TASK_INPUT_FIELD_ACTION_RESULTS, DEFAULT_AGENT_TASK_INPUT_FIELD_AGENT_STATES, \
    DEFAULT_AGENT_TASK_INPUT_FIELD_TRIGGER_ACTION, DEFAULT_AGENT_TASK_INPUT_FIELD_PREVIOUS_AGENT_RESULTS, \
    DEFAULT_AGENT_TASK_INPUT_FIELD_TASK_LABEL, DEFAULT_AGENT_TASK_INPUT_FIELD_ATTACHMENTS, \
    DEFAULT_AGENT_TASK_INPUT_FIELD_MAX_NUM_LOOPS
from agent_foundation.ui.interactive_base import InteractiveBase, InteractionFlags
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import ResultPassDownMode
from rich_python_utils.common_objects.workflow.common.worknode_base import WorkGraphStopFlags
from rich_python_utils.common_objects.workflow.workgraph import WorkGraphNode, WorkGraph
from rich_python_utils.common_utils import (
    dict_,
    get_,
    set_,
    is_none_or_empty_str,
    solve_as_single_input,
    get_relevant_named_args, iter_, append_, list_
)
from rich_python_utils.common_utils.attr_helper import getattr_
from rich_python_utils.common_utils.workflow import CommonWorkflowStatus
from rich_python_utils.string_utils import join_, add_prefix, remove_prefix
from rich_python_utils.string_utils.misc import snake_to_camel_case
from rich_python_utils.string_utils.xml_helpers import mapping_to_xml

LOG_TYPE_SET_AGENT_ACTIVE_LAST_NODE = 'SetAgentActiveLastNode'
LOG_TYPE_AGENT_WORKSTREAM_COMPLETED = 'AgentWorkstreamCompleted'

# Type aliases for improved type clarity
ReasonerInput: TypeAlias = Any  # Can be str, dict, or structured input
ReasonerInferenceConfig: TypeAlias = Optional[Any]  # Configuration for inference (e.g., temperature, model params)
ReasonerResponse: TypeAlias = Any  # Raw response from reasoner (to be parsed by _parse_raw_response)
ReasonerArgs: TypeAlias = Mapping[str, Any]  # Additional keyword arguments for reasoner


@runtime_checkable
class ReasonerProtocol(Protocol):
    """
    Protocol defining the interface for reasoner callables.

    A reasoner is a callable that processes user input and generates responses.
    It takes the prepared input, optional inference configuration, and additional
    keyword arguments, then returns a raw response to be parsed by the agent.

    Typical implementations include:
    - LLM-based reasoners (e.g., GPT, Claude)
    - Rule-based reasoning systems
    - Hybrid decision-making systems

    Note:
        This protocol is runtime_checkable, allowing isinstance() checks at runtime.
    """
    def __call__(
        self,
        reasoner_input: ReasonerInput,
        reasoner_inference_config: ReasonerInferenceConfig = None,
        **kwargs: Any
    ) -> ReasonerResponse:
        """
        Process input and generate a response.

        Args:
            reasoner_input: The prepared input (from _construct_reasoner_input)
            reasoner_inference_config: Optional configuration for inference
            **kwargs: Additional reasoner-specific arguments (from reasoner_args)

        Returns:
            Raw response to be parsed by _parse_raw_response
        """
        ...


class AgentLogTypes(StrEnum):
    AgentState = 'AgentState'
    AgentResponse = 'AgentResponse'
    AgentNextActions = 'AgentNextActions'
    ActionResult = 'AgentActionResults'
    ActionError = 'AgentActionError'


class AgentCompletionReason(StrEnum):
    """
    Reasons for agent execution completion.

    Attributes:
        Normal: Agent completed normally (agent_state == Completed, no more next_actions)
        MaxLoops: Agent completed due to reaching max_num_loops limit
    """
    Normal = 'Normal'
    MaxLoops = 'MaxLoops'


class AgentControls(StrEnum):
    """
    Agent-specific workflow control signals.

    Extends CommonWorkflowControls with agent-specific controls.
    Note: Must redefine all values because Python enum doesn't allow extending enums with new members.

    Attributes:
        Stop: Signal to stop execution immediately
        Pause: Signal to pause execution (can be resumed)
        Continue: Signal to continue or resume execution
        StepByStep: Execute one step at a time, pausing at each checkpoint (agent-specific)
    """
    Stop = 'Stop'
    Pause = 'Pause'
    Continue = 'Continue'
    StepByStep = 'StepByStep'


@attrs
class Agent(Debuggable):
    """
    An interactive agent class that manages user input, integrates user profile data and contextual information,
    and generates responses using a reasoning function. This class provides a flexible framework for handling
    interactive flows, enabling subclassing for specific implementations as needed.

    The `Agent` class is designed for use in user-agent interactions, capturing input, processing it through
    various components, and delivering appropriate responses. It integrates:
    - `reasoner`: a function or callable implementing ReasonerProtocol for processing user input,
    - `user_profiler`: a user profile handler (static data or callable function),
    - `context_provider`: a context handler (static data or callable function),
    - `interactive`: an interface handler for user interactions.

    While `Agent` provides a core implementation, subclasses are encouraged to override key methods to
    tailor interactions and behavior for specific applications.

    Attributes:
        user_profile (Union[str, Callable[[Any], Any], Any]): Static user profile data or a callable that
            returns user profile information based on input. If `None` or an empty string, no profile data is used.
        context (Union[str, Callable[[Any], Any]]): Static context data or a callable that returns
            context based on input. If `None` or an empty string, no additional context is used.
        reasoner (ReasonerProtocol): A callable that implements the ReasonerProtocol interface, processing
            reasoner input (constructed by _construct_reasoner_input) along with optional inference config
            and additional kwargs (from reasoner_args) to generate a raw response.
        reasoner_args (Optional[ReasonerArgs]): Additional keyword arguments passed to the reasoner during invocation.
        actor (Callable[[Any], Any]): Optional callable that performs actions based on responses.
        interactive (InteractiveBase): Interface for handling user input and sending responses.

    Methods:
        start() -> None:
            Orchestrates the user interaction flow by capturing input, applying user profile and context data,
            processing the input through the `reasoner`, parsing the response, and delivering the result.
            Subclasses can override this method to customize the interaction flow.

    Abstract Methods:
        _process_user_input(user_input: Any, user_profile: Any, context: Any) -> Any:
            Processes raw user input with optional profile and context data, readying it for the reasoner.
            Subclasses should override to implement specific input processing requirements.

        _parse_raw_response(raw_response: Any) -> Tuple[str, Union[AgentStates, str]]:
            Parses the raw response from the reasoner into a formatted response for the user. Returns the
            formatted response and an agent state, indicating if further input is required.
            Subclasses should override to customize response formatting and flow.

    Usage:
        An `Agent` instance requires an `InteractiveBase` object for handling interactions, a `reasoner`
        function to process inputs, and optional user profiling and context management.
    """
    user_profile: Union[str, Callable[[Any], Any], Any] = attrib(default=None)
    context: Union[str, Callable[[Any], Any], Any] = attrib(default=None)
    knowledge_provider: Union[Callable[[str], Dict[str, str]], Dict[str, str], None] = attrib(default=None)
    _reasoner: ReasonerProtocol = attrib(default=None, alias='reasoner')
    reasoner_args: Optional[ReasonerArgs] = attrib(default=None)
    actor: Union[Callable[[Any], Any], MultiActionExecutor] = attrib(default=None)
    summarizer: Callable[[Any], Any] = attrib(default=None)
    actor_args_transformation: Union[Callable, Mapping] = attrib(default=partial(add_prefix, prefix='action', sep='_'))
    interactive: InteractiveBase = attrib(default=None)

    task_input_field_user_input: str = attrib(default=DEFAULT_AGENT_TASK_INPUT_FIELD_USER_INPUT)
    task_input_field_user_profile: str = attrib(default=DEFAULT_AGENT_TASK_INPUT_FIELD_USER_PROFILE)
    task_input_field_context: str = attrib(default=DEFAULT_AGENT_TASK_INPUT_FIELD_CONTEXT)
    task_input_field_action_results: str = attrib(default=DEFAULT_AGENT_TASK_INPUT_FIELD_ACTION_RESULTS)
    task_input_field_previous_agent_results: str = attrib(default=DEFAULT_AGENT_TASK_INPUT_FIELD_PREVIOUS_AGENT_RESULTS)
    task_input_field_agent_states: str = attrib(default=DEFAULT_AGENT_TASK_INPUT_FIELD_AGENT_STATES)
    task_input_field_trigger_action: str = attrib(default=DEFAULT_AGENT_TASK_INPUT_FIELD_TRIGGER_ACTION)
    task_input_field_attachments: str = attrib(default=DEFAULT_AGENT_TASK_INPUT_FIELD_ATTACHMENTS)
    task_input_field_max_num_loops: str = attrib(default=DEFAULT_AGENT_TASK_INPUT_FIELD_MAX_NUM_LOOPS)

    states: AgentStates = attrib(factory=AgentStates)
    branching_agent_start_as_new: bool = attrib(default=False)
    actor_state: Any = attrib(default=None)
    anchor_action_types: Iterable = attrib(default=None)

    # Maximum number of while-loop iterations in __call__. -1 means infinite loops (default behavior)
    max_num_loops: int = attrib(default=-1)

    # Base action: Pre-configured action to execute BEFORE first reasoner call
    # Two supported formats:
    #   1. StructuredResponse string (e.g., containing <ImmediateNextActions>)
    #      - Skips reasoner, uses _parse_raw_response() to parse it
    #   2. Custom format (any non-StructuredResponse type)
    #      - Skips reasoner, uses _parse_base_action() to parse it (subclass must implement)
    # Example use case: Navigate to a predefined URL before reasoning starts
    base_action: Optional[Any] = attrib(default=None)

    additional_reasoner_input_feed: Mapping[str, Any] = attrib(default={})

    # Mapping from UserInputsRequired subtype (e.g. 'Authentication', 'MissingInformation')
    # to InputModeConfig or callable(action) -> InputModeConfig for structured input collection.
    # When the agent sends a UserInputsRequired action, the subtype is looked up here to
    # determine the input_mode passed to interactive.send_response().
    user_input_mode_mapping: Dict[str, Any] = attrib(factory=dict)

    # Metadata fields that must remain consistent throughout the session
    # Example: ['session_id'] ensures session_id doesn't change during agent execution
    # If a follow-up input contains a different value for these fields, a warning is logged
    # and the original value is preserved
    ensure_consistent_session_metadata_fields: Iterable[str] = attrib(default=None)

    # Store the currently active WorkGraph for graph exploration
    # Allows external tools to traverse the agent's execution graph via BFS/DFS
    # The WorkGraph object contains start_nodes which can be accessed for traversal
    _active_last_node: Debuggable = attrib(default=None, init=False)

    # Workflow control signal for managing agent execution (Stop, Pause, Continue, StepByStep)
    # External code should use stop(), pause(), resume(), step_by_step() methods to set control
    # Or read via the public control property
    # Defaults to Continue to allow normal execution
    _control: AgentControls = attrib(default=AgentControls.Continue, init=False)

    # Workflow status representing the actual execution state (Stopped, Paused, Running)
    # This reflects the current state of the agent, updated when control signals are executed
    # External code should read via the public status property
    # Defaults to Stopped - changes to Running when execution begins
    _status: CommonWorkflowStatus = attrib(default=CommonWorkflowStatus.Stopped, init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        # Auto-wrap Mapping actor into MultiActionExecutor
        if self.actor is not None and isinstance(self.actor, Mapping) and not isinstance(self.actor, MultiActionExecutor):
            self.actor = MultiActionExecutor(self.actor)
        if self.interactive is not None and isinstance(self.interactive, Debuggable):
            self.interactive.set_parent_debuggable(self)
        if self._reasoner is not None and isinstance(self._reasoner, Debuggable):
            self._reasoner.set_parent_debuggable(self)

    @property
    def reasoner(self) -> ReasonerProtocol:
        """Get the reasoner instance."""
        return self._reasoner

    @reasoner.setter
    def reasoner(self, value: ReasonerProtocol):
        """
        Set the reasoner instance and properly configure its parent debuggable.

        This setter ensures that when a reasoner is replaced (e.g., during inferencer swapping),
        the new reasoner's parent_debuggable is correctly set to this agent, maintaining the
        debugging chain.

        Args:
            value: New reasoner instance that implements the ReasonerProtocol
        """
        self._reasoner = value
        if value is not None and isinstance(value, Debuggable):
            value.set_parent_debuggable(self)

    @property
    def status(self) -> CommonWorkflowStatus:
        """
        Get the current execution status of the agent.

        Returns:
            CommonWorkflowStatus: Current status (Stopped, Paused, or Running)
        """
        return self._status

    @property
    def control(self) -> 'AgentControls':
        """
        Get the current control signal of the agent.

        Returns:
            AgentControls: Current control signal (Stop, Pause, Continue, or StepByStep)
        """
        return self._control

    def add_actor(self, actor_name: str, actor: Callable):
        """
        Dynamically add an actor to the agent's actor registry.

        This method allows adding actors after the agent is initialized. Actors can be
        associated with specific action types or used as the default actor.

        Args:
            actor_name: The name/identifier for this actor. Can be an action type
                       (e.g., 'Navigation.VisitURL', 'Search', 'ElementInteraction.Click')
                       or 'default' for the fallback actor when no specific match is found.
            actor: The callable actor to add. Should implement the actor interface.

        Example:
            # Add specific actors for different action types
            agent.add_actor('Navigation.VisitURL', url_navigation_actor)
            agent.add_actor('Search', search_actor)
            agent.add_actor('default', default_web_actor)

        Raises:
            ValueError: If self.actor has an unexpected type
        """
        # Case 1: actor is None - initialize as MultiActionExecutor
        if self.actor is None:
            self.actor = MultiActionExecutor({actor_name: actor})

        # Case 2: actor is a MultiActionExecutor - use add_executor
        elif isinstance(self.actor, MultiActionExecutor):
            self.actor.add_executor(actor_name, actor)

        # Case 3: actor is a single callable - convert to MultiActionExecutor
        elif callable(self.actor):
            default_actor = self.actor
            self.actor = MultiActionExecutor({'default': default_actor})
            self.actor.add_executor(actor_name, actor)

        # Case 4: Unexpected type
        else:
            raise ValueError(
                f"Cannot add actor: self.actor has unexpected type {type(self.actor)}. "
                f"Expected None, Callable, or MultiActionExecutor."
            )

    def _extract_user_input_and_metadata(
            self,
            raw_input: Any,
            existing_metadata: Dict = None
    ) -> Tuple[Any, Dict, Dict]:
        """
        Extracts user input, metadata, and agent fields from raw input received from interactive.get_input().

        This utility method separates:
        1. User-facing input (the actual message)
        2. Internal metadata (session_id, etc.) - for response routing
        3. Agent-specific fields (user_profile, context, etc.) - for agent processing

        This prevents metadata from appearing in conversation prompts while preserving it for
        response routing, and cleanly separates agent fields for downstream processing.

        Additionally, if ensure_consistent_session_metadata_fields is set, this method validates
        that specified fields (e.g., 'session_id') remain consistent across follow-up inputs.
        If a field value changes, an error is logged and the original value is preserved.

        Args:
            raw_input: Raw input from interactive.get_input(), can be:
                - A string: direct user input, no metadata or agent fields
                - A dict with 'user_input' key: structured input with potential metadata and agent fields
                - Other types: treated as direct user input
            existing_metadata: Optional dict of existing metadata to update. Also used for consistency
                validation of fields specified in ensure_consistent_session_metadata_fields.

        Returns:
            Tuple[Any, Dict, Dict]: (user_input, metadata_dict, agent_fields_dict)
                - user_input: Clean user input (string or other types) for prompt construction
                - metadata_dict: Dict containing metadata fields (session_id, etc.) for response routing
                - agent_fields_dict: Dict containing agent fields (user_profile, context, etc.) for processing

        Example:
            # Case 1: String input
            user_input, metadata, agent_fields = agent._extract_user_input_and_metadata("hello")
            # Returns: ("hello", {}, {})

            # Case 2: Dict with metadata only
            raw = {"session_id": "session_1", "user_input": "hello"}
            user_input, metadata, agent_fields = agent._extract_user_input_and_metadata(raw)
            # Returns: ("hello", {"session_id": "session_1"}, {})

            # Case 3: Dict with metadata and agent fields
            raw = {"session_id": "session_1", "user_input": "hello", "user_profile": {"name": "Alice"}}
            user_input, metadata, agent_fields = agent._extract_user_input_and_metadata(raw)
            # Returns: ("hello", {"session_id": "session_1"}, {"user_profile": {"name": "Alice"}})

            # Case 4: Consistency validation (with ensure_consistent_session_metadata_fields=['session_id'])
            # First call
            raw1 = {"session_id": "session_1", "user_input": "hello"}
            user_input1, metadata1, _ = agent._extract_user_input_and_metadata(raw1)
            # Returns: ("hello", {"session_id": "session_1"}, {})

            # Follow-up call with different session_id (ERROR!)
            raw2 = {"session_id": "session_2", "user_input": "how are you"}
            user_input2, metadata2, _ = agent._extract_user_input_and_metadata(
                raw2, existing_metadata=metadata1
            )
            # Logs ERROR: "Metadata field 'session_id' changed from 'session_1' to 'session_2'..."
            # Returns: ("how are you", {"session_id": "session_1"}, {})  # Preserved original!
        """
        # Initialize metadata dict
        if existing_metadata is None:
            metadata = {}
        else:
            metadata = dict(existing_metadata)

        # Extract user_input, metadata, and agent fields from structured dict input
        if isinstance(raw_input, Dict) and self.task_input_field_user_input in raw_input:
            raw_input_copy = dict(raw_input)
            user_input = raw_input_copy.pop(self.task_input_field_user_input)

            # Define agent-specific fields
            agent_field_names = {
                self.task_input_field_user_profile,
                self.task_input_field_context,
                self.task_input_field_action_results,
                self.task_input_field_agent_states,
                self.task_input_field_trigger_action
            }

            # Separate agent fields from metadata
            agent_fields = {k: v for k, v in raw_input_copy.items()
                            if k in agent_field_names}
            new_metadata = {k: v for k, v in raw_input_copy.items()
                            if k not in agent_field_names}

            # Validate consistency of critical metadata fields
            if self.ensure_consistent_session_metadata_fields and existing_metadata:
                for field_name in self.ensure_consistent_session_metadata_fields:
                    if field_name in new_metadata and field_name in existing_metadata:
                        existing_value = existing_metadata[field_name]
                        new_value = new_metadata[field_name]

                        if existing_value != new_value:
                            # Log error and preserve original value
                            self.log_error(
                                f"Metadata field '{field_name}' changed from '{existing_value}' to '{new_value}'. "
                                f"Preserving original value '{existing_value}' to maintain session consistency."
                            )
                            # Override new_metadata with existing value to maintain consistency
                            new_metadata[field_name] = existing_value

            # Update metadata dict with new metadata
            metadata.update(new_metadata)

            return user_input, metadata, agent_fields
        else:
            # Not a structured dict, treat as direct user input
            return raw_input, metadata, {}

    @property
    def source(self):
        return self.states.last_action.source

    def _construct_reasoner_input(
            self,
            task_input: Any,
            user_input: Any,
            user_profile: Any = None,
            context: Any = None,
            action_results: Any = None,
            attachments: Sequence = None,
            knowledge: Dict[str, str] = None
    ) -> ReasonerInput:
        """
        Prepares the input for the reasoning function by integrating user input, profile data,
        additional context, and optional action results.

        This method combines the raw user input with user profile information and context (if available)
        to create a single input ready for processing by the reasoning function. Subclasses may override
        this method to apply specific transformations or validations based on interaction requirements.

        Args:
            task_input (Any): Task-level metadata or input context
            user_input (Any): The raw input provided by the user.
            user_profile (Any): User profile data, if available, for personalizing the interaction.
            context (Any): Additional contextual information relevant to the interaction, if any.
            action_results (Any): Optional result of a previous action that may influence further responses.
            attachments (Sequence): Optional attachments to include.
            knowledge (Dict[str, str]): Resolved knowledge dict from knowledge_provider.
                Ignored by base implementation; PromptBasedAgent merges it into the prompt feed.

        Returns:
            ReasonerInput: The combined input, structured and ready for processing by the reasoning function.
        """
        return join_(
            task_input,
            user_input,
            user_profile,
            context,
            action_results,
            *(attachments if attachments is not None else ()),
            sep='\n\n'
        )

    def _parse_raw_response(self, raw_response: ReasonerResponse) -> Tuple[
        Union[str, AgentResponse],
        Union[AgentTaskStatusFlags, str, AgentStateItem, Any]
    ]:
        """
        Converts the raw response from the reasoning function into a user-friendly format
        and determines the next agent state.

        This method takes the raw response output by the reasoning function and parses it
        into a format suitable for user interaction. It also assesses whether the conversation
        should continue, be completed, or await further input.

        Args:
            raw_response (Any): The unformatted response generated by the reasoning function.

        Returns:
            Tuple[str, Union[AgentTaskStatusFlags, str, Any]]:
                - str: A user-friendly formatted response.
                - Union[AgentStates, str, Any]: An agent state indicating the interaction's next step.
                    Can be one of `AgentStates` or a custom state defined by subclasses.
        """
        return raw_response, AgentTaskStatusFlags.Completed

    def _get_next_actions_from_response(self, response: Any):
        next_action = get_(response, 'next_actions', default=None)
        return response if next_action is None else next_action

    # region actor related methods

    @staticmethod
    def _is_stateless_actor(actor):
        return not (
                (
                        hasattr(actor, 'set_state')
                        and callable(actor.get_state)
                        and hasattr(actor, 'get_state')
                        and callable(actor.set_state)
                )
                or hasattr(actor, 'state')
        )

    @staticmethod
    def _set_actor_state(actor, state):
        if hasattr(actor, 'set_state') and callable(actor.set_state):
            actor.set_state(state)
        elif hasattr(actor, 'state'):
            actor.state = state
        else:
            raise ValueError(
                "Actor does not have a callable `set_state` or an attribute `state`."
            )

    @staticmethod
    def _get_actor_state(actor):
        if hasattr(actor, 'get_state') and callable(actor.get_state):
            return actor.get_state()
        elif hasattr(actor, 'state'):
            return actor.state
        else:
            raise ValueError(
                "Actor does not have a callable `get_state` or an attribute `state`."
            )

    def _get_all_actor_states(self):
        if self.actor is None:
            # certain agents might not have an actor (such as those for responses and memories)
            return None
        if isinstance(self.actor, MultiActionExecutor):
            # MultiActionExecutor has built-in state management
            return self.actor.executor_states
        elif callable(self.actor):
            # Single callable actor assumed stateless
            return None
        else:
            raise ValueError(f"Actor must be a Callable or MultiActionExecutor; got {type(self.actor)}")

    def _resolve_actor_for_next_action(self, next_action):
        # NOTE: `self.actor_state` might be empty if this method is called before agent execution.
        #  For example, when `self._run_single_action` is invoked before agent execution as an initial action
        #  to bring the actor to the right state, but not an official action in the agent execution loop.

        if isinstance(self.actor, MultiActionExecutor):
            next_action_type = (
                next_action.type if isinstance(next_action, AgentAction)
                else next_action
            )
            actor = self.actor.resolve(next_action_type)
            # Get actor state using MultiActionExecutor's built-in state management
            actor_state = self.actor.get_state(next_action_type)
            if isinstance(actor, Agent):
                actor = partial(actor, trigger_action=next_action)
            if not self._is_stateless_actor(actor):
                self._set_actor_state(actor, actor_state)
        elif callable(self.actor):
            # Single callable actor assumed stateless, no state management needed
            actor = self.actor
        else:
            raise ValueError(f"Actor must be a Callable or MultiActionExecutor; got {type(self.actor)}")

        return actor

    def _create_actor_args(self, raw_action_items, attachments=None, **kwargs) -> Mapping:
        actor_args = dict_(raw_action_items, key_transformation=self.actor_args_transformation)
        if attachments is not None:
            actor_args[self.task_input_field_attachments] = attachments
        actor_args.update(kwargs)
        return actor_args

    # endregion

    @staticmethod
    def _resolve_task_input_field(user_input: Any, task_input: Dict, field_name: str, default: Any):
        if not task_input:
            return default
        field_value = task_input.pop(field_name, default)

        if is_none_or_empty_str(field_value):
            return None
        elif callable(field_value):
            return field_value(user_input)
        else:
            return field_value

    def copy(self, clear_states: bool = True) -> 'Agent':
        """
        Create a copy of the agent.

        Args:
            clear_states: If True, clears states and actor_state in the copy.
                         If False, preserves states from the original agent.

        Returns:
            A new Agent instance with copied attributes.

        Note:
            Parent-child debuggable relationships for actors are established
            when actors are triggered (in _run_single_action), not during copying.
        """
        new_agent = copy(self)

        new_agent.parent_debuggables = []
        new_agent._active_last_node = None
        new_agent.new_id()

        if clear_states:
            new_agent.states = AgentStates()
            new_agent.actor_state = None
        elif self.states:
            new_agent.states = copy(self.states)

        # Copy MultiActionExecutor to avoid sharing with original agent
        if isinstance(new_agent.actor, MultiActionExecutor):
            new_agent.actor = new_agent.actor.copy(clear_states=clear_states)

        return new_agent

    def _construct_reasoner_inference_config(self) -> ReasonerInferenceConfig:
        """
        Constructs configuration for reasoner inference.

        This method can be overridden by subclasses to provide specific inference
        configurations such as temperature, max_tokens, model selection, etc.

        Returns:
            ReasonerInferenceConfig: Configuration for reasoner inference, or None for defaults
        """
        return None

    def _run_single_action(
            self,
            action: Union[AgentAction, Any],
            task_input,
            agent_state: Union[AgentStateItem, Any],
            agent_response: Union[AgentResponse, Any],
            previous_action_results,
            task_input_metadata: Dict = None,
            attachments: Sequence = None
    ):
        if action.type.startswith('UserInputsRequired.'):
            self.log_info(f"Detected user input required action: {action.type}")
            self.log_info(f"User input required reasoning: {action.reasoning}")

            self.states.set_last_action(action, self.anchor_action_types)
            question = action.target

            # Construct response list: [instant_response, question]
            agent_response.instant_response = [
                agent_response.instant_response,
                question
            ]
            self.log_info(
                f"Constructed user input required response with {len(agent_response.instant_response)} parts:\n"
                f"  [0] instant_response: {agent_response.instant_response[0]}\n"
                f"  [1] question: {question}"
            )

            # Wrap with metadata for multi-session support
            if task_input_metadata is None:
                task_input_metadata = {}
            response_with_metadata = {
                **task_input_metadata,
                'response': agent_response.instant_response
            }

            # Determine input_mode from mapping based on action subtype
            subtype = action.type.split('.', 1)[1] if '.' in action.type else ''
            input_mode_entry = self.user_input_mode_mapping.get(subtype)
            if callable(input_mode_entry):
                input_mode = input_mode_entry(action)
            else:
                input_mode = input_mode_entry  # InputModeConfig or None

            self.log_info("Sending user input required response to user (is_pending=True)")
            if self.interactive is not None:
                self.interactive.send_response(
                    response=response_with_metadata,
                    flag=InteractionFlags.PendingInput,
                    input_mode=input_mode,
                )

            # Get follow-up input and extract metadata for multi-session support
            self.log_info("Waiting for user's response...")
            raw_input = self.interactive.get_input()
            self.log_info(f"Received user response: {raw_input}")

            follow_up_user_input, updated_metadata, _ = self._extract_user_input_and_metadata(
                raw_input, existing_metadata=task_input_metadata
            )
            # Update task_input_metadata for next iteration
            task_input_metadata.update(updated_metadata)

            self.log_info(f"Extracted follow-up user input: {follow_up_user_input}")
            self.log_info("User input required flow complete, continuing to next reasoning iteration")

            # Determine output action results.
            # Unlike normal actions which produce a fresh result via an actor call,
            # UserInputsRequired must construct its output from prior state.
            # Two cases: copilot (user changed the page) needs fresh capture via no_op;
            # non-copilot uses the last accumulated snapshot (matches live state).
            is_copilot = False
            if input_mode is not None and getattr(input_mode, 'options', None):
                selected_opt = next(
                    (o for o in input_mode.options if o.value == follow_up_user_input),
                    None
                )
                if selected_opt is not None:
                    is_copilot = getattr(selected_opt, 'needs_user_copilot', False)
                else:
                    follow_up_options = [
                        o for o in input_mode.options if getattr(o, 'follow_up_prompt', '')
                    ]
                    if follow_up_options:
                        is_copilot = any(
                            getattr(o, 'needs_user_copilot', False) for o in follow_up_options
                        )
                    else:
                        is_copilot = any(
                            getattr(o, 'needs_user_copilot', False) for o in input_mode.options
                        )

            if is_copilot and isinstance(self.actor, MultiActionExecutor):
                self.log_info(
                    "needs_user_copilot=True: capturing fresh page state "
                    "via no_op after user browser interaction"
                )
                try:
                    default_actor = self.actor.resolve(self.actor.default_key)
                    output_action_results = default_actor(
                        action_type='no_op'
                    )
                except Exception as e:
                    self.log_error(
                        f"Failed to capture fresh page state after copilot: {e}"
                    )
                    raise e
            else:
                # Use the last snapshot from accumulated results (matches live state).
                # previous_action_results may be an accumulated list from sequential
                # actions earlier in this WorkGraph (via _merge_action_results_to_list).
                if isinstance(previous_action_results, (list, tuple)):
                    output_action_results = previous_action_results[-1] if previous_action_results else None
                else:
                    output_action_results = previous_action_results

            return (
                WorkGraphStopFlags.Terminate,  # graph's stop flag
                {
                    self.task_input_field_user_input: follow_up_user_input,
                    self.task_input_field_action_results: output_action_results
                }
            )
        else:
            # Wrap with metadata for multi-session support
            if task_input_metadata is None:
                task_input_metadata = {}
            response_with_metadata = {
                **task_input_metadata,
                'response': agent_response.instant_response
            }
            if self.interactive is not None:
                self.interactive.send_response(
                    response=response_with_metadata,
                    flag=InteractionFlags.MessageOnly
                )

            # NOTE: the actor will be set at its state in `self.actor_state` when it is resolved
            actor: Callable = self._resolve_actor_for_next_action(action)
            if isinstance(actor, Debuggable):
                actor.set_parent_debuggable(self)
            action.source = getattr_(actor, 'source', None)
            self.states.set_last_action(action, self.anchor_action_types)

            actor_args = self._create_actor_args(
                raw_action_items=action,
                task_input=task_input,
                action_results=previous_action_results,
                attachments=attachments
            )
            actor_args = get_relevant_named_args(actor, **actor_args)

            task_status_description_extended = mapping_to_xml(
                {
                    snake_to_camel_case(remove_prefix(k, 'action_')): v
                    for k, v in actor_args.items()
                    if v is not None and k in ('action_type', 'action_target', 'action_args')
                },
                root_tag='Action'
            )
            agent_state.task_status_description_extended = task_status_description_extended
            try:
                # ACTION EXECUTION: Call the actor and capture raw operational results
                # The actor performs the actual work (e.g., web scraping, API calls, etc.)
                # and returns new_action_results containing raw operational data
                # Example: {'body_html_before': '<div>...', 'body_html_after': '<div>...',
                #           'is_follow_up': False, 'source': 'https://...'}
                new_action_results = actor(**actor_args)
            except Exception as action_error:
                self.log_error(action_error, AgentLogTypes.ActionError)
                raise action_error
            set_(action, 'result', new_action_results)
            self.log_info(new_action_results, AgentLogTypes.ActionResult, artifacts_as_parts=True, parts_min_size=0)

            # Return the raw action_results to be passed to the next i teration
            # This will become the action_results parameter in the next reasoner call
            return (
                WorkGraphStopFlags.Continue,
                {
                    self.task_input_field_user_input: None,
                    self.task_input_field_action_results: new_action_results  # Raw operational data
                }
            )

    def _get_agent_results(self, trigger_action, trigger_action_results, new_states):
        """
        Transform raw operational data into presentation-ready results.

        This method is called ONLY when the agent completes (agent_state == Completed).
        It transforms action_results (raw operational data) and execution states into
        agent_results (structured, human-readable output).

        TRANSFORMATION EXAMPLE:
        ----------------------
        Input (from action_results and states):
            action_results = {
                'body_html_before_last_action': '<body tabindex="-1" class="sk-client-md...',
                'body_html_after_last_action': '<body tabindex="-1" class="sk-client-theme...',
                'is_cleaned_body_html_only_incremental_change': False,
                'is_follow_up': False,
                'source': 'https://app.slack.com/client/T08TOPUOL57/C0875GJL7PV'
            }

        Output (agent_results):
            AgentActionResult(
                action=None,
                anchor_action=None,
                details='<div id="41" class="p-ia4_client...',
                source='https://app.slack.com/client/T08TOPUOL57/C0875GJL7PV',
                summary='''markdown## Slack Channel Summary. #seo-link-building-news-updates
                ## November 14-23, 2025

                ### Overview
                This channel saw active participation from link building professionals
                sharing available sites, pricing, and collaboration opportunities. Key co...'''
            )

        KEY DISTINCTIONS:
        ----------------
        - action_results: Machine-oriented (raw HTML, flags, internal state)
        - agent_results: Human-oriented (markdown summaries, structured details)

        USAGE:
        ------
        - Called at: Line 1138 when agent completes
        - Used at: Line 1154-1159 to augment final response with summaries
        - Returned: Line 1172 as the final return value of __call__()

        Args:
            trigger_action: The initial action that triggered this agent execution (if any)
            trigger_action_results: Results from the trigger action
            new_states: List of AgentStateItems created during this execution

        Returns:
            Structured, presentation-ready results (e.g., AgentActionResult objects)
            or None if no special formatting is needed. The base implementation returns None,
            allowing subclasses to define their own transformation logic.
        """
        pass

    def _default_summarizer(self, *args, **kwargs):
        """
        Default summarizer that merges results from parallel branches.
        Returns a dict with combined action_results.
        """
        # When called after parallel execution, receives results from all branches
        # Return a minimal result dict that the agent expects
        return {
            self.task_input_field_user_input: None,
            self.task_input_field_action_results: args[0] if args else None
        }

    def _merge_action_results_to_list(self, result, shared_list, *_args, **_kwargs):
        """
        Extract action_results from result dict and append to shared list.

        Used as result_pass_down_mode callable for sequential WorkGraph nodes.
        Designed to work with partial(self._merge_action_results_to_list, shared_list=...).

        Why this design:
        - _run_single_action returns: {'user_input': None, 'action_results': new_action_results}
        - We need JUST the action_results field, not the entire dict
        - Returns None to leverage in-place mutation pattern (keeps original args/kwargs)
        - The shared list accumulates results from all sequential action nodes

        Args:
            result: Result dict from _run_single_action
            shared_list: The list to accumulate action_results into (updated in-place)
            *_args, **_kwargs: Original args/kwargs (unused, passed through by WorkGraph)

        Returns:
            None (keeps original args/kwargs, shared_list mutated in-place)
        """
        if isinstance(result, dict) and self.task_input_field_action_results in result:
            action_results_data = result[self.task_input_field_action_results]
            append_(action_results_data, arr=shared_list)
        return None

    def _make_attachments(self, base_obj) -> Sequence[AgentAttachment]:
        pass

    def _parse_base_action(self, base_action: Any) -> Tuple[
        Union[str, AgentResponse],
        Union[AgentTaskStatusFlags, str, AgentStateItem, Any]
    ]:
        """
        Parse base_action when it's not a StructuredResponse string.

        This method is called on the first iteration when base_action is configured
        but is not a StructuredResponse XML string. Subclasses must override this to
        handle custom base_action formats (e.g., dict, object, etc.).

        Args:
            base_action: The configured base_action value (can be any type)

        Returns:
            Tuple[Union[str, AgentResponse], Union[AgentTaskStatusFlags, str, AgentStateItem, Any]]:
                - First element: agent_response (formatted response or AgentResponse object)
                - Second element: agent_state (state indicating next step)

        Example:
            If base_action is a dict like {"action": "navigate", "url": "https://..."},
            this method should convert it to (AgentResponse, AgentStateItem) format.
        """
        pass

    def _finalize_and_send_agent_results(
        self,
        agent_response: Union[AgentResponse, Any],
        agent_results: Any,
        task_input_metadata: Dict,
        completion_reason: AgentCompletionReason
    ):
        """
        Finalize agent results and send the final response to the user.

        This method encapsulates the common logic for formatting and sending the final response
        when the agent completes execution, either normally (agent_state == Completed) or
        due to reaching max_num_loops limit.

        Args:
            agent_response: The agent response object containing instant_response (must not be None)
            agent_results: Structured results from _get_agent_results()
            task_input_metadata: Metadata dict to attach to response (e.g., session_id)
            completion_reason: Reason for completion (Normal or MaxLoops)

        Raises:
            ValueError: If completion_reason is not a valid AgentCompletionReason
        """
        # AUGMENT RESPONSE: Enhance the final response with agent_results summaries
        # The agent_results (human-readable summaries) are appended to the instant_response
        # to provide the user with structured output beyond the raw response text
        agent_response_string = agent_response if isinstance(agent_response, str) else agent_response.instant_response

        if agent_results:
            agent_response_string = agent_response_string.replace('<html>', '').replace('</html>', '').replace(
                '<body>', '').replace('</body>', '')
            # Extract summary field from each agent_result_item (e.g., markdown summaries)
            for agent_result_item in iter_(agent_results):
                try:
                    agent_response_string += '\n' + agent_result_item.summary
                except:
                    agent_response_string += '\n' + str(agent_result_item)
            agent_response_string = '<html><body>' + agent_response_string + '</body></html>'

        # Attach metadata from task_input (like session_id) to response
        # This enables multi-session support for queue-based systems
        response_with_metadata = {
            **task_input_metadata,
            'response': agent_response_string
        }

        if self.interactive is not None:
            self.interactive.send_response(
                response=response_with_metadata,
                flag=InteractionFlags.TurnCompleted  # Final response - agent is done
            )

        # Log completion with appropriate message based on completion reason
        if completion_reason == AgentCompletionReason.Normal:
            log_message = f"Agent work stream completed at node `{self._active_last_node.id}`."
        elif completion_reason == AgentCompletionReason.MaxLoops:
            log_message = f"Agent work stream completed due to max_num_loops limit at node `{self._active_last_node.id}`."
        else:
            raise ValueError(f"Invalid completion reason: {completion_reason}. Expected {AgentCompletionReason.Normal} or {AgentCompletionReason.MaxLoops}.")

        self._active_last_node.log_info(log_message, LOG_TYPE_AGENT_WORKSTREAM_COMPLETED)

    @staticmethod
    def _resolve_task_input_from_call_args(args: tuple, kwargs: dict) -> Any:
        """
        Resolve task_input from __call__ arguments, unwrapping nested task_input dicts.

        Handles special case where Agent is wrapped in WorkGraphNode with partially filled
        arguments. When an agent is used as a callable in a WorkGraphNode, some of its __call__
        arguments may be pre-filled (e.g., user_input). To pass additional arguments from parent
        agents, they are wrapped in a 'task_input' dict. This method unwraps such nested dicts
        to avoid double-wrapping while preserving pre-filled arguments (which take priority).

        Args:
            args: Positional arguments from __call__
            kwargs: Keyword arguments from __call__

        Returns:
            Resolved task_input with nested task_input dicts unwrapped, or None if no args/kwargs
        """
        # Handle the case where agent is called with no arguments (e.g., agent())
        # solve_as_single_input() raises ValueError for empty args and kwargs
        if not args and not kwargs:
            return None

        task_input = solve_as_single_input(*args, **kwargs)

        # Special treatment in case the Agent has arguments partially filled
        # Agent might be wrapped in a WorkGraphNode with some of its `__call__`'s arguments partially filled;
        #   for example, the `user_input` argument filled by a planning agent.
        # As the base class, Agent does not know whether this would happen,
        #   and has to pass parent up-stream Agent's arguments like `user_profile`, `context` or those from `task_input_metadata`
        #   inside a `task_input` dict.
        # Here we unwrap the `task_input`, but the arguments inside it has lower priority than the pre-filled partial arguments.
        # We forbit a child class to fill the `task_input` argument. This argument is preserved for up-stream Agent to pass down arguments.
        if isinstance(task_input, Dict) and 'task_input' in task_input:
            _task_input = task_input['task_input']
            if isinstance(_task_input, Dict):
                task_input = dict(task_input)
                for k, v in _task_input.items():
                    if k not in task_input:
                        task_input[k] = v
                del task_input['task_input']

        return task_input

    def __call__(self, *args, **kwargs):
        """
        Manages the full interaction cycle given a user input, and deliver a response.

        This method orchestrates each step in the interaction flow:
        1. Captures user input through `interactive`.
        2. Retrieves or generates user profile data and context, if applicable.
        3. Passes processed input to `reasoner` for generating a raw response.
        4. Parses the raw response using `_parse_raw_response`.
        5. Sends the final response to the user and evaluates if further input is required.

        Subclasses can override `start` to adjust the interaction flow, logging, or other
        custom requirements for more specialized conversational behavior.

        BRANCHED RECURSION ARCHITECTURE:
        ================================
        This method implements branched recursion when the reasoner requests parallel actions.

        Execution Flow:
        - The agent runs in a while True loop, calling the reasoner repeatedly
        - When reasoner returns parallel actions [A, B, C], the agent:
          1. Creates a copy of itself for each action (branched_agent)
          2. Wraps each copy in a WorkGraphNode (branched_agent_node)
          3. When branched_agent_node executes, it calls branched_agent.__call__()
             This is the RECURSIVE CALL - starting a new agent loop
          4. Each branched agent can independently:
             - Make decisions via its own reasoner
             - Execute actions
             - Create MORE branched agents (recursion depth is unbounded)
          5. All branches merge at a summary_node

        Simple Recursion Tree:
            main_agent.__call__()
                └─ reasoner → [ActionA, ActionB]
                    ├─ branched_agent_A.__call__()  [RECURSIVE]
                    │   └─ reasoner → [ActionA1, ActionA2]
                    │       ├─ branched_agent_A1.__call__()  [DEEPER RECURSION]
                    │       └─ branched_agent_A2.__call__()  [DEEPER RECURSION]
                    └─ branched_agent_B.__call__()  [RECURSIVE]
                        └─ reasoner → []  (completes)

        DETAILED EXAMPLE: Complex Multi-Step Execution
        ===============================================
        Consider a research agent that needs to: search → analyze results → write report

        next_actions format: List of action groups
        - Single-element group [Action1] → executes sequentially
        - Multi-element group [ActionA, ActionB, ActionC] → executes in parallel

        Execution Timeline:

        MAIN AGENT (depth=0):
        ├─ Iteration 1: while True loop starts
        │  ├─ reasoner(user_input="Research quantum computing applications")
        │  └─ returns: next_actions = [[SearchAction]]  ← single action, sequential
        │     └─ WorkGraph: SearchAction → (loop continues)
        │
        ├─ Iteration 2: loop continues with search results
        │  ├─ reasoner(action_results=[search_results])
        │  └─ returns: next_actions = [[AnalyzeAction1, AnalyzeAction2, AnalyzeAction3]]
        │     ↑ Multiple actions in one group → PARALLEL EXECUTION
        │     └─ WorkGraph creates 3 branches:
        │        ├─ Branch A: AnalyzeAction1 → branched_agent_A.__call__() → summary
        │        ├─ Branch B: AnalyzeAction2 → branched_agent_B.__call__() → summary
        │        └─ Branch C: AnalyzeAction3 → branched_agent_C.__call__() → summary
        │
        │  Each branched agent runs independently (RECURSIVE CALLS):
        │
        │  BRANCHED_AGENT_A (depth=1):
        │  ├─ Executes AnalyzeAction1 (analyze quantum algorithms)
        │  ├─ Iteration 1: reasoner(action_results=[analysis_results_1])
        │  └─ returns: next_actions = [[DetailAction1, DetailAction2]]
        │     └─ Creates 2 MORE branches (depth=2):
        │        ├─ branched_agent_A1.__call__()  [DEEPER RECURSION]
        │        │  ├─ Executes DetailAction1
        │        │  ├─ Iteration 1: reasoner() → next_actions = []
        │        │  └─ Returns: analysis_detail_1 ✓
        │        └─ branched_agent_A2.__call__()  [DEEPER RECURSION]
        │           ├─ Executes DetailAction2
        │           ├─ Iteration 1: reasoner() → next_actions = []
        │           └─ Returns: analysis_detail_2 ✓
        │     └─ summary merges [analysis_detail_1, analysis_detail_2]
        │  └─ Returns: comprehensive_analysis_A ✓
        │
        │  BRANCHED_AGENT_B (depth=1):
        │  ├─ Executes AnalyzeAction2 (analyze quantum hardware)
        │  ├─ Iteration 1: reasoner(action_results=[analysis_results_2])
        │  └─ returns: next_actions = []  ← No further actions needed
        │  └─ Returns: analysis_B ✓
        │
        │  BRANCHED_AGENT_C (depth=1):
        │  ├─ Executes AnalyzeAction3 (analyze use cases)
        │  ├─ Iteration 1: reasoner(action_results=[analysis_results_3])
        │  └─ returns: next_actions = [[RefineAction]]
        │     └─ WorkGraph: RefineAction → (continues)
        │  ├─ Iteration 2: reasoner(action_results=[refined_results])
        │  └─ returns: next_actions = []
        │  └─ Returns: analysis_C ✓
        │
        │  └─ All branches complete → summary_node merges:
        │     [comprehensive_analysis_A, analysis_B, analysis_C]
        │     → combined_analysis
        │
        ├─ Iteration 3: MAIN AGENT continues with combined_analysis
        │  ├─ reasoner(action_results=[combined_analysis])
        │  └─ returns: next_actions = [[WriteReportAction]]
        │     └─ WorkGraph: WriteReportAction → (loop continues)
        │
        ├─ Iteration 4: loop continues with report
        │  ├─ reasoner(action_results=[report])
        │  └─ returns: next_actions = []  ← Task complete!
        │     └─ agent_state = Completed
        │
        └─ EXIT: return agent_results  (line 629)

        Key Observations:
        - MAIN AGENT had 4 iterations of while True loop
        - Iteration 2 spawned 3 parallel branches (depth=1)
        - Branch A spawned 2 MORE branches (depth=2)
        - Branch C had 2 iterations in its own loop
        - Total recursive agent calls: 1 (main) + 3 (depth-1) + 2 (depth-2) = 6 agents
        - Each agent independently decides when to complete (via its reasoner)
        - The recursion depth is unbounded - branches can create branches infinitely

        MIXED SEQUENTIAL AND PARALLEL EXAMPLE:
        ======================================
        next_actions = [
            [Action1],              # Group 1: sequential
            [Action2, Action3],     # Group 2: parallel (2 branches)
            [Action4]               # Group 3: sequential
        ]

        WorkGraph structure created:
            Action1 → [Action2, Action3] → summary → Action4
                      (parallel)          (merge)

        Execution order:
        1. Action1 executes
        2. Action2 and Action3 execute in parallel (via branched agents)
        3. summary_node merges results from Action2 and Action3
        4. Action4 executes with merged results
        5. Main agent loop continues

        This shows how action groups are CHAINED together:
        - Single-action groups create sequential nodes
        - Multi-action groups create parallel branches with summary
        - Groups are connected in order: prev_node → current_group → becomes prev_node

        Termination: Each recursive call terminates when its reasoner returns no next_actions,
        setting agent_state to Completed and breaking the while True loop.

        Returns:
            None
        """

        # Reset workflow control and status for fresh execution
        # Each __call__ starts with Continue control and Running status
        self._control = AgentControls.Continue
        self._status = CommonWorkflowStatus.Running

        # region STEP1: resolve user_input and preserve extra fields
        task_input = self._resolve_task_input_from_call_args(args, kwargs)

        if task_input:
            max_num_loops = task_input.pop(self.task_input_field_max_num_loops, self.max_num_loops)
            attachments = task_input.pop(self.task_input_field_attachments, [])
            task_label = task_input.pop(DEFAULT_AGENT_TASK_INPUT_FIELD_TASK_LABEL, None)

            # PREVIOUS AGENT RESULTS → ATTACHMENTS CONVERSION:
            # When this agent is a node in a PromptBasedActionPlanningAgent's WorkGraph,
            # the previous node's result is passed via `result_pass_down_mode='previous_agent_results'`.
            # This injects the result as a named kwarg: previous_agent_results=<result_from_previous_node>
            #
            # Flow:
            #   1. PlanningAgent creates WorkGraph with result_pass_down_mode=task_input_field_previous_agent_results
            #   2. Node A completes → result passed to Node B as kwargs['previous_agent_results']
            #   3. Node B (this agent) extracts it here from task_input
            #   4. _make_attachments() converts the result into AgentAttachment objects
            #   5. Attachments are included in _construct_reasoner_input() for the LLM to see
            #
            # This enables chained agents to receive context from their predecessors in the dependency graph.
            previous_agent_results = task_input.pop(self.task_input_field_previous_agent_results, None)
            if previous_agent_results is not None:
                attachments.extend(self._make_attachments(previous_agent_results))
        else:
            max_num_loops = self.max_num_loops
            task_label = None
            attachments = []

        # Preserve extra fields from task_input (like session_id) to attach to response
        task_input_metadata = {}

        # Determine input source: explicit argument vs interactive queue
        if task_input:
            # Case 1: Direct string input (e.g., agent("hello"))
            if isinstance(task_input, str):
                user_input = task_input
                task_input = None
                self.log_debug(f"Using explicit string input: {user_input}", 'UserInput')
            # Case 2: Dict with user_input field (e.g., agent({"user_input": "hello", "session_id": "..."}))
            elif isinstance(task_input, Dict) and self.task_input_field_user_input in task_input:
                # Use utility method to extract user_input, metadata, and agent fields
                user_input, task_input_metadata, agent_fields = self._extract_user_input_and_metadata(task_input)
                # Keep only agent-specific fields in task_input for downstream processing
                task_input = agent_fields if agent_fields else None
                self.log_debug(
                    f"Using explicit dict input: user_input={user_input}, metadata={task_input_metadata}",
                    'UserInput'
                )
            # Case 3: Invalid explicit input - log error and fall back to queue
            else:
                self.log_error(
                    f"Invalid explicit task_input: expected string or dict with '{self.task_input_field_user_input}' field, "
                    f"got {type(task_input).__name__}: {task_input}. Falling back to interactive.get_input().",
                    'UserInput'
                )
                # Fall back to reading from interactive queue
                if self.interactive is None:
                    raise ValueError(
                        f"Cannot get user input: task_input is invalid (type={type(task_input).__name__}, value={task_input}) "
                        f"and self.interactive is None. Either provide valid task_input (string or dict with "
                        f"'{self.task_input_field_user_input}' field) or configure self.interactive."
                    )
                user_input, task_input_metadata, _ = self._extract_user_input_and_metadata(
                    self.interactive.get_input()
                )
                self.log_debug(f"Read from interactive queue (invalid explict input): {user_input}", 'UserInput')
                task_input = None
        else:
            # Case 4: No explicit input - read from interactive queue (e.g., agent())
            if self.interactive is None:
                raise ValueError(
                    f"Cannot get user input: no explicit task_input provided and self.interactive is None. "
                    f"Either call agent with input (e.g., agent('message') or agent({{'{self.task_input_field_user_input}': 'message'}})) "
                    f"or configure self.interactive."
                )
            self.log_debug("Reading input from interactive queue (no explicit task_input)", 'UserInput')
            # Get input and extract metadata
            user_input, task_input_metadata, _ = self._extract_user_input_and_metadata(
                self.interactive.get_input()
            )
            self.log_debug(f"Read from interactive queue: {user_input}", 'UserInput')

        # endregion

        # region STEP2: resolve other agent args
        if task_input:
            trigger_action = self._resolve_task_input_field(
                user_input=user_input,
                task_input=task_input,
                field_name=self.task_input_field_trigger_action,
                default=None
            )
            user_profile = self._resolve_task_input_field(
                user_input=user_input,
                task_input=task_input,
                field_name=self.task_input_field_user_profile,
                default=self.user_profile
            )
            context = self._resolve_task_input_field(
                user_input=user_input,
                task_input=task_input,
                field_name=self.task_input_field_context,
                default=self.context
            )
            action_results = self._resolve_task_input_field(
                user_input=user_input,
                task_input=task_input,
                field_name=self.task_input_field_action_results,
                default=None
            )
            self.states = self._resolve_task_input_field(
                user_input=user_input,
                task_input=task_input,
                field_name=self.task_input_field_agent_states,
                default=self.states
            )
        else:
            trigger_action = action_results = None
            user_profile = self.user_profile
            context = self.context
            task_input = {}

        # Resolve knowledge_provider (callable or static dict)
        knowledge = {}
        if self.knowledge_provider is not None:
            if callable(self.knowledge_provider):
                knowledge = self.knowledge_provider(user_input)
            elif isinstance(self.knowledge_provider, dict):
                knowledge = self.knowledge_provider

        if self.states is None:
            self.states = AgentStates()
            if trigger_action is not None:
                self.states.set_last_action(trigger_action)
        num_states_when_starting = len(self.states)

        # region Retrieve the actor states at the beginning.
        # NOTE: We assume the actor is at the right state at this moment (right before this agent executes).
        #  The actor(s) should return to this state before their action.
        self.actor_state = self._get_all_actor_states()
        # endregion

        self.log_debug(user_input, 'UserInput')
        self.log_debug(user_profile, 'UserProfile')
        self.log_debug(context, 'Context')
        self.log_debug(action_results, 'ActionResults')
        self.log_debug(task_input, 'TaskInput')

        # endregion

        # region STEP3: starts looping the agent
        # This while True loop is the heart of the agent's execution
        # It continues until the reasoner returns agent_state == Completed (no more next_actions)

        # Initialize result tracking variables:
        # - action_results: Raw operational data from the most recent action execution
        #                   Example: {'body_html_before': '<div>...', 'is_follow_up': False, 'source': 'https://...'}
        #                   Purpose: Feeds into the next reasoner call as context (feedback loop)
        #                   Updated: After each action execution (line 709, 1332-1334)
        #                   Machine-oriented: Used internally for agent's decision-making
        #
        # - agent_results: Structured, human-readable summary of the entire agent execution
        #                  Example: AgentActionResult(summary='markdown## Summary...', details='<div>...')
        #                  Purpose: Final presentation-ready output returned to caller
        #                  Computed: Only when agent_state == Completed (line 1116-1120)
        #                  Human-oriented: Used in final response to user (line 1129-1134)
        agent_results = None
        trigger_action_results = action_results
        self._active_last_node = self
        loop_count = 0
        agent_response = None  # Will be set during reasoning; initialized for defensive programming

        while True:
            loop_count += 1

            # Check workflow control signal
            if not self._check_workflow_control():
                return agent_results  # Early exit without completion processing

            # Check if max loops reached
            if max_num_loops > 0 and loop_count > max_num_loops:
                self.log_info(f"Max loops limit reached ({max_num_loops}). Wrapping current results and completing.")
                completion_reason = AgentCompletionReason.MaxLoops
                break

            # Flag to track if _parse_raw_response should be skipped
            skip_parse_raw_response = False

            # BASE ACTION HANDLING: Skip reasoner on first iteration if base_action is configured
            if loop_count == 1 and self.base_action is not None:
                # Check if base_action is a StructuredResponse string
                reasoner_input = None
                if isinstance(self.base_action, str) and '<StructuredResponse>' in self.base_action:
                    # Use as raw_response (will be parsed by _parse_raw_response below)
                    self.log_info("First iteration with base_action (StructuredResponse string) - skipping reasoner")
                    raw_response: ReasonerResponse = self.base_action
                    self.log_info(raw_response, 'BaseActionRawResponse', artifacts_as_parts=True, parts_min_size=0)
                else:
                    # Use _parse_base_action for custom formats
                    self.log_info("First iteration with base_action (custom format) - calling _parse_base_action")
                    agent_response, agent_state = self._parse_base_action(self.base_action)
                    skip_parse_raw_response = True
            else:
                # FEEDBACK LOOP: Construct reasoner input with action_results from previous iteration
                # The action_results (raw operational data) from the last action execution is fed
                # into the reasoner to inform its next decision. This creates a feedback loop:
                # reasoner → action → action_results → reasoner → action → ...
                # Example: action_results might contain HTML changes, status flags, etc.
                reasoner_input: ReasonerInput = self._construct_reasoner_input(
                        task_input=task_input_metadata,
                        user_input=user_input,
                        user_profile=user_profile,
                        context=context,
                        action_results=action_results,  # Raw data from previous action (or None on first iteration),
                        attachments=attachments,
                        knowledge=knowledge
                )
                reasoner_inference_config: ReasonerInferenceConfig = self._construct_reasoner_inference_config()
                self.log_info(reasoner_input, 'ReasonerInput', is_artifact=True, parts_min_size=0)
                self.log_debug(reasoner_inference_config, 'ReasonerInferenceConfig')

                self.log_info(f"Start reasoning with reasoner '{self.reasoner}'")

                # TODO: we need to retry the two steps
                raw_response: ReasonerResponse = self.reasoner(reasoner_input, reasoner_inference_config, **(self.reasoner_args or {}))
                self.log_info('End reasoning')
                self.log_info(raw_response, 'ReasonerResponse', artifacts_as_parts=True, parts_min_size=0)

            # Only parse raw_response if we didn't use _parse_base_action
            if not skip_parse_raw_response:
                agent_response, agent_state = self._parse_raw_response(raw_response)

            if isinstance(agent_response, str) and agent_state is None:
                # direct response - wrap with metadata for multi-session support
                response_with_metadata = {
                    **task_input_metadata,
                    'response': agent_response
                }
                if self.interactive is not None:
                    self.interactive.send_response(response=response_with_metadata, flag=InteractionFlags.PendingInput)

                    # Get follow-up input and extract metadata
                    user_input, task_input_metadata, _ = self._extract_user_input_and_metadata(
                        self.interactive.get_input(),
                        existing_metadata=task_input_metadata
                    )
                self.log_info(agent_response, AgentLogTypes.AgentResponse, is_artifact=True, parts_min_size=0)
            else:
                # region special support for build-in AgentState class
                if isinstance(agent_state, AgentStateItem):
                    agent_state.last_action_source = self.states.last_action_source
                    agent_state.last_action_type = self.states.last_action_type
                    agent_state.last_anchor_action_type = self.states.last_anchor_action_type
                    agent_state.user_input = user_input
                    agent_state.reasoner_input = reasoner_input
                    agent_state.raw_response = raw_response
                    agent_state.response = agent_response
                    agent_state.task_label = task_label
                    if not agent_response.next_actions:
                        agent_state.task_status = AgentTaskStatusFlags.Completed
                else:
                    if not agent_response.next_actions:
                        agent_state = AgentTaskStatusFlags.Completed

                self.log_info(agent_response, AgentLogTypes.AgentResponse, artifacts_as_parts=True, parts_min_size=0)
                self.log_info(agent_state, AgentLogTypes.AgentState, artifacts_as_parts=True, parts_min_size=0)

                # endregion

                self.states.append(agent_state)

                # TERMINATION CONDITION: Check if the agent's work is complete
                # The loop breaks when agent_state == Completed (no more next_actions)
                if str(agent_state) == AgentTaskStatusFlags.Completed:
                    # Set completion reason and break to finalization block
                    completion_reason = AgentCompletionReason.Normal
                    break
                else:
                    # Agent needs to continue - execute the next actions
                    # Check workflow control before executing actions
                    if not self._check_workflow_control():
                        return agent_results

                    next_actions = self._get_next_actions_from_response(agent_response)
                    self.log_info(next_actions, AgentLogTypes.AgentNextActions)
                    if callable(next_actions):
                        # Send instant_response before executing the WorkGraph
                        # This ensures the planning agent's decomposition/reasoning is visible to the user
                        if task_input_metadata is None:
                            task_input_metadata = {}
                        response_with_metadata = {
                            **task_input_metadata,
                            'response': agent_response.instant_response
                        }
                        if self.interactive is not None:
                            self.interactive.send_response(
                                response=response_with_metadata,
                                flag=InteractionFlags.MessageOnly  # Work is still ongoing (WorkGraph execution)
                            )

                        # Execute the WorkGraph (e.g., from PromptBasedActionPlanningAgent)
                        action_results = next_actions(
                            task_input={
                                self.task_input_field_user_input: user_input,
                                self.task_input_field_user_profile: user_profile,
                                self.task_input_field_context: context,
                                self.task_input_field_action_results: action_results,
                                **task_input_metadata
                            }
                        )
                    else:
                        # Build a WorkGraph to execute actions
                        # next_actions is a list of action groups: [[A1], [B1, B2, B3], [C1]]
                        # - Single-element groups [A1] execute sequentially
                        # - Multi-element groups [B1, B2, B3] execute in parallel

                        # Filter out empty action groups to prevent creating a WorkGraph
                        # with no start nodes (which would return an empty tuple and crash)
                        next_actions = [group for group in next_actions if group]
                        if not next_actions:
                            continue  # No valid actions — loop back for next reasoning iteration

                        start_nodes = last_node = None
                        previous_action_results = list_(action_results)
                        for next_action_group in next_actions:
                            # CASE 1: Sequential execution - single action in the group
                            if len(next_action_group) == 1:
                                # Create a node that executes ONE action, then returns control
                                # Flow: action_node → (next iteration of this agent's while loop)
                                next_action = next_action_group[0]
                                action_node = WorkGraphNode(
                                    value=partial(
                                        self._run_single_action,
                                        action=next_action,
                                        task_input=task_input,
                                        agent_state=agent_state,
                                        agent_response=agent_response,
                                        previous_action_results=previous_action_results,
                                        task_input_metadata=task_input_metadata,
                                        attachments=attachments
                                    ),
                                    result_pass_down_mode=partial(
                                        self._merge_action_results_to_list,
                                        shared_list=previous_action_results
                                    ),
                                    copy_debuggable_config_from=self,
                                    id=f'$class-Actioner',
                                    enable_suffix_for_initial_id=True
                                )

                                # Chain sequential actions: action1 → action2 → action3 ...
                                if start_nodes is None:
                                    start_nodes = [action_node]
                                else:
                                    last_node.add_next(action_node)

                                last_node = action_node
                            else:
                                # CASE 2: Parallel execution - multiple actions in the group
                                # This implements BRANCHED RECURSION:
                                # After the immediate action, each parallel branch gets its own independent agent that can
                                # make decisions, execute actions, and even create more branches
                                # A summarizer is placed after all the branches concluded
                                # Flow: action_node → branched_agent_node → summary_node

                                # Create summary node to merge results from all parallel branches
                                summary_node = WorkGraphNode(
                                    value=(
                                        self.summarizer
                                        if self.summarizer is not None
                                        else self._default_summarizer
                                    ),
                                    result_pass_down_mode=partial(
                                        self._merge_action_results_to_list,
                                        shared_list=previous_action_results
                                    ),
                                    copy_debuggable_config_from=self,
                                    id=f'$class-Summarizer',
                                    enable_suffix_for_initial_id=True
                                )

                                action_nodes = []
                                for next_action in next_action_group:
                                    # Create a branched agent (independent copy) as a mean for parallel recursion
                                    branched_agent = self.copy(clear_states=self.branching_agent_start_as_new)

                                    # Create action_node to execute the immediate next action
                                    action_node = WorkGraphNode(
                                        value=partial(
                                            branched_agent._run_single_action,
                                            action=next_action,
                                            task_input=task_input,
                                            agent_state=agent_state,
                                            agent_response=agent_response,
                                            previous_action_results=previous_action_results,
                                            task_input_metadata=task_input_metadata,
                                            attachments=attachments
                                        ),
                                        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
                                        copy_debuggable_config_from=self,
                                        id=f'$class-Actioner',
                                        enable_suffix_for_initial_id=True
                                    )

                                    # Create branched_agent_node for RECURSIVE agent execution
                                    # This is the KEY to branched recursion
                                    # When this node executes, it calls branched_agent.__call__(),
                                    # which starts a COMPLETE NEW agent loop
                                    branched_agent_node = WorkGraphNode(
                                        branched_agent,  # Callable via __call__ method
                                        result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
                                        copy_debuggable_config_from=self,
                                        id=f'$class-Agent',
                                        enable_suffix_for_initial_id=True
                                    )

                                    # Wire up the branch: action → recursive agent → summary
                                    action_node.add_next(branched_agent_node)
                                    branched_agent_node.add_next(summary_node)
                                    action_nodes.append(action_node)

                                # Connect parallel branches to the workflow
                                if start_nodes is None:
                                    # This is the first action group - set as start nodes
                                    start_nodes = action_nodes
                                else:
                                    # Chain this parallel group after the previous node
                                    # Creates fork: last_node branches into multiple action_nodes
                                    for action_node in action_nodes:
                                        last_node.add_next(action_node)

                                # After parallel execution, all branches merge at summary_node
                                last_node = summary_node

                        # Execute the constructed WorkGraph
                        # This runs all the action nodes (and any branched agents recursively)
                        work_graph = WorkGraph(
                            start_nodes=start_nodes,
                            copy_debuggable_config_from=self
                        )

                        if last_node is not None:
                            work_graph.set_parent_debuggable(self._active_last_node)
                            last_node.log_info(
                                f"`{last_node.id}` is set as agent `{self.id}`'s active last node",
                                log_type=LOG_TYPE_SET_AGENT_ACTIVE_LAST_NODE
                            )
                            self._active_last_node = last_node

                        work_graph_results = work_graph.run(
                            task_input={
                                self.task_input_field_user_input: user_input,
                                self.task_input_field_user_profile: user_profile,
                                self.task_input_field_context: context,
                                self.task_input_field_action_results: action_results,
                                **task_input_metadata
                            }
                        )

                        # Extract final results from the WorkGraph execution
                        work_graph_final_result = (
                            work_graph_results[-1]
                            if isinstance(work_graph_results, Sequence)
                            else work_graph_results
                        )

                        # UPDATE action_results: Extract raw operational data from completed WorkGraph
                        # The WorkGraph has executed one or more actions (possibly in parallel branches)
                        # Extract the action_results dict which contains raw operational data from the last action
                        # This action_results will feed into the next reasoner call (line 1072), closing the feedback loop
                        # Example: {'body_html_after': '<div>...', 'is_follow_up': False, 'source': 'https://...'}
                        user_input = work_graph_final_result[self.task_input_field_user_input]
                        agent_state.action_results = action_results = (
                            work_graph_final_result[self.task_input_field_action_results]
                        )
                        # Loop continues: the agent will call the reasoner again with updated action_results

        # POST-LOOP HANDLING: Unified finalization for all completion reasons
        # This code executes when the while loop is exited via break (either Normal or MaxLoops)

        # TRANSFORMATION: Convert raw operational data to presentation-ready results
        # _get_agent_results() transforms action_results (and all states) into agent_results
        # Example transformation:
        #   action_results (input):  {'body_html_before': '<div>...', 'is_follow_up': False}
        #   agent_results (output):  AgentActionResult(summary='## Slack Channel Summary...', details='<div>...')
        agent_results = self._get_agent_results(
            trigger_action=trigger_action,
            trigger_action_results=trigger_action_results,
            new_states=self.states[num_states_when_starting:]
        )

        # Finalize and send response (handles both Normal and MaxLoops completion)
        self._finalize_and_send_agent_results(
            agent_response=agent_response,
            agent_results=agent_results,
            task_input_metadata=task_input_metadata,
            completion_reason=completion_reason
        )

        # EXIT POINT: Return agent results after successful completion
        # For branched agents (recursive calls), this returns control to the parent
        # For the main agent, this completes the entire execution
        return agent_results

        # endregion

    def _check_workflow_control(self):
        """
        Check workflow control signals and handle Stop/Pause states.

        This method checks the current workflow control signal and handles Stop and Pause states.
        It should be called at strategic points in the execution loop to allow responsive control.
        The Pause state is handled internally by entering a wait loop.
        The _status attribute is updated to reflect the actual execution state.

        Returns:
            bool: True if execution should continue, False if execution should stop.
        """
        if self._control == AgentControls.Stop:
            self._status = CommonWorkflowStatus.Stopped
            self.log_info("Agent execution stopped by control signal")
            return False

        elif self._control == AgentControls.Pause:
            self._status = CommonWorkflowStatus.Paused
            self.log_info("Agent execution paused by control signal")
            # Wait loop until resumed or stopped
            import time
            while self._control == AgentControls.Pause:
                time.sleep(0.1)  # Small sleep to prevent CPU busy-waiting

            # Check if stopped while paused
            if self._control == AgentControls.Stop:
                self._status = CommonWorkflowStatus.Stopped
                self.log_info("Agent execution stopped while paused")
                return False

            self._status = CommonWorkflowStatus.Running
            self.log_info("Agent execution resumed from pause")

        elif self._control == AgentControls.StepByStep:
            self._status = CommonWorkflowStatus.Paused
            self.log_info("Agent in step-by-step mode - pausing before next step")
            # Wait loop until user calls resume() or stop()
            import time
            while self._control == AgentControls.StepByStep:
                time.sleep(0.1)  # Small sleep to prevent CPU busy-waiting

            # Check if stopped while in step-by-step mode
            if self._control == AgentControls.Stop:
                self._status = CommonWorkflowStatus.Stopped
                self.log_info("Agent execution stopped during step-by-step mode")
                return False

            # User called resume() - execute one step then return to step-by-step mode
            if self._control == AgentControls.Continue:
                self._status = CommonWorkflowStatus.Running
                self.log_info("Executing one step in step-by-step mode")
                self._control = AgentControls.StepByStep  # Reset for next checkpoint
                return True

        # Continue execution
        return True

    def stop(self):
        """
        Stop the agent execution.

        Sets the control signal to Stop, which will cause the agent's execution loop
        to terminate at the next iteration. The agent will break out of the __call__ loop
        and return any partial results.

        Note: This does not automatically call close(). Call close() explicitly for cleanup.
        """
        self._control = AgentControls.Stop
        self.log_info("Agent stop signal set")

    def pause(self):
        """
        Pause the agent execution.

        Sets the control signal to Pause, which will cause the agent's execution loop
        to enter a waiting state. The agent will remain paused until resume() or stop()
        is called.
        """
        self._control = AgentControls.Pause
        self.log_info("Agent pause signal set")

    def resume(self):
        """
        Resume the agent execution from a paused state.

        Sets the control signal to Continue, allowing a paused agent to resume execution.
        If the agent is not paused, this has no effect.
        """
        self._control = AgentControls.Continue
        self.log_info("Agent resume signal set")

    def step_by_step(self):
        """
        Enable step-by-step debugging mode for the agent.

        Sets the control signal to StepByStep, which causes the agent to pause at each
        workflow control checkpoint. The agent will execute one step each time resume()
        is called, then automatically pause again for the next step.

        This is useful for debugging and inspecting agent behavior at each decision point.

        To exit step-by-step mode:
        - Call resume() twice in quick succession to switch to normal Continue mode
        - Call stop() to terminate execution
        """
        self._control = AgentControls.StepByStep
        self.log_info("Agent step-by-step mode enabled")

    def close(self):
        """
        Clean up resources used by the agent.

        This method should be called when the agent is no longer needed to ensure
        proper cleanup of any resources (e.g., browser instances, file handles).
        Subclasses can override this to clean up their specific resources.
        """
        # Base implementation
        self.stop()
