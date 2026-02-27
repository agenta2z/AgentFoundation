from collections.abc import Mapping
from functools import partial
from itertools import product
from typing import List

from attr import attrs, attrib

from agent_foundation.agents.prompt_based_agents.prompt_based_action_agent import PromptBasedActionAgent
from rich_python_utils.common_objects.workflow.workgraph import WorkGraphNode, WorkGraph
from rich_python_utils.string_utils import strip_, split_

DEFAULT_RESPONSE_FIELD_PROBLEMS = 'Problems'
DEFAULT_RESPONSE_FIELD_PROBLEM = 'Problem'
DEFAULT_RESPONSE_FIELD_PROBLEM_ID = 'ProblemID'
DEFAULT_RESPONSE_FIELD_PROBLEM_GRAPH = 'ProblemGraph'
DEFAULT_ACTION_SEP = '->'
DEFAULT_SECONDARY_ACTION_SEP = ','

@attrs
class PromptBasedActionPlanningAgent(PromptBasedActionAgent):
    """
    A planning agent that decomposes complex tasks into sub-problems with dependencies.

    KEY DIFFERENCE FROM PromptBasedActionAgent:
    ==========================================
    - PromptBasedActionAgent._create_next_actions() returns List[List[AgentAction]]
      → Base Agent.__call__() builds WorkGraph dynamically with branched recursion

    - PromptBasedActionPlanningAgent._create_next_actions() returns WorkGraph (callable)
      → Base Agent.__call__() detects callable and executes it directly

    This polymorphic design allows the planning agent to have FULL CONTROL over
    the execution graph structure, parsed from the LLM's ProblemGraph response.

    WORKFLOW:
    =========
    1. LLM receives user request and decomposes into Problems with ProblemIDs
    2. LLM specifies dependencies in ProblemGraph (e.g., "A -> B" means B depends on A)
    3. This agent parses ProblemGraph into paths and builds a WorkGraph
    4. WorkGraph is returned as next_actions (callable)
    5. Base Agent.__call__() executes the WorkGraph directly
    6. Results flow between nodes via 'previous_agent_results' named argument

    EXAMPLE LLM RESPONSE:
    ====================
<StructuredResponse>
 <Problems>
  <Problem>
   <Request>Search customized recipe variations for Starbucks French roast coffee</Request>
   <SolutionRequirement>Specific recipe modifications needed with exact customization options and quantities for ordering</SolutionRequirement>
   <ProblemID>SUGGEST_RECIPE_STARBUCKS</ProblemID>
  </Problem>
  <Problem>
   <Request>Place online order for one customized French roast coffee from Starbucks</Request>
   <ProblemID>ORDER_STARBUCKS</ProblemID>
  </Problem>
  <Problem>
   <Request>Place online order for one French roast coffee from McDonald's</Request>
   <ProblemID>ORDER_MCDONALDS</ProblemID>
  </Problem>
 </Problems>
 <ProblemGraph>
  SUGGEST_RECIPE_STARBUCKS -> ORDER_STARBUCKS
  ORDER_MCDONALDS
 </ProblemGraph>
</StructuredResponse>

    RESULTING EXECUTION GRAPH:
    =========================
      SUGGEST_RECIPE_STARBUCKS ──► ORDER_STARBUCKS
      ORDER_MCDONALDS (independent, runs in parallel)

    RESULT PASSING:
    ==============
    When ORDER_STARBUCKS runs, it receives SUGGEST_RECIPE_STARBUCKS's result via:
      kwargs['previous_agent_results'] = <result from SUGGEST_RECIPE>

    This is then converted to attachments in Agent.__call__() so the LLM can see
    the previous task's output when reasoning about the current task.
    """
    enable_action_groups: bool = attrib(default=False)

    # Planning agents typically only need one iteration to decompose and plan
    # After planning and executing the plan, the agent should complete rather than re-planning
    max_num_loops: int = attrib(default=1)

    # Response field mappings for parsing LLM output
    response_field_next_actions: str = attrib(default=DEFAULT_RESPONSE_FIELD_PROBLEMS)
    response_field_action: str = attrib(default=DEFAULT_RESPONSE_FIELD_PROBLEM)
    response_field_action_id: str = attrib(default=DEFAULT_RESPONSE_FIELD_PROBLEM_ID)
    response_field_action_graph: str = attrib(default=DEFAULT_RESPONSE_FIELD_PROBLEM_GRAPH)

    # Graph parsing separators:
    # action_sep: Sequential dependency (A -> B means B depends on A)
    # secondary_action_sep: Parallel alternatives (A, B means A and B can run in parallel)
    action_sep: str = attrib(default=DEFAULT_ACTION_SEP)
    secondary_action_sep = attrib(default=DEFAULT_SECONDARY_ACTION_SEP)

    def _parse_instant_response(self, instant_response):
        return instant_response

    def _create_action_item(self, raw_action_item: Mapping):
        """
        Create a partial function (actor) for a single Problem from the LLM response.

        Unlike PromptBasedActionAgent which creates AgentAction objects,
        this creates partial(self.actor, **args) - a callable that will be
        wrapped in a WorkGraphNode.

        The 'name' attribute is set on the partial so it can be looked up
        when building the dependency graph from ProblemGraph text.

        Args:
            raw_action_item: Parsed Problem dict from LLM response
                             e.g., {'Request': '...', 'ProblemID': 'ORDER_STARBUCKS'}

        Returns:
            partial: A partial function partial(self.actor, **actor_args) with
                     .name attribute set to ProblemID for graph building
        """
        actor_args = self._create_actor_args(raw_action_item)
        action_id = raw_action_item.get(self.response_field_action_id, None)
        actor = partial(self.actor, **actor_args)
        if action_id:
            setattr(actor, 'name', action_id)
        return actor

    def _create_next_actions(self, action_items: List[WorkGraphNode], raw_response_parse: Mapping):
        """
        Build a WorkGraph from the LLM's ProblemGraph dependency specification.

        CRITICAL: This method returns a WorkGraph (callable), NOT a list!
        ================================================================
        This is the KEY difference from PromptBasedActionAgent:
        - Parent class returns: List[List[AgentAction]] → Agent builds WorkGraph
        - This class returns: WorkGraph (callable) → Agent executes directly

        When Agent.__call__() receives next_actions, it checks:
            if callable(next_actions):
                action_results = next_actions(task_input={...})  # Direct execution
            else:
                # Build WorkGraph with branched recursion...

        GRAPH PARSING:
        =============
        ProblemGraph text format:
            "A -> B -> C"  means: A executes first, then B, then C (sequential)
            "A, B -> C"    means: A and B in parallel, then C (Cartesian product)
            "A\\nB"        means: A and B are independent paths (both are start nodes)

        Example:
            "SUGGEST_RECIPE -> ORDER_STARBUCKS
             ORDER_MCDONALDS"

            Becomes paths: [(SUGGEST_RECIPE, ORDER_STARBUCKS), (ORDER_MCDONALDS,)]
            Becomes DAG:
                SUGGEST_RECIPE ──► ORDER_STARBUCKS
                ORDER_MCDONALDS

        RESULT PASSING:
        ==============
        result_pass_down_mode=self.task_input_field_previous_agent_results (='previous_agent_results')

        This means when Node A completes, its result is passed to Node B as:
            kwargs['previous_agent_results'] = result_from_A

        In Agent.__call__(), this is extracted and converted to attachments:
            previous_agent_results = task_input.pop('previous_agent_results')
            attachments.extend(self._make_attachments(previous_agent_results))

        This enables context propagation through the dependency graph.
        """
        if self.response_field_action_graph in raw_response_parse:
            raw_graph_text = strip_(raw_response_parse[self.response_field_action_graph])
            if not raw_graph_text:
                self.log_warning("No action graph is identified.")
                return

            # Build lookup map: ProblemID -> partial(actor, **args)
            # This allows us to convert ProblemID strings from the graph text
            # into the actual callable actors
            action_id_to_action_items_map = {
                action_item.name: action_item
                for action_item in action_items
            }

            # Parse graph text line by line
            # Each line represents an independent path or dependency chain
            raw_action_dependency_texts = split_(raw_graph_text, sep='\n', lstrip=True, rstrip=True)

            action_graph = []
            for raw_action_dependency_text in raw_action_dependency_texts:
                if not raw_action_dependency_text:
                    continue

                # Split by '->' to get sequential dependencies
                # "A -> B -> C" becomes ["A", "B", "C"]
                action_item_names = split_(raw_action_dependency_text, sep=self.action_sep, lstrip=True, rstrip=True)

                # Handle ',' for parallel alternatives using Cartesian product
                # "A, B -> C" with action_item_names = ["A, B", "C"]
                # After split by ',': [["A", "B"], ["C"]]
                # Cartesian product: [("A", "C"), ("B", "C")] - two paths!
                for action_path in product(*map(lambda x: split_(x, sep=self.secondary_action_sep, lstrip=True, rstrip=True), action_item_names)):
                    action_graph.append(
                        tuple(
                            action_id_to_action_items_map[action_item_name]
                            for action_item_name in action_path
                        )
                    )
        else:
            # Missing the action graph, we assume a linear dependency
            # All action_items execute in sequence as a single path
            action_graph = (action_items, )

        # BUILD AND RETURN WORKGRAPH (CALLABLE):
        # =====================================
        # WorkGraph constructor uses build_nodes_from_paths() to create DAG
        # from the action_graph paths. Nodes with common prefixes are merged.
        #
        # node_cls configures each WorkGraphNode with:
        #   result_pass_down_mode='previous_agent_results'
        # This causes results to be passed as named kwargs between nodes,
        # enabling context propagation through the dependency graph.
        #
        # IMPORTANT: Returning WorkGraph (callable) instead of list!
        # Agent.__call__() will detect callable(next_actions)==True and
        # execute it directly rather than building its own WorkGraph.
        return WorkGraph(
            action_graph,
            node_cls=partial(
                WorkGraphNode,
                result_pass_down_mode=self.task_input_field_previous_agent_results
            )
        )


    def _create_agent_state(self, raw_response_parse: Mapping):
        """
        Planning agents don't track state in the traditional way.

        Returns None because:
        1. Planning agents typically run once (max_num_loops=1)
        2. The "state" is effectively the WorkGraph being executed
        3. Individual sub-tasks (nodes) may have their own agent states
        """
        return None