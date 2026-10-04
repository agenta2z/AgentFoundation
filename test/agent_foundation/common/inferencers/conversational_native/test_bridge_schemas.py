"""MCP schemas and names of the AF tool bridge (plan §5.1-§5.2).

Plan §12.1 row coverage (assertion: tests; Class.method or a whole Class):
- every AF tool.json converts:
    ActionSchemaDetailsTest.test_the_full_registry_offers_one_valid_named_schema_per_tool
    SchemaTest.test_every_action_tool_converts_with_executor_keys
- reverse name map:
    SchemaTest.test_reverse_name_map_returns_the_executor_key
    SchemaTest.test_cli_spellings_reach_the_executor_under_its_key
    ActionArgumentNamesTest
- enum / required / default:
    ActionSchemaDetailsTest.test_choices_become_an_enum_and_required_and_defaults_are_kept
- subcommand flattening:
    SchemaTest.test_subcommands_flatten_behind_a_required_enum (+ its collision tests)
- widget schemas incl. then_run / output / flags round-trip:
    WidgetSchemaRoundTripTest
    WidgetThenRunMarkerTest
    SchemaTest.test_widget_schemas_round_trip_through_the_parser
- names <= 64 incl. the prefix:
    SchemaTest.test_mcp_names_are_valid_and_bounded
    ActionSchemaDetailsTest.test_the_full_registry_offers_one_valid_named_schema_per_tool
Also here: the SOP-tool allowlist and the tool_argument_form flag
(BridgeSurfaceTest).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import jsonschema
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_response_parser import (
    tool_invocation_to_conversation_tool,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.schema import (
    action_tool_schema,
    argument_names,
    canonical_arguments,
    executor_arguments,
    MCP_PREFIX,
    mcp_tool_name,
    property_name,
    SOP_COMMAND_SCHEMAS,
    WIDGET_SCHEMAS,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.tool_bridge import (
    END_TURN_MARKER,
)
from agent_foundation.resources.tools.models import (
    ParameterDef,
    SubcommandDef,
    ToolDefinition,
)
from agent_foundation.resources.tools.registry import load_all_tools
from fakes import text, tools
from helpers import make_native
from later.unittest import TestCase


class SchemaTest(unittest.TestCase):
    def _tool(self, subcommands) -> ToolDefinition:
        return ToolDefinition(
            name="kn",
            description="Knowledge base.",
            tool_type="Action",
            parameters=[ParameterDef(name="--workflow-target-path", type="path")],
            subcommands=subcommands,
        )

    def test_flag_names_normalize_to_executor_keys(self) -> None:
        self.assertEqual(
            property_name("--workflow-target-path"), "workflow_target_path"
        )
        self.assertEqual(property_name("topic"), "topic")

    def test_subcommands_flatten_behind_a_required_enum(self) -> None:
        schema = action_tool_schema(
            self._tool(
                [
                    SubcommandDef(
                        name="add",
                        parameters=[ParameterDef(name="text", required=True)],
                    ),
                    SubcommandDef(
                        name="search",
                        parameters=[ParameterDef(name="--top-k", type="int")],
                    ),
                ]
            )
        )
        props = schema["properties"]
        self.assertEqual(schema["required"][0], "subcommand")
        self.assertEqual(props["subcommand"]["enum"], ["add", "search"])
        self.assertIn("absolute path", props["workflow_target_path"]["description"])
        self.assertEqual(props["top_k"]["type"], "integer")
        self.assertEqual(props["text"]["type"], "string")
        self.assertNotIn(
            "text", schema["required"]
        )  # subcommand parameters stay optional

    def test_conflicting_subcommand_parameter_types_are_rejected(self) -> None:
        tool = self._tool(
            [
                SubcommandDef(
                    name="a", parameters=[ParameterDef(name="--limit", type="int")]
                ),
                SubcommandDef(
                    name="b", parameters=[ParameterDef(name="--limit", type="flag")]
                ),
            ]
        )
        with self.assertRaises(ValueError):
            action_tool_schema(tool)

    def test_a_parameter_shared_by_subcommands_is_one_explicit_property(self) -> None:
        props = action_tool_schema(load_all_tools()["knowledge"])["properties"]
        self.assertEqual(props["space"]["type"], "string")
        self.assertIn(
            "(subcommands: add, search, load, list)", props["space"]["description"]
        )
        self.assertIn(
            "add: Knowledge space to add to; search: Filter by knowledge space",
            props["space"]["description"],
        )
        self.assertIn(
            "(subcommands: get, update, delete, restore; required by: get, "
            "update, restore)",
            props["piece_id"]["description"],
        )
        self.assertEqual(
            props["query"]["description"],
            "search: Search query text; delete: Query-based deletion (mode 2, "
            "alternative to piece_id). (subcommands: search, delete; required by: "
            "search)",
        )
        self.assertIn("(subcommand: history; required)", props["since"]["description"])

    def test_tool_level_parameter_redeclared_identically_stays_tool_level(
        self,
    ) -> None:
        schema = action_tool_schema(
            self._tool(
                [
                    SubcommandDef(
                        name="add",
                        parameters=[
                            ParameterDef(name="--workflow-target-path", type="path")
                        ],
                    )
                ]
            )
        )
        self.assertEqual(
            schema["properties"]["workflow_target_path"]["description"],
            "(absolute path)",
        )

    def test_any_other_name_collision_is_rejected(self) -> None:
        cases = {
            "choices differ": [
                SubcommandDef(
                    name="a",
                    parameters=[ParameterDef(name="--mode", choices=["x", "y"])],
                ),
                SubcommandDef(name="b", parameters=[ParameterDef(name="--mode")]),
            ],
            "defaults differ": [
                SubcommandDef(
                    name="a", parameters=[ParameterDef(name="--limit", default="5")]
                ),
                SubcommandDef(name="b", parameters=[ParameterDef(name="--limit")]),
            ],
            "path vs string": [
                SubcommandDef(
                    name="a", parameters=[ParameterDef(name="target", type="path")]
                ),
                SubcommandDef(name="b", parameters=[ParameterDef(name="target")]),
            ],
            "same subcommand": [
                SubcommandDef(
                    name="a",
                    parameters=[
                        ParameterDef(name="--top-k"),
                        ParameterDef(name="top_k"),
                    ],
                )
            ],
            "tool-level differs": [
                SubcommandDef(
                    name="a",
                    parameters=[ParameterDef(name="--workflow-target-path")],
                )
            ],
            "selector name": [
                SubcommandDef(name="a", parameters=[ParameterDef(name="--subcommand")])
            ],
        }
        for case, subcommands in cases.items():
            with self.subTest(case), self.assertRaises(ValueError):
                action_tool_schema(self._tool(subcommands))

    def test_reverse_name_map_returns_the_executor_key(self) -> None:
        long_flag = "--" + "very-long-option-" * 5
        tool = ToolDefinition(
            name="odd",
            tool_type="Action",
            parameters=[
                ParameterDef(name="--workflow-target-path", type="path"),
                ParameterDef(name="--max depth/v2", type="int"),
                ParameterDef(name=long_flag),
            ],
        )
        names = argument_names(tool)
        props = action_tool_schema(tool)["properties"]
        self.assertEqual(set(names), set(props))
        for prop in props:
            self.assertRegex(prop, r"^[a-zA-Z0-9_.-]{1,64}$")
        self.assertEqual(names["workflow_target_path"], "workflow_target_path")
        self.assertEqual(names["max_depth_v2"], "max depth/v2")
        self.assertIn(property_name(long_flag), names)
        self.assertEqual(
            names[property_name(long_flag)], long_flag.lstrip("-").replace("-", "_")
        )
        self.assertEqual(
            executor_arguments(tool, {"max_depth_v2": 3, property_name(long_flag): 1}),
            {"max depth/v2": 3, long_flag.lstrip("-").replace("-", "_"): 1},
        )

    def test_cli_spellings_reach_the_executor_under_its_key(self) -> None:
        tool = load_all_tools()["research_propose"]
        canonical = canonical_arguments(
            tool,
            {
                "request": "x",
                "--docs-path": "/d",
                "workflow-target-path": "/w",
                "--no-dual": True,
            },
        )
        self.assertEqual(
            canonical,
            {
                "request": "x",
                "docs_path": "/d",
                "workflow_target_path": "/w",
                "--no-dual": True,
            },
        )
        self.assertEqual(
            executor_arguments(tool, canonical),
            {
                "request": "x",
                "docs_path": "/d",
                "workflow_target_path": "/w",
                "no_dual": True,
            },
        )
        with self.assertRaises(ValueError):
            canonical_arguments(tool, {"docs_path": "/a", "--docs-path": "/b"})
        with self.assertRaises(ValueError):
            executor_arguments(tool, {"no-dual": True, "no_dual": False})

    def test_every_action_tool_converts_with_executor_keys(self) -> None:
        for tool in load_all_tools().values():
            if tool.tool_type != "Action":
                continue
            schema = action_tool_schema(tool)
            for key in schema["properties"]:
                self.assertNotIn("-", key, f"{tool.name}.{key}")
        research = load_all_tools()["research_propose"]
        props = action_tool_schema(research)["properties"]
        self.assertIn("workflow_target_path", props)
        self.assertEqual(props["research_only"]["type"], "boolean")

    def test_widget_schemas_round_trip_through_the_parser(self) -> None:
        tool = tool_invocation_to_conversation_tool(
            {
                "name": "single_choice",
                "arguments": {
                    "prompt": "Depth?",
                    "choices": [{"label": "Quick", "value": "quick"}, "deep"],
                    "parallel_group": 1,
                },
                "output": ["depth"],
            }
        )
        self.assertEqual(tool.output_vars, ["depth"])
        self.assertEqual([c.value for c in tool.choices], ["quick", "deep"])
        self.assertEqual(tool.parallel_group, 1)
        self.assertEqual(
            set(WIDGET_SCHEMAS),
            {
                "clarification",
                "single_choice",
                "multiple_choice",
                "confirmation",
                "proposal_selection",
            },
        )

    def test_mcp_names_are_valid_and_bounded(self) -> None:
        long_name = "x" * 80
        name = mcp_tool_name(long_name)
        self.assertLessEqual(len(MCP_PREFIX + name), 64)
        self.assertEqual(mcp_tool_name("research-propose"), "research-propose")
        self.assertEqual(mcp_tool_name("a b/c"), "a_b_c")


class ActionArgumentNamesTest(TestCase):
    async def test_a_cli_spelled_call_runs_with_the_executor_keys(self) -> None:
        native, factory, _, executor = make_native(
            [
                [
                    tools(("write_brief", {"--topic": "lidar", "--depth": "deep"})),
                    tools(("write_brief", {"topic": "a", "--topic": "b"})),
                    text("done"),
                ]
            ]
        )
        async with native:
            await native.run_agentic_loop("go")
            self.assertEqual(
                executor.calls, [("write_brief", {"topic": "lidar", "depth": "deep"})]
            )
            ok, duplicate = factory.last.tool_results
            self.assertFalse(ok[2])
            self.assertTrue(duplicate[2])
            self.assertIn("'topic' is given more than once", duplicate[1])


class BridgeSurfaceTest(TestCase):
    _SOP_TOOLS = {"enter_sop", "resume_sop", "pause_sop", "exit_sop", "sop_status"}

    async def test_sop_control_tools_follow_the_allowlist(self) -> None:
        native, _, _, _ = make_native([])
        self.assertLessEqual(
            self._SOP_TOOLS, {s.name for s in native.bridge.manifest()}
        )
        native, factory, _, _ = make_native(
            [[text("ok")]], sop_control_tools=["sop_status"]
        )
        async with native:
            await native.run_agentic_loop("go")
            offered = {t.name for t in factory.last.open_request.tools}
            self.assertEqual(offered & self._SOP_TOOLS, {"sop_status"})
            refused = await native.bridge.call("enter_sop", {"name": "mini_research"})
            self.assertTrue(refused.is_error)
            self.assertIn("Unknown AgentFoundation tool", refused.text)
            self.assertIsNone(native.sop_state)

    def test_an_unknown_sop_control_tool_fails_at_construction(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            make_native([], sop_control_tools=["enter_sop", "start_sop"])
        self.assertIn("start_sop", str(ctx.exception))

    def test_tool_argument_form_is_offered_only_behind_its_flag(self) -> None:
        native, _, _, _ = make_native([])
        self.assertNotIn(
            "tool_argument_form", {s.name for s in native.bridge.manifest()}
        )
        native, _, _, _ = make_native([], expose_tool_argument_form=True)
        self.assertFalse(hasattr(native.prompt_renderer, "set_variable"))
        (spec,) = [
            s for s in native.bridge.manifest() if s.name == "tool_argument_form"
        ]
        self.assertEqual(spec.input_schema["required"], ["prompt"])
        self.assertIn("then_run", spec.input_schema["properties"])
        tool = tool_invocation_to_conversation_tool(
            {
                "name": "tool_argument_form",
                "arguments": {"prompt": "Topic?", "parallel_group": 2},
                "output": ["topic"],
            }
        )
        self.assertEqual(
            (tool.tool_type, tool.prompt, tool.output_vars, tool.parallel_group),
            ("tool_argument_form", "Topic?", ["topic"], 2),
        )

    async def test_tool_argument_form_answer_feeds_its_then_run(self) -> None:
        form = {
            "prompt": "Which topic should the brief cover?",
            "output": ["topic"],
            "then_run": {"name": "write_brief", "arguments": {"topic": "__topic__"}},
        }
        native, factory, interactive, executor = make_native(
            [[tools(("tool_argument_form", form))], [text("Brief written.")]],
            answers=["lidar"],
            expose_tool_argument_form=True,
        )
        async with native:
            result = await native.run_agentic_loop("write a brief")
            self.assertTrue(factory.last.tool_results[0][1].startswith(END_TURN_MARKER))
            self.assertEqual(len(interactive.widgets), 1)
            self.assertEqual(executor.calls, [("write_brief", {"topic": "lidar"})])
            self.assertEqual(native.prior_context["topic"], "lidar")
            self.assertEqual(native.prior_context["tool_argument_form__topic"], "lidar")
            self.assertIn("topic: lidar", factory.last.turn_requests[1].text)
            self.assertEqual(result.text, "Brief written.")


class ActionSchemaDetailsTest(unittest.TestCase):
    def test_choices_become_an_enum_and_required_and_defaults_are_kept(self) -> None:
        tool = ToolDefinition(
            name="brief",
            tool_type="Action",
            parameters=[
                ParameterDef(name="topic", required=True, positional=True),
                ParameterDef(
                    name="--depth", choices=["quick", "deep"], default="quick"
                ),
                ParameterDef(name="--max-pages", type="int", default=3),
                ParameterDef(name="--dry-run", type="flag", default=False),
                ParameterDef(name="--out", type="path", description="Where to write."),
            ],
        )
        schema = action_tool_schema(tool)
        self.assertEqual(schema["required"], ["topic"])
        self.assertEqual(
            schema["properties"],
            {
                "topic": {"type": "string"},
                "depth": {
                    "type": "string",
                    "enum": ["quick", "deep"],
                    "default": "quick",
                },
                "max_pages": {"type": "integer", "default": 3},
                "dry_run": {"type": "boolean", "default": False},
                "out": {
                    "type": "string",
                    "description": "Where to write. (absolute path)",
                },
            },
        )

    def test_the_full_registry_offers_one_valid_named_schema_per_tool(self) -> None:
        native, *_ = make_native([])
        registry = load_all_tools()
        native.tool_registry.clear()
        native.tool_registry.update(registry)
        specs = native.bridge.manifest()
        actions = {
            mcp_tool_name(t.name)
            for t in registry.values()
            if t.tool_type == "Action" and getattr(t, "agent_enabled", True)
        }
        self.assertTrue(actions)
        self.assertEqual(
            sorted(s.name for s in specs),
            sorted(actions | set(WIDGET_SCHEMAS) | set(SOP_COMMAND_SCHEMAS)),
        )
        for spec in specs:
            with self.subTest(spec.name):
                self.assertRegex(MCP_PREFIX + spec.name, r"^[a-zA-Z0-9_-]{1,64}$")
                jsonschema.validators.validator_for(spec.input_schema).check_schema(
                    spec.input_schema
                )
                self.assertEqual(spec.input_schema["type"], "object")
                self.assertTrue(spec.description)


_THEN_RUN = {"name": "write_brief", "arguments": {"topic": "__answer__"}}
_COMMON = {"output": ["answer"], "parallel_group": 2, "then_run": _THEN_RUN}
# One call per widget passing every argument its schema declares.
_WIDGET_CALLS = {
    "clarification": {
        **_COMMON,
        "prompt": "Target path?",
        "default": "/srv/model",
        "expected_input_type": "path",
        "prefix": "/srv",
        "allow_multiple_input": True,
        "serialization": "json",
    },
    "single_choice": {
        **_COMMON,
        "prompt": "Depth?",
        "default": "quick",
        "choices": [
            {
                "label": "Quick",
                "value": "quick",
                "description": "One pass.",
                "input": {"name": "pages", "label": "Pages"},
            },
            "deep",
        ],
        "allow_custom": False,
    },
    "multiple_choice": {
        **_COMMON,
        "prompt": "Which?",
        "default": "a",
        "choices": ["a", {"label": "B", "value": "b"}],
        "allow_custom": True,
    },
    "confirmation": {
        **_COMMON,
        "prompt": "Write it?",
        "default": "yes",
        "view": "/tmp/brief.md",
        "yes_label": "Write",
        "no_label": "Stop",
    },
    "proposal_selection": {
        **_COMMON,
        "prompt": "Pick proposals.",
        "default": "P1",
        "proposals_path": "/tmp/proposals.json",
        "preselected_ids": ["P1"],
        "allow_zero": True,
        "experiment_hub": True,
        "host_dashboard": "hub_dash",
    },
}
_FIELDS = {
    "prompt": lambda t: t.prompt,
    "output": lambda t: t.output_vars,
    "parallel_group": lambda t: t.parallel_group,
    "expected_input_type": lambda t: t.expected_input_type,
    "prefix": lambda t: t.prefix,
    "allow_multiple_input": lambda t: t.allow_multiple_input,
    "serialization": lambda t: t.serialization,
    "allow_custom": lambda t: t.allow_custom,
}


def _choice(item) -> dict:
    if isinstance(item, str):
        return {"label": item, "value": item, "description": "", "input": None}
    spec = item.get("input")
    return {
        "label": item["label"],
        "value": item.get("value", ""),
        "description": item.get("description", ""),
        "input": (spec["name"], spec["label"]) if spec else None,
    }


class WidgetSchemaRoundTripTest(unittest.TestCase):
    """Plan §5.2: every argument a widget schema declares survives
    ``tool_invocation_to_conversation_tool`` the way the bridge calls it
    (``output`` as the bindings, ``then_run`` kept beside the widget)."""

    def test_every_declared_widget_argument_reaches_the_conversation_tool(
        self,
    ) -> None:
        self.assertEqual(set(_WIDGET_CALLS), set(WIDGET_SCHEMAS))
        for name, call in _WIDGET_CALLS.items():
            with self.subTest(name):
                schema = WIDGET_SCHEMAS[name]
                self.assertEqual(set(call), set(schema["properties"]))
                jsonschema.validate(call, schema)
                arguments = {k: v for k, v in call.items() if k not in _COMMON}
                arguments["parallel_group"] = call["parallel_group"]
                tool = tool_invocation_to_conversation_tool(
                    {"name": name, "arguments": arguments, "output": call["output"]}
                )
                self.assertEqual(tool.tool_type, name)
                for key, value in call.items():
                    if key == "then_run":
                        continue
                    if key == "choices":
                        carried = [
                            _choice(
                                {
                                    "label": c.label,
                                    "value": c.value,
                                    "description": c.description,
                                    "input": c.input
                                    and {"name": c.input.name, "label": c.input.label},
                                }
                            )
                            for c in tool.choices
                        ]
                        self.assertEqual(carried, [_choice(c) for c in value])
                    elif key in _FIELDS:
                        self.assertEqual(_FIELDS[key](tool), value, key)
                    else:
                        self.assertEqual(tool.metadata[key], value, key)


class WidgetThenRunMarkerTest(TestCase):
    async def test_then_run_is_kept_beside_the_widget_in_its_durable_marker(
        self,
    ) -> None:
        # The confirmation handler drops a view path that is not a file.
        view = os.path.join(tempfile.mkdtemp(prefix="af_native_view_"), "brief.md")
        Path(view).write_text("# Brief\n")
        call = {**_WIDGET_CALLS["confirmation"], "view": view}
        native, _, interactive, executor = make_native(
            [[tools(("confirmation", call))]], answers=[None]
        )
        async with native:
            result = await native.run_agentic_loop("write a brief?", turn_number=1)
        self.assertTrue(result.has_conversation_tool)
        (marker,) = interactive.persisted
        self.assertEqual(marker["action_tools"], [_THEN_RUN])
        (tool,) = marker["tools"]
        self.assertEqual(
            (tool.tool_type, tool.prompt, tool.output_vars, tool.parallel_group),
            ("confirmation", "Write it?", ["answer"], 2),
        )
        self.assertEqual(tool.metadata["view"], view)
        self.assertEqual(executor.calls, [])
