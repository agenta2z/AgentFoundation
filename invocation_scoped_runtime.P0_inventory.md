# Invocation-scoped runtime: P0 inventory

**Purpose.** This is the P0 inventory required by plan v8 (`snappy-doodling-star.md`), §11 P0 commit 6. It is laid out as follows:
- §0 holds the decisions P0 settles.
- §1–§30 each map to one commit-6 bullet, in plan order.
- §31 classifies the existing tests (plan §9).
- §32 records characterization findings from the P0 goldens.
- §33 resolves the items the drafts left unverified.
- §34 lists deviations from the plan.
- §35 rolls up every `[OPEN→Pn]`.

**Anchors.**
- Every `file:line` is at base commit `69ea2d3762af` unless it says otherwise.
- Since base, the stack has changed these production files:
  - `RC/purity.py` (B8, `39178f39d551`);
  - `RC/state.py`, `RC/store.py` and TPL (B12, `2ca561e6b299`);
  - WG and AU (B14, `2c02f7615283`);
  - SIB: `_SessionSlot` and the `active_session_id` getter/setter (B6(a), `8b92cead4ca6`) shift SIB lines by +7 after `:131` and +9 after `~:602`; B4 (`1eb6a19db727`) replaces the recovery write at base SIB `:1368` with `self.active_session_id = None` (now SIB:1377).
  - Other shifts: `RC/store.py` +6 after `~:84`; `RC/state.py` +2 after `~:210`; WG +1 after `:45`, base WG:2740-2753 collapsed to WG:2741-2742, −11 after base `:2756`; B12 leaves TPL with no net shift.
  - AU: B14 adds `run_async_joined`, and E5 (`933507187199`) removes 5 lines near base `:324`; `_run_async` :440 → :437, `run_async_joined` at AU:478.
  - RPU `common_utils/function_helper.py`: E6 (`45f0996cb685`) and E5 (`933507187199`) rewrite the sync `execute_with_retry` terminal and single-attempt path (base `:375-660`).
  - IB: B5 (`cbb056d03259`, guardrail window, base `:591-607`, `:4226-4261`, `:4386`, `:4485`), B9 (`dc02a72082e4`, dead surface, base `:547`, `:2211-2232`, `:2828`, `:4551-4569`, `:4909`) and E1 (`972c1cda155c`, `parallel_infer`, base `:4004-4114`). Net −21 lines after base `:4909`; `sl diff -r 69ea2d3762af` gives the exact shift at any line.
  - B9 also deletes dead lines in BTA.py (base `:2033`, −1 after), DUAL (base `:1013`, −1 after) and PTI (base `:410-413`, `:1505`, `:2411-2437`, `:2579`), and adds 2 lines to TPL after base `:121`.
  - devmate: B11 (`59f0b0115ec6`, `dump_output`, base `:1364-1442` and `:1549-1665`).
  - LWI: E2 (`cae65efbbca1`, `_record_iteration`, +4 after base `:976`) and E4 (`e456b12e3147`, the `_auto_enable_checkpointing` docstring, −2 after base `:1025`).
  - WF: E4 (`e456b12e3147`) adds one line at each resume site (+1 after base `:1112`, +2 after base `:1467`).
  - P2 (`384f501b6c0d` … `6941db7c5e55`) rewrites the streaming entries of SIB, rovodev, devmate, claude and OpenClaw, touches IB (the fan-out comment and the `_validate_bta_inferencer_spec` error text) and TAI (`_ainfer` consumes the pipeline). §24 and §27 carry the post-P2 lines; elsewhere, lines in those files are at base.
  - The pre-P3 fixes (a)–(f) (`8b552c9de7` … `a393f0248c`) and the logger race (`02a7a80250`) touch CI, `AF/ui/graph_interactive_adapter.py`, `OS/services/conversation_service.py`, `OS/routes/manager_websocket_routes.py`, `FLOW/reflective_inferencer.py`, `AFN/decorator.py`, `AFS/agents/agent.py` and IB (`_rc_child` subslots, the workspace-logger lock). Rows about those sites carry the fix tag; their line numbers stay at base.
  - A row whose fact changed since base is tagged "(fixed in `<hash>`)"; rows are not renumbered.
- Every other production file is unchanged since base.
- For the changed files, read the base copy (`sl cat -r 69ea2d3762af <file>`), or diff against it (`sl diff -r 69ea2d3762af <file>`).

**Tags.**

| Tag | Meaning |
|---|---|
| `[V]` | Verified in source by reading the anchor. |
| `[G]` | Pinned by a committed golden under `TEST/run_context/goldens/` (bta, getters, legacy_workspace, locations, openclaw, streaming/{claude_code,devmate,rovodev}, lifetimes). |
| `[P]` | Pinned by a P0 probe in `TEST/run_context/test_p0_probes.py`. Probes are strict xfails that name their fix phase. Verdicts below are from the committed run (1 passed, 23 xfailed); each xfail fails for the reason its marker states. |
| `[X]` | Run ad hoc during P0 from a script that is not committed. Reproducible, but not a regression test. |
| `[I]` | Inferred from reading the code; not executed. |
| `[OPEN→Pn]` | Deferred to phase Pn, with the reason stated. |

**Search method.**
- Scoped `grep -rn` only, inside `AgentFoundation/`, `RichPythonUtils/` and `OpenStartup/`, plus AST scans of those trees.
- No recursive search over fbsource.
- `search_files` omits some subtrees, so empty results were re-checked with a scoped grep.

**Paths.**

| Short | Path |
|---|---|
| AF | `fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/` |
| RC | `AF/run_context/` |
| FLOW | `AF/agentic_inferencers/flow_inferencers/` |
| EXT | `AF/agentic_inferencers/external/` |
| BTA.py | `FLOW/breakdown_then_aggregate_inferencer.py` |
| RPU | `fbcode/_tony_dev/CoreProjects/RichPythonUtils/src/rich_python_utils/` |
| TEST | `fbcode/_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/` |
| AFS | `fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/` |
| OS | `fbcode/_tony_dev/CoreProjects/OpenStartup/src/openteam/server/` |
| OST | `fbcode/_tony_dev/CoreProjects/OpenStartup/test/openteam/` |
| IB | `AF/inferencer_base.py` |
| SIB | `AF/streaming_inferencer_base.py` |
| TPL | `AF/templated_inferencer_base.py` |
| TIB / TSIB | `AF/terminal_inferencers/terminal_inferencer_base.py` / `terminal_session_inferencer_base.py` |
| MFI / MFD / DUAL / PTI / LWI | `FLOW/multi_flow_inferencer.py` / `multi_flow_dual_inferencer.py` / `dual_inferencer.py` / `plan_then_implement_inferencer.py` / `linear_workflow_inferencer.py` |
| CI | `AF/agentic_inferencers/conversational/conversational_inferencer.py` |
| TAI | `AF/agentic_inferencers/tool_inferencers/tool_as_inferencer.py` |
| AFN | `AF/agentic_functions/` |
| WG / WF / AU | `RPU/common_objects/workflow/workgraph.py` / `.../workflow/workflow.py` / `RPU/common_utils/async_utils.py` |
| TEX | `AFS/resources/tools/task/executor.py` |
| claude / claude-sdk | `EXT/claude_code/claude_code_cli_inferencer.py` / `claude_code_sdk_inferencer.py` |
| codex / codex-sdk | `EXT/codex/codex_cli_inferencer.py` / `codex_sdk_inferencer.py` |
| kiro | `EXT/kiro/kiro_cli_inferencer.py` |
| rovodev / rovodev-serve | `EXT/rovodev/rovodev_cli_inferencer.py` / `rovodev_serve_inferencer.py` |
| devmate / devmate-sdk | `EXT/devmate/devmate_cli_inferencer.py` / `devmate_sdk_inferencer.py` |
| openclaw | `EXT/openclaw/openclaw_inferencer.py` |

---

## 0. Decisions

| # | Decision | Evidence | Resulting action |
|---|---|---|---|
| D1 | **`fresh_instance` copies what the fan-out needs.** Override values are shared by reference. Recipe `InferencerBase` instances (breakdown, aggregator, entries of a worker list) are rebuilt fresh for each copy, memoized per pass. Plain callable factories are shared, but each copy calls them. `partial` is rebuilt and `LazyConfigFactory` gets `.fresh()`. Duck-typed stages, closures, `OrderedDict`, namedtuples and `template_manager` are shared. | [V] Recipe capture:<br>• `_capture_init_recipe` IB:318-334 captures explicit init args.<br>• Lazy fields are captured as `_LIVE_FIELD`: `worker_inferencers` (BTA.py:642) and `bta_inferencer` (IB:532).<br>• Everything else goes through `_snapshot` (IB:308-315), which copies exact dict/list/tuple/set and keeps other values by reference.<br>[V] Rebuild:<br>• `fresh_instance` IB:1462-1483 → `_InstanceRebuilder.build` IB:345-373 → `type(inf)(**kwargs, **overrides)`.<br>• `_rebuild_kwargs` IB:375-388 skips override keys, so overrides are never rebuilt.<br>• `rebuild_value` IB:390-410:<br>&nbsp;&nbsp;– `InferencerBase` → recursive `build`, memoized;<br>&nbsp;&nbsp;– `_PrototypeCloneFactory` (IB:194) / `_FreshCloneFactory` (IB:219) → the same wrapper around a rebuilt prototype;<br>&nbsp;&nbsp;– `LazyConfigFactory` → `.fresh()`;<br>&nbsp;&nbsp;– `partial` → rebuilt;<br>&nbsp;&nbsp;– a bound method of an IB → rebound to the rebuilt instance;<br>&nbsp;&nbsp;– exact containers → recursive;<br>&nbsp;&nbsp;– anything else → by reference.<br>[V] The fan-out today (`_materialize_fanout` IB:3444-3469):<br>&nbsp;&nbsp;1. `worker = self.fresh_instance(**_p_derived_overrides())` (IB:3471-3483; `template_manager` passed by reference);<br>&nbsp;&nbsp;2. `fanout = proto.fresh_instance(worker_inferencers=_FreshCloneFactory(worker), worker_inference_args={..., "prepared_input": True}, workspace=None, **_fanout_slot_overrides(proto))` (IB:3485-3497; blank slots become per-call `self.fresh_instance(...)` overrides);<br>&nbsp;&nbsp;3. `fanout.build_aggregator()`.<br>`_fanout_prototype` (IB:3351-3359) returns a live template instance, whose slots `fresh_instance` rebuilds, or else calls the template factory on each call. | **P6 registers the fanout itself in the caller's ledger.** This resolves the plan §5.7 conditional as "copies": every `InferencerBase` stage of the fanout is a per-call copy.<br>Caveats:<br>(a) A shared duck-typed stage that defines `adisconnect` would also be closed. No in-tree duck-typed stage defines one: `AF/mock_inferencers/*.py` has none [V]. Only test-local doubles could.<br>(b) P6 must **not** pass the prototype's live aggregator instance as an override. That would flip it from rebuilt to shared, and seeding would write into the prototype. Resolve from the fanout's own constructed slot (`resolve_stage(fanout.aggregator_inferencer)` after `fresh_instance`) or through the rebuild path. This contradicts plan §5.7 text; see §34 X2.<br>(c) `adisconnect` stays idempotent (plan §5.7). |
| D2 | **B15: CONFIRMED.** | [P] `test_b15_iteration_dirs_derive_from_call_start_workspace` xfails with "iteration N's dir nests under iteration N-1's (lwi/iteration_2/iteration_3)"; `test_b15_reused_lwi_keeps_its_workspace` xfails with "the call leaves the LWI bound to its last iteration dir, so the next call starts there" (both P9).<br>[V] LWI:911-918 rebinds `self._workspace` to the iteration workspace.<br>[G] `test_golden_locations.py:198` `lwi_reused_bare_b15`: the second bare call nests under `<WS>/lwi/iteration_2/iteration_3` and returns call 1's result. | Fix in P9, as the probe names it. |
| D3 | **B31: CONFIRMED** (sync and async). | [P] `test_b31_shared_nested_bta_uses_each_worker_node_name` xfails with "both workers sharing one nested BTA run under the last worker's node name"; `test_b31_nested_bta_name_unchanged_after_call` xfails with "BTA writes its per-call node name onto the nested BTA worker" (P7).<br>[V] Writes `worker.name` at BTA.py:2552-2554 and :2799-2800. | Fix in P7. |
| D4 | **B34: CONFIRMED.** | [P] `test_b34_parallel_infer_inits_call_state_per_item` / `test_b34_aparallel_infer_inits_call_state_per_item` xfail with "(a)parallel_infer items never run _init_call_state"; `test_b34_iterator_inits_call_state_per_item[sync/async]` xfails with "infer(iterator) runs _init_call_state/state_factory once, on the iterator" (P3).<br>[V] `parallel_infer` (IB:3980-4111) and `aparallel_infer` (IB:5325-5401) never call `_init_call_state`.<br>[G] `test_golden_streaming_lifetimes.py:367`: `state_factory` receives the `list_iterator`. | Fix in P3 (fixed in `f0e15b47af`: `_init_call_state` runs inside the seam, once per invocation). |
| D5 | **B35: CONFIRMED.** | [P] `test_b35_resume_keeps_max_breakdown_truncation[sync/async]` xfails with "resume rebuilds workers from the untruncated promoted breakdown"; `test_b35_resume_keeps_interactive_selection` xfails with "resume rebuilds workers from the pre-selection promoted breakdown" (P8).<br>[V] Ordering in the async `_breakdown_fn`:<br>&nbsp;&nbsp;1. promotion BTA.py:3194-3196;<br>&nbsp;&nbsp;2. then the `max_breakdown` cap :3198-3203;<br>&nbsp;&nbsp;3. then the zero-subquery guard :3205-3216;<br>&nbsp;&nbsp;4. then `breakdown_only` :3218-3220;<br>&nbsp;&nbsp;5. then the inline interactive selection :3222-3238.<br>So the promoted file holds the uncapped, unselected list, and resume lambdas BTA.py:942-953 rebuild from it. | Fix in P8. |
| D6 | **B36: CONFIRMED.** | [P] `test_b36_substitute_result_is_canonical_output[default|external × sync|async]` xfails with "the substitute result is returned but never written as the canonical output"; `test_b36_failed_aggregator_output_is_not_canonical[default|external]` xfails with "finalize links the failed attempt's aggregator output as canonical" (P6).<br>[G] `test_golden_bta_finalize.py:60` / `:76`.<br>For workspace-bound BTAs, the observable failure today is: the fallback result is returned but not finalized (no `outputs/`, report, manifest or symlink). See D11 and §32. | Fix in P6. |
| D7 | **B37 (new, found in P0).** A resume misfires when worker results exist but no promoted breakdown does. The setup is a fresh instance with the same workspace and `resume_with_saved_results=True`. The resume raises `TypeError` in `_build_subgraph_spec`. A sync retry also reaches it. | **CONFIRMED.** [P] `test_b37_resume_without_promoted_breakdown_completes[sync/async]` xfails with "resume feeds _load_promoted_breakdown()'s None into _build_subgraph_spec"; `test_b37_sync_retry_after_worker_failure_completes` xfails with "a sync retry rebuilds the saved expansion from a missing promoted breakdown" (P8).<br>[G] `test_golden_bta.py:259` `test_resume_without_promoted_breakdown` (sync and async).<br>[V] Mechanism:<br>&nbsp;&nbsp;1. Registry lambdas BTA.py:942-953 call `_build_subgraph_spec(self._load_promoted_breakdown()[0], _original_query=...)`.<br>&nbsp;&nbsp;2. `_load_promoted_breakdown` (BTA.py:1763-1805) returns `(None, None)` when `breakdown/decomposed_subtasks.json` is missing.<br>&nbsp;&nbsp;3. `for sq in sub_queries:` (BTA.py:2389) then raises `TypeError`.<br>[V] Sync path: `_infer_recovery` (IB:4149) re-runs `BTA._infer`. The re-run reconstructs the saved expansion and hits the same `TypeError`, and the chain then moves to the external fallback ([G] `finalize_after_fallback` external_sync). The async `_ainfer_recovery` (IB:4172) re-raises for orchestrators (IB:4200-4201). | Fix in P8, with the manifest / plan-commit work, or earlier if a phase needs resume to work without a promoted breakdown. |
| D8 | **Nested same-path public calls** are refactored before P3 lands. The claim is never relaxed. | See §9 for every site. [V]:<br>• `FLOW/reflective_inferencer.py:323` and `:372-374` call `self.reflection_inferencer(processed_reflection_input)` with no `run_context`. `__call__` → `infer` → `enter_run(None)` reuses the reflective node's ctx, so a different object runs at the same path. The base calls already use `_rc_child("base")`.<br>• `AFN/decorator.py:288-297` returns a static `infer_kwargs["run_context"]` as-is, so every call runs at the caller's path.<br>• `AFS/agents/agent.py:1612` calls `self.reasoner(...)` with no `run_context`. | Before P3:<br>• reflective → `run_context=self._rc_child("reflect")` (fixed in `f2622c8fbb`; slot `reflect`, see X26);<br>• decorator → in the static-rc case derive `caller_rc.child(slot)`, and run `_reset_session` under `rc` (fixed in `cbada6a67b`; see X27);<br>• agent.py:1612 → a child slot (fixed in `12c48ee8ad`: `reasoner`; see X28);<br>• by-design same-path adapters → route to the private pipeline (P2/P3). |
| D9 | **Stale callers** are pre-existing defects and out of scope, unless a phase touches the file. | [X] The Python kwargs `worker_factory=` / `workspace_root=` raise `TypeError` at construction.<br>[V] YAML `worker_factory:` is **not** stripped silently. RPU `config_utils/_instantiate.py:1453-1459` logs a warning and `:1460` deletes the key, so the BTA is built without `worker_inferencers`. The runtime effect is [I]. Full hit list below the table. | Record only. A phase that edits one of these files fixes its hits in the same commit. |
| D10 | **Target coverage.** P0 commit 1 repointed the terminal-inferencer target to CoreProjects. | [V] `TEST/common/inferencers/terminal_inferencers/BUCK` now has one `python_pytest` target, `terminal_inferencers`. Deps (:14-15): `//_tony_dev/CoreProjects/AgentFoundation/src:agent_foundation`, `//_tony_dev/CoreProjects/RichPythonUtils/src:rich_python_utils`.<br>Baselines:<br>• buck `common/inferencers/...`: 725 pass / 0 fail, plus 2 pre-existing build failures (`test_metamate_standalone_inferencer`: `fbcode//metamate_standalone` missing).<br>• `run_context` target at P0 c1: 171 pass / 5 fail pre-existing: 4 × `example_runcontext_*`, and `test_m9_conversation_resume.py::test_rehydrate_from_resumed_store_wires_restore_into_the_run`. After the P0 commits (probes, goldens, ratchet, B8): 404 pass / 1 fail (the same m9 test) / 23 skip (buck reports strict xfails as skips).<br>• After B4: `common/inferencers/...` 729 pass (725 + 4 new B4 tests) / 0 fail, same 2 build failures.<br>• `terminal_inferencers`: 72 pass.<br>• After the pre-P3 fixes (`02a7a80250`): `run_context` local and buck 467 pass / 1 fail (the same m9 test) / 23 skip.<br>Per-target counts are in the table below. | These baselines are the regression floor for every phase. |
| D11 | **U3b reachability (new).** The B36 U3b guard (BTA.py:1841-1850) cannot be reached through the public API **for workspace-bound BTAs**. It **is** reachable for workspace-less and `checkpoint_dir`-only BTAs. | [V] For workspace-bound BTAs, `_read_child_workspace` (IB:915-933) resolves the aggregator workspace through the child ctx even before any subgraph is built. The explicit bind happens at BTA.py:3016, and also via :1197 → `_rebind_aggregator_workspace` :1205-1214.<br>[X] Workspace-bound BTA, breakdown fails, external fallback: `fb:hello`, no U3b.<br>[X] No workspace and no `checkpoint_dir`, workers failing, external fallback, sync and async: the expansion fails, `fb` runs, then U3b `RuntimeError` is raised. **The fallback result is lost.** The same config with no fallback raises `ExpansionConfigError`.<br>[X] `checkpoint_dir`-only BTA with an aggregator: U3b after `[w0,w1,w2,agg]` on **every** call, including the happy path and with `default_return_or_raise`. With `disable_aggregator=True` the call succeeds. | Fold into the B36 fix (P6). The P6 regression tests must cover the workspace-less and `checkpoint_dir`-only shapes. `[OPEN→P6]`: add a committed golden or probe for them; today they exist only as `[X]`. |

**D9 stale-caller hits** (status: *live* = production code path, *doc* = documentation):

| Hit | Kind | Status |
|---|---|---|
| `AFS/experiment_hub/experiment_bridge.py:687`, `:690`, `:727`, `:730` (also `checkpoint_dir` at :689 / :729) | Python kwargs | live |
| `AFS/experiment_hub/implement_hypothesis_bridge.py:928`, `:931` (`checkpoint_dir` at :930) | Python kwargs | live |
| `OS/resources/tools/create_role/executor.py:420` (BTA call about :417) | Python kwarg | live module. The tool's execute path builds from YAML (`create_role_bta.yaml`). |
| `OS/resources/tools/role_setup/executor.py:334-335`, `:629`, `:826`, `:830`, `:1192` | Python kwargs | live. The assignment at :1202 is dead code. |
| `OS/resources/tools/mock_task/profiles/`: `default.yaml:16,27`, `flat.yaml:22`, `huge.yaml:18,34`, `slow.yaml:13`, `error.yaml:13` | YAML key | live (mock task tool; duck-typed stages) |
| `OS/resources/tools/project_onboarding/project_onboarding.yaml:48`; `OS/resources/tools/role_setup/role_setup.yaml:51`, `role_setup_skill_tool_creation.yaml:54`; `OS/resources/tools/create_role/create_role_bta.yaml:117` | YAML key | live |
| `OpenStartup/_dev/role_setup.yaml:50` | YAML key | dev config |
| About 19 plan docs; `RC/README.md:37` (historical) | text | doc |

Not stale:
- `role_setup/executor.py:1013` and `create_role/executor.py:321` (different callees);
- `deliverable_boundary.py:189`, `:218` (`child_workspace_root`);
- test `StubInferencer` / `LazyHolder` hits;
- `OST/resources/tools/task/test_task_real_cli.py:808`.

**D10 per-target collected-test counts** (buck, `common/inferencers/...`; 35 targets, sum 725):

| Target | n | Target | n |
|---|---|---|---|
| agentic_functions:test_body_roles | 12 | :test_breakdown_block_registry | 13 |
| agentic_functions:test_config | 17 | :test_breakdown_then_aggregate | 40 |
| agentic_functions:test_decorator | 17 | :test_bta_checkpoint_promotion | 15 |
| agentic_functions:test_output | 11 | :test_bta_inferencer_fanout | 68 |
| agentic_functions:test_parsers | 39 | :test_bta_outcome_pairing | 18 |
| agentic_functions:test_plus | 4 | :test_bta_resume_original_query | 4 |
| agentic_functions:test_rendering | 22 | :test_bta_resume_workspace_binding | 6 |
| agentic_functions:test_run_context | 4 | :test_bta_worker_resume | 6 |
| agentic_functions:test_stage0_normalize | 10 | :test_fanout_primitives | 33 |
| agentic_functions:test_trace | 5 | :test_fresh_instance | 27 |
| external/metamate:test_code_scope_judge | 42 | :test_function_inferencer | 13 |
| external/metamate:test_metamate_timeout_scopes | 4 | :test_inferencer_variable_expansion | 23 |
| conversational:test_phase_progression_guards | 14 | :test_lwi_step_workspace | 2 |
| conversational:test_widget_recovery | 9 | :test_output_guardrail | 80 |
| terminal_inferencers:terminal_inferencers | 72 | :test_research_propose_contract | 4 |
| | | :test_research_propose_fanout_config | 19 |
| | | :test_reviewer_guardrail_cascade | 2 |
| | | :test_streaming_recovery | 31 |
| | | :test_task_instructions_snapshot | 16 |
| | | :test_unified_finalize_output | 23 |

The offline baseline failures (`failures.txt`, 883 lines) depend on the environment. They were not used as a regression floor.

---

## 1. §7 writers and host-mode presets

### 1a. Writers of the §7 fields (all `[V]`)

| Owner | Field / effect | Writers | Readers / notes |
|---|---|---|---|
| IB | `_guardrail_recent_empty_fingerprints` (IB:598) | IB:4250 (reset), 4255 (in-place `append`), 4257-4258 (replace), 4390, 4489 (accept resets) | read IB:4249, 4253, 4389, 4488; shared across calls of the leaf (fixed in `cbb056d03259`, B5: the field is deleted; the window is `_current_fallback_state["guardrail_empty_fingerprints"]`, IB:4218, and spans the attempts of one call) |
| IB | `_last_inference_input` | IB:2831, 4912 | IB:4564 (legacy fallback in `_render_guardrail_prompt`) (fixed in `dc02a72082e4`, B9: deleted; the guardrail reads only the current call's `rendered_input`) |
| IB | `_output_finalized` | IB:2214 (class attr), 2227 | read only at IB:2223 inside `_complete_inference` (IB:2216-2228), which has no caller in AF, RPU or OpenStartup, so the pair is dead (fixed in `dc02a72082e4`, B9: both deleted) |
| IB | judge mutation | `judge.template_manager = None` IB:4526-4527; legacy `judge._workspace = guardrail_ws` IB:4534 | permanent mutation of the judge instance |
| IB | role audit | `_role_history` lazy init, `_pending_role_changes` consumed, history append (IB:1201-1214) | instance writes during `switch_role` |
| IB | `_init_call_state` call sites | IB:3900 (`infer`), 5267 (`ainfer`) | see §11 |
| MFI | `_last_winner_idx`, `_last_reviewer_alias`, `_last_fixer_alias`, `_last_ranking` | override MFI:1486-1494 → `_reset_dispatch_state_for_call` MFI:1254-1263; `_dispatch_set` MFI:1289-1310 (ctx node, plus a backing mirror under legacy_mint); `_extract_dispatch_state` MFI:1539-1583 | ctx-aware properties MFI:1312-1342 |
| MFI | legacy input propagation | MFI:1878-1881 (no-ctx branch mutates `cfg["input"]` and `self.predefined_sub_queries`) | legacy only |
| CLI leaves | `active_session_id` post-call | claude 884-904 / 951-960; codex 656-673 / 703-710; kiro 333-344 / 393-403; rovodev 879-894 / 944-951; devmate 1265-1333 (async), 1199-1206 (sync, outside the bridge) | §12 |
| rovodev | `_current_output_file` ContextVar (rovodev:65-68) | `.set` 753; `.set(None)` 801 (no token, no `reset`) | reads 466, 572, 617 |
| rovodev | `_last_clean_output` (undeclared) | 582, 624, 792-803 | 656, 787, 818 (`getattr` after `super()._ainfer` at 817) |
| rovodev | `_last_raw_stdout` (undeclared) | 709 (reset), 715 | 666 |
| TPL | RoleState | TPL:465-477 (`_active_role_state` claims the node), 660-678 (`_record_role_state`) | B12 moved it to `node.role_state` after base |
| TPL | `_last_rendered_task_instructions` | TPL:167 (decl), 454 | TPL:463 (`_proposer_task_instructions`) |
| DUAL | indirect TPL writes | DUAL:1552, 2039 `_role_get(...)._render_prompt(...)` preview renders under the review/fix child ctx | write the TPL field above |
| DUAL | `_current_config` | DUAL:1013 (raw instance write; siblings use `_run_set`) | (fixed in `dc02a72082e4`, B9: deleted) |
| SIB | `active_session_id` | getter SIB:581, setter SIB:623 (range 580-640) | host → handle slot; legacy/no ctx → `_session_id` backing. (fixed in `8b92cead4ca6`, B6(a): a host None write stores `_SessionSlot.RESET`, SIB:639-643 now; the getter returns None for RESET, never the backing, :600-607; the no-ctx read skips RESET, :625) |
| SIB | raw `_session_id = None` | SIB:1368 in `_ainfer_recovery` (SIB:1320) | under a host ctx the `live_session_id` slot is not cleared [I]. (fixed in `1eb6a19db727`, B4: now `self.active_session_id = None`, SIB:1377) |
| SIB | Tier-3 | `_get_live_handle_store` SIB:515; `_tier3_get` 529; `_tier3_set` 548; `_iter_live_handle_sets` 562 | §26 |
| devmate | `_consecutive_error_count` (:288) | `+= 1` :1281; `= 0` :1332 and :1150 | read :1147. Only async `ainfer` feeds it. |
| devmate | `dump_output` flip | `ainfer_streaming` :1367-1374 (restore :1438-1439); `infer_streaming` :1552-1560 (restore :1662) | read :769, :848 (fixed in `59f0b0115ec6`, B11: no flip; both streaming entries pass `dump_output=False` per call, :1373 / :1553, and `construct_command` reads `kwargs.get("dump_output", self.dump_output)`, :770) |
| devmate | `_output_file` | :770 (mkstemp), :965 (clear) | :848-850 |
| BTA | `_worker_task_instructions` (:778) | async worker harvest :2655 (guard :2650). **The sync worker_fn (:2737-2770) has no harvest write.** | getter return :811 |
| BTA | `use_async` | :2259-2260 flip, :2309 restore | :2446 → `_make_worker_fn` :2850, `_make_agg_fn` :3041; `_make_breakdown_fn` :3098-3100 |
| BTA | durable child-ws binds | `_bind_rebuilt_child_ws` at BTA :1208/1214, :2530/2532, :3015/3016; IB:3308; def IB:935-956 | |
| BTA | breakdown ws (bare backing) | `_configure_for_workspace` :2062 (`breakdown_inferencer._workspace = ws.child("breakdown")`) | |
| BTA | `_last_aggregation_guidance` | :786 (decl), :1396 (resume restore), :1429 (parser reset), :1477 (set) | :1393, :1398 |
| BTA | `_promoted_breakdown_cache` | :922, :1802, :2165, :2249 | :1782 (:1779 is docstring text) |
| BTA | `_cached_original_query` | :931, :2168, :2252 | registry lambdas :942-953 |
| BTA | `_graph_topology_emitted` | :771, :2171, :2255, :2348, :3340 | :2314, :3289 |
| BTA | `_pending_topology` | :1641, :2346, :3338 | :1626 (`getattr`) |
| BTA | `_worker_instances` | :2166, :2250, :2443-2444, :2551 | :1009, :1871, :2138, :2146, IB:3342 |
| BTA | `start_nodes` | :2196, :2287 | §4 |
| BTA | stage mutations | `worker.name` :2552-2554 / :2799-2800; `worker.interactive` :2808-2809; `worker.stream_observer` :2810-2815; agg `stream_observer` :3031; breakdown `stream_observer` :3144; `build_aggregator` write-back :1993 | §21 |
| PTI | `_current_*` (PTI:417-420) | `_workspace` :2654 / :2656 | |
| PTI | dead surface (B9) | `_setup_iteration_children` (def :2411-2437, no caller; its `child._workspace` :2435 and `child.output_path` :2437 writes are therefore dead; docstring mention :1508); `_partial_iteration_history` (decl :414-416, written only at :2579, never read); `_next_iteration_input` (:413, no reference) | deleted in P1 (B9) (fixed in `dc02a72082e4`) |
| LWI | `self._workspace` | LWI:911-918 (iteration workspace; B15) | |
| LWI | `_result_root_override` on child workflows | LWI:942, :947, :961 | |
| TAI | `_last_response` | TAI:156 (decl), :372 | :387; :426-430 after `ainfer_streaming` (:422) |
| claude / codex | `_last_stream_result` (undeclared) | claude :591 / :701; codex :470 / :527-529 / :549-551 | claude :890-897; codex :659-668 |
| SDK leaves | raw `_session_id` | codex :536; codex-sdk :258 / :316; devmate-sdk :407; `EXT/metamate/metamate_sdk_inferencer.py:415`; `EXT/rovochat/rovochat_inferencer.py:468` | bypass the property (§26) |
| metamate-sdk | `_conversation_uuid` / `_fbid` | `EXT/metamate/metamate_sdk_inferencer.py:413-414` | |
| OpenClaw | `_session_initialized` (:203) | :1206, :1284 | :1193, :1258 |

`[OPEN→P1]` PTI dead-code sites. P1's dead-code pass deletes or migrates them. P0 does not classify them beyond listing them above.

### 1b. Host-mode presets: tests that set instance state before a call `[V]`

| File | Lines | What |
|---|---|---|
| `TEST/common/inferencers/test_dual_inferencer/test_pti_resume.py` | 241, 242, 262, 263, 278, 279, 316, 317, 454, 467, 483, 486, 490 | 13 PTI preset writes (plan §9 names only :490) |
| `TEST/common/inferencers/test_dual_inferencer/test_recursive_resume.py` | 109, 130, 150 | `_current_base_workspace` |
| `TEST/common/inferencers/test_dual_inferencer/test_resume_detection.py` | 1458 | raw `_step_was_previously_attempted` set |
| `TEST/common/inferencers/test_bta_resume_workspace_binding.py` | 173 | `bta.use_async = True` |
| `TEST/common/inferencers/test_breakdown_block_registry.py` | 43-44 | mailbox fields on a `__new__` object |
| `TEST/common/inferencers/test_bta_resume_original_query.py` | 119-120 | `_promoted_breakdown_cache`, `_cached_original_query` |
| raw `_session_id` writes | `test_streaming_recovery` 234 / 267 / 457; `test_metamate_offline` 127; `test_dual_streaming` 323-324; `test_m6_session_handle` 57; `test_purity` 44 (base; B8 moved it) | leaf session presets. At `1eb6a19db727` the B4 / B6(a) tests add more, all in scope for the P3 commit 11 sweep: `test_streaming_recovery` 240 / 273 / 463 (shifted) plus 511 / 529 / 542 / 551; `test_m6_session_handle` 62 plus 112 / 121 / 130 / 158 / 166; `test_purity` 51 |

`[OPEN→each phase]` The full sweep of presets. Each phase that moves a field greps the tests for that field and updates the presets in the same commit. A global sweep now would go stale.

## 2. Post-call instance readers

| Reader | Reads | Anchor | Tag |
|---|---|---|---|
| CI after `base_inferencer.ainfer` | `get_final_output()`, gated by `streams_differ_from_final_output` (True only for rovodev :135; SIB:207 default) | CI:939 | [V] |
| MFD after the MFI call | `mfi._last_winner_idx`, `mfi._last_ranking` (ctx-aware) | MFD:1080-1081 | [V] |
| cross-node contract reads | `_proposer_task_instructions()` | DUAL:785, :1438; BTA.py:2652 | [V] |
| `_conclude_fanout` | `fanout._worker_instances` → `n_subtasks` | IB:3342 | [V] |
| TSIB `_ainfer` after the transport | `_last_streaming_output` / `_stderr` / `_return_code` | TSIB:485-487 | [V] |
| claude / codex `ainfer` | `_last_stream_result` | claude :890-897; codex :659-668 | [V] |
| rovodev `_ainfer` | `_last_clean_output` | rovodev :787, :818 | [V] |
| TAI `_ainfer` | `_last_response` | TAI:426-430 | [V] |
| SDK `_ainfer` | `_last_tool_use_count`, `_last_usage` | claude-sdk :396; codex-sdk :354-356 | [V] |
| devmate `parse_output` | `_output_file` | devmate :848-850 | [V] |
| guardrail prompt (legacy) | `_last_inference_input` | IB:4564 | [V] (fixed in `dc02a72082e4`, B9: the fallback is removed) |
| CI widgets | `_last_rendered_prompt` / `_template_source` / `_template_feed` / `_template_config` (CI:311-314) | CI:720-1228, :2757-2760; one-shot `_last_handler_bindings` :2891-2892 / :3097-3098 | [V] |
| tests | `_worker_instances` after a call | `TEST/common/inferencers/test_fresh_instance.py:577` | [V] |
| tests | `aggregator.template_extra_feed` after a call | `test_bta_inferencer_fanout.py:555`, `test_bta_outcome_pairing.py:531` | [V] |
| tests | `_cached_original_query` after `infer` / `ainfer` | `test_bta_resume_original_query.py:140`, `:155` | [V] |
| tests | LWI resume flags | `test_resume_detection.py:1433`, `:1449`, `:1563-1568`, `:1765` | [V] |

The getters are in §3.

## 3. Documented getters → `RuntimeKey.compat`

| Getter | Anchor | Reads | Tag |
|---|---|---|---|
| `get_final_output()` | SIB:216-230 (base default; returns `None` at :230) | none | [V] |
| `get_final_output()` | rovodev :641 | `_last_clean_output` (legacy), `_last_raw_stdout` | [V]; callers CI:939, rovodev :881 |
| `get_streaming_result()` | claude :1060-1079 | 3 TIB/TSIB fields via `getattr` | [V]; stale on the async path (§23 F1.2) |
| `get_streaming_result()` | devmate :1664-1702 | output and rc (:1690-1691), `parse_output(stderr="")` | [V]; **side effect**: sets `active_session_id` (:1698-1699) |
| `_proposer_task_instructions()` | IB:3625; TPL:461; DUAL:774; BTA.py:798 (returns `self._worker_task_instructions or ""` at :811); MFI:1606 | TPL `_last_rendered_task_instructions`; BTA `_worker_task_instructions` | [V] |
| MFI getters (`get_winner_flow_idx` :1587 … `get_non_winner_inferencers`) | MFI:1587-1680 | ctx-aware dispatch fields | [V] |
| `get_messages()` | CI:1326 | conversation messages | [V] |
| `last_call` | AFN `decorator.py:589` | trace | [V]; outside the inferencer tree |
| `get_response_text(result)` | devmate :1117, kiro :240, claude :1081 | its argument only | [V]; pure, not a post-call getter |

Proposed compat declarations [I]:

| Group | Fields | Used by |
|---|---|---|
| `TerminalStreamResult` | `_last_streaming_output`, `_last_streaming_return_code`, `_last_streaming_stderr` | claude and devmate `get_streaming_result` |
| rovodev | `_last_clean_output`, `_last_raw_stdout` | `get_final_output` |
| `_RENDERED_CONTRACT` | `_last_rendered_task_instructions` | TPL `_proposer_task_instructions` |
| BTA | `_worker_task_instructions` | BTA `_proposer_task_instructions` |

The MFI getters are already ctx-aware and need no entry.

## 4. `start_nodes` / `subgraph_registry` introspection users `[V]`

- **OS src:** 0 hits. The AFS hits outside BTA are separate fields: `agents/agent.py:1741-1866`, `automation/schema/action_graph.py:774-1809`, `ui/dash_interactive/utils/dummy_graph_executor.py:105, 181, 291`.
- **BTA fields and writes:**
  - `start_nodes = attrib(factory=list)` (:790-792);
  - `self.subgraph_registry = self.subgraph_registry or {}` (:942), which mutates the user's dict in place;
  - lambdas `"bta_diamond"` :943-948 and `"bta_workers"` :949-953;
  - `start_nodes = [breakdown_node]` at :2196 (`_infer`) and :2287 (`_ainfer`).
- **BTA reads:**
  - resume already-attached check `start_nodes[0].next` at :3270 (async) and :3446 (sync);
  - `GraphTopologyEvent.from_work_graph(self)` at :2321, which walks `start_nodes` via `AF/graph_events.py:54-98`.
- **WG indirect readers** (base anchors; WG changed in B14):
  - field :2001-2003 and post-init :2005-2011;
  - `_propagate_expansion_settings` :2013-2033;
  - `to_serializable_obj` :2038-2072;
  - `_clear_all_node_queues` :2084-2110;
  - `_all_nodes` :2127-2146;
  - `_reconstruct_graph_expansions` :2148-2290, with the registry lookup at :2227-2231;
  - initial-task creation :2482, :2718; `_run` :2793; `_arun` :2869-2912.
- **Tests:**
  - `test_fresh_instance.py:221-225` (registry identity and closures);
  - `test_bta_resume_original_query.py:124`, `:129`;
  - `test_bta_checkpoint_promotion.py:252`.
  - No test reads `.start_nodes`, `._graph`, `_pending_topology` or `_graph_topology_emitted`.

## 5. Thread and process hops on the call path `[V]` (C §2)

| Site | Mechanism | Copies ContextVars? | Note |
|---|---|---|---|
| WG:2748-2751 (base) | `ThreadPoolExecutor(1).submit(asyncio.run, coro)` | **No** | B14 hop; blocks the caller loop (§17). (fixed in `2c02f7615283`: now `run_async_joined`, WG:2742, runs under `copy_context()`, so Yes; writes stay in the copy and the caller loop is still blocked) |
| WG:2753 | same-thread `asyncio.run` | yes | only when no loop is running (now the no-loop branch of `run_async_joined`, AU:~500) |
| WG:2758-2775 | `executor.run_async` | process hop | no AF user sets `executor` |
| AU:440-478 `_run_async` | same-thread `asyncio.run` (:477) | yes | raises inside a running loop |
| IB `parallel_infer(use_threading=True)` (IB:4040-4080) | `multiprocessing.pool.ThreadPool` (`RPU/mp_utils/parallel_process.py:154-157`) | no, but re-bound by `_ctx_worker` → `enter_run(parent.child(f"parallel_{i}"))` (IB:4070-4078) | with no parent ctx: no bridge, each item legacy-mints |
| IB `parallel_infer(use_threading=False)` | `multiprocessing.Pool` | n/a | raises `NotImplementedError` when a ctx is present (IB:4041-4047) |
| SIB `infer_streaming` (SIB:1061-1114) | `copy_context()` + daemon `Thread` + `asyncio.run` | yes | throwaway loop per call (§18) |
| metamate-sdk :279 / :319 / :361 / :473 | `asyncio.to_thread` | yes | |
| IO pumps (claude :666-667; codex :507-508; devmate-sdk :370 / :381; claude-sdk :509; TAI:314-315; TSIB:284 / :290 / :294) | `create_task` | yes (snapshot) | not dispatch |
| CI:2200 | `create_task(_run_async())` async tool | yes (turn snapshot) | §6 (fixed in `a393f0248c`: the dispatch runs under `tool/<name>/async_<n>`) |
| `OS/services/tool_dispatcher.py:920` | `create_task` background tool | yes | outlives the turn |
| subprocess leaves (claude / codex CLI, kiro, rovodev, TSIB, rovodev-serve) | process | n/a | only env and args cross |
| WG gathers :2913, :1673-1705 | `asyncio.gather` | yes | every branch sees the same `_active_ctx` |

Other ContextVars a hop loses:
- `IB._current_fallback_state` (IB:67-68);
- rovodev `_current_output_file` (:66);
- `AFN/trace.py:51`;
- `AFS/ui/interactive_base.py:11`;
- `RPU/config_utils/_resolvers.py:14`.

## 6. Concurrent dispatch sites and child ctx (fix before P3)

**Derive a distinct child per branch** `[V]`:

| Site | Child slot |
|---|---|
| `parallel_infer` / `aparallel_infer` with a parent ctx (IB:4070-4078; `_aparallel_one` IB:5403-5417) | `parallel_{i}` |
| BTA async workers (BTA.py:2604-2728) | `worker_{i}` |
| MFI flows | `flow_{i}_workflow` |
| BTA breakdown / aggregator | `breakdown` / `aggregator` |
| fan-out (IB:3296-3331) | `BTA_INFERENCER_SLOT` |
| guardrail (IB:4501-4535) | `guardrail` |
| external fallback (~IB:3026 / ~5098) | `fallback/external_{i}` |

DUAL panelists (DUAL:1762-1800), LWI / PTI steps and MFD are **sequential** and use named slots.

**Branches that share a path, or have none:**

| # | Site | Finding | Resolution |
|---|---|---|---|
| 1 | BTA `adisconnect` gather (BTA.py:1003-1017, gather :1012) | Every child's `adisconnect` runs under the BTA's ctx. [V] | Leaf `adisconnect` drains every path regardless of ctx, so the shared path does not change what is closed. P6 replaces this with the ledger close. No claim is made, so it needs no fix before P3 [I]. |
| 2 | CI:2200 async tool | Inherits the finished turn's ctx snapshot, with no child derivation. The OpenStartup `task` tool mints its own root (TEX:1046-1055). [V] | Fixed in `a393f0248c`: a sync tool body runs under `tool/<name>`, an async dispatch under `tool/<name>/async_<n>` (a per-instance counter, so two overlapping dispatches of one tool never share a path). Both keep the turn's workspace object; commands dispatched as tools stay at the turn ctx (see X29). |
| 3 | ctx-less `parallel_infer` / `aparallel_infer` (IB:5410-5411; the no-parent branch of IB:4040-4080) | Each item legacy-mints `/` on one instance. [V] | Legacy mint gets no claim, so it is unaffected. P3 adds `_init_call_state` per item (B34). |
| 4 | WG gathers | Isolation relies on node fns calling `_rc_child`. No in-tree plain WorkGraph calls inferencers without `run_context=` under a ctx. [I] | None needed. |
| 5 | same-class, same-path claims | `RunStateStore.node` (`RC/store.py:94-113`, base) raises only on different non-None creators. Two concurrent same-class calls at one path silently share a node. [V] | The P3 strict claim closes this. |

## 7. Judges with an `input_preprocessor` `[V]`

**None in-tree.**
- `input_preprocessor` (IB:509, applied at IB:2761-2762) is assigned nowhere in AF, RPU or OS sources, except:
  - the attrib;
  - docstrings;
  - `_FANOUT_RESET_FIELDS` (IB:93-98).
- YAML judges set no preprocessor: `AFS/resources/tools/task/configs/breakdown-multiflow-plan.yaml:141-143, 250-252, 258-260, 270-272, 277-279`.
- The only test use sets it on the host, not on a judge: `test_bta_inferencer_fanout.py:386`.

Gap [I]: `_prepare_guardrail_judge` (IB:4501-4538) neutralizes only `template_manager`. A judge with a preprocessor would transform the pre-rendered judge prompt.

## 8. Host concurrent use of one instance (P6 guard)

| Case | Evidence | Resolution |
|---|---|---|
| BTA with a list-typed `worker_inferencers` (`wi[i % len(wi)]`, BTA.py:2470) | [V] `_validate_worker_isolation` (BTA.py:2089-2120, called :3073) only warns. No shipped config uses a list. | factory / `fresh_instance` per item (P6 c5/c7: in host mode the guard rejects one `InferencerBase` worker serving two concurrent workers, and graph build rejects a shared duck-typed one; no shipped config uses a list) |
| S1 concurrent turns on one session | [V] `OS/routes/manager_websocket_routes.py:1948-1965`: `cancel()` is not awaited, then a new turn starts. Two WS connections share the cached root and `_inferencers[sid]`. `turn_N` comes from disk (`OS/services/conversation_service.py:700-716`). | `[OPEN→P6 guard]`: the guard serializes or rejects. Which one is a product decision. (resolved in P6 c7: the guard rejects; `ConversationService` serializes a session's turns, X66) |
| S2 `/stream` fallback streams | [V] Each stream mints a fresh root (`.child("stream")`), so two streams never share a path (§10). | none needed (decided; see X32) |
| MFD reviewer / fixer reuse of flow leaves (`multiflow-plan.yaml:20-27`; MFD:457, 838-840, 921-929) | [V] sequential, after the flows join | certify (P6 c7: sequential reuse passes the guard; nothing to change) |
| CI async tools (CI:2200) | [V] The `task` tool mints its own root (TEX:1046). Each async dispatch now runs at its own child path (`a393f0248c`). | certify (P6 c7: the async tool runs the tool executor, never the CI's own leaves, so nothing overlaps) |
| `parallel_infer` / `aparallel_infer` | [V] No production callers. Only `test_inferencer_recovery.py` and `run_context/test_cancellation.py`. | `fresh_instance` per item, or certify (P6 c7: refused up front under a host ctx when items would overlap, X65; tests run one item at a time) |
| guardrail judges | [V] one judge per leaf (IB:519). It is shared only when the leaf itself runs concurrently. The judge ctx paths differ; only the IB:4526-4534 mutations race. | certify (B22 in P5) (P6 c7: a concurrently running leaf is rejected by the guard before its judge runs) |
| YAML `_import_shared_` (RPU `_instantiate.py` 412-425) | [V] no in-tree use | none |
| Python-built OpenStartup graphs | [I] not exhaustively read | `[OPEN→P6]`: audit when the guard lands. The guard fails loudly, so a miss surfaces as an error, not as silent sharing. (audited in P6 c7: the create_role executor builds a fresh worker per sub-query and distinct breakdown / aggregator instances; background tool tasks run tool executors, never a session's inferencers; every session-inferencer call is a turn under `_turn_lock`, which now also orders the per-turn `_tool_dispatcher` writes. No concurrent sharing found) |

## 9. Nested same-path public calls (before P3)

| Site | What it does `[V]` | Resolution |
|---|---|---|
| `FLOW/reflective_inferencer.py:323`, `:372-374` | `self.reflection_inferencer(processed_reflection_input)` has no `run_context`. `IB.__call__` → `infer` → `enter_run(None)` reuses the reflective ctx, so a **different object** claims the reflective path while the reflective call is live. | `run_context=self._rc_child("reflection")` before P3 (fixed in `f2622c8fbb`, slot `reflect`; see X26) |
| `AFN/decorator.py:288-297` `_resolve_run_context` | A static `infer_kwargs["run_context"]` is returned as-is, so every call runs at the caller's path, the same path as a live caller frame. Otherwise it returns `active.child(slot)`. `rc` is resolved once (:369) and reused by every parse-retry attempt (:372-378). `_reset_session(inferencer)` (:377, def :679) runs under the **caller** ctx, not `rc`. | Static case: `caller_rc.child(slot)`. Run `_reset_session` under `rc`. Consequence of today's behaviour: §33 A:289. (Fixed in `cbada6a67b`: an explicit ctx is the parent, like the active one; the reset runs under `rc`. Concurrent same-slot calls get no auto-suffix; see X27.) |
| `AFS/agents/agent.py:1612` | `self.reasoner(...)` has no `run_context`. With an active ctx, it runs at the enclosing path. | child slot before P3. The agent graphs are sync and ctx-less today [I]. (Fixed in `12c48ee8ad`: fixed slot `reasoner`; see X28.) |
| SIB `_ainfer` → `self.ainfer_streaming` (SIB:1041); TAI:422; SIB `infer_streaming` → `ainfer_streaming` (SIB:1082); IB adapters `iter_infer` :3967, `__call__` :4127, `aiter_infer` :5310; OpenClaw `infer` → `ainfer` :1056, `new_session` :1325, `anew_session` :1342 | By-design same-object adapters. | Route to the private pipeline (P2 / P3), so the adapter never re-enters a public entry. |
| SIB `new_session` / `anew_session` / `resume_session` / `aresume_session` (SIB:1155-1290) | `enter_run(run_context)`, then `self.infer` / `ainfer` | Stay frameless adapters; the inner call owns the frame. |

Already child slots [V]:
- `AF/templated_inferencer.py:112`, `:136`;
- guardrail (IB:4307 / 4418);
- fallback;
- fan-out (IB:3264 / 3279);
- CI `agent` / `context_compression`;
- flow orchestrators;
- TEX:1055, which is a root.

## 10. Root sites and handle-store reuse (B32)

| # | Site | Store | Handle store | Reused across turns | Decision |
|---|---|---|---|---|---|
| S1 | `OS/services/conversation_service.py:343` `_get_session_root` | loaded (:340) or fresh; saved per turn (:1160-1166) | fresh, cached with the root | **yes**: turns are `root.child(f"turn_{N}")` (:1134-1138); eviction (:378, :402) only pops | Behaviour-neutral under `(scope_id, path)` keying: same store, and turn paths differ. [I] The handle slot never carries across turns today. Any cross-turn session continuity comes from the instance backing (SIB getter fallback) or from history replay. |
| S2 | same file :1262 `astream_response` fallback | fresh | fresh | no; `.child("stream")` | Decided (X32): `/stream` is the manager-unavailable fallback. It replays history with `set_messages` and mints a fresh ephemeral root per call by design (the §9.4 comment at the call site), so no handle continuity is expected. `ab81b93a66` added only ownership closing. P3 commit 12 keeps it. |
| S3 | TEX:670 `_run_conversational_router` | loaded (:667) / saved (:687-697) | fresh | per call | fresh per call; no change |
| S4 | TEX:1046 `_run_topology` | loaded (:1043) / saved (:1072-1080) | fresh | per call | fresh per call; no change |
| S5 | `AFS/resources/tools/sop/cli.py:254` `run_sop` | loaded (:251) / saved (:293-299) | fresh | one root per process | continuity preserved (one store) |
| S6 | `RC/bridge.py:68` `mint_root(legacy_mint=True)` | fresh | fresh | no | legacy: one fixed scope (plan §5.5) |
| S7 | SIB:515-527 per-instance `_live_handle_store` | n/a | lazy, per leaf | lives as long as the leaf | becomes `(scope_id, path)` keyed |

[V] Mechanics:
- `RunContext.root()` (`RC/context.py:62-86`, root path `/` at :78) creates fresh `RuntimeBindings`, `RunStateStore` and `LiveHandleStore`.
- No host passes `handle_store=` or `runtime=`.
- `LiveHandleStore.teardown()` has no production caller.
- No host calls `adisconnect()` on the inferencer it built.

## 11. `_init_call_state` callers and `state_factory` classes (B34) `[V]`

| Kind | Anchor | Behaviour |
|---|---|---|
| attrib | IB:466 | `state_factory = attrib(default=None)` |
| base | IB:3717-3733 | No-op without a factory or a ctx. Otherwise claims `ctx.node(creator=(qualname, path))` (:3731) and sets `node.call = state_factory(inp)` **only if** `node.call is None`. Resets nothing. |
| override | MFI:1486-1494 | `super()`, then `_reset_dispatch_state_for_call()` (MFI:1254-1263) nulls the 4 dispatch fields via `_dispatch_set` |
| default factory | MFI:494-497 | `lambda _inp: MultiFlowState()` |
| src callers | IB:3900, IB:5267, MFI:1493 | |
| tests | `TEST/run_context/test_afi1_mfi_dispatch.py:109, 115, 127`; `test_c10_runner_state_isolation.py:199`; `test_m4_state_factory.py:20-58` | |
| example | `AgentFoundation/examples/.../example_runcontext_state_factory.py:29` | `BTAState` |

**Not reached by:**
- `parallel_infer` / `aparallel_infer` (B34);
- `SIB.ainfer_streaming` / `infer_streaming`;
- every `@bridge_entrypoint` CLI `ainfer` / `infer`;
- OpenClaw `ainfer` / `ainfer_streaming`;
- direct `_ainfer` / `_infer` calls.

devmate `infer` does reach it, via `super().infer`. `infer(iterator)` calls it once, on the iterator ([G] lifetimes :367).

Separate mechanism: LWI `initial_state_factory` (LWI:155, :1709-1710).

## 12. CLI session policy (seam hooks) `[V]`

The columns split each entry into:
- (a) argument adaptation, which stays in the adapter;
- (b) `_prepare_call`;
- (c) `_conclude_call`.

"del" is the delegate short-circuit.

| Leaf.entry | del | (b) (+ (a) kwargs writes) | CALL | (c) |
|---|---|---|---|---|
| claude `ainfer` (:836-904, `@bridge_entrypoint` :835) | 855-858 | 861-863 `new_session` pop/reset; 866-879 resolve `session_id` / `resume` into kwargs | 883 | 884-904: session from result / dict / `_last_stream_result`; enrich TIR; set `active_session_id` |
| claude `infer` (:909-962) | 926-927 | 930-932; 935-945 | 949 | 951-960 (no `_last_stream_result` fallback) |
| claude `infer_streaming` (:1000-1056, no bridge) | — | 1017-1027 | direct `_infer_streaming` (bypass) | **none** |
| codex `ainfer` (:627-675) | 631-634 | 636-638; 640-650 | 653 | 656-673 |
| codex `infer` (:678-712) | 682-683 | 685-687; 689-699 | 701 | 703-710 |
| kiro `ainfer` (:288-345) | 307-310 | 313-315; 318-327 | 330 | 333-344 |
| kiro `infer` (:350-405) | 369-370 | 373-375; 378-387 | 390 | 393-403 |
| rovodev `ainfer` (:828-900) | 845-848 | 858-860; 863-873 | 878 | 879-894: wrap via `get_final_output`, `find_latest_session_id`, `ensure_session_metadata` |
| rovodev `infer` (:903-951) | 919-920 | 923-925; 928-938 | 941 | 944-951, only if `result.success` |
| rovodev `ainfer_streaming` (:722-803) | 740-746 (`_delegates_execution_under`) | (a) 749-758 output-file / ContextVar; (b) 760-771, the **second** application on the `ainfer` path | 775 `super().ainfer_streaming` | `finally` 780-803 |
| devmate `_apply_session_policy` (:1134-1165) | — | pops `new_session`; NEW_SESSION_PER_CALL / ON_CONSECUTIVE_ERRORS (reset :1150) | — | — |
| devmate `infer` (:1167-1208, not decorated) | 1186-1189 | 1191, **before** the bridge | 1192-1194 `super().infer` | 1199-1206, **after** the bridge exits |
| devmate `ainfer` (:1211-1333) | 1253-1256 | 1258 | 1260 | 1265-1333: error count, NEW_SESSION_ON_ERROR, ACL-flake / port-race sleeps, tool-use / max-iterations resets, `InferencerExecutionError`; `= 0` on success |
| devmate `ainfer_streaming` (:1335-1439) | 1359-1364 | (a) `dump_output` off :1367-1374; (b) 1377-1391 | direct `_ainfer_streaming` :1409 (bypass) | **none** (fixed in `59f0b0115ec6`, B11: (a) is a per-call `dump_output=False`, :1368-1373) |
| OpenClaw `ainfer` (:980-1032, own `enter_run` :996-1005 / `exit_run` :1031-1032) | — | 1007-1009 `new_session`; 1011-1017 resolve; 1022 `_maybe_initialize_session` | 1024 `_ainfer_with_retry` (**not** `_ainfer_single`) | 1027-1030 |
| OpenClaw `ainfer_streaming` (:1061-1112, no bridge) | — | 1080-1095 | 1099-1109 | 1111-1112, skipped on abandonment |

`[OPEN→P3]` devmate `infer`: the policy (:1191) and the post-call write (:1206) resolve `active_session_id` under the caller's ctx, while the call itself runs under `run_context`. P3 moves both into `_prepare_call` / `_conclude_call`, inside the frame. (fixed in `2c1976b6d953`: the override is deleted; base `infer` reaches `_infer_single`, whose hooks run under the call's ctx.)

**After P3 c6** (`c718ae93e0e9`, `8dfbe0517baf`, `20e0c9cc3bff`, `e8aeda6af865`, `2c1976b6d953`, `a720eb3ba76f`): every (b) column is the leaf's `_prepare_call` (claude, codex, kiro, rovodev: `_apply_session_policy`; devmate: its existing `_apply_session_policy`; OpenClaw: `new_session` + `session_id` resolution). Every (c) column is `_conclude_call` (sync entries) or `_aconclude_call` (async entries): the sync/async columns above differ for claude and codex (async-only `_last_stream_result` fallback), rovodev (sync: success-gated capture; async: wrap, then capture) and devmate (async-only failure promotion with `await asyncio.sleep` backoffs). The "del" column is the seam's `_runs_provider` (`795c57a23aa4`). The `ainfer` / `infer` overrides are thin adapters (`@bridge_entrypoint` for the five TSIB leaves; OpenClaw keeps validation, `enter_run` and `_drop_per_call_fallback_mode`). The streaming rows are unchanged: their policy stays in the pipeline overrides (plan §5.2).

`[OPEN→P2]` devmate non-delegate streaming leaves `run_context` in kwargs for `_ainfer_streaming` / `construct_command`. The effect is not traced. P2 routes this entry through the template.

## 13. BTA path sites with their rule `[V]` (`checkpoint_dir` field :695)

**Workspace first, then `checkpoint_dir`:**
- `_get_result_path` :1690-1700;
- breakdown node ckpt, `_infer` :2184-2195 and `_ainfer` :2275-2286;
- aggregator node ckpt :3053-3056.

**Workspace only:**
- post-init :879-880 and :904;
- `_resolve_worker_output_paths` :1087, `_resolve_worker_output_path` :1114;
- `_rebind_aggregator_workspace` :1205-1214;
- graph status :1573, topology :1649;
- `_load_promoted_breakdown` :1785-1788 (**no `checkpoint_dir` fallback**, so a `checkpoint_dir`-only BTA cannot resume-short-circuit);
- `_finalize_output` :1835, :1856, :1869-1877;
- `_finalize_response` :2018 / :2027 / :2038;
- `_configure_for_workspace` :2057-2062;
- worker binding :2524-2532 and log :2541;
- `_make_worker_fn` ws :2860-2867;
- worker node ckpt :2877-2889 (hand-built `join(ws.root, "children", ...)`, no fallback);
- aggregator bind :3011-3017;
- deprecated `_configure_child_workspace` :1682-1688.

**Dead:** `_try_extract_proposal_index` (:1895 / :1923 / :1936) and `_read_aggregator_output_text` (:1960). Their only caller is `TEST/common/inferencers/test_bta_proposal_index_fallback.py`. Delete them rather than migrate.

**Base helpers (workspace only):**
- `_workspace` / `_workspace_under` IB:833-856;
- `_read_child_workspace` IB:915-933;
- `_bind_rebuilt_child_ws` IB:935-956;
- `_finalize_leaf_output` IB:2160;
- `_symlink_child_output` IB:2280;
- `_promote_child_checkpoints` IB:2377-2436;
- `resolve_output_path` IB:3598-3623;
- `_prepare_fanout` IB:3306-3308;
- `_validate_fanout` IB:3382.

`_finalize_response` exists only in BTA.py:2012 and DUAL:708.

(P6 c1: every in-call site above reads the call record `_bta_call()`, keeping its own rule; the post-init sites, `_configure_for_workspace` (P9) and the deprecated `_configure_child_workspace` stay. The dead pair is deleted with its test. `_promote_child_checkpoints` takes the parent workspace as `parent_ws`, X56.)

## 14. Roots with distinct `RuntimeBindings` sharing a store

[V] Only `store=` is ever passed to `RunContext.root()` (S1, S3-S5), and each loads its own `RunStateStore` object from disk. No in-memory store object is shared across distinct `RuntimeBindings`.

[I] On disk, two roots over one `working_dir` (S3 / S4 / S5 concurrently) are last-writer-wins. `save` is an atomic temp-file rename (`RC/store.py:162-196`, base) with no lock.

Decided (X32): in one process, a store object backs at most one live runtime; each root loads its own. Two processes sharing one on-disk store stay last-writer-wins, as today; P8's lease covers BTA checkpoint roots.

## 15. `.aggregator_inferencer` hits (35) `[V]`

- **Needs the resolved stage (19):**
  - IB:3410, IB:3512; MFI:878;
  - BTA :263, :288, :330, :1008, :1242-1243, :1253-1254, :1361, :1831, :1955 (dead), :1990, :2023, :2072-2073, :2083.
  - :2072 / :2083 are traversals.
- **Presence check (12):**
  - IB:3428; MFI:456, :860, :1701;
  - BTA :1359, :1843, :2139, :2920, :3208, :3347, :3428, :3460.
- **Write-back:** BTA :1993 (`build_aggregator`).
- **Definition-level:** BTA :888-889 (`output_path` default). With a factory it silently falls back to `"aggregation_report.md"`.
- **Doc:** BTA :806.
- Plus 3 `getattr(bta, "aggregator_inferencer", None)` presence forms (BTA :262, :286, :328), not counted in the 35.

Pairs on adjacent lines (1242-1243, 1253-1254, 2072-2073, 888-889) are one hit per line in the count.

`build_aggregator` (:1987-1994) callers:
- IB:3468, before `_validate_fanout_aggregator` (IB:3303) and `_seed_aggregator_feed` (IB:3304);
- BTA :3010, also reached by the resume lambdas.

No test calls `.build_aggregator()`.

Every RESOLVED site that runs before the first `_build_subgraph_spec` sees the raw factory when the BTA is used directly: :888, :1242, :1253, the traversals.

## 16. `ctx.node(creator=…)` sites `[V]`

| Anchor | Function | Inside the owner's frame? |
|---|---|---|
| IB:3731 | `_init_call_state` | yes (after `enter_run`) |
| TPL:476 | `_active_role_state` (a "read" that claims) | yes, at render |
| TPL:670 | `_record_role_state` via `switch_role` | **no.** Its only src caller is MFD:665 (`_reassign_role_workspace`), under MFD's role child ctx, which pre-claims the leaf's future node. No-ctx → early return. |
| MFI:1222 | `_reset_cross_flow_state` | yes |
| MFI:1303 | `_dispatch_set` | yes |
| MFI:1865 | `_apply_runtime_input_propagation` | yes |
| BTA :1714 | `_get_effective_predefined_sub_queries` | yes |

`creator=None` claims skip the guard:
- LWI:186-300;
- DUAL:372 / 383 / 417 / 432 / 478 / 490;
- CI:152 / 1437 / 1456 / 1483.

`load()` does not restore creator tags.

`evict_subtree` (`RC/store.py:129-149`, base) has one caller, DUAL:999. It is a no-op at root `/`.

## 17. `WorkGraph._run` in a running loop (B14) `[V]` (base WG)

**The hop** (WG:2740-2753): with `use_async` set and a running loop, it runs `ThreadPoolExecutor(1).submit(asyncio.run, self._arun(...)).result()`. That drops every ContextVar and freezes the caller loop (B14 in `2c02f7615283`: the context is now copied; the loop is still blocked). `use_async` defaults to False (WG:1986) and is not propagated to subgraphs (WG:672-693).

| # | Path to `_run` | Hops? |
|---|---|---|
| a | `IB.infer` → `BTA._infer` → `WorkGraph._run` (BTA.py:2206) | Only if `use_async` is True at that moment. It is True only inside the same instance's `_ainfer` window (:2259-2260 / :2309). |
| b | `MFI._infer` → `BTA._infer` (MFI:1916) | as (a) |
| c | `parallel_infer` worker thread | no loop in that thread → same-thread `asyncio.run` keeps the `parallel_{i}` ctx [I] |
| d | outer BTA sync worker → nested BTA `infer` (BTA.py:2748) | only when the nested instance is shared and in its own `_ainfer` [I] |
| e | `agent.py:1865-1884`, `prompt_based_planning_agent.py:251` | no (`use_async` never set) |
| f | BTA `run` / `arun` | blocked (BTA.py:1020-1030) |
| g | nested WorkGraph as a node value | no in-tree setter; dynamic construction [I] |

**Reachability:**
- [I] Overlapping `ainfer` calls on one BTA race the save/restore and can leave `use_async=True` permanently. After that, every sync `infer` from a loop thread hops.
- [V] `use_async=True` is set only at BTA.py:2260 and in `test_bta_resume_workspace_binding.py:173`.
- `AU._run_async` (AU:440-478) always raises inside a running loop. Its consumers: claude-sdk :426 / :449, codex-sdk :368 / :406, devmate-sdk :486 / :488, rovochat :547 / :549, openclaw :1053 / :1306 / :1376, metamate-sdk :523 / :525, rovodev-serve :373 / :375, TAI:447 / :449, LWI:1778, DUAL:948.

P1 (B14, `2c02f7615283`) replaced only the WG hop (base WG:2740-2753 → WG:2742 `run_async_joined`). `run_async_joined` copies the caller's context but still blocks the caller's loop. `AU._run_async` (now AU:442) is unchanged and still raises inside a running loop; all its consumers above still call it (P2/P3 own that).

## 18. Loop-bound resources `[V]` unless tagged

| Leaf | Handles (Tier-3 unless noted) | Created | Closed | Stale-loop handling |
|---|---|---|---|---|
| claude-sdk | `_client`, `_disconnect_fn`, `_connected_loop` (:198-220); instance `_connect_lock` (:195) | `aconnect` :455-519 (task runs connect → wait → disconnect); lazy connect :329-337 | `adisconnect` :520-534 (all branches) | sync only: :429 (closed loop) and :436-447 (cross-loop raise). Sync `_infer` (:449) never closes → **the known violation**. The async path has no check [I]. |
| codex-sdk | `_client`, `_thread`, `_connected_loop` (:134-156); `_connect_lock` (:128) | `aconnect` :~208-257; lazy connect :283-301 | `adisconnect` :259-281; sync `_infer` `finally` :404 drains all branches | sync only: :372-392 [I] Under sync `parallel_infer`, one thread's `finally` closes siblings on other loops. |
| rovodev-serve | `_server_process` (PIPEs), `_base_url`, `_http_client` (:94-116) | `aconnect` :127-205; auto-connect :287 | `adisconnect` :207-235 | **none**, and no lock. Two concurrent connects orphan a server [I]. |

**Nothing held across calls:** devmate-sdk, metamate-sdk, rovochat, openclaw (websocket per call :448-477, closed :860-864), claude / codex CLI, kiro, rovodev CLI, TSIB.

**Cross-cutting hazards [I]:**
- A leaf lazily connected inside `infer_streaming` binds to the throwaway loop.
- One branch's reset drains every branch: `anew_session` SIB:1201 (`await self.adisconnect()`; sync `new_session` :1155 only clears `active_session_id` :1175), `resume_session` :1287, recovery :1362, and the DUAL / PTI resets.

Each close must happen inside the creating loop (plan §5.1).

## 19. `fresh_instance` sharing

Decided in D1. The fan-out-relevant cases in brief:

| Value | Shared or fresh in a copy |
|---|---|
| override values | shared |
| configured breakdown / aggregator instance | fresh (post-construction setattrs dropped) |
| aggregator plain factory | shared callable, called per copy |
| `partial` / `LazyConfigFactory` | rebuilt / `.fresh()` |
| worker list | each rebuilt |
| bare worker instance | wrapped as `_PrototypeCloneFactory` by `_wrap_bare_factory_inferencers` (IB:1623), prototype rebuilt |
| duck-typed stages, closures, `template_manager` | shared |
| `subgraph_registry` | lambdas re-created bound to the copy (`test_fresh_instance.py:221-225`) |

## 20. LWI resume flags `[V]`

**Writers:** the RPU WF runner.

| Event | sync | async |
|---|---|---|
| per-run reset | :1184-1185 | :1538-1539 |
| marker detect | :1198-1199 | :1552-1553 |
| loop-back clear | :1382-1383 | :1740-1741 |
| advance clear | :1404-1405 | :1767-1768 |

**LWI:**
- Instance-backed properties at :644-649 and :652-657.
- The note at :630-642 declares concurrency out of scope. It cites PTI :571, but the read is at PTI :563.

**PTI:**
- Read at :563 (`_build_executor_input` :525), which appends a "Resume Context" block.
- Write `= True` at :2146 (comments :2141-2144, :1820).

**Other readers:**
- `_previous_attempt_info` has no src reader.
- Tests: `test_resume_detection.py` reads at :1433, :1449, :1563-1568, :1765, and does a raw set at :1458.
- Ratchet debt: `test_template_purity_ratchet.py:331-332` covers both `_backing` names (P9).

**RPU tests:** 0 direct references (scoped grep of `RichPythonUtils/test`). `test_loop_resume.py` and `test_recursive_resume.py` under `RichPythonUtils/test/rich_python_utils/common_objects/workflow/` exercise the runner's resume only indirectly [I].

## 21. Non-`InferencerBase` stages; `interactive` / `stream_observer` readers `[V]`

**Duck-typed stages:**
- `AF/mock_inferencers/mock_bta_components.py`:
  - `MockBreakdownInferencer` :26 (`stream_observer` :52);
  - `MockWorker` :63 (`interactive` :95, `stream_observer` :96);
  - `MockAggregator` :112 (:132).
- `mock_clarification_inferencer.py:13`.
- Registered at `AFS/common/configs/registered_targets.py:215-228` and used by the mock_task profiles (D9).
- Test doubles:
  - `test_bare_object_worker_factory.py:24-25, 70`;
  - `test_bta_worker_inferencers_yaml.py:75`;
  - `test_bta_conflict_detection.py:204-208`;
  - `test_graph_visualization.py:200-242`;
  - `test_breakdown_block_registry.py:30-35`.

**BTA guards for duck-typed stages:**
- ws binding is gated by `isinstance` (:2524, :2863, :3011);
- `hasattr` dispatch at :2603, :2747 / :2755, :2932, :2998 / :3004.

**Readers (the `_INVOCATION_KEYWORDS` declarations):**
- SIB `stream_observer` (field :214, hoisted at :1039). This is the only non-mock reader.
- BTA `interactive` (field :716): :2353, :2360 (results-review rerun), and `_bta.interactive` at :3223 / :3229 (selection).
- CI `interactive` (field :186): :549, :2343, :2641, :2693, :2859.
- `AF/agentic_inferencers/conversational/flow_node_adapter.py` :130 / :285.
- PTI :331 / :2051 / :2055.
- `AFS/agents/agent.py` :221 (not a stage).
- `RC/bindings.py:32` `RuntimeBindings.interactive` already exists; there is no `stream_observer` equivalent.

## 22. Private-hook tests; post-BTA `worker.*` reads `[V]`

**Direct `._ainfer(` / `._infer(` calls:**
- 162 calls on non-`self` receivers across 27 files in `TEST/`.
- 0 in `RichPythonUtils/test`.
- The largest blocks: `test_conversational_flow_node_adapter.py` 25, `test_resume_detection.py` 25, `test_dual_inferencer_resume.py` 24, `test_dual_consensus.py` 11. The Dual / PTI group is about 110.
- Every call skips the bridge, `_init_call_state`, the seed and `_(a)infer_single`.
- There are 10 more `self._infer` delegations inside test-double hooks. Those are not bypasses.

**Direct `_build_subgraph_spec` calls:**
- `test_bta_resume_workspace_binding.py:127`;
- `test_bta_worker_resume.py:72, 87, 101, 119, 132, 146`;
- `test_workspace_propagation.py:257`.

**Other private hooks:**
- `_make_breakdown_fn`: `test_bta_checkpoint_promotion.py:321`, `:363`.
- Mailbox hooks: see §31(c).

**Post-BTA `worker.name` / `.stream_observer` / `.interactive` reads: none.**
- `test_graph_visualization.py:210` sets it at setup.
- `test_task_real_cli.py:471 / 500 / 510` read at construction.
- The closest post-call stage read is `aggregator.template_extra_feed` (`test_bta_outcome_pairing.py:531`).

## 23. B28 family (transport → `_ainfer` handoffs) `[V]`

| Field | Declared | Writers | Readers | Getter |
|---|---|---|---|---|
| `_last_streaming_output` | TIB:70 | TIB:408 / 413 / 456; TSIB:442 (dead), :570 (sync), :608 (async) | TSIB:485 | claude :1066, devmate :1690 |
| `_last_streaming_return_code` | TIB:71 | TIB:409 / 414 / 457; TSIB:443 (dead), :571, :609 | TSIB:487 | claude :1067, devmate :1691 |
| `_last_streaming_stderr` | TSIB:92 | TSIB:440 (dead); claude :716; codex :566 | TSIB:486; claude :733-741; codex :578-579; devmate :927 | claude :1073 |
| `_last_stream_result` | undeclared | claude :591 / :701; codex :470 / :527-551 | claude :890-897; codex :659-668 | none |
| `_last_clean_output` / `_last_raw_stdout` | undeclared | rovodev §1 | rovodev :656 / :666 / :787 / :818 | `get_final_output` :641 |
| `_last_tool_use_count` (/ `_last_usage`) | claude-sdk :196; codex-sdk :129-130 | claude-sdk :350 / :352 / :385; codex-sdk :325 / :333 / :345-346 | claude-sdk :396; codex-sdk :354-356 | none |
| `_last_token_count` | devmate-sdk :120; metamate-sdk :194; rovochat :211 | per file | per file | none |
| `_last_response` | TAI:156 | :372 | :387, :426-430 | none |
| `_output_file` | devmate :299 | :770, :965 | :848-850 | none |

**Findings:**
- **F1.1.** TSIB defines `_ainfer_streaming` twice, at TSIB:391 and TSIB:575, and Python keeps the second. So TSIB:391-443 is dead, including the only TSIB stderr write (:440). devmate / kiro / rovodev CLI never capture stderr.
- **F1.2.** claude CLI (:560) and codex CLI (:454) override `_ainfer_streaming` without writing output or rc. TSIB:485-487 and claude `get_streaming_result` therefore see stale or default values.
- **F1.3.** The TIB docstring at :444 references a `get_streaming_result` that TIB does not define.

## 24. `_delegates_execution_under` callers `[V]`

Lines in this section are post-P2 (`6941db7c5e55`). Definition: IB:3198-3207 (the property `_delegates_execution` is at IB:3194).

**Remaining callers** (public entries without `@bridge_entrypoint`):
- devmate `infer` :1191. P3 commit 6 moves devmate's session policy into the seam hooks, which removes this caller; the helper is deleted with it;
- devmate `ainfer_streaming` (fixed in `71c8b0ab1513`: the adapter calls `super()`, and the base template decides delegation);
- rovodev `ainfer_streaming` (fixed in `8043f9828092`: the public override is deleted).

**Bridged entries and the templates use plain `_delegates_execution`:**
- claude :855 / :928, codex :631 / :682, devmate `ainfer` :1262, kiro :307 / :369, rovodev :842 / :917;
- SIB:814 (`_astreaming_source`) and SIB:1142 (`_streaming_source`); TPL:289.

## 25. devmate `inference_config` callers (B27) `[V]`

**Mechanism.** devmate `ainfer_streaming(self, prompt, filter_session_info=True, **kwargs)` (:1335-1339) receives SIB `_ainfer`'s positional `inference_config` (SIB:1041-1042) as `filter_session_info`. The config never reaches the backend: `None` disables filtering, and a truthy config enables it ([G] `test_golden_streaming_devmate.py:301`).

**Generic forwarders that can carry a non-None config:**
- IB 3026, 3273, 3288, 5098;
- SIB 1041, 1082;
- CI 2284, 2305, 3150, 3155;
- `flow_node_adapter.py:475`;
- PTI 2014, 2162, 2210;
- BTA 2632, 2748, 2949, 2999, 3149, 3156, 3394;
- LWI 1265 and ~1470.

**Experiment hub.** The `AFS/experiment_hub/` bridges build devmate (`experiment_bridge.py:444-456`, `implement_hypothesis_bridge.py:526-538`) inside Dual. Dual forwards only `_current_extra_inference_args`, not `inference_config`.

There are no direct devmate-typed call sites, and no OS `.py` references devmate.

## 26. Tier-3 handles other than `live_session_id` (P2 stop condition) `[V]`

- **devmate CLI, OpenClaw and claude_code CLI** make no `_tier3_*` calls. On direct streaming they touch the handle store only through `live_session_id` (via `active_session_id`). **The P2 stop condition holds for these three.**
- **The claude_code SDK** (claude-sdk) uses `_client` / `_disconnect_fn` / `_connected_loop` (:199-220). If P2 counts the SDK under "claude_code", its streaming path touches three more Tier-3 handles.
- **codex-sdk** (:136-156) and **rovodev-serve** (:96-116) are the only other `_tier3_*` users.
- **`_tier3_*` has no `legacy_mint` check** (SIB:529-560), unlike `active_session_id` (SIB:580-640).

## 27. Public-entry overrides `[V]`

"SINGLE" means the entry reaches `_(a)infer_single`. The "Post-P2" columns are at `6941db7c5e55`.

**Streaming entries (8):**

| Entry | Anchor | Reaches | Post-P2 |
|---|---|---|---|
| `SIB.ainfer_streaming` | SIB:751-786 | template | SIB:762, a plain method → template (`_ainfer_streaming_entry` :786, `_astreaming_source` :809; pipeline :827) (fixed in `384f501b6c0d`) |
| `SIB.infer_streaming` | SIB:1061-1114 | adapter | SIB:1100 → template (`_infer_streaming_entry` :1121, `_streaming_source` :1138, `_infer_streaming_pipeline` :1155, whose default is the thread bridge) (fixed in `384f501b6c0d` / `d1cafd6b4e16`) |
| claude `infer_streaming` | :1000-1056 | bypass | public override deleted; `_infer_streaming_pipeline` :1000 (fixed in `e10dbeefbc6a`) |
| rovodev `ainfer_streaming` | :722-803 | template via `super` | public override deleted; `_ainfer_streaming_pipeline` :722 (fixed in `8043f9828092`) |
| devmate `ainfer_streaming` | :1335-1439 | bypass on the non-delegate path | :1340, a signature adapter → template; pipeline :1374 (fixed in `71c8b0ab1513`) |
| devmate `infer_streaming` | :1499-1662 | own sync path | :1519, a signature adapter → template; pipeline :1584 (fixed in `e10dbeefbc6a`) |
| OpenClaw `ainfer_streaming` | :1061-1112 | bypass | :1095 and a new sync `infer_streaming` :1110, validating adapters → template; pipeline :1125 (fixed in `e9c79a0c96e4`) |
| `CI.ainfer_streaming` | CI:3129-3158 | delegates to `base_inferencer` under an `agent` child | unchanged (the recorded exemption) |

**CLI `ainfer` / `infer` override classes (6):**

| Class | Anchors | Reaches | Post-P2 |
|---|---|---|---|
| claude | :836 / :909 | SINGLE | :836 / :909 |
| codex | :627 / :678 | SINGLE | :627 / :678 |
| kiro | :288 / :350 | SINGLE | :288 / :350 |
| rovodev | :828 / :903 | SINGLE | :825 / :902 |
| devmate | `ainfer` :1211 (SINGLE); `infer` :1167 → `IB.infer` | SINGLE | `ainfer` :1216 (SINGLE at :1263 / :1269); `infer` :1172 |
| OpenClaw | `ainfer` :980 (bypass via `_ainfer_with_retry`); `infer` :1034 (adapter) | bypass / adapter | `ainfer` :1019, SINGLE at :1062 (fixed in `2274f592af6c`, B30); `infer` :1068 (adapter) |

**Other overrides:**
- IB `infer` / `ainfer` / `iter_infer` / `aiter_infer` / `__call__` / `parallel_infer` / `aparallel_infer` (IB:3854, 5231, 3946, 5295, 4113, 3980, 5325; post-P2 IB:3833, 5209, 3925, 5273, 4091, 3959, 5303);
- `AF/templated_inferencer.py` `__call__` / `infer_raw` / `infer` (:64 / :115 / :139; not an IB);
- `context_compressor.py:61-103`;
- `AFN/decorator.py:641` / :646;
- the mocks (duck-typed).

No other overrides exist in AFS (AST scan). This is the fixture list for `test_streaming_entry_contract.py` (added in `6941db7c5e55`; it checks the override shape and the P2 ctx half).

## 28. OpenClaw configs (B30) `[V]`

**No config sets framework knobs on OpenClaw.**
- It has no alias in `registered_targets.py`.
- No YAML or JSON in AgentFoundation/ or OpenStartup/ references it.
- Only its own `max_retries` (openclaw:173) is set:
  - `examples/.../example_openclaw_cli_mode.py:83`;
  - `example_openclaw_gateway_mode.py:68`, `:315`;
  - `TEST/common/inferencers/external/openclaw/test_openclaw_inferencer.py:297`.

**Latent risk [I].** `breakdown-multiflow-plan.yaml:139-143, 246-279` sets `max_retry`, `attempt_timeout_seconds` and `output_guardrail_inferencer` on `${_params.main_inferencer}`. That value can be overridden by env `DEFAULT_MAIN_INFERENCER` (`default.yaml:65`). All of those knobs are ignored if it points at OpenClaw ([G] :318).

## 29. Early-stop stream consumers (`aclosing` before P3)

**Syntactic early exits: 0** [V, AST scan of AgentFoundation/{src,test,examples}, RichPythonUtils, OpenStartup/{src,test}]. There is no manual `anext` / `aclose`.

**Implicit abandonment paths:**

| Site | Mechanism | Fix |
|---|---|---|
| SIB `infer_streaming` (SIB:1105-1114) | no `try/finally` around `yield`; the daemon thread keeps running; `thread.join` (:1111) and the error re-raise are skipped | P2/P3 `framed_gen` closes the inner stream (plain-method shape fixed in `384f501b6c0d`; post-P2 the bridge is `_infer_streaming_pipeline` SIB:1155 and still a daemon thread, so cancel+join lands in P3 commit 7) |
| `OS/services/websocket_interactive.py:292-306` `stream_token_batches` | `_send` raising would abandon `token_gen`. `send_safe` (`OS/routes/manager_websocket_routes.py:857-863`) catches `Exception`, so only `CancelledError` abandons. §33 A:316. | `aclosing` around `token_gen` before P3 (fixed in `8b552c9de7` by ownership: the CI turn that creates `token_gen` closes it, and `NodeStreamInteractive` closes its own `_tagged_stream`. `stream_token_batches` is a consumer, so it does not close the caller's upstream) |
| `OS/services/conversation_service.py:1272-1281` `astream_response` | an async generator; abandonment propagates and `add_message` is skipped | `aclosing` before P3 (fixed in `ab81b93a66`: `astream_response` closes the backend `ainfer_streaming`, and the new `_stream_fallback_tokens` closes `astream_response`) |
| pass-throughs (CI:3150, devmate :1361, rovodev :742 / :776) | propagate | covered by the entry's frame (post-P2: devmate :1340 is a signature adapter; the rovodev public override is deleted, `8043f9828092`). CI:3150 creates the leaf stream, so it owns and closes it (fixed in `8b552c9de7`). The internal pipeline wrappers get `aclosing` in P3 commit 7 |
| SIB `ainfer_streaming` `finally: exit_run` (SIB:785-786) | finalization from another Context raises "Token … created in a different Context" ([G] lifetimes) | B29 (P3) |

## 30. Between-yield readers (B29)

- **Writes after a yield** [V]:
  - claude-sdk :358 (`active_session_id` in the `ResultMessage` case);
  - rovodev-serve :319-320;
  - openclaw :1111-1112.
  - All three are skipped on abandonment, and each resolves against whatever ctx is active when the consumer resumes.
- **Consumer reads** [V]: none. There is no direct `active_run_context()` or `.active_session_id` read in any consumer loop body.
- **Observers** [V]: `stream_observer` (SIB:1049-1058) and `send_graph_event` (`AFS/ui/graph_interactive_adapter.py`; `_tagged_stream` :66-74; breaker :149) are guarded callbacks. See §33 A:329.
- **Goldens** [G]: B29a/b/c/d are confirmed by the lifetimes goldens (§32).

---

## 31. Existing-test classification (feed / modes, plan §9) `[V]` (each line read)

**(a) Purity assertions that stay green:**
- `TEST/common/inferencers/test_bta_inferencer_fanout.py:555` (`aggregator.template_extra_feed == {}`);
- `test_bta_outcome_pairing.py:531`;
- `TEST/run_context/test_afi3_mfi_feed_isolation.py:128, 164, 189, 206, 275, 342` (`"upstream_artifacts" not in …template_extra_feed`).

**(b) Legacy no-ctx instance writes:**
- `test_afi3_mfi_feed_isolation.py:304` (`== "LEGACY_UPSTREAM"`);
- `TEST/inferencers/test_propagate_to_children.py`: 87 matching lines across 30 tests. The prompt said "~40"; the path is `TEST/inferencers/`, not `TEST/common/inferencers/`.
- `TEST/common/inferencers/test_templated_inferencer_modes.py:254`.

**(c) BTA private hooks (move with P6):**
- `test_breakdown_block_registry.py:127` (the assertion spans :127-129, after `_inject_aggregator_extra_feed` :125) and :139;
- `test_research_propose_fanout_config.py:112` (`_seed_aggregator_feed`), :118 (`_inject_aggregator_extra_feed`);
- `test_bta_inferencer_fanout.py:487` (`_seed_aggregator_feed`).

**(d) Construction-level:**
- `test_mfdual_workspace_anomalies_integration.py:1195`;
- `test_slot_defaults_real_orchestrators.py:821`, `:835`;
- `test_bta_inferencer_fanout.py:407`, `:417`;
- `test_fanout_primitives.py:530-531`;
- `TEST/resources/tools/task/test_task_helpers.py:451`;
- `test_modes_slot_defaults.py:110`;
- `test_template_defaults.py:335`, `:375`;
- `test_research_propose_fanout_config.py:237`.

**(e) Out of scope:**
- `TEST/common/inferencers/conversational/test_soft_max_iterations.py:54, 93, 105` (CI `_last_template_feed`);
- `TEST/common/inferencers/test_dual_inferencer/test_preflight_template_variable_coverage.py:390`.

B17 needs no test rewrite.

## 32. Characterization findings from the P0 goldens (all pre-existing)

### OpenClaw (`test_golden_openclaw.py`)

| Finding | Evidence | Status |
|---|---|---|
| B26 | :285 | [G] |
| B19 | :351 | [G] |
| B30: `ON_EXHAUSTED` and the framework knobs are ignored | :301 rate limit; :318 settings ignored → 1 request | [G] |
| Three attempts, then "OpenClaw inference failed after 3 retries" | raise at openclaw:935-937 (message :936). The knob is OpenClaw's own `max_retries` (:173), not the framework `max_retry`. | [G] pre-existing quirk (off-by-one wording), not in scope |
| Retry delays `[8.0, 16.0]` | `retry_delay * attempt` (:174) | [G] |
| Warm-up flag set even with no warm-up | `_maybe_initialize_session` def :1181; `_session_initialized = True` at :1206 | [G] pre-existing quirk, not in scope |

### rovodev (`test_golden_streaming_rovodev.py:221`, parametrized over 3 entries × bare / ctx_fresh / ctx_seeded)

| Finding | Evidence | Status |
|---|---|---|
| B26: the session is resolved before the bridge | rovodev :760-771, before `super().ainfer_streaming` at :775 (the prompt's :764 / :776 are off by one) | [G] |
| B26b | not pinned; plan §11 c5 runs the fan-out only on the devmate and claude_code sync entries | n/a |
| Streaming captures no session: it runs with `--restore <backing>` and writes no `metadata.json` | golden output | [G] |

### claude_code (`test_golden_streaming_claude_code.py:236`, `:245`)

| Finding | Status |
|---|---|
| `ainfer_streaming` never resolves a session | [G] |
| `get_streaming_result` after `ainfer` returns `""` and rc 0 (F1.2) | [G] |
| `claude_code_bta_ainfer` → `"FANOUT_TEXT"` | [G] |
| B26b: sync `infer_streaming` with a fan-out runs the CLI and ignores the fan-out | [G] |

### Lifetimes (`test_golden_streaming_lifetimes.py:210`, `:251`, `:260`, `:299`, `:367`)

| Finding | Evidence | Status |
|---|---|---|
| B29a/b/c/d | goldens `async_abandoned_gc`, `async_abandoned_loop_shutdown`, `async_between_yields`, `sync_stream_abandoned`: the finalizer raises "Token … created in a different Context"; the sync transport thread is still alive at 0.5 s | [G] |
| B34: `state_factory` receives the `list_iterator` | `lazy_infer_iterator` | [G] |

### devmate (`test_golden_streaming_devmate.py:264`, `:275`, `:290`, `:301`, `:316`)

| Finding | Evidence | Status |
|---|---|---|
| Async stderr is lost | TSIB defines `_ainfer_streaming` twice, at TSIB:391 (dead; stderr write :440) and TSIB:575 (live). The prompt's "440 and 575" names the dead write, not the first def. | [G]/[V], F1.1 |
| A failed `ainfer` sets `consecutive_error_count` to 1 and keeps `active_session_id` | `devmate_ainfer_failure` | [G] pre-existing quirk, not in scope |
| Streaming entries never raise on CLI failure and never set `active_session_id` | golden | [G] |
| The session logger leaks onto the instance | `_ensure_ctx_workspace_logger` IB:2000 | [G] |
| B27: `None` → 9 raw lines; truthy → 2 filtered | :301 | [G] |
| B26b: sync `infer_streaming` ignores the fan-out (CLI run). Separately, `ainfer` with a fan-out returns a plain `str` (`"FANOUT(fanout question)"`) instead of a response object. | :316 | [G] |
| `dump_output` is always off in streaming | flip :1367-1374 / :1552-1560 | [G] (fixed in `59f0b0115ec6`, B11: still always off, now through a per-call `dump_output=False`, :1373 / :1553) |
| `infer_streaming` has no `auto_resume` check | golden | [G] |
| ANSI regex bug: `re.sub(r"\\x1b\\[[0-9;]*[A-Za-z]", …)` matches a literal backslash sequence, not ESC | devmate:1464 | [V] pre-existing quirk, not in scope |

### BTA (`test_golden_bta.py`, `test_golden_bta_finalize.py`)

**1. The U3b guard and the fallback path.**
- For workspace-bound BTAs, the B36 U3b guard (BTA.py:1841-1850) never fires: the aggregator workspace resolves (bound at :3016; D11).
- The fallback path returns before `_finalize_response` (BTA.py:2012) runs, so it produces no `outputs/`, report, manifest or symlink ([G] `finalize_after_fallback`: external_sync → `"fb:hello"`, no outputs).
- The plan's "links the failed attempt's aggregator output" needs an aggregator that wrote its output before the attempt failed. No golden exercises that; the B36 probe does [P].

**2. B37 through a sync retry.** A sync retry hits B37, and the chain then moves to the external fallback [G].

**3. Call sequences.**
- Sync: `[w0,w0,w0,fb]`.
- Async: `[w0,w0,w1,w1,w2,w2,fb]`.
- [G] `finalize_after_fallback`; default mode → `"default answer"`.

**4. A failing aggregator.**
- Async swallows it: `AggregatorFallback` (BTA.py:2967) → `_build_synthetic_aggregation` (:1303). The calls are `[w0,w1,w2,agg,agg]`, the result is the synthetic `"## Upstream Outcome 1..3"`, and outputs are present.
- Sync raises, then falls back: `[w0,w1,w2,agg,agg,agg,fb]`, `"fb:hello"`, no outputs.
- [G] `aggregator_failure`.

**5. `disable_aggregator` with an aggregator configured** skips the `outputs/workers/` symlinks, because `_finalize_output` takes the `if agg_ws is not None:` branch (:1852).
- Unset: `worker_00..02` symlinks, and the report holds only the last worker's answer.
- [G] `no_aggregator`.

**6. Oddities** [G], each a pre-existing quirk, not in scope:
- jsonfy node checkpoints are still `.pkl` (12 refs);
- the breakdown completes twice;
- the aggregator stream id is `"aggregator"` but the status id is `"gbta.aggregator"`;
- the host store holds only `/`;
- `child_reporter` is never called;
- `output_path` is empty;
- the MFI rerun parsers run twice (`test_mfi_interactive_rerun` :372; the §5.7 double post-processing);
- the async resume logs `ConcurrencyDiag` after `WorkerWsAssigned`.

**7. U3b through the public API.** Unreachable for workspace-bound BTAs [G], but reachable for workspace-less and `checkpoint_dir`-only ones [X] (D11).

**8. Manifest contributor order is not a contract.** In `artifacts/aggregation_report_manifest.json`, the contributors come from `sorted(os.listdir(session_dir))` in `_emit_output_manifest` (IB:2641-2697, listdir at :2652), over random logger-id filenames [V].

### Other goldens

| Golden | Finding | Status |
|---|---|---|
| `test_golden_locations.py` (:149 / :162 / :178 / :198 / :212 / :224) | B15 (D2) | [G] |
| `test_golden_legacy_workspace.py` (:251, :300) | the pre-refactor workspace fixture | [G] |
| `test_golden_bare_getters.py` (:374 … :651) | the §3 getters on bare calls | [G] |

## 33. Draft UNVERIFIED items, resolved

| Item | Verdict | Evidence |
|---|---|---|
| A:289 agentic-fn `_reset_session` | Mechanism [V]: `_reset_session(inferencer)` (`AFN/decorator.py:377` / `:466`, def :679) calls `inferencer.reset_session()` under the **caller's** ctx; the call runs under `rc`. Under a host ctx that clears the caller-path slot, not `rc`'s. Under legacy / no ctx both hit the instance backing, so bare behaviour is correct. | Fixed in `cbada6a67b`: the reset runs under `rc`. A regression test reuses a real `StreamingInferencerBase` leaf twice under one host root and asserts it never resumes. |
| A:316 websocket `_send` raising | Only `CancelledError` abandons the stream [V] | `send_safe` (`OS/routes/manager_websocket_routes.py:857-863`) catches `Exception` |
| A:329 observer / `send_graph_event` | Guarded [V]: an observer exception cannot abort the stream | SIB:1049-1058 (base); `_tagged_stream` :66-74; `send_graph_event` breaker :149 |
| B:253 `"### Result N"` vs `"### Upstream Outcome N"` | The test is stale [V]. It expects `"### Result 1"` (OST `resources/tools/task/test_task_real_cli.py:425-431`), but BTA emits `"### Upstream Outcome N"` (BTA.py:268-271, :1283). Not run (real CLI). | out of scope; the phase that touches BTA feed formatting (P6) may update it |
| B:318 YAML `worker_factory:` "silently stripped" | Corrected [V]: a warning, then delete (RPU `_instantiate.py:1453-1460`). Runtime effect [I]: the BTA has no workers. | D9 |
| B:319 Python `worker_factory=` / `workspace_root=` "likely TypeError" | Confirmed [X]: `TypeError` at construction | D9 |
| B:455 promoted file vs cap / selection | Promotion precedes the cap and the selection, so the file holds the uncapped, unselected list (= B35) [V] | D5 |

## 34. Deviations from the plan found in P0

| # | Deviation | Plan text | P0 finding |
|---|---|---|---|
| X1 | B37 added | "36 numbered defects" (plan line 91; §8 table) | 37 (D7) |
| X2 | `fresh_instance` conditional resolved: register the fanout | §5.7: "The fanout object itself is not registered by default … Only if `fresh_instance` copies … does the caller also register the fanout." | It copies (D1). **Also:** §5.7 says P6 "passes the instance in the same `proto.fresh_instance(..., aggregator_inferencer=stage.inferencer)` call". For an instance-valued prototype slot, that would hand every call the prototype's live aggregator, and seeding would write into it. P6 resolves from the fanout's own rebuilt slot instead. |
| X3 | Terminal targets repointed | §2.3 row h cites 7 ScienceModelingTools targets (lines 19, 32, 44, 57, 68, 81, 92) | one CoreProjects target (D10) |
| X4 | U3b reachability | §5.7 / B36: finalize "links the failed attempt's aggregator output … or raises U3b and loses the result" | Workspace-bound: returned but not finalized [G]. Workspace-less: U3b, result lost [X]. **New:** a `checkpoint_dir`-only BTA with an aggregator raises U3b on every call [X] (D11). |
| X5 | Root sites | §10 lists four per-turn-root callers (`conversation_service.py:1262`, TEX:670, :1046, `sop/cli.py:254`) | omits S1 `conversation_service.py:343`, the only root reused across turns (§10) |
| X6 | Host presets | §9 names `test_pti_resume.py:490` | 13 writes there, plus `test_recursive_resume.py`, `test_resume_detection.py:1458` and others (§1b) |
| X7 | §5.7 needs-stage list | [Dec] lists `:2139`, `:2920` as needs-stage | both are presence checks. The list also omits the resolved hits :263 / :288 / :330 / :1008 / :1831 / :1990 / MFI:878 / IB:3410 / :3512 (§15). |
| X8 | Private-hook tests | "about 30" | 27 files / 162 calls (§22) |
| X9 | Anchor drift (plan → actual) | | `_init_call_state` 3901 → 3900; MFI override 1483-1491 → 1486-1494; rovodev ContextVar 61-64 → 65-68; BTA `use_async` 2847 / 3038 → 2850 / 3041; BTA sync harvest 2737-2753 → **no write exists** (async only, :2655); devmate `dump_output` 1552-1557 → 1552-1560 (fixed in `59f0b0115ec6`: the flip is gone); WF resume-flag lines +1 off (§20); PTI 564 → 563 and 2144-2145 → 2146; LWI note cites PTI :571 → :563; B35 promotion 3189-3195 → 3194-3196; `_select_sub_queries` does not exist (inline 3222-3238); registry lambdas 942-955 → 942-953; §7 CLI post-call ranges differ by a few lines (actual in §12) |
| X10 | "production code has not changed since base" (task brief) | | stale: B8 (`39178f39d551`), B12 (`2ca561e6b299`), B14 (`2c02f7615283`), B6(a) (`8b92cead4ca6`) and B4 (`1eb6a19db727`) are committed (header). B4 depends on B6(a): it relies on the host setter writing `RESET` to the branch slot |
| X11 | Extra pre-existing defects E1–E4 | not in the plan | §36; fixed as P1 add-on commits: E1 (`972c1cda155c`), E2 with E3 (`cae65efbbca1`), E4 (`e456b12e3147`) |
| X12 | P2 commit order | §11 P2: commit 5 switches base `_ainfer`, the sync bridge and tool_as to the private pipeline and fixes B27 | the base `_ainfer` / tool_as switch and B27 are folded into commit 3 (`71c8b0ab1513`): once devmate's pipeline is keyword-only, a positional `_ainfer` → public-adapter call binds `inference_config` to `filter_session_info` through the adapter, so the switch and the keyword fix must land together to stay green. The sync-bridge switch landed last, as "5b" (`d1cafd6b4e16`). OpenClaw's sync `infer_streaming` adapter was added in commit 6 (`e9c79a0c96e4`) |
| X13 | No-mint delta is broader | §5.2 "Behaviour deltas" names bare rovodev `ainfer()` | a bare `ainfer()` on every `@bridge_entrypoint` leaf (claude_code, codex, kiro, rovodev, devmate) no longer legacy-mints in the nested streaming call, because base `_ainfer` now consumes the private pipeline. Session reads and writes are identical in legacy and no-ctx modes. devmate direct streaming now legacy-mints (as §5.2 says) |
| X14 | Sync template lifetime | §5.2: P2 changes no lifetime | the commit 1 sync template (`384f501b6c0d`) runs `enter_run` in the caller's thread on first `next()`, and the bridge thread reuses the root through `copy_context`. An abandoned sync stream therefore leaves the caller's ctx set until finalization (B29, P3 commit 7). The sync fan-out runs `_run_fanout` in the caller's thread without a per-call BTA disconnect (B25, P6) |
| X15 | Verbatim moves over 50 lines | "functions ≤ 50 lines" (conventions) | devmate `_ainfer_streaming_pipeline` / `_infer_streaming_pipeline` were moved verbatim (§11 P2 "Logic preserved"); both carry a pre-existing C901 (async 13, was 15; sync 19) |
| X16 | `_delegates_execution_under` kept | §11 P2 exit: delete it if P0 found no other callers | devmate `infer` still calls it (§24); P3 commit 6 moves that caller into the seam and deletes the helper (fixed in `2c1976b6d953`) |
| X17 | `live_store` creation | — | created lazily (`{}`) on the first host-ctx read, not eagerly |
| X18 | OpenClaw streaming entries | §5.2: public override = validate, then `super().ainfer_streaming(...)` | the public entries are plain methods that validate eagerly (at stream creation, not first `next()`); a new sync `infer_streaming` adapter validates the same way; `type: ignore[override]` removed |
| X19 | Unrelated autofixes | — | `arc lint -a` suggestions in touched files unrelated to the change (e.g. `NoGetEventLoop`) were not applied |
| X20 | OpenClaw session after a terminal failure | §5.2 B30: the attempt count and session are unchanged | `ainfer` / `infer` reset `active_session_id` to `None` after a terminal failure: RPU's retry helper calls `pre_retry` after the last attempt too, and OpenClaw's `pre_retry` calls `reset_session`. This matches OpenClaw streaming today |
| X21 | OpenClaw retry-override WARNING | §5.2: "logged at WARNING" | through `logger.warning`, following the Q3 precedent (not `warnings.warn`) |
| X22 | OpenClaw per-call `fallback_mode` | not in the plan | a per-call `fallback_mode` kwarg to `ainfer` / `infer` is dropped with a WARNING (`_drop_per_call_fallback_mode`); `ON_FIRST_FAILURE` would otherwise mean two attempts |
| X23 | OpenClaw BTA fan-out | not in the plan | `_SUPPORTS_BTA_FANOUT = False`, with a new reason: every fan-out worker would resolve the one configured `session_id`. The error reads "<Class> does not support bta_inferencer; leave it None" (§35) |
| X24 | OpenClaw `on_retry_callback` | not in the plan | fires once, on the terminal failure (one base attempt); the base retry path adds its own log lines |
| X25 | `_FANOUT_DROPPED_ARGS` | not in the plan | devmate adds `filter_session_info`, `stream_callback` and `output_stream`, which have no meaning for fan-out workers |
| X26 | reflection child slot (D8) | `_rc_child("reflection")` | `reflect` (`f2622c8fbb`): the step name the async LWI workflow already uses, so sync and async reflective runs agree |
| X27 | agentic-fn concurrent same-slot calls (§9, §35) | "needs a per-call slot suffix decision" | decided: no automatic suffix. The documented per-call `run_context_slot` callable is the mechanism, and the P3 claim rejects an overlap. The one production caller (the Metamate code-scope judge) runs sequentially per `_ainfer` (`cbada6a67b`) |
| X28 | agent reasoner slot (D8) | "a child slot" | fixed slot `reasoner` (`12c48ee8ad`). Safe because the agent's sync `WorkGraph` has no executor and no `use_async`, so branches run sequentially in one thread, and `Agent.copy` is shallow, so branches share one reasoner and never overlap at `/…/reasoner` |
| X29 | CI tool child path (§5, §6) | `tool/<name>` | sync `tool/<name>`; async `tool/<name>/async_<n>`; the turn's workspace object is kept; commands dispatched as tools stay at the turn ctx; `_rc_child(slot, *subslots)` chains components and replaces `..` inside a component with `_` (`a393f0248c`) |
| X30 | workspace-logger un-defer (plan §4.7 ALLOWED row) | "idempotent" | it was a check-then-act race under `parallel_infer` threads, and the in-place dict insert broke a concurrent `Debuggable.log` iteration (E7). Now a module `threading.RLock` with a re-check, and copy-on-write for `logger`, `_resolved_logger_configs` and `_ws_log_relpaths` (`02a7a80250`). The ALLOWED row stands |
| X31 | agents suite baseline | not in the plan | 6 fail / 18 pass before and after every pre-P3 fix: `test_agent.py`'s `MockInteractive` defines `get_input` instead of the abstract `_get_input` (pre-existing) |
| X32 | S2 continuity (§10) and the store policy (§14) | `[OPEN→P3]` | S2 keeps a fresh ephemeral root per `/stream` call by design, with continuity from history replay. A store object backs at most one live runtime in one process; cross-process sharing stays last-writer-wins, and P8's lease covers BTA checkpoint roots |
| X33 | P3 commit 1 contents (§11 P3) | commit 1 lands `framed_agen` / `framed_gen` with the frame | `framed_agen` / `framed_gen` land in commit 7: they compose the claim (c3), the outcome clear (c4) and the seam steps (c5). Commit 1 lands `open_invocation` / `aopen_invocation` with steps 2 and 9–11 (frame, ledger, cleanup errors, release); c3 / c4 / c5 / c10 extend them with the claim, the outcome clear and publish, the keyword pop and hooks, and the compat flush. `ResourceLedger` also gains `close_joined` (sync close through `run_async_joined`), used by `open_invocation` |

Corrections to the prompt:
- TSIB "440 / 575" → defs at 391 / 575; 440 is the dead stderr write.
- OpenClaw ":1181" → def; the flag is set at :1206.
- rovodev ":764 / :776" → :763 / :775.
- `max_retry=3` → OpenClaw `max_retries`.
- `test_propagate_to_children` "~40" → 87 lines / 30 tests.
- rovodev B26b is not pinned.
- U3b reachability (X4).

Corrections to the drafts:
- C: `_wrap_bare_factory_inferencers` / `_PrototypeCloneFactory` live in IB (:1623 / :194), not BTA.
- D: rovodev `_last_clean_output` write 584 → 582.
- B: "silently" → warn then delete. The stale-caller list omitted the OS Python builders and 7 YAML files.

## 35. `[OPEN→Pn]` roll-up

| Item | Phase | Why deferred |
|---|---|---|
| PTI dead-code sites (§1a) | P1 | P1's dead-code pass owns deletion |
| Full host-preset sweep (§1b) | each phase | presets follow the field each phase moves |
| devmate `infer` pre/post outside the bridge (§12) | P3 | moves into `_prepare_call` / `_conclude_call` (fixed in `2c1976b6d953`) |
| devmate non-delegate streaming `run_context` in kwargs (§12) | P2 | P2 routes the entry through the template (fixed in `71c8b0ab1513` / `e10dbeefbc6a`) |
| CI:2200 async tool under the turn ctx (§6) | P3 | tool bodies are dynamic; derive a child before the claim (fixed in `a393f0248c`) |
| Concurrent agentic-fn calls sharing `parent/<fn>` (§9) | P3 | the claim rejects them; needs a per-call slot suffix decision (decided in `cbada6a67b`: no suffix; X27) |
| A:289 consequence (§33) | P3/P5 | runtime slot contents; the fix lands with D8 (fixed in `cbada6a67b`) |
| S2 `/stream` continuity (§10) | P3 | product decision on whether streams continue a session (decided: fresh root per stream; X32) |
| Distinct `RuntimeBindings` store policy (§14) | P3 | policy, not a code fact (decided; X32) |
| Guardrail judge retry-path persistence (the same `guardrail` path across host retries keeps the judge's slot) [I] | P5 | P5 owns judge isolation (B22) |
| S1 concurrent turns (§8) | P6 | the guard's serialize-vs-reject choice |
| Python-built OpenStartup graphs (§8) | P6 | not exhaustively read; the guard fails loudly |
| Workspace-less / `checkpoint_dir`-only U3b golden (D11) | P6 | only `[X]` today; add it with the B36 fix |
| BTA sync harvest anchor (plan 2737-2753) (X9) | P6 | re-check when the contract harvest moves |
| rovodev B23 user `output_file` clobber (rovodev :802-803 sets `_last_clean_output = None` when the user supplied `output_file`) [I] | P10 | P10 owns rovodev output handling (fixed in `e6b7b8a40b04`; X85) |
| YAML `worker_factory:` runtime effect (D9) | out of scope | `[I]`; a pre-existing defect |
| OpenClaw `bta_inferencer` fan-out support (X23) | P10 | rejected at validation today; decide before P10 (decided in P10: stays rejected; X90) |

Resolved: the B15 / B31 / B34 / B35 / B36 / B37 verdicts are all CONFIRMED by strict-xfail probes (D2–D7); their fixes stay in P9 / P7 / P3 / P8 / P6 / P8.

## 36. Extra pre-existing defects (outside the numbered register)

E1–E4 were reproduced [X] at `1eb6a19db727`. E5 and E6 were found while writing B5's sync regression test. All six are fixed in P1. E7 was found as a buck flake during the pre-P3 fixes and is fixed before P3. Each fix has a regression test that fails on its parent.

| # | Defect (anchor) | Failure | Root cause | Fix |
|---|---|---|---|---|
| E1 | `parallel_infer` result merge (IB:4101-4109 → RPU `mp_utils/common.py:236`, `_default_merger_1`: `return sum(results, [])`) | `parallel_infer([...], num_workers=2, use_threading=True)` on a `str`-returning inferencer raises `TypeError: can only concatenate list (not "str") to list`; with `num_workers=1` it returns an `MPResultTuple`, not a list. The B34 probes use `num_workers=1` to get past it | the `"list"` merger assumes each worker returns a list; `sum(..., [])` is also quadratic | P1 add-on commit, with a regression test for both worker counts (fixed in `972c1cda155c`: the merger is dropped and the workers' contiguous chunks are flattened in order; `debug=True` runs every input with `num_p=1`) |
| E2 | LWI default iteration record (LWI:963-979, `_record_iteration`) | a looping LWI with a workspace and no `iteration_record_builder` fails with `RecursionError` in `map_helper.dict__` (:455 / :476), reached from `_save_loop_checkpoint` (LWI:1852) → `Workflow._save_checkpoint` (RPU `workflow.py:599`) → `_save_result` (LWI:1816). The retry chain then re-runs the whole workflow twice more. The B15 probes pass an `iteration_record_builder` to get past it | the default snapshot copies the non-underscore `iteration_records` list into the record, then appends that record to the same list, so every record, starting with the first, contains the list it is appended to: a reference cycle | P1 add-on commit: the default snapshot excludes `iteration_records`; regression test on a looping LWI with a workspace (fixed in `cae65efbbca1`; the `run_context` golden `locations/lwi_loop_default_record_builder.json`, which characterized the `RecursionError`, is regenerated: final state `iteration` 3 with two records, backing workspace `<WS>/lwi/iteration_2/iteration_3`, the B15 nesting; regenerated again with the B15 fix in `7a1c4167a411`) |
| E3 | `test_lwi_properties.py:156`, `:200` | both call `_make_lwi_with_workspace(..., workspace=InferencerWorkspace(...))`, but the helper (:28) takes `workspace_root`, so each would raise `TypeError`. The file is in no BUCK target (it imports `hypothesis`), so nothing runs it | stale helper keyword | with E2 (fixed in `cae65efbbca1`: the tests pass `workspace_root=`; `test_iteration_workspace_directory_creation` is a strict xfail citing B15, removed in P9) |
| E4 | RPU `Workflow` resume start index (`common_objects/workflow/workflow.py:1111-1112` in sync `_run`, def :1051; `:1466-1467` in async `_arun`, def :1415) | on a workflow without loops, `resume_with_saved_results=True` is read as the step index `1`, because `bool` is a subclass of `int`. A 4-step workflow with all four results saved resumes by scanning back from step 1, so it re-runs steps 2 and 3; a 1-step workflow raises `IndexError: list index out of range` (`self._steps[1]`). `resume_with_saved_results=3` re-runs nothing. A static-mode LWI with a workspace sets `True` (LWI:1030), so a resumed LWI without loops and with at least three steps re-executes completed steps. LWI's `_auto_enable_checkpointing` docstring (LWI:1018-1022) records the `IndexError` as intrinsic to the scan and leaves resume off in dynamic mode to avoid it | `isinstance(self.resume_with_saved_results, int)` accepts `True`; the docstring (:71-75) defines `True` as "resume from the last saved step" and an `int` as a step index. The loop-checkpoint branch (:1082 / :1438) compares `is True`, so only workflows with loops take the documented path | P1 add-on RPU commit: exclude `bool` at both sites; regression test for `True` on 1-step and 4-step workflows, sync and async (fixed in `e456b12e3147`; LWI's `_auto_enable_checkpointing` docstring now gives the real reason dynamic mode leaves resume off: its initial step list does not describe the expanded steps, and it resumes through `expansion_step_registry`) |
| E5 | RPU sync `execute_with_retry` single-attempt fast path (`common_utils/function_helper.py`) | with `max_retry <= 1`, no `total_timeout` and no fallback, the result is returned without calling `output_validator`; async validates every attempt, so a guardrail with `fallback_mode=NEVER` judged `ainfer()` but not `infer()` | the fast path returned before validation | the single attempt is validated and a rejection takes the exhausted-retry terminal (fixed in `933507187199`) |
| E6 | RPU sync `execute_with_retry` terminal with no default (`common_utils/function_helper.py`) | an `OutputValidationExhaustedError` from the last attempt is wrapped in a generic `Exception`, so `except OutputValidationExhaustedError` handlers and `non_retryable_exceptions` never see the type; async re-raises it unchanged | sync-only generic wrapper | re-raised unchanged, as async does (fixed in `45f0996cb685`) |
| E7 | workspace-logger un-defer (IB `_ensure_ctx_workspace_logger`, `_configure_for_workspace`, `_add_workspace_logger`, `_tag_ws_log_relpath`) | under `parallel_infer`, one item un-defers the shared instance's workspace logger while another item's `Debuggable.log` iterates `self.logger`: `RuntimeError: dictionary changed size during iteration` (the buck flake `test_parallel_infer_results::test_str_results_under_a_run_context[N]`); two threads can also both create a JsonLogger | a check-then-act flag plus in-place dict inserts on a shared instance | `_undefer_workspace_logger` re-checks the flag under a module `threading.RLock`; the three logger dicts are replaced copy-on-write (fixed in `02a7a80250`; X30) |
| E8 | `checkpoint_dir`-only BTA with an aggregator (BTA `_finalize_output`, U3b) | the run raises "BTA aggregator is configured but its workspace could not be resolved" after every stage succeeded | the aggregator gets a workspace only from a BTA workspace (§13: aggregator bind is workspace-only), and U3b refuses the no-aggregator branch | `[OPEN]`, outside the register: found while writing P6 c1's `checkpoint_dir` layout test; identical at `6018ed964db2`. The P6 c1 test pins the layout with `disable_aggregator=True` |
| E9 | `checkpoint_dir`-only BTA with `enable_result_save=True` | a worker node raises `NotImplementedError` from `Resumable._get_result_path` (`WG:1155`) | the expansion nodes inherit the breakdown node's `enable_result_save=True` (`_propagate_settings_to_subgraph`), and worker nodes have no `checkpoint_dir` rule (§13: worker node ckpt is workspace-only) | `[OPEN]`, outside the register; found and verified with E8 |
| E10 | multi-iteration PTI with a workspace (LWI `_setup_iteration` → PTI `_get_iteration_workspace`) | the second meta-iteration's first step raises `TypeError: PlanThenImplementInferencer._get_iteration_workspace() takes 2 positional arguments but 3 were given`; the retries re-run and fail the same way, so no multi-iteration PTI with a workspace ever completed (present before this stack, at `2cbacc7066a9^`) | `_setup_iteration` called `self._get_iteration_workspace(base, n, factory)`, which resolves PTI's two-argument staticmethod override instead of LWI's three-argument helper | P9.5: `_setup_iteration` names `LinearWorkflowInferencer._get_iteration_workspace` explicitly; regression test `test_lwi_iteration_workspace.py::test_a_multi_iteration_pti_runs_every_meta_iteration` (fixed in `7a1c4167a411`) |
| E11 | PTI iteration history | once E10 was fixed, a three-iteration run reported `iteration_history` `[1, 2, 2, 3, 3]`, `total_meta_iterations=5` and, with `max_meta_iterations=3`, `meta_iterations_exhausted=True` | PTI's analysis step records each meta-iteration, and PTI also set `iteration_record_builder`, so the workflow's `_record_iteration` at each iteration change added a second record built after `iteration` advanced (the new number with the previous outputs) | P9.6: PTI overrides `_record_iteration` as a no-op and drops the builder (fixed in `0bf2a6247dde`) |
| E12 | MFI with `propagate_runtime_input` and `inject_upstream_artifacts` (followup builder, `cfg.get("input")`) | under any ctx — every public call since P3 — each followup step received the configured placeholder (`""` when the YAML omits `input`) instead of the runtime task | the M5 read-flip published the runtime inputs to the ctx node instead of rewriting `flow_configs`, but the followup builder kept reading `cfg["input"]` | P9.9: the inputs live on the attempt (`_BtaAttempt.effective_sub_queries`) and the builder reads `_flow_input(index, cfg)`; regression test `test_m5_definition_immutable.py::test_every_step_of_every_flow_gets_the_runtime_input` (fixed in `cb50055f7ccd`) |
| E13 | terminal exit poller (TSIB `_poll_process_exit`, used by `_read_stdout_with_exit_detection`) | a successful CLI call intermittently reads back as "Command failed with code 255" (the P10 buck gate: a devmate golden, full stdout, return code 255; three reruns passed) | `os.waitpid(pid, WNOHANG)` reaps the child, so asyncio's child watcher finds no child and reports 255; the poller wins the race when the loop is busy as the child exits | poll with `os.waitid(..., WEXITED | WNOHANG | WNOWAIT)`, which leaves the zombie for asyncio to reap (`5cbc7ce2e189`); deterministic regression test |
| E14 | OpenStartup BTA configs still pass `worker_factory` (`role_setup.yaml`, `_dev/role_setup.yaml`, `role_setup_skill_tool_creation.yaml`, `project_onboarding.yaml`, `create_role_bta.yaml`, the five `mock_task/profiles/*.yaml`, nested BTA nodes included) [V] | the config loader drops the key with a warning ("Removing YAML key 'worker_factory' — not a valid __init__ param"), so those BTAs get no workers from the YAML; the Python constructors in `role_setup/executor.py`, `create_role/executor.py` and the two experiment bridges pass it too and raise `TypeError` | BTA's worker field was consolidated into `worker_inferencers`; C8 migrated `breakdown-multiflow-plan.yaml` but not the OpenStartup tools, whose tests and overrides (`worker_factory.<key>`) also use the old name | `[OPEN]`: an OpenStartup migration (configs, executors, preflight tests), outside this plan; found by the P11 config audit |
| E15 | a BTA call with `resume_with_saved_results` off, on a root holding an earlier call's checkpoints [X] | it overwrites only the checkpoints it reaches; if it dies, a later resume loads the earlier call's results for the rest. The P8 probe resumed such a call and got the earlier call's aggregation back without running a stage | nothing recorded which call wrote which checkpoint; P8.4 as first written let the second call's header vouch for the tree | P8.4: such a call leaves the root without a manifest (X102), so the resume fails closed |
| E16 | an `@agentic_function`'s cached inferencer under concurrent host calls (`agentic_functions/config.py` `InferencerProvider`) [X] | G5 (research_propose with flow_03 fanned out): the Metamate fan-out workers judge their tasks concurrently through `judge_code_scope`, whose one cached `PlugboardApiInferencer` is not host-pure certified, so every overlapping call raised `UncertifiedConcurrentUseError` and the judge fell back to the default `fbsource` scope | P6 c7's guard refuses overlapping host invocations of one uncertified instance; the provider shares one instance per decorated function, and the §8 inventory did not list it (the instance lives inside the decorator) | `InferencerProvider.for_call`: under a host context, a call whose shared inferencer is not certified runs on its `fresh_instance()`; certified inferencers and bare calls keep the shared one |
| E17 | Devmate CLI passes the prompt on its command line (`devmate run ... freeform "prompt=$PROMPT"` through `/bin/sh`) [X] | a prompt over 128 KiB fails before Devmate starts: `[Errno 7] Argument list too long: '/bin/sh'`. G5: flow_01's round01 step for two planner workers (initial outputs 97 KB and 83 KB, carried into the follow-up) produced no response after four attempts | Linux's per-argument limit (`MAX_ARG_STRLEN`, 128 KiB); pre-existing, the command line is unchanged since before this plan | `[OPEN]`: pass long prompts through a file or stdin; outside this plan |

**Fixed so far** (P1, in order): B12 (`2ca561e6b299`), B14 (`2c02f7615283`), B6(a) (`8b92cead4ca6`, pulled forward from P3 commit 11 because B4 depends on it), B4 (`1eb6a19db727`), E6 (`45f0996cb685`), E5 (`933507187199`), B5 (`cbb056d03259`), B9 (`dc02a72082e4`), B11 (`59f0b0115ec6`), E1 (`972c1cda155c`), E2 and E3 (`cae65efbbca1`), E4 (`e456b12e3147`). B6(b) stays in P3 commit 11 (`LiveHandleField`).

**Fixed so far** (P2): B26 (`384f501b6c0d`, `8043f9828092`, `71c8b0ab1513`, `e10dbeefbc6a`, `e9c79a0c96e4`), B26b (`e10dbeefbc6a`), B27 (`71c8b0ab1513`), B30 (`2274f592af6c`).

**Fixed so far** (pre-P3): (a) `8b552c9de7`, (b) `ab81b93a66`, (c) `f2622c8fbb`, (d) `cbada6a67b`, (e) `12c48ee8ad`, (f) `a393f0248c`, E7 `02a7a80250`.

## 34. P3 c3 deviations (path claims)

- The contract errors live in the leaf module `run_context/errors.py`, not in `invocation.py`: `store.py` raises them and `invocation.py` imports `store` through `bridge`, so defining them in `invocation.py` would create an import cycle.
- The store lock is a `threading.RLock`: `evict_subtree` holds it while calling `claims.live_below`, which takes the same lock.
- The claim is acquired in `_claim_path` before `_current_invocation` is set, so a rejected call binds nothing and leaves the frame chain unchanged.

## 34. P3 c6 deviations (seam hooks)

| # | Deviation | Plan text | Finding / decision |
|---|---|---|---|
| X34 | Provider hooks skip calls that run no provider | §5.1 steps 6 and 8 run for every entry | `_runs_provider(render_only)` (`795c57a23aa4`): a render-only call and a call delegated to a per-call BTA (`_delegates_execution`) never reach the backend, so neither hook runs. This is the CLI adapters' `_delegates_execution` short-circuit, decided once per invocation. A render-only call through a CLI adapter used to apply the session policy and run the post-call code (devmate reset its error counter); now it touches no session state |
| X35 | Async post-hook `_aconclude_call` | §5.1: both hooks are sync, with identity defaults | `_aconclude_call(result)` (`795c57a23aa4`), defaulting to `_conclude_call`, runs on the async entries. devmate's async post-call code `await`s backoffs, and claude/codex/rovodev/devmate run different post-call code on the two paths (§12 note). `_prepare_call` stays one sync hook: every leaf's pre-call code is identical on both paths |
| X36 | `parallel_infer` / iterator items run the provider hooks | §4.4 lists `_init_call_state` per item | the hooks run for every invocation, so items of a CLI leaf now follow its session policy. No production caller (§8). Under a host ctx each item has its own `parallel_{i}` slot; bare items share the instance backing, as bare concurrent calls already do (§4.6) |
| X37 | devmate `infer` override deleted | §4.4: overrides become thin adapters | with the policy in the hooks, devmate's `infer` was a pure `super().infer(...)` pass-through; deleting it keeps base `infer`'s mint policy and fixes the §12 `[OPEN→P3]` ctx mismatch |
| X38 | Tests that mocked `_(a)infer_single` | — | rovodev `TestSessionManagement` / `TestAinferAcceptsWrappedResponse` and `test_large_input_mode.py::test_ainfer_preserves_session_kwargs` patched the seam entry to observe the adapters' pre/post code; they now patch the transport (`_ainfer` / `_infer`) the seam runs |
| X39 | Real-CLI session isolation test | — | `external/claude_code/test_claude_code_cli_session_isolation.py` runs the real `claude` binary and fails at the P3 c6 baseline too: Claude Code's auto-memory (`~/.claude/projects/-tmp/memory/secret_word.md`) carries the "secret" across sessions. Excluded from the routine local runs as environment-dependent |

## 34. P3 c7–c12 and P4 c1 deviations

| # | Deviation | Plan text | Finding / decision |
|---|---|---|---|
| X40 | P4 commit 2 (compat takeover of `_last_rendered_task_instructions`) moves into P6 commit 4 | §11 P4 c2: the leaf field is written only through `publish_result`; BTA, MFI and Dual readers switch in P6 | BTA (`_worker_task_instructions` harvest), MFI and Dual read the leaf through `_proposer_task_instructions()` until P6. A host-silent compat field would blank `<OriginalTaskInstructions>` for every host Dual / MFI / BTA between P4 and P6, so the takeover lands in the P6 commit that switches those readers (the §13 rule "publishers and their readers land in one commit"). P4 c1 (`67dd5d06bedd`) publishes the typed contract and adds `_task_contract_at`; its preview test shows a Dual preview never publishes or changes the leaf's published contract |
| X41 | Streaming chains close by ownership | §5.1: "aclose / close of the inner pipeline … run inside the same binding" | `59f205cbd6fc`: the pipelines iterated the generators they create with a bare `async for`, so an early close left the transport to asyncio's GC finalizer, in another task and context (and the base pipeline's `_generator_cleanup_timeout` guard, which nothing sets, was the only explicit close). The base pipeline, rovodev, devmate, OpenClaw and the two TSIB transports now hold nested generators with `contextlib.aclosing` (devmate: explicit `aclose` in `finally`) |
| X42 | `_init_call_state` on the streaming entries | §5.1: steps 1–5 at the first resumption | the templates' `start` callable runs it inside the frame before choosing the fan-out or the pipeline (`c90af1105498`) |
| X43 | Loop-shutdown lifetime golden | — | at loop shutdown asyncio closes every live async generator concurrently, in no fixed order; after X41 the transport is closed either through the chain (stream ctx) or directly (none). The golden pins "one of the two" (folded into `59f205cbd6fc`) |
| X44 | rovodev output-file fallback | §5.1: readers fall back to `self.output_file` when frameless | `_call_output_file()` (`1e039b0df8e0`); the new consumer-`copy_context()` contract test failed exactly for rovodev's async entry before the fix |
| X45 | `LiveHandleField` and session-scoped helpers | §5.5 | `_session_scoped_get(name, default, *, backing)` / `_set` (backing defaults to `_<name>`; `active_session_id` uses `_session_id`); a no-host reset clears `name` in every branch (B6(b)), which also stops a failed bare call's recovery from resuming a host branch's session (`getters/streaming_session` golden). The Tier-3 helpers move to `InferencerBase` unchanged (`7c8faefbb167`) |
| X46 | B32 keying | §5.5 | `LiveHandleStore.scope_id` is a uuid4 hex; `RunContext.handle_scope` / `live_branch_key`; the leaf store keys are `(scope, path)` pairs (`LiveHandleStore` keys are any hashable); golden views render them with `live_branch_label` (`8842284f9482`). No per-turn-root caller needed a migration (§10) |
| X47 | `BARE_EXPECTED_CHANGES` | I8 names it | created in `test_golden_bare_getters.py` (`7c8faefbb167`), listing every intentional bare-visible change so far |

**Fixed so far** (P3): B34 (`f0e15b47af`), B29 (`c90af1105498`, `59f205cbd6fc`, `1e039b0df8e0`, `51b2fd0c8827`), B6(b) (`7c8faefbb167`), B32 (`8842284f9482`). P3 exit: `test_p3_stop_gate.py` (`2afb659d285d`); buck `run_context` + `common/inferencers/...`: Pass 1506, Fail 1 (the pre-existing m9), Skip 19, Build failure 2 (pre-existing `metamate_standalone`).

**Fixed so far** (P4): B10, generic half (`67dd5d06bedd`): the templated leaf and LWI publish their contract; BTA / MFI / Dual in P6.

## 34. P5 deviations

| # | Deviation | Plan text | Finding / decision |
|---|---|---|---|
| X48 | A second ctx channel for a templated parent's own feed / modes | §5.9 B17: "publishes at the parent's own node, via `publish_child_template_feed` and a new `TEMPLATE_MODES_OVERRIDE_HANDLE`" | `publish_child_template_feed`'s handle is also where an orchestrator puts a per-call override for a child at the child's own node (BTA's aggregator `upstream_artifacts`), and a renderer reads its own node's override. A templated leaf with children (judge, fallback) publishing its own feed there would let its definition beat the orchestrator's per-call keys at its own render and persist that into the connection-scoped handle. `publish_propagated` / `resolve_propagated` (`TEMPLATE_PROPAGATED_FEED_HANDLE` / `_MODES_HANDLE`) are read only from strictly above, stop at the scope barrier, and an outer publication wins, matching the legacy push (`77f34bcf9058`) |
| X49 | `_applied_role` stays no-ctx-only | §5.9: "`_role_history` and `_applied_role` are written in no-ctx and legacy modes only" | a legacy ctx records the role in `RoleState` and never writes the template fields, so an instance `_applied_role` would outlive the call and make later bare calls report a role their fields don't carry (`_select_bta_template`). `_role_history` is written in no-ctx and legacy, `_applied_role` in no-ctx only (`373d92341a20`) |
| X50 | Host role audit entry shape | §5.9: "bounded `node.provenance`" | the entry records the role, time and the changed attribute *names*, last 64 entries; the values live in the node's typed `RoleState`, so provenance stays JSON-serializable (I3) (`373d92341a20`) |
| X51 | `interactive` declarers | §5.9: BTA, PTI, Conversational | `ConversationalFlowNodeAdapter` declares it (it reads a per-call `interactive`); `ConversationalInferencer` stays the documented bypass (O2) and does not (`fdc5721c9ba2`) |
| X52 | BTA asks the graph reporter for what it asked before | §5.9 dispatch | an interactive handle for every leaf worker, an observer for a stage that has one; only the delivery changes (declared keywords; duck-typed stages keep attribute writes), so the reporter interaction in the BTA goldens is unchanged (`fdc5721c9ba2`) |
| X53 | P4 c1 reader test harness | — | the inferencer tests' `conftest.py` prepends its sibling `src`, so a baseline comparison must run a self-contained snapshot tree (`/tmp/isr/mkbase.sh` copies `src` and `test`); a PYTHONPATH-only snapshot tests the live sources on both sides. The corrected whole-directory comparison found three resume tests broken since P3 c7 (a stub re-entered `ainfer_streaming`), fixed in `01d927f3db1d` |

**Fixed so far** (P5): B16 / B16b (`373d92341a20`), B17 (`77f34bcf9058`), B22 (`47387d7814db`), B18 (`fdc5721c9ba2`). B31 (`bta_node_name`) stays P7.

**Fixed so far** (P6): the BTA call record (`ee434e66d69b`), B3 / B21 with the attempt loop (`2e8e53c8a5ea`), B36 (`a1705aafb9f0`), B1 / B13 and the X40 compat takeover (`e07e80f5b3de`), B25 (`0ee6ae4208f9` workers; `d666fc1e931a` aggregator and sync fan-out), B2 (`d666fc1e931a`), the single-flight guard with every §8 row resolved (`e3fd74cf8568`). Pre-existing, outside the register: E8, E9 (`[OPEN]`). Buck at `e3fd74cf8568`: `run_context` Pass 897 / Fail 1 (m9) / Skip 16; the rest of `test/agent_foundation/...` Pass 794 / Fail 0, Build failure 2 (metamate standalone, pre-existing).

## 34. P6 deviations

| # | Deviation | Plan text | Finding / decision |
|---|---|---|---|
| X54 | `_bta_call()` takes the record on first use, from the invocation's own ctx | §5.7: "put at the start of BTA's `_ainfer` / `_infer`" | `_ainfer` / `_infer` call it first, so production behaves as specified. A private hook a test drives inside `open_invocation` takes it on first use. Either way the workspace is `_workspace_under(frame.ctx)`, so even a first read inside a child's call resolves the BTA's own ctx. A retried `_ainfer` reuses the record: one snapshot per invocation |
| X55 | More path sites read the record than §5.7 lists | §5.7 names `_get_result_path`, `_load_promoted_breakdown`, the node path lambdas, the graph setup, `_finalize_output`, `_finalize_response`, `_promote_child_checkpoints` | every in-call workspace read in §13 migrates too: `_resolve_worker_output_paths` / `_resolve_worker_output_path` (now takes the workspace), `_rebind_aggregator_workspace`, the graph status callback and topology emit output paths, worker binding and its log, the aggregator bind. They run inside node functions, where the same child-ctx hazard applies |
| X56 | `_promote_child_checkpoints(child, slot, *, parent_ws)` | §5.7: the base helper "joins the `_BTA_CALL` path sites" | the base helper cannot see a BTA component, so the caller passes the parent workspace explicitly; BTA passes its record's |
| X57 | The `use_async` flip on `self` goes in P6 c2, not P7 | §3: "`_BtaAttempt.use_async` (the flip on `self` goes with `_BtaGraph` in P7)" | the only readers of `self.use_async` were the node-function builders, which now read the attempt; `WorkGraph._arun` never reads it (only the sync `WorkGraph._run` does, to delegate). Keeping the flip would keep a definition write with no reader |
| X58 | Attempt plumbing | §5.7: "`_begin_attempt(attempt, inference_input)`"; definition-level callbacks use `invocation_of(self).require(_BTA_ATTEMPT)` | `_open_attempt(input, *, use_async)` puts the attempt and runs the hook (c3 discards the summary there); `_bta_attempt()` is the `require`. The registry factories call `_rebuild_subgraph()`. The JSON breakdown parse is pure: `_parse_json_breakdown` returns `(sub_queries, guidance)` and the breakdown records the guidance on its attempt; `_parse_json_subtasks` returns the queries only |
| X59 | `BtaCallSummary` field shapes | §5.7: `worker_child_names` and `worker_output_roots`, `worker_count` | the summary holds each worker's workspace root (`worker_workspace_roots`), from which finalize derives `outputs/` through `InferencerWorkspace` (one `has_deliverables` rule); `worker_count` is a property over `worker_child_names`, so it can't disagree with the names. The summary is built in the tail by `_run_summary` (before `_conclude_attempt`) and published after it by `_conclude_run`; it reads the aggregator slot as finalize did until P6 c6 makes the slot a factory again. `discard_result` (new in `invocation.py`) withdraws a published component and its queued compat fields; `_open_attempt` uses it |
| X60 | The getters stay true-no-ctx projections | §5.3: readers Dual `:785` / `:1438` and MFI `:1606-1630` via `_task_contract_at`; §9: the `_FakeMFI` / `_FakeDual` tests unchanged | the run-time readers use the typed channel: BTA/MFI harvest each worker's contract at its ctx (`_harvest_worker_contract`), Dual captures its proposer's at the propose ctx. The documented getters keep their shape for a parent in true no-ctx: MFI's walks winner-then-flows and Dual's falls back to its base, both through `_task_contract_at(child, None)`. `_task_contract_state_at` is the typed reader; a getter returning a non-`str` (a test double) counts as no contract |
| X61 | One key carries both BTA compat fields | §5.4: "`_worker_task_instructions` … the declared compat field of the BTA contract key" | `_BTA_SUMMARY` declares `_last_call_summary` and `_worker_task_instructions` (through `BtaCallSummary.selected_task_contract`); the relayed contract lives in the summary, so a second key would duplicate it. The summary is built after `_conclude_attempt`, because MFI's winner comes from the dispatch state that hook extracts |
| X62 | The aggregator's call ledger lands in P6 c6 | §11 P6 c5: "the attempt ledger (workers) and the call ledger (aggregator)" | until c6 `build_aggregator` writes its product into the slot (B2), so a call that closed it would leave a closed stage in the definition for the next call. c5 records only whether this attempt built it (`aggregator_owned`, for `_bind_rebuilt_child_ws`); c6 resolves it per call (`_BTA_AGGREGATOR`) and registers an owned one in the call ledger |
| X63 | Branch-scoped sync close in both SDK leaves; BTA `adisconnect` covers definition stages | §5.1 loop affinity; §5.7 "`adisconnect` stays idempotent and covers borrowed and definition children" | claude_code's sync bridge adopts codex's `_run_and_close`; both close only the active branch's client (`_adisconnect_branch`), since another branch's client may be bound to another loop (inventory §18 hazard). Every `adisconnect` was already idempotent (handles cleared). BTA's `adisconnect` now closes the breakdown, the aggregator slot and a static worker list's instances, no longer `_worker_instances` (owned workers close with their attempt) |
| X64 | Aggregator scope details | §5.7: `build_aggregator()` becomes `resolve_stage(...)` stored with `frame.get_or_create(_BTA_AGGREGATOR)`; the fan-out passes the resolved instance into `fresh_instance` | `get_or_create` needs a key factory without the owner, so `_aggregator_stage()` resolves on first need and puts the stage (or `None`) in the frame, registering an owned product in the call ledger; `build_aggregator()` returns its inferencer. Reads that may run before resolution or outside a call (`_iter_child_inferencers`, `_iter_child_slots`, the summary, `_finalize_response`, the worker-result formatter) use `_current_aggregator()`: the call's stage when resolved, else the slot. The fan-out resolves from the fanout's own constructed slot after `fresh_instance` and writes the instance back into that per-call object (D1 / X2), then the caller's ledger registers the fanout itself (D1: every `InferencerBase` stage of it is a per-call copy). `ResolvedStage` / `resolve_stage` live in `inferencer_base.py`, which the fan-out needs and BTA imports |
| X65 | Host parallel entries refuse overlapping items up front | §5.1: "a host `parallel_infer` / `aparallel_infer` on an uncertified class … raises, or runs per-item instances as P0 decided" | the guard alone would fire on whichever item happened to overlap, so the result would depend on timing. `_refuse_overlapping_items` raises `UncertifiedConcurrentUseError` before any item runs when a host ctx, more than one concurrent item and an uncertified class meet; one item at a time and bare / legacy calls run as before. Per-item `fresh_instance` copies were not adopted: a copy drops everything set after construction, a silent semantic change; there are no production callers (§8) |
| X66 | S1 resolved by serializing a session's turns | §8 S1: "the guard serializes or rejects. Which one is a product decision" | the guard rejects (it is never weakened); `ConversationService` serializes: `run_conversation_turn`, `resume_conversation_from_round` and `resume_conversation_from_widget` each hold the session's `asyncio.Lock`, so a new turn starts after the previous one — a cancelled turn still unwinding included — has finished, and two connections' turns queue instead of racing on the shared inferencers and the on-disk turn number |


**Fixed so far** (P7): B24 in every mode, with BTA and MFI certified (`cf8560c43ef5`), B31 (`0eec51eb2cbc`). Buck at `0eec51eb2cbc`: `test/agent_foundation/...` Pass 1716 / Fail 1 (m9) / Skip 12, Build failure 2 (metamate standalone, pre-existing).

## 34. P7 deviations

| # | Deviation | Plan text | Finding / decision |
|---|---|---|---|
| X67 | BTA and MFI are certified in P7 c1, not P9 | §11 P9: "BTA, MFI, Dual, LWI, PTI certified as their debt empties"; §11 P6 test: "two overlapping host calls on one BTA raise `UncertifiedConcurrentUseError` until BTA is certified (P9)" | `_BtaGraph` cleared the last measured BTA debt (the graph writes), and MFI's host-mode writes already go to its ctx node (`_dispatch_set`, `_reset_cross_flow_state`, the effective sub-queries), so both `KNOWN_DEBT` entries are empty. The ratchet holds certification and empty debt together in both directions, so it requires certifying them in the commit that empties the debt; leaving them uncertified would keep rejecting overlapping host calls that are now safe. B20 (MFI's no-ctx `predefined_sub_queries` write) stays in P9: it is bare-mode only and the guard is host-only. Overlapping host calls on one BTA now run (`test_bta_graph.py`); borrowed uncertified stages they share are still refused (`test_single_flight_guard.py`) |
| X68 | `_BtaGraph` copies the engine fields; it does not hold a reference to the BTA's graph state | §5.8: "a per-attempt `WorkGraph`" | the engine reads its configuration through the graph object (`checkpoint_mode`, `executor`, `max_concurrency`, the result-save flags, `_result_root_override`), so the attempt graph copies each one; per-attempt fields (`start_nodes`, `name`, `use_async`, the expansion limits, `subgraph_registry`) are set from the attempt; every `Debuggable` field keeps its default because `log` and `_get_result_path` forward to the owner, which keeps session-log records and checkpoint paths byte-identical (the BTA goldens). A test fails when `WorkGraph` gains an unclassified field |
| X69 | Every per-call reader of a BTA's name reads `bta_node_name`, including its log records | §5.9: "`_bta_prefix` and `_BtaGraph.name` read `_effective("bta_node_name")`"; "not passed" falls back to `self.name` | the written `name` had three more readers: the `bta_name` field of BTA's diagnostics, the `max_worker_query_chars` error, and `Debuggable.log`, which puts an instance's `name` in each record in place of the class-name `log_name`. All read the per-call name now, so session logs are unchanged. `Debuggable` gets a `_log_display_name()` hook (default: the `name` attribute), which BTA overrides. `_effective` falls back to the instance attribute of the same name, so BTA exposes a read-only `bta_node_name` property returning `name`. The ratchet gains a `bta_nested` fixture (an outer BTA borrowing a nested one), whose I2 check fails on the parent commit with the `name` write |

**Fixed so far** (P9): the workflow resume marker (`a2f86f7d6942`), Dual's iteration record (`229d87b9d879`), B7 (`3a7a34957481` call context; `d011fa9ae1a0` workspace), B15 with E10 (`7a1c4167a411`), E11 (`0bf2a6247dde`), the save-flag subphase with LWI, Dual, MFDual and PTI certified (`5f9573b83b46`), BTA's breakdown workspace (`e0d87b27de00`), B20 with E12 (`cb50055f7ccd`). A lint audit of every Python file the stack changed found two issues, fixed in `de9775ed451c`. Buck at `de9775ed451c`: `test/agent_foundation/...` Pass 1818 / Fail 1 (m9) / Skip 6, Build failure 2 (metamate standalone, pre-existing).

## 34. P9 deviations

| # | Deviation | Plan text | Finding / decision |
|---|---|---|---|
| X70 | The resume marker is strict | §13: "a frame component behind the same properties; direct-hook tests wrapped in `open_invocation`" | as specified: reading or writing `_step_was_previously_attempted` / `_previous_attempt_info` outside an invocation raises `NoInvocationError`. About 86 direct `_ainfer` call sites in 10 test files run inside `aopen_invocation`; reads after the call moved inside it. The same holds for Dual's `_last_iteration_record` and PTI's `_current_*` |
| X71 | The ratchet measures more before it certifies | §11 P9: "certified as their debt empties" | the fixtures did not exercise checkpointing, iteration changes or child workflows, so an empty debt would have been vacuous. Added: `dual_ckpt`, `mfdual_ckpt` (checkpointing on), `lwi_loop` (a loop that advances `iteration`), `pti_duals`, `lwi_dyn_dual`, `dual_lwi_base` (child workflows), `bta_shared_breakdown` (P9.8). Each fails on the parent of the commit that fixed what it measures |
| X72 | PTI's `_current_*` are one component | §13: "components" | the four attributes become one `_PtiCall` (`RuntimeKey` with a factory) behind properties of the same names (`_call_field`); the `init=False` attrs fields are removed, so construction is unchanged |
| X73 | One generic re-rooting primitive | §5.13: per-class host equivalents of the setter's three steps | `InferencerBase._set_call_workspace` records a workspace in the frame; `_workspace` returns it for that owner while the invocation runs. The setter's host equivalents: (1) a workspace-derived logger already follows `_workspace` per write (`_log_path_override`); (2) PTI's children are propagation-skipped, a static LWI's children already have workspaces from construction (iteration propagation was a no-op), a dynamic LWI passes explicit step workspaces; (3) `_DERIVED_FROM_WORKSPACE` is empty everywhere. PTI uses it only under a host ctx; a bare call keeps the setter, so `resume_workspace` stays the backing |
| X74 | LWI iterations re-root the invocation in every mode | §5.13: "Legacy and no-ctx modes keep the setter"; "LWI iteration paths are computed from the call-start workspace held in a frame component" | leaving the last iteration in the backing is B15 itself (the bare P0 probe `test_b15_reused_lwi_keeps_its_workspace`), so the iteration re-rooting is frame-scoped in all modes. The workflow's own checkpoints — loop checkpoint, step results, `final_result.json` — stay at the call root (`_checkpoint_workspace`), as PTI's `_get_result_path` already did: scattered across iteration directories, a resume read the root's iteration-1 checkpoint, and once the backing stopped leaking a repeated bare call lost its cached final result. `_end_iterations` returns the invocation to its root after the steps |
| X75 | Checkpoint settings: own decision, then handed, then configured | §13: "the PTI/LWI child save/resume flags and `_result_root_override`" as an atomic subphase | `enable_result_save`, `resume_with_saved_results`, `max_expansion_events` and `_result_root_override` are LWI properties over the configured `__dict__` values. While the instance runs: its own per-call decision (`_set_call_policy`, frame) wins, then the settings the nearest enclosing host invocation handed it (`_CHILD_POLICIES`, through the RPU `_configure_child_workflow` hook), then the configured value — the precedence the writes produced. Only LWI-family children are handed settings; other workflows and non-host modes keep the writes; `checkpoint_mode` is written only when it differs. Own and child settings land in one commit because certification needs both |
| X76 | Child debt blocks certification | §6: "certified when its debt is empty" | the ratchet counts the `CHILD_DEBT` of the fixtures a class owns as that class's debt, so a class that writes into its children cannot be certified |
| X77 | BTA's breakdown is bound at dispatch | §5.13: "under a host ctx it is published to the `breakdown` child ctx instead" | under a host ctx `_configure_for_workspace` leaves the breakdown alone and `_bind_breakdown_workspace` publishes `<call workspace>/children/breakdown` before every dispatch — the location the setter gave it |
| X78 | MFI's runtime inputs have one channel | §13: "`_BtaAttempt.effective_sub_queries`" | the ctx-node publication goes too: BTA reads the attempt's sub-queries, MFI's followups read `_flow_input`. `MultiFlowState.effective_sub_queries` / `flow_inputs` are no longer written and stay for decoding earlier stores |

**Fixed so far** (P10): B28 — the terminal transport result (`541d9dc23636`), the CLI streams' final metadata (`900b4f5bcc17`), tool_as's response (`0b4f385ab63a`), the stream counters (`b69f90b0a64c`), devmate's dump file (`fb5a6d80d121`); B6 leaves — the six streamed-session writers and the metamate / rovochat conversation ids (`199c16d5dce8`), the four SDK leaves' session reads and rovochat's event-stream close (`d61e79b56558`); B33 (`f191de3f4d99`); B19 (`3a469ac2f7c7`); B23 with rovodev's `final_output`, Conversational's `_final_output_at` and the §35 `output_file` clobber (`e6b7b8a40b04`); the leaf ratchet fixtures with 14 leaves certified (`e7490d256f18`); outside the register, E13 (`5cbc7ce2e189`). Buck at `5cbc7ce2e189`: `test/agent_foundation/...` Pass 2076 / Fail 1 (m9) / Skip 6, Build failure 2 (metamate standalone, pre-existing).

## 34. P10 deviations

| # | Deviation | Plan text | Finding / decision |
|---|---|---|---|
| X79 | TSIB's shadowed `_ainfer_streaming` is deleted, not migrated | §13: the terminal transports publish `_TERMINAL_RESULT` | its first definition was shadowed by the second in the same class body and never ran (§23 F1.1), so P10.1 deletes it; the live transports publish |
| X80 | claude / codex CLI async streams still report no stdout or return code | §13: "the transports … call `publish_result(self, _TERMINAL_RESULT, …)`" | they publish their stderr; stdout and the return code keep the defaults, as before (§23 F1.2), because `_ainfer` parses the stream it accumulated itself. Reporting them would change `get_streaming_result` after a bare call; not a per-call-state defect |
| X81 | devmate's `_output_file` is dual-mode | §13: a component | a property over the invocation component inside a call and the instance attribute outside one, so the hook-level unit tests that drive `construct_command` / `parse_output` directly keep working; a public call never touches the instance |
| X82 | The conversation ids back themselves | §13: "metamate and rovochat conversation ids become Tier-3" | they are `LiveHandleField`s whose backing is their own `__dict__` entry: the session policy (branch slot under a host ctx, instance otherwise) rather than raw Tier-3 handles, so `reset_session` and the tests that preset them keep their names. `_session_scoped_get` / `_set` read and write backings through `__dict__` |
| X83 | tool_as tracks every running process | B28: tool_as `_last_response` | `_proc` (the last process started) becomes the set `_procs`, which each call adds to and removes from, so `cancel()` terminates every subprocess the instance runs; empty between calls |
| X84 | OpenClaw's initialized sessions are per connection branch | §13 (B19): "Tier-3 set keyed by `session_id`" | as specified: a host branch records its own set, so a sibling branch probes the transcript once more (and warms up only a new session); an explicit `initialize_session()` outside any call writes the shared backing, which a branch reads until it records its own |
| X85 | rovodev: the §35 clobber and the sync entry | §35: "P10 owns rovodev output handling" | the stream no longer clears the clean output when the user configured `output_file`, so `get_final_output()` and the published `final_output` hold that file; `_ainfer` still rewraps only for the temp file (`_uses_auto_output_file`; a configured file outlives the call and the base result already reads it). A sync `infer()` resets the output component and records none: its result is already the clean output |
| X86 | The typed channel is read under every ctx | §5.3: `_final_output_at` reads "the typed outcome under a ctx, the getter only in true no-ctx" | as specified, so a base inferencer that sets `streams_differ_from_final_output` must publish `final_output` (only rovodev does). Conversational's public entries always install a ctx (a legacy root for a bare call), so its getter fallback is reached only when `_run_agentic_loop_impl` runs with no ctx |
| X87 | Four SDK leaves also read the session backing | §13 names the six writers | found by the leaf ratchet's probes: codex SDK's connect-time auto-resume, devmate SDK's auto-resume, and the `SDKInferencerResponse.session_id` of devmate SDK, metamate SDK and rovochat read `_session_id`; under a host ctx a branch's next call started over and the response reported no session. They read `active_session_id` (P10.10). RovoChat's unclosed NDJSON stream, finalized at loop shutdown outside the call, is closed by ownership in the same commit |
| X88 | Leaf certification is measured in three modes, over fake transports | §11 P10: "as each family's debt reaches zero, set `_HOST_PURE_CERTIFIED = True`" | the ratchet measures 14 real leaves through `_leaf_fixtures` (fake CLIs as real subprocesses, the scripted gateway, fake SDK clients, `httpx.MockTransport`), in sync, async and a drained `ainfer_streaming`; `test_every_leaf_fixture_reaches_its_transport` keeps a certification from being vacuous (under buck it caught the kiro and metamate CLI fakes failing before their shebang was replaced by a `sh` wrapper). A setattr trace of every leaf during host calls showed only `LiveHandleField` / Tier-3 property writes. Not measured and so not certified: the API inferencers (`api_inferencers/*`, bedrock), `FunctionInferencer`, `HttpRequestInferencer`, `RemoteInferencerBase` (outside the plan's leaf families), Conversational (plan O2) and the abstract terminal bases; the guard keeps refusing overlapping host use of them |
| X89 | The P10 buck gate found two commits red under buck only | §2 critique (g): intermediate commits green | P10.4's strict `_reset_stream_stats` broke the buck-only `test_code_scope_judge.py` (direct `_ainfer`), and P10.10's `_leaf_fixtures.py` broke `run_context`'s `python_pytest` (sources must be test modules) and, as a library, its type-checking target (an `Optional` regex match). The fixes were amended into the commit that introduced each break: the test runs `_ainfer` inside `aopen_invocation`; `_leaf_fixtures` is a `python_library` that imports no test module, holding the fakes it shares with the goldens. The same gate surfaced E13 |
| X90 | OpenClaw fan-out stays rejected | §35: "decide before P10" (X23) | re-checked after B19: the session id is configuration (`session_id`, default `main`), not per branch, so every fan-out worker would talk to the same gateway session; `_SUPPORTS_BTA_FANOUT` stays `False` |

Observations, not changed in P10 (outside its scope):
- Tier-3 connections live per host branch until `adisconnect`: an instance used under many independent host roots keeps one SDK client / Codex thread / serve process per root (M6 design; the host owns the lifecycle). `test_leaf_connections.py` pins that `adisconnect` closes them all.
- A no-ctx read of a session-scoped value with no backing returns the one live branch value (`_session_scoped_get`, B6(b) policy), so a bare read after a single host call shows that branch's session.
- The SDK leaves' `_connect_lock` is created lazily and reused across event loops.
- claude CLI's streaming entries do not adopt the streamed session (only `ainfer`'s post-hook does), as the P0 goldens pin.

**Fixed so far** (P11): the certification audit and the ratchet's stale-allowance check (`0034a6b81815`), BTA no longer a `WorkGraph` (`fc563e8149d2`), the source checks (`7f2051aaedc1`), the run_context README (`faf2b618793d`). Buck at `faf2b618793d`: `test/agent_foundation/...` Pass 2106 / Fail 1 (m9) / Skip 6, Build failure 2 (metamate standalone, pre-existing).

## 34. P11 deviations

| # | Deviation | Plan text | Finding / decision |
|---|---|---|---|
| X91 | The audit also shrinks `ALLOWED` | §11 P11 item 3: "shrink `KNOWN_DEBT` to empty or documented stop-point rows" | `KNOWN_DEBT` and `CHILD_DEBT` were already empty. Six `ALLOWED` names were never written by a host call any more (`_init_recipe`, set in `__new__`; BTA's `output_path`, derived at construction; MFI's four `_last_*` dispatch mirrors, whose backings are written only outside a host ctx). They are removed, and a stale allowance now fails like stale debt. Every §8 host-concurrent-use row stays resolved (P6 c7); with the leaves certified, a static `worker_inferencers` list may now overlap a shared certified worker |
| X92 | Engine settings the BTA never configured stay at `WorkGraph`'s defaults | §11 P11 item 2: "Redeclare only `name`, `max_concurrency` and `group_max_concurrency`" | as specified. The attempt graph copies what the BTA configures (result-save / resume flags, checkpoint mode, result root override, the two concurrency limits); seven engine settings no caller sets on a BTA form `_BTA_GRAPH_ENGINE_DEFAULTS` and keep the values the BTA held. The expansion-reconstruction registry is built per attempt; sync `_infer` opens its attempt with `use_async=False`; `name` and `group_max_concurrency` are keyword-only. The `start_nodes` override and the `run` / `arun` blockers go (shims, item 3) |
| X93 | Shipped configs are checked for keys the base took | — | the config loader drops a key its target does not accept with only a warning, so a YAML relying on a `WorkGraph`-only argument would lose it silently; `test_no_shipped_config_passes_a_constructor_argument_the_graph_base_took` scans every BTA / MultiFlow node in AgentFoundation and OpenStartup YAML. The same scan surfaced E14 |
| X94 | The source check is static where the ratchet is dynamic | §11 P11 item 4 | `test_runtime_source_checks.py` scans the sources: declared compat fields are never written by code (only the compat flush, through `setattr`) and read only by the documented getters and their no-invocation projections (`_stream_result`, rovodev `_call_output`); registered state fields are JSON-shaped, with ten `Any` fields listed with why. Host-call writes outside `ALLOWED` and live objects in a host store stay measured by the ratchet (I1–I3); `MultiFlowAttemptState.judgments` gets its precise type |
| X95 | One README "still open" item stays open | §11 P11 item 5: "close its 'still open' items" | five are done (write-purity, Tier-3 for the leaves, `_iter_child_slots`, HITL checkpoints, the §9.2 layer-2 forwarder). Conversation-resume rehydrate is wired but its regression test is the pre-existing m9 failure, Conversational's own follow-up (O2) |

**Fixed so far** (P8): the RPU `FileLock` (`b1d55bba2605`), the checkpoint-root lease (`ba935c3c9ecf`), canonical resume identities (`87c60726e54a`), the resume manifest and the committed effective plan (`519cbffdc756`) — B35, B37, E15. Buck at `519cbffdc756`: `test/agent_foundation/...` Pass 2155 / Fail 1 (m9) / Skip 0 (the B35 / B37 probes pass), Build failure 2 (metamate standalone, pre-existing).

## 34. P8 deviations

| # | Deviation | Plan text | Finding / decision |
|---|---|---|---|
| X96 | The retry archive keeps call-scoped files at the root | §5.11: the lease spans the whole logical call | U3c's archive moved everything in `checkpoints/` into `.attempts/<n>/`, the held lock file included, which freed the root mid-call. `_retry_archive_keeps()` names the entries that belong to the call; BTA keeps the lease and the manifest. A refused call (`BtaWorkspaceBusyError`, never retried) archives nothing. The lease also refuses `/`, `$HOME` and repository roots |
| X97 | A module-level function identifies as `module:qualname` | §5.12: "an opaque factory or callable: a stable identity the factory supplies (`resume_identity`); none → error" | its import path is the stable identity it supplies; lambdas, closures and bound methods still have none, and mocks are rejected |
| X98 | A fresh run whose identity can't be computed still runs | §5.12: "fails before any I/O with that error; it can still run fresh with `enable_result_save` off" | failing those calls would break every fresh run with a lambda worker factory or a closure stage. The header records why (`unverifiable`), a WARNING says so when resume is enabled, and any resume of the tree fails closed (`ResumeIdentityUnavailableError`) unless `trust_legacy` |
| X99 | `trust_legacy` also covers unverifiable headers and unreplayable plans | §5.12: the override "resumes a legacy workspace" | the same emergency override for every tree the manifest can't vouch for, logged on every use |
| X100 | The committed plan holds the sub-queries and the worker node names | §5.12 item 4: per worker, "its query, arguments, task type, node name and stage identity" | dispatch and todo expansion are functions of the sub-queries (queries with their arguments, after truncation and selection) and the definition, which the header verifies; reconstruction re-runs `_build_subgraph_spec` on them and must reproduce the committed worker names (else `BtaResumeCorruptionError`). Sub-queries that change in a JSON round trip are recorded `unavailable` |
| X101 | A legacy tree is rebuilt from what its breakdown node saved | §5.12 does not say where `trust_legacy` rebuilds from | the breakdown node's saved result (the list after truncation and selection), else the promoted breakdown (before them), else the predefined sub-queries capped like a fresh run. Before P8 only the promoted breakdown was read (B35), or `None` (B37). `certify_resume_manifest` rebuilds the same way and keeps a replayable plan already committed, logging a replaced header (e.g. after an upgrade changed the definition's identity) |
| X102 | A call that doesn't resume never vouches for an earlier call's tree | §5.12's policy table covers resuming calls | a call with `resume_with_saved_results` off on a root holding checkpoints overwrites only what it reaches, so it removes the manifest and writes none (its plan stays in memory for its own rebuilds); a later resume fails closed like a legacy tree. On a clean root it writes the manifest as usual (E15) |
| X103 | Goldens changed beyond the manifest and the lock file, and record digests as placeholders | §5.12: "the I7 compatibility goldens allow exactly two new files" | the pinned B35 (`resume_partial_*`) and B37 (`resume_without_promoted_breakdown*`, the sync `finalize_*` retries) defects are fixed here, as their docstrings say; the legacy fixture fails closed and resumes under `trust_legacy`. SHA-256 digests normalize to numbered `<SHA…>` placeholders: the definition digest covers qualified names, which differ between buck and pytest module paths |
