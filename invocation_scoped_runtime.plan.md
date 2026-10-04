# Plan v8 (integrated): inferencers as definition-stable templates, per-call state owned by the invocation

**Status.**
- **v8 (2026-09-28) amends v7 in place.** Three reviewer reports (two against v6, one against v7) were checked item by item against the live tree. Only confirmed defects changed the plan. §2.3 lists each change with its root cause and file:line evidence. §2.3 also lists feedback that was outdated or rejected, with reasons.
- v7 replaced v6 in this file. It integrates five plans at their current revisions (re-diffed 2026-09-27 16:10):
  - Sep, Dec and Harm are byte-identical to the revisions v6 read.
  - **Idem was rewritten (16:04) as "Required Amendments to Snappy v6":** a review of v6 that keeps it as the chosen plan and lists nine gaps plus a resume specification. Each claim was checked in source; §2.4 records what was adopted, refined or rejected.
- On 2026-09-27, every anchor in v6 was re-verified in the live tree by independent source passes (entries/streaming, BTA/stages, run-context infrastructure, entry order and iterators, BTA workspace/rerun/cleanup, Tier-3 handle keying), and the design was put through an adversarial critique. §2.4 lists what changed.
- No code has changed. The local fan-out stack `5b2ec41573aa…69ea2d3762af` (plan v4.1) is untouched.
- The fan-out e2e gates **G1–G6 stay paused** until the user explicitly says GO.

**Sources (provenance tags):**

| Tag | Plan (revision read) |
|---|---|
| [Snappy] | this file, v5/v6 (mine) |
| [Sep] | `fbcode/_tony_dev/CoreProjects/AgentFoundation/inferencer_template_runtime_separation.plan.md`, "Integrated Plan v2" (13:09) |
| [Dec] | `~/.llms/plans/inferencer_template_runtime_decoupling.plan.md`, "Plan v2 (integrated)" (13:35, 705 lines; it was being edited by another session when first read, and was unchanged at the 16:10 re-diff) |
| [Idem] | `~/.llms/plans/idempotent-enchanting-lake.md`, "Required Amendments to Snappy v6" (v4, 16:04; supersedes its "Canonical Plan" v3) |
| [Harm] | `~/.claude/plans/here-is-some-context-harmonic-liskov.md`, v4 (13:13) |
| [new] | found while writing v6, v7 or v8 |
| [R1] / [R2] / [R3] | the three reviewer reports v8 answers (R1 and R3 against v6, R2 against v7) |

**Paths.** Line numbers are at `69ea2d3762af`.

| Short | Full path |
|---|---|
| `AF/` | `fbcode/_tony_dev/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/` |
| `RC/` | `AF/run_context/` |
| `FLOW/` | `AF/agentic_inferencers/flow_inferencers/` |
| `EXT/` | `AF/agentic_inferencers/external/` |
| `BTA.py` | `FLOW/breakdown_then_aggregate_inferencer.py` |
| `RPU/` | `fbcode/_tony_dev/CoreProjects/RichPythonUtils/src/rich_python_utils/` |
| `TEST/` | `fbcode/_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/` |

---

## 0. TL;DR

1. **The goal is right, and it is the codebase's own stated direction.**
   - `RC/README.md:41` lists "Still open: full write-purity".
   - M1–M7 already moved role, workspace, sessions, sub-queries and typed call state into the run context.
   - BTA, PTI/LWI, the templated snapshot, the leaves and `switch_role` are what's left.
2. **The precise target is definition stability [Sep]:** under a host-owned `RunContext`, a call writes no call-specific state into its own definition or into any borrowed child definition. Bare calls keep their documented getters.
3. **Placement rule: the narrowest scope wins.** There are five homes (definition, invocation, node state, live handles, workspace) plus the Tier-2 transport (§4.2).
4. **Mechanism: explicit typed components on an `InvocationFrame`** (`RuntimeKey`, `get` / `require` / `get_or_create` / `put`). There are no descriptors for call state and no write-through [Sep/Dec]. `frame_for(owner)` walks the frame ancestry by identity.
   - A key that backs a documented bare getter declares it (`RuntimeKey.compat`). One `publish_result` call then feeds both the component and, in bare modes only, the getter field, so the two can never drift (§5.4).
   - Per-call values that a parent assigns to a stage (stream observer, interactive, a nested BTA's node name) travel as **declared invocation keywords into the stage's own frame** (`_INVOCATION_KEYWORDS`). They are never written onto the stage (§5.9, B18/B31).
5. **Every public entry is a new invocation, runs one fixed seam, and public overrides are thin** (validate or adapt arguments, then delegate). Behaviour lives in private pipeline overrides: `_ainfer_streaming_pipeline` (async) and `_infer_streaming_pipeline` (sync).
   - **The seam [Idem, refined]** runs in this order for every entry (ordinary, iterator item, parallel item, CLI, streaming):
     1. bind the ctx;
     2. open the frame, claim the path, register the single-flight guard;
     3. clear the node's outcome;
     4. pop invocation keywords;
     5. `_init_call_state`;
     6. provider pre-hook `_prepare_call`;
     7. the pipeline (retries, finalize);
     8. provider post-hook `_conclude_call`;
     9. close the ledger;
     10. publish the outcome (success only), then flush compat fields (every close);
     11. release;
     12. unbind.
   - The CLI leaves' pre-call session policy and **all** of their post-call code become the two hooks. That post-call code covers session writes, result wrapping, and devmate's failure promotion and error counting. So a rejected call touches nothing, and a failure devmate promotes publishes no outcome (B34, §5.1).
   - This template-method split [new] fixes the private-pipeline split in Sep/Idem/Dec.
   - Their split would make `ainfer()` skip the session, output-file and filter logic that rovodev and devmate keep in their public overrides.
   - It would also leave the streaming leaves (rovodev, devmate, OpenClaw, and claude_code's sync entry) ignoring an explicit `run_context` on direct streaming (B26). See §3, row 2.
   - OpenClaw's `ainfer`, the one CLI entry that skips `_ainfer_single`, converges onto it (B30).
   - Streaming entries, and the lazy `infer(iterator)`, bind their ContextVars only while the pipeline runs, never across yields (B29). This is what makes the path claim sound for abandoned streams and iterators.
   - The public streaming templates are therefore **plain methods that return a framed generator**. The wrapper resolves the ctx once, at first resumption (`enter_run`'s rule: explicit, then active, then mint). It binds the ctx and frame around each resumption and runs the fan-out branch inside the frame. No entry-level `enter_run` runs in a generator body.
   - The sync bridge owns its event loop and task explicitly, so closing a sync stream cancels and joins its transport.
   - A private hook called on *another* object (Dual's preview `_render_prompt` on its review and fixer leaves) runs outside that object's invocation. Under a host ctx it may only read (§4.4).
6. **Concurrency.**
   - **A strict `(store, path)` claim:** any overlapping public invocation at one path raises before any I/O. Nested work always runs at a child path; this was verified for fan-out, guardrail, fallback and agentic functions. Private calls stay in the current frame.
   - **A single-flight guard (P6):** in host mode, an instance whose class is not certified host-pure runs at most one invocation at a time in the process, whatever its path or stage role. Certified classes may be shared.
   - **A workspace lease:** rejects two writers of one BTA checkpoint root, across threads and processes.
7. **Results read after the call use the typed `NodeRunState.outcome = NodeOutcomeState(task_contract, summary, final_output, invocation_id, cleanup_errors)`.**
   - Each node's outcome is cleared when its invocation opens, and published once, at that invocation's successful end. A stream succeeds only when it is exhausted without an exception.
   - So a failed call can never expose an earlier call's result.
   - Parents read their direct child, at the exact child ctx. This includes Conversational's read of a leaf's final output (`_final_output_at`).
8. **Bare compatibility is declared on the keys** (`RuntimeKey.compat`). Whenever a legacy or no-ctx invocation closes, success or not, the documented getter fields it published are flushed. That matches today, where the transports write those fields during the call. Host mode never touches them.
9. **BTA:**
   - an explicit attempt loop, one local `_BtaAttempt` per attempt. It replaces the recursive interactive rerun;
   - a workspace and checkpoint root frozen once per invocation, which every path derivation reads (the child-checkpoint promotion included);
   - a typed, frozen `BtaCallSummary`, put as the run's last step (after `_conclude_attempt`, the subclass post-processing hook). When the result came from an external fallback or `default_return_or_raise` instead, there is no summary and finalize takes the base path (B36);
   - `_build_subgraph_spec`'s output as the one effective-plan commit point, which resume rebuilds from (B35, P8);
   - `ResolvedStage` owned vs borrowed for **all three stage kinds** (breakdown, workers, aggregator). Owned stages are closed at attempt or call end, sync included;
   - duck-typed (non-`InferencerBase`) stages keep a documented attribute protocol and are never certified;
   - a pure `build_aggregator`;
   - a per-attempt `_BtaGraph` composition that copies every `WorkGraph` field (coverage-tested), with log and result-path forwarding;
   - staged `WorkGraph` base removal.
10. **36 numbered defects** (§8). 35 are confirmed in source; B15 is a hypothesis that P0 confirms or drops (B16b is now confirmed). Each has one commit and a regression test that reproduces it. There are no interim band-aid resets. B26–B36 are found only here (B32 and B33 first raised by [Idem], B35 and B36 found while checking [R1]):
    - B26: explicit `run_context` ignored by direct streaming;
    - B27: devmate's positional `filter_session_info` binding;
    - B28: leaf call results (terminal stdout/return code, stream results, tool responses, token counts) handed from the transport to `_ainfer` through instance fields;
    - B29: streaming entries leak `_active_ctx` into the consumer across yields, and abandoned streams reset it in the wrong Context and orphan the sync bridge thread;
    - B30: OpenClaw's `ainfer` bypasses `_ainfer_single`, so it would run frameless and unclaimed;
    - B31: BTA writes a per-call node name onto nested-BTA workers, including borrowed ones, so two workers sharing one instance share one graph name and result path;
    - B32: live handles are keyed by path alone, so two independent host roots (both `"/"`) share one leaf's session and client;
    - B33: devmate's consecutive-error counter is per instance, so one branch's failures reset another branch's session;
    - B34: per-call initialization (`_init_call_state`) runs only in `infer` / `ainfer`, and outside any claim. `parallel_infer` items, CLI entries and direct streaming skip it, and `infer(iterator)` runs it once, on the iterator object;
    - B35: BTA resume rebuilds workers from the pre-selection promoted breakdown, so interactive selection and truncation are lost;
    - B36: when every attempt raised and an external fallback or a configured `default_return_or_raise` produced BTA's result, `_finalize_output` still finalizes the failed attempt's aggregator output, or raises U3b and loses the result.
11. **Phases P0–P11, plus O1/O2** (§11).
    - Each commit is green when it lands. Rollback reverts from the stack tip in reverse dependency order; a commit with no dependents reverts alone.
    - Stop points: P7 and P9. §11 lists what stopping after each phase buys.
    - P8 (lease and resume manifest) needs its own explicit approval to run. The plan is **complete only after P8**; without it, only partial completion can be reported, with I4 and I11 still open.
12. **If only one plan can be picked: this v8** (§15; 42 of 42 rows ✅). Of the four other plans as they stand, the best is **Dec v2** (18 ✅), then Sep v2. Idem v4 is not a standalone plan: it is a set of amendments to v6, and v7 adopted most of them.

---

## 1. Context

**Why.** Inferencer instances are meant to be **templates**:
- built once from config;
- shared by concurrent branches;
- re-roled by orchestrators (MFDual's fixer and reviewer);
- cloned per call (`fresh_instance`, the BTA fan-out).

State that belongs to one call still lives on `self`, which causes three classes of defect:

| Defect | Example |
|---|---|
| **Concurrency** | two branches running one instance overwrite each other (the task-contract snapshot, BTA topology and graph) |
| **Staleness** | a reused instance carries its previous call's values (B1, B3, B5, B6, B23, B25) |
| **Opacity** | a post-call reader can't tell which call a value belongs to (B10, B13) |

**What prompted it.**
- While answering the template-vs-run-state questions, I found live bugs (B2 is in my own fan-out stack).
- Four other plans proposed overlapping designs. The user asked for one integrated, non-ad-hoc plan, and for a single-plan pick with a side-by-side comparison.

**Intended outcome:**
- Under a host ctx, no call mutates any inferencer instance outside the audited ALLOWED set (§4.7). A measured ratchet enforces this.
- Bare calls keep documented behaviour. Every intentional change is a numbered bug fix.
- Checkpoint paths, node names, graph-event order, the e2e `n_subtasks` metric, and resume of new workspaces are unchanged.

---

## 2. Honest assessment

### 2.1 Three corrections to "move run state into the context"

All five plans agree on these.

| Possible assumption | Reality | Consequence |
|---|---|---|
| "Split Template and Runner classes" | Hooks such as `_render_prompt`, `_finalize_output` and the `_ainfer` overrides have fixed signatures across about 30 subclasses. | Keep the classes; move state into the right home. |
| "Put per-call state in the ctx node" | Nodes are keyed by path. Dual's fixer re-enters one `…/fix` node every round, and concurrent calls from one parent path share a node. | The per-call home is the **invocation**. The node holds typed facts needed after the call. |
| "Resume needs instance state" | Resume reads promoted checkpoints, WorkGraph results and `NodeRunState` from disk. The one case that looked like it (BTA guidance, `BTA.py:1390-1398`) is bug B3. | Durable facts go to node state or workspace files. |

### 2.2 Value

- Production exposure today is low: research_propose builds a fresh BTA per fan-out call, and nothing overlaps calls on one BTA.
- The value is design correctness, 36 latent defects, and completing the `fresh_instance` contract ("the recipe is the definition").

### 2.3 What v8 corrects in v7

Each row gives the v7 defect and its root cause, then the fix. Every row was verified in source at `69ea2d3762af`. Letters are stable IDs for the review verdicts.

| # | v7 defect → root cause | v8 fix | Evidence |
|---|---|---|---|
| a | **Streaming entries still leak their ctx [R1#1].** Step 1 kept `enter_run` at the public entry, and the streaming templates are async generators. So the entry itself sets `_active_ctx` in the consumer's context across yields, outside whatever `framed_agen` binds. The fan-out branch also ran outside the frame, so its ledger registration had no frame. | The public `ainfer_streaming` / `infer_streaming` become plain methods that return `framed_agen` / `framed_gen`. The wrapper resolves the ctx once, at first resumption, with `enter_run`'s rule, and holds it locally. It binds `_active_ctx` and `_current_invocation` around each resumption, and runs the fan-out branch inside the frame. A minted root needs no teardown. P2 lands the plain-method shape, returning today's generator; P3 swaps in the wrapper (§5.1, §5.2). | `AF/streaming_inferencer_base.py:751-786` (`_rc_token = enter_run(...)` in the generator body; `finally: exit_run`); `RC/bridge.py:50-69` (explicit → active → mint), `:72-74` (`exit_run` is only `_active_ctx.reset(token)`) |
| b | **Finalize after a result no BTA run produced [R1#4, the valid part].** v7 left `_finalize_output` without a summary unspecified. The retry helper can return a result that no BTA run produced, in two cases, both after the last attempt raised: an external `fallback_inferencer` (orchestrator recovery re-raises, so the chain moves on), or a configured non-exception `default_return_or_raise`. Today BTA's `_finalize_output` ignores `response` and symlinks whatever aggregator workspace `_read_child_workspace` resolves. So it links the failed attempt's aggregator output as the canonical output, or raises the U3b `RuntimeError` and loses the real result (B36). R1's guardrail scenario can't happen: guardrails are leaf-only (not-adopted table). [new] A summary put at the end of BTA's own `_ainfer` would still not be the run's last step for MFI, whose post-processing runs after BTA's `_ainfer` returns and can raise. | `_BTA_SUMMARY` is discarded when a run starts and put as its **last step**: after `_emit_graph_reconcile`, `_finalize_response` and a new `_conclude_attempt(result)` hook, with nothing after it that can raise or await. An orchestrator is never judged, and the helper stops at the first run that returns, so the summary is present iff the call's result came from a BTA run, and it belongs to that run. With a summary, `_finalize_output` keeps today's branches, reading the summary. Without one, it finalizes `response` through the base path. MFI moves its post-processing (`_normalize_aggregator_output`, `_extract_dispatch_state`, `_maybe_strip_response`) into its `_conclude_attempt` override. Its `_ainfer` override is deleted, and its `_infer` keeps only the coordination guard (§5.7). No base acceptance hook is needed. | leaf-only guardrail `AF/inferencer_base.py:1486-1499`; judge skipped on orchestrators `:4297-4302`, `:4408-4413`; orchestrator `_ainfer_recovery` re-raises `:4200-4201`; chain `[_recovery_wrapper] + external_wrappers` `:5117`; non-exception default returned at exhaustion `RPU/common_utils/async_utils.py:203-208`, `RPU/common_utils/function_helper.py:574-584`; finalize after the helper `:5205-5207`; `BTA.py:1824-1886` (U3b `:1841-1850`); run tail `:2352-2367`; MFI post-processing `FLOW/multi_flow_inferencer.py:1892-1899`, `:1915-1920` |
| c | **The effective plan had no commit point, and resume ignored it [R1#5, #10].** The manifest committed an "effective plan", but reconstruction still went through the registry lambdas, which rebuild from the promoted breakdown. Promotion runs before selection, so resume loses interactive selection and truncation today (B35). | `_build_subgraph_spec`'s output is the effective plan: it is the single function that decides the worker list. The plan is committed there, before any worker result persists. The P8 registry lambdas rebuild from the committed plan. §5.12 adds an explicit resume state table. | registry lambdas `BTA.py:942-955` call `_build_subgraph_spec(self._load_promoted_breakdown()[0], …)`; promotion `:3189-3195` precedes `_select_sub_queries` (`:3211+`); `_build_subgraph_spec` `:2369+` (dispatch `:2398-2418`, todo expansion `:2419-2430`, nodes `:2845-2875`) |
| d | **Aggregator traversal and fan-out ownership [R1#6].** (1) After B2 the slot may hold a factory, but `_iter_child_slots` / `_iter_child_inferencers` still yield the slot. So `pre_retry` stops archiving the call-scoped aggregator between retries; today the slot write makes it do so. (2) v7 registered "the whole per-call fanout" in the caller's ledger. But BTA `adisconnect` closes every child, borrowed ones included, so it would close a prototype's configured aggregator passed in by instance. | (1) Both traversals yield `frame_for(self).get(_BTA_AGGREGATOR).inferencer` when present, else the slot. (2) The caller's ledger closes exactly what the fan-out call created: the resolved aggregator when owned. The fanout's owned stages close in its own invocation. The fanout itself is registered only if P0 shows that `fresh_instance` copies instance-valued recipe entries and overrides instead of sharing them (unverified today). | `pre_retry` `AF/inferencer_base.py:1855-1900`, `_archive_workspace_for_retry` `:1902-1960`; `BTA.py:2064-2087`; `adisconnect` `:1003-1017`; `fresh_instance` `:1462-1483` |
| e | **`use_async` has more readers than the graph [R1#7, R2#1].** v7 moved only the async path's `use_async` flip onto `_BtaGraph`. The node-function builders also read `self.use_async` to choose sync or async functions, so without the flip they would build sync functions for the async graph. | `_BtaAttempt.use_async` (`True` on the async path; the definition's value on sync), read by the builders and by `_BtaGraph` (§5.8). | flip `BTA.py:2259-2260`, `:2309`; readers `:2446` (`_build_subgraph_spec` → worker fn `:2847`, aggregator factory `:3038`), `:3098-3100` (`_make_breakdown_fn`); `WorkGraph.use_async` defaults to `False` (`RPU/common_objects/workflow/workgraph.py:1986`) |
| f | **Post-call code outside the frame; Conversational reads a last-call getter [R1#8, R2#4].** (1) v7 moved only devmate's promotion and rovodev's `find_latest_session_id` into `_conclude_call`. The claude_code, codex and kiro session writes and rovodev's result wrap also run after `_(a)infer_single` returns, so after release and publication. They race the next same-path call, and they read `_last_stream_result`, which P10 makes a frame component. (2) Conversational calls the leaf's `get_final_output()` after streaming. After P10 that reads host-silent compat fields. | (1) All post-call code moves into `_conclude_call` (P3 commit 6). (2) New `NodeOutcomeState.final_output`, published by leaves whose `streams_differ_from_final_output` is true. Conversational reads `_final_output_at(child, child_ctx)`, which mirrors `_task_contract_at`; only true no-ctx calls the getter. It lands with the P10 rovodev commit. A stream succeeds only when it is exhausted without an exception (§5.3). | claude_code CLI `:883-899` / `:950-957`; codex CLI `:653-670` / `:701-707`; rovodev `:879-893`; devmate `:1264-1333`; kiro `:331-340`; `conversational_inferencer.py:909-913`, `:938-941`; rovodev `streams_differ_from_final_output = True` (`:135`), `get_final_output` (`:641-663`) |
| g | **Intermediate commits were not green [R1#9].** (1) P2 commit 1 switched base `_ainfer` to the private pipeline while rovodev and devmate still kept their logic in public overrides, so `ainfer()` skipped it until commits 3–4. (2) P6 commit 2 deleted `_worker_instances` while finalize, `adisconnect` and the fan-out still read it. (3) P9 was listed as needing only P4, yet it uses `_BtaAttempt` (P6) and certifies BTA/MFI, whose graph debt clears in P7. | P2 order: templates first (consumers still call the public entry), then the leaf moves, then the `_ainfer` / bridge / tool_as switch (B27 fixed there), then OpenClaw. P6: commit 2 deletes the five mailbox fields; `_worker_instances` goes in commit 6, after its readers migrate (commits 3, 5, 6). P9 needs P4 and P6, and BTA/MFI certification also needs P7 (§11). | `_worker_instances` readers `BTA.py:1871`, `:2138-2146`, `:1003-1017`, `AF/inferencer_base.py:3342`; base `_ainfer` → public entry `AF/streaming_inferencer_base.py:1041` |
| h | **Some relied-upon tests don't test CoreProjects [R1#12].** Several terminal-inferencer targets depend on the ScienceModelingTools copy of `agent_foundation`, so they would stay green whatever CoreProjects does. A target that collects zero tests also passes. | P0 checks the dependency of every target the plan relies on, and repoints or twins the wrong ones. Every phase gate compares per-target collected-test counts with the P0 baseline. | `TEST/common/inferencers/terminal_inferencers/BUCK` (lines 19, 32, 44, 57, 68, 81, 92) → `//_tony_dev/ScienceModelingTools/src:agent_foundation`; `TEST/common/inferencers/BUCK` → CoreProjects |
| i | **New state classes could fail to decode [R1 (a)].** Codec registration happens at import. A store loaded in a fresh process before the defining module is imported degrades the unknown tag to a dict. | `NodeOutcomeState`, `RenderedTaskContractState` and `BtaCallSummary` live in `RC/state.py` and are exported from `RC/__init__`, beside the existing state classes. A fresh-import rehydration test covers them. | `register_state` `RC/state.py:31-45`; `decode_state` warns and degrades `:82-108`; `RC/__init__.py:30-43` eagerly imports `BTAState`, `DualState`, … |
| j | **A released frame could still be written [R1 (b)].** A task or callback created during a call copies `_current_invocation` and can outlive the call. It would find the frame through `frame_for` and write components or register resources into a ledger that never closes again. | `InvocationFrame.closed` is set at release. `frame_for` skips closed frames, and `ledger.register` on a closed frame raises `InvocationContractError` (§5.1). | ContextVar copy semantics (`asyncio` tasks, `copy_context`) |
| k | **"Every commit reverts alone" was false [R1 (e)].** Later commits build on earlier ones. | "Each commit is green when it lands. Rollback reverts from the stack tip in reverse dependency order; a commit with no dependents reverts alone." (§11, §13) | §11 dependency table |
| l | **The sync ledger close had the wrong safety argument [R2#2 (b)–(d)].** v7 called `_run_async(ledger.aclose())` "exactly as safe as `adisconnect` from any loop". A loop-bound resource can't be closed from another loop: the claude_code SDK itself drops a stale-loop client without disconnecting it. [new] `_run_async` also raises inside a running loop, which is where a sync BTA called from async code runs today (via the B14 thread hop). | **Loop affinity:** a loop-bound resource is closed inside its creating loop. Async entries close their ledger in the entry's loop. On sync paths, SDK leaves close per-call clients inside their own `_run_async` loop (codex's `_run_and_close`; the claude_code SDK adopts it no later than P6 commit 5). The sync ledger then does only loop-independent teardown or idempotent `adisconnect`. It runs through `run_async_joined`, a small RPU helper beside `_run_async` that hops to a context-copying worker thread when a loop is running. B14's fix in P1 uses the same helper (§5.1). | claude_code SDK sync `_infer` `:425-451` (clears a stale-loop client without disconnecting), `_client` / `_connected_loop` `:516-520`; codex SDK `_run_and_close` `:360-406`; `RPU/common_utils/async_utils.py:440-478`; `workgraph.py:2746-2751` |
| m | **`_bind_rebuilt_child_ws` was ALLOWED on shared stages [R2#3].** It writes a backing workspace onto the child instance, and BTA calls it on borrowed instances too, so under a host ctx it is a definition write that races between calls. | Owned stages only (they are the run). Borrowed stages get a ctx publication in host mode and keep the setter in bare modes. The row leaves ALLOWED (§4.7). Lands in P6 commit 5. | docstring "Safe ONLY for non-shared nodes" (`AF/inferencer_base.py:935-956`); calls `BTA.py:1205-1215`, `:2524-2534`, `:3008-3017`, base `:3307-3310` |
| n | **LWI resume flags missing from the inventory [R2#5].** `_step_was_previously_attempted` and `_previous_attempt_info` are per-call values that the RPU workflow engine writes onto the LWI instance through properties. | The properties route to a frame component in P9; direct-hook tests are wrapped in `open_invocation` (§7). | `FLOW/linear_workflow_inferencer.py:630-657`; writers `RPU/common_objects/workflow/workflow.py:1185-1186`, `:1199-1200`, `:1384-1385`, `:1405-1406`, `:1539-1540`, `:1553-1554`, `:1742-1743`, `:1768-1769`; readers `FLOW/plan_then_implement_inferencer.py:564`, `:2144-2145` |
| o | **rovodev's output-file ContextVar was cited as a precedent, but it is B29 [R2#6].** It is set in the streaming body and cleared with `.set(None)` and no token, so it leaks into the consumer between yields, just like `_active_ctx`. | It becomes a frame component, and its three readers fall back to `self.output_file`. The row leaves the §10 precedents. The B29 contract test asserts that the consumer's `copy_context()` is unchanged between yields. That test catches any leaf ContextVar. Lands in P3 after commit 7. | `EXT/rovodev/rovodev_cli_inferencer.py:61-64`, set `:753`, cleared `:801`, read `:466`, `:572`, `:617` |
| p | **`_BtaGraph` copied only ten fields; one path site and B16b were open [R2 minor (a)–(b), new].** Fields the engine reads through `self` would silently take `WorkGraph` defaults. Child-checkpoint promotion re-evaluates the ambient workspace. B16b was still marked a hypothesis. | `_BtaGraph` copies every `attrs.fields(WorkGraph)` field except the per-attempt ones, and a coverage test fails on any unclassified field (§5.8). `_promote_child_checkpoints` joins the `_BTA_CALL` path sites (§5.7). B16b is confirmed and fixed with the B16 typed fields (§5.9). | `WorkGraph` fields incl. `node_cls`, `verbose_repr`, `result_pass_down_mode`, `unpack_single_result`, `ignore_stop_flag_from_saved_results`, `executor`; `AF/inferencer_base.py:2378-2440`; `_effective_role` `master = state.template_version` (`AF/templated_inferencer_base.py:479-494`) |
| q | **No view of partial stopping [R3 gaps 1–2].** | §11 adds "What stopping after Pn buys", including that the P6 guard rejects host-mode B24 while bare B24 remains until P7. | — |
| r | **G1–G6 were only "paused" [R3 gap 3].** P6 changes the fan-out's aggregator, ledger and `n_subtasks` source, so the stack needs a baseline and a re-check. | Recommendation, run only on the user's GO: run G1–G6 against `69ea2d3762af` before P0, and again after P6 (§17). | `AF/inferencer_base.py:3333-3349`, `:3444-3469` |
| s | **Success-only compat flush would create stale getters [R3 gap 4].** Today the transports write the getter fields during the call, whatever its result. Flushing only on success would leave a failed bare call's getters showing the previous call. | Every non-host close flushes the compat fields the call published; the outcome stays success-only (§5.4). The "compat on success only" entry leaves `BARE_EXPECTED_CHANGES`. | TSIB `:440-443`; rovodev `finally` `:783-803`; devmate counter `:1281`, `:1332` |

**Feedback not adopted:**

| Feedback | Verdict | Why |
|---|---|---|
| R1#2: the claim opens after `_init_call_state`; ancestor stacking shares one node | outdated | v7's seam claims at step 2, before `_init_call_state` (step 5), and the claim is strict (§3 rows 3, 19) |
| R1#3: the outcome has no invocation freshness | outdated | v7 clears it at open and stamps `invocation_id` (§5.1 step 3, §5.3) |
| R1#6, first half: sync-created stages wait for `adisconnect` | outdated | v7 closes sync stages at attempt or call end. v8 corrects only the mechanism (row l); the other two halves are row d |
| R1#9, P4 half: P4 consumes `BtaCallSummary` | outdated | v7 moved BTA/MFI/Dual publication to P6 (§11, P4 note); the other halves are row g |
| R1#10, lease half: the lease must span retries, reruns, finalize and cleanup, and cover `checkpoint_dir`-only runs | outdated | v7 leases `checkpoint_root` for the whole logical call (§5.11); the state-machine half is row c |
| R1#11: certification comes too late and is inherited | outdated | the single-flight guard lands in P6 and reads `vars(type(owner))` (§5.1) |
| R1#12, target-type half: a `python_unittest` target collects nothing | outdated | v7 uses `python_pytest`; the dependency and collection half is row h |
| R1 (c): tool_as `_last_response` has no migration commit | outdated | P10's B28-family commit migrates it (§7, P10) |
| R1 (d): keep pre- and post-P8 goldens apart | outdated | §5.12: post-P8 goldens differ by exactly the manifest and the lock file |
| R2#2 (a): a deferred-close set keeps sync stages until `adisconnect` | wrong for v7 | v7 had no deferred-close set (§3 row 9 and §16 reject it); the valid part of R2#2 is row l |
| R1#4: a guardrail-rejected attempt's summary reaches `_finalize_output` after a fallback | partly wrong | guardrails are leaf-only: attaching one to an orchestrator raises `ValueError` (`AF/inferencer_base.py:1486-1499`), and the judge is skipped on orchestrators (`:4297-4302`, `:4408-4413`). An orchestrator's recovery re-raises (`:4200-4201`), so no BTA run is ever rejected after returning. The valid residue (a result from an external fallback or a non-exception `default_return_or_raise`) is row b |
| R3 claim D: B29 frame lifetime is sound | partly wrong | true of `framed_agen`'s own bindings, but it misses the `enter_run` in the generator body (row a) |
| R3 gap 2: close B24 more cheaply than P7 | not adopted | the only cheaper closure is an interim per-instance lock, a band-aid the user ruled out. Host B24 is rejected from P6 by the guard (BTA is uncertified); bare B24 has no production caller (§2.2, §11 stopping table) |

### 2.4 What v7 corrected in v6

Each row was verified in source before adoption. Critique points that did not survive verification are recorded in §16.

| v6 said | v7 | Evidence |
|---|---|---|
| `_COMPAT_SINK: ClassVar[Mapping]` plus a separate `_compat_publish` | `RuntimeKey.compat` declares the getter fields a component feeds; `publish_result` / `read_result` are the one API (§5.4) | the v6 design declared each fact twice (key and sink map), so the two could drift |
| BTA writes the observer into borrowed stages via `child_rc.handles.stream_observer`; instance value first | declared invocation keywords (`stream_observer=`, `interactive=`, `bta_node_name=`), consumed by the stage's entry into its frame; **per-call value first** (§5.9) | Tier-3 handles are connection-scoped, so an observer would outlive its call. Today BTA *overwrites* the configured value whenever a reporter exists (`BTA.py:2803-2815`, `:3026-3034`, `:3139-3147`), so v6's precedence was inverted; Conversational already resolves `interactive` per-call first (`conversational_inferencer.py:548-549`) |
| stage ownership covered workers and the aggregator | the breakdown stage too (always borrowed: it has no factory path; `BTA.py:600`, `:3149`) | its observer (`:3144`) and workspace (`:2062`) are written today |
| duck-typed stages not mentioned | documented attribute protocol for non-`InferencerBase` stages (`mock_inferencers/mock_bta_components.py`), with an explicit I2 exemption and never certifiable | they have no entry and read `self.stream_observer` directly |
| nested-BTA `worker.name` write not listed | B31: a per-call node name, passed as a call keyword from P7 | `BTA.py:2552-2554` writes it on every nested BTA worker, borrowed ones included |
| the terminal handoff (B28) was the only transport→`_ainfer` handoff | B28 covers every leaf call-result handoff: tool_as `_last_response`, `_last_stream_result`, `_last_clean_output`, token and usage counters | `tool_as_inferencer.py:372` → `:395-430`; the same pattern in 8 leaves |
| "no frame → written immediately" | frameless is mode-aware: under host, publish is a no-op; in bare modes, publish writes the declared compat fields. A frameless component *read* raises `NoInvocationError`, because compat fields are a lossy projection and an in-call read outside an invocation is a broken precondition | Dual renders previews by calling `_render_prompt` directly on its review and fixer leaves, frameless, under their host child ctx (`FLOW/dual_inferencer.py:1552`, `:2039`) |
| the sync bridge "cancels the task" | an explicit loop/task owner (`asyncio.Runner`), with the handle published before the thread starts, cancel through `call_soon_threadsafe`, and a bounded join (§5.1) | today there is no loop or task handle (`AF/streaming_inferencer_base.py:1077-1114`) |
| OpenClaw `_maybe_initialize_session` "idempotent per session" | guarded by the per-*instance* flag today (B19); per-session after P10 | `EXT/openclaw/openclaw_inferencer.py:1193-1206` |
| OpenClaw `fallback_mode = NEVER` as a soft class default | warn and override a non-default `max_retry` / `fallback_mode` / `fallback_inferencer` at construction (the Q3 precedent) | those settings are silently ignored today; convergence would otherwise make them live |
| claim: acquire and release | the claim records its owner, entry and start time for diagnostics; a stream acquires at its first resumption; `evict_subtree` asserts no live claim strictly below the prefix | Dual evicts its own subtree between consensus attempts (`FLOW/dual_inferencer.py:999`) |
| ActivePathClaims "an `eq=False` member" | a plain member of the hand-written `RunStateStore`, excluded from `to_json` | `RC/store.py:88-92` is not an attrs class |
| B16: key, root, version honoured | also the master version: `RoleState` has no typed field for it, so a host `switch_role(template_master_version=…)` reaches only `changes` | `RC/state.py:204-215`; `AF/templated_inferencer_base.py:660-678` |
| B16b at `:491-492` | `:492-493` | off by one |
| B14 as a one-line fix | the same one line, plus a repo-wide audit of `WorkGraph._run` callers inside a running loop | shared RPU behaviour changes for every such caller |
| dependency graph: P3 after P2 only | P3 also needs P1 (B14 is in P3's stop gate), and P5 needs P1 (B12); §17's order is one valid serialization | §11 |
| aggregator reader list "exhaustive" | P0 classifies all 35 `.aggregator_inferencer` hits in `AF/` as *needs the resolved stage* or *presence check* | four presence checks at `BTA.py:3207-3208`, `:3346-3347`, `:3427-3428`, `:3459-3460` were missing |
| fan-out "pre-builds its aggregator" | spelled out against today's code: resolve with a pure `resolve_stage`, pass the instance into the one `fresh_instance` call, and register the per-call fanout in the caller's ledger, so the sync fan-out is finally closed; `worker_count` is read through `_summary_at`, with a no-ctx branch | `AF/inferencer_base.py:3265-3266` (sync path never disconnects), `:3444-3469`; the fan-out's `child_rc` is `None` in true no-ctx (`:3307`) |
| **From [Idem] v4, each re-verified:** the frame opens in `_(a)infer_single`, after `_init_call_state`; CLI session policy runs outside the frame | **one seam, fixed order** (TL;DR 5, §5.1): claim first, then `_init_call_state`, then the provider hooks `_prepare_call` / `_conclude_call`, all inside the frame (B34) | `infer` / `ainfer` run `_init_call_state` after `enter_run` and before dispatch (`AF/inferencer_base.py:3888-3901`, `:5262-5268`). MFI's override resets dispatch state (`FLOW/multi_flow_inferencer.py:1483-1491`). `parallel_infer` / `aparallel_infer` call `_(a)infer_single` directly (`:4064-4079`, `:5403-5417`). devmate turns `success=False` into an exception only after `_ainfer_single` returns (`EXT/devmate/devmate_cli_inferencer.py:1278-1330`), which is after v6 would have published the outcome |
| claim with ancestor stacking, "e.g. fan-out delegation at the same path" | **strict claim, no stacking** [Idem] | That example is false: the fan-out runs at child `bta_inferencer` (`AF/inferencer_base.py:3306-3307`). Others checked: guardrail at `guardrail` (`:4529`); fallback at `fallback/external_{i}` (`:3019-3030`, `:5092-5104`); flow-node fallback at `fallback` (`AF/agentic_inferencers/conversational/flow_node_adapter.py:474-479`); agentic functions at `ctx.child(slot)` unless the caller passes a ctx (`AF/agentic_functions/decorator.py:288-297`). The session helpers, `iter_infer` and `__call__` call the same instance's `infer` without an outer frame |
| lazy `infer(iterator)` not mentioned | binds per `next()`, like streaming (B29 extended) [Idem] | `infer` defers `exit_run` into the returned generator's `finally` (`AF/inferencer_base.py:3891-3915`); `iter_infer` wraps it (`:3946-3978`). `ainfer` collects eagerly (`:5274-5293`) |
| sync owned stages go to a deferred-close set drained by `adisconnect` | **every owned stage closes at attempt or call end.** Sync closes through the framework's joined bridge, `_run_async(ledger.aclose())`. There is no deferred-close set [Idem, refined] | Nothing defines a sync close hook: `InferencerBase` has only `aconnect`, `adisconnect`, `__aenter__` and `__aexit__` (`AF/inferencer_base.py:5421-5448`), and BTA adds none. A default sync BTA (`use_async=False`) runs each worker's `infer` on that call's own short-lived `_run_async` loop (`RPU/common_objects/workflow/workgraph.py:2777-2842`; `BTA.py:2735-2777`), so no creating loop survives the call |
| cleanup errors logged and recorded, never raised after success | recorded as strings. They never replace an in-flight exception (logged and attached as a note). After a success, the outcome is published first, then `InvocationCleanupError` is raised carrying the result [Idem] | Today's only per-call close, the async fan-out's `finally: await fanout.adisconnect()` (`AF/inferencer_base.py:3288-3292`), raises. BTA `adisconnect` attempts every child, then re-raises the first failure (`BTA.py:1003-1017`). Logging alone would hide leaks |
| outcome published on success, never cleared | cleared when the invocation opens, and stamped with `invocation_id` [Idem] | nodes are path-keyed and reused (Dual rounds): success A followed by a failed B at one path would leave A readable as B's |
| BTA paths read `self._workspace` at each site | one snapshot per invocation (`effective_workspace`, `checkpoint_dir`, `checkpoint_root`) that every site reads. Precedence is unchanged [Idem, refined] | every site re-evaluates the property under whichever ctx is active (`BTA.py:1695-1703`, `:2185-2189`, `:2276-2280`, `:2877-2889`, `:3053-3057`, `:1758-1789`, `:1869`, `:2012`). `_workspace_under` consults the *active* ctx's `workspace_override` handle first (`AF/inferencer_base.py:838-857`), so a read inside a child's callback can resolve to the child's workspace |
| recursive interactive rerun untouched | explicit attempt loop with a `_begin_attempt` hook [Idem] | `return await self._ainfer(...)` (`BTA.py:2363`) dispatches polymorphically back through MFI's `_ainfer` (`FLOW/multi_flow_inferencer.py:1882-1891`). It would overwrite `_BTA_ATTEMPT` while the outer attempt is still live |
| P4 publishes BTA contracts from `_BtaAttempt.worker_contracts` | P4 covers the generic channel, the templated leaf and LWI. BTA, MFI and Dual publication and the parent readers move to P6 [Idem] | `_BtaAttempt` exists only from P6 |
| P11 static gate: one borrowed instance dispatched to more than one concurrently runnable worker | a **dynamic single-flight guard** at frame open, from P6 [Idem, refined] | A class-level worker check misses the breakdown and aggregator roles, configured descendants, and one borrowed object shared by two BTA calls. P7 makes overlapping BTA calls possible, which is before P11 |
| Tier-3 keyed by `ctx.path` | B32: keyed by `(handle scope, path)` (§5.5) [Idem] | `AF/streaming_inferencer_base.py:515-560`; every root is `"/"` (`RC/context.py:78`) |
| devmate `_consecutive_error_count` ALLOWED as connection health | B33: session-scoped, beside the session it governs [Idem] | it forces a new session after N failures (`EXT/devmate/devmate_cli_inferencer.py:1134-1150`, `:1281`, `:1332`) |
| manifest with a "semantic definition digest"; lease at `<workspace>/checkpoints` | canonical identity rules, a reconstructable effective plan, a lease on `checkpoint_root` for the whole logical call; P8 required for completion (§5.11, §5.12) [Idem] | digests alone can't replay a plan. `checkpoint_dir`-only BTA is supported for its breakdown, aggregator and graph result paths (`BTA.py:695`, `:1695-1703`, `:2185-2189`, `:3053-3057`) |
| `TEST/run_context/BUCK` following `python_unittest` conventions | `python_pytest`, including `conftest.py` [Idem] | the suite is pytest functions with an autouse `conftest.py` fixture; the template is `TEST/knowledge/BUCK` |
| smaller items | I2 forbids writes to definition-owned or borrowed objects (invocation-owned products may be initialized); the P4 stop gate uses the P0 scoped search; `arc f` / `arc lint -a` run on explicit paths; §9 asserts observable behaviour, not `bta._graph`; P0 keeps the frozen plan and inventory as versioned files in `AgentFoundation/` [Idem] | the working copy has unrelated changes; `_graph` is a transient private |

Idem proposals that v7 rejects, with reasons, are in §16.

### 2.5 What v6 corrected in v5

| v5 said | v6 | Why |
|---|---|---|
| `RunScoped` descriptor with bare write-through | explicit typed components plus an explicit compatibility sink | A descriptor hides mode-dependent semantics at every `self._x`, is fragile above `@attrs`, and preserves accidental `__dict__` bytes rather than documented behaviour. |
| owner-keyed guard | `(store, path)` claim with ancestor stacking | an owner key misses two instances at one path |
| re-entrant frames share values | every public entry is new; template-method streaming split | it removes the only public self-re-entry (`AF/streaming_inferencer_base.py:1041`) |
| MFI reads LWI's internal step node | recursive layering: LWI publishes its own contract, MFI reads `flow_N` [Sep/Dec] | encapsulation |
| lease, manifest, cleanup and stage ownership deferred | adopted | cross-process writes, resume identity, and B25 are real gaps |
| P1 reset lines for B1/B3 | structural fixes only (P4/P6) | the resets were band-aids |
| finalize reads live workers | typed `BtaCallSummary` [Idem/Harm] | resume-safe |
| `TEST/run_context/` runs in buck | it has no BUCK file; P0 adds ownership [Idem] | otherwise the tests can't run |
| purity fix (c): "compare identity first" | recursive fingerprint, normalizers, negative tests | otherwise a nested mutation is invisible |
| ALLOWED: `_propagate_to_children` markers; OpenClaw `_session_initialized` | the markers don't exist; the OpenClaw flag is bug B19 | accuracy |
| "batch `ThreadPool`" (§3.5) | it is `parallel_infer`, which derives a `parallel_{i}` ctx per input (`AF/inferencer_base.py:4064-4080`, `:5413`) | accuracy; it also means the claim can't trip on it |
| aggregator stored on the run object | a call-scoped owned component; the slot is never written | §5.7 |

---

## 3. Adjudications (contested points, verified in source)

| # | Point | Positions | Decision and evidence |
|---|---|---|---|
| 1 | Per-call access | Snappy/Harm: `RunScoped` descriptor · Sep/Idem/Dec: typed components | **Typed components.** Ownership is visible at each site; attrs, deepcopy, pickle and `fresh_instance` are untouched. |
| 2 | Public re-entry / streaming entry | Snappy/Harm: shared frames · Sep/Idem/Dec: base `_ainfer` calls the private `_ainfer_streaming_pipeline` · **v6: template-method split** | **Template-method split [new].** See the evidence and the per-class plan below this table. |
| 3 | Concurrency guard | Snappy/Harm: `(path, owner)` · Sep: `(store, path)` · Dec: `(id(store), path)` registry on Tier-2 · v6: `(store, path)` with ancestor stacking · Idem v4: strict | **A strict `(store, path)` claim, with the registry owned by the `RunStateStore`** (not serialized). The store defines path identity, so every ctx over one store sees the claim whatever `RuntimeBindings` it carries. **No stacking:** every verified nested call runs at a child path (§2.4), so stacking would only hide bugs. If P0 finds a same-path nested *public* call, it is refactored to a child slot or to the private pipeline; the claim is never relaxed. |
| 4 | Guardrail empty-fingerprint window | Sep/Idem: per attempt · Snappy/Dec: per call | **Per call, spanning attempts, in `_current_fallback_state` (`_fs`).** It counts consecutive identical empties *across retries* (`AF/inferencer_base.py:4229-4272`). A per-attempt window resets every attempt, so fail-fast could never fire. `_fs` is set and reset around the whole retry loop (`:3069-3120`, `:5145-5203`). |
| 5 | Generic `AttemptFrame` | Sep/Idem: yes · Dec: no | **No.** Only BTA has attempt-scoped live state, held in a local `_BtaAttempt`. |
| 6 | Post-call result shape | Snappy/Sep/Harm: `task_contract` field · Idem/Dec: `NodeOutcomeState` | **`NodeRunState.outcome = NodeOutcomeState(task_contract, summary, final_output)`,** typed only (no free-form dict), **published once per invocation** at its successful close (§5.3). |
| 7 | Stage slots and shared-instance safety | Sep v1: blanket fresh · Snappy: share · Sep v2/Dec: borrowed vs owned plus static certification (Dec from P4) · v6: static gate in P11 · Idem v4: a gate over the whole reachable tree in P6 | **Borrowed vs owned, plus a dynamic single-flight guard from P6** (§5.1). In host mode, an instance whose class isn't `_HOST_PURE_CERTIFIED` may run only one invocation at a time in the process, whatever its role, path or depth. A static tree walk can't see runtime reachability (round-robin lists, factories, configured descendants, one object shared by two BTA calls). The guard checks the actual overlap at the one place every invocation passes. It must land in P6, because P7 makes overlapping BTA calls possible. Concurrent host configs that P0 finds sharing an uncertified instance are migrated to a factory slot in the same commit. Sequential borrowing stays valid. |
| 8 | Aggregator scope | Dec: per attempt · v6: per call | **Per call.** It keeps today's within-call reuse across retries (the slot write persists across attempts today) and never crosses calls. |
| 9 | Worker cleanup timing | Snappy: `adisconnect` only · Sep/Dec: attempt end, "sync hooks" · v6: sync defers to `adisconnect` · Idem v4: `close_sync` with a native hook, else a joined bridge, plus a loop-affinity preflight | **Every owned stage closes at attempt end (workers) or call end (aggregator, fan-out), sync included, under loop affinity: a loop-bound resource is closed inside the loop that created it.** Async: `await ledger.aclose()` in the entry's loop. Sync: an SDK leaf closes its per-call client inside its own `_run_async` loop before that loop ends (codex's `_run_and_close`, `EXT/codex/codex_sdk_inferencer.py:360-406`; the claude_code SDK adopts it in P6 commit 5). The sync ledger then does only loop-independent teardown or idempotent `adisconnect`, through `run_async_joined`, a new RPU helper beside `_run_async` that hops to a context-copying thread when a loop is running (`_run_async` raises there, `RPU/common_utils/async_utils.py:440-478`). No native sync hook exists (`AF/inferencer_base.py:5421-5448`), so Idem's first branch would be dead code, and a preflight is replaced by the leaves closing in their own loop. v6's deferral was tracked leakage. Tier-3 keeps only connection-scoped resources. |
| 10 | Resume of legacy workspaces | Harm: `legacy_warn` · Sep/Idem/Dec: fail closed | **Fail closed, with an explicit logged opt-in `resume_identity_policy="trust_legacy"`** and a certify helper. |
| 11 | Foreign `worker_inferencers` in a fan-out template | Idem: raise | **Warn and override** (user decision Q3). |
| 12 | Interim reset lines | Harm S0; Snappy v5 P1 | **None.** B1 and B3 are fixed structurally in P4/P6; no production path hits them today. |
| 13 | Judge | Dec v1: `prepared_input` alone | **`prepared_input=True`, plus an explicit `input_preprocessor` step.** `prepared_input` skips preprocessing; nulling the manager didn't. |
| 14 | `WorkGraph` base | remove together with composition vs staged | **Staged:** composition in P7, base removal in P11 after a repo-wide audit [Sep/Dec]. |
| 15 | Per-call values a parent assigns to a stage (observer, interactive, nested name) | Sep/Dec/v6: ctx handles · Snappy v5: allowed writes · **v7: call keywords** | **Declared invocation keywords into the stage's own frame** (`_INVOCATION_KEYWORDS`; the `extra_feed` / `prepared_input` pop precedent, `AF/inferencer_base.py:2777-2779`, `:4861-4863`). They die with the call, need no new ctx machinery, and are concurrency-safe by construction. Precedence is per-call value, then configured value. That matches today's overwrite and Conversational's existing `interactive` rule (`conversational_inferencer.py:548-549`). Tier-3 handles would outlive the call; a per-child Tier-2 overlay would be a new concept (`RunContext` has only `child()`, `RC/context.py:89-121`). |
| 16 | Duck-typed stages (no `InferencerBase`) | not addressed by any plan | **Documented attribute protocol** (`stream_observer`, `interactive`), `hasattr`-gated as today. It gets a scoped I2 exemption, and such stages are never certified. They have no frame, so the guard can't see them; BTA refuses to dispatch one borrowed duck-typed instance to two concurrently runnable workers in host mode (§5.7). |
| 17 | Certification: `_HOST_PURE_CERTIFIED` ClassVar vs deriving it from measured debt | Dec: ClassVar · critique: derive | **ClassVar, verified by the ratchet.** The single-flight guard runs in production and can't import test data (`KNOWN_DEBT` lives under `TEST/`). The ClassVar declares the fact and the ratchet proves it, the same declared-then-verified pattern as a type annotation checked by pyre. |
| 18 | `LiveHandleField` vs "no descriptors" | critique: inconsistent | **Allowed only to implement an existing public property.** `active_session_id` is already a public property with this mode-dependent contract (`AF/streaming_inferencer_base.py:580-640`). The descriptor implements that one documented contract once. Private handle state uses explicit `_tier3_get` / `_tier3_set`. The §16 rejection targets private call state hidden behind `self._x`. |
| 19 | Where the claim opens relative to per-call work | v6: frame inside `_(a)infer_single`, after `_init_call_state` and outside the CLI wrappers · Idem v4: one seam, claim first | **One seam with a fixed order** (TL;DR 5, §5.1), inside the frame. Leaf session policy moves into two sync hooks with identity defaults, `_prepare_call(kwargs)` and `_conclude_call(result)`. The CLI `@bridge_entrypoint` `ainfer` / `infer` stay as thin adapters that keep their no-mint policy. `_seed_graph_reporter_into_runtime` stays at the entry: it is idempotent Tier-2 transport, not call state. |
| 20 | Live-handle scope for independent host roots | v6: path only · Idem v4: `(handle scope, path)`, or route through `ctx.handles` | **`(scope_id, path)` on the leaf's own store** (§5.5). Routing through `ctx.handles` would break leaf `adisconnect` teardown and share one bag across every instance at a path. |
| 21 | Cleanup failure after a successful inference | v6: logged and recorded · Idem v4: raise a typed error after recording | **Record, publish, then raise `InvocationCleanupError(result, errors)`** (§5.1). Today's fan-out close and BTA `adisconnect` raise. A result already produced is never lost: it is on the exception and in the published outcome. An in-flight exception always wins, with the cleanup failures attached as a note. |
| 22 | BTA workspace precedence under a host ctx | today: `_workspace_under` (active ctx override, then backing, then ctx workspace) · Idem v4: ctx workspace over configured backing | **Keep today's precedence; snapshot it once at invocation open** (§5.7). Flipping precedence inside `_workspace_under` would silently re-root every class that configures a workspace. It is a behaviour change unrelated to purity. The real defect is re-evaluating the property under whichever ctx is active, and the snapshot fixes that. Two overlapping host calls on one definition with a configured workspace would share one root. They are rejected before any I/O: by the single-flight guard while BTA is uncertified, and by the lease with `BtaWorkspaceBusyError` (§5.11) once it is certified, or when two distinct instances share the root. Callers wanting independent trees configure no backing workspace, or use a factory. |

**Evidence for row 2** (`AF/streaming_inferencer_base.py`, `AF/terminal_inferencers/`, `EXT/`):

| Class | Today |
|---|---|
| base | The public `ainfer_streaming` (`:751-786`) = `enter_run` + the fan-out branch + `self._ainfer_streaming_pipeline`. Base `_ainfer` (`:1041`) and `infer_streaming` (`:1082`) consume the **public** method, passing `(inference_input, inference_config, **kwargs)` positionally. |
| rovodev, devmate | Both subclass `TerminalSessionTemplatedInferencerBase`. Its `_ainfer` (`AF/terminal_inferencers/terminal_session_inferencer_base.py:464-490`) calls `super()._ainfer`, which reaches the base `_ainfer`, which calls the leaf's **public** `ainfer_streaming`. Their `ainfer` overrides are `@bridge_entrypoint` (`RC/bridge.py:77`; rovodev `:827`, devmate `:1210`), which does **not** mint for a true bare call. |
| rovodev | Public override `:722-803`: output file, then session resolution, then `super().ainfer_streaming` (`:776`). Its `finally` sets `_last_clean_output`, which `_ainfer` reads after `super()._ainfer` (`:805-826`). |
| devmate | Public `ainfer_streaming(prompt, filter_session_info=True, **kwargs)` (`:1335-1439`) is devmate's whole pipeline: session/resume/`new_session`, cache, filter, the `dump_output` flip, and a direct `self._ainfer_streaming(...)` (`:1409`). Under `ainfer()`, `inference_config` binds positionally to `filter_session_info` (B27). |
| OpenClaw | Subclasses `StreamingInferencerBase` directly. Public `:1061` is its streaming pipeline (validation `:1080`, session, gateway/CLI), and it never calls `enter_run`. Its `ainfer` (`:980-1031`) calls `enter_run` itself, then `_maybe_initialize_session` and `_ainfer_with_retry` (`:892`) directly. It never reaches `_ainfer_single` or `ainfer_streaming`, so it skips the base per-call machinery, and under v6 it would get no frame, no claim and no outcome (B30). Its `infer` (`:1034-1054`) is `_run_async(self.ainfer(...))`. Its `_ainfer` (`:1380-1396`) already does the prompt/session/retry body. |
| claude_code CLI | `ainfer` / `infer` (`:835-960`) route through `_(a)infer_single`, as do codex CLI (`:626-700`) and kiro (`:287-400`). But its public sync `infer_streaming` (`:1000-1049`) is a native sync transport: it resolves the session, opens the cache and calls `self._infer_streaming(...)` (TSIB `:518`) with no `enter_run`, so `run_context` stays in `kwargs` and flows into `construct_command` (B26). |
| all | `grep "def ainfer_streaming\|def infer_streaming"` over `AF/` gives 8 public streaming entries: base ×2, Conversational, claude_code sync, devmate ×2, OpenClaw, rovodev. The table covers all of them. |

What goes wrong with the private split:
- **Under `ainfer()`:**
  - **devmate** would lose session handling and the filter.
  - **rovodev** would lose the output file and session handling, and `_ainfer` would then read a stale `_last_clean_output` (B23).
- **On direct streaming with an explicit `run_context`:**
  - OpenClaw, devmate (sync and async) and claude_code's sync entry never call `enter_run`; `run_context` stays in `kwargs` and is dropped or forwarded to the transport.
  - rovodev resolves the session *before* `super()` enters the ctx.
  - So all of them read and write the caller's session slot (B26). The private split fixes none of this.

The template-method split keeps the leaf behaviour, fixes B26, and still makes every public entry a new invocation (§5.2).

---

## 4. The model

### 4.1 Modes (decided when an invocation opens, from `active_run_context()`)

| Mode | Condition | Store | Purity target |
|---|---|---|---|
| **host** | a ctx with `not ctx.legacy_mint` | owned by the host; may be saved and resumed | **yes**: zero writes outside ALLOWED |
| **legacy** | `ctx.legacy_mint` (a bare `ainfer()` minted a throwaway root, `RC/bridge.py:39-47`; children inherit the flag) | discarded at root exit | isolated per invocation; documented getters via declared compat fields |
| **no-ctx** | no active ctx: a true-bare `@bridge_entrypoint` `ainfer` / `infer` (claude_code, codex, kiro, rovodev, devmate), setup, or reads between calls | — | explicit reconfiguration APIs only; documented getters via declared compat fields |

### 4.2 Homes: narrowest scope wins

| # | Home | Lifetime | Holds | Mechanism |
|---|---|---|---|---|
| 1 | **Definition** (instance) | object | config fields, post-init derivations, `_init_recipe`, pure caches | attrs (existing) |
| 2 | **`InvocationFrame`** | one public call: retries, guardrail, finalize and the streaming generator's life | typed components (PTI/LWI current values, the templated render, stream results, BTA summary and aggregator) plus the call's `ResourceLedger` | **new**; a ContextVar chain (`RC/invocation.py`) |
| 2a | **Attempt local** | one iteration of BTA's attempt loop (§5.7) | `_BtaAttempt`: graph, stages, caches, flags, worker contracts | captured by the attempt's closures |
| 3 | **`NodeRunState`** | one run; persisted by the host | `call`, `role_state`, **`outcome`** (new), `provenance`, `checkpoints` | `RC/store.py` |
| 4 | **Live handles** | connection | sessions, session-scoped health (devmate's error counter), SDK clients, subprocesses | `RC/handles.py`, keyed `(scope_id, path)`; `_tier3_get` / `_tier3_set`, `_session_scoped_get` / `_set`, `LiveHandleField` |
| 5 | **Workspace** | across restarts | checkpoints, the BTA manifest, the lease file | files |
| — | **Transport** (Tier-2 `RuntimeBindings`) | run | reporter, interactive, cancellation, stream observer handle | existing |

**Choosing a home:**
- the same for every call → **1**;
- needed only during the call → **2** (**2a** for per-attempt BTA state);
- needed after the call by someone else, or after a restart, and serializable → **3**;
- a live resource that outlives the call → **4**;
- must survive a process restart → **5**.

### 4.3 Lookup rules

- **In-call, by the owner:** `frame_for(self)` components, then its own node, then its own handles. **Never a compat field.**
- **Post-call, by a parent:** `read_outcome(child_ctx)` through `RunStateStore.peek`, which never claims, so it can't raise `CollisionError`.
- **External getters after a bare call:** the declared compat fields (§5.4).

### 4.4 Entry rule

- **Public entries open a new invocation.** A frame opens in exactly three places: `_(a)infer_single`, base `ainfer_streaming` and base `infer_streaming`. The two streaming templates are plain methods that return a framed generator, so no entry-level ContextVar is set in a generator body (§5.1, §5.2). Every public entry reaches one of them:
  - `infer` / `ainfer`, through `_infer_single` / `_ainfer_single` (`AF/inferencer_base.py:2767`, `:4856`). This includes the six CLI overrides: claude_code, codex, kiro, rovodev, devmate, and OpenClaw after B30;
  - `parallel_infer` / `aparallel_infer`, one per input under `parallel_{i}`;
  - `iter_infer`, sequentially;
  - base `ainfer_streaming` (async template);
  - base `infer_streaming` (sync template, §5.2). Its default pipeline is the thread bridge. devmate and claude_code override the pipeline with their native sync transports;
  - the session helpers (`AF/streaming_inferencer_base.py:1171-1290`), through `ainfer`;
  - OpenClaw's `infer`, a thin `_run_async(self.ainfer(...))` adapter.
- **Private calls stay in the caller's invocation:**
  - `_ainfer` / `_infer`;
  - `_ainfer_streaming_pipeline` / `_infer_streaming_pipeline`;
  - recovery;
  - `super()._ainfer` / `super()._infer`, including MFI's `_infer` delegating to BTA's (MFI's `_ainfer` override is deleted in P6, §5.7; the only other callers of BTA's `_ainfer` / `_infer` today are MFI's two overrides, `FLOW/multi_flow_inferencer.py:1892`, `:1915`);
  - the BTA attempt hooks `_begin_attempt` / `_conclude_attempt`, which run inside the owner's frame.
- **Rules for overrides:**
  - A public `ainfer_streaming` / `infer_streaming` override may only validate or adapt arguments and then call `super()`. Behaviour goes in a pipeline override.
  - A public `ainfer` / `infer` override must reach `_(a)infer_single`, or be a thin adapter over another public entry.
  - The `ainfer` / `infer` overrides of the CLI leaves (claude_code, codex, kiro, rovodev, devmate, OpenClaw) become thin adapters: validation, `new_session` / `session_id` argument adaptation, and the no-mint `@bridge_entrypoint` policy. They then call `_(a)infer_single`. Their session policy moves into the seam's hooks: `_prepare_call(kwargs)` (session resolution, `_apply_session_policy`) and `_conclude_call(result)`, which takes **all** of their post-call code (the claude_code, codex and kiro session writes, rovodev's result wrap and `find_latest_session_id`, devmate's `success=False` promotion and error counting). Both run inside the frame, so a rejected call touches nothing, and a failure that devmate promotes publishes no outcome. Model I/O, such as OpenClaw's session initialization, belongs inside `_ainfer`.
  - `_init_call_state` runs inside the seam, once per invocation: per `parallel_infer` item, per iterator item, and for CLI and streaming entries too (B34).
  - `test_streaming_entry_contract.py` enforces these rules (I9). It enumerates every class in `AF/` that overrides a public entry and checks that a stub transport observes `frame_for(inst)`.
  - Conversational is the only recorded exemption (§5.2).
- **Non-inferencer callers** reach a public entry and need nothing new:
  - `TemplatedInferencer.__call__` (`AF/templated_inferencer.py`) renders, then calls `base_inferencer(...)` with a derived `run_context`;
  - the agentic-function wrappers call `infer` / `ainfer` with an explicit `run_context` (`AF/agentic_functions/decorator.py:378`, `:467`).
- **Private hooks called on another object** run outside that object's invocation, so `frame_for(obj)` is `None` there.
  - Today's only production case: Dual's logging-preview `_render_prompt` on its review and fixer leaves (`FLOW/dual_inferencer.py:1552`, `:2039`), under the leaf's child ctx.
  - Rule: under a host ctx such a call may only read. `publish_result` is a no-op there (§5.4), and the ratchet's I2 catches any other write.
  - The real dispatch that follows renders again inside the leaf's own frame and publishes the contract.
- **Private hooks called directly by tests** (about 30 test files call `_ainfer` / `_infer` directly) run frameless. `publish_result` keeps bare semantics there (the documented getters still work), but any in-call component read raises `NoInvocationError` (§5.4), because a private hook's precondition is that it runs inside its owner's invocation. P0 lists these tests. The migration commit that first makes a class's private hook read a component (terminal and tool_as in P10, BTA in P6, PTI/LWI in P9, including the RPU workflow tests that drive LWI's resume-flag properties) wraps that class's direct-call tests in `open_invocation(inst)` / `aopen_invocation(inst)` in the same commit.

### 4.5 Why `frame_for(owner)` walks by identity

- A parent's callback can run inside a child's call (e.g. `on_node_stream`).
- An LWI/PTI owner reads its own components while a step child runs.

In each case the owner's own frame is an ancestor in the chain.

A streaming consumer is **not** such a case. Streaming frames are bound only while the pipeline runs (§5.1), so between yields the consumer runs in its own context.

### 4.6 Concurrency contract

| Case | Behaviour |
|---|---|
| Same path, overlapping, any instances, any nesting | `ConcurrentInvocationError` at frame open, before `_init_call_state`, the hooks, or any node or model I/O (host and legacy) |
| Nested work | always at a child path (fan-out, guardrail, fallback, agentic functions; §2.4). Private calls stay in the current frame and claim nothing |
| Sequential same-path re-entry (Dual rounds) | allowed after release; the outcome is cleared at open, so round B never exposes round A's result |
| `infer(iterator)`, lazy (no merger) | the ctx is resolved (minted at most once) before `infer` returns; each `next()` binds it and runs that item's `_infer_single`, which opens its own invocation at the same path. The items run one after another, so the strict claim holds. Abandoning the iterator leaks nothing (B29). The merger path is eager and gets the same per-item invocations |
| A stream abandoned without closing (`async for … break`) | still a live invocation: its transport may still be running. It keeps its claim until it is closed or finalized (GC finalizes it; `asyncio.run` shuts down live async generators). A same-path call before then raises `ConcurrentInvocationError`. The message names the holder's owner class, entry and age, and tells the caller to close the stream (`contextlib.aclosing` / `closing`). A stream object that is created but never iterated holds no claim, because the claim is acquired at the first resumption. |
| `evict_subtree(prefix)` (Dual between consensus attempts, `FLOW/dual_inferencer.py:999`, evicting its own path) | removes node state only. It asserts that no claim is live strictly below `prefix`, since children must have finished. The caller's own claim at `prefix` stays. |
| One BTA workspace written by two threads or processes | `BtaWorkspaceBusyError` (lease, P8) |
| One instance, concurrent, host mode (any paths, any stage roles, any two parent calls) | certified class: supported. Uncertified: the second overlapping invocation raises `UncertifiedConcurrentUseError` at frame open (streams: at first resumption), naming the class and suggesting a factory slot. Enforced from P6 by the single-flight guard |
| Two independent host roots sharing one leaf | separate sessions and clients: handles are keyed `(scope_id, path)` (B32). Bare concurrent calls are not session-isolated unless callers pass distinct roots |
| Concurrent bare calls on one instance | isolated per invocation; the guard doesn't apply; compat fields are last closing writer wins (documented) |

### 4.7 ALLOWED instance residents

Anything else that changes under a host call fails I1.

| Resident | Why |
|---|---|
| config fields, `_init_recipe`, `_extension_manager_cache` | definition, or a pure function of it |
| logger un-defer: `logger`, `_logger_awaiting_workspace`, `_resolved_logger_configs`, `_ws_log_relpaths` | workspace-derived definition cache, idempotent (normalized by the fixed purity tool) |
| lazily created locks | definition cache |
| `_live_handle_store` | Tier-3 connection holder, keyed by `(scope_id, ctx.path)` (B32) |
| `attach_state_graph` registrations, debug toggles | explicit reconfiguration |
| `_applied_role` and template fields from a **no-ctx** `switch_role` | explicit reconfiguration |
| MultiFlow `_last_*` legacy mirror | public post-call getters, already ctx-backed in host mode (`FLOW/multi_flow_inferencer.py:1268-1339`) |
| PTI `output_path` normalization | idempotent |
| **compat fields declared by `RuntimeKey.compat`** on the class's keys | written only by the non-host flush at every close, or by a frameless bare hook call (§5.4); the ratchet derives this set from the declared keys |

---

## 5. Mechanisms

### 5.1 `InvocationFrame`, `RuntimeKey`, `ResourceLedger`, and the path claim (`RC/invocation.py`, new)

```python
@attrs.define(frozen=True)
class RuntimeKey(Generic[T]):
    name: str                                   # "<Class>.<purpose>", a ClassVar beside the owner class
    factory: Optional[Callable[[], T]] = None   # used by get_or_create
    compat: Mapping[str, str] = attrs.field(factory=dict)   # instance field -> attribute of the value ("" = the value)

@attrs.define(eq=False, slots=True)
class InvocationFrame:
    owner: Any                                  # compared by identity only
    ctx: Optional[RunContext]
    mode: Literal["host", "legacy", "no_ctx"]
    parent: Optional["InvocationFrame"]
    entry: str                                  # "ainfer", "infer_streaming", ... (diagnostics)
    opened_at: float                            # time.monotonic() (diagnostics)
    invocation_id: str                          # uuid4().hex; stamped on the published outcome
    components: dict = attrs.field(factory=dict)       # RuntimeKey -> value
    pending_compat: dict = attrs.field(factory=dict)   # compat field -> value (non-host only)
    ledger: "ResourceLedger" = attrs.field(factory=lambda: ResourceLedger())
    cleanup_errors: list = attrs.field(factory=list)   # strings from every ledger closed in this invocation
    closed: bool = False                        # set at release; frame_for skips closed frames
    def get(self, key: RuntimeKey[T]) -> Optional[T]: ...
    def require(self, key: RuntimeKey[T]) -> T: ...          # KeyError naming key and owner class
    def get_or_create(self, key: RuntimeKey[T]) -> T: ...    # requires key.factory
    def put(self, key: RuntimeKey[T], value: T) -> None: ...
    def discard(self, key: RuntimeKey[T]) -> None: ...

class ResourceLedger:                            # owned resources, closed in reverse registration order
    def register(self, resource: Any, label: str) -> None: ...   # InvocationContractError once its frame is closed
    async def aclose(self) -> list[str]: ...     # attempts every close; returns failure strings; idempotent

_current_invocation: ContextVar[Optional[InvocationFrame]]
def frame_for(owner) -> Optional[InvocationFrame]           # walks .parent by owner identity
def invocation_of(owner) -> InvocationFrame                 # frame_for, or NoInvocationError with the fix
def publish_result(owner, key: RuntimeKey[T], value: T) -> None       # §5.4
def read_result(owner, key: RuntimeKey[T]) -> Optional[T]             # §5.4; raises frameless
@contextmanager
def open_invocation(owner) -> Iterator[InvocationFrame]      # sync entries
@asynccontextmanager
async def aopen_invocation(owner) -> AsyncIterator[InvocationFrame]   # async entries
def framed_gen(owner, entry: str, run_context, start: Callable[[], Iterator[T]]) -> Iterator[T]    # sync streaming templates
async def framed_agen(owner, entry: str, run_context, start: Callable[[], AsyncIterator[T]]) -> AsyncIterator[T]
def ctx_bound_gen(rc: RunContext, gen: Iterator[T]) -> Iterator[T]   # lazy infer(iterator): binds the ctx only
class InvocationContractError(RuntimeError): ...           # never retried: in both non_retryable_exceptions tuples
class ConcurrentInvocationError(InvocationContractError): ...
class UncertifiedConcurrentUseError(InvocationContractError): ...
class InvocationCleanupError(InvocationContractError):
    result: Any                                  # the completed result; never lost
    errors: tuple[str, ...]                      # one "<resource>: <exc type>: <msg>" per failed close
class NoInvocationError(RuntimeError): ...
```

**The seam [Idem, refined].** Every public entry reaches it through `_(a)infer_single` or a streaming template (§4.4). Order:

| # | Step | Notes |
|---|---|---|
| 1 | bind the ctx | `enter_run` at the public entry, as today (`AF/inferencer_base.py:3888-3890`, `:5262-5264`). `_seed_graph_reporter_into_runtime` stays here: it is idempotent Tier-2 transport. Streaming templates resolve the ctx at the first resumption and bind it per resumption (below) |
| 2 | open the frame | compute the mode; build the frame (`parent` = current frame, fresh `invocation_id`); with a ctx, `ctx.store.claims.acquire(ctx.path, frame)`; in host mode, register the single-flight guard; set the ContextVar |
| 3 | clear the outcome | `ctx.store.clear_outcome(ctx.path)`: non-claiming, under the store lock |
| 4 | pop invocation keywords | `_INVOCATION_KEYWORDS` into components, beside `extra_feed` / `render_only` / `prepared_input` (`:4861-4863`) |
| 5 | `_init_call_state(inference_input)` | moved out of `infer` / `ainfer` (`:3901`, `:5267`); now per item for iterators and `parallel_infer`, and for CLI and streaming entries (B34) |
| 6 | `self._prepare_call(kwargs) -> kwargs` | sync; identity default. CLI leaves put session resolution and `_apply_session_policy` here |
| 7 | the pipeline | `__(a)infer_single_impl` (retries, guardrail, fan-out, finalize, post-processing) or the streaming pipeline. `_fs` alone is too narrow: it resets at `:3119-3124` |
| 8 | `self._conclude_call(result) -> result` | sync; identity default. devmate promotes `success=False` to its exception and counts errors here; rovodev records `find_latest_session_id` here. It may raise, in which case the call failed |
| 9 | close the ledger | every owned resource closed; failures recorded as strings (below) |
| 10 | publish (success only), then flush (every close) | on success, the outcome (§5.3), stamped with `invocation_id` and `cleanup_errors`. Then, on **every** close (success, failure, cancellation), the compat fields this call published, in non-host modes (§5.4) |
| 11 | release | the guard, then the claim; always, in `finally` |
| 12 | unbind | `reset(token)` for `_current_invocation`, then `exit_run` at the public entry |

Steps 2–11 are `open_invocation` / `aopen_invocation`. The streaming templates run the same steps through `framed_gen` / `framed_agen`, except the provider hooks (6, 8), which belong to `_(a)infer_single`: a streaming leaf's session handling lives in its pipeline override (§5.2). Steps 1–5 run at the first resumption, 7 across resumptions, and 9–12 in the wrapper's `finally`, each under the per-resumption binding (B29, below). A rejected call (step 2 raising) has executed nothing past step 1.

**Ledger close (steps 9–10).**
- **Loop affinity.** A loop-bound resource (an SDK client, its anyio task group, subprocess pipes) is closed inside the loop that created it. `_run_async`'s own docstring names the cross-loop hazard (`RPU/common_utils/async_utils.py:446-455`), and the claude_code SDK drops a client whose loop has changed without disconnecting it (`EXT/claude_code/claude_code_sdk_inferencer.py:425-451`, `:516-520`).
- **Async entries:** `await ledger.aclose()` in the entry's own loop, where the call's resources were created.
- **Sync entries:** every `_run_async` loop inside a sync call ends with its coroutine (a sync BTA runs each worker's `infer` on its own loop, `RPU/common_objects/workflow/workgraph.py:2777-2842`). So an SDK leaf closes any per-call client inside that loop before it ends, as codex already does (`_run_and_close`, `EXT/codex/codex_sdk_inferencer.py:360-406`); the claude_code SDK adopts the pattern in P6 commit 5. What reaches the sync ledger is then loop-independent: idempotent `adisconnect` of stages whose clients are already closed, and plain teardown. It runs through `run_async_joined(ledger.aclose())`, a new RPU helper beside `_run_async` (P1 commit 2): with no running loop it is `_run_async`; inside a running loop, where `_run_async` raises (`:466-471`), it runs the coroutine on a worker thread under `contextvars.copy_context()` and joins. B14's fix uses the same helper.
- Every entry is attempted, in reverse registration order, and each failure is recorded as `"<resource>: <type>: <message>"`. Exception objects are never stored. A nested ledger closed earlier in the same invocation (a BTA attempt's) appends its failures to `frame.cleanup_errors` too, so step 10 reports every failure of the invocation in one place.
- **Inference raised:** the original exception propagates unchanged. The cleanup failures are logged at ERROR and attached with `exc.add_note(...)`. No outcome is published; non-host compat fields are still flushed (§5.4).
- **Inference succeeded, cleanup failed:** publish the outcome with `cleanup_errors`, flush compat, release, then raise `InvocationCleanupError(result, errors)`. This matches today, where `finally: await fanout.adisconnect()` (`AF/inferencer_base.py:3288-3292`) and BTA's `adisconnect` (`BTA.py:1003-1017`, which re-raises the first failure) propagate cleanup errors. It improves on today because the result is kept.
- `InvocationContractError` and its subclasses are added to both `non_retryable_exceptions` tuples (`AF/inferencer_base.py:3096`, `:5156`). A contract violation or a cleanup failure is never retried.

**Single-flight guard (P6).**
- **Registry:** a process-wide `dict[int, InvocationFrame]` keyed by `id(owner)`, under a `threading.Lock`, in `RC/invocation.py`. The frame holds a strong reference to its owner for its lifetime, so the id can't be reused while registered.
- **Rule:** in host mode, if `vars(type(owner)).get("_HOST_PURE_CERTIFIED", False)` is false and another live frame is registered for the same owner, raise `UncertifiedConcurrentUseError` naming the class, both entries, the holder's age, and "use a factory slot". Using `vars(type(...))` means certification is never inherited by a subclass that adds state.
- **Scope:** only overlapping invocations of one object. A nested frame of the same owner is impossible after the template-method split (private calls open none). Sequential reuse always passes. Legacy and no-ctx modes are unchanged, because bare calls are already isolated per invocation (§4.6).
- **Why dynamic:** it sees every stage role (breakdown, worker, aggregator, fan-out), every depth, and one object shared by two different parent calls. No static walk over config can see these (§3 row 7).
- **It also covers the instance's own concurrency APIs.** A host `parallel_infer` / `aparallel_infer` on an uncertified class runs overlapping invocations of one object, and so does a guardrail judge shared by those items. Today that is a live race (the B28 handoff: item A can parse item B's stdout). P0 inventories every host caller. The P6 guard commit resolves each one in the same commit: certify the class if its measured debt is already empty, or give the caller a factory or `fresh_instance` per item. If a case can't be resolved either way, only the guard commit stops (stop-point semantics, §11). The guard is never weakened.

**Outcome invalidation (step 3).** `RunStateStore.clear_outcome(path)` sets `NodeRunState.outcome = None` without claiming the node. It runs under the store's `threading.Lock`, which also guards claims and publishes. A node skipped entirely by a parent-level resume opens no frame, so its saved outcome stays readable. A child that is invoked and then resumes from its own cache clears and republishes, with a new `invocation_id`.

**Lazy `infer(iterator)` (B29 extended).** `infer` resolves the ctx (minting at most once, as today), calls `exit_run` before returning, and returns `ctx_bound_gen(ctx, inner)`. Each `next()` binds `_active_ctx` to that ctx around the inner step and resets it before yielding; it opens no frame, because the item's own `_infer_single` opens that item's invocation. `close()`, a `break`, GC, and a close from another thread all leave the consumer's ContextVars untouched. `ainfer(iterator)` is eager and needs no change.

**Streaming binding rule (B29).**
- **The problem today.** Base `ainfer_streaming` sets `_active_ctx` inside the async generator and keeps it set "across all yields" (`AF/streaming_inferencer_base.py:758-786`). An async generator runs in its consumer's task context, so the leaf's ctx leaks into the consumer between yields.
- **Abandoned streams make it worse.** If the consumer abandons the stream without `aclose`:
  - the leak becomes permanent in that task, and later bare calls there reuse the abandoned root through `enter_run`'s reuse branch (`RC/bridge.py:66-68`);
  - the finalizer's `exit_run` (`:72-74`) runs in another Context, so `reset(token)` raises `ValueError`.
  - A frame opened the same way would inherit both problems, and would also leave its claim unreleasable from the right Context.
- **The rule.** The public streaming templates are plain methods (not generator functions) that return `framed_agen(self, entry, run_context, start)` / `framed_gen(...)`, so no entry-level `enter_run` ever runs in a generator body. The wrapper resolves the ctx once, at its first resumption, by `enter_run`'s rule (explicit, then active, then mint) and holds it locally; a minted root is only a local value, so it needs no teardown. It binds `_active_ctx` and `_current_invocation` only for each resumption of the inner pipeline (around each `__anext__` / `next`), and resets them before yielding. `start` runs inside the first binding and decides between the fan-out branch and the private pipeline, so the fan-out runs inside the frame and its ledger registration has one.
  - `aclose` / `close` of the inner pipeline, the ledger and the claim release all run inside the same binding in the wrapper's `finally`.
  - Tokens are always reset in the Context that created them.
  - The pipeline itself sees exactly what it sees today.
  - The consumer sees only its own context.
  - An abandoned stream keeps its claim until finalized (§4.6).
- **Leaf ContextVars follow the same rule.** rovodev's `_current_output_file` (`EXT/rovodev/rovodev_cli_inferencer.py:61-64`) is set in its streaming body (`:753`) and cleared with `.set(None)` and no token (`:801`), so it leaks into the consumer between yields exactly like `_active_ctx`. It becomes a frame component, and its three readers (`:466`, `:572`, `:617`) fall back to `self.output_file` when frameless. The B29 contract test asserts that the consumer's `copy_context()` is unchanged between yields for every streaming entry, which catches any leaf ContextVar, not just the known ones.
- **The sync bridge.** Today it has no loop or task handle: a daemon thread runs `asyncio.run(pump())` and the generator only drains a queue (`AF/streaming_inferencer_base.py:1077-1114`; the join at `:1111` runs only after the queue drains), so an abandoned sync stream leaves its transport running. The default `_infer_streaming_pipeline` delegates to a new RPU helper `iterate_async_in_thread(agen_factory)` in `RPU/common_utils/async_utils.py`, next to `_run_async` (`:440`). The helper:
  - is started during the first bound `next`, so `copy_context()` carries the frame and ctx into the thread for the whole call;
  - runs an `asyncio.Runner` in the thread and publishes its loop to the generator (through a `threading.Event`) before it yields anything;
  - has the pump record `asyncio.current_task()` and check a `cancel_requested` flag before starting, which closes the race where close arrives before the task exists;
  - on `close` (or GC finalization) sets the flag, calls `loop.call_soon_threadsafe(task.cancel)`, drains the queue, and joins with a bound (`join(5.0)`). A thread still alive after that is logged at ERROR with the owner class; it is never silently abandoned.

  The frame's ledger and claim release run after the join, in the wrapper's `finally`.

**Path claim (`ActivePathClaims`):**
- It is a plain member of the hand-written `RunStateStore` (`RC/store.py:88-92`, not an attrs class). It is created in `__init__`, never serialized (excluded from `to_json`), and guarded by the store's `threading.Lock` (`parallel_infer` threads share the store), the same lock `clear_outcome` and `publish_outcome` take.
- `acquire(path, frame)` records `holders[path] = frame` only if no live holder exists. Otherwise it raises `ConcurrentInvocationError`, naming the path and, for both holder and caller, the owner class, `frame.entry` and the holder's age (`time.monotonic() - opened_at`). If the holder is a streaming entry, the message adds the `aclosing` / `closing` hint. There is no ancestor exception: nested work runs at child paths (§2.4).
- `release(path, frame)` asserts that `holders[path] is frame`, then removes it.
- `RunStateStore.evict_subtree(prefix)` (`RC/store.py:129-149`) asserts that no claim is live at a path strictly below `prefix`. Its only caller, Dual's consensus retry (`FLOW/dual_inferencer.py:999`), evicts its own path after its children have finished, so the assert holds by construction and catches any future misuse.
- **Streams acquire at their first resumption,** inside the first binding of `framed_agen` / `framed_gen`, not when the generator object is created. A never-iterated stream holds nothing; one abandoned mid-iteration holds its claim until it is closed or finalized.
- **What it closes:** `RunStateStore.node` (`RC/store.py:94-115`) treats a same-creator claim as a no-op, so two instances of one class at one path silently share a node today. The claim catches that, as well as sibling concurrency under one parent.

**Thread hops:**
- `asyncio.run` (`RPU/.../async_utils.py:440-478`), `asyncio.gather` tasks, and `infer_streaming`'s `copy_context` thread all copy the chain.
- `parallel_infer` workers re-enter a child ctx and open their own frame.
- The one hop that drops the chain is B14, which P1 fixes first by routing the thread-hop branch (`workgraph.py:2746-2751`) through `run_async_joined`, which copies the context. The fix is shared RPU behaviour, so P0 audits every `WorkGraph._run` caller that can run inside a running loop. Each caller must be correct with the caller's ContextVars visible; a caller that relied on a clean context is fixed explicitly in the same commit.

**Frame closure.** A task or callback created during a call copies `_current_invocation` and can outlive the call. `InvocationFrame.closed` is set at release (step 11). `frame_for` skips closed frames, so a late reader sees `None` and follows the frameless rules (§5.4), and `ledger.register` on a closed frame raises `InvocationContractError`, so a late resource is never silently orphaned.
- **`ctx.node(creator=…)` sites** (`AF/inferencer_base.py:3731`, `AF/templated_inferencer_base.py:476`, `:670`) must run inside the owner's frame, so the node claim and the path claim agree. P0 lists every site; any that runs outside a frame (e.g. `switch_role` before a call) is recorded, and it is either moved or documented as setup-time reconfiguration.

### 5.2 Streaming template-method split [new]

**Shape.**
- **Two public templates, both plain methods** (not generator functions) that return a generator. Each does the entry-level work exactly once: resolve and bind the ctx, then the frame (from P3), then the fan-out branch, then the private pipeline.
- **Pipelines are pure transports.** They carry no entry policy.
- **Staging.** P2 lands the plain-method shape returning an inner generator whose body is today's (`enter_run`, fan-out branch, pipeline), so P2 changes no lifetime. P3 commit 7 swaps that inner generator for `framed_agen` / `framed_gen`, whose `start` callable picks the fan-out branch or the pipeline inside the frame (§5.1). The only leaf changes in P3 are the CLI session-policy moves into `_prepare_call` / `_conclude_call` (P3 commit 6) and rovodev's output-file component (P3 commit 8).
- **P2 order keeps every commit green.** The templates land first while every consumer still calls the public entry; then the leaf bodies move into pipeline overrides; only then do base `_ainfer`, the sync bridge and tool_as switch to the private pipeline (B27 is fixed in that commit, because that switch is what changes devmate's binding); OpenClaw last. Switching `_ainfer` before the leaves move would make `ainfer()` skip rovodev's and devmate's logic in between.

| Class | After P2 | P3 adds |
|---|---|---|
| `StreamingInferencerBase`, async | public `ainfer_streaming(inference_input, inference_config=None, *, run_context=None, **kwargs)`, a plain method returning an async generator: `enter_run`, then the fan-out branch (`_arun_fanout`, unchanged), else `self._ainfer_streaming_pipeline(...)`. `_ainfer` (`:1041`) consumes **`self._ainfer_streaming_pipeline`**, so it stays in the caller's invocation. Fan-out under `ainfer()` is decided earlier, in `_ainfer_single`. | returns `framed_agen(self, "ainfer_streaming", run_context, start)`; `start` picks fan-out or pipeline inside the frame |
| `StreamingInferencerBase`, sync | public `infer_streaming(inference_input, inference_config=None, *, run_context=None, **kwargs)`, a plain method returning a generator: `enter_run`, then the fan-out branch (`_run_fanout`, `AF/inferencer_base.py:3264`), else `self._infer_streaming_pipeline(...)`. New private hook `_infer_streaming_pipeline`: its base default is today's thread bridge (`:1061-1114`), now driving `self._ainfer_streaming_pipeline` instead of the public async entry. That removes the last public→public self-call. | returns `framed_gen(self, "infer_streaming", run_context, start)`; bridge cancel+join on close (B29) |
| `ToolAsInferencer` | `_ainfer` (`AF/agentic_inferencers/tool_inferencers/tool_as_inferencer.py:422`) consumes the pipeline. | `_last_response` becomes a frame component, with a compat field if P0 lists it as a documented getter (P10) |
| rovodev | Its body (output file, session resolution, `finally`) becomes a `_ainfer_streaming_pipeline` override that wraps `super()._ainfer_streaming_pipeline`. The base clean-output hook (`:998`) is inside the base pipeline, so `finally` order is unchanged. The public override and its delegation branch (`:741-747`) are deleted; the base template handles delegation. | `_current_output_file` becomes a frame component (§5.1, B29) |
| devmate, async | Its body becomes `_ainfer_streaming_pipeline(self, inference_input, inference_config=None, *, filter_session_info=False, **kwargs)`; `inference_input` plays today's `prompt` role. The keyword default `False` keeps today's `ainfer()` behaviour exactly, where the positional `inference_config` (normally `None`) lands in `filter_session_info` (B27). The two defaults encode a real distinction: the pipeline is the *transport* (raw chunks, which `_ainfer` accumulates and parses from raw stdout), while the public streaming entry is *presentation* for a human-facing consumer. Keyword-only makes a positional mis-bind impossible. The public override stays only as a signature adapter: `(prompt, filter_session_info=True, **kwargs)` → `super().ainfer_streaming(prompt, filter_session_info=filter_session_info, **kwargs)`, so direct streaming still filters by default. The delegation branch (`:1359-1365`) is deleted. | — |
| devmate, sync | Its body (`:1499-1560`: session, cache, filter, `dump_output` handling) becomes `_infer_streaming_pipeline(self, inference_input, inference_config=None, *, stream_callback=None, output_stream=None, filter_session_info=True, **kwargs)`. The public override stays only as a signature adapter for its positional `(prompt, stream_callback, output_stream, filter_session_info)` → `super().infer_streaming(prompt, stream_callback=…, output_stream=…, filter_session_info=…, **kwargs)`. | — |
| claude_code CLI, sync | Its body (`:1000-1049`) becomes `_infer_streaming_pipeline`, and the public override is deleted: the signatures already match. | — |
| OpenClaw, streaming | Its body becomes a pipeline override. The public override becomes `self._validate_bta_inferencer_spec()` then `super().ainfer_streaming(...)`; drop `type: ignore[override]` if the signature now matches. | — |
| OpenClaw, `ainfer` (B30) | Converges onto the entry every other CLI leaf uses. `ainfer` keeps `_validate_bta_inferencer_spec`, its explicit `enter_run` (mint policy unchanged), the `new_session` pop and the session-id resolution, then returns `await self._ainfer_single(inference_input, inference_config, session_id=session_id, **kwargs)`. `_ainfer` already resolves `session_id` from `kwargs` and sets `active_session_id` from the result (`:1386-1395`). `_maybe_initialize_session` (model I/O) moves into `_ainfer` (`:1380`), inside the frame. It is guarded by the per-instance `_session_initialized` flag today (`:1193-1206`; B19), and by a per-session Tier-3 set after P10, so a second attempt is a no-op in both. OpenClaw owns its rate-limit retries (`_ainfer_with_retry`, `:892`). Its `__attrs_post_init__` therefore normalizes the base retry settings, following the Q3 warn-and-override precedent: a non-default `max_retry`, `fallback_mode` or `fallback_inferencer` is logged at WARNING ("OpenClaw owns its retries; base setting ignored") and reset to `max_retry=1`, `fallback_mode=FallbackMode.NEVER`, `fallback_inferencer=None`. Today those settings are silently ignored by `ainfer`; without the override, convergence would make them live and change behaviour. The base loop then makes exactly one `_ainfer` attempt (`AF/inferencer_base.py:3041`; the async attempt count equals `max_retry`). `infer` stays the thin `_run_async(self.ainfer(...))` adapter. | — (frame via `_ainfer_single`) |
| Conversational (`AF/agentic_inferencers/conversational/conversational_inferencer.py:3129`) | Out of scope: its own loop, and it skips super post-init. Recorded as a documented bypass: an inventory row plus a ratchet exemption with the reason. | — |

**Behaviour deltas** (each is covered by a P0 golden):
- **devmate and OpenClaw direct streaming** now run inside the base entry's `enter_run`, which legacy-mints when there is no ctx. This is the same policy every other streaming leaf already has on direct streaming.
  - Session semantics are unchanged: under `legacy_mint`, the `active_session_id` getter takes the no-ctx branch and the setter writes the instance backing (`AF/streaming_inferencer_base.py:595-636`).
  - `bridge_entrypoint`'s docstring warns against minting because a minted root used to drop session ids. That warning predates the setter's legacy branch at `:629-636`.
- **Bare rovodev `ainfer()`** (a no-ctx `@bridge_entrypoint` call) no longer legacy-mints inside `super().ainfer_streaming`. The pipeline now runs in the caller's mode (no-ctx). Session reads and writes are identical in both modes. `_effective_cache_folder()` is golden-checked.
- **devmate and claude_code sync direct streaming** now run inside the sync template's `enter_run` (legacy mint, as above). They also honour a configured `bta_inferencer` fan-out, which today they silently skip, unlike their async entries (B26b).
- **B26:** an explicit `run_context` is now honoured by every streaming entry.
- **OpenClaw `ainfer()` (B30)** now passes through the base per-call machinery: `_fs`, guardrail (off by default), `total_timeout_seconds`, preprocessing and finalize, all identity for a default OpenClaw. The attempt count and the rate-limit retries are unchanged. Configs that set `max_retry`, `fallback_mode` or `fallback_inferencer` on an OpenClaw leaf are listed by P0; they now get a construction-time WARNING and the override, instead of being silently ignored. Configs that set a guardrail or timeout on OpenClaw get them honoured for the first time; P0 lists them and each is confirmed as intended.

**Proof:**
- The P0 streaming-entry goldens stay unchanged apart from the B26 characterization. They cover `ainfer` and direct streaming (sync and async) per leaf, and two sequential bare calls per path for session continuity.
- OpenClaw golden: a stub gateway that raises a rate-limit N times gives the same number of transport calls, continuation prompts and final session before and after B30.
- Parity test: `ainfer()`, direct `ainfer_streaming()` and direct `infer_streaming()` give the same output on a stub transport.
- Under a host ctx, the stub transport observes `active_run_context() is ctx` (P2), plus `frame_for(inst) is not None` (P3).

**Stop condition:** if a golden shows a leaf's other Tier-3 handles losing continuity under legacy mint, P2 stops for that leaf. The leaf keeps its current entry until the handle is moved to `LiveHandleField` (P10). The mint policy is not special-cased.

### 5.3 Typed outcome channel [Sep/Idem/Dec + new single publish point]

```python
@register_state
@attrs.define(frozen=True)
class RenderedTaskContractState(InferencerStateBase):
    text: str
    sha256: str
    role: Optional[str]            # active role at render time; None = own
    source_path: str               # ctx.path of the node that rendered it

@register_state
@attrs.define(frozen=True)
class NodeOutcomeState(InferencerStateBase):
    task_contract: Optional[RenderedTaskContractState] = None
    summary: Optional[InferencerStateBase] = None     # e.g. BtaCallSummary
    final_output: Optional[str] = None                # leaves whose streams differ from their final output
    invocation_id: str = ""                           # the frame that produced it
    cleanup_errors: tuple[str, ...] = ()              # recorded strings, never exception objects
```

- **Freshness:** the seam clears the node's outcome when an invocation opens (§5.1 step 3), and publishes at most once, at that invocation's successful close. Failed, rejected or cancelled invocations leave `None`. A stream succeeds only when it is exhausted without an exception; one closed early publishes nothing. A reader can tell which invocation produced a value from `invocation_id`.

- **Storage:**
  - `NodeRunState.outcome` is serialized; `from_json` uses `.get`, so old stores load it as `None`.
  - `RenderedTaskContractState`, `NodeOutcomeState` and `BtaCallSummary` live in `RC/state.py` and are exported from `RC/__init__`, beside the existing state classes. Codec registration happens at import (`RC/state.py:31-45`), and `decode_state` degrades an unknown tag to a dict (`:82-108`), so a class defined in `BTA.py` would fail to rehydrate in a fresh process that loads a store before importing BTA. A fresh-import rehydration test covers all three.
  - Helpers in `RC/outcome.py`: `publish_outcome(ctx, creator, outcome)` and `read_outcome(ctx)` (via `peek`).
- **Single publish point.** At the successful close of its invocation, the owner calls `self._outcome_for(frame) -> Optional[NodeOutcomeState]`, a polymorphic hook with base default `None`. If the frame has a ctx, the result is published at the owner's own node.
  - Classes contribute through frame components during the call; nobody publishes mid-call.
  - This makes MFI (a BTA subclass that runs BTA's `_ainfer` as a private super-call) publish exactly once, with its own selection.

| Class | `task_contract` | `summary` |
|---|---|---|
| templated leaf | its render: `_RENDERED_CONTRACT`, captured at `AF/templated_inferencer_base.py:454` | — |
| LWI | its first step child's outcome contract | — |
| BTA | `BtaCallSummary.selected_contract` of the run that returned: the **lowest successful worker index**, sync and async | that run's `BtaCallSummary` (§5.7); none when a fallback produced the result |
| MFI | the winner first, then ascending flow index, read from each `flow_N` child's outcome | `BtaCallSummary` |
| Dual | `state["prior_task_instructions"]` when non-empty, else the propose child's contract | — |

- **Readers:**
  - Dual (`FLOW/dual_inferencer.py:785`, `:1438`);
  - MFI (`FLOW/multi_flow_inferencer.py:1606-1630`);
  - the fan-out's `_conclude_fanout` (`AF/inferencer_base.py:3333-3349`), which reads `summary.worker_count`, so the `BtaFanOutComplete.n_subtasks` metric is preserved (`~/bta_fanout_e2e/metrics.py:277`).
- **Reader API:** `_task_contract_at(child, child_ctx) -> str`.
  - With a ctx, it uses the typed channel only (`""` if absent).
  - Only in true no-ctx does it call the child's documented getter.
- **Final output.** A leaf whose `streams_differ_from_final_output` is true (rovodev, `EXT/rovodev/rovodev_cli_inferencer.py:135`) publishes `final_output` from `_outcome_for`. Conversational, which today calls the leaf's `get_final_output()` after streaming (`conversational_inferencer.py:909-913`, `:938-941`), reads `_final_output_at(child, child_ctx)` instead, the same shape as `_task_contract_at`: the typed outcome under a ctx, the getter only in true no-ctx. It lands with the P10 rovodev commit, which is when the getter's fields become host-silent compat fields.

### 5.4 Bare compatibility, declared on the keys [Sep's sink, made declarative + new]

**Declaration.** A key whose value backs a documented bare getter says so in `RuntimeKey.compat`, mapping each getter field to an attribute of the value (`""` = the value itself). One declaration per fact, so the component and the getter field can't drift. For example:

```python
_RENDERED_CONTRACT = RuntimeKey[RenderedTaskContractState](
    "TemplatedInferencerBase.rendered_contract",
    compat={"_last_rendered_task_instructions": "text"})
_TERMINAL_RESULT = RuntimeKey[TerminalStreamResult](
    "TerminalInferencerBase.stream_result",
    compat={"_last_streaming_output": "stdout",
            "_last_streaming_stderr": "stderr",
            "_last_streaming_return_code": "return_code"})
```

Other declarations: BTA/MFI `_worker_task_instructions` (read by `_proposer_task_instructions`); rovodev `_last_clean_output` / `_last_raw_stdout` (`get_final_output`); the stream-result fields behind `get_streaming_result`. P0 inventories the exact list of documented getters.

**Field names stay the same as today.** The getters and the duck-typed `_FakeBTA` / `_FakeMFI` tests in `TEST/common/inferencers/test_task_instructions_snapshot.py` therefore stay green unchanged.

**API.** `publish_result(owner, key, value)` and `read_result(owner, key)` are the only way a class writes or reads a result-bearing component. They are mode-aware:

| Situation | `publish_result` | `read_result` |
|---|---|---|
| owner frame, host | `frame.put(key, value)` only | `frame.get(key)` (`None` if unset) |
| owner frame, legacy or no-ctx | `frame.put(key, value)`, plus the compat fields into `frame.pending_compat`; flushed to the instance on every close (success, failure, cancellation) | `frame.get(key)` |
| frameless, under a host ctx (a private hook called on another object, e.g. Dual's preview `_render_prompt`, §4.4) | no-op | raises `NoInvocationError` |
| frameless, no host ctx (a test calling `leaf._render_prompt(...)` directly, e.g. `TestLeafCapturesItsOwnRender`) | writes the declared compat fields immediately (none for a key without `compat`) | raises `NoInvocationError` |

- **Why frameless reads raise instead of rebuilding from the compat fields:** compat fields are a lossy projection (the rendered contract keeps only `text`, not `sha256`, `role` or `source_path`), and an in-call read outside any invocation is a broken precondition, not a mode. The error names the key and owner class and says "public entries open an invocation; tests that call a private hook directly wrap it in `open_invocation(inst)`".
- **Why frameless publishes still write in bare mode:** a hook called on its own is a bare call, and the documented getter is the one thing a bare caller can observe.

**Rules:**
- The outcome is success-only (§5.3). Compat fields are different: today the transports write the getter fields during the call, whatever its result (TSIB `:440-443`; rovodev's `finally`, `:783-803`), so a bare caller sees the failed call's values. Flushing only on success would leave a failed call's getters showing the *previous* call. So every non-host close flushes exactly the compat fields that call published, and nothing else.
- Internal readers use `read_result` or the typed outcome, never a compat field; P11 adds a source check.
- The ratchet derives the ALLOWED compat set from the keys a class declares (walking the MRO's `RuntimeKey` ClassVars), and allows those fields to change only in non-host modes.

### 5.5 `LiveHandleField`: exact session semantics [Snappy]

`active_session_id` becomes `LiveHandleField("live_session_id", "_session_id")`, a generalization of `AF/streaming_inferencer_base.py:580-640`. It fixes both halves of B6:
- **Host reset:** writes `_TOMBSTONE` to the branch slot, and the getter returns `None` for it. Today the getter falls back to the shared backing (`:593-601`).
  - The tombstone lives until the next write.
  - Teardown (`_iter_live_handle_sets`) never reads `live_session_id`.
- **No-ctx reset:** also clears `live_session_id` in every branch the no-ctx getter reads (`:612-624`).

**Handle scope (B32) [Idem, refined].**
- **Today:** `_tier3_get`, `_tier3_set` and `active_session_id` all use one instance-owned `LiveHandleStore` keyed by `ctx.path` alone (`AF/streaming_inferencer_base.py:515-560`, `:580-640`). Every root has path `"/"` (`RC/context.py:78`). So one leaf shared by two independent host roots (two OpenTeam sessions, two requests) shares one session id and one SDK client. The ctx's own connection-scoped `_handle_store` (`ctx.handles`, `RC/context.py:129-132`), which the README documents as the Tier-3 home (`RC/README.md:18`, `:124-126`), is never consulted.
- **Fix:** the leaf stays the connection holder, and its branches are keyed by `(scope, ctx.path)`:
  - **host ctx:** scope = the root's `LiveHandleStore`, identified by a `scope_id` assigned when the store is constructed. `id()` is never used, because it is reused after GC;
  - **legacy mint:** one fixed legacy scope, so bare calls keep today's cross-call continuity exactly;
  - **no ctx:** the instance backing, unchanged.
- **Why not move the handles into `ctx.handles`:** leaf `adisconnect` tears down every branch the leaf holds (`_iter_live_handle_sets`, `:562-578`; claude_code SDK `:527`, codex SDK `:267`, rovodev serve `:212`). The ctx bag is also shared by every instance at a path. Scope keying keeps teardown as it is; `_iter_live_handle_sets` yields the branches of every scope.
- **Continuity contract:** a host that wants cross-turn continuity reuses its root's handle store (README Note B).
  - OpenTeam's session roots already do: they are cached per session (`OpenStartup/src/openteam/server/services/conversation_service.py:343`).
  - Some callers mint a fresh root per turn and rely on today's path-only continuity: `conversation_service.py:1262`, `AgentFoundation/src/agent_foundation/resources/tools/task/executor.py:670` and `:1046`, and `resources/tools/sop/cli.py:254`. P0 lists them. Where they expect continuity, the same commit switches them to pass their session's handle store.

**Session-scoped state (B33) [Idem].**
- **Today:** devmate's `_consecutive_error_count` (`EXT/devmate/devmate_cli_inferencer.py:288`) is per instance. `_apply_session_policy` (`:1134-1150`) forces a new session once it reaches a threshold, and `ainfer` increments or resets it (`:1281`, `:1332`). So one host branch's failures reset another branch's session.
- **Fix:** two helpers on `StreamingInferencerBase`, `_session_scoped_get(name, default)` and `_session_scoped_set(name, value)`, apply exactly the session policy that `active_session_id` uses:
  - host: the branch slot, keyed `(scope_id, path)`;
  - legacy and no-ctx: the instance backing.
- `LiveHandleField` is built on the same two helpers, so session and session-health can never follow different policies. The counter leaves ALLOWED.

`LiveHandleField` is used only to implement an existing public property with this documented mode-dependent contract (today, `active_session_id`; §3 row 18). Private handle state (conversation ids, the OpenClaw initialized-session set) uses explicit `_tier3_get` / `_tier3_set` calls. The `_tier3_get` / `_tier3_set` policy (any ctx → store) stays as it is for SDK clients; it is a different, deliberate policy.

### 5.6 Guardrail window per call (B5)

`_guardrail_recent_empty_fingerprints` (`AF/inferencer_base.py:598-603`) is deleted. The window becomes `_fs["guardrail_empty_fingerprints"]`, which spans all attempts of one call. This is bare-visible and tagged B5.

### 5.7 BTA: attempt, summary, stage ownership, aggregator [Idem + Sep v2 + Dec + new]

```
public call ─ InvocationFrame (seam, §5.1)
   _BTA_CALL (_BtaCallRecord, put at the start of BTA's _ainfer/_infer):
       effective_workspace, checkpoint_dir, checkpoint_root      # frozen snapshot; every path site reads it
   _BTA_AGGREGATOR (call-scoped ResolvedStage, frame ledger)
   └─ one BTA run = BTA._ainfer (sync _infer: exactly one attempt)
        loop:
          attempt = _BtaAttempt(); frame.put(_BTA_ATTEMPT, attempt); frame.discard(_BTA_SUMMARY)
          self._begin_attempt(attempt, inference_input)          # hook; MFI: propagation + cross-flow reset
          graph: _BtaGraph (P7)   stages: [ResolvedStage]   ledger: ResourceLedger (attempt-owned stages)
          original_query, promoted_breakdown, aggregation_guidance, worker_contracts{index: contract}
          topology_emitted, pending_topology
          try → run the graph; interactive review (async only, as today)
          finally → close the attempt ledger (async: await aclose(); sync: run_async_joined(aclose())); failures → frame.cleanup_errors
          review says "rerun" → continue                         # the tail below is skipped, as the recursive return skips it today
        tail (today :2365-2366; sync :2210):
          _emit_graph_reconcile (async) → _finalize_response(result)
          result = self._conclude_attempt(result)                # hook; MFI: normalize → extract dispatch state → strip
          frame.put(_BTA_SUMMARY, BtaCallSummary(...))           # the run's last statement: nothing after it raises or awaits
          return result
   base retry helper returns (a BTA run, an external fallback, or a non-exception default_return_or_raise)
   _finalize_output (:1824-1886): _BTA_SUMMARY present → today's branches, reading it and _BTA_CALL, never live stages
                                  absent → super()._finalize_output(response) (B36)
   seam steps 9–11: frame ledger closes the aggregator and any fan-out → publish outcome → release lease (P8), guard, claim
```

- **The attempt loop [Idem].** Today an interactive "rerun" decision calls `return await self._ainfer(...)` recursively (`BTA.py:2353-2363`). That call dispatches polymorphically, back through MFI's `_ainfer` (`FLOW/multi_flow_inferencer.py:1882-1891`), and would nest a second attempt inside a live one. v7 replaces it with an explicit loop in BTA's async `_ainfer`. Each iteration builds a new `_BtaAttempt`, discards any `_BTA_SUMMARY`, calls `self._begin_attempt(attempt, inference_input)`, then runs today's body (`:2215-2363`: the initial topology, the graph, the interactive review). The attempt's ledger closes in the iteration's `finally`, before the next iteration or the tail. The only step before the loop is the `_BTA_CALL` snapshot.
  - **Two hooks, both identity in BTA.** `_begin_attempt(attempt, inference_input)` runs at the start of every attempt. `_conclude_attempt(result) -> result` runs once, in the tail, after `_finalize_response`.
  - **MFI's `_ainfer` override is deleted** (`FLOW/multi_flow_inferencer.py:1883-1899`). What it runs before BTA's `_ainfer` (`_apply_runtime_input_propagation`, `_reset_cross_flow_state`) becomes its `_begin_attempt`, so a rerun still redoes them. What it runs after (`_normalize_aggregator_output` → `_extract_dispatch_state` → `_maybe_strip_response`) becomes its `_conclude_attempt`, so it runs before the summary is put. MFI's `_infer` (`:1901-1920`) keeps only the `_coordination_enabled` guard, then returns `super()._infer(...)`.
  - **Rerun.** "rerun" continues the loop and skips the tail, just as today's `return await self._ainfer(...)` skips the outer run's `_emit_graph_reconcile` and `_finalize_response`. So the tail, and with it the summary, runs once, for the attempt whose result is returned.
  - **Sync `_infer` has exactly one attempt** (interactive review is async-only today). Its tail is today's `_finalize_response(result)` (`BTA.py:2210`), then `_conclude_attempt`, then the summary. Its attempt ledger closes through `run_async_joined` (§5.1).
  - **Behaviour delta (P0 golden, `BARE_EXPECTED_CHANGES`).** On an MFI interactive rerun today, the inner MFI `_ainfer` post-processes and returns, and the outer one post-processes the already-stripped string again. After P6 it runs once. `_normalize_aggregator_output` (`:1681`) is a no-op on a `str`, and `_extract_dispatch_state` (`:1539`) sets fields only on non-`None` parses. But `_maybe_strip_response` (`:1803`) applies the user-supplied `response_parser`, which need not be idempotent, so the single pass is the intended result. The P0 golden records today's double pass on a stub parser.
- **The call record and workspace snapshot [Idem, refined].** At the start of BTA's `_ainfer` / `_infer`, before the loop, BTA computes the snapshot once and puts it in `_BTA_CALL`:
  - `effective_workspace = self._workspace_under(ctx)`, today's precedence exactly (`AF/inferencer_base.py:838-857`);
  - `checkpoint_dir = self.checkpoint_dir` (`BTA.py:695`);
  - `checkpoint_root = effective_workspace.checkpoints_dir` if there is a workspace (`AF/inferencer_workspace.py:80`), else `checkpoint_dir`.

  Every BTA path derivation reads the snapshot instead of re-evaluating `self._workspace` under whichever ctx is active: `_get_result_path` (`:1695-1703`), `_load_promoted_breakdown` (`:1758-1789`), the breakdown, worker and aggregator node path lambdas (`:2185-2189`, `:2877-2889`, `:3053-3057`), the async and sync graph setup (`:2276-2280`), `_finalize_output` (`:1869`) and `_finalize_response` (`:2012`). Each site keeps its own rule unchanged: workspace-first then `checkpoint_dir`, or workspace-only. P0 lists every site with its rule, and the I7 goldens prove the paths are byte-identical. The lease and the manifest live at `checkpoint_root` (§5.11, §5.12), so `checkpoint_dir`-only runs are covered.
- **When a summary exists (B36).** Guardrails are leaf-only: attaching one to an orchestrator raises `ValueError` (`AF/inferencer_base.py:1486-1499`), and the judge is skipped on orchestrators (`:4297-4302`, `:4408-4413`). An orchestrator's recovery re-raises (`:4200-4201`). So the base retry helper stops at the first BTA run that returns, and nothing rejects it afterwards. The result reaches `_finalize_output` without a BTA run in only two cases, both after every attempt raised: an external `fallback_inferencer` (the chain `[_recovery_wrapper] + external_wrappers`, `:5117`), or a non-exception `default_return_or_raise` (`RPU/common_utils/async_utils.py:203-208`, `RPU/common_utils/function_helper.py:574-584`). The summary is put as the run's last statement and discarded when a run starts, so it is present exactly when the result came from a BTA run, and it belongs to that run. Without it, `_finalize_output` returns `super()._finalize_output(response)`. Today it ignores `response` and links the failed attempt's aggregator output, or raises U3b (`BTA.py:1841-1850`) and loses the fallback's result. No base acceptance hook is needed.
- **`BtaCallSummary`** (frozen, codec-registered in `RC/state.py`, §5.3), built in the run's tail from the returned attempt:
  - `worker_child_names` and `worker_output_roots`, ordered by worker index;
  - `aggregator_output_name` and `aggregator_workspace_root`;
  - `worker_count`, `selected_contract_index`, `disable_aggregator`;
  - `selected_contract: Optional[RenderedTaskContractState]`, the contract itself, so `_outcome_for` reads only the returned run's summary and never another attempt's `_BtaAttempt`.

  Its key declares `compat={"_last_call_summary": ""}` behind a read-only `last_call_summary` property, for the one true no-ctx reader (the fan-out, below). Cleanup status is not in the summary: the summary is frozen before the call-scoped close, so `cleanup_errors` lives on `NodeOutcomeState` and `InvocationCleanupError` (§5.1).
- **Who reaches the attempt, and how:**
  - Closures built per attempt, such as `_make_breakdown_fn` (`_bta = self`, `BTA.py:3095`), capture it directly.
  - Definition-level callbacks (the post-init registry lambdas, `BTA.py:942-955`, until P7 moves the registry into `_BtaGraph`) resolve it via `invocation_of(self).require(_BTA_ATTEMPT)`.
  - MFI's private super-call runs in MFI's frame, so it sees its own attempt.
- **Fields deleted from `self`:** `_cached_original_query`, `_promoted_breakdown_cache`, `_pending_topology`, `_graph_topology_emitted`, `_last_aggregation_guidance` (P6 commit 2), and `_worker_instances` (P6 commit 6, after its readers in finalize, `adisconnect`, the harvest and the fan-out have migrated: `BTA.py:1871`, `:2138-2146`, `:1003-1017`, `AF/inferencer_base.py:3342`).
  - `_worker_task_instructions` survives only as a declared compat field of the BTA contract key.
- **`use_async` is per attempt.** `_BtaAttempt.use_async` is `True` on the async path and the definition's value on the sync path. The node-function builders read it (`_build_subgraph_spec` `BTA.py:2446` → worker function `:2847` and aggregator factory `:3038`; `_make_breakdown_fn` `:3098-3100`), and so does `_BtaGraph` (§5.8). This replaces the async path's temporary flip of `self.use_async` (`:2259-2260`, `:2309`), whose builders would otherwise build sync functions for an async graph.
- **Traversals see the call-scoped aggregator.** `_iter_child_slots` / `_iter_child_inferencers` yield `frame_for(self).get(_BTA_AGGREGATOR).inferencer` when present, else the slot. Today the slot write in `build_aggregator` is what lets `pre_retry` archive the built aggregator's workspace between retries (`AF/inferencer_base.py:1855-1900`, `:1902-1960`); once the slot holds a factory again, the traversal must supply the built stage.
- **Child workspace rebinding.** `_bind_rebuilt_child_ws` writes a backing workspace onto the child instance, and its docstring says it is "Safe ONLY for non-shared nodes" (`AF/inferencer_base.py:935-956`), yet BTA calls it on borrowed stages too (`BTA.py:1205-1215`, `:2524-2534`, `:3008-3017`; base `:3307-3310`). It stays for owned stages (the object is the run). For a borrowed stage it becomes a publication to the stage's child ctx in host mode; legacy and no-ctx keep the setter. P6 commit 5.
- **Every path site reads the snapshot,** including `_promote_child_checkpoints` (`AF/inferencer_base.py:2378-2440`), which re-evaluates the ambient workspace today.
- **`ResolvedStage(inferencer, ownership: "owned" | "borrowed")`:**

| Stage | Slot value | Ownership | Closed by |
|---|---|---|---|
| worker, aggregator | factory (`LazyConfigFactory`, `_FreshCloneFactory`, callables) | owned | the attempt ledger (workers) or the call ledger (aggregator) |
| worker, aggregator | instance (including static round-robin lists, `BTA.py:2470`) | borrowed | never by the call; by the definition's `adisconnect` |
| breakdown | always an instance (`BTA.py:600`; no factory path) | borrowed | never by the call; by the definition's `adisconnect` (`:1003-1017`) |

- **Resolution** is one pure module function, `resolve_stage(slot_value) -> ResolvedStage`: a callable that isn't an `InferencerBase` is called and the product is owned; anything else is borrowed. It is today's `build_aggregator` body (`BTA.py:1987-1994`) without the slot write.
- **Per-call values into stages.** For every stage kind, per-call values (stream observer, interactive, a nested BTA's node name) travel as declared invocation keywords into the stage's own entry (§5.9), never as writes. The breakdown's per-call workspace (`_configure_for_workspace`, `BTA.py:2057-2062`) becomes a publication to the `breakdown` child ctx under a host ctx (§5.13 step 2); legacy and no-ctx modes keep the setter.
- **Duck-typed stages** (not `InferencerBase`, e.g. `mock_inferencers/mock_bta_components.py`) have no entry that could consume invocation keywords, and no frame, so the single-flight guard can't see them. They keep today's documented, `hasattr`-gated attribute protocol (`stream_observer`, `interactive`), with a scoped I2 exemption. In host mode BTA refuses to dispatch one borrowed duck-typed instance to two concurrently runnable workers, raising `UncertifiedConcurrentUseError` at graph build. They are never certified.
- **Stage rules:**
  - A factory that returns the same live object twice in one attempt raises `StageOwnershipError`.
  - The ledger calls `adisconnect` on an owned stage only if it has one, matching today's `hasattr` guard (`BTA.py:1011`).
  - Shared-instance safety for `InferencerBase` stages is the single-flight guard (§5.1), from P6, for every role.
- **Cleanup** follows §5.1: every close is attempted; an in-flight exception wins and gets the failures as a note; after success the outcome is published with `cleanup_errors`, then `InvocationCleanupError` is raised with the result.
  - Workers close at attempt end; the aggregator (and whatever a fan-out call created) closes at call end. Loop-bound clients are closed inside their own loop by the leaf (§5.1); the sync ledger's close goes through `run_async_joined(ledger.aclose())`, so nothing waits for a later `adisconnect`. This fixes B25 on both paths.
  - `adisconnect` stays idempotent and covers borrowed and definition children.
- **Aggregator (B2):**
  - `build_aggregator()` becomes `resolve_stage(self.aggregator_inferencer)`, stored with `frame.get_or_create(_BTA_AGGREGATOR)`. The slot is never written.
  - **Readers.** P0 classifies every `.aggregator_inferencer` hit in `AF/` (35 today: 28 in `BTA.py`, 4 in MFI, 3 in the base fan-out) into one of two kinds:
    - *needs the resolved stage*: switches to `invocation_of(self).require(_BTA_AGGREGATOR).inferencer`. [Dec] lists `:1242`, `:1359-1363`, `:1955`, `:2023`, `:2072`, `:2083`, `:2139`, `:2920`, `:3010`;
    - *presence check* (`not disable_aggregator and aggregator_inferencer is not None`): stays on the slot, which is definition-level and correct for a factory too. Examples: `:3207-3208`, `:3346-3347`, `:3427-3428`, `:3459-3460`, which [Dec] omitted.
  - All *needs-the-stage* readers switch in one commit. `:888` (the post-init `output_path` default) stays definition-level.
  - **The fan-out today** (`AF/inferencer_base.py:3444-3469`): `_materialize_fanout` builds the fresh fanout, then calls the mutating `fanout.build_aggregator()` "before seeding", so `_validate_fanout_aggregator` (`:3409`) and `_seed_aggregator_feed` (`:3507-3522`) see the built instance. `_arun_fanout` closes the fanout with `adisconnect` in a `finally` (`:3288-3292`), and the sync `_run_fanout` never closes it (its docstring says so, `:3265-3266`).
  - **The fan-out after P6:**
    - `_materialize_fanout` resolves the aggregator slot value the fanout would receive (the blank-slot override, else the prototype's) with `resolve_stage`, and passes the instance in the same `proto.fresh_instance(..., aggregator_inferencer=stage.inferencer)` call. Validation and seeding apply to that instance, unchanged. Seeding writes into an object this call just built, which is allowed: the object is the run.
    - The fanout's own call sees an instance slot, so it borrows the aggregator and doesn't close it. The caller's invocation ledger closes exactly what the fan-out call created: the resolved aggregator when it is owned (a factory product). The fanout's owned workers close in the fanout's own invocation, at its attempt end. Both paths get this: async replaces the explicit `try/finally: await fanout.adisconnect()`, and sync ends today's leak (B25).
    - The fanout object itself is not registered by default, because BTA `adisconnect` closes every child, borrowed ones included (`BTA.py:1003-1017`). If `fresh_instance` (`:1462-1483`) shares instance-valued recipe entries and overrides by reference, closing the fanout would close the prototype's configured breakdown and workers, and the owned aggregator a second time. P0 checks this. Only if `fresh_instance` copies those entries (so the copies belong to this call) does the caller also register the fanout.
    - `_conclude_fanout` reads `worker_count` through `_summary_at(fanout, child_rc)`, which mirrors `_task_contract_at`: the typed outcome under a ctx; in true no-ctx (`child_rc is None`), the fresh fanout's `last_call_summary`. That fanout is per-call, so its last-call getter can't be stale.
- **Q3:** a foreign `worker_inferencers` in a fan-out template → warn and override (unchanged).

### 5.8 `_BtaGraph`: per-attempt composition with byte-identical logs and checkpoints [Dec]

```python
@attrs(slots=False)
class _BtaGraph(WorkGraph):                        # module-private; one per attempt
    owner: Any = attrib(kw_only=True)
    start_nodes = attrib(factory=list)             # mirrors BTA.py:792, so it can be built empty
    def log(self, *a, **kw):              return self.owner.log(*a, **kw)
    def _get_result_path(self, *a, **kw): return self.owner._get_result_path(*a, **kw)
```

- **Log forwarding is load-bearing.**
  - The engine logs through `self` (`RPU/common_objects/workflow/workgraph.py:957`, `:1006`, `:1338`, `:1437-1442`).
  - `Debuggable.log` writes only via the object's own logger (`RPU/common_objects/debuggable.py:756-960`).
  - Forwarding keeps every engine record in BTA's session log, with BTA's id and name, including `_log_path_override` rebasing (`AF/inferencer_base.py:2105-2129`).
- **The graph-level result path** stays identical: `workgraph.py:2957-2963` uses `self.name`, which is set to `owner.name`. Per-node paths are lambdas on the nodes (`BTA.py:2183-2195`, `:2871-2890`, `:3051-3062`) and don't move.
- **Build order mirrors today,** which preserves start-node parenting (`workgraph.py:2005-2011`):
  1. construct empty, copying **every** `attrs.fields(WorkGraph)` field from the owner except the per-attempt ones, which are set explicitly: `name` (the owner's, or `bta_node_name` from P7), `use_async` (from `_BtaAttempt`, §5.7), `max_expansion_depth=1`, `max_total_nodes`, the per-attempt `subgraph_registry`, and `start_nodes` (empty). The engine reads many fields through `self` (`node_cls`, `verbose_repr`, `result_pass_down_mode`, `unpack_single_result`, `ignore_stop_flag_from_saved_results`, `executor`, besides `enable_result_save`, `resume_with_saved_results`, `checkpoint_mode` and the concurrency limits), so a hand-picked list would silently give the attempt graph `WorkGraph` defaults for whatever it misses. A coverage test fails on any `WorkGraph` field that is neither copied nor in the per-attempt set, so a new engine field can't be dropped silently;
  2. set `start_nodes = [breakdown_node]`;
  3. call `_propagate_expansion_settings()`.
- **Readers switch to the attempt graph:**
  - the resume check `start_nodes[0].next` (`BTA.py:3270-3273`, sync `:3446`), which fixes B24;
  - `GraphTopologyEvent.from_work_graph` (`AF/graph_events.py:54-79`; `BTA.py:2316-2326`);
  - `_all_nodes` (`:1814`);
  - `set_graph_event_callback` (`:2304`).
- **`self` is no longer written:**
  - the async path's temporary `use_async` flip, saved and restored around the graph run (`BTA.py:2259-2260`, `:2309`), a definition write visible to any concurrent reader;
  - `start_nodes`, `max_expansion_depth` and `max_total_nodes` (`:2196-2200`).

  All of them move onto the attempt's `_BtaGraph`.
- **Paths come from the call record:** `_BtaGraph._get_result_path` forwards to the owner, which reads `_BTA_CALL` (§5.7), never the ambient `self._workspace`.
- **Staging:** BTA keeps subclassing `WorkGraph` until P11.

### 5.9 Definition channels: feed, modes, role, observer, judge

- **Precedence, highest first [Sep]:**
  1. explicit per-call `extra_feed`;
  2. the nearest ctx publication (walk-up, stopping at the scope barrier);
  3. the `RoleState` overlay;
  4. the child definition.
- **Feed and modes (B17).**
  - Under any ctx, `_propagate_to_children` (`AF/templated_inferencer_base.py:531-579`; called at `AF/inferencer_base.py:2822-2832`, `:4903-4913`) *publishes* at the parent's own node, via `publish_child_template_feed` (`AF/template_feed_scope.py:68-118`) and a new `TEMPLATE_MODES_OVERRIDE_HANDLE`. It no longer merges into child instances. `_effective_modes()` resolves the ladder.
  - No-ctx setup keeps the explicit push as a setup-time API.
- **Role (B16).** `_effective_role_state()` returns the full overlay the no-ctx path applies (the `:636-657` block): key, root, version, master version, `template_variables`, extra feed and modes. Today `_effective_role` (`:479-494`) honours only key, root and version.
  - `RoleState` (`RC/state.py:204-215`) has typed fields only for role, key, root, version and modes; the master version, `template_variables` and extra feed reach it only inside the free-form `changes` dict (`AF/templated_inferencer_base.py:660-678`). B16 adds typed fields for all seven attributes `switch_role` sets (`:605-658`). Old stores migrate on load: `from_json` lifts the known keys out of `changes`.
  - **B16b (confirmed):** `_effective_role` sets `master = state.template_version` (`:492-493`), so the role's template *version* lands in the *master* slot. It is fixed by the same typed fields: each `RoleState` field maps to its own slot.
- **Role audit.**
  - `_pending_role_changes` becomes a keyword-only `_role_changes: RoleTransition` argument.
  - The host audit goes to bounded `node.provenance`. `_role_history` and `_applied_role` are written in no-ctx and legacy modes only.
  - A workspace argument under a host ctx is published to the role ctx.
- **Per-call values a parent assigns to a stage: invocation keywords (B18, B31).**
  - **Today.** BTA writes per-call values onto its stages, borrowed ones included:
    - `worker.stream_observer` and `worker.interactive` on leaf workers (`BTA.py:2803-2815`);
    - the same on the breakdown and aggregator (`:3139-3147`, `:3026-3034`);
    - `worker.name` on nested BTA workers (`:2552-2554`, cleared at `:2799-2800`; B31).

    The observer write *overwrites* a configured value whenever a reporter exists, so per-call wins today.
  - **Declaration.** A class declares the per-call values it accepts, `_INVOCATION_KEYWORDS: ClassVar[Mapping[str, RuntimeKey]]`, merged along the MRO:
    - `StreamingInferencerBase`: `stream_observer`;
    - the classes that read `interactive` (BTA, PTI, Conversational): `interactive`;
    - BTA: `bta_node_name`.
  - **Consumption.** `_(a)infer_single` and the streaming templates pop the declared keywords from the call kwargs into frame components before anything else, beside `extra_feed` and `prepared_input` (`AF/inferencer_base.py:2777-2779`, `:4861-4863`). A value is stored only when it was passed: a sentinel keeps "passed `None`" apart from "not passed".
  - **Resolution.** `_effective(name)` returns the component when passed, else the configured instance value. For the observer, the reader at `AF/streaming_inferencer_base.py:1039` switches to it. This is the precedence today's overwrite produces, and the one Conversational already uses for `interactive` ("prefer per-call arg, fallback to self.interactive", `conversational_inferencer.py:548-549`; `flow_node_adapter.py:283-285`).
  - **Dispatch.** BTA passes a keyword only to a stage whose class declares it. This replaces today's `hasattr` probing with a declared protocol, and it stops the dead ad-hoc `interactive` attribute that BTA now creates on leaves that never read it.
    - Duck-typed stages keep the attribute protocol (§5.7).
    - Owned per-attempt workers also receive keywords. Writes to them would be allowed (the object is the run), but one mechanism for every stage is simpler and keeps I2 trivially true.
  - **B31.** A nested BTA's `_bta_prefix` (`BTA.py:2445`) and its `_BtaGraph.name` (which feeds the graph-level result path, `RPU/common_objects/workflow/workgraph.py:2957-2963`) read `_effective("bta_node_name")`. "Passed `None`" reproduces today's reporter-case clear (`:2799-2800`); "not passed" falls back to `self.name`. It lands in P7 with `_BtaGraph`; until then the `worker.name` write stays `KNOWN_DEBT` for BTA.
- **Judge (B22).**
  - Delete `judge.template_manager = None` (`AF/inferencer_base.py:4526-4527`).
  - The judge calls (`:4305-4308`, `:4416-4419`) pass `prepared_input=True`, and apply the judge's `input_preprocessor` explicitly first when one is configured.

### 5.10 Purity tool (B8) [Snappy + Dec]

`RC/purity.py` changes:
- report added, changed **and removed** keys;
- a recursive fingerprint with cycle detection;
- identity first, then value;
- class normalizers for loggers, locks and handle stores;
- `assert_pure(obj, before)` takes a baseline;
- import `Iterable`.

**Negative tests (I10):** an added key, a removed key, and a nested mutation inside a list or dict must each fail the gate.

### 5.11 Checkpoint-root lease [Idem/Sep/Dec]

- **What:** a non-blocking exclusive advisory lock on `<checkpoint_root>/.bta_execution.lock`, where `checkpoint_root` is the call record's frozen root (§5.7): the effective workspace's `checkpoints_dir`, else the configured `checkpoint_dir`. So `checkpoint_dir`-only runs are covered. A second holder gets `BtaWorkspaceBusyError`.
- **When:** acquired right after `_BTA_CALL` is built, before any manifest, checkpoint or model I/O, and held for the whole logical call: every attempt, finalize, and the call-scoped close. It is registered first in the frame ledger, so it is released last (reverse order), after the aggregator and any fan-out have closed.
- **No root:** a BTA with neither a workspace nor a `checkpoint_dir` persists nothing, so it takes no lease.
- **Safety:** process death releases it, and the path is validated (never a repo root or `$HOME`).
- **Implementation:** a small RPU helper extracted from `RPU/service_utils/queue_service/storage_based_queue_service.py:205-254` (`fcntl` / `msvcrt`).

### 5.12 Resume identity manifest: two-phase, fail-closed [Idem + Sep + Dec]

**Protocol [Idem]**, all under the lease, all files at `checkpoint_root`, each written atomically (temp file, `fsync`, `os.replace`):
1. **Freeze the root.** `checkpoint_root` comes from `_BTA_CALL` (§5.7) and never changes during the call.
2. **Lease** (§5.11), before any checkpoint is read or any model is invoked.
3. **Identity header**, written on a fresh run or validated on resume, before any old artifact is read:
   - schema and checkpoint versions;
   - the input's type, length and canonical digest;
   - the semantic BTA definition identity (below);
   - the digest of the relevant invocation arguments.
4. **Committed effective plan**, written after parsing, truncation, interactive selection, todo expansion and heterogeneous dispatch, and before any worker result can be persisted. It holds the **ordered, reconstructable worker specifications**: for each worker, its query, arguments, task type, node name and stage identity, or an immutable artifact reference plus its digest. Digests alone can't replay a plan, so they only accompany the specs. **The commit point is the return of `_build_subgraph_spec`** (`BTA.py:2369+`): it is the one function that turns the selected sub-queries into the worker list (dispatch `:2398-2418`, todo expansion `:2419-2430`, nodes `:2845-2875`), and it runs before any worker node exists.
5. **Resume reconstructs from the committed plan**, not from the raw promoted breakdown. Today the registry lambdas (`BTA.py:942-955`) call `_build_subgraph_spec(self._load_promoted_breakdown()[0], …)`, and promotion (`:3189-3195`) happens before `_select_sub_queries` (`:3211+`), so a resumed run rebuilds the *unselected* worker list and loses interactive selection and truncation (B35). From P8 the lambdas rebuild from the committed plan.

**Canonical identity** (one pure function in `RC/`, used for the input, the definition and the arguments):

| Value | Encoding |
|---|---|
| `str` | UTF-8 bytes |
| `bytes` | raw bytes |
| JSON-compatible values | canonical JSON: sorted keys, no whitespace, UTF-8 |
| an inferencer | qualified class name plus an explicit, secret-free semantic identity: its config fields minus scheduling-only settings (concurrency) and minus fields marked secret, each value encoded by these same rules |
| an opaque factory or callable | a stable identity the factory supplies (`resume_identity`); none → error |
| anything else | `ResumeIdentityUnavailableError`, naming the value's type and path |

`repr()`, memory addresses, clients, tokens and secrets are never hashed. A BTA whose identity can't be computed fails before any I/O with that error; it can still run fresh with `enable_result_save` off.

**Policy:**

| Situation | Result |
|---|---|
| no artifacts | fresh run |
| header matches | resume, by the state table below |
| mismatch | `BtaResumeIdentityMismatch` |
| corrupt | `BtaResumeCorruptionError` |
| legacy artifacts without a manifest | `UnverifiedLegacyBtaResumeError` |

**Resume state table** (header already validated):

| On disk at the crash | Resume action |
|---|---|
| header only, or a missing or malformed breakdown checkpoint | re-run the breakdown, then plan and commit |
| a valid breakdown, no committed plan | re-plan from that breakdown (selection, truncation, expansion), then commit |
| a committed plan, with zero or some worker results | rebuild the workers from the plan; skip the saved ones |
| worker results but no committed plan | `BtaResumeCorruptionError`: results can't exist before the commit point, so the tree was written by something else |
| the aggregator completed | finalize from the saved result |

- The opt-in `resume_identity_policy="trust_legacy"` is an emergency override: it resumes a legacy workspace, logs at WARNING on every use, and never writes a "verified" manifest.
- A certify helper writes a manifest only when identity is provable.
- Scheduling-only settings (concurrency) are excluded from identity.
- **Goldens:** the I7 compatibility goldens allow exactly two new files, the manifest and the lock file. Every pre-existing path and byte is unchanged.
- **Crash points** (P8 tests), each followed by a fresh-process resume: header only; breakdown saved; plan committed; some workers persisted; aggregator completed. Each resumes or fails exactly as the table says.

### 5.13 PTI/LWI workspace publication (B7) [Snappy + Dec]

**What the instance workspace setter does** (`AF/inferencer_base.py:859-906`), and the host-mode equivalent of each:

| Setter step | Host equivalent |
|---|---|
| (1) `_configure_for_workspace` (base `:1388`; BTA `:2057`) | audited per class: logger un-defer is ALLOWED; anything else moves to a frame component. BTA's write into its borrowed breakdown stage (`self.breakdown_inferencer._workspace = bd_ws`, `BTA.py:2062`) is really a step-(2) child publication: under a host ctx it is published to the `breakdown` child ctx instead |
| (2) `_propagate_workspace_to_children` (base `:1234`; LWI `:690`, MFI `:571`, Dual `:591`) | each child's derived workspace is published to that child's ctx, under the same slot names (`_dynamic_child_name`, attribute slots) |
| (3) `_DERIVED_FROM_WORKSPACE` pops | recomputed from the resolved workspace |

- **Pattern reused:** `FLOW/linear_workflow_inferencer.py:1269` and `_prepare_guardrail_judge` (`AF/inferencer_base.py:4529-4534`).
- **Legacy and no-ctx modes** keep the setter.
- **LWI iteration paths** are computed from the call-start workspace held in a frame component, which fixes B15 if P0 confirms it.

---

## 6. Invariants

These are proven by the P0 ratchet, `TEST/run_context/test_template_purity_ratchet.py`, plus goldens.

**Fixtures** (from `TEST/common/inferencers/_helpers/factories.py` and `mock_inferencer.py`): a templated leaf, a streaming stub leaf, Dual, MFI, MFDual, BTA, LWI (static and dynamic), PTI, and a fan-out P.

| ID | Invariant | Proof |
|---|---|---|
| I1 | **Host purity:** each instance's `__dict__` delta (after warming pure caches) ⊆ ALLOWED ∪ `KNOWN_DEBT[class]` | the fixed purity tool. `KNOWN_DEBT` is **measured**, each entry names its removal phase, and a stale entry **fails**. A class's `_HOST_PURE_CERTIFIED` ClassVar must be `True` exactly when its measured debt is empty; the ratchet fails in both directions. |
| I2 | Under a host ctx, no call writes into a definition-owned or borrowed object (a borrowed stage, a configured child, a judge). Objects the invocation itself created (owned stages, the fan-out's per-call BTA) may be initialized normally. The one recorded exemption is the documented attribute protocol of duck-typed stages (§5.7) | the ratchet snapshots direct children and borrowed stages |
| I3 | Invocation state is unique per call, not per path; no live object is serialized | overlapping same-instance tests; a `NodeRunState.to_json` scan |
| I4 | No two overlapping public invocations share a `(store, path)`; in host mode no uncertified instance runs two overlapping invocations; and (after P8) no two holders share a BTA `checkpoint_root`, across threads and processes. All are rejected before any I/O | claim, guard and lease tests |
| I5 | Parents read child results only from the exact child ctx | concurrent contract-chain test |
| I6 | Definition slots are never overwritten; every owned stage is closed exactly once, at its attempt's end or the call's end, on every exit, sync included; a cleanup failure is never silent | identity and ledger tests |
| I7 | BTA is byte-compatible (workspace tree, ordered session-log `log_type` sequence, event order, the node set re-executed on resume), and PTI/LWI locations (`ctx.path`, workspace root, `output_path`) are unchanged | goldens in pickle and jsonfy modes; location goldens |
| I8 | Bare byte-compatibility of documented behaviour: output, documented getters, and the node-path set equal the `69ea2d3762af` goldens except `BARE_EXPECTED_CHANGES = {key: bug_id}`; the host never touches compat fields | getter goldens; compat-isolation tests |
| I9 | Every public entry runs inside run and invocation; sync, async and streaming agree on context, role, output and cleanup | `test_streaming_entry_contract.py`, parity tests |
| I10 | The gate can't false-pass | negative tests (§5.10) |
| I11 | New-manifest resume is unchanged; old stores load (a missing `outcome`; a `RoleState` in `call` is migrated) | resume suites; old-store fixtures |

---

## 7. Per-class inventory (state → home → phase)

| Area | State (anchor) | Home after | Phase |
|---|---|---|---|
| base | guardrail window (`AF/inferencer_base.py:598`) | `_fs` | P1 |
| base | `_last_inference_input` (`:2831`, `:4912`; fallback read `:4564`) | deleted; `_fs["rendered_input"]` is primary (`:2955-2963`) | P1 |
| base | `_output_finalized` / `_complete_inference` (`:2214-2227`) | deleted (dead) | P1 |
| base | judge mutation (`:4526-4527`) | `prepared_input` plus preprocessor | P5 |
| base | `_role_history`, `_pending_role_changes` (`:1201-1214`) | provenance / `RoleTransition` argument | P5 |
| base | `_init_call_state` run by `infer` / `ainfer` after `enter_run`, before any claim (`:3901`, `:5267`); skipped by `parallel_infer` items (`:4064-4079`, `:5403-5417`), CLI entries and direct streaming; run once on the iterator object by `infer(iterator)` | seam step 5, inside the frame, once per invocation (B34) | P3 |
| base | lazy `infer(iterator)` keeps `_active_ctx` set until the returned generator finishes (`:3891-3915`; `iter_infer` `:3946-3978`) | ctx resolved and `exit_run` before return; `ctx_bound_gen(ctx, inner)` binds per `next()` (B29) | P3 |
| CLI leaves | session code around `_(a)infer_single` in the public `ainfer` / `infer`: pre-call resolution and `_apply_session_policy`, and **all** post-call code (claude_code CLI `:883-899`, `:950-957`; codex CLI `:653-670`, `:701-707`; kiro `:331-340`; rovodev `:879-893`; devmate `:1264-1333`) | `_prepare_call` / `_conclude_call`, inside the frame (B34) | P3 |
| rovodev | `_current_output_file` ContextVar set in the streaming body and cleared with `.set(None)`, no token (`EXT/rovodev/rovodev_cli_inferencer.py:61-64`, `:753`, `:801`; read `:466`, `:572`, `:617`) | frame component; readers fall back to `self.output_file` frameless (B29) | P3 |
| templated | `RoleState` in `node.call` (`AF/templated_inferencer_base.py:465-477`, `:660-678`) | `node.role_state` | P1 |
| templated | `_last_rendered_task_instructions` (`:167`, `:454`); also written by Dual's preview renders on its leaves (`FLOW/dual_inferencer.py:1552`, `:2039`) | `_RENDERED_CONTRACT` component (with `compat`) + outcome; the preview publish is a no-op under host | P4 |
| templated | feed/modes cascade (`:531-579`); role overlay (`:479-494`); master version, `template_variables` and extra feed only in `RoleState.changes` (`:660-678`) | ctx publication; `_effective_role_state()`; typed `RoleState` fields + load migration | P5 |
| streaming | raw `_session_id = None` in recovery (`AF/streaming_inferencer_base.py:1368`) | `active_session_id` | P1 |
| streaming | `active_session_id` reset semantics (`:580-640`) | `LiveHandleField` | P3 |
| streaming | Tier-3 branches keyed by `ctx.path` alone (`:515-560`); every root is `"/"` (`RC/context.py:78`) | keyed `(scope_id, path)`; `scope_id` assigned in `LiveHandleStore.__init__` (B32) | P3 |
| devmate | `_consecutive_error_count` per instance (`EXT/devmate/devmate_cli_inferencer.py:288`; policy `:1134-1150`; counted `:1281`, `:1332`) | `_session_scoped_get` / `_set` (helpers P3; devmate moves in its P10 commit) (B33) | P3 → P10 |
| BTA | `_worker_task_instructions` (`BTA.py:778`, harvest `:2649-2664`, sync `:2737-2753`) | `_BtaAttempt.worker_contracts` → outcome; declared `compat` field of the BTA contract key | P6 |
| BTA | path derivations re-evaluated under whichever ctx is active: `_get_result_path` (`:1695-1703`), `_load_promoted_breakdown` (`:1758-1789`), node path lambdas (`:2185-2189`, `:2877-2889`, `:3053-3057`), graph setup (`:2276-2280`), `_finalize_output` (`:1869`), `_finalize_response` (`:2012`), and the base `_promote_child_checkpoints` (`AF/inferencer_base.py:2378-2440`) | read the frozen `_BTA_CALL` snapshot (§5.7) | P6 |
| BTA | recursive interactive rerun `return await self._ainfer(...)` (`:2353-2363`), re-dispatched through MFI's `_ainfer` (`FLOW/multi_flow_inferencer.py:1883-1899`) | explicit attempt loop with `_begin_attempt`; the tail (`_emit_graph_reconcile`, `_finalize_response`, `_conclude_attempt`, summary) runs once, skipped on rerun | P6 |
| MFI | `_ainfer` / `_infer` overrides wrap BTA's run: propagation and cross-flow reset before, `_normalize_aggregator_output` → `_extract_dispatch_state` → `_maybe_strip_response` after (`FLOW/multi_flow_inferencer.py:1883-1920`) | `_begin_attempt` / `_conclude_attempt` overrides; `_ainfer` deleted; `_infer` keeps only the coordination guard (B36) | P6 |
| BTA | `self.use_async` read by the node-function builders (`_build_subgraph_spec` `:2446` → worker fn `:2847`, aggregator factory `:3038`; `_make_breakdown_fn` `:3098-3100`) | `_BtaAttempt.use_async` (the flip on `self` goes with `_BtaGraph` in P7) | P6 |
| base, BTA | `_bind_rebuilt_child_ws` writes a backing workspace onto the child, borrowed stages included (`AF/inferencer_base.py:935-956`; `BTA.py:1205-1215`, `:2524-2534`, `:3008-3017`; base `:3307-3310`) | owned stages: unchanged; borrowed: child-ctx publication in host mode, setter in bare modes | P6 |
| claude_code SDK | sync `_infer` drops a stale-loop client without disconnecting it (`EXT/claude_code/claude_code_sdk_inferencer.py:425-451`, `:516-520`) | per-call client closed inside its own `_run_async` loop (codex's `_run_and_close`, `EXT/codex/codex_sdk_inferencer.py:360-406`) | P6 |
| BTA | `_last_aggregation_guidance` (`:786`, `:1390-1398`, `:1429`, `:1477`) | `_BtaAttempt` | P6 |
| BTA | `_cached_original_query`, `_promoted_breakdown_cache` (`:1782`, `:1802`, resets `:2161-2168`) | `_BtaAttempt` (captured) | P6 |
| BTA | `_graph_topology_emitted`, `_pending_topology` (`:771`; `:1626-1641`; `:2314-2348`; `:3288-3340`) | `_BtaAttempt` | P6 |
| BTA | `_worker_instances` (`:2443-2444`, `:2551`; finalize `:1871`, `:2138-2146`; `adisconnect` `:1003-1017`) | stages + ledger + summary | P6 |
| BTA | built aggregator (`:1987-1994`) | `_BTA_AGGREGATOR`; `resolve_stage` | P6 |
| BTA | observer/interactive onto leaf workers, breakdown and aggregator (`:2803-2815`, `:3139-3147`, `:3026-3034`) | invocation keywords | P5 |
| BTA | `worker.name` on nested BTA workers (`:2552-2554`, `:2799-2800`) | `bta_node_name` invocation keyword (B31) | P7 |
| BTA | breakdown `_workspace` (`:2062`) | `breakdown` child ctx publication | P9 |
| base fan-out | per-call fanout closed only on the async path (`AF/inferencer_base.py:3265-3266`, `:3288-3292`); `n_subtasks` from `_worker_instances` (`:3342`) | caller's invocation ledger; `_summary_at` | P6 |
| BTA | graph state: `start_nodes` and expansion limits (`:2196-2200`); the async path's `use_async` flip and restore (`:2259-2260`, `:2309`) | `_BtaGraph` | P7 |
| MFI | `predefined_sub_queries` rewrite in no-ctx mode (`FLOW/multi_flow_inferencer.py:1881`) | `_BtaAttempt.effective_sub_queries` | P9 |
| MFI | dispatch `_last_*`, `_role_get` / `_role_set` | unchanged (ctx-backed in host; legacy mirror ALLOWED) | — |
| Dual | `_last_iteration_record`; `_current_config` (`FLOW/dual_inferencer.py:1013`, dead) | frame component; deleted | P9; P1 |
| Dual | `_run_get` / `_run_set` (`:405-435`) | unchanged (already ctx-routed) | O1 |
| PTI | `_current_*` (`FLOW/plan_then_implement_inferencer.py:417-420`, writes `:1814-2659`); `self._workspace` (`:2654`, `:2656`); `child._workspace` (`:2435`); child save/resume flags (`:1504-1558`, `:2644-2682`); dead `_partial_iteration_history` (`:2579`) and `_setup_iteration_children` (`:2411-2437`) | components; ctx publication; atomic subphase; deleted | P9; P1 (dead code) |
| LWI | `self._workspace = ws` (`FLOW/linear_workflow_inferencer.py:911-918`); `_result_root_override` (`:941-961`) | ctx publication; derived from the child ctx | P9 |
| LWI | `_step_was_previously_attempted`, `_previous_attempt_info` (`FLOW/linear_workflow_inferencer.py:630-657`), written through properties by the RPU workflow engine (`RPU/common_objects/workflow/workflow.py:1185-1186`, `:1199-1200`, `:1384-1385`, `:1405-1406`, `:1539-1540`, `:1553-1554`, `:1742-1743`, `:1768-1769`), read by PTI (`FLOW/plan_then_implement_inferencer.py:564`, `:2144-2145`) | a frame component behind the same properties; direct-hook tests wrapped in `open_invocation` | P9 |
| LWI | `_state`, `_runner_scratch`, `_pending_state` (`:236-367`) | unchanged (already ctx-scoped) | — |
| terminal base | transport → `_ainfer` handoff through `_last_streaming_output`, `_last_streaming_stderr` and `_last_streaming_return_code`. Declared at `AF/terminal_inferencers/terminal_inferencer_base.py:70-71` and `terminal_session_inferencer_base.py:92`. Written at TIB `:408-457`, TSIB `:440-443`, `:570-609`, codex CLI `:566`, claude_code `:716`. Read by TSIB `_ainfer` `:485-487`, devmate `:927`, and the `get_streaming_result` getters (devmate `:1664-1691`, claude_code `:1066-1073`) | a `TerminalStreamResult(stdout, stderr, return_code)` component whose key declares the three fields as `compat` (B28) | P10 (first) |
| tool_as | `_last_response` (`AF/agentic_inferencers/tool_inferencers/tool_as_inferencer.py:156`, written `:372`, read in `_ainfer` `:395-430`) | component via `publish_result` / `read_result` (B28 family); `compat` only if P0 finds a documented getter | P10 |
| devmate | positional `inference_config` → `filter_session_info` (`EXT/devmate/devmate_cli_inferencer.py:1335-1339`, reached through base `:1041`) | explicit keyword on the pipeline (B27) | P2 |
| devmate, claude_code | public sync `infer_streaming` holding entry policy (devmate `:1499-1560`; claude_code `EXT/claude_code/claude_code_cli_inferencer.py:1000-1049`) | `_infer_streaming_pipeline` overrides under the sync template (B26, B26b) | P2 |
| OpenClaw | `ainfer` bypasses `_ainfer_single` (`EXT/openclaw/openclaw_inferencer.py:980-1031`); `_maybe_initialize_session` runs before inference (`:1181-1206`); base `max_retry` / `fallback_mode` / `fallback_inferencer` silently ignored | converged entry; initialization inside `_ainfer`; post-init warn-and-override of the base retry settings (B30) | P2 |
| streaming base | `_active_ctx` held across yields (`AF/streaming_inferencer_base.py:758-786`); the sync bridge's daemon thread outlives an abandoned consumer (`:1102`) | `framed_agen` / `framed_gen`; cancel+join (B29) | P3 |
| leaves | raw `_session_id` writes: codex CLI `:536`, codex SDK `:258`, `:316`, devmate SDK `:407`, metamate `:415`, rovochat `:468` | `LiveHandleField` | P10 |
| leaves | `_last_stream_result` (claude_code `:591`, `:701`, `:886-891`; codex CLI `:470`, `:527-551`, `:660-662`); `_last_token_count` (rovochat, metamate SDK, devmate SDK); codex SDK `_last_usage` / `_last_tool_use_count`; claude_code SDK `_last_tool_use_count` | components via `publish_result` / `read_result` (B28 family); `compat` where P0 finds a documented getter | P10 |
| leaves | metamate `_conversation_uuid` / `_fbid` (`:413-414`), rovochat `_conversation_id` | Tier-3 fields | P10 |
| leaves | OpenClaw `_session_initialized` (`EXT/openclaw/openclaw_inferencer.py:203`, `:1193-1206`, `:1258-1284`) | Tier-3 set keyed by `session_id` | P10 |
| leaves | rovodev `_last_clean_output`, `_last_raw_stdout` (`EXT/rovodev/rovodev_cli_inferencer.py:582`, `:624`, `:709-715`, `:783-803`; read `:656-666`, `:817`) | component whose key declares both fields as `compat` (B28 family, B23) | P10 |
| Conversational | reads the leaf's `get_final_output()` after streaming (`conversational_inferencer.py:909-913`, `:938-941`) | `_final_output_at(child, child_ctx)` over `NodeOutcomeState.final_output`; the getter only in true no-ctx | P10 (with rovodev) |
| leaves | devmate `dump_output` flip (`EXT/devmate/devmate_cli_inferencer.py:1367-1374`, `:1552-1557`) | a parameter to the builder (read at `:769`) | P1 |

---

## 8. Bug register

- Each bug gets one commit and a regression test that fails on the parent commit.
- "Dec/Harm #" is the other plans' numbering; they agree for B1–B19.

| # | Bug (anchor) | Failure scenario | Fix | Phase | Dec/Harm # |
|---|---|---|---|---|---|
| B1 | `_worker_task_instructions` is never reset, and only the async path harvests it (`BTA.py:778`, `:2649-2664` vs sync `:2737-2753`) | a reused BTA: task B's reviewer gets A's contract; sync `infer` yields `""` | per-attempt `worker_contracts`, harvested on both paths, published through `_outcome_for` | P6 | B1 |
| B2 | `build_aggregator` writes the definition slot (`:1987-1994`, my commit `4e67d970404c`) | the next call reuses call 1's aggregator, with its feed and session | pure resolver + `_BTA_AGGREGATOR` | P6 | B2 |
| B3 | the resume restore of guidance runs only if the value is None (`:1390-1398`); the parse reset (`:1429`) doesn't run on resume | call 2 resumes, but the aggregator gets call 1's guidance | `_BtaAttempt.aggregation_guidance` | P6 | B3 |
| B4 | streaming recovery writes `self._session_id = None` (`AF/streaming_inferencer_base.py:1368`) | host: the dead session stays in the branch slot, and the shared backing is wiped for everyone | `self.active_session_id = None` | P1 | B4 |
| B5 | the guardrail window persists across calls (`AF/inferencer_base.py:598-603`, `:4229-4272`) | a new call's first empty output trips fail-fast | `_fs` | P1 | B5 |
| B6 | session reset doesn't reset: (a) host falls back to the backing (`:593-601`); (b) no-ctx returns the single live branch (`:612-624`); six raw writers | a sibling branch resumes another branch's conversation | `LiveHandleField` (P3); leaves (P10) | P3, P10 | B6 |
| B7 | PTI/LWI per-call workspace and `_current_*` writes under a host ctx | two host branches overwrite each other's iteration workspace | §5.13 | P9 | B7 |
| B8 | purity tool: removals invisible; `assert_pure` always pure; deepcopy false changes; missing import (`RC/purity.py:18`, `:30-39`, `:42-54`, `:59`, `:77-79`) | the gates miss deletions and over-report | §5.10 | P0 | B8 |
| B9 | dead surface: `_complete_inference` / `_output_finalized`, `_deliverables_copied` (`BTA.py:2033`), Dual `_current_config`, PTI `_partial_iteration_history` / `_setup_iteration_children`, `_last_inference_input` | latent per-instance flags | delete (reader audit first) | P1 | B9 |
| B10 | the templated snapshot is last-call-on-object (`AF/templated_inferencer_base.py:167`, `:454`); Dual's preview renders (`FLOW/dual_inferencer.py:1552`, `:2039`) overwrite it too | one leaf in two branches: the parent reads whichever rendered last, or a preview | outcome channel; frameless host publish is a no-op | P4 | B10 |
| B11 | devmate flips its definition field `dump_output` (`EXT/devmate/devmate_cli_inferencer.py:1367-1374`, `:1552-1557`) | concurrent calls see each other's flip | parameter | P1 | B11 |
| B12 | `RoleState` in `node.call` (`AF/templated_inferencer_base.py:476-477`, `:671`) collides with typed call state (`AF/inferencer_base.py:3730-3733`; `FLOW/multi_flow_inferencer.py:493-497`) | a switch clobbers `MultiFlowState`, or suppresses it | `node.role_state` + one-way migration | P1 (first) | B12 |
| B13 | the BTA contract pick is first completion wins (`BTA.py:2649-2664`) | the published contract depends on timing | lowest successful index | P6 | B13 |
| B14 | the WorkGraph thread hop drops ContextVars (`RPU/common_objects/workflow/workgraph.py:2746-2751`) | sync `infer` from async code: workers legacy-mint and vanish from the host store | route the hop through the new RPU `run_async_joined`, which runs the coroutine on a worker thread under `contextvars.copy_context()` (§5.1) | P1 | B14 |
| B15? | suspected: a reused LWI nests under the previous call's last iteration dir (`FLOW/linear_workflow_inferencer.py:911-918`) | wrong checkpoint dirs on reuse | call-start workspace component | P0 probe → P9 | B15? |
| B16 | ctx role ignores `RoleState` master version, variables, feed and modes (`AF/templated_inferencer_base.py:479-494` vs the `:636-657` block); `RoleState` has no typed field for the master version, variables or feed, which reach it only via `changes` (`RC/state.py:204-215`; `:660-678`). B16b (confirmed): `_effective_role` puts the role's template *version* in the *master* slot (`master = state.template_version`, `:492-493`) | a role switch under a host ctx renders with the wrong template version or variables | typed `RoleState` fields + load migration; `_effective_role_state()` | P5 | B16 |
| B17 | parent feed/modes merged into child instances on every call (`:531-579`) | shared children accumulate other parents' feed | ctx publication | P5 | B17 |
| B18 | observer/interactive written into stages, borrowed ones included: leaf workers (`BTA.py:2803-2815`), breakdown (`:3139-3147`), aggregator (`:3026-3034`) | a borrowed stage streams into another call's observer, or keeps a finished call's observer; a PTI worker's configured `interactive` is overwritten | declared invocation keywords, per-call first (§5.9) | P5 | B19 |
| B19 | OpenClaw `_session_initialized` is per instance, not per session (`:1193-1206`, `:1258-1284`) | a second session skips its startup sequence | Tier-3 set keyed by `session_id` | P10 | — |
| B20 | MFI rewrites the definition's `predefined_sub_queries` in no-ctx mode (`FLOW/multi_flow_inferencer.py:1881`) | the next call inherits the previous runtime sub-queries | attempt-local `effective_sub_queries` | P9 | — |
| B21 | `_pending_topology` is undeclared, per-call state that is never reset (only `_graph_topology_emitted` is, `BTA.py:2171`, `:2255`). **Low severity:** `_emit_pending_graph_topology` clears it before awaiting (`:1641`), so it survives only if the reporter resolves differently between the check and the emit (`:1627`) | a stale topology is emitted to a later call's reporter | `_BtaAttempt` | P6 | — |
| B22 | the judge's `template_manager` is nulled permanently (`AF/inferencer_base.py:4526-4527`) | the judge definition loses its templates | §5.9 | P5 | B18 |
| B23 | rovodev keeps a stale `_last_clean_output`: its `finally` refills it only when empty (`EXT/rovodev/rovodev_cli_inferencer.py:787`), and `_ainfer` returns it (`:817-824`) | cache disabled and output file present: call 2 returns call 1's output | a component whose key declares both fields as `compat` (P10); split (P2) | P2, P10 | — |
| B24 | overlapping calls on one BTA share `self.start_nodes`, so B's breakdown sees A's workers (`BTA.py:3270-3273`) and returns raw sub-queries | silent wrong result | `_BtaGraph` per attempt | P7 | B20 |
| B25 | workers from earlier calls are never disconnected: the per-call reset drops `_worker_instances` (`BTA.py:2166`, `:2250`) and `adisconnect` sees only the last call's (`:1003-1017`). The sync fan-out never disconnects its per-call BTA (`AF/inferencer_base.py:3265-3266`) | subprocess or client leaks on reuse | owned stages close at attempt end (workers) or call end (aggregator) through the ledger, under loop affinity: SDK leaves close per-call clients inside their own `_run_async` loop (the claude_code SDK's stale-loop drop is replaced by codex's `_run_and_close` pattern), and the sync ledger closes through `run_async_joined(ledger.aclose())`; the caller's ledger closes what the fan-out call created (§5.7); a failed close after success raises `InvocationCleanupError` with the result (§5.1) | P6 | B21 |
| B26 [new] | (a) an explicit `run_context` is ignored by direct streaming. devmate async (`:1359-1409`) and sync (`:1499`), OpenClaw (`:1061-1112`) and claude_code sync (`EXT/claude_code/claude_code_cli_inferencer.py:1000-1049`) never `enter_run`, and `run_context` stays in `kwargs`. rovodev resolves the session (`:760-772`) before `super()` enters the ctx (`:776`). (b) The devmate and claude_code sync entries ignore a configured fan-out, unlike their async entries | (a) a direct host-ctx streaming call reads and writes the caller's session slot, not its branch's; claude_code forwards `run_context` into `construct_command`. (b) a fan-out-configured leaf silently runs locally on sync streaming | sync and async template-method split; entry policy only in the templates | P2 | — |
| B27 [new] | base `_ainfer` passes `(inference_input, inference_config, …)` positionally (`AF/streaming_inferencer_base.py:1041`), so devmate's `filter_session_info` receives `inference_config` (`EXT/devmate/devmate_cli_inferencer.py:1335-1339`) | `ainfer()` with a truthy `inference_config` silently filters session lines; with `None` it never filters, unlike direct streaming. The parse uses raw stdout, so the impact is limited to the accumulated fallback | explicit keyword: the pipeline defaults to `False` (today's `None` behaviour), the public adapter to `True` | P2 | — |
| B28 [new] | leaf transports hand call results to `_ainfer` through instance fields. Terminal: `_last_streaming_output`, `_last_streaming_stderr` and `_last_streaming_return_code` (`AF/terminal_inferencers/terminal_session_inferencer_base.py:440-443` → `:485-487`). The same pattern: `_last_stream_result` (claude_code, codex CLI), rovodev `_last_clean_output` / `_last_raw_stdout`, tool_as `_last_response` (`tool_as_inferencer.py:372` → `:395-430`), and the token/usage counters (§7) | two concurrent calls on one leaf (a borrowed leaf in a host fan-out): call A parses call B's stdout, return code, session id or tool response; the result is silently wrong | one component per family via `publish_result` / `read_result`; documented getters via the key's `compat` (§5.4) | P10 (terminal first) | — |
| B29 [new] | streaming entries hold ContextVars across yields. Base `ainfer_streaming` keeps `_active_ctx` set "across all yields" (`AF/streaming_inferencer_base.py:758-786`) inside a generator that runs in the consumer's task. The sync bridge's daemon thread (`:1102`) is not stopped when the consumer stops. Lazy `infer(iterator)` does the same: it defers `exit_run` into the returned generator's `finally` (`AF/inferencer_base.py:3891-3915`; `iter_infer` `:3946-3978`) | (a) between yields the consumer's own calls resolve the leaf's ctx; (b) after `async for … break`, the leaf's (possibly minted) root stays active in that task forever, later bare calls there join it (`RC/bridge.py:66-68`), and the finalizer's `exit_run` raises `ValueError` (reset in another Context); (c) an abandoned sync stream keeps its subprocess running; (d) the same as (a) and (b) for an abandoned lazy `infer(iterator)` | `framed_agen` / `framed_gen` bind per resumption and close inside the binding; the bridge cancels and joins on close; lazy `infer(iterator)` resolves the ctx and exits the run before returning, then binds per `next()` (§5.1) | P3 | — |
| B30 [new] | OpenClaw's `ainfer` bypasses `_ainfer_single` (`EXT/openclaw/openclaw_inferencer.py:980-1031`) and runs session initialization, which is model I/O, before inference (`:1181-1206`) | under v6 an OpenClaw call would run with no frame, claim or outcome, so I4 and I9 fail for it. Today, base per-call settings configured on it (timeouts, guardrail, preprocessing) are silently ignored by `ainfer` | converge onto `_ainfer_single`; initialization inside `_ainfer`; post-init warn-and-override of `max_retry` / `fallback_mode` / `fallback_inferencer` keeps exactly one base attempt (§5.2) | P2 | — |
| B31 [new] | BTA writes a per-call node name onto nested-BTA workers, borrowed ones included (`BTA.py:2552-2554`; cleared at `:2799-2800` when a reporter exists). The nested BTA's `_bta_prefix` (`:2445`) and its graph-level result path (`RPU/common_objects/workflow/workgraph.py:2957-2963`, via `self.name`) read it | two round-robin workers `i` and `j` sharing one nested BTA instance, no reporter: both names are written at graph-build time, so both run under `j`'s name. Worker `i`'s inner node ids and graph name carry `j`'s prefix, and where the two share a result root, their graph-level result files collide. A reused nested BTA also keeps a stale name. P0 adds a reproducer that confirms the exact effect before P7 | `bta_node_name` invocation keyword into the nested BTA's frame, read by `_bta_prefix` and `_BtaGraph.name` (§5.9) | P7 | — |
| B32 [Idem] | live handles are keyed by `ctx.path` alone in one instance-owned `LiveHandleStore` (`AF/streaming_inferencer_base.py:515-560`, `:580-640`), and every root has path `"/"` (`RC/context.py:78`) | one leaf shared by two independent host roots (two OpenTeam sessions, two requests): both read and write one session id and one SDK client, so conversations cross | branches keyed `(scope_id, path)`, `scope_id` assigned when a `LiveHandleStore` is constructed; legacy mint uses one fixed scope; per-turn-root callers that expect continuity pass their session's store (§5.5) | P3 | — |
| B33 [Idem] | devmate's `_consecutive_error_count` is per instance (`EXT/devmate/devmate_cli_inferencer.py:288`), yet it decides when to force a new session (`:1134-1150`; counted `:1281`, `:1332`) | host branch A fails N times; branch B's next call is forced onto a new session and loses its conversation | `_session_scoped_get` / `_set`, the same policy as `active_session_id` (§5.5) | P10 (helpers P3) | — |
| B34 [Idem] | per-call initialization runs only in `infer` / `ainfer`, after `enter_run` and outside any claim (`AF/inferencer_base.py:3888-3901`, `:5262-5268`). `parallel_infer` / `aparallel_infer` items call `_(a)infer_single` directly (`:4064-4079`, `:5403-5417`); CLI entries and direct streaming never reach it; `infer(iterator)` runs it once, with the iterator object as its input | a `parallel_infer` item or a CLI call starts with the previous call's typed call state (MFI dispatch state, reset at `FLOW/multi_flow_inferencer.py:1483-1491`); an iterator's `state_factory` sees the iterator, not the item. Under v6 a claim-rejected call would already have reset shared state | seam step 5: `_init_call_state` inside the frame, once per invocation, after the claim (§5.1). P0 adds a reproducer | P3 | — |
| B35 [new] | BTA resume rebuilds workers from the promoted breakdown, not from the selected worker list: the registry lambdas (`BTA.py:942-955`) call `_build_subgraph_spec(self._load_promoted_breakdown()[0], …)`, and promotion (`:3189-3195`) runs before `_select_sub_queries` (`:3211+`) | a run with interactive selection or truncation crashes after some workers finish; the resumed run rebuilds the unselected list, so it runs sub-queries the user dropped, and a worker index can pair with a saved result from a different sub-query. P0 adds a reproducer | the return of `_build_subgraph_spec` is the effective-plan commit point; from P8 the registry lambdas rebuild from the committed plan (§5.12) | P8 | — |
| B36 [new] | BTA `_finalize_output` (`BTA.py:1824-1886`) ignores `response` and finalizes whatever aggregator workspace `_read_child_workspace` resolves, raising U3b when there is none (`:1841-1850`) | every attempt raises; an external `fallback_inferencer` (`AF/inferencer_base.py:5117`) or a non-exception `default_return_or_raise` (`RPU/common_utils/async_utils.py:203-208`) supplies the result; finalize links the failed attempt's aggregator output as the canonical output, or raises U3b and loses the fallback's result | `_BTA_SUMMARY` discarded at run start and put as the run's last statement (after `_conclude_attempt`); no summary → `super()._finalize_output(response)` (§5.7) | P6 | — |

**`BARE_EXPECTED_CHANGES`** (bare-visible, each tagged):
- B1 (reuse), B3, B5, B6, B15 (if confirmed), B20, B23, B25;
- B35 (a resumed run keeps its selection and truncation; lands with P8);
- B36 (a BTA result produced by an external fallback or a non-exception `default_return_or_raise` is finalized through the base path instead of the failed attempt's aggregator output, or U3b);
- MFI interactive rerun: post-processing (`_normalize_aggregator_output`, `_extract_dispatch_state`, `_maybe_strip_response`) runs once instead of twice (§5.7; P0 golden);
- B25's cleanup raise: a close that fails after a successful call now raises `InvocationCleanupError` carrying the result, where today the failure surfaced only at a later `adisconnect` or never;
- B34 (`parallel_infer` items, CLI entries and direct streaming now initialize call state; an iterator's `state_factory` receives each item);
- B26 (explicit ctx honoured); B26b (sync fan-out honoured);
- B27 (only for callers that pass a truthy `inference_config` to devmate `ainfer`; P0 inventories them);
- B29 (the consumer no longer sees the leaf's ctx between yields; an abandoned sync stream is cancelled);
- B30 (only for OpenClaw configs that set base per-call settings; P0 inventories them);
- B18 (a stage's configured observer or `interactive` is no longer left overwritten after the call; leaves that don't declare `interactive` no longer get the attribute);
- B25's sync fan-out close (the per-call BTA is disconnected when the caller disconnects);
- B31 (a nested BTA keeps its own `name` after the call).

B32 and B33 are not bare-visible: legacy mint keeps one fixed handle scope and no-ctx keeps the instance backing, so bare continuity is unchanged. Compat fields need no entry either: they are flushed on every non-host close, success or not, which matches today's transports writing them during the call (§5.4).

---

## 9. Existing tests that change

| Test | Change | Phase |
|---|---|---|
| `TEST/run_context/test_m7_purity_gate.py:53`, `:117-127` | carve-outs updated (fixed tool; role audit) | P0, P5 |
| `TEST/common/inferencers/test_output_guardrail.py:575`, `:607`, `:617` | window in `_fs`; `_last_inference_input` removed | P1 |
| `test_resume_detection.py:1568`; `subgraph_registry` / `use_async` tests | read the attempt graph | P7 |
| `test_dual_inferencer/test_pti_resume.py:490` (presets `_current_iteration_workspace`) | `with open_invocation(pti) as f: f.put(_PTI_CURRENT, …)` | P9 |
| LWI tests and RPU workflow tests that set or read `_step_was_previously_attempted` / `_previous_attempt_info` directly (P0 lists them) | wrapped in `open_invocation(lwi)`, because the properties now route to a frame component | P9 |
| `test_task_instructions_snapshot.py` (`_FakeBTA` / `_FakeMFI`, `TestLeafCapturesItsOwnRender`) | **unchanged**: the compat fields keep today's names, and a frameless bare `_render_prompt` still writes them | — |
| tests that call a private hook directly (`_ainfer` / `_infer`; about 30 files, listed by P0) | unchanged while the hook only publishes; wrapped in `open_invocation(inst)` / `aopen_invocation(inst)` by the commit that first makes that class's hook read a component | P6, P9, P10 |
| tests that read `worker.name`, `worker.stream_observer` or `worker.interactive` after a BTA call (P0 lists them) | assert the value the stage's frame received | P5, P7 |
| `test_mfdual_workspace_anomalies_integration.py:231` | **unchanged** (no-ctx path) | — |
| `test_bta_inferencer_fanout.py:473-496` | pre-built aggregator; summary-driven `n_subtasks` | P6 |
| `test_bta_checkpoint_promotion.py:247-257`; `test_bta_resume_original_query.py:98-159`; `test_fresh_instance.py:216-225`, `:572-592`; `test_breakdown_block_registry.py`; `test_parse_json_subtasks.py`; `test_bta_resume_workspace_binding.py:168-178` | mailbox fields deleted → assert **observable** behaviour: graph events, checkpoint paths and bytes, outputs, and cleanup calls on stub stages. In particular: `test_bta_resume_original_query.py:124`, `:129` assert the query the resumed aggregator receives; `test_bta_resume_workspace_binding.py:173` asserts the checkpoint path written; `test_bta_checkpoint_promotion.py:252` asserts the promoted file's path and content; `test_fresh_instance.py:221`, `:223` assert the clone's outputs and that its stages were closed. Never the transient private `bta._graph` [Idem] | P6 |
| any test asserting a child's instance feed or modes after a parent call (P0 lists them) | assert the rendered feed | P5 |

---

## 10. Critical files and reused utilities

**Modified or new:**

| File | Changes |
|---|---|
| `RC/invocation.py`, `RC/outcome.py` | new: frame, keys, ledger, seam context managers, single-flight guard, contract errors; outcome helpers |
| `RC/resume_identity.py` | new: the canonical identity function and `ResumeIdentityUnavailableError` (§5.12, P8) |
| `RC/store.py`, `RC/state.py`, `RC/handles.py`, `RC/purity.py`, `RC/README.md` | `outcome`, `clear_outcome`, claims; `RC/state.py` gains `NodeOutcomeState`, `RenderedTaskContractState` and `BtaCallSummary`, exported from `RC/__init__` (§5.3); `LiveHandleStore.scope_id` (B32); B8; docs |
| `AF/inferencer_base.py` | the seam in `_(a)infer_single` (`_init_call_state` moved in, `_prepare_call` / `_conclude_call` hooks, invocation keywords); lazy `infer(iterator)` binding; `InvocationContractError` in both `non_retryable_exceptions` tuples; B5, B9; `_outcome_for`; lifted handle helpers; judge; role audit; the fan-out's ledger registration |
| `AF/templated_inferencer_base.py` | B12, B16, B17; render capture |
| `AF/streaming_inferencer_base.py` | template-method split; frame open; `LiveHandleField`, `_session_scoped_get` / `_set`, `(scope_id, path)` keying; B4 |
| `BTA.py`, `FLOW/multi_flow_inferencer.py`, `FLOW/dual_inferencer.py`, `FLOW/plan_then_implement_inferencer.py`, `FLOW/linear_workflow_inferencer.py` | per §7; BTA also gets `_BTA_CALL`, the attempt loop with `_begin_attempt` / `_conclude_attempt`, the summary-or-base `_finalize_output` (B36), and the lease and manifest (P8). MFI overrides both hooks; its `_ainfer` override is deleted and its `_infer` keeps only the coordination guard |
| `EXT/{rovodev,devmate,openclaw}`, `EXT/claude_code/claude_code_cli_inferencer.py` | split and OpenClaw convergence (P2); session policy into `_prepare_call` / `_conclude_call` (P3); leaves (P10) |
| `EXT/{codex,kiro}` CLI | thin `ainfer` / `infer` adapters; session policy into the hooks (P3) |
| per-turn-root callers: `OpenStartup/src/openteam/server/services/conversation_service.py:1262`, `AgentFoundation/src/agent_foundation/resources/tools/task/executor.py:670`, `:1046`, `resources/tools/sop/cli.py:254` | pass their session's handle store where they expect cross-turn continuity (B32; P0 decides each) |
| `EXT/{codex,claude_code,metamate,rovochat}`, `AF/agentic_inferencers/tool_inferencers/tool_as_inferencer.py` | leaves (P10); tool_as (P2) |
| `EXT/claude_code/claude_code_sdk_inferencer.py` | sync `_infer` closes its per-call client inside its own `_run_async` loop, replacing the stale-loop drop (loop affinity; P6 commit 5) |
| `AF/terminal_inferencers/terminal_inferencer_base.py`, `terminal_session_inferencer_base.py` | B28 handoff component (P10) |
| `RPU/common_objects/workflow/workgraph.py` | B14: the thread hop goes through `run_async_joined` (P1) |
| `RPU/common_utils/async_utils.py` | new `run_async_joined` beside `_run_async` (`:440`): `_run_async` with no running loop, else a joined worker thread under `copy_context()` (P1); new `iterate_async_in_thread` sync bridge helper (P3); each with its own tests |
| RPU lock helper (extracted from `storage_based_queue_service.py:205-254`) | lease (P8) |
| `TEST/run_context/BUCK` | new `python_pytest` target (P0) |
| `AgentFoundation/invocation_scoped_runtime.plan.md`, `AgentFoundation/invocation_scoped_runtime.P0_inventory.md` | new: the frozen plan revision and the P0 inventory, versioned as review evidence (P0) |

**New tests:**
- `test_template_purity_ratchet.py`, `goldens/`;
- `test_invocation_frame.py` (seam order, ledger, cleanup errors), `test_path_claims.py`, `test_single_flight_guard.py`, `test_outcome_state.py`, `test_compat_fields.py`, `test_live_handle_field.py` (scope keying, session-scoped helpers), `test_streaming_entry_contract.py`, `test_lazy_iterator_binding.py`;
- the BTA call-record, attempt-loop, ledger, summary and graph tests;
- the lease, canonical-identity and manifest crash-point tests;
- per-bug regression tests beside the existing suites.

**Reused, not re-invented:**

| Need | Existing utility |
|---|---|
| per-call ContextVar pattern | `bridge._active_ctx`, `enter_run` / `exit_run`; `_current_fallback_state` (`AF/inferencer_base.py:67`) |
| child ctx | `_rc_child` (`:3664`), `_with_child_ctx` (`:1849`) |
| workspace publish / read | `_publish_workspace_to_ctx` (`:908`), `_read_child_workspace`, `_workspace_under` (`:838`) |
| node claims and non-claiming reads | `RunStateStore.node` / `peek` |
| feed publication | `publish_child_template_feed` (`AF/template_feed_scope.py:68-118`) |
| Tier-3 | `LiveHandleStore` (`RC/handles.py`), `_tier3_get` / `_tier3_set` |
| fixtures | `TEST/common/inferencers/_helpers/factories.py`, `mock_inferencer.py` |
| lock | `RPU/service_utils/queue_service/storage_based_queue_service.py:205-254` |
| closing a loop-bound client inside its own loop | codex SDK `_run_and_close` (`EXT/codex/codex_sdk_inferencer.py:360-406`) |
| sync close of the ledger | `_run_async` (`RPU/common_utils/async_utils.py:440`), wrapped by the new `run_async_joined` for callers inside a running loop |
| BUCK template for a pytest suite | `TEST/knowledge/BUCK` |

---

## 11. Phases

**Conventions for every commit:**
- one concern, committed by explicit path;
- `arc f <paths>`, `arc lint -a <paths>` and `arc pyre check-owning-targets` on the explicitly listed touched files, never on the whole working copy, which holds unrelated changes (buck test runs no pyre);
- find owners with `buck2 uquery "owner('<file>')"`, then `buck2 test <owners> <inferencer + run_context targets> 2>&1 | tee /tmp/<phase>.log; echo ${PIPESTATUS[0]}`;
- never edit files while buck runs; on a CAS 404, use the `#link-tree` workaround;
- compare failure IDs with the P0 baseline, not just counts, and compare each target's collected-test count with its P0 count (a target that collects nothing also passes);
- the ratchet stays green, and `KNOWN_DEBT` only shrinks, exactly as annotated.

**Dependencies** (a phase starts only after everything it needs is green):

| Phase | Needs | Why |
|---|---|---|
| P1 | P0 | baseline, ratchet |
| P2 | P0 | streaming goldens |
| P3 | P1, P2 | B14 is in P3's stop gate; frames go into the P2 templates |
| P4 | P3 | frames, outcome, `publish_result` |
| P5 | P1, P3 | `node.role_state` (B12); invocation keywords need frames |
| P6 | P4 | BTA/MFI/Dual publication uses the P4 channel; `_summary_at` reads it. P6 also lands the single-flight guard, before P7 makes overlapping BTA calls possible |
| P7 | P5, P6 | `bta_node_name` uses the P5 invocation keywords; `_BtaGraph` uses the P6 attempt and call record |
| P8 | P6, plus separate approval | the lease and manifest use the P6 `checkpoint_root` and record the P6 effective plan |
| P9 | P4, P6; P7 for BTA/MFI certification | PTI/LWI contracts and components; MFI's `effective_sub_queries` lives on `_BtaAttempt` (P6); certifying BTA and MFI needs their graph debt cleared (P7) |
| P10 | P3 | components, `LiveHandleField`, the session-scoped helpers |
| P11 | P5, P7, P9, P10 | the final purity contract needs every family measured |

§17's order is one valid serialization of this table. **The plan is complete only after P8.** Without P8, report partial completion, with I4's lease clause and I11 still open.

**Stop-point semantics:** when a stop condition hits, the affected rows stay in `KNOWN_DEBT` with the reason, the phase ends green, independent phases continue, and the user is told.

**What stopping after Pn buys** (cumulative; each row assumes its dependencies landed):

| Stop after | Holds | Still open |
|---|---|---|
| P0 | measurement only: `run_context` suites in buck, target coverage and collection counts, ratchet with measured debt, goldens, inventory | every defect |
| P1 | B4, B5, B9, B11, B12, B14; `run_async_joined` | all per-call instance state; no frames or claims |
| P2 | entry shapes: B26, B26b, B27, B30 | ctx still held across yields (B29); no claim |
| P3 | the seam: strict claim, outcome freshness, B6 (a/b), B29, B32, B34 | no class publishes an outcome yet; transport handoffs (B28) |
| P4 | leaf and LWI contracts on the typed channel (B10) | BTA, MFI and Dual still read last-call getters |
| P5 | B16/B16b, B17, B18, B22 | BTA per-call state |
| P6 | BTA per-call state off `self` (B1, B2, B3, B13, B21, B25, B36); the single-flight guard, so **host B24 is rejected** (BTA is uncertified) | **bare B24** (graph state still on `self`, no production caller, §2.2) until P7; B31 |
| P7 | B24 in every mode; B31; concurrent BTA calls on distinct instances | lease and manifest; PTI/LWI; leaves |
| P8 | B35; I4's lease clause; I11 | — (required for completion) |
| P9 | B7, B15 (if confirmed), B20; LWI flags; BTA, MFI, Dual, LWI, PTI certified as their debt empties | provider leaves |
| P10 | B19, B23, B28, B33, B6 leaves; Conversational's `_final_output_at` | enforcement |
| P11 | source checks and `WorkGraph` base removal | — |

### P0: infrastructure, measurement, inventory (no production behaviour change)

**Commits:**
1. **`TEST/run_context/BUCK`**: a `python_pytest` target copied from the `TEST/knowledge/BUCK` template. It includes `conftest.py` (the suite uses an autouse fixture), the golden resources, `network_access_utils.none()`, `typing=False`, deps `pytest` and `//_tony_dev/CoreProjects/AgentFoundation/src:agent_foundation`, and `oncall("ml_oncall")`. Confirm with `buck2 uquery "owner('…/test_m7_purity_gate.py')"`. Record the baseline pass/fail IDs.
   - **Target coverage.** For every test target the plan relies on, check with `buck2 uquery "deps(<target>, 1)"` that it depends on `//_tony_dev/CoreProjects/AgentFoundation/src:agent_foundation`. The terminal-inferencer targets (`TEST/common/inferencers/terminal_inferencers/BUCK`, lines 19–92) depend on the ScienceModelingTools copy today; each is repointed, or twinned if ScienceModelingTools still needs it. Record every target's collected-test count as the gate baseline.
2. **Frozen plan and inventory, versioned [Idem].** Copy this plan revision to `AgentFoundation/invocation_scoped_runtime.plan.md`. The inventory (commit 6) goes to `AgentFoundation/invocation_scoped_runtime.P0_inventory.md`. Both are review evidence in the repo, not only files under `~/.claude/plans`.
3. **Purity tool (B8) and negative tests (I10).** Re-run `test_m7_purity_gate.py`; each newly visible delta becomes a carve-out or `KNOWN_DEBT`, with its reason.
4. **Ratchet (I1–I4)** with measured `KNOWN_DEBT`. The `_HOST_PURE_CERTIFIED` assertion starts with no class certified.
5. **Goldens**, normalized before comparison: timestamps, invocation and session ids, and absolute temp paths are replaced by stable placeholders, and records whose order is not part of the contract are sorted. Ordered records that are part of the contract (session-log `log_type` sequence, graph events) are never sorted.
   - bare getters (I8);
   - BTA (I7): fresh sync, fresh async, partial resume (two worker results deleted), no aggregator, MFI; each in pickle and jsonfy;
   - PTI/LWI locations;
   - **streaming entries**: rovodev, devmate, OpenClaw and claude_code CLI on a stub transport, each through `ainfer`, direct async streaming, and direct sync streaming (devmate and claude_code native; the others through the bridge). Each path runs two sequential bare calls (session continuity) and one direct call with an explicit `run_context` (B26 characterization). devmate also runs with a truthy `inference_config` (B27). The devmate and claude_code sync entries also run with a fan-out configured (B26b);
   - **OpenClaw retry**: a stub gateway raising a rate-limit N times, via `ainfer` and `infer`. Record the transport-call count, continuation prompts, `_maybe_initialize_session` calls and final session (B30);
   - **streaming lifetimes**: after `async for … break` on a bare leaf, record `active_run_context()` in the consumer task and the finalizer's exception (B29 characterization); after dropping a sync bridge generator mid-stream, record whether the transport is still running;
   - **MFI interactive rerun**: a stub review that says "rerun" once, and a recording `response_parser`. Today it records two post-processing passes (§5.7);
   - **finalize after a non-BTA result** (B36): every attempt raises, then (a) an external `fallback_inferencer` returns, or (b) a non-exception `default_return_or_raise` is returned. Record the canonical output link and any raise;
   - a pre-refactor workspace fixture.
6. **Inventory**, committed as `AgentFoundation/invocation_scoped_runtime.P0_inventory.md`:
   - the writers of every §7 field, and host-mode presets;
   - post-call instance readers;
   - the documented-getter list, which becomes the `RuntimeKey.compat` declarations;
   - `start_nodes` / `subgraph_registry` introspection users;
   - every thread or process hop on the call path;
   - every concurrent dispatch site on a call path (`asyncio.gather`, task groups, thread pools), and whether each branch derives its own child ctx. A branch that shares its parent's path would trip the claim, so each such site is fixed to derive a child ctx before P3 lands;
   - judges with an `input_preprocessor`;
   - **host concurrent use of one instance** (for the P6 guard): configs that borrow one instance in concurrently runnable stages or across two parent calls, host callers of `parallel_infer` / `aparallel_infer`, and guardrail judges shared by concurrent items. Each row records how the guard commit resolves it (certify, or a factory / `fresh_instance` per item);
   - **nested same-path public calls**: any public entry that calls another public entry at its own ctx path while its invocation is live. This includes agentic functions called with an explicit caller ctx (`AF/agentic_functions/decorator.py:288-297`). Each one found is refactored to a child slot or the private pipeline before P3 lands; the claim is never relaxed;
   - **root sites and handle-store reuse** (B32): every place that constructs a root `RunContext`, and whether it reuses one handle store across turns. The four per-turn-root callers in §10 are the known cases;
   - **`_init_call_state` callers and `state_factory` classes** (B34): every override of `_init_call_state` and every class with a `state_factory`, with what each resets;
   - **CLI session policy** (seam hooks): for claude_code, codex, kiro, rovodev, devmate and OpenClaw, the exact code before and after `_ainfer_single` in their `ainfer` / `infer`, split into argument adaptation (stays in the adapter), `_prepare_call` and `_conclude_call`;
   - **BTA path sites** (§5.7): every derivation from `self._workspace` or `checkpoint_dir`, with its rule (workspace-first then `checkpoint_dir`, or workspace-only);
   - whether roots with distinct `RuntimeBindings` can share a store;
   - every `.aggregator_inferencer` hit in `AF/` (35 today), classified as *needs the resolved stage* or *presence check* (§5.7);
   - every `ctx.node(creator=…)` site (`AF/inferencer_base.py:3731`, `AF/templated_inferencer_base.py:476`, `:670`, and any others), with whether it runs inside the owner's frame;
   - every `WorkGraph._run` caller that can run inside a running loop (the B14 audit);
   - **loop-bound resources** (loop affinity, §5.1): every leaf that holds an SDK client, an anyio task group or subprocess pipes, the loop that creates it, and where it is closed. Each must be closed inside its creating loop; the claude_code SDK's stale-loop drop is the known violation;
   - **`fresh_instance` sharing**: whether instance-valued recipe entries and overrides (`AF/inferencer_base.py:1462-1483`) are shared by reference or copied. This decides whether P6 registers the per-call fanout in the caller's ledger (§5.7);
   - **LWI resume flags**: every writer and reader of `_step_was_previously_attempted` / `_previous_attempt_info`, including the RPU workflow tests that set them;
   - stage configurations whose stages are not `InferencerBase` (duck-typed, including `mock_inferencers/`), and every class that reads `interactive` or `stream_observer` (the `_INVOCATION_KEYWORDS` declarations);
   - tests that call a private hook directly, and tests that read `worker.name`, `worker.stream_observer` or `worker.interactive` after a BTA call;
   - the B28 family: every leaf field written by a transport and read by `_ainfer`, with its documented getter if any;
   - remaining callers of `_delegates_execution_under`;
   - callers that pass an `inference_config` to devmate `ainfer` (B27);
   - Tier-3 handles other than `live_session_id` that devmate, OpenClaw and claude_code touch on direct streaming (for the P2 stop condition);
   - every class in `AF/` that overrides a public entry (`ainfer`, `infer`, `ainfer_streaming`, `infer_streaming`, `parallel_infer`, `iter_infer`, `__call__`), with whether it reaches `_(a)infer_single` or a streaming template. This is the fixture list for `test_streaming_entry_contract.py`; today there are 8 streaming entries and 6 CLI `ainfer`/`infer` overrides;
   - OpenClaw configs that set `max_retry`, `fallback_mode`, `fallback_inferencer`, a guardrail or a timeout (B30);
   - consumers that stop a stream early without closing it (`break` inside `async for` / `for` over a public streaming entry). Under a host ctx each such consumer must close the stream, or the claim rejects its next same-path call (§4.6); each is fixed with `aclosing` / `closing` before P3 lands;
   - consumer code that reads `active_run_context()` or a leaf's `active_session_id` between yields (B29).
7. **Probes:** B15 (two bare calls on one multi-iteration LWI), the B35 reproducer (interactive selection, a crash after some workers persist, then a fresh-process resume), the B36 reproducer (the finalize golden above), the B31 reproducer (two round-robin workers sharing one nested BTA, no reporter), and the B34 reproducer (`parallel_infer` on an MFI with stale dispatch state; `infer(iterator)` with a recording `state_factory`).

**Search method:** all P0 searches are scoped to `AgentFoundation/` and `RichPythonUtils/` (scoped `grep -rn` or `search_files` with an explicit path; the index omits some subtrees, so an empty result is re-checked with a scoped grep). Never a recursive search over fbsource.

**Exit:**
- `buck2 test` runs the `run_context` targets;
- the negative tests fail on injected mutations and pass otherwise;
- the ratchet is green with measured debt;
- the goldens are committed;
- every relied-upon target depends on the CoreProjects `agent_foundation` and has a recorded collected-test count;
- the frozen plan and the inventory are committed; B15 is confirmed or dropped, and the B31, B34, B35 and B36 effects are confirmed.

**Risk:** L. **Rollback:** revert the test and document files.

### P1: foundational fixes (no new framework)

**Commits**, in order:
1. **B12:** writer and reader use `node.role_state`; one-way migration. Coexistence test with `MultiFlowState`, `DualState` and `BTAState`, for switch-before-call and switch-after-call.
2. **`run_async_joined` and B14.** The new RPU helper beside `_run_async`, with its own tests: no running loop; inside a running loop; the caller's ContextVars visible in the coroutine; exceptions propagate. Then the thread hop at `workgraph.py:2746-2751` goes through it. RPU regression test: a ContextVar set before `_run` is visible inside a node. Every caller from the P0 `WorkGraph._run` audit is checked in the same commit.
3. **B4.**
4. **B5.**
5. **B9:** delete the dead surface, after a reader audit; update the guardrail tests.
6. **B11.**

**Logic preserved:** each fix touches only the faulty path; bare-visible fixes are tagged.

**Exit:**
- each regression test fails on the parent commit and passes on the fix;
- the matching `KNOWN_DEBT` rows (window, `_last_inference_input`, `_output_finalized`, devmate `dump_output`) are removed.

**Risk:** L. Each commit is green; rollback in reverse order (§13).

### P2: streaming template-method split (§5.2)

This phase lands the entry **shapes** with `enter_run` only. No frame exists yet; P3 commit 7 adds frames to the two templates.

**Commits:**
1. **templates:** the base async and sync templates as plain methods returning today's generator; the `_infer_streaming_pipeline` hook, whose default is today's thread bridge; the sync fan-out branch. Every consumer (base `_ainfer`, the bridge, tool_as) still calls the public entry, so a leaf's public override still runs;
2. rovodev: body into a `_ainfer_streaming_pipeline` override; public override deleted;
3. devmate async: pipeline (keyword-only `filter_session_info=False`) + the public signature adapter. `_ainfer` still binds positionally through the adapter, so behaviour is unchanged in this commit;
4. devmate sync and claude_code sync: `_infer_streaming_pipeline` overrides (B26b regression test);
5. **switch:** base `_ainfer` (`AF/streaming_inferencer_base.py:1041`), the sync bridge default and tool_as `_ainfer` consume the private pipeline. This is the commit that changes devmate's binding, so it fixes B27, with its regression test;
6. OpenClaw streaming;
7. OpenClaw `ainfer` convergence (B30), with the retry golden.

**Logic preserved:**
- The pipeline bodies move verbatim, and `finally` order is unchanged.
- Session resolution now runs inside the call's ctx (B26).
- Direct streaming legacy-mints; the session getter and setter behave identically in legacy and no-ctx modes.
- OpenClaw's attempt count, continuation prompts and session initialization are unchanged (golden).

**Tests:**
- the P0 streaming and OpenClaw goldens (only the B26/B26b rows change);
- the three-way parity test (`ainfer`, direct async, direct sync);
- the B26, B26b and B30 regression tests;
- the `test_streaming_entry_contract.py` skeleton: the ctx half, and the override-shape check over the P0 entry list.

**Exit:**
- the goldens are green;
- no public streaming override contains behaviour beyond validation or argument adaptation, and every public `ainfer` / `infer` override reaches `_(a)infer_single` or is a thin adapter (reviewed, plus the contract test);
- `_delegates_execution_under` is deleted if P0 found no other callers.

**Risk:** M; one class per commit.

### P3: invocation seam, strict claim, lazy iterators, handle scope, outcome invalidation (no field migrated yet)

**Commits:**
1. `RC/invocation.py` and unit tests: `RuntimeKey` (with `compat`), `InvocationFrame` (with `invocation_id`, `cleanup_errors`, `discard`), `frame_for` / `invocation_of`, `ResourceLedger`, `framed_agen` / `framed_gen`, the `InvocationContractError` family and `NoInvocationError`. `InvocationContractError` joins both `non_retryable_exceptions` tuples (`AF/inferencer_base.py:3096`, `:5156`);
2. the RPU `iterate_async_in_thread` helper and its tests: a stub async generator is cancelled and joined when the sync consumer closes early, when it closes before the first item, and when it is garbage-collected; a thread that ignores cancellation is logged at ERROR after the bounded join;
3. `ActivePathClaims` on the store (strict, no ancestor exception), diagnostics in the error, the `evict_subtree` assert;
4. `NodeOutcomeState`, `RenderedTaskContractState`, `NodeRunState.outcome`, `clear_outcome` / `publish_outcome` under the store lock, and the helpers (codec plus old-store load);
5. **the seam** in `_(a)infer_single` (`open_invocation` / `aopen_invocation`, §5.1 steps 2–11): claim, outcome clear, `_init_call_state` moved in from `infer` / `ainfer` (B34), the `_prepare_call` / `_conclude_call` hooks with identity defaults, ledger close (async `await aclose()`; sync `run_async_joined(aclose())`), cleanup-error handling, publication through `_outcome_for` (base default `None`), release. Step 4's keyword pop gets its declarations in P5;
6. **CLI session policy into the hooks**, one commit per leaf (claude_code, codex, kiro, rovodev, devmate, OpenClaw): their `ainfer` / `infer` become thin adapters; session resolution and `_apply_session_policy` move into `_prepare_call`; **all** post-call code moves into `_conclude_call`: the claude_code, codex and kiro session writes, rovodev's result wrap and `find_latest_session_id`, devmate's `success=False` promotion and error counting. Each commit is proven by the P0 streaming goldens;
7. frames in the two streaming templates, with the streaming binding rule (B29): the templates' inner generator is replaced by `framed_agen` / `framed_gen`, which resolve the ctx and acquire at first resumption and run the fan-out branch inside the frame; the default sync pipeline uses the bridge helper;
8. rovodev's `_current_output_file` becomes a frame component, and its three readers fall back to `self.output_file` when frameless (B29, §5.1). The contract test that the consumer's `copy_context()` is unchanged between yields, for every streaming entry, lands here;
9. lazy `infer(iterator)` binding (B29 extended): resolve the ctx and `exit_run` before returning; `ctx_bound_gen(ctx, inner)` binds it per `next()`;
10. `publish_result` / `read_result` with `pending_compat`, flushed on every non-host close (success, failure, cancellation); the ratchet derives ALLOWED compat fields from the declared keys;
11. `LiveHandleField` (B6 a/b), `_session_scoped_get` / `_set`, and the lifted handle helpers;
12. handle scope (B32): `LiveHandleStore.scope_id`, `(scope_id, path)` keying with the fixed legacy scope, `_iter_live_handle_sets` over every scope, and the per-turn-root caller migrations P0 decided.

**Stop gate [Sep]:** every entry gets a correct, unique frame in:
- nested child calls, `parallel_infer`, `aparallel_infer`;
- direct async and sync streaming (bridge and native);
- OpenClaw and the other CLI `ainfer` / `infer` adapters;
- cancellation, and streams or lazy iterators abandoned or closed early;
- sync-from-async (B14);
- copied-context sibling tasks;
- the session helpers, `iter_infer`, `__call__`.

**Seam tests [Idem]:**
- a same-path rejection happens before `state_factory`, MFI's dispatch reset, any session mutation, and any model or checkpoint I/O (spies on each);
- each `parallel_infer` item and each iterator item initializes its own call state (the B34 reproducer passes);
- a devmate call whose result `_conclude_call` promotes to failure publishes no outcome, and its bare getters still show this failed call, as today (instance fields until P10, then the every-close compat flush);
- success A then failure B at one path: `read_outcome` is `None` after B;
- cleanup: a stub resource whose close fails after a successful call → the outcome is published with `cleanup_errors`, then `InvocationCleanupError` carries the result; the same failure during a failing call → the original exception propagates with the failure in a note, logged at ERROR; neither is retried;
- lazy iterator: after a `break`, an explicit `close()`, garbage collection, and a `close()` from another thread, the consumer's `active_run_context()` is its own, no claim remains, and no finalizer error is logged;
- streams: after `async for … break`, the consumer's `active_run_context()` is its own and no finalizer error is logged; a dropped sync stream stops its stub transport.

**Claim tests:**
- sibling same-path calls raise;
- two instances at one path raise;
- a nested public call at the caller's own path raises (there is no ancestor stacking);
- private calls (`super()._ainfer`, recovery, MFI's BTA super-call) claim nothing;
- sequential re-entry is allowed;
- a same-path call while an earlier stream is unclosed raises, naming the holder's owner class, entry and age, with the `aclosing` hint; after `aclose` it is allowed;
- a stream object that was never iterated doesn't block a same-path call;
- `evict_subtree` with a live claim below the prefix fails the assert; Dual's consensus retry passes.

**Handle-scope tests (B32):** two host roots, both at path `"/"`, with distinct handle stores, never share a session id or client; one root reused across turns keeps continuity; two sequential bare calls keep today's continuity; leaf `adisconnect` tears down every scope's branches.

**Frameless tests:** `publish_result` under a host ctx without a frame is a no-op; in bare modes it writes only the declared compat fields; `read_result` without a frame raises `NoInvocationError` whose message names the key, the owner class and the `open_invocation` fix.

**Exit:**
- the full suite and ratchet are green;
- zero claim trips in the existing suites (a trip is a real bug to fix, never a claim to relax);
- the contract test is fully green (frame half included).

**Risk:** M–H (lifetimes). Each commit is green; rollback in reverse order (§13).

### P4: outcome channel, generic half (B10)

BTA, MFI and Dual publication and the parent readers need `_BtaAttempt`, so they land in P6 [Idem].

**Commits:**
1. **Templated leaf and LWI publishers.** The templated leaf publishes `_RENDERED_CONTRACT` (captured at `AF/templated_inferencer_base.py:454`); LWI publishes its first step child's contract; both through `_outcome_for`. Also `_task_contract_at(child, child_ctx)` as the shared reader API, with no caller switched yet.
2. **Compat takeover for the leaf.** `_last_rendered_task_instructions` is written only through `publish_result`, as the declared `compat` field of `_RENDERED_CONTRACT`. Dual's preview renders (`FLOW/dual_inferencer.py:1552`, `:2039`) become no-ops under host; a regression test shows that a preview no longer overwrites the leaf's published contract.

**Tests:**
- a templated leaf and an LWI publish in host mode, and publish nothing on failure;
- one leaf in two host branches: each branch's outcome holds its own render;
- `read_outcome` at the child ctx after a leaf call, in host and legacy modes; true no-ctx falls back to the getter;
- `test_task_instructions_snapshot.py` green and unchanged.

**Exit:** the tests above pass, and the `KNOWN_DEBT` rows for the templated snapshot are removed. **Risk:** L–M.

### P5: definition channels (B16, B17, B18, B22; role audit)

**Commits:**
1. role (`RoleTransition`, provenance, `_effective_role_state`, typed `RoleState` fields with the load migration, B16/B16b);
2. feed and modes publication (B17), first rewriting the tests P0 listed;
3. judge (B22);
4. invocation keywords (B18): `_INVOCATION_KEYWORDS`, consumption in the entries, `_effective(name)`, BTA dispatch of `stream_observer` / `interactive` to leaf workers, breakdown and aggregator; the duck-typed protocol kept behind its exemption.

**Tests:**
- the precedence ladder;
- a role switch with a master version and variables under a host ctx renders like the no-ctx switch; an old store with those keys in `changes` loads into the typed fields;
- a borrowed leaf reused by two sequential BTA calls: each call's observer receives only its own events, and the leaf's configured observer is unchanged afterwards; a PTI worker's configured `interactive` survives the call;
- the render-once barrier in the fan-out;
- the judge gets the exact prompt and is not mutated;
- the updated purity-gate carve-outs;
- the MFDual suites.

**Exit:** tests green, and the role, feed and judge `KNOWN_DEBT` rows are removed. **Risk:** M–H (feed). Each commit is green; rollback in reverse order (§13).

### P6: BTA call record, attempt loop, summary, publication, ownership, aggregator, single-flight guard (B1, B2, B3, B13, B21, B25, B36)

**Commits:**
1. **Call record (`_BTA_CALL`).** The frozen snapshot (`effective_workspace`, `checkpoint_dir`, `checkpoint_root`, §5.7), put at the start of BTA's `_ainfer` / `_infer`. Every path site from the P0 list reads it, each keeping its own rule, including the base `_promote_child_checkpoints` (`AF/inferencer_base.py:2378-2440`). The I7 goldens stay byte-identical.
2. **Attempt loop.** `_BtaAttempt` + `_BTA_ATTEMPT`; the recursive rerun (`BTA.py:2353-2363`) becomes the loop, and the tail (`_emit_graph_reconcile`, `_finalize_response`, `_conclude_attempt`) runs once, after it. The `_begin_attempt` / `_conclude_attempt` hooks are added (identity in BTA). MFI overrides both (`_begin_attempt`: `_apply_runtime_input_propagation` + `_reset_cross_flow_state`; `_conclude_attempt`: normalize → extract dispatch state → strip), its `_ainfer` override is deleted, and its `_infer` keeps only the coordination guard. `_BtaAttempt.use_async` feeds the node-function builders. The five mailbox fields are deleted (B3, B21); `_worker_instances` stays until commit 6. The direct-`_ainfer` BTA tests are wrapped in `aopen_invocation`. The MFI rerun golden changes exactly as recorded in `BARE_EXPECTED_CHANGES`.
3. **`BtaCallSummary`** (with its `last_call_summary` compat getter): discarded at the start of each attempt, put as the run's last statement; the finalize readers switch to it; without a summary, `_finalize_output` takes the base path (B36).
4. **BTA, MFI and Dual publication and readers (B1, B13)**, moved here from P4 because they need `_BtaAttempt` [Idem]. Publishers, each through `_outcome_for`: BTA (lowest successful index, sync and async), MFI (winner first, from the `flow_N` outcomes), Dual. Readers: Dual (`FLOW/dual_inferencer.py:785`, `:1438`) and MFI (`FLOW/multi_flow_inferencer.py:1606-1630`), both via `_task_contract_at`. `_worker_task_instructions` is written only through `publish_result`, as the declared `compat` field of the BTA contract key.
5. **Stage ownership.** `resolve_stage` and `ResolvedStage` for all three stage kinds (breakdown always borrowed). The attempt ledger (workers) and the call ledger (aggregator) call `adisconnect` only when it is present. Sync closes via `run_async_joined(ledger.aclose())`, and failures go to `frame.cleanup_errors` (§5.1). The claude_code SDK's sync `_infer` closes its per-call client inside its own `_run_async` loop (codex's `_run_and_close`), replacing the stale-loop drop. `adisconnect` becomes idempotent. `_bind_rebuilt_child_ws` stays for owned stages only; a borrowed stage gets a child-ctx publication in host mode. In host mode, graph build refuses to dispatch one borrowed duck-typed instance to two concurrently runnable workers.
6. **Aggregator, fan-out and `_worker_instances`.** A pure `build_aggregator`, with every *needs-the-stage* reader migrated in the same commit; `_iter_child_slots` / `_iter_child_inferencers` yield the call-scoped aggregator. The fan-out resolves the aggregator before `fresh_instance`; the caller's ledger closes the owned aggregator, and registers the fanout itself only if P0 showed `fresh_instance` copies instance-valued entries. `worker_count` is read via `_summary_at`. `_worker_instances` is deleted once its readers (finalize, `adisconnect`, the harvest, the fan-out) have migrated.
7. **Single-flight guard** (§5.1), registered at seam step 2 and released at step 11. The same commit resolves every P0 "host concurrent use" row: certify the class when its measured debt is already empty, otherwise give that caller a factory slot or a `fresh_instance` per item. A row that can't be resolved either way stops only this commit (stop-point semantics); the guard is never weakened.

**Stop gate (moved from P4):** no internal parent reader uses a child instance's last-call getter. Checked by a scoped search, `grep -rn "_proposer_task_instructions()"` over `AgentFoundation/src/`: the only hits are no-ctx branches of `_task_contract_at` and the getters themselves.

**Tests:**
- **Workspace snapshot:**
  - with no configured backing workspace, sequential host calls under two ctx workspaces write two independent checkpoint trees;
  - with a configured backing workspace, both calls resolve to the backing root (today's precedence);
  - a callback running under a child ctx that carries a `workspace_override` resolves the parent's snapshot;
  - a `checkpoint_dir`-only BTA puts its breakdown, aggregator and graph result paths exactly where it does today.
- **Attempt loop:** an interactive rerun runs two sequential attempts with distinct `_BtaAttempt` objects; the first attempt's ledger is closed before the second begins; MFI's propagation and cross-flow reset run once per attempt; MFI's post-processing (and its `response_parser`) runs once per call; the async graph's node functions are async (`_BtaAttempt.use_async`).
- **Finalize (B36):**
  - every attempt raises, then an external `fallback_inferencer` returns: `_finalize_output` finalizes the fallback's response through the base path and links no aggregator output;
  - the same with a non-exception `default_return_or_raise`;
  - a result a BTA run produced still takes the summary branches (the P0 goldens);
  - a `_conclude_attempt` that raises leaves no summary.
- **Aggregator traversal:** `pre_retry` archives the call-scoped aggregator's workspace between retries, as the slot write makes it do today.
- **Loop affinity:** a sync BTA called from inside a running loop closes its ledger through `run_async_joined`; SDK clients are closed inside the loop that created them (a stub client records its closing loop).
- **Reuse:** a second sequential call sees nothing from the first. Slot identity is unchanged after success, failure and resume.
- **Ledgers:**
  - exercised on success, retry, failure, timeout and cancellation;
  - a sync BTA closes its owned workers before `infer` returns;
  - B25: earlier workers are disconnected, and a sync fan-out's per-call BTA is closed at the caller's call end;
  - a stage close that fails after a successful call publishes the outcome with `cleanup_errors` and then raises `InvocationCleanupError` carrying the result. The same failure during a failing call becomes a note on the original exception.
- **Publication:**
  - the chain leaf → LWI → BTA → MFI → Dual → MFDual, in host, legacy and no-ctx modes;
  - two parents sharing a child;
  - the MFDual fixer re-render leaves the propose contract unchanged;
  - a sync BTA publishes the lowest successful index.
- **Fan-out:** the fan-out suites and `n_subtasks`, in host, legacy and true no-ctx (a bridge-entrypoint leaf with a fan-out).
- **Guard:**
  - two overlapping host calls on one uncertified leaf: the second raises before `_init_call_state`;
  - one borrowed breakdown shared by two overlapping BTA calls raises;
  - a host `parallel_infer` on an uncertified class raises, or runs per-item instances as P0 decided;
  - sequential reuse passes, and overlapping bare calls are unaffected;
  - a certified class passes, and a subclass of a certified class doesn't inherit certification.
- **Duck-typed stages:** such a configuration runs unchanged; one borrowed duck-typed instance on two concurrently runnable workers raises in host mode.
- The P0 I7 goldens are unchanged, plus the §9 rewrites.

**Exit:**
- the BTA per-call rows are gone from `KNOWN_DEBT`, except the P7 graph rows and the P9 breakdown-workspace row;
- every P0 host-concurrent-use row is resolved, or recorded as a stop-point row.

**Risk:** H; one commit per group.

### P7: `_BtaGraph` composition (B24); BTA still subclasses `WorkGraph`

- Implement §5.8, then B31 as a separate commit: `bta_node_name` joins BTA's `_INVOCATION_KEYWORDS`; the worker dispatch passes it instead of writing `worker.name` (`BTA.py:2552-2554`, `:2799-2800`); `_bta_prefix` and `_BtaGraph.name` read `_effective("bta_node_name")`.
- **Gate:**
  - the I7 goldens match byte-for-byte;
  - the B31 reproducer passes, and a nested BTA's `name` is unchanged after the call;
  - the overlapping-calls reproducer passes: one BTA, A's breakdown slow and B's fast, both correct (it fails today). It runs as two concurrent bare calls, where B24 is just as real and the guard doesn't apply;
  - two host calls on distinct BTA instances (or a factory), with distinct ctxs and workspaces, run concurrently;
  - two overlapping host calls on one BTA raise `UncertifiedConcurrentUseError` until BTA is certified (P9).
- **Stop point:** any drift in checkpoint paths, expansion replay, event order, log parentage or MFI behaviour → stop. The graph rows stay `KNOWN_DEBT`, and concurrent same-instance BTA calls stay unsupported, as they are today.
- **Risk:** very high.

### P8: checkpoint-root lease + two-phase manifest, B35 (required for completion; separately approved)

**Commits:**
1. the RPU lock helper (extracted from `storage_based_queue_service.py:205-254`), with its tests;
2. the lease (§5.11): acquired right after `_BTA_CALL`, registered first in the frame ledger;
3. `RC/resume_identity.py`: the canonical identity function and `ResumeIdentityUnavailableError`;
4. the manifest (§5.12): header, committed effective plan, reconstruction from the plan, the policy table, `trust_legacy`, the certify helper.

**Tests:**
- **Lease:**
  - one `checkpoint_root` across two processes → `BtaWorkspaceBusyError` before any read;
  - the same across two threads on distinct BTA instances;
  - a `checkpoint_dir`-only BTA takes and releases the lease;
  - the lease is held across a retry, finalize and the call-scoped close, and released last;
  - a BTA with no root takes no lease.
- **Identity:** canonical encodings for `str`, `bytes`, JSON-compatible values, inferencers and a factory with `resume_identity`. An unsupported value raises `ResumeIdentityUnavailableError` before any I/O, and nothing is ever derived from `repr()` (a test object whose `repr` changes between runs keeps its identity, or raises). Scheduling-only changes don't change identity.
- **Manifest:**
  - fresh-process resume with a match;
  - mismatch, corruption and legacy failures;
  - `trust_legacy` logs a WARNING and writes no verified manifest;
  - the certify helper;
  - resume reconstructs from the committed plan after interactive selection and truncation: the P0 B35 reproducer passes.
- **Crash points,** each followed by a fresh-process resume: header only; breakdown saved; plan committed; some workers persisted; aggregator completed.
- **Goldens:** the I7 goldens differ only by the manifest and the lock file.

**Rollback:** tolerate the new files, and never delete user checkpoints.

**Behaviour change:** legacy resume needs `trust_legacy`, which is why the phase needs explicit approval. Without P8 the plan reports partial completion (§11 dependencies).

### P9: PTI / LWI / Dual / MFI (B7, B15, B20)

- `_current_*` and Dual `_last_iteration_record` become components.
- Host publication follows §5.13.
- MFI uses `effective_sub_queries` on the attempt.
- **LWI resume flags:** `_step_was_previously_attempted` / `_previous_attempt_info` keep their properties, which now route to a frame component; the LWI and RPU workflow tests P0 listed are wrapped in `open_invocation(lwi)` in the same commit.
- **Separate atomic subphase:** the PTI/LWI child save/resume flags and `_result_root_override`, protected by location goldens.
- BTA's breakdown workspace write (`BTA.py:2062`) becomes a publication to the `breakdown` child ctx under a host ctx (§5.13).
- **Certification:** each of BTA, MFI, Dual, LWI and PTI sets `_HOST_PURE_CERTIFIED = True` in the commit that empties its measured debt, and the ratchet verifies it. From then on, overlapping host calls on one instance of that class are supported.
- **Stop point:** location-golden drift → those writes stay `KNOWN_DEBT`, while `_current_*` and the child writes still convert. A class with remaining debt stays uncertified, so the guard keeps rejecting overlapping use of it.
- **Risk:** very high.

### P10: provider leaves (one family per commit)

- **First, the terminal base (B28).**
  - The transports (TIB, TSIB, codex CLI, claude_code) call `publish_result(self, _TERMINAL_RESULT, …)`; TSIB `_ainfer` and devmate read it with `read_result`.
  - The `_last_streaming_*` fields are the key's `compat` fields, still behind `get_streaming_result`.
  - Direct-`_ainfer` terminal tests are wrapped in `aopen_invocation` in the same commit.
  - Regression test: two overlapping calls on one stub terminal leaf under distinct host ctxs each parse their own stdout and return code.
- the rest of the B28 family, one family per commit: `_last_stream_result` (claude_code, codex CLI), tool_as `_last_response`, the token/usage counters (rovochat, metamate SDK, devmate SDK, codex SDK, claude_code SDK). Each gets a component, `compat` only for a documented getter, and a two-overlapping-calls regression test;
- the six `_session_id` writers go through `LiveHandleField`;
- metamate and rovochat conversation ids become Tier-3;
- B19 (OpenClaw);
- B23 (rovodev), in the same commit as rovodev's `final_output` publication and Conversational's switch to `_final_output_at`;
- B33 (devmate): `_consecutive_error_count` goes through `_session_scoped_get` / `_set`.

As each family's debt reaches zero, set `_HOST_PURE_CERTIFIED = True`; the ratchet verifies it.

**Tests:**
- per-family branch isolation and disconnect tests, before moving to the next family;
- the host fan-out never inherits a sibling's conversation;
- B33: N failures in host branch A leave branch B's next call on its existing session, and bare calls keep today's counter behaviour.

### P11: enforcement and hierarchy cleanup

1. **Certification audit.** There is no static gate: the P6 single-flight guard already enforces shared-instance safety at runtime, and the P6 graph-build check covers duck-typed stages. P11 re-verifies that every `_HOST_PURE_CERTIFIED` ClassVar matches the measured debt (the ratchet, both directions), and that every P0 host-concurrent-use row is resolved or recorded as a stop-point row.
2. **Repo-wide audit, then base removal.**
   - Audit: BTA `isinstance` / `issubclass`, inherited WorkGraph methods, constructor fields, `.start_nodes`, `.subgraph_registry`, direct graph execution, config aliases.
   - Then remove `WorkGraph` from BTA's bases; MFI loses it transitively.
   - Redeclare only `name`, `max_concurrency` and `group_max_concurrency`, which YAML `breakdown-multiflow-plan.yaml:114-126`, `:203-208` and `FLOW/multi_flow_dual_inferencer.py:414-430` use.
3. Remove shims; shrink `KNOWN_DEBT` to empty or documented stop-point rows.
4. **Source check:** forbid host-call instance writes outside ALLOWED, internal reads of declared compat fields (outside the documented getters), and live objects in serialized state.
5. **`RC/README.md`:** the placement rule, ALLOWED, "how to add runtime state", the entry rule; close its "still open" items.

### Optional follow-ups

- **O1:** Dual `_run_get` / `_run_set` → components (skip if any post-call reader exists).
- **O2:** a separate plan for `ConversationalInferencer`.

---

## 12. Logic preservation (before → after)

| Behaviour | Before | After | Why it holds |
|---|---|---|---|
| In-call reads | instance field | owner's component | the frame spans the whole call, including finalize; only the owner writes it |
| Self-calls, recovery, MFI super-call | same instance | same frame | private calls don't pass through an entry |
| Public re-entry | nested public call | private pipeline, with public wrappers kept as thin adapters | template-method split; the public streaming templates are plain methods returning `framed_agen` / `framed_gen` |
| Leaf wrapper logic (session, output file, filter) | public overrides | pipeline overrides | moved verbatim; `finally` order unchanged (`:998` inside the base pipeline); devmate's `ainfer()` filter default kept (B27) |
| Streaming ContextVars | set across yields, leaked to the consumer | bound per resumption of the pipeline | the pipeline sees identical values while it runs; only the consumer's view between yields changes (B29) |
| OpenClaw `ainfer` | own retry loop, outside `_ainfer_single` | the same loop inside `_ainfer`, one base attempt | `fallback_mode=NEVER` with `max_retry=1` gives exactly one `_ainfer` call; retry golden (B30) |
| Terminal result handoff | instance `_last_streaming_*` | `TerminalStreamResult` component | same values, same reader; scoped to the call (B28) |
| Propagation | `_active_ctx` rules | same rules for the frame | copy semantics; B14 fixed first |
| Per-node checkpoint paths | node lambdas | unchanged | bound on the nodes |
| Graph-level records and engine logs | through `self` | forwarded from `_BtaGraph` | same inputs, `name` and logger |
| `_finalize_output` | live children | `BtaCallSummary`; no summary → the base path | same values for a result a BTA run produced; resume-safe. A result from an external fallback or `default_return_or_raise` is finalized as itself (B36) |
| Bare documented getters | instance fields | compat fields declared on the keys, same names | flushed at every non-host close, as today's transports write them during the call |
| Per-call values into stages | written onto the stage (per-call overwrites configured) | declared invocation keywords into the stage's frame | same precedence (per-call first), scoped to the call (B18, B31) |
| Sync streaming bridge | daemon thread, no loop/task handle | `iterate_async_in_thread`: `asyncio.Runner`, cancel + bounded join | same items in the same order; only early close changes (B29) |
| Per-call initialization | `_init_call_state` in `infer` / `ainfer`, before dispatch | seam step 5, inside the frame, once per invocation | same reset for an ordinary call; now also per `parallel_infer` item, iterator item, CLI and streaming call (B34) |
| CLI session policy | around `_ainfer_single` in the public `ainfer` / `infer` | `_prepare_call` / `_conclude_call` inside the frame | same code, same order relative to the pipeline; streaming goldens (P3) |
| Lazy `infer(iterator)` | ctx active until the returned generator finishes | ctx bound per `next()` | each item runs under the same ctx as today; only the consumer's view between items changes (B29) |
| Owned stage close | async fan-out `finally`; workers only at a later `adisconnect`; sync never | attempt or call end through the ledger; sync via `run_async_joined(ledger.aclose())`; loop-bound clients closed by the leaf inside their own loop | the same `adisconnect` call on the same objects, earlier and exactly once, never from a foreign loop (B25) |
| Cleanup failure | fan-out close and BTA `adisconnect` raise; the result is lost | outcome published, then `InvocationCleanupError(result, errors)`; an in-flight exception wins, with a note | still loud, but the result is kept |
| BTA paths | `self._workspace` re-evaluated at each site | the `_BTA_CALL` snapshot | same precedence, evaluated once at the start of the call; I7 goldens |
| Interactive rerun | recursive `self._ainfer` | explicit attempt loop, `_begin_attempt` / `_conclude_attempt`; the tail runs once | same per-attempt steps (MFI propagation and reset), sequential attempts; the tail runs for the returned attempt only, as the recursive `return` skips the outer tail today |
| MFI post-processing | in MFI's `_ainfer` / `_infer`, after BTA's run; twice on an interactive rerun | MFI's `_conclude_attempt`, before the summary | same steps in the same order; once instead of twice on rerun (P0 golden, `BARE_EXPECTED_CHANGES`) |
| Live-handle continuity | path-keyed on the leaf | `(scope_id, path)`; legacy uses a fixed scope | bare continuity identical; hosts keep continuity by reusing their root's handle store (B32) |
| Fan-out | fresh BTA per call | unchanged | now structurally immune |

---

## 13. Risks and rollback

| Area | Risk | What can break → mitigation |
|---|---|---|
| purity tool | M | new deltas surface → carve-outs or debt; negative tests |
| streaming split | M | wrapper logic lost → verbatim moves, goldens, parity test, one class per commit |
| streaming binding (B29) | M | an early-exit consumer now hits the claim → P0 lists them and each is fixed with `aclosing` first; the error message names the fix |
| OpenClaw convergence (B30) | M | retry count or initialization order changes → retry golden; post-init warn-and-override; the P0 config list |
| invocation keywords (B18, B31) | M | a stage stops receiving a per-call value, or a keyword leaks into a leaf's `**kwargs` → dispatch only to declaring classes; the entries pop declared keywords first; observer-isolation and `interactive`-survival tests |
| frameless semantics | L–M | a direct-hook test or a cross-object hook breaks → P0 lists them; `NoInvocationError` names the fix; each migration commit wraps its own tests |
| sync bridge helper | M | a transport ignores cancellation → bounded join, ERROR log naming the owner; helper tests cover close-before-first-item and GC |
| invocation lifetimes | M–H | aliasing or leaks → P3 stop gate; ContextVar semantics identical to `_active_ctx` |
| path claim | M | a legitimate overlap trips → each trip is triaged as a real bug and refactored to a child slot or the private pipeline (P0 lists nested same-path public calls); the claim is never relaxed |
| seam (B34) | M | a hook move changes CLI session behaviour, or `_init_call_state` now runs where a subclass didn't expect it → P0 inventories every override and every CLI before/after block; streaming goldens per leaf; one leaf per commit |
| single-flight guard | M–H | a host `parallel_infer` or a shared judge on an uncertified class now raises → P0 inventories them and the guard commit resolves each (certify, or factory / `fresh_instance`); only that commit stops if one can't be resolved |
| cleanup raise | M | a caller that used to ignore a failed close now sees `InvocationCleanupError` → the result rides on the exception and in the outcome; listed in `BARE_EXPECTED_CHANGES`; the error is non-retryable |
| handle scope (B32) | M | a per-turn-root caller loses cross-turn continuity → P0 lists root sites; each caller that expects continuity passes its session's handle store in the same commit |
| outcome channel | M | readers miss results → publishers and their readers land in one commit (leaf in P4, BTA/MFI/Dual in P6); the chain test |
| feed/modes | H | precedence or render-once changes → ladder and barrier tests |
| BTA attempt / aggregator | H | stale finalize reads, lost sessions → summary; ledger tests; call-scoped aggregator. MFI's hooks change its post-processing order relative to BTA's tail → the P0 MFI rerun golden and the B36 finalize golden |
| loop affinity | M | a loop-bound client closed from a foreign loop hangs or leaks → the P0 loop-bound inventory; SDK leaves close in their own `_run_async` loop; `run_async_joined` for sync closes inside a running loop; a stub client records its closing loop |
| `_BtaGraph` | very high | checkpoints, logs, events → forwarding, build order, goldens; stop point |
| lease / manifest | M–H | legacy rejection → `trust_legacy`, the certify helper, separate approval |
| PTI/LWI | very high | path drift → atomic subphase, location goldens, stop point |
| leaves | M–H | continuity → one family per commit; `LiveHandleField` and the session-scoped helpers keep today's bare continuity exactly |
| base removal | M | hidden consumers → repo-wide audit first |

**Rollback:** each commit is green when it lands. Rollback reverts from the stack tip in reverse dependency order; a commit with no dependents reverts alone. Phases after P3 depend only on the primitives, never on each other's internals, except where the §11 dependency table says so.

---

## 14. Acceptance criteria

1. The `TEST/run_context/` suites run in buck (P0). Every relied-upon test target depends on the CoreProjects `agent_foundation`, and its collected-test count never drops below the P0 baseline.
2. Under a host ctx, every fixture's instance delta is within ALLOWED, and `KNOWN_DEBT` is empty or holds only documented stop-point rows (I1).
3. Under a host ctx, no writes into definition-owned or borrowed objects (configured children, borrowed stages, judges); invocation-owned products may be initialized (I2).
4. Overlapping same-path calls, and overlapping host calls on one uncertified instance, raise before any I/O. Two holders of one BTA `checkpoint_root`, across threads or processes, raise (I4; the lease clause after P8).
5. Parents read results only at the exact child ctx; the chain test passes in all three modes. A failed call never exposes an earlier call's outcome (I5).
6. Definition slots are never overwritten; the ledger closes each owned stage exactly once on every exit, sync included, and no cleanup failure is silent (I6).
7. The BTA byte goldens and the PTI/LWI location goldens are unchanged (I7).
8. Bare documented getters are unchanged except the tagged `BARE_EXPECTED_CHANGES` (I8).
9. Every public entry runs inside run and invocation; `ainfer` and direct streaming agree (I9).
10. The purity gate's negative tests pass (I10).
11. The resume suites are green; old stores load (I11).
12. Each of B1–B36 has a regression test that fails on the parent commit (B15 only if P0 confirms it).
13. The e2e metric contract `n_subtasks` is unchanged.
14. `arc lint` is clean and `arc pyre check-owning-targets` passes on every touched target.
15. G1–G6 ran only on the user's explicit GO (§17).
16. P8 has landed. Without it, only partial completion is reported, with I4's lease clause and I11 open.

---

## 15. Plan comparison and single-plan choice

✅ correct and complete · ⚠️ partial or flawed · ❌ absent or wrong · ↪ Idem v4 is silent here, so it inherits v6's position (it is a set of amendments to v6)

Each plan is judged at the revision in the Sources table, against the source-verified findings in §2.3, §2.4 and §3. Several ✅ from v6's table became ⚠️, because Idem v4 found gaps that apply to every plan (lease scope, sync close, identity rules).

| # | Item | Harm v4 | Sep v2 | Dec v2 | Idem v4 (amends v6) | **v8** |
|---|---|---|---|---|---|---|
| | **A. Architecture** | | | | | |
| 1 | Target: definition stability under a host ctx | ✅ | ✅ | ✅ | ↪ ✅ | ✅ |
| 2 | Per-call home is the invocation, not the node | ✅ | ✅ | ✅ | ↪ ✅ | ✅ |
| 3 | Access mechanism for per-call state | ⚠️ descriptor | ✅ typed components | ✅ typed components | ↪ ✅ | ✅ typed components; bare getters declared on the key |
| 4 | Attempt state local to BTA (no generic `AttemptFrame`) | ✅ | ⚠️ generic, heavier | ✅ | ✅ | ✅ |
| 5 | Typed outcome with a summary | ✅ | ⚠️ contract only | ✅ | ✅ | ✅ one publish point per invocation |
| 6 | Recursive contract layering (parents read direct children) | ⚠️ | ✅ | ✅ | ↪ ✅ | ✅ |
| 7 | Summary-driven finalize | ✅ | ✅ | ✅ | ✅ | ✅ + base path when no BTA run produced the result (B36) |
| | **B. Entries and lifetimes** | | | | | |
| 8 | Public self-re-entry removed | ⚠️ shared frames | ✅ | ✅ | ↪ ✅ | ✅ |
| 9 | Leaf wrapper logic kept (session, output file, filter) | ⚠️ kept, but direct streaming runs frameless | ❌ private split: `ainfer()` skips it | ❌ same | ↪ ✅ | ✅ template-method split |
| 10 | Explicit `run_context` honoured by direct streaming (B26) | ❌ | ❌ | ❌ | ↪ ✅ | ✅ |
| 11 | Leaf result handoff (B28) and devmate argument binding (B27) | ❌ | ❌ | ❌ | ↪ ⚠️ terminal only | ✅ whole B28 family |
| 12 | Complete entry inventory (OpenClaw `ainfer` B30, claude_code sync streaming) | ❌ | ❌ | ❌ | ↪ ✅ | ✅ 8 streaming + 6 CLI entries, contract-tested |
| 13 | ContextVars not held across streaming yields (B29) | ❌ holds them across yields | ❌ | ❌ | ↪ ✅ | ✅ |
| 14 | Lazy `infer(iterator)` binds per `next()` | ❌ | ❌ | ❌ | ✅ found it | ✅ |
| 15 | Sync bridge cancels and joins its transport on close | ❌ | ❌ | ❌ | ↪ ⚠️ stated, no loop or task handle | ✅ `iterate_async_in_thread` |
| 16 | Claim before per-call init; CLI session policy inside the frame (B34) | ❌ | ❌ | ❌ | ✅ found it: one seam | ✅ fixed 12-step seam with `_prepare_call` / `_conclude_call` |
| 17 | Semantics for a hook called outside its owner's invocation | ❌ | ❌ | ❌ | ↪ ⚠️ "written immediately" | ✅ publish is mode-aware; a read raises |
| | **C. Concurrency and freshness** | | | | | |
| 18 | Same-path claim | ⚠️ `(path, owner)`, ancestors allowed | ⚠️ `(store, path)`, ancestors allowed | ⚠️ strict, but the registry lives on Tier-2 bindings | ✅ strict | ✅ strict, owned by the store |
| 19 | One uncertified instance in overlapping host calls (any stage role, `parallel_infer`) | ⚠️ owner guard at one path only | ⚠️ static gate from P4 | ⚠️ static gate from P4, before any leaf can be certified | ⚠️ static reachable-tree gate in P6 | ✅ dynamic single-flight guard from P6 |
| 20 | Independent host roots don't share a leaf's session (B32) | ❌ | ⚠️ via `ctx.handles` (breaks leaf teardown); collision not identified | ❌ | ✅ found it | ✅ `(scope_id, path)` keys |
| 21 | devmate error counter scoped to the session (B33) | ❌ left per instance | ❌ same | ❌ same | ✅ found it | ✅ |
| 22 | Outcome cleared at open, stamped with `invocation_id` | ❌ | ⚠️ has `call_id`, never cleared | ❌ | ✅ found it | ✅ |
| 23 | Guardrail window spans attempts (B5) | ✅ | ❌ per attempt: fail-fast never fires | ✅ | ↪ ✅ | ✅ |
| | **D. BTA runtime** | | | | | |
| 24 | Workspace and checkpoint root frozen per invocation | ❌ | ⚠️ partial | ⚠️ outline | ⚠️ snapshot, but flips precedence | ✅ snapshot, today's precedence |
| 25 | Interactive rerun as an explicit attempt loop | ❌ | ❌ | ❌ | ✅ found it | ✅ loop + `_begin_attempt` / `_conclude_attempt` (MFI overrides both); tail once |
| 26 | Owned vs borrowed, for all three stage kinds | ⚠️ | ⚠️ workers + aggregator | ⚠️ same | ↪ ⚠️ same | ✅ + breakdown; duck-typed protocol |
| 27 | Owned stages closed at call end, sync included (B25) | ⚠️ assumes sync hooks | ⚠️ same | ⚠️ same | ⚠️ native sync hook (dead code); names loop affinity, but as a preflight | ✅ loop affinity: leaves close in their own loop; `run_async_joined(ledger.aclose())` |
| 28 | Cleanup failure after success is surfaced | ✅ | ✅ | ✅ | ✅ | ✅ result kept on `InvocationCleanupError` |
| 29 | Per-call values into stages: observer, `interactive` (B18) | ⚠️ handles, instance first | ⚠️ ctx handles | ⚠️ ctx handles | ↪ ⚠️ Tier-3, precedence inverted | ✅ invocation keywords, per-call first |
| 30 | Nested-BTA node name (B31) | ❌ | ❌ | ❌ | ❌ | ✅ |
| 31 | `_BtaGraph` log/result-path forwarding + build order | ✅ | ✅ | ✅ | ↪ ✅ | ✅ |
| 32 | Staged `WorkGraph` base removal | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **E. Durability** | | | | | |
| 33 | Lease on the checkpoint root, held for the whole call | ⚠️ workspace only | ⚠️ not whole-call | ⚠️ not whole-call | ✅ | ✅ |
| 34 | Resume manifest: fail closed, canonical identity, reconstructable plan, crash points | ⚠️ `legacy_warn` default | ⚠️ two-phase; no identity rules | ⚠️ fail closed; no identity rules | ✅ | ✅ `RC/resume_identity.py`; commit point at `_build_subgraph_spec`, resume rebuilds from it (B35); P8 required for completion |
| | **F. Bugs, process, document** | | | | | |
| 35 | Core bugs: B12 role, B8 purity, B14 hop, B6 session | ✅ | ✅ | ✅ | ↪ ✅ | ✅ |
| 36 | Feed, modes and judge defects | ✅ | ✅ | ✅ | ↪ ✅ | ✅ + typed `RoleState` fields (B16) |
| 37 | Measured ratchet (a stale entry fails) | ✅ | ✅ | ✅ | ↪ ✅ | ✅ |
| 38 | BUCK target for the `run_context` tests | ❌ | ❌ | ⚠️ target type unspecified | ✅ `python_pytest` | ✅ |
| 39 | `n_subtasks` metric contract | ❌ | ❌ | ✅ | ↪ ✅ | ✅ |
| 40 | User decisions (Q3 warn and override; no band-aids) | ⚠️ S0 reset lines | ✅ | ✅ | ↪ ✅ | ✅ |
| 41 | Line anchors, machine-followable phases, exit criteria | ✅ | ⚠️ few anchors | ✅ | ⚠️ phase reorder only | ✅ |
| 42 | Standalone, executable document | ✅ | ✅ | ⚠️ was being edited when read | ❌ amendments to v6, which v7 has replaced | ✅ |
| | **Tally ✅ / ⚠️ / ❌** | **14 / 12 / 16** | **14 / 13 / 15** | **18 / 10 / 14** | **31 / 9 / 2** (v6's included) | **42 / 0 / 0** |

**Choice: v8.**
- It is the only plan with no ⚠️ or ❌ row, and every row is backed by source evidence in §2.3, §2.4 and §3.
- What it takes from each plan:
  - **Sep / Dec:** the architecture. Typed components on an invocation frame, borrowed vs owned stages, fail-closed resume, recursive layering, staged `WorkGraph` removal; Dec's strict claim and its `n_subtasks` contract.
  - **Idem v4:** most of its nine gaps, adopted as found: the claim-first seam, lazy iterators, outcome freshness, the attempt loop, B32, B33, the strict claim, the resume specification, `python_pytest`, P8 required for completion. Four are refined:
    - loop affinity met directly (leaves close in their own loop; the sync ledger closes through `run_async_joined`), instead of a native hook that doesn't exist plus a preflight;
    - a dynamic single-flight guard, instead of a static tree gate;
    - today's workspace precedence, kept and snapshotted instead of flipped;
    - `(scope_id, path)` keys, instead of routing through `ctx.handles`.
  - **Harm:** summary-driven finalize; cleanup never replaces an in-flight exception.
  - **Its own:** the template-method split, B26–B31, B35, B36, frameless semantics, declared invocation keywords, sync bridge ownership, compat declared on the keys, loop affinity.
- All ✅ means v8 addresses every item; it does not mean the risky phases are proven. P7 and P9 are stop points, and P8 needs its own approval.

**If it must be one of the others as-is: Dec v2** (18 ✅, the most). Its defects:
1. the private streaming split (row 9), so `ainfer()` skips the leaf session logic;
2. B26–B34 are missed (rows 10–17, 20–22, 30);
3. a static gate from P4, before any leaf can be certified (row 19), and a claim registry on Tier-2 bindings (row 18);
4. it assumes sync close hooks that don't exist (row 27);
5. no workspace snapshot, attempt loop, lazy-iterator binding or outcome freshness (rows 14, 22, 24, 25).

**Second: Sep v2.** Sound architecture, but ancestor stacking, a per-attempt guardrail window that disables fail-fast, handles routed through `ctx.handles`, few anchors, and no BUCK step. Harm v4 ties with it on ✅ count, but keeps descriptors, shared frames and an owner-keyed guard.

**Idem v4 is not a candidate on its own.** It is "Required Amendments to Snappy v6" and keeps v6 as the chosen plan, so its 31 ✅ are v6's plus its amendments. v6 no longer exists as a document: v7 replaced it in this file, and v8 amends v7 in place. v7 is v6 with those amendments applied, four of them refined, plus B31, frameless semantics and invocation keywords; v8 adds the verified reviewer corrections of §2.3.

---

## 16. Rejected approaches

| Approach | From | Why rejected |
|---|---|---|
| `RunScoped` descriptors with bare write-through | Snappy v5, Harm | hidden mode-dependent semantics; fragile above `@attrs`; preserves accidental bytes, not documented behaviour |
| Canonical live state in node scratch | Dec v1 | the node is path-keyed |
| Owner-keyed guard | Snappy v5, Harm | misses two instances at one path |
| Frames that share values on re-entry | Snappy v5 | removing the re-entry is cleaner |
| Private pipeline split that bypasses public overrides | Sep, Idem v3, Dec | `ainfer()` would skip leaf session, output-file and filter logic |
| Frames and `_active_ctx` held across streaming yields | Harm (explicitly); the first v6 draft | leaks into the consumer; an abandoned stream resets in the wrong Context and leaves a dead frame holding its claim |
| Self-framed OpenClaw `ainfer` (a frame opened inside the leaf) | the first v6 draft | a fourth frame-opening site and an exception to the entry rule. Converging onto `_ainfer_single` is behaviour-neutral once post-init pins the base retry settings (`max_retry=1`, `fallback_mode=NEVER`, no fallback inferencer), with a WARNING for any non-default |
| `weakref.finalize` to release a stream's claim | critique of v6 | a finalizer runs on an arbitrary thread and Context, so it can't reset the tokens it didn't create. The wrapper's `finally` already runs when the generator is closed or finalized, inside its own binding; a second release path would race it |
| Tier-2 claim registry | Dec | the store defines path identity; bindings may differ across roots |
| Ancestor stacking at the same path | v6 | its one cited use (fan-out at the same path) is false: every verified nested call runs at a child path (§2.4). Stacking would only hide real overlaps; a same-path nested public call is refactored instead [Idem] |
| Static certification gate (a worker-only check in P11; a reachable-tree walk in P6; a gate from P4) | v6, Idem v4, Dec | a static walk over config can't see runtime reachability (round-robin lists, factories, configured descendants, one object shared by two parent calls, a host `parallel_infer`). Dec's P4 gate would also break host configs before any leaf could be certified. Replaced by the dynamic single-flight guard at frame open, from P6 |
| Certification derived from measured `KNOWN_DEBT` | critique of v6 | the guard runs in production and can't import test data; the ClassVar declares, the ratchet verifies (§3 row 17) |
| Per-attempt guardrail window | Sep, Idem v3 | fail-fast never fires |
| Generic `AttemptFrame` | Sep, Idem v3 | only BTA needs attempt state; a local suffices |
| Free-form `outputs` dict | Dec v1 | untyped |
| `_COMPAT_SINK: ClassVar[Mapping]` beside the keys | v6 | each fact declared twice (key and sink), so the two can drift; replaced by `RuntimeKey.compat` |
| Rebuilding a component from compat fields on a frameless read | considered for v7 | compat fields are a lossy projection (no `sha256`, `role` or `source_path`); an in-call read outside an invocation is a broken precondition, so it raises `NoInvocationError` |
| Stream observer (and other per-call stage values) in Tier-3 handles or a per-child Tier-2 overlay | v6, Sep, Dec | Tier-3 is connection-scoped, so the observer would outlive its call; an overlay is a new concept. Replaced by declared invocation keywords (§5.9) |
| Routing leaf live handles through `ctx.handles` | Idem v4 (alternative) | breaks leaf `adisconnect` teardown (`_iter_live_handle_sets`), and the ctx bag is shared by every instance at a path; `(scope_id, path)` keying fixes B32 without either problem |
| Flipping workspace precedence (ctx workspace over configured backing) | Idem v4 | silently re-roots every class that configures a workspace, a behaviour change unrelated to purity. The defect is re-evaluation under the active ctx, which the `_BTA_CALL` snapshot fixes (§3 row 22) |
| A deferred-close set for sync owned stages, drained by `adisconnect` | v6 | tracked leakage: many sync callers never disconnect |
| `close_sync` with a native sync hook, else a bridge, plus a loop-affinity preflight | Idem v4 | no sync close hook exists anywhere (`AF/inferencer_base.py:5421-5448`), so the first branch is dead code. Loop affinity is the right requirement (R2#2 confirmed it: the claude_code SDK drops a stale-loop client, `EXT/claude_code/claude_code_sdk_inferencer.py:425-451`), but a preflight can only refuse; it can't close. v8 meets it directly: SDK leaves close per-call clients inside their own `_run_async` loop (codex's `_run_and_close`), and the sync ledger does only loop-independent teardown through `run_async_joined(ledger.aclose())` (§5.1) |
| Cleanup failures only logged after a successful call | v6 | hides leaks; today's fan-out close and BTA `adisconnect` raise. v7 publishes, then raises `InvocationCleanupError` carrying the result |
| Blanket fresh stages | Sep v1, Idem v2 | breaks session continuity |
| Interim reset lines; in-flight bit on the object | Harm S0, Snappy v5 P1 | band-aids |
| `legacy_warn` resume default | Harm | silently resumes unverified work |
| Foreign fan-out worker raises | Idem v3 | contradicts user decision Q3 |
| Removing the `WorkGraph` base together with composition | — | prove composition first |
| Sync cleanup on an untracked event loop or a fire-and-forget thread | — | either could outlive the call. v8's `run_async_joined` runs the close to completion (on a context-copying worker thread when a loop is running) and joins before the call returns |
| An interim per-instance lock to close bare B24 before P7 | R3 gap 2 | a band-aid the user ruled out, and one P7 deletes. Host B24 is already rejected from P6 by the single-flight guard; bare B24 has no production caller (§2.2, §11 stopping table) |
| A provisional summary plus a base "attempt accepted" hook, so a guardrail-rejected BTA run can't leave a summary | R1#4 | its premise can't occur: guardrails are leaf-only (`AF/inferencer_base.py:1486-1499`), the judge is skipped on orchestrators (`:4297-4302`, `:4408-4413`), and orchestrator recovery re-raises (`:4200-4201`), so no BTA run is rejected after it returns. Discarding the summary at run start and putting it as the run's last statement is enough (§5.7, B36) |

---

## 17. Execution notes

- **Base:** the working copy at `69ea2d3762af`, the top of the local `bta_inferencer` stack. New commits stack on top, one concern each, committed by explicit path.
- **No `jf submit` and no push** without approval.
- **Order:** P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 (stop point) → P9 (stop point) → P10 → P11. This is one valid serialization of the §11 dependency table. P8 runs after P6, only once it has its own explicit approval.
- **Completion:** the plan is complete only after P8. Without it, report partial completion, with I4's lease clause and I11 open.
- **Paused:** the fan-out e2e gates G1–G6 (`~/bta_fanout_e2e/`). They are never launched without the user's explicit GO; notifications are not approval.
- **Recommendation for G1–G6, on the user's GO only:** P6 changes the fan-out's aggregator resolution, its ledger and the source of `n_subtasks` (`AF/inferencer_base.py:3333-3349`, `:3444-3469`). So run G1–G6 once against `69ea2d3762af` before P0, as the baseline, and once after P6, comparing against it.
- **After approval:** archive v4.1 as `~/.claude/plans/snappy-doodling-star.ARCHIVE-2026-09-27-bta-fanout-v4.1.md`, recovered from session transcript `bf686ee8-a4e5-4428-942b-6293b7fc6a7d.jsonl`.
