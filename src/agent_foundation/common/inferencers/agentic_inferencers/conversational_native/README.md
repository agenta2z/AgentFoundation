# conversational_native

`NativeConversationalInferencer` hands the conversation to a vendor agent —
Claude Code (Agent SDK or CLI), Devmate `dm`, Codex CLI or Metamate. The
vendor owns the session, transcript, agent loop, tool calling and compaction.
AgentFoundation adds three things through the vendor's own channels:

| Lane | What | Carrier |
|---|---|---|
| **L1** session instructions | identity, how the host works, SOP model + catalog, widget/async rules | the backend's session-instruction channel (system-prompt append on Claude and dm, `developer_instructions` on Codex, inline agent config on Metamate; see `l1_route` in the backend table); session-static |
| **L2** turn context | `<af_context nonce turn generation origin>` with the active / paused SOPs, catalog changes and one-shot notices; sent only when due (see "Turn context (L2)") | a hook (`UserPromptSubmit` additional context) or, where a backend opts in, a labelled envelope before the user's text |
| **L3** state update | after an AF tool changed the SOP state: the full active-SOP block when the SOP identity changed since the agent last saw it (e.g. another SOP became active, the paused / in-progress SOPs or the catalog changed, the vendor compacted), else the active SOP's status and next step | appended to that tool's result, within its size budget (see "Result size") |

AF tools (action tools, SOP control, conversation widgets) are served to the
vendor as the MCP server `af` (`mcp__af__*`). Widgets are deferred: the vendor
turn ends, one (compound) widget is shown, and the answer becomes the next
vendor turn. AF keeps its domain state — SOP state machine, widgets, the
durable session record — and the host contract shared with
`ConversationalInferencer` (`conversational/host_protocol.py`).

Design and decisions: `/home/zgchen/.claude/plans/take-a-look-at-curious-beacon.md`.

## Layout

```
native_inferencer.py   NativeConversationalInferencer: host protocol, commands, record, policies
turn_loop.py           turn driver: sessions, rewind, L2, vendor turns, rounds, widgets, cancellation
turn.py events.py errors.py
context/composer.py    L1 / L2 / L3 rendering (templates: resources/prompt_templates/conversation_native/)
bridge/                tool bridge, schemas, MCP transports (in-process, HTTP, unix socket + stdio relay), CLI hook relay
session/               backends, actor, runtime manager, durable record + stores, backend factory
```

## Backends

`BackendCapabilities` records what each backend can do and the evidence for it
(`verified` = exercised against the real vendor; `experimental` = not yet).
A configuration that needs a missing capability fails at construction, before
any turn reaches a vendor. A backend uses a capability only on `verified`
evidence: when the backend is built, every capability it relies on
(`relies_on`) and a configured non-`inherit` environment must be `verified`,
else it raises `NativeCapabilityError`. `allow_experimental: true` in the
backend spec admits `experimental` evidence (logged as a warning), never
`unsupported` or untagged.

| | Claude SDK | Claude CLI | Devmate `dm` | Codex CLI | Metamate |
|---|---|---|---|---|---|
| process | persistent client | `claude -p` per turn | `dm -p` per turn | `codex exec` per turn | remote (Metamate SDK) |
| AF tools | in-process MCP | localhost HTTP MCP (bearer) | unix-socket MCP via stdio relay | localhost HTTP MCP (bearer env var) | none (tool-less) |
| L1 | preset + `append-system-prompt-file` | `--append-system-prompt-file` | `--append-system-prompt`, every turn | `-c developer_instructions=…` | inline agent config, `system_prompt_mode="append"` |
| L2 | hook | `--settings` command hook | envelope | envelope | envelope |
| turn stop | `PostToolUse continue:false` | command hook | `AF_END_TURN` directive | `AF_END_TURN` directive | n/a |
| subagent AF calls refused | PreToolUse `agent_id` | hook `agent_id` | structurally: dm-core gives `--mcp-servers` to root agents only | — | n/a |
| session | pinned id, resume | pinned id, `--resume` | pinned id, `--resume` | recorded thread id, `exec resume` | conversation id |
| exact rewind | `fork_session` ✔ | `fork_session` on the transcript ✔ | — (recap) | — (recap) | — (recap) |
| `/compact` passthrough | ✔ | ✔ | — | — | — |
| environment | inherit, hermetic | inherit, hermetic | inherit | inherit, hermetic | inherit |
| tested on | claude 2.1.289 + claude-agent-sdk 0.1.58 | claude 2.1.289 | dm 2026.10.03-0249 | codex-cli 0.159.3 | `//msl/metamate/sdk` b037b06c2d47 |
| evidence | verified: tools, hooks, L2, resume, fork, compaction, subagent `agent_id`, hermetic, `af` health, `MCP_TOOL_TIMEOUT` | verified: tools, hooks (fail closed), L2, resume, fork, compaction, subagent `agent_id`, hermetic, `af` health, `MCP_TOOL_TIMEOUT` | verified: transport, MCP tool timeout, resume (scripted model); hooks and hermetic unsupported | verified: tools, L1, resume, hermetic, `af` health (`required`), `tool_timeout_sec` | verified: L1, resume, envelope; interrupt unsupported |

Caveats:

* **Devmate**: dm-core stalls a real-model turn when an MCP server is attached
  unless the model promptly calls a tool. dm's workflow mode
  (`DM_CORE_WORKFLOW=1`) removes the stall but requires CAT credentials
  (`--cats-file`; see `session/devmate_dm.py:resolve_cats_file`). Without them
  the backend is not usable with AF tools; hosts should report it unavailable.
  Workflow mode loads no user or additional hooks, so dm can have no
  hook-based L2, turn stop or deny. dm resolves its build from the cwd.
* **Metamate**: the Metamate SDK is Buck-only. Add the target
  `…/conversational_native/session/metamate:metamate` (it carries
  `//msl/metamate/sdk`) to the host binary; without it the backend factory raises
  `NativeCapabilityError`. The append is fixed when a
  conversation is created, so changes reach the model as L2 notices. Its legacy
  `internal_prompt` is shown to the model as pasted text and distrusted; it is
  not used. Replies arrive whole (no token streaming); interrupt only stops
  reading (the SDK cannot cancel a running request).

  *Stalls.* Metamate sometimes stops answering a request server-side; this
  reproduces with the SDK alone, without AgentFoundation code, on about 1 in 5
  resumed requests. The backend passes `extra.idle_timeout_s` (default 120 s;
  `backend/metamate.yaml`) to the SDK, whose own default is 300 s. After that
  long with no new output block (any block, tool calls included) the turn fails
  `uncertain`: the request may still finish server-side, so it is not re-sent,
  and the next turn gets a `turn_failed` notice.

  *Memory.* Metamate's server-side persistent memory (core, personal and
  session memory) belongs to the user, not to one conversation. A new session
  after `/new` or `/clear` starts without the conversation's history, but
  anything Metamate saved to that memory can come back in a later session.
  This is a vendor property that AF cannot isolate (one reason `hermetic` is
  unsupported). In the OpenStartup run, fresh sessions searched that memory
  first and recalled nothing; the codewords had been marked "do not save".

  *OpenStartup.* Run it with the `:server` binary, which carries the SDK:
  `buck2 run @fbcode//mode/dev //_tony_dev/CoreProjects/OpenStartup/src:server -- --real-sessions <dir> --llm-backend native_metamate`.
  A source checkout lists `native_metamate` as unavailable.
* **Codex** emits whole messages (no token deltas).

## Spike results (`scripts/native_spikes/`)

Final evidence run 2026-10-03 on the final code: Claude Code 2.1.289 (SDK
0.1.58), Codex CLI 0.159.3. Each script prints `[PASS]/[FAIL] <check>` and
exits non-zero on a failure (S11 prints one `[s11 PASS]`/`[s11 FAIL]` line,
S16 and S16b print answers, S17 measures); it asserts on transport evidence
(stream events, hook inputs, MCP calls received, the Claude transcript JSONL)
wherever possible. `n/m` is checks passed / run. Run logs are kept outside the
repo (`~/native_spike_logs/final_<script>_<YYYYMMDD-HHMM>.log`; S7, S11 and S14
re-run after that as `z2_<script>_…`; the earlier S17 logs as named).

| # | Question | Result |
|---|---|---|
| S1 | L1 append frozen across resume and `fork_session`; re-recorded after compaction | partial (`s1_l1_snapshot.py` 26/26, `final_s1_l1_snapshot_20261003-1839.log`, CLI + SDK): the first request records a `prompt_snapshot` (preset + append); a resume and a resumed `fork_session` send that record although the L1 file was rewritten (frozen ✔); `/compact` re-records from the append text the *process* read at launch, so a per-turn CLI or an SDK resume picks up a rewritten file but a running SDK process does not. Native (`s1_s3_s5_s8_claude.py --only S1,S3,S5,S8` 61/61, `final_s1_s3_s5_s8_claude_20261003-1842.log`, SDK + CLI): persona holds across a restart; drift arrives as a notice, session kept |
| S2 | `UserPromptSubmit.additionalContext`: visible, not user-attributed, persisted, size | ✔ SDK + CLI (`s2_s4_s12_claude_cli_hooks.py` 50/50, `final_s2_s4_s12_claude_cli_hooks_20261003-1845.log`): visible; stored as a `hook_additional_context` transcript attachment rendered as a `<system-reminder>`, never in the user message; persisted (a later resumed turn without context still knows it, so every L2 block stays in history); the model attributes it to the hook, not the user; up to 10,000 chars inline, 10,001+ chars saved to a file and replaced by a ~2 KB preview + path |
| S3 | compaction observable; next turn re-sends L2 | ✔ (`s1_s3_s5_s8_claude.py`, 61/61 above): `/compact` passes through and the next turn re-sends L2 (SDK + CLI); after an L1 drift the post-compaction snapshot carries the new L1 on both (the SDK only since a drift reopens its process, see "Instruction drift") |
| S4 | `PostToolUse continue:false` ends the turn; resume works | ✔ SDK + CLI (`s2_s4_s12_claude_cli_hooks.py`, 50/50 above): the AF tool runs once, nothing follows, result `success` (`stop_reason` `tool_use`; CLI `terminal_reason` `hook_stopped`), the tool_use has its tool_result; the next resume works and remembers the result |
| S5 | cancel during an AF tool call; `MCP_TOOL_TIMEOUT` | ✔ cancel (`s1_s3_s5_s8_claude.py`, 61/61 above): SDK and CLI `interrupted`; next turn works, tool not re-run. The CLI was `uncertain` until each per-turn CLI ran in its own process group, which a cancel ends: the `claude` launcher's child kept the turn's pipes open, so the interrupt was never acknowledged. Since then: `interrupted`, no surviving process (`test_real_integration/test_real_native_claude_cli.py`, 34/34, `final_real_native_claude_cli_20261003-1939.log`). Timeout (`s5_mcp_tool_timeout.py` 8/8, `final_s5_mcp_tool_timeout_20261003-1850.log`, SDK in-process + HTTP): 3 s vs a 20 s tool: after ~3.5 s the model gets an `is_error` result "MCP server "af" tool "slow_lookup" timed out after 3s" while the handler keeps running to completion (not cancelled); 7,200,000 ms (the native default): completes |
| S6 | AF tools under tool search; `af` under managed settings | ✔ (`s6_tool_search_managed.py` 20/20, `final_s6_tool_search_managed_20261003-1853.log`, SDK + CLI): with 14 and 40 AF tools every `mcp__af__*` tool is deferred behind ToolSearch (first request ≈ 23K prompt tokens either way); the model still finds `enter_sop(name=model_optimization)` and `single_choice` from task wording; `af` connected (init event, SDK `get_mcp_status`). Default managed profile: no `allowManagedMcpServersOnly` / `allowManagedHooksOnly`. Sensitive profiles (read from `/etc/claude-code`, not run): all three set `allowManagedMcpServersOnly` (`af` is not allowlisted), `-high` / `-low` also `allowManagedHooksOnly` (the native hooks would be ignored), `-high` disables `bypassPermissions` |
| S7 | SDK 0.1.58 + system CLI: preset + append-file, in-process MCP, hooks, `get_mcp_status`, pinned `cli_path` | ✔ (`s7_sdk_cli_pin.py` 12/12, `z2_s7_sdk_cli_pin_20261003-2133.log`, haiku): one SDK session configured like the backend, with `cli_path` pinned to a wrapper that records its argv and execs the system `claude`: the SDK started the pinned path, and the CLI that answered reports 2.1.289 (the SDK's bundled CLI would be 2.1.97; no bundled binary is installed here); argv has `--append-system-prompt-file`, no `--system-prompt` (preset kept), the pinned `--session-id` and the SDK `af` server in `--mcp-config`; `get_mcp_status` lists `af` connected before the turn and `init` lists `mcp__af__echo`; the `UserPromptSubmit`, `PreToolUse` and `PostToolUse` callbacks fired; the in-process handler ran and its result was the reply; the transcript's `prompt_snapshot` holds the preset and the append |
| S8 | hermetic: auth OK, no user CLAUDE.md / hooks / MCP servers | ✔ SDK + CLI (`s1_s3_s5_s8_claude.py`, 61/61 above, canaries in a temp `CLAUDE_CONFIG_DIR` and the project): `inherit` loads user + project CLAUDE.md, runs the user hook, starts the user MCP server; `hermetic` loads only the managed instructions, runs no user hook, starts no user server, and authenticates |
| S9 | two AF action calls in one message; event ordering | ✔ SDK + CLI (`s9_parallel_actions.py` 8/8, `final_s9_parallel_actions_20261003-1855.log`): both tool_use blocks in one API message (one `AssistantMessage` per block); both handlers run once, sequentially (no overlap); per call `AssistantMessage` → `PreToolUse` → handler → `PostToolUse`, and the second block's message arrives before the first handler starts |
| S10 | pinned id + resume across a restart; cwd binding | ✔ (`s10_cwd_binding.py` 10/10 on the re-run, `final_s10_cwd_binding_rerun_20261003-1917.log`, CLI + SDK): a pinned id resumes after a restart; a resume from another cwd continues the session (memory kept), runs in the new cwd and appends to the transcript under the original project dir — not cwd-bound. The first run was 9/10 (`final_s10_cwd_binding_20261003-1855.log`): the CLI resume from cwd B was chained onto the session's last message as usual, but haiku answered `NONE` because it read the transcript's `environment` attachment (working directory changed) as a new session. That is a model judgment; the transport behaved the same |
| S11 | `--mcp-config` reaches a localhost HTTP server under the Meta launcher | ✔ (`s11_http_mcp.py`, `z2_s11_http_mcp_20261003-2111.log`): `claude -p --mcp-config` pointing at a 127.0.0.1 streamable-HTTP server with an `Authorization: Bearer` header: the tool ran once and its result was the reply (rc 0). Control: with a wrong token the server answers 401, no call reaches it, and claude reports `af` failed to connect |
| S12 | subagent `agent_id` in PreToolUse | ✔ SDK + CLI (`s2_s4_s12_claude_cli_hooks.py`, 50/50 above): a Task subagent's AF call carries `agent_id` (+ `agent_type`); the main thread's has none |
| S13 | `fork_session(up_to_message_id)` → exact rewind, remapped boundaries | ✔ SDK + CLI (`s13_claude_sdk_fork.py` 6/6 each, `final_s13_claude_sdk_fork_{sdk,cli}_20261003-1858.log`): forked session forgets later turns; second rewind of the fork works |
| S14 | dm session pinning/resume, MCP, events | ✔ with dm's scripted model (`s14_dm_socket_mcp.py` 12/12, `z2_s14_dm_socket_mcp_20261003-2130.log`, dm 2026.10.03-0249; fixtures indexed by the number of tool results in the request dm assembles, tools really run): the pinned `--session-id` is the id of every session event; `--resume` continues that session (same id, and the resumed request holds turn 1's tool result: the turn-2 fixture makes a second AF call and answers `HISTORY-OK` only then, while the same fixture in a fresh session answers `NO-HISTORY` — the control); the events the backend maps carry the fields it reads (`session_start/update/end` `session.id`, `step_start/end`, `action_end` `llm_action` `output.info` and `tool_use_action` `tool_use_id`/`tool_name`, `session_end.exit_code` `COMPLETE`); the AF tool round trip works over the stdio relay to the 0600 unix socket (the production path; dm exits by itself) and over dm's direct socket transport. Not asserted: real-model turns (they need workflow mode, i.e. CATs; `test_real_native_devmate_dm.py` skips here: no CAT file), the append after a resume, hooks |
| S15 | Codex `developer_instructions`, HTTP MCP + bearer env var, resume, hermetic auth | ✔ (`s15_codex_http_mcp.py` 15/15, `final_s15_codex_http_mcp_20261003-1900.log`, temp `CODEX_HOME`): AF tool over HTTP MCP with the bearer env var, `mcp_tool_call` items, L1 followed; `--ignore-user-config --ignore-rules` authenticates on fresh and `exec resume` threads and starts no user `config.toml` MCP server; the user `AGENTS.md` still loads (outside those flags) |
| S16 | Metamate per-request instructions | Buck binaries. `internal_prompt`: distrusted (✘; not followed, not persisted, a changed one has no effect; `s16_metamate_internal_prompt`, `final_s16_metamate_internal_prompt_20261003-1933.log`); inline append: ✔ followed, uses host state, resumes by id, fixed per conversation (a changed append has no effect; `s16b_metamate_inline_agent`, `final_s16b_metamate_inline_agent_20261003-1935.log`); a native Metamate conversation (codeword, SOP by slash command, SOP state used, same conversation resumed) 6/6 (`s16c_metamate_native`, `final_s16c_metamate_native_20261003-1936.log`) |
| S17 | classic CI re-sends history into a continuing vendor session | ✔ measured on classic CI, not native (`s17_ci_nonstreaming_duplication.py`, haiku — Sonnet 5.5 refuses the classic prompt; turn 1 takes 2 CI rounds, turn 2 recalls a codeword). Before (`s17_knob_off_2026-10-03.log`, `fresh_vendor_session_per_round: false`): CLI non-streaming without a run context resumes every round and across turns (1 session, 3 rendered prompts, 30,782 chars); CLI non-streaming and SDK streaming with a per-turn host continue within a turn (2 sessions, one holding 2 prompts, ~20.2K chars); CLI streaming is fresh every round. After (`s17_after_fix_2026-10-03.log`, `fresh_vendor_session_per_round: true`): 3 sessions × 1 prompt (~10K chars) in all four scenarios; turn 2 still answers the codeword. Final run (`final_s17_ci_nonstreaming_duplication_20261003-1901.log`, `true`): the same in A–D (3 sessions × 1 prompt, 9,936–10,661 chars; codeword recalled). A fix exists in classic CI, opt-in (default `false`, pending the user's decision): see below |
| S18 | two widget calls in one message | ✔ one compound widget (clarification + single_choice) on SDK, CLI, Codex. SDK and CLI: the host checklist's C8 (`test_e2e_native.py`, sonnet; `z2_e2e_native_claude_sdk_20261003-2132.log` 82/82, `z2_e2e_native_claude_cli_20261003-2149.log` 81/81): `model_optimization` Phase 0a's `clarification` and `single_choice` reach the UI as one compound widget after the vendor turn (on Claude, widgets are grouped by assistant message id, so one compound widget means one message), both answers are applied, and the next vendor turn's context has Phase 0b active. Codex (`s18_codex_compound_widget.py` 8/8, `final_s18_codex_compound_widget_20261003-1904.log`): both calls of one `exec` script carry the same `_meta.itemId`, a later response's call another, and `--json` shows the same events between the calls in both cases; the native SOP run shows one compound widget (before the per-item rule it split into two: the second question was refused) |
| G3 | classic CI vs native: same 3-turn script (chat, SOP entry with 2 answered widgets, recall), haiku | ✔ 33/33 twice (`g3_native_vs_classic.py`, `w_g3_full_{1,2}.log`, 2026-10-03). Classic: 8 fresh sessions, each prompt re-rendering the history (all 8 carry turn 1's text); native: one vendor session, each user text sent once verbatim, none carries an earlier one, codeword recalled. Wall time 176.5 / 123.9 s classic, 68.6 / 74.6 s native CLI, 36.8 / 40.6 s native SDK; input-equivalent tokens (input + 1.25 × cache write + 0.1 × cache read) 102.8K / 102.7K, 52.4K / 61.4K, 55.3K / 53.2K; output tokens 9,999 / 3,832, 1,788 / 1,983, 2,170 / 2,516. Haiku's choice between the question tool and prose after `enter_sop` varies run to run, so it is measured with `--repeat`/`--templates` (logs `w_g3_<L1 variant>_native_sdk_<n>.log`): Phase 0's question went through `mcp__af__clarification` in 34/36 native SDK runs and 5/5 native CLI runs with the current L1 ("Asking the user": an SOP phase's input is asked with the question tool its guidance names, open questions included), against 28/48 and 3/5 with the earlier rule ("whenever you need structured input") and 25/33 with the wording before the "SOP control" section. The earlier ✘ runs (`final_g3_*_20261003-19*.log`) also counted runs whose agent exited the finished SOP; the spike now reads Phase 0 from the suspended SOP too |

S17 measured the duplication native avoids by construction: a continued vendor
session holds every rendered prompt, each a full copy of the conversation.
Classic CI has a fix, opt-in: with `fresh_vendor_session_per_round: true`
(`resources/configs/conversational/default.yaml`, or the
`ConversationalInferencer` attribute), before each round's base call
(streaming and non-streaming) `ConversationalInferencer` calls
`base_inferencer.areset_conversation(run_context=<the round's context>)`, a
per-branch reset on every leaf family (`new_session=True` would reach the AI
Gateway API request bodies, and the SDK leaves' `adisconnect()` drains every
branch of a shared instance). Measured (S17 row): every round then reaches a
fresh session holding one rendered prompt (~10K chars), where a continued
session held up to 3 (30,782 chars), and turn 2 still recalls the codeword.
The trade-off: a continued vendor session is the only carrier of the vendor's
own built-in tool results (e.g. Claude's Read/Bash) between rounds; a fresh
one keeps only what the rendered prompt carries. It changes the classic
route's behavior (plan §15: needs the user's OK; invariant 11: classic is
unchanged by default), so the default stays `false` (the continuation) until
the user decides.

Host checklist (plan §12.3) against OpenStartup:
`OpenStartup/test/openteam/server/backends/test_e2e_native.py --backend <name>`
— chat, SOP catalog, cancel mid-stream, server restart (same session),
resume-from-turn (vendor forgets dropped turns), `/compact`, `/new`, an SOP
compound widget answered after a page refresh, backend switch (SOP kept, recap
on return), delete, graceful shutdown. Passing: `native_claude_sdk`,
`native_claude_cli`, `native_codex`, and `native_metamate` in tool-less mode
through the `:server` binary (42/42; the 5 checks that need AF tools are
skipped).

## Configuration

`resources/configs/conversational_native/default.yaml` (policies) +
`backend/<kind>.yaml` (`NativeBackendSpec`), loaded by
`resources/tools/_ci_host.build_native_from_config(path, backend=<kind>, ...)`.
Key policies: `on_l1_drift` (notice | rotate | fail), `on_session_loss`
(recap | fresh | fail), `rewind_on_repeat_turn`, `on_rewind_unsupported`
(fail | recap), `vendor_drain_timeout_s`, `vendor_stall_timeout_s`,
`max_vendor_turns_per_call`, `idle_close_seconds`. Tool surface:
`sop_control_tools`, `expose_tool_argument_form` (see "AF tool bridge").
Backend-specific options go under the spec's `extra`; for example, Metamate
takes `api_key`, `surface`, `entry` and `idle_timeout_s`.

Host integration points (beyond `ConversationalHost`):

* `runtime_manager` — inject one `NativeRuntimeManager` per host so live vendor
  sessions survive evict-then-rebuild; `aclose_all()` at shutdown.
* `record_store` — where the durable record lives (`InMemoryRecordStore`,
  `CallbackRecordStore`, `RunContextRecordStore`; OpenStartup keeps it in
  `session["native_session"]`). `end_vendor_session(store, key)` retires a
  session without an inferencer (e.g. the host switched backend).
* `rewind_to(turn)` — call before truncating host history; raises
  `RewindUnsupported` (policy `fail`) and leaves the session intact.
* `accepts_command(text)` — slash commands the orchestrator (or the vendor,
  e.g. `/compact`) handles.
* `supports_round_resume = False` — resume a native turn as a whole.

## Submission safety and cancellation

The record tracks each vendor turn: `prepared` (not sent) → `submitted` (the
vendor may hold it) → `committed`. A turn that does not complete ends as
`interrupted` (a cancel the vendor confirmed) or as `uncertain` (a cancel it did
not confirm, or a failure after the turn may have reached the vendor). Nothing
that may have reached the vendor is sent again. The native inferencer sets
`max_retry = 0`. The only automatic re-send is a turn whose session the vendor
reports missing: the vendor never accepted that turn, and it goes to a fresh
session with a recap. A host may safely send a `prepared` turn (spawn or connect
failure) again. Instead of a replay, the next turn's L2 carries one-shot
notices: `interrupted` after a cancel (also for a record still `submitted` when
the next host call starts, which means the host died during that turn),
`turn_failed` after a failure that may have reached the vendor, and
`widget_unshown` for a question the cut-off turn had queued.

* **Cancel.** A host cancel interrupts the vendor turn and drains it, both
  bounded by `vendor_drain_timeout_s` (30 s). If the turn does not drain, its
  session is closed: the actor retires and closes the backend (the SDK
  disconnects, which ends the `claude` process), so nothing the vendor still
  produces reaches a later turn. The next turn opens a new session that resumes
  the vendor session by id (the SDK reports the session at `init`, so a first
  turn cut off during its first request resumes too). The per-turn CLIs run
  each turn in its own process group. A cancel ends the group (SIGTERM, then
  SIGKILL after 5 s), so the vendor's child processes stop with the launcher,
  and the cancel is acknowledged.
* **Stall.** If the vendor sends no event for `vendor_stall_timeout_s`
  (1800 s) while no AF tool is running, the turn is interrupted and fails
  `uncertain`. Metamate's idle timeout (120 s by default) fires first (see
  Caveats).
* **Turn cap.** `max_vendor_turns_per_call` (50) can end a call before content
  reaches the agent: a widget answer (already applied) or the error of a widget
  batch that failed validation. The mirror then records that text, and the next
  host turn's L2 delivers it once as an `unsent_turn` notice. The notice body is
  kept in memory; after a restart it is read back from the mirror. The result
  reports `exhausted_max_iterations`.
* **View Prompt.** A turn's prompt manifest is recorded before the turn is
  submitted. After a failed, stalled or cancelled turn, `last_prompt_data()`
  shows that turn with a `## Turn outcome — failed | uncertain | interrupted`
  section and `template_feed["turn_outcome"]` (plus `turn_outcome_detail`). A
  host that saves prompt data only per completed round must also save it on its
  error path.

## Resume policy

Before a persisted vendor session is contacted again, the record is checked in
this order. The invariant: a vendor session is never continued across an
incompatible fingerprint. Plan §7.3 lists backend, adapter and model mismatches
as "fail closed" while D9 makes a backend switch a recap by default; a session
that is not continued (D9 starts a fresh one with a labelled recap) does not
violate the invariant, and a host that wants a hard stop sets
`on_session_loss: fail`.

| Recorded vs current | Behaviour | Why |
|---|---|---|
| principal differs | fail closed (`SessionResumeRejected`) | identity; checked first, so not even a recap of another principal's history reaches a new session |
| backend differs | session loss per `on_session_loss` (recap / fresh / fail) | another vendor cannot continue the session (D9 backend switch) |
| adapter version differs | session loss per `on_session_loss` | the stored coordinates mean something else to this adapter: the old session is unusable |
| cwd differs | session loss per `on_session_loss` | a policy, not a vendor limit: a Claude session resumed from another directory keeps its memory, runs there and gets a new `environment` snapshot naming it (S10); but its earlier turns' files belong to the other tree, and the other backends' directory binding is unverified |
| permission policy differs | fail closed until `/new` | checked only for a session that would be continued: a session may not continue under another policy; a new one (after a loss) starts under the current policy |
| model differs | the session continues on the new model (§7.3 `/model`: SDK `set_model`, CLIs from the next process); a switch the vendor refuses reopens the session, which starts on the new model; the record is updated | the vendor supports switching models within a session; failing would force `/new` for a routine change |
| L1 core hash differs | `on_l1_drift` (D8, below) | |

User commands are not losses. `/new`, `/clear` (which also clears the mirror)
and `/root <path>` rotate the session themselves — a new session the user asked
for — so `on_session_loss: fail` does not refuse them and the next turn starts
the new session (in the new cwd for `/root`). `/root` carries the conversation
over as a recap under `recap`; `/new` and `/clear` start without one; `/root`
at the current root keeps the session, and it never rotates a session another
principal started (the identity check refuses that turn). `/model <name>`
records the model; the next turn applies it as in the "model differs" row.

## Instruction drift

A changed `l1_core_hash` on an existing session is handled per `on_l1_drift`
(D8). The hash covers the session instructions and the AF tool set (tool names
and input schemas), not the SOP catalog or the nonce: a catalog change is not
drift (L2 reports it as `catalog_changes`) and keeps the live session. Under
`notice` the session is kept: changed instructions reach the model once as a
superseding L2 notice (`instructions_updated`, the whole new L1), and the L1
file is rewritten so that the vendor's next compaction records it; a change of
the tool set alone gets a `tools_updated` notice instead. A vendor process that
outlives turns read its L1 and connected its tool server when it was opened —
the Claude SDK process reads the append file at launch and re-records that text
at `/compact` (S1); dm, Codex and Metamate hold the text they were opened with
— so a live session opened with other instructions or another tool set is
reopened by resume (same vendor session id and history).

## Turn context (L2)

An L2 is one `<af_context nonce turn generation origin>` block: the active SOP
(or "No SOP is active"), the paused and in-progress SOPs, catalog changes since
the session's L1 snapshot, and one-shot notices. Each block is complete; the
newest generation supersedes earlier ones. It is sent only when due
(`turn_loop._l2_due`):

* on the first turn of a vendor session: new, forked or rotated;
* when the SOP state differs from the last one the session received (the state
  hash of its last L2 or L3, `record.l2_hash`). That hash is cleared wherever the
  vendor may no longer hold the state: a compaction, a session reopened by
  resume (drift, a changed tool set, a refused model switch), and a vendor that
  started a new session instead of resuming;
* when notices are pending;
* on host-written turns (`validation_retry`, `tool_completion`, `host_event`,
  `resumed_turn`), whose L2 states the origin; user text and widget answers
  speak for themselves;
* on every turn of a channel the vendor does not keep in its history (the
  per-request config channel; no backend uses it today). Hook context and
  envelopes stay in the transcript (S2), so a session continued after a process
  restart is not sent the state it already holds.

**Slash commands.** Our commands (`/help`, `/status`, `/sop`, `/model`, `/new`,
…) are handled locally. A command in the backend's `slash_passthrough`
(Claude: `/compact`, `/context`, `/cost`, `/usage`) is sent verbatim without an
L2. Any other `/…` is an ordinary message, as typed, with the L2: Claude Code
treats unknown names and paths as text, and expands user commands and skills
into a prompt its `UserPromptSubmit` hook sees. Only its local commands
(`/compact`, `/context`, `/cost`/`/usage`, also `/release-notes`, which only
says it is not available under `-p`) skip the hook and the model (claude
2.1.289, CLI and SDK); its `/help` is never reached, ours answers. An L2 sent
via the hook counts as delivered only if the hook ran, so one sent with an
unlisted local command stays due. Codex `exec` treats `/x` as text.

**Budget** (UTF-16 code units, as the vendor counts): hook 9,000 (Claude Code
inlines hook context up to 10,000 characters, S2); envelope and per-request
config 16,000. The SOP state comes first; it is cut when it exceeds the
budget, less 2,000 units kept for notices if any are pending. Then the
notices: whole where they fit, newest first; the rest cut to equal shares of
the room left (a recap keeps its first line and its end, the others their
head), newest first; any for which not even a 400-unit excerpt fits are
bundled into one file named by a last `more_notices` notice. Every pending
notice is delivered, inline, cut or by file. A cut text says where it was cut
and names the file with its full text under `<session dir>/turn_context/`
(0700 directory, 0600 files); logs carry sizes only.

## Hook liveness (Claude SDK, Claude CLI)

Claude Code drops AF's hooks without a trace in its output when a managed
policy allows only managed hooks (`allowManagedHooksOnly`, set by the
sensitive profiles) or settings disable hooks (`disableAllHooks`); the turn
would run without its L2, turn stop and subagent guard. The effective managed
settings cannot be read up front — the Meta launcher composes them per run
(presets, Configerator, sensitive launchers) and mounts them over
`/etc/claude-code/managed-settings.json` in the vendor's namespace — so the
backends check behaviour (`session/claude_hooks.py`): every turn on the hook
channel, a passed-through local command excepted (Claude Code runs it without
`UserPromptSubmit` and without the model), must reach AF's `UserPromptSubmit`
hook before the model starts, else the turn fails with an actionable error.
That includes a turn whose L2 is not due: the hook runs on every turn and adds
context only when there is some. The SDK checks at `init` (Claude Code awaits
the callback before `init`); the CLI at the first model output (a relay that
cannot reach AF blocks the prompt, and Claude Code reports that block only
after `init`). Any other `/…` prompt is checked at the first model output on
both: a local command the backend does not list (`/release-notes`) also runs
without the hook, but never reaches the model (its output is a `<synthetic>`
message); with a project `disableAllHooks`, `/foo` is stopped when the model
starts (claude 2.1.289). The prompt
passed the hooks, so it is already in the transcript: the turn is recorded
`uncertain`. Evidence, claude 2.1.288: a project
`disableAllHooks` drops the CLI's `--settings` hooks and the turn is stopped;
SDK callback hooks are not affected by `disableAllHooks`; `allowManagedHooksOnly`
cannot be exercised without a managed profile.

## AF tool bridge

The MCP server `af` offers (`bridge/tool_bridge.py`, schemas in `bridge/schema.py`):

* **action tools**: registry tools of type `Action` with `agent_enabled`;
* **SOP control**: `enter_sop`, `resume_sop`, `pause_sop`, `exit_sop`,
  `sop_status`, filtered by `sop_control_tools` (default all five; `[]` leaves
  SOP control to the user's slash commands; an unknown name fails at
  construction). A tool left out is not offered; a call to it is refused as an
  unknown AF tool;
* **questions** (widgets): `clarification`, `single_choice`, `multiple_choice`,
  `confirmation`, `proposal_selection`, and `tool_argument_form` when
  `expose_tool_argument_form` is set (default off; it has no `tool.json`: one
  free-text answer, bound to `output` and feeding `then_run` like any question).

**Argument names.** An action tool's MCP argument is its parameter's executor
key (leading dashes stripped, hyphens → underscores, the hosts' tool-argument
convention: `--workflow-target-path` → `workflow_target_path`), made valid for
vendor schemas (`^[a-zA-Z0-9_.-]{1,64}$`; a longer name is shortened with a
hash). A call may also use the CLI spelling the tool docs show (`--docs-path`,
`docs-path`): the bridge canonicalizes the arguments, validates them against the
schema and passes them to the executor through the reverse name map (MCP name →
executor key; arguments the tool does not declare by the same convention, as
some executors read keys their `tool.json` omits). Two arguments naming one
parameter are an `Invalid arguments` error. Subcommands flatten behind a
required `subcommand` enum; a name declared in several places must be the same
parameter (type, choices, default), and any other collision fails when the
manifest is built.

**Ending a turn.** A queued question, a background dispatch, a dashboard
handoff or a pause closes the turn's gate, and the call's result starts with
`AF_END_TURN — …` (a truncated or spilled result still leads with it). The
Claude backends stop the vendor turn from `PostToolUse` (`continue: false`)
once the message's last AF call finished; Codex and dm rely on the directive.
While the gate is closed, further AF calls are refused with a result that is
not an error and starts with `AF_END_TURN — not run …` (`… — not queued …` for
a question from a later message): a refusal ends the turn, it is not a tool
failure. That matters on Claude Code, which runs `PostToolUseFailure` instead of
`PostToolUse` for an error result and ignores `continue: false` from it (claude
2.1.288); the backends register it only to mark the call finished. Limit: when
a stop is due and the message's last AF call returned a real error (invalid
arguments, a failing tool, a vendor timeout), no hook can stop the turn; the
gate refuses every further AF call (each refusal can stop it) and the earlier
end-turn result asks the model to end its turn. A call outside a live turn is
an error (Claude's `PreToolUse` denies it first).

**Result size.** An AF tool result stays within `native_tool_result_max_chars`
(default 16,000) UTF-16 code units, the unit of Claude Code's
`maxResultSizeChars`. The end-turn directive leads and the L3 state update ends
the result, both whole; the tool output gets the room left, cut where the result
says so, with the whole output in a private file it names
(`<session dir>/tool_results/`, 0600). A state update that does not fit beside
that cut is cut and spilled the same way. The Claude SDK backend declares the
limit as the tools' `maxResultSizeChars`, and Claude Code spills a larger result
itself, showing the model only a 2,000-character preview and the file path: that
keeps the directive but would hide a trailing state update, so the vendor gets a
result whole only when it carries no state update. The other backends' results
are always sized by the bridge, and the Claude CLI's HTTP tools declare the limit
too (`_meta["anthropic/maxResultSizeChars"]`), so Claude Code leaves a result the
bridge sized whole. An undeclared MCP tool's result would be spilled above
50,000 characters or above Claude Code's MCP output token limit
(`MAX_MCP_OUTPUT_TOKENS`, 25,000 by default), which a declaring tool skips
(claude 2.1.288: a 60,000-character result was spilled undeclared and arrived
whole under a declared 100,000; one of 100,001 was spilled). Claude Code caps a
declared limit at 500,000, so both Claude backends refuse a larger
`native_tool_result_max_chars` at construction (`ValueError`).

**Compound questions.** Once a question is queued, question calls from the same
assistant message join it (one compound widget after the turn, as classic CI
batches them); a question from a later message and any other AF call are
refused. Which message a call belongs to depends on the backend:

* Claude SDK / CLI — exact: `PreToolUse` announces each AF call's tool use, and
  the message events carry each tool use's API message id (the bridge waits up
  to 2 s for that event).
* Codex — exact per model output item. Codex reaches MCP tools only from a
  code-mode `exec` script, and every `tools/call` carries `_meta.callId` (the
  call) and `_meta.itemId` (the `exec` call that made it); the backend announces
  each AF call as an AF message of that item before the call runs. So the
  questions one script asks (`Promise.all([...])`; Codex still runs them one
  after the other) form one compound widget, and a question from a later model
  response — another item — is refused. `--json` alone cannot tell the two
  apart: it reports neither the `exec` item nor response boundaries, only each
  MCP call once it finished, with nothing in between in either case. A call
  without `_meta.itemId` is a message of its own, and two `exec` items in one
  response (not observed) would count as two messages: both refuse rather than
  merge (`s18_codex_compound_widget.py`, codex-cli 0.159.3).
* dm — a fallback: its calls are not attributed (no pre-call hook), so calls
  join only until the vendor reports an AF message after the first question
  was queued. dm reports a step (one model response) only after all of its tool
  calls ran (a step's calls run concurrently; scripted-model capture, dm
  2026.10.03-0249), so a step's questions join and a later step's are refused.

**Results that come too late.** Every tool-capable backend gets the spec's
`mcp_tool_timeout_ms` (Claude `MCP_TOOL_TIMEOUT`, Codex `tool_timeout_sec`, dm
`toolCallTimeoutMs`). When it passes, the vendor tells the model the call failed
while AF's handler runs on (S5), so a call counts as timed out slightly earlier
(by min(1 s, a tenth of the timeout)) and is then not applied: no context
updates, phase completion, recorded action or state update; a call whose
deadline passed while it waited for another is not run. The next turn's L2
carries a one-shot `late_tool_result` notice naming the tool and the timeout,
whether it ran, that its result was not applied, and its output (kept in memory
only: after a restart the notice says it was not retained). A yolo question's
bundled `then_run` action follows the same rule, and a result returning after
its turn ended is not applied either.

## Design deviations

Where the code departs from the plan, and why:

* **Actors per generation (§7.2).** The plan has one `SessionActor` per
  conversation, with `fork` and `rotate` commands and the active `TurnScope`
  inside the actor. The runtime manager keys actors by `(conversation, backend
  kind, generation)` instead, and an actor takes only `turn`, `set_model` and
  `close`. A rewind fork, `/new`, `/clear`, `/root` or a session loss moves the
  record to the next generation (`NativeSessionRecord.rotate`,
  `turn_loop._rewind`); the next turn opens that generation's actor (a fork
  opens on the forked transcript), and opening it closes the older
  generations' actors, so a stale generation cannot act. The active
  `TurnScope` is the inferencer's `current_turn`; the backend's tool handlers
  and hooks reach it through `session/binding.py`, which the turn driver
  rebinds to itself before every vendor turn. Why: an actor outlives the
  inferencer that opened it (OpenStartup evicts and rebuilds its
  per-conversation inferencer, e.g. for a resume or a model change, and the
  runtime manager keeps the vendor session for the rebuilt one), and the
  vendor registered those handlers and hooks once, at session open. A
  rebuilt inferencer that adopts the live actor must take its tools and hooks
  over; a scope held in the actor would keep serving the old inferencer.
* **Shared core and mixins (D2).** The functional core extracted from
  `ConversationalInferencer` — `conversational/sop_feed.py`, the module
  functions of `tool_dispatch.py`, `widget_core.py` and `sop_sections.py` — is
  shared by both orchestrators. Both also inherit four mixins from
  `conversational/` (`ConversationStateMixin`, `SOPCommandsMixin`,
  `ToolDispatchMixin`, `ConversationToolsMixin`) whose methods delegate to that
  core or forward to the orchestrator's own objects (`sop_controller`,
  `mailboxes`, the transcript mirror); they keep classic CI's method names. They
  declare no fields and no `__init__` (each orchestrator declares the
  attributes they use), so they are not a shared stateful base. Two methods
  leave a flag on the instance for the step that follows:
  `ToolDispatchMixin._execute_tool_call` sets `_async_tool_dispatched` (read and
  cleared by classic CI's round loop) and `ConversationToolsMixin._apply_widget_answer`
  sets `_last_handler_bindings` (read by the answer decode that follows).
* **A backend switch keeps background tool runs.** A tool run in the
  background (an async action, a slash command's run) is a host task, not
  part of the vendor session. OpenStartup's backend switch evicts the
  inferencer and closes the old backend's vendor sessions but leaves those runs
  going; when one finishes, the UI's auto-advance turn (origin `host_event`)
  reports it on the session's current backend. Deleting the session and server
  shutdown cancel and await them (`evict_session_inferencer(close_sessions=True)`,
  `aclose_all`), as resume and restore do before they rewrite the session's
  files.

## Adding a backend

1. `session/<kind>.py`: a class with `capabilities` (evidence-tagged; the
   keys of what it uses in `relies_on`, checked by
   `capabilities.require_spec(spec)` in its constructor),
   `open(SessionOpenRequest)`, `run_turn(TurnRequest) -> VendorEvent stream`,
   `interrupt()`, `set_model()`, `close()`, `session_id`; optional
   `fork_message_map()` for exact rewinds. Emit `SessionStarted` when the vendor
   confirms the session, `MessageEnd` per main-thread assistant message
   (`message_uuid` = the vendor's transcript id, used as the turn boundary),
   `TurnEnd`, and `VendorError(submitted=…, session_missing=…)`.
2. Register it in `session/factory.py` and `NativeBackendSpec`'s inferencer map.
3. Add `resources/configs/conversational_native/backend/<kind>.yaml`.
4. Unit-test request construction and event mapping without the vendor; prove
   each capability with a spike before marking it `verified`.
