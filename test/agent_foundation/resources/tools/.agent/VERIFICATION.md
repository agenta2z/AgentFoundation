# Tool Run Verification — Common Catalog

> **Canonical location**: `AgentFoundation/test/agent_foundation/resources/tools/.agent/VERIFICATION.md`
> (migrated here from `OpenStartup/test/openteam/resources/tools/VERIFICATION.md` — the BTA/Dual
> machinery and the `task` tool now live in AgentFoundation. OS-side tools that still consume this
> catalog (`create_role`, `role_setup`) reach it via a redirect stub at the old OS path.)
>
> **Purpose**: Shared, tool-agnostic post-run verification checklist for ALL BTA-based tools (`task`, `create_role`, `role_setup`, `sop`, `understand_codebase`, future tools). Each entry describes **WHAT a problematic run looks like** — not why. Treat as a black-box observer: "If you see X, the run is unhealthy." Causes evolve; observations remain stable.
>
> **Scope**: This catalog covers observations that apply to ANY tool whose execution flows through a BreakdownThenAggregate orchestrator (breakdown → workers → aggregator). Tool-specific observations live in each tool's own `.agent/VERIFICATION.md`.
>
> **Last updated**: 2026-06-29 (added O-24 byte-stream-stall / turn-truncation; scoped O-23's ratio test to RovoDevCli; fixed stale breakdown filename + the A16 aggregation-preamble string. Earlier: moved OS→AF `.agent/`; per-tool docs now live at `<tool>/.agent/VERIFICATION.md`)

---

## How to use

After every tool run:

1. Set `WS=<latest_workspace_path>` (each tool's `.agent/VERIFICATION.md` provides a one-liner)
2. Walk **§1 Common Audit Pack** below — universal yes/no checks (A1–A20)
3. Then walk the tool-specific audit pack in `<tool>/.agent/VERIFICATION.md`
4. If any check FAILS, find the matching observation entry (common §2 or tool-specific) to understand what the bad pattern looks like
5. If the failure is genuinely new, add a new observation per the **§3 Authoring Guide** in the appropriate doc (common-if-cross-tool, tool-specific-if-narrow)

> **Important — placement rule**:
> - If the observation could apply to ANY BTA tool → add here (this common catalog)
> - If the observation references tool-specific deliverable names, prompt slots, or topology → add to that tool's `.agent/VERIFICATION.md`

---

## §1 Common Audit Pack (A1–A20)

| #   | Observation                                            | Pass criterion (✅ = healthy)                                                                                       | Ref  |
|-----|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|------|
| A1  | Run terminated cleanly                                 | Log has no `Traceback`, no `WorkflowAborted`, ends with completion line                                            | O-1  |
| A2  | Workspace at canonical location                        | `$WS` resolves to `<repo>/_runtime/tasks/<tool>/...`, NOT under `src/`                                             | O-2  |
| A3  | All inferencers logged artifacts                       | every `.jsonl.parts/` contains `InferenceInput/` and `InferenceResponse/` subdirs each non-empty; `InferenceArgs/` is present OR JSONL has `InferenceArgs` entry with empty `item={}` (no kwargs passed is valid) | O-3  |
| A4  | Breakdown received the real request                    | `breakdown/.../InferenceInput/*.txt` is non-empty AND contains the user request verbatim                           | O-4  |
| A5  | Each worker got a DISTINCT facet                       | `worker_N/.../InferenceInput/*.txt` are pairwise non-equal AND each matches breakdown JSON index N                  | O-5  |
| A6  | Inferencer inputs look templated (not raw text)        | InferenceInput contains expected template scaffolding (task preamble, sections, etc.), not just the raw user text  | O-6  |
| A7  | Aggregator used file references                        | aggregator `InferenceInput/*.txt` contains `(See file:` references AND is <300 lines (not bloated with `<Response>`) | O-7  |
| A8  | Aggregator invoked exactly ONCE                        | `aggregator/.../InferenceInput/` has exactly 1 file                                                                | O-8  |
| A9  | Canonical deliverable surfaced under final_deliverables | `$WS/outputs/final_deliverables/<deliverable>` exists and contains substantive content (size + structure thresholds defined in tool VERIFICATION) | O-9  |
| A10 | Top-level run summary distinguishable from canonical   | `$WS/outputs/run_summary.md` (or equivalent) exists; no name-collision with the canonical deliverable name at top level | O-10 |
| A11 | No deliverable double-nesting                          | `$WS/outputs/final_deliverables/final_deliverables/` does NOT exist                                                | O-11 |
| A12 | No silent argument loss                                | No "Removing YAML key" warnings for user-relevant params; arguments declared in CLI all appear in logs              | O-12 |
| A13 | Workers' inputs match breakdown sub-tasks              | For each index `i`, `worker_i` `InferenceInput/*.txt` contains the matching sub-task description from `breakdown/outputs/output.md` | O-14 |
| A14 | Aggregator references AND actively uses worker outputs | aggregator input has `(See file:` ref per worker AND `(See deliverables:` ref per worker (two-reference format, Mode A) AND response shows self-report OR unique fragments from each `worker_i/outputs/<artifact>` appear in `outputs/final_deliverables/*` (Mode B) | O-15 |
| A15 | Inferencer outputs honor `<Response>...</Response>` protocol when instructed | for each inferencer whose input instructs `<Response>` format, the STREAM file (`_runtime/inferencer_cache/.../stream_*.txt`) contains both `<Response>` and `</Response>` | O-16 |
| A16 | Aggregator received aggregation-specific preamble (not default) | aggregator `InferenceInput/*.txt` contains the canonical preamble opening `"You are aggregating/integrating the following upstream outcomes"` AND does NOT contain the default planning preamble (`## Planning Context`). Catches wrong template variant, missing template roots, prompt builder bypass, or stale wrapper composition. | O-17 |
| A17 | Every aggregator's response is substantive and integrative | For EACH aggregator in the workspace tree: (1) `outputs/final_deliverables/output.md` exists and is >1 KB, (2) the response contains integration language (aggregat/integrat/consolid/synthesiz/combin), (3) the response is not a verbatim copy of a single upstream source. Catches hollow aggregation, empty promotion chains, and aggregators that echo one worker while ignoring others. | O-20 |
| A18 | Review inputs reference the correct prior artifact | For EACH `review/` node: round_01 review input contains `<ImplementationUnderReview>` or `<ArtifactUnderReview>` referencing the propose output; round_02+ review input references the previous round's fix output path (not the original propose). Catches mislinked review chains where the reviewer evaluates a stale artifact. | O-21 |
| A19 | Review JSON correctly parsed and consensus decisions structurally consistent | Every review response contains a ` ```json ` block with valid JSON having `approve`/`approved` (boolean), `severity`/`overall_severity` (valid Severity string), `issues` (array). Round structure matches verdicts: `approved=false` or above-threshold issues → fix node + next round exist; final review approved with ≤ threshold → no further rounds. Catches malformed review JSON, field name mismatches, and framework parsing failures that silently default to wrong consensus decisions. | O-22 |
| A20 | Logged `output` matches stream cache (not noisy transcript) | For CLI-based inferencers (RovoDevCli): `output_*.txt` in `InferenceResponse/` should match `stream_*.txt` in `_runtime/inferencer_cache/` (clean LLM response, typically 5–30 KB). If `output_*.txt` ≈ `raw_output_*.txt` in size (both ~50–100 KB of terminal noise), the clean output pipeline is broken — downstream consumers receive the noisy TUI transcript instead of the parsed LLM response, breaking `<Response>` tag extraction and review JSON parsing. | O-23 |

### Generic one-liner sanity script

Each tool's `.agent/VERIFICATION.md` provides a wrapper that sets `$WS`, `$TOOL`, and (optionally) a canonical deliverable name; below is the common audit body:

```bash
# Assumes $WS, $TOOL, and (optional) $CANONICAL_DELIVERABLE are exported by tool wrapper.
echo "=== A1 abort/error scan ==="
find "$WS" -name "*.log" -exec grep -l "Traceback\|WorkflowAborted\|NameError" {} \; 2>/dev/null

echo "=== A2 canonical workspace location ==="
case "$WS" in
  */src/*) echo "FAIL: workspace under src/ ($WS)";;
  */_runtime/tasks/$TOOL/*) echo "OK: under _runtime/tasks/$TOOL/";;
  *) echo "WARN: non-canonical workspace ($WS)";;
esac

echo "=== A3 .jsonl.parts integrity ==="
for d in $(find "$WS" -type d -name "*.jsonl.parts"); do
  for sub in InferenceInput InferenceResponse; do
    n=$(ls "$d/$sub" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || echo "MISSING: $d/$sub"
  done
done

echo "=== A7 aggregator input shape (per aggregator) ==="
for agg in $(find "$WS" -path "*/aggregator/logs/session/*.jsonl.parts/InferenceInput" -type d); do
  inp=$(ls "$agg"/*.txt 2>/dev/null | head -1)
  [ -f "$inp" ] && {
    echo "  $agg:"
    wc -l "$inp"
    echo "    file refs: $(grep -c '(See file:' "$inp")"
    echo "    inlined responses: $(grep -c '<Response>' "$inp")"
  }
done

echo "=== A8 aggregator invocation count (per aggregator) ==="
for agg in $(find "$WS" -path "*/aggregator/logs/session/*.jsonl.parts/InferenceInput" -type d); do
  echo "  $agg: $(ls $agg | wc -l) invocations"
done

echo "=== A11 double-nesting ==="
for nest in $(find "$WS" -path "*/final_deliverables/final_deliverables" -type d 2>/dev/null); do
  echo "FAIL double-nested: $nest"
done || echo "OK: no double-nesting"

echo "=== A13 worker input presence (sizes) ==="
for w in $(find "$WS" -type d -name "worker_*" -not -path "*/_runtime/*" -not -path "*/logs/*"); do
  wid=$(basename "$w")
  win=$(ls "$w"/logs/session/*.jsonl.parts/InferenceInput/*.txt 2>/dev/null | head -1)
  [ -f "$win" ] && echo "  $wid: $(wc -c <"$win") bytes input" || echo "  $wid: MISSING input"
done

echo "=== A14 aggregator references AND uses worker outputs ==="
# Mode A: file ref count (includes two-reference format check)
for agg in $(find "$WS" -path "*/aggregator/logs/session/*.jsonl.parts/InferenceInput" -type d); do
  inp=$(ls "$agg"/*.txt 2>/dev/null | head -1)
  agg_parent=$(dirname "$(dirname "$(dirname "$(dirname "$agg")")")")
  parent=$(dirname "$agg_parent")
  siblings=$(find "$parent" -maxdepth 1 -type d -name "worker_*" -o -name "flow_*" | wc -l | tr -d ' ')
  [ -f "$inp" ] && {
    file_refs=$(grep -c "(See file:" "$inp")
    dir_refs=$(grep -c "(See deliverables:" "$inp")
    agg_label=$(echo "$agg" | sed "s|$WS/children/||;s|/logs/.*||")
    echo "  $agg_label: file_refs=$file_refs dir_refs=$dir_refs siblings=$siblings"
    [ "$dir_refs" -ge 1 ] && echo "    two-reference format: YES" || echo "    two-reference format: NO (pre-v2 or broken)"
  }
done

# Mode B content traceability — manual per-tool:
#   For each worker, pick a unique fragment from its output and grep the deliverable.

echo "=== A15 <Response> protocol compliance ==="
for stream in $(find "$WS" -path "*_runtime/inferencer_cache/*/stream_*.txt" 2>/dev/null); do
  open=$(grep -c '<Response>' "$stream" 2>/dev/null)
  close=$(grep -c '</Response>' "$stream" 2>/dev/null)
  [ "$open" -ge 1 ] && [ "$close" -ge 1 ] && tag="OK" || tag="MISSING"
  echo "  $tag open=$open close=$close $(basename $stream)"
done

echo "=== A16 aggregator preamble variant ==="
for agg in $(find "$WS" -path "*/aggregator/logs/session/*.jsonl.parts/InferenceInput" -type d); do
  inp=$(ls "$agg"/*.txt 2>/dev/null | head -1)
  [ -f "$inp" ] && {
    agg_preamble=$(grep -c "You are aggregating/integrating the following upstream outcomes" "$inp")
    default_pre=$(grep -c "## Planning Context" "$inp")
    echo "  aggregation preamble: $agg_preamble (want >=1)"
    echo "  default preamble (BAD): $default_pre (want 0)"
    [ "$agg_preamble" -ge 1 ] && [ "$default_pre" -eq 0 ] && echo "  PASS" || echo "  FAIL"
  }
done

echo "=== A17 aggregator response substantiveness + integration ==="
for agg_dir in $(find "$WS" -type d -name "aggregator" -not -path "*/logs/*"); do
  output="$agg_dir/outputs/final_deliverables/output.md"
  agg_label=$(echo "$agg_dir" | sed "s|$WS/children/||")
  if [ ! -f "$output" ]; then
    echo "  FAIL $agg_label: no output.md in final_deliverables/"
    continue
  fi
  size=$(wc -c < "$output" | tr -d ' ')
  lines=$(wc -l < "$output" | tr -d ' ')
  integration=$(grep -ci "aggregat\|integrat\|consolid\|synthesiz\|combin\|upstream\|sibling" "$output")
  if [ "$size" -lt 1024 ]; then
    echo "  FAIL $agg_label: output too small (${size}B < 1KB)"
  elif [ "$integration" -eq 0 ]; then
    echo "  WARN $agg_label: ${size}B, ${lines}L but 0 integration keywords"
  else
    echo "  PASS $agg_label: ${size}B, ${lines}L, integration=$integration"
  fi
done

echo "=== A18 review inputs reference correct prior artifact ==="
for review_dir in $(find "$WS/children" -type d -name "review" -not -path "*/logs/*" | sort); do
  label=$(echo "$review_dir" | sed "s|$WS/children/||")
  round_dir=$(dirname "$(dirname "$review_dir")")
  round_name=$(basename "$round_dir")
  round_num=$(echo "$round_name" | sed 's/round_0*//')
  inp=$(find "$review_dir" -path "*/logs/session/*/InferenceInput/*.txt" -type f 2>/dev/null | head -1)
  [ ! -f "$inp" ] && { echo "  FAIL $label: no input file"; continue; }
  has_artifact=$(grep -c "UnderReview>\|ArtifactUnderReview>\|ImplementationUnderReview>" "$inp")
  if [ "$round_num" -gt 1 ]; then
    prev=$((round_num - 1))
    prev_fix_ref=$(grep -c "round_0*${prev}/children/fix/outputs\|round_0*${prev}.*fix.*output" "$inp")
    [ "$has_artifact" -ge 1 ] && [ "$prev_fix_ref" -ge 1 ] \
      && echo "  PASS $label: references round_${prev} fix output" \
      || echo "  WARN $label: round_${round_num} but no fix-output ref (artifact=$has_artifact fix_ref=$prev_fix_ref)"
  else
    [ "$has_artifact" -ge 1 ] \
      && echo "  PASS $label: references propose artifact" \
      || echo "  WARN $label: no artifact-under-review section found"
  fi
done

echo "=== A19 review JSON parsing + consensus structural consistency ==="
for review_dir in $(find "$WS/children" -type d -name "review" -not -path "*/logs/*" | sort); do
  label=$(echo "$review_dir" | sed "s|$WS/children/||")
  round_dir=$(dirname "$(dirname "$review_dir")")
  round_name=$(basename "$round_dir")
  round_num=$(echo "$round_name" | sed 's/round_0*//')
  resp=$(find "$review_dir" -path "*/logs/session/*/InferenceResponse/*output_*.txt" -type f 2>/dev/null | head -1)
  [ ! -f "$resp" ] && { echo "  FAIL $label: no response file"; continue; }
  # Extract and validate JSON
  result=$(sed -n '/```json/,/```/p' "$resp" | sed '1d;$d' | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    approved = data.get('approved', data.get('approve', '__MISSING__'))
    severity = data.get('severity', data.get('overall_severity', '__MISSING__'))
    issues = data.get('issues', '__MISSING__')
    errs = []
    if approved == '__MISSING__': errs.append('no approved/approve field')
    elif not isinstance(approved, bool): errs.append(f'approved is {type(approved).__name__} not bool')
    if severity == '__MISSING__': errs.append('no severity/overall_severity field')
    if issues == '__MISSING__': errs.append('no issues field')
    elif not isinstance(issues, list): errs.append(f'issues is {type(issues).__name__} not list')
    if errs:
        print(f'JSON_WARN|{\"|\".join(errs)}|approved={approved}|severity={severity}')
    else:
        print(f'JSON_OK|approved={approved}|severity={severity}|issues={len(issues)}')
except json.JSONDecodeError as e:
    print(f'JSON_FAIL|{e}')
except Exception as e:
    print(f'JSON_FAIL|{e}')
" 2>&1)
  json_status=$(echo "$result" | cut -d'|' -f1)
  approved_val=$(echo "$result" | grep -o 'approved=[A-Za-z]*' | cut -d= -f2)
  # Structural consistency: check if fix + next round exist when expected
  fix_exists="NO"
  [ -d "$(dirname "$review_dir")/fix" ] && fix_exists="YES"
  next_round_num=$(printf "%02d" $((round_num + 1)))
  next_round="$(dirname "$round_dir")/round_${next_round_num}"
  next_round_exists="NO"
  [ -d "$next_round" ] && next_round_exists="YES"
  if [ "$json_status" = "JSON_FAIL" ]; then
    echo "  FAIL $label: $result"
  elif [ "$json_status" = "JSON_WARN" ]; then
    echo "  WARN $label: $result"
  else
    # Consistency check
    if [ "$approved_val" = "False" ]; then
      [ "$fix_exists" = "YES" ] \
        && echo "  PASS $label: $result → rejected, fix created" \
        || echo "  INCONSISTENT $label: $result → rejected but NO fix"
    elif [ "$approved_val" = "True" ]; then
      [ "$next_round_exists" = "NO" ] \
        && echo "  PASS $label: $result → approved, no more rounds" \
        || echo "  CHECK $label: $result → approved but next round exists (threshold override?)"
    fi
  fi
done

echo "=== A20 output vs stream cache consistency (CLI inferencers) ==="
for parts_dir in $(find "$WS" -type d -name "*.jsonl.parts" 2>/dev/null); do
  output=$(find "$parts_dir/InferenceResponse" -name "*output_*.txt" -not -name "*raw_output*" -type f 2>/dev/null | head -1)
  raw=$(find "$parts_dir/InferenceResponse" -name "*raw_output_*.txt" -type f 2>/dev/null | head -1)
  [ ! -f "$output" ] || [ ! -f "$raw" ] && continue
  out_size=$(wc -c < "$output" | tr -d ' ')
  raw_size=$(wc -c < "$raw" | tr -d ' ')
  label=$(echo "$parts_dir" | sed "s|$WS/children/||;s|/logs/.*||")
  # If output ≈ raw_output (within 5%), the clean pipeline is broken
  if [ "$raw_size" -gt 0 ]; then
    ratio=$((out_size * 100 / raw_size))
    if [ "$ratio" -gt 95 ]; then
      echo "  WARN $label: output=${out_size}B ≈ raw=${raw_size}B (${ratio}%) — output may be noisy transcript"
    elif [ "$out_size" -gt 0 ]; then
      echo "  PASS $label: output=${out_size}B vs raw=${raw_size}B (${ratio}%) — output is clean"
    fi
  fi
done
```

> **Tip — auto-discover $WS**: each tool's `.agent/VERIFICATION.md` provides a `WS=$(ls -td …/$TOOL/* | head -1)` snippet.

---

## §2 Common Observation Catalog (O-1 – O-24)

### O-1 — Run aborts unexpectedly
- **Look for**: log/stderr contains `Traceback`, `NameError`, `WorkflowAborted`, OR the launcher process exits non-zero before reaching the completion line
- **Distinguishes from healthy**: healthy runs end with an explicit "completed" line and zero exception traces in any log file

### O-2 — Workspace created in the wrong location
- **Look for**: workspace path contains `/src/` (i.e., under source tree) OR is created in cwd OR under `/tmp/` instead of `<repo>/_runtime/tasks/<tool>/`
- **Distinguishes from healthy**: healthy runs always place workspace under `<repo>/_runtime/tasks/<tool>/<tool>_<YYYYMMDD>_<HHMMSS>_<8hash>/`

### O-3 — Hollow `.jsonl.parts/` directories
- **Look for**: an inferencer's `logs/session/*.jsonl.parts/` directory exists but is empty, OR is missing `InferenceInput/` or `InferenceResponse/` subdirs, OR those subdirs exist but have 0 files. (Note: `InferenceArgs/` may be absent when no kwargs were passed — the framework only externalizes non-primitive values; the JSONL still has a record with `item={}`)
- **Distinguishes from healthy**: every inferencer that ran should have ≥1 file in EACH of the 2 mandatory subdirs (`InferenceInput`, `InferenceResponse`)

### O-4 — Inferencer received empty / wrong input
- **Look for**: `InferenceInput/*.txt` is empty, OR contains a placeholder like `<no_query>`/`<empty>`, OR contains a value that doesn't match what the user/upstream should have passed (e.g., breakdown received some unrelated string)
- **Distinguishes from healthy**: the input text should contain the actual upstream content (user request, parent's output, sub-facet description)

### O-5 — Workers received duplicate / wrong facets
- **Look for**: two or more `worker_N/.../InferenceInput/*.txt` files are byte-identical OR don't correspond to distinct entries in `breakdown/outputs/output.md`
- **Distinguishes from healthy**: in an N-way breakdown, each worker should receive a UNIQUE sub-query matching its position in the breakdown plan

### O-6 — Inferencer input is raw, not templated
- **Look for**: `InferenceInput/*.txt` contains ONLY the bare user request (or upstream content) with NO surrounding template scaffolding — no task preamble, no instruction header, no rendered Jinja2 blocks (like deep_mode, elegant_mode, output_path directives)
- **Distinguishes from healthy**: a properly-templated input has 80–200 lines of scaffolding around a small embedded query; a raw input is just the query itself
- **Bonus signal**: log warnings like `Removing YAML key 'template_*' — not a valid __init__ param` indicate the inferencer can't accept template attribs and will silently render nothing

### O-7 — Aggregator input bloated with inlined worker text
- **Look for**: aggregator's `InferenceInput/*.txt` exceeds ~300 lines AND contains literal `<Response>...</Response>` blocks holding full worker outputs (multi-KB chunks of markdown) inlined directly into the prompt; OR has ZERO `(See file: ...)` references when workers produced output files
- **Distinguishes from healthy**: a healthy aggregator input is compact (<300 lines), references upstream artifacts via `(See file: <path>)` lines, and lets the agent read them on demand

### O-8 — Aggregator invoked multiple times within ONE run
- **Look for**: `aggregator/.../InferenceInput/` contains 2 or more `.txt` files in a single run (each timestamped seconds-to-minutes apart with near-identical content)
- **Distinguishes from healthy**: a single run should produce exactly 1 aggregator input. Repeat invocations indicate a retry loop, cache replay, or framework re-entry bug

### O-9 — Canonical deliverable missing or insubstantial
- **Look for**: `$WS/outputs/final_deliverables/<deliverable>` does not exist, OR exists but is suspiciously small (size threshold depends on tool — see tool VERIFICATION), OR has insufficient structure (heading count below tool threshold)
- **Distinguishes from healthy**: a healthy canonical deliverable is the full aggregator output with substantive multi-section content (specific size/heading thresholds in each tool's VERIFICATION.md)

### O-10 — Top-level output ambiguity
- **Look for**: two files with the same name (e.g., `<deliverable>.md`) exist at BOTH `$WS/outputs/` AND `$WS/children/aggregator/outputs/` AND they have different content/size — leaving the user unsure which is "the real one"; OR the top-level `outputs/<deliverable>.md` contains a short summary blurb instead of the real document
- **Distinguishes from healthy**: top-level should have a distinctly-named summary (e.g., `run_summary.md`); the canonical deliverable should live exclusively under `outputs/final_deliverables/`

### O-11 — Deliverable double-nesting
- **Look for**: `$WS/outputs/final_deliverables/final_deliverables/<any_file>` exists (one level too deep)
- **Distinguishes from healthy**: `final_deliverables/` should appear EXACTLY once on any path

### O-12 — Silent argument loss
- **Look for**: a parameter the user explicitly passed at CLI (or via `/tool` slash invocation) does NOT appear in the downstream `InferenceInput/` content; log may show "Removing YAML key" warnings OR no warning at all (silent); OR the breakdown clearly didn't use a provided value
- **Distinguishes from healthy**: every user-provided argument should be observably consumed (either by appearing in rendered prompts, or by altering observable behavior like worker count)

### O-13 — Subagent gives up on out-of-workspace path
- **Look for**: subagent output contains phrases like `"path you provided is outside the current workspace"` OR `"I cannot access that path"` AND the parent agent did NOT fall back to `bash cat <path>` AND the run completed without ever reading the referenced material
- **Distinguishes from healthy/known-workaround**: it's acceptable for a subagent to refuse, AS LONG AS the parent retries via `bash cat` and successfully reads the content. The failure mode is when the parent gives up entirely

### O-14 — Workers don't receive content matching the breakdown
- **Look for**: `breakdown/outputs/output.md` defines N sub-tasks with specific descriptions/scopes, but the corresponding `worker_N/.../InferenceInput/*.txt` does NOT contain the matching sub-task description (workers received a different facet, an outdated one, or a generic placeholder)
- **Distinguishes from healthy**: for every entry index `i` in the breakdown plan, `worker_i` should have an `InferenceInput/*.txt` that demonstrably reflects that specific entry's description/scope/todos (verbatim or near-verbatim substring match)
- **Verify**: cross-check by extracting the breakdown JSON, then `grep` each entry's distinguishing phrase against the corresponding worker's InferenceInput

### O-15 — Aggregator either doesn't reference worker outputs OR doesn't actively use them despite the reference
- **Look for** (either failure mode):
  - **Mode A — Missing reference**: aggregator `InferenceInput/*.txt` does NOT contain `(See file: <worker_path>)` lines pointing to each worker's output file (and the worker outputs ARE missing from any other form in the prompt too)
  - **Mode B — Reference ignored**: aggregator `InferenceInput/*.txt` DOES contain the file references, BUT the aggregator's `InferenceResponse/*.txt` (or `_runtime/inferencer_cache/.../stream_*.txt`) shows the agent never read those files (no `view_file`/`read_file`/`bash cat` tool calls against worker output paths), AND the synthesized output is missing content/facts that exist ONLY in worker outputs
- **Distinguishes from healthy**: a healthy run has BOTH (a) per-worker file references in the aggregator input AND (b) evidence of active consumption — either explicit file-read tool calls OR verbatim/unique fragments from each worker file appearing in the synthesis
- **Verify**:
  - Mode A: `grep -c "(See file:" $WS/children/aggregator/logs/session/*.jsonl.parts/InferenceInput/*.txt` should equal N (worker count)
  - Mode B (content traceability — most reliable): for each `worker_i`, pick a unique fragment from `worker_i/outputs/<artifact>` (e.g., a distinctive heading or named concept) and `grep` it in the aggregator's final deliverable; at least 1 fragment per worker should match
  - Mode B (self-report — secondary): `grep -E "I read|consulted|Worker [0-9]+|worker_[0-9]+" $WS/children/aggregator/logs/session/*.jsonl.parts/InferenceResponse/*output*.txt` should show the aggregator naming each worker it consumed
  - **Note**: `_runtime/inferencer_cache/.../stream_*.txt` for RovoDev/RovoChat CLI inferencers contains only the FINAL `<Response>` text, NOT intermediate tool calls; do not rely on it for tool-call detection

### O-16 — Inferencer output missing `<Response>...</Response>` protocol delimiters
- **Look for**: an inferencer's STREAM file (`_runtime/inferencer_cache/<class>/<id>_<ts>/stream_*.txt`) does NOT contain BOTH `<Response>` and `</Response>` tags wrapping the actual deliverable, despite the prompt template's task-response-format instruction (often phrased as "Use literal `<Response>` and `</Response>` in your actual reply — `<ResponseSchema>` is just the example container") being present in the corresponding `InferenceInput/*.txt`
- **Distinguishes from healthy**: a compliant stream contains exactly one `<Response>` and one `</Response>` tag-pair surrounding the deliverable; the framework's `extract_delimited()` then cleanly extracts the inner content. Missing tags force fallback to "use whole output" — silently lenient but fragile.
- **Important**: do NOT check `InferenceResponse/*output*.txt` for the tags — for CLI-based agents (RovoDev/RovoChat) that file captures the full terminal dump (banner, agent setup, ✓ tool calls, formatted output blocks) where `<tag>`-like text may be stripped by terminal rendering. The stream file is the authoritative source of what the agent actually emitted.
- **Verify** (per response file):
  - `open=$(grep -c '<Response>' $stream); close=$(grep -c '</Response>' $stream)` — both should equal 1
  - Confirm input asked for the format: `grep -q '<Response>' $WS/children/<inferencer>/logs/session/*.jsonl.parts/InferenceInput/*.txt`
- **Caveat**: if the corresponding `InferenceInput/*.txt` does NOT instruct the `<Response>` format, then absence in output is expected — not a bug. Always check the input first.

### O-17 — Aggregator received default preamble instead of aggregation-specific preamble
- **Look for**: aggregator's `InferenceInput/*.txt` contains the default planning preamble (`## Planning Context`, generic plan creation instructions) instead of the aggregation-specific preamble (`You are aggregating/integrating the following upstream outcomes...` with `{{ upstream_artifacts }}` slot and `{{ aggregation_guidance }}` conditional). The aggregator functions without it (the LLM still synthesizes) but produces hollow output — worker research is not referenced, no `(See file:)` paths, and synthesis quality degrades.
- **Distinguishes from healthy**: a healthy aggregator input starts its `<UserRequest>` block with "You are aggregating/integrating the following upstream outcomes..." followed by `### Upstream Outcome N` entries with `(See file: <worker_path>)` references. An unhealthy one starts with `## Planning Context` (the generic default).
- **Root causes** (ordered by frequency):
  1. **Missing template root**: the TemplateManager has only the consumer root (OpenStartup) but not the framework root (AgentFoundation) where `aggregation/default.jinja2` lives. Typically caused by `_template_manager.templates` override clobbering the YAML's two-root list with a single path.
  2. **Wrapper variable composition**: the wrapper variable `context.user_request_with_task_preamble` bakes in the default `task_preamble` during auto-discovery composition, and the caller's correct aggregation preamble arrives too late. Requires `enable_templated_feed=True` AND the wrapper recomposition step in `_resolve_templated_feed()`.
  3. **SLOT_DEFAULTS override**: `template_version` set explicitly on the aggregator gets overridden back to `"aggregation"` by SLOT_DEFAULTS during BTA execution. Use per-variable overrides in `template_variables` instead of `template_version` for task-specific variants.
- **Verify**:
  ```bash
  for agg in $(find "$WS" -path "*/aggregator/logs/session/*.jsonl.parts/InferenceInput" -type d); do
    inp=$(ls "$agg"/*.txt 2>/dev/null | head -1)
    [ -f "$inp" ] && {
      agg_preamble=$(grep -c "You are aggregating/integrating the following upstream outcomes" "$inp")
      default=$(grep -c "## Planning Context" "$inp")
      echo "  aggregation preamble: $agg_preamble (want >=1)"
      echo "  default preamble: $default (want 0)"
      [ "$agg_preamble" -ge 1 ] && [ "$default" -eq 0 ] && echo "  PASS" || echo "  FAIL"
    }
  done
  ```
- **Cross-ref**: A14a regression (2026-05-21); root cause documented in `AgentFoundation/_docs/_plan/template_and_variable_versioning_formalization/`

### O-18 — Inner aggregator files written to root workspace instead of own workspace
- **Look for**: in a nested BTA (e.g., role_setup), skill/tool files appear at `$WS/outputs/skills/` or `$WS/outputs/tools/` (root level) instead of `$WS/children/worker_N/children/aggregator/outputs/skills/`. The inner aggregator's own `outputs/` directory is empty or missing the expected deliverables.
- **Distinguishes from healthy**: a healthy nested BTA has files in `children/worker_N/children/aggregator/outputs/skills/` (the inner aggregator's own workspace). Root-level `outputs/skills/` should only contain files written by the OUTER aggregator.
- **Root cause**: `_target_path` cascade from `_run_topology()` propagates the root workspace path to ALL descendant inferencers. The inner aggregator's `effective_cwd` uses `target_path` (root workspace) instead of `_workspace.root` (correct inner workspace). Fix: use `{{ workspace_outputs }}` absolute paths in templates instead of relative `outputs/` paths.
- **Source**: Run `role_setup_20260525_085001_1e13c7db` — 70 files at root `outputs/`, 0 in inner aggregator workspace. Fixed in run `role_setup_20260525_213032_57d3f86c` via `{{ workspace_outputs }}` template variable.

### O-19 — Files in `outputs/` but `final_deliverables/` empty (promotion chain broken)
- **Look for**: an aggregator's `outputs/` directory contains skill/tool files (e.g., `outputs/skills/<name>/SKILL.md`, `outputs/tools/<name>/tool.json`) but `outputs/final_deliverables/` is empty. The parent BTA's `final_deliverables/` is also empty, and the outer aggregator reports no deliverables to integrate.
- **Distinguishes from healthy**: a healthy run has files MOVED from `outputs/` to `outputs/final_deliverables/` by `_finalize_output()`, which then cascade up via `_symlink_child_output()` to the parent BTA's `final_deliverables/`.
- **Root cause**: `output_is_deliverable` not set to `true` on the aggregator inferencer in the YAML. Without this flag, `_finalize_output()` skips the move from `outputs/` to `outputs/final_deliverables/`, breaking the entire promotion chain.
- **Source**: Run `role_setup_20260525_213032_57d3f86c` — inner aggregator had 28 files in `outputs/`, 0 in `final_deliverables/`. Fix: add `output_is_deliverable: true` to aggregator YAML config.

### O-20 — Aggregator response hollow or non-integrative
- **Look for**: an aggregator's `outputs/final_deliverables/output.md` is missing, empty, or suspiciously small (<1 KB). OR the output exists but contains zero integration language — no mention of "aggregat", "integrat", "consolidat", "synthesiz", "combin", "upstream", or "sibling". The output reads like a single worker's plan copy-pasted, not a synthesis of multiple upstream sources.
- **Distinguishes from healthy**: a healthy aggregator output is substantive (typically >5 KB for multi-worker BTAs) and explicitly signals integration — e.g., "This document integrates three upstream plans", "Consolidated from Flow 0 and Flow 1", "Aggregator artifact". At minimum it references more than one upstream source.
- **Severity tiers**:
  - **FAIL**: output.md missing or <1 KB — the aggregator produced nothing usable
  - **WARN**: output.md exists but has 0 integration keywords — possible verbatim echo of a single worker
  - **PASS**: output.md >1 KB with ≥1 integration keyword
- **Applies to all aggregator levels**: inner MFDual aggregators (aggregating flows), BTA aggregators (aggregating workers), and outer aggregators in nested topologies. In a nested BTA, check EVERY aggregator in the workspace tree, not just the top-level one.
- **Verify**:
  ```bash
  for agg_dir in $(find "$WS" -type d -name "aggregator" -not -path "*/logs/*"); do
    output="$agg_dir/outputs/final_deliverables/output.md"
    label=$(echo "$agg_dir" | sed "s|$WS/children/||")
    if [ ! -f "$output" ]; then
      echo "FAIL $label: no output.md in final_deliverables/"
      continue
    fi
    size=$(wc -c < "$output" | tr -d ' ')
    integration=$(grep -ci "aggregat\|integrat\|consolid\|synthesiz\|combin\|upstream\|sibling" "$output")
    [ "$size" -lt 1024 ] && echo "FAIL $label: ${size}B < 1KB" && continue
    [ "$integration" -eq 0 ] && echo "WARN $label: ${size}B but 0 integration keywords" && continue
    echo "PASS $label: ${size}B, integration=$integration"
  done
  ```
- **Source**: task full mode v3 run `task_20260526_214348_8c81288d` — all 4 aggregators verified: worker_0 (35KB, 27 keywords), worker_1 (34KB, 15), worker_2 (46KB, 46), outer BTA (56KB, 51). Codified as a durable check after manual verification caught promotion-chain and template-cascade issues in earlier runs.

### O-21 — Reviewer evaluates stale or wrong artifact
- **Look for**: a review node's `InferenceInput/*.txt` does NOT contain `<ImplementationUnderReview>` or `<ArtifactUnderReview>` sections; OR for round_02+ reviews, the referenced artifact path points to the original propose output instead of the previous round's fix output; OR the `<ImplementationUnderReview>` content is empty / placeholder text.
- **Distinguishes from healthy**: a healthy round_01 review references the propose output (e.g., `.../propose/outputs/output.md` or `.../propose/outputs/final_deliverables/output.md`). A healthy round_02+ review references the prior fix output (e.g., `.../round_01/children/fix/outputs/output.md`). The artifact-under-review section contains a substantive summary of the prior output, not the raw user request.
- **Why it matters**: if the reviewer evaluates a stale artifact, the fix cycle is wasted — the fixer addresses issues that may already be resolved, or the reviewer approves something that was already superseded. The Dual loop converges but on the wrong version.
- **Verify**:
  ```bash
  for review_dir in $(find "$WS/children" -type d -name "review" -not -path "*/logs/*" | sort); do
    label=$(echo "$review_dir" | sed "s|$WS/children/||")
    round_num=$(basename "$(dirname "$(dirname "$review_dir")")" | sed 's/round_0*//')
    inp=$(find "$review_dir" -path "*/logs/session/*/InferenceInput/*.txt" -type f 2>/dev/null | head -1)
    [ ! -f "$inp" ] && { echo "FAIL $label: no input"; continue; }
    has_artifact=$(grep -c "UnderReview>" "$inp")
    if [ "$round_num" -gt 1 ]; then
      prev=$((round_num - 1))
      fix_ref=$(grep -c "round_0*${prev}/children/fix\|round_0*${prev}.*fix.*output" "$inp")
      [ "$has_artifact" -ge 1 ] && [ "$fix_ref" -ge 1 ] \
        && echo "PASS $label: round_${round_num} references round_${prev} fix" \
        || echo "WARN $label: round_${round_num} missing fix ref (artifact=$has_artifact fix_ref=$fix_ref)"
    else
      [ "$has_artifact" -ge 1 ] && echo "PASS $label" || echo "WARN $label: no artifact section"
    fi
  done
  ```
- **Source**: task full mode v3 run `task_20260526_214348_8c81288d` — all 18 reviews verified: every round_01 review referenced propose output, every round_02+ review referenced the previous fix output. Codified to prevent regression in review chain linking.

### O-22 — Review JSON malformed or framework consensus decision inconsistent with verdict
- **Look for** (either failure mode):
  - **Mode A — Malformed JSON**: reviewer response has no ` ```json ` block, OR the block contains invalid JSON, OR the JSON is missing `approve`/`approved` (must be boolean) or `severity`/`overall_severity` (must be a valid Severity string: NONE, COSMETIC, MINOR, MAJOR, CRITICAL) or `issues` (must be array). The framework's `_default_parse_review` falls back to `{approved: false, severity: "MAJOR", issues: [{parsing_error}]}` — silently treating parse failure as rejection.
  - **Mode B — Structural inconsistency**: review verdict says `approved=false` (or has above-threshold per-issue severity) but NO fix node or next round exists (premature termination); OR verdict says `approved=true` with all issues ≤ threshold but a fix + next round WAS created anyway (unnecessary iteration).
  - **Mode C — Threshold override**: review says `approved=true` but individual issues have severity above the consensus threshold (default: COSMETIC). The framework correctly OVERRIDES the approval via Gate 1 (per-issue severity check) and creates a fix round. This is HEALTHY behavior — the `CHECK` label in the script distinguishes it from Mode B.
- **Distinguishes from healthy**: a healthy review has: (1) valid ```json block with boolean `approve`/`approved`, string severity, array `issues`; (2) round structure exactly matching the parsed verdict — rejected → fix + next round, approved (with all issues ≤ threshold) → no more rounds.
- **Verify**:
  ```bash
  for review_dir in $(find "$WS/children" -type d -name "review" -not -path "*/logs/*" | sort); do
    label=$(echo "$review_dir" | sed "s|$WS/children/||")
    round_dir=$(dirname "$(dirname "$review_dir")")
    round_num=$(basename "$round_dir" | sed 's/round_0*//')
    resp=$(find "$review_dir" -path "*/logs/session/*/InferenceResponse/*output_*.txt" -type f 2>/dev/null | head -1)
    [ ! -f "$resp" ] && { echo "FAIL $label: no response"; continue; }
    result=$(sed -n '/```json/,/```/p' "$resp" | sed '1d;$d' | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    a = d.get('approved', d.get('approve', '__MISSING__'))
    s = d.get('severity', d.get('overall_severity', '__MISSING__'))
    i = d.get('issues', '__MISSING__')
    e = []
    if a == '__MISSING__': e.append('no approved field')
    elif not isinstance(a, bool): e.append(f'approved={type(a).__name__}')
    if s == '__MISSING__': e.append('no severity field')
    if not isinstance(i, list): e.append('issues not list')
    print(f'WARN|{\"|\".join(e)}' if e else f'OK|approved={a}|severity={s}|issues={len(i)}')
except: print('FAIL|json parse error')
" 2>&1)
    approved=$(echo "$result" | grep -o 'approved=[A-Za-z]*' | cut -d= -f2)
    fix_exists=$( [ -d "$(dirname "$review_dir")/fix" ] && echo Y || echo N )
    next=$( [ -d "$(dirname "$round_dir")/round_$(printf '%02d' $((round_num+1)))" ] && echo Y || echo N )
    echo "  $label: $result | fix=$fix_exists next_round=$next"
  done
  ```
- **Field name aliasing**: the review template instructs `approve` and `overall_severity`; the parser accepts both via fallback `.get()` chains (`parsed.get("approved", parsed.get("approve", False))`). Both naming conventions are valid.
- **Source**: task full mode v3 run `task_20260526_214348_8c81288d` — all 18 reviews parsed successfully (zero fallbacks to `parsing_error` path). 3 threshold-override cases correctly identified (reviewer said `approve: true` with MINOR issues, framework overrode via Gate 1). Codified after discovering the multi-layer extraction pipeline (RovoDevCli `extract_json_from_output` → DualInferencer `_default_parse_review`) and the field name aliasing behavior.

### O-23 — Logged `output` is noisy transcript instead of clean LLM response
- **Look for**: for CLI-based inferencers (RovoDevCli), `output_*.txt` in `InferenceResponse/` is nearly the same size as `raw_output_*.txt` (both ~50–100 KB). The `output_*.txt` starts with `Working in /path...` and contains MCP server errors, tool call blocks, and terminal chrome — instead of the clean LLM response (typically 5–30 KB starting with the actual content). Meanwhile `stream_*.txt` in `_runtime/inferencer_cache/` IS correct (contains the clean `<Response>`-tagged output).
- **Distinguishes from healthy**: a healthy `output_*.txt` contains ONLY the clean LLM response (matching `stream_*.txt` in content and size). `raw_output_*.txt` contains the full noisy transcript — that's expected. **SCOPE (important — the ratio test is RovoDevCli-specific):** RovoDevCli interleaves a noisy terminal transcript, so for it `output_size / raw_size` well below 50% is healthy and >95% means the output IS the transcript. For **clean-stdout CLIs (ClaudeCodeCli, CodexCli)** `output ≈ raw` (often a 100% ratio) is **NORMAL and healthy** — both are the clean response; a ~100% ratio there is NOT a finding (the A20 script's `WARN …output≈raw` line is expected for these and should be ignored). The real failure to catch for those is `out_size == 0` with `raw_size > 0` (empty clean output). Verified on v9 (`multimodal_plan_3flow_20260629_103951_192264bc`): every ClaudeCodeCli/CodexCli node has output==raw (e.g. aggregator 5424B==5424B) and is healthy.
- **Impact**: downstream consumers (e.g., `DualInferencer._default_parse_review`) receive the noisy transcript instead of the clean response. `<Response>` tag extraction fails (tags are stripped by terminal rendering), forcing fallback to ` ```json ``` ` block search inside 50–100 KB of noise. Review JSON parsing still works (the JSON is embedded in the noise) but is fragile and wasteful.
- **Verify**:
  ```bash
  for parts_dir in $(find "$WS" -type d -name "*.jsonl.parts" 2>/dev/null); do
    output=$(find "$parts_dir/InferenceResponse" -name "*output_*.txt" -not -name "*raw_output*" -type f 2>/dev/null | head -1)
    raw=$(find "$parts_dir/InferenceResponse" -name "*raw_output_*.txt" -type f 2>/dev/null | head -1)
    [ ! -f "$output" ] || [ ! -f "$raw" ] && continue
    out_size=$(wc -c < "$output" | tr -d ' ')
    raw_size=$(wc -c < "$raw" | tr -d ' ')
    label=$(echo "$parts_dir" | sed "s|$WS/children/||;s|/logs/.*||")
    ratio=$((out_size * 100 / raw_size))
    [ "$ratio" -gt 95 ] && echo "WARN $label: output≈raw (${ratio}%)" || echo "PASS $label: output=${out_size}B vs raw=${raw_size}B"
  done
  ```
- **Source**: task full mode v3 run `task_20260526_214348_8c81288d` — outer Dual round_01 review had `output_*.txt` = 68,992B ≈ `raw_output_*.txt` = 68,994B (99.9% ratio), while `stream_*.txt` was 8,265B (the correct clean response). Fixed by pushing `TerminalInferencerResponse` wrapping from `ainfer()` down to `_ainfer()` so the base-class logging captures the correct clean output. (RovoDevCli only — see SCOPE above.)

### O-24 — Agent turn truncated / required deliverable never emitted (byte-stream stall)
- **Look for**: an inferencer whose captured `output`/deliverable is pure **narration** with no `<Response>` block / no JSON / no written file (e.g. a breakdown `outputs/output.md` that reads "I'll start by investigating…" with 0 `subtasks`); a `Response stalled mid-stream` / `isApiErrorMessage` marker in the raw output or the underlying `~/.claude/projects/.../<session>.jsonl` transcript; a turn that ended with `stop_reason: stop_sequence` + a synthetic error rather than `end_turn`. Most damaging on the **breakdown** step, where 0 subtasks → the BTA skips ALL worker fan-out (see task **TK-O-12**). The trigger is the nested `claude` CLI's byte-stream idle watchdog: when `CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS` is inherited too low (e.g. 20000 ms from an interactive session), a legitimate >20 s think-pause before a tool call / large write trips a synthetic "Response stalled mid-stream" that ends the turn (subprocess still exits 0).
- **Distinguishes from healthy**: O-23 is a DIFFERENT mechanism (a *complete* response captured as a noisy transcript); O-24 is an *incomplete/truncated* turn — the required structured output was never produced at all. Healthy: the deliverable contains the required structure (`<Response>`/JSON/file); no `stalled mid-stream`; turns end `end_turn`. Mitigations in current code: `ClaudeCodeCliInferencer.byte_stream_idle_timeout_ms=120000` forced into the spawned subprocess env (`_build_subprocess_env`), and, for the breakdown specifically, a `SubtaskStructureJudge` output-guardrail + `max_retry` that REJECT+retry a 0-subtask breakdown, with a LOUD `BREAKDOWN_EMPTY_DEGRADE` warning if all retries fail (never a silent skip).
- **Source**: Run `multimodal_plan_3flow_20260629_021219_1ca9970a` (v8 task run) — the breakdown turn stalled mid-stream (~21 s gap → synthetic error), emitted narration only, 0 subtasks parsed, the 3-flow pipeline was silently skipped, yet the run exited 0 in 41 min (vs a healthy ~2 h). 8 of 9 byte-identical sibling runs were unaffected (it's a transient ~11% event amplified by the too-low inherited timeout). Fixed across `claude_code_cli_inferencer.py` (scoped 120000 ms env) + `breakdown_then_aggregate_inferencer.py` (loud degrade) + `guardrails/subtask_structure_judge.py` + the two task configs.

---

## §3 Authoring Guide — Adding a NEW Common Observation

When a run reveals a NEW unhealthy pattern that:
- Could apply to ANY BTA-based tool (not tied to one tool's deliverable names or topology)
- Is purely a **runtime observation** (something an auditor can see post-run by inspecting workspace files or logs) — NOT a static config issue

Then add it HERE (not in a tool-specific `.agent/VERIFICATION.md`):

1. Add a new `O-N` entry in **§2** with exactly two sub-sections:
   - **Look for**: concrete, file-grep-able signals
   - **Distinguishes from healthy**: contrast with the expected pattern
2. Add a row to the **§1 Common Audit Pack** with the verify command + new reference
3. Update the "Last updated" date at the top
4. Update the per-tool `.agent/VERIFICATION.md` files if the new check needs tool-specific parameters

> **Important**: Do NOT inline root causes or past fix plan references. Causes vary across recurrences; including them risks anchoring future investigators to outdated diagnoses. Keep observations purely as "what unhealthy looks like."

> **Placement decision rule**:
> - Generic across BTA tools → here (`VERIFICATION_COMMON.md`)
> - References specific tool deliverable names, prompt slots, expected facet count, output shape → tool's own `VERIFICATION.md`

---

## §4 Tool-Specific Verification Docs

Each BTA-based tool has its own `VERIFICATION.md` that:
1. References this common catalog as the baseline
2. Provides tool-specific values for parametrized common checks (e.g., canonical deliverable name + size threshold for A9, expected facet count for A5/A13)
3. Adds tool-specific observations (e.g., role_setup's nested BTA structure, create_role's role-document format)
4. Provides a tool-specific wrapper that sets `$WS`, `$TOOL`, and runs the common one-liner

| Tool | Catalog location |
|------|-----------------|
| `task` | `task/.agent/VERIFICATION.md` (AF — relative: `../task/.agent/VERIFICATION.md`) |
| `sop`  | `sop/.agent/VERIFICATION.md` (AF, if present) |
| `understand_codebase` | `understand_codebase/.agent/VERIFICATION.md` (AF, if present) |
| `create_role` | `create_role/VERIFICATION.md` (OpenStartup-resident tool) |
| `role_setup`  | `role_setup/VERIFICATION.md` (OpenStartup-resident tool) |
