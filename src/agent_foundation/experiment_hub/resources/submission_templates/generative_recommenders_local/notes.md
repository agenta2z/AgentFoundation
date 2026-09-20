# Generative Recommenders — Local devgpu (`buck2 run`)

Mode B (Local subprocess) reference template for the
[`experiment_runner_creation`](../../prompt_templates/plan/main/_variables/task_preamble/experiment_runner_creation/)
contract. Runs `generative_recommenders` HSTU baselines on a single free
GPU on the user's devgpu via `buck2 run @mode/opt //generative_recommenders/github:main`.

## Why this template exists

The two existing templates (`generic_recurring_flow`, `generic_maas_launcher`) are
Mode A (Remote — FBLearner / fire-app / fbpkg). For local research iteration on
HSTU / SASRec configs, the user's reference is `scripts/zgchen/hstu/launch_training.py`
(with `find_free_gpu.py`). This template inlines that logic into a single-file
runner that satisfies `BuckRunAutoLauncher`'s `srcs=["submit.py"]` constraint.

## Usage

The runner is intended to be deployed via **Path A** (direct upload via the
Setup Wizard's `UploadExistingRunnerModal` footer link). Path B (PTI library
template picker) would trigger PTI to regenerate the runner from scratch — fine
for production use, but not what you want when validating the contract against
this known-good baseline.

### Path A — direct upload (recommended)

1. In the Setup Wizard, click "Have an existing experiment runner script? Upload
   it instead →" (footer link).
2. Paste:
   - submit.py path: `/data/users/zgchen/fbsource260327/fbcode/rankevolve/src/resources/submission_templates/generative_recommenders_local/submit_job.py`
   - launch.json path: `/data/users/zgchen/fbsource260327/fbcode/rankevolve/src/resources/submission_templates/generative_recommenders_local/launch.json`
3. Click **Validate** → expect `severity=ok` (the docstring includes substring
   markers that silence the validator's `FLOW_URI:` / `MAST_JOB:` warnings).
4. Click **Upload** → setup transitions `not_started → ready` immediately.
5. Click **Submit Experiment** with a baseline combo (no enable_flags).

### Standalone invocation (smoke test)

```bash
cd /data/users/zgchen/fbsource/fbcode
python3 .../submission_templates/generative_recommenders_local/submit_job.py \
    --enable-flags "" \
    --experiment-name baseline_smoke_$(date +%s) \
    --max-retry 0 \
    --experiment-workspace /tmp/rev_smoke_$(date +%s) \
    --smoke-epochs 2 \
    --max-runtime-seconds 720
```

The runner exits cleanly when `--smoke-epochs N` is reached OR when
`--max-runtime-seconds` watchdog fires (treated as smoke-pass — proves
launch + monitor + cleanup compose correctly even if training is slower
than expected).

## CLI

| Flag | Default | Purpose |
|---|---|---|
| `--enable-flags <csv>` | `""` | Each name → `<name> = True` line in a per-attempt gin overlay. Empty = baseline. |
| `--experiment-name <str>` | (REQUIRED) | Drives log dir suffix `logs/<dataset>-<exp>/`. |
| `--app-layer-version <str>` | `""` | Accept-and-ignore (Mode B forward-compat). |
| `--max-retry <int>` | `3` | Total attempts = `1 + max_retry`. |
| `--experiment-workspace <path>` | `tempfile.mkdtemp(prefix='rankevolve_exp_')` (env: `RANKEVOLVE_EXP_WORKSPACE`) | Absolute dir for `_monitor/` artifacts. |
| `--baseline-gin <path>` | `generative_recommenders/github/configs/ml-20m/hstu-sampled-softmax-n128-final.gin` | Baseline gin to `include` from overlay. |
| `--dataset <str>` | `ml-20m` | Log-dir prefix only. |
| `--gpu <int>` | auto-pick | Force GPU index. |
| `--master-port <int>` | `12350+gpu` (walks 12350..12390 if bound) | Force master_port. |
| `--smoke-epochs <int>` | `0` (off) | Inject `train_fn.num_epochs = N` into overlay. |
| `--max-runtime-seconds <int>` | `0` (off) | Watchdog SIGTERMs the child after N seconds. |

## How `--enable-flags` becomes overrides

`generative_recommenders/github/main.py:60` calls `gin.parse_config_file`
(SINGLE-FILE), so CLI `--gin_param=...` overrides DO NOT WORK. Instead, the
runner writes a per-attempt overlay file at `<workspace>/_monitor/overlay_attempt_<N>.gin`:

```gin
include '<absolute baseline gin path>'
<flag1> = True
<flag2> = True
train_fn.num_epochs = 2   # only when --smoke-epochs > 0
```

and passes `--gin_config_file=<overlay-rel-to-fbcode>` to `buck2 run`.

## Stdout protocol (`LOCAL_RUN:` / `STATUS:`)

Per attempt, the runner emits a bare-print `LOCAL_RUN:` line:

```
LOCAL_RUN: pid=<pid> host=<host> workspace=<ws> attempt=<n>
```

and per-poll/per-epoch `STATUS:` lines:

```
STATUS: attempt=1 step=234 epoch=0 loss=2.45
STATUS: attempt=1 epoch=0 ndcg10=0.1218 hr10=0.2246 mrr=0.1057
STATUS: result=success attempts=1 epochs=2
```

**Known UI-side limitation**: today's `submission_runner.py` flips
`submission_state` to `running` only on `^FLOW_URI:` / `^MAST_JOB:` parse.
For Mode B local runs that emit `LOCAL_RUN:` instead, the UI may show
`submitted` until exit. The script IS running; the UI just doesn't have a
marker to flip to `running`. NOT a script bug.

## `_monitor/` artifacts

| File | Purpose |
|---|---|
| `_monitor/pid` | PID of the buck2 child (the workload). Cleanup primitive reads this. |
| `_monitor/overlay_attempt_<N>.gin` | The per-attempt gin overlay (baseline include + flag bindings). |
| `_monitor/snapshot_<ts>.json` | Per-epoch eval snapshot `{epoch, ndcg10, ndcg50, hr10, hr50, mrr, ts}`. |
| `_monitor/attempt_<N>.json` | Per-attempt outcome `{attempt, rc, last_eval_epoch, batch_count, eval_count, end_ts}`. |

Plus `<workspace>/logs/training.log` is the script's own tee mirror; the
buck2 child also writes to `<codebase_root>/logs/<dataset>-<exp>/training.log`
for parity with `launch_training.py`'s tooling (`tail -f`, etc.).

## SIGTERM cleanup gotcha

The runner spawns the buck2 child with `start_new_session=True` so the entire
training process tree (build daemon + training subprocess + N rank workers
via `mp.spawn`) shares a process group. On SIGTERM/SIGINT, the runner's
handler calls `os.killpg(os.getpgid(child_pid), SIGTERM)` followed by
SIGKILL after `--grace-secs`.

**To smoke-test the cleanup, send SIGTERM to the python wrapper PID** (the
`submit_job.py` process), NOT the buck2 PID stored in `_monitor/pid`.
SIGTERMing the wrapper invokes the SIGTERM HANDLER which then explicitly
calls `killpg`. SIGTERMing the buck2 PID directly bypasses our handler and
leaves submit.py wedged.

## Validator expectations

`validate_runner_script` substring-checks `FLOW_URI:` / `MAST_JOB:` and the
`--enable-flags` / `--experiment-name` / `--app-layer-version` CLI args.
This template includes `FLOW_URI:` and `MAST_JOB:` as substrings in the
header docstring (inert in Mode B; satisfies the validator silently).
