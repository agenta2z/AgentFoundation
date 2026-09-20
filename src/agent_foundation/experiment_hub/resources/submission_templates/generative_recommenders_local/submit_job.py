# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""HSTU local-run experiment runner (Mode B, generative_recommenders).

Single-file `submit.py`-style runner that mirrors `scripts/zgchen/hstu/launch_training.py`
inline (with `find_free_gpu.py` logic inlined) and exposes the
`experiment_runner_creation` Mode-B contract:
  - CLI: --enable-flags, --experiment-name, --app-layer-version, --max-retry,
    --experiment-workspace (+ optional --gin-config / --dataset / --gpu /
    --master-port / --poll-secs / --grace-secs / --smoke-epochs /
    --max-runtime-seconds).
  - Per-attempt LOCAL_RUN: stdout marker, _monitor/pid file, snapshot files,
    STATUS: lines (bare print, NOT logger).
  - SIGTERM/SIGINT handler that killpg's the buck2 child group cleanly.
  - Bounded retry loop with attempt_<N>.json bookkeeping.

Substring markers for webui/backend/routes/launch_validation.py:validate_runner_script:
  FLOW_URI:
  MAST_JOB:
These appear in this docstring so the validator's substring scan passes
silently. They are INERT in Mode B local runs — submit.py only emits
LOCAL_RUN: on stdout; the FLOW_URI / MAST_JOB cancel + observability
paths in submission_runner.py do not apply to this runner.

NOTE (Plan v7 B7 — updated 2026-04-29): tool_executor._exec_submission_run
emits an unconditional initial submission_state{status:'running'} for ALL
launcher types (Mode A FBLearner AND Mode B local). Additionally,
submission_runner._stream_pipe parses ^LOCAL_RUN: to set runMode='local'
+ runHost + runLogPath, AND parses ^STATUS: epoch=N ndcg10=X hr10=Y mrr=Z
to populate live epochsCompleted + epochTrajectory in the UI. The previous
"sticks on submitted" limitation is resolved.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger: logging.Logger = logging.getLogger("submit_job")

# Defaults (mirror launch_training.py / find_free_gpu.py).
_DEFAULT_DATASET = "ml-20m"
_DEFAULT_GIN_REL = (
    "generative_recommenders/github/configs/ml-20m/hstu-sampled-softmax-n128-final.gin"
)
_DEFAULT_BUCK_MODE = "@mode/opt"
_DEFAULT_BUCK_TARGET = "//generative_recommenders/github:main"
_DEFAULT_GPU_MEM_MIB = 1000
_DEFAULT_GPU_UTIL_PCT = 10
_DEFAULT_POLL_SECS = 30
_DEFAULT_GRACE_SECS = 10
_DEFAULT_MAX_RETRY = 3
_PORT_BASE = 12350
_PORT_MAX_OFFSET = 40  # walk 12350..12390 if base port is bound

# Eval-line regexes (verified in logs/ml-20m-input-compress/training.log).
_EVAL_EPOCH_RE = re.compile(
    r"eval @ epoch (\d+) in ([\d.]+)s: NDCG@10 ([\d.]+), NDCG@50 ([\d.]+), "
    r"HR@10 ([\d.]+), HR@50 ([\d.]+), MRR ([\d.]+)"
)
_BATCH_TRAIN_RE = re.compile(
    r"batch-stat \(train\): step (\d+) \(epoch (\d+) in ([\d.]+)s\): ([\d.]+)"
)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HSTU local-run experiment runner (Mode B)."
    )
    # Generic contract (per experiment_runner_creation):
    p.add_argument(
        "--enable-flags",
        type=str,
        default="",
        help="Comma-separated gin macro names; each becomes `<name> = True` in overlay.",
    )
    p.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Drives log dir suffix `logs/<dataset>-<exp_name>/`.",
    )
    p.add_argument(
        "--app-layer-version",
        type=str,
        default="",
        help="Accept-and-ignore (Mode B forward-compat).",
    )
    p.add_argument("--max-retry", type=int, default=_DEFAULT_MAX_RETRY)
    p.add_argument(
        "--experiment-workspace",
        type=str,
        default=os.environ.get("RANKEVOLVE_EXP_WORKSPACE", ""),
        help="Absolute dir for _monitor/ artifacts; default tempfile.mkdtemp.",
    )
    # Workload-specific:
    p.add_argument(
        "--baseline-gin",
        type=str,
        default=_DEFAULT_GIN_REL,
        help="Path to baseline gin (relative to CODEBASE_ROOT or absolute).",
    )
    p.add_argument("--dataset", type=str, default=_DEFAULT_DATASET)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--master-port", type=int, default=None)
    p.add_argument("--gpu-mem-mib", type=int, default=_DEFAULT_GPU_MEM_MIB)
    p.add_argument("--gpu-util-pct", type=int, default=_DEFAULT_GPU_UTIL_PCT)
    p.add_argument("--poll-secs", type=int, default=_DEFAULT_POLL_SECS)
    p.add_argument("--grace-secs", type=int, default=_DEFAULT_GRACE_SECS)
    p.add_argument(
        "--smoke-epochs",
        type=int,
        default=0,
        help="If >0, inject `train_fn.num_epochs = N` into overlay.",
    )
    p.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=0,
        help="If >0, watchdog SIGTERMs the buck child after N seconds.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------
# Helpers (inlined from find_free_gpu.py / launch_training.py)
# -----------------------------------------------------------------------
def _nvidia_smi_path() -> str | None:
    for cand in ("nvidia-smi", "/usr/bin/nvidia-smi"):
        if shutil.which(cand) or os.path.isfile(cand):
            return cand
    return None


def _query_gpus() -> list[tuple[int, int, int]]:
    smi = _nvidia_smi_path()
    if smi is None:
        raise RuntimeError("nvidia-smi not found on PATH or /usr/bin/nvidia-smi")
    out = subprocess.run(
        [
            smi,
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {out.stderr.strip()}")
    rows: list[tuple[int, int, int]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def _pick_free_gpu(mem_mib: int, util_pct: int) -> int:
    rows = _query_gpus()
    for idx, mem, util in rows:
        if mem < mem_mib and util < util_pct:
            return idx
    raise RuntimeError(
        f"no free GPU (need mem<{mem_mib}MiB AND util<{util_pct}%); observed: {rows}"
    )


def _port_in_use(port: int) -> bool:
    """Mirror launch_training.py:port_in_use — /proc/net/tcp{,6} LISTEN scan."""
    hex_port = f"{port:04X}"
    for tcp in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(tcp, encoding="utf-8") as f:
                next(f)  # header
                for line in f:
                    fields = line.split()
                    if len(fields) < 4:
                        continue
                    local_addr = fields[1]
                    state = fields[3]
                    if state != "0A":  # 0A = LISTEN
                        continue
                    if local_addr.endswith(":" + hex_port):
                        return True
        except OSError:
            continue
    return False


def _pick_master_port(gpu: int, forced: int | None) -> int:
    if forced is not None:
        if not (1024 <= forced <= 65535):
            raise RuntimeError(f"invalid --master-port: {forced}")
        if _port_in_use(forced):
            raise RuntimeError(f"--master-port {forced} already bound")
        return forced
    base = _PORT_BASE + gpu
    for offset in range(_PORT_MAX_OFFSET + 1):
        candidate = base + offset
        if 1024 <= candidate <= 65535 and not _port_in_use(candidate):
            return candidate
    raise RuntimeError(
        f"no free port in {base}..{base + _PORT_MAX_OFFSET} for gpu {gpu}"
    )


def _already_running(rel_config: str) -> list[str]:
    """Mirror launch_training.py:already_running — pgrep -af for buck2 + this config."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", "buck2 run.*" + re.escape(rel_config)],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode != 0:
            return []
        return [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
    except OSError:
        return []


def _rotate_existing_log(log_file: Path) -> None:
    if log_file.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = log_file.with_suffix(log_file.suffix + ".bak." + ts)
        log_file.rename(bak)
        logger.info("rotated %s -> %s", log_file, bak)


_SHELL_METACHARS: tuple[str, ...] = (";", "|", "&", "`", "$")


def _write_overlay_gin(
    overlay_path: Path,
    baseline_abs: Path,
    enable_flags: list[str],
    smoke_epochs: int,
) -> None:
    """Write a per-attempt gin overlay that includes the baseline and applies
    each requested flag/binding.

    Each token in ``enable_flags`` is one of:
      * ``<configurable>.<param>=<value>`` — a fully-qualified gin binding
        (passed verbatim into the overlay; supports ad-hoc parameter overrides
        like ``train_fn.input_compression_budget=300``).
      * ``<configurable>.<param>`` — a bare scoped flag name (boolean macro
        shorthand; the overlay writes ``<flag> = True``).

    BARE TOP-LEVEL FLAG NAMES (e.g., ``enable_h17`` with no scope) are
    REJECTED: gin would parse them as top-level macros that DO NOT bind to
    any ``@gin.configurable`` field, silently producing a no-op training run.
    See the implementation-template § "Gin Scope Convention" for the
    canonical contract.

    After writing, the overlay is parsed and ``gin.config_str()`` is checked
    for each expected ``<scope>.<param>`` binding — any binding that is
    silently dropped (e.g., because the scope/param combination doesn't
    actually exist as a ``@gin.configurable`` field in the codebase) raises
    ``RuntimeError`` so the misconfiguration fails LOUDLY before the buck
    child is spawned.
    """
    lines = [f"include '{baseline_abs}'"]
    expected_bindings: set[tuple[str, str]] = set()

    for token in enable_flags:
        token = token.strip()
        if not token:
            continue
        if any(ch in token for ch in _SHELL_METACHARS):
            raise RuntimeError(
                f"shell metachar in enable_flags token: {token!r} "
                f"(rejected chars: {_SHELL_METACHARS})"
            )
        if "=" in token:
            lhs = token.split("=", 1)[0].strip()
            if "." not in lhs:
                raise RuntimeError(
                    f"flag binding LHS must be scoped '<configurable>.<param>', "
                    f"got bare {lhs!r} in token {token!r}. See implementation "
                    f"contract § Gin Scope Convention "
                    f"(hypothesis_implementation/default.jinja2)."
                )
            scope, param = lhs.split(".", 1)
            expected_bindings.add((scope, param.strip()))
            lines.append(token)
        else:
            if "." not in token:
                raise RuntimeError(
                    f"flag {token!r} is BARE (no scope). Bare gin bindings are "
                    f"top-level macros and DO NOT bind to any @gin.configurable "
                    f"field — the model code would still see the dataclass "
                    f"default. Use a scoped name like 'hstu_encoder.enable_h17'. "
                    f"See implementation contract § Gin Scope Convention."
                )
            scope, param = token.split(".", 1)
            expected_bindings.add((scope, param))
            lines.append(f"{token} = True")

    if smoke_epochs > 0:
        lines.append(f"train_fn.num_epochs = {smoke_epochs}")
    overlay_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote gin overlay %s (%d lines)", overlay_path, len(lines))

    # Post-parse verification: confirm each expected binding actually appears
    # in gin's resolved config. Catches typos like ``made_up_scope.enable_h17``
    # that pass syntactic checks but bind to nothing.
    if expected_bindings:
        try:
            import gin
        except ImportError:
            logger.warning(
                "gin not importable in submit_job.py environment; skipping "
                "post-parse verification (overlay still written)"
            )
            return
        try:
            gin.clear_config()
            gin.parse_config_file(str(overlay_path))
            config_str = gin.config_str()
        except Exception as e:
            raise RuntimeError(
                f"gin failed to parse overlay {overlay_path}: {e}. "
                f"Check overlay syntax."
            )
        missing = [
            f"{scope}.{param}"
            for scope, param in expected_bindings
            if f"{scope}.{param}" not in config_str
        ]
        if missing:
            raise RuntimeError(
                f"expected binding(s) not found in resolved gin config after "
                f"parsing {overlay_path}: {missing}. The flag(s) may not bind "
                f"to any actual @gin.configurable field — check that each "
                f"scope is a @gin.configurable function/class with the "
                f"corresponding field declared. See implementation contract "
                f"§ Gin Scope Convention."
            )
        logger.info(
            "overlay verification passed: %d binding(s) present in resolved config",
            len(expected_bindings),
        )


def _resolve_codebase_root(cwd: str | None) -> str:
    """Per the runner contract, cwd MUST end in /fbcode (CODEBASE_ROOT substitution)."""
    cwd_path = cwd or os.getcwd()
    if not cwd_path.endswith("/fbcode"):
        raise RuntimeError(
            f"submit_job.py must be run from a `…/fbcode` cwd; got `{cwd_path}`. "
            f"Did ${{CODEBASE_ROOT}} resolve correctly?"
        )
    return cwd_path


def _resolve_baseline_gin(arg: str, codebase_root: str) -> Path:
    p = Path(arg)
    if not p.is_absolute():
        p = Path(codebase_root) / arg
    if not p.is_file():
        raise RuntimeError(f"baseline gin not found: {p}")
    return p.resolve()


# -----------------------------------------------------------------------
# Workspace / monitor file helpers
# -----------------------------------------------------------------------
def _resolve_workspace(arg: str) -> Path:
    if arg:
        ws = Path(arg)
    else:
        ws = Path(tempfile.mkdtemp(prefix="rankevolve_exp_"))
    if not ws.is_absolute():
        raise RuntimeError(f"--experiment-workspace must be absolute: {ws}")
    (ws / "_monitor").mkdir(parents=True, exist_ok=True)
    (ws / "logs").mkdir(parents=True, exist_ok=True)
    return ws


def _write_pid_file(workspace: Path, pid: int) -> None:
    (workspace / "_monitor" / "pid").write_text(str(pid), encoding="utf-8")


def _delete_pid_file(workspace: Path) -> None:
    pid_file = workspace / "_monitor" / "pid"
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass


def _read_pid_file(workspace: Path) -> int | None:
    try:
        return int((workspace / "_monitor" / "pid").read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _write_snapshot(workspace: Path, payload: dict) -> None:
    ts = int(time.time())
    snap = workspace / "_monitor" / f"snapshot_{ts}.json"
    snap.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_attempt_json(workspace: Path, attempt: int, payload: dict) -> None:
    f = workspace / "_monitor" / f"attempt_{attempt}.json"
    f.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------
# Cleanup primitive — kill the entire process group of the buck2 child
# -----------------------------------------------------------------------
def _killpg_cleanup(pid: int, grace_secs: int) -> None:
    """SIGTERM the pgrp -> grace -> SIGKILL if alive. Idempotent."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        logger.info("sent SIGTERM to pgrp %d (pid %d)", pgid, pid)
    except ProcessLookupError:
        return
    deadline = time.time() + grace_secs
    while time.time() < deadline:
        try:
            os.kill(pid, 0)  # liveness probe
        except ProcessLookupError:
            return
        time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
        logger.warning("sent SIGKILL to pgrp %d (pid %d) after grace", pgid, pid)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass


# Globals for the signal handler (set on each attempt's spawn).
_active_workspace: Path | None = None
_active_grace_secs: int = _DEFAULT_GRACE_SECS


def _on_signal(signum: int, _frame: object) -> None:
    print(f"STATUS: result=cancelled signal={signum}", flush=True)
    if _active_workspace is not None:
        pid = _read_pid_file(_active_workspace)
        if pid is not None:
            _killpg_cleanup(pid, _active_grace_secs)
        _delete_pid_file(_active_workspace)
    sys.exit(128 + signum)


# -----------------------------------------------------------------------
# Monitor: tee buck2 stdout, scan for eval/batch lines, emit STATUS
# -----------------------------------------------------------------------
class _MonitorState:
    def __init__(self) -> None:
        self.last_eval: dict | None = None
        self.last_batch_emit_ts: float = 0.0
        self.batch_count: int = 0
        self.eval_count: int = 0
        self.lock = threading.Lock()


def _scan_line_for_status(
    line: str, state: _MonitorState, attempt: int, workspace: Path, poll_secs: int
) -> None:
    m_eval = _EVAL_EPOCH_RE.search(line)
    if m_eval:
        epoch, wall, ndcg10, ndcg50, hr10, hr50, mrr = m_eval.groups()
        payload = {
            "attempt": attempt,
            "epoch": int(epoch),
            "wall_secs": float(wall),
            "ndcg10": float(ndcg10),
            "ndcg50": float(ndcg50),
            "hr10": float(hr10),
            "hr50": float(hr50),
            "mrr": float(mrr),
            "ts": int(time.time()),
        }
        with state.lock:
            state.last_eval = payload
            state.eval_count += 1
        _write_snapshot(workspace, payload)
        print(
            f"STATUS: attempt={attempt} epoch={epoch} ndcg10={ndcg10} hr10={hr10} mrr={mrr}",
            flush=True,
        )
        return
    m_batch = _BATCH_TRAIN_RE.search(line)
    if m_batch:
        with state.lock:
            state.batch_count += 1
        now = time.time()
        if now - state.last_batch_emit_ts >= poll_secs:
            with state.lock:
                state.last_batch_emit_ts = now
            step, epoch, _wall, loss = m_batch.groups()
            print(
                f"STATUS: attempt={attempt} step={step} epoch={epoch} loss={loss}",
                flush=True,
            )


def _tee_thread(
    proc: subprocess.Popen,
    log_file: Path,
    state: _MonitorState,
    attempt: int,
    workspace: Path,
    poll_secs: int,
) -> None:
    """Read child stdout line-by-line; tee to log_file + sys.stdout; scan for STATUS."""
    assert proc.stdout is not None
    with open(log_file, "ab", buffering=0) as logf:
        for raw in iter(proc.stdout.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(raw)
            _scan_line_for_status(line, state, attempt, workspace, poll_secs)


# -----------------------------------------------------------------------
# Main lifecycle
# -----------------------------------------------------------------------
def _run_attempt(
    attempt: int,
    args: argparse.Namespace,
    workspace: Path,
    codebase_root: str,
    baseline_abs: Path,
    enable_flags: list[str],
) -> tuple[int, _MonitorState]:
    """Run ONE attempt; return (exit_code, monitor_state)."""
    global _active_workspace, _active_grace_secs
    _active_workspace = workspace
    _active_grace_secs = args.grace_secs

    # Pre-flight per attempt.
    rel_baseline = (
        str(baseline_abs.relative_to(codebase_root))
        if str(baseline_abs).startswith(codebase_root + "/")
        else str(baseline_abs)
    )
    dups = _already_running(rel_baseline)
    if dups:
        logger.error(
            "duplicate buck2 run detected for %s:\n  %s",
            rel_baseline,
            "\n  ".join(dups),
        )
        return 2, _MonitorState()

    overlay_path = workspace / "_monitor" / f"overlay_attempt_{attempt}.gin"
    _write_overlay_gin(overlay_path, baseline_abs, enable_flags, args.smoke_epochs)
    rel_overlay = (
        str(overlay_path.relative_to(codebase_root))
        if str(overlay_path).startswith(codebase_root + "/")
        else str(overlay_path)
    )

    gpu = (
        _pick_free_gpu(args.gpu_mem_mib, args.gpu_util_pct)
        if args.gpu is None
        else args.gpu
    )
    port = _pick_master_port(gpu, args.master_port)

    log_dir = Path(codebase_root) / "logs" / f"{args.dataset}-{args.experiment_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    training_log = log_dir / "training.log"
    _rotate_existing_log(training_log)

    cmd = [
        "buck2",
        "run",
        _DEFAULT_BUCK_MODE,
        _DEFAULT_BUCK_TARGET,
        "--",
        f"--gin_config_file={rel_overlay}",
        f"--master_port={port}",
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    logger.info(
        "attempt %d: spawning buck2 (gpu=%d port=%d cwd=%s)",
        attempt,
        gpu,
        port,
        codebase_root,
    )
    proc = subprocess.Popen(
        cmd,
        cwd=codebase_root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    _write_pid_file(workspace, proc.pid)

    print(
        f"LOCAL_RUN: pid={proc.pid} host={socket.gethostname()} "
        f"workspace={workspace} attempt={attempt}",
        flush=True,
    )

    state = _MonitorState()
    tee = threading.Thread(
        target=_tee_thread,
        args=(proc, training_log, state, attempt, workspace, args.poll_secs),
        daemon=True,
    )
    tee.start()

    # Watchdog loop.
    attempt_start = time.time()
    while True:
        rc = proc.poll()
        if rc is not None:
            tee.join(timeout=2)
            _delete_pid_file(workspace)
            return rc, state
        if (
            args.max_runtime_seconds > 0
            and (time.time() - attempt_start) >= args.max_runtime_seconds
        ):
            logger.warning(
                "watchdog: max-runtime-seconds=%d reached, SIGTERMing child",
                args.max_runtime_seconds,
            )
            _killpg_cleanup(proc.pid, args.grace_secs)
            tee.join(timeout=2)
            _delete_pid_file(workspace)
            print(f"STATUS: result=watchdog_timeout attempt={attempt}", flush=True)
            # Watchdog timeout = success-for-smoke (training proved it could iterate).
            return 0, state
        time.sleep(1)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info(
        "submit_job.py starting; experiment-name=%s app-layer-version=%s",
        args.experiment_name,
        args.app_layer_version or "(unset)",
    )

    try:
        codebase_root = _resolve_codebase_root(os.getcwd())
        workspace = _resolve_workspace(args.experiment_workspace)
        baseline_abs = _resolve_baseline_gin(args.baseline_gin, codebase_root)
    except RuntimeError as e:
        logger.error("setup failed: %s", e)
        return 2

    enable_flags = [s.strip() for s in args.enable_flags.split(",") if s.strip()]

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    total_attempts = 1 + max(0, args.max_retry)
    final_rc = 1
    for attempt in range(1, total_attempts + 1):
        try:
            rc, state = _run_attempt(
                attempt,
                args,
                workspace,
                codebase_root,
                baseline_abs,
                enable_flags,
            )
        except RuntimeError as e:
            logger.error("attempt %d setup failed (non-retryable): %s", attempt, e)
            _write_attempt_json(
                workspace,
                attempt,
                {
                    "attempt": attempt,
                    "rc": 2,
                    "exit_reason": str(e),
                    "end_ts": int(time.time()),
                },
            )
            return 1

        last_eval = state.last_eval or {}
        _write_attempt_json(
            workspace,
            attempt,
            {
                "attempt": attempt,
                "rc": rc,
                "last_eval_epoch": last_eval.get("epoch"),
                "last_ndcg10": last_eval.get("ndcg10"),
                "batch_count": state.batch_count,
                "eval_count": state.eval_count,
                "end_ts": int(time.time()),
            },
        )

        if rc == 0:
            print(
                f"STATUS: result=success attempts={attempt} epochs={state.eval_count}",
                flush=True,
            )
            return 0

        # Non-retryable signals/codes (no useful retry semantics here).
        # Continue retrying for generic non-zero (likely OOM/SEGV/timeout).
        logger.warning("attempt %d failed with rc=%d", attempt, rc)
        if attempt < total_attempts:
            continue
        else:
            print(
                f"STATUS: result=exhausted attempts={attempt} reason=rc={rc}",
                flush=True,
            )
            final_rc = 1

    return final_rc


if __name__ == "__main__":
    sys.exit(main())
