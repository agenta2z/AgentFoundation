"""Quick smoke test for the 3-path consolidation refactor.

Run via:
    buck2 run fbcode//_tony_dev/CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers:contract_smoke
"""

import os
import sys
import tempfile

import attr


def main() -> int:
    from agent_foundation.common.inferencers.inferencer_base import InferencerBase

    # 1. IB has target_path + effective_cwd
    ib_fields = {f.name for f in attr.fields(InferencerBase)}
    assert "target_path" in ib_fields, "target_path missing from IB"
    assert attr.fields(InferencerBase).target_path.default is None
    assert isinstance(InferencerBase.effective_cwd, property)
    print("OK: InferencerBase has target_path + effective_cwd")

    # 2. TIB inherits — same default, same name
    from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_base import (
        TerminalInferencerBase,
    )
    tib_target = attr.fields(TerminalInferencerBase).target_path
    assert tib_target.default is None, (
        f"TIB target_path default should be None, got {tib_target.default!r} — "
        "indicates TIB redeclared with a different default"
    )
    # Confirm TIB's effective_cwd resolves through IB's property
    assert "effective_cwd" not in TerminalInferencerBase.__dict__, (
        "TIB has its own effective_cwd in __dict__ — should inherit IB's"
    )
    print("OK: TIB inherits target_path + effective_cwd from IB")

    # 3. ClaudeCodeSdk migrated
    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer import (
        ClaudeCodeSdkInferencer,
    )
    cc = {f.name for f in attr.fields(ClaudeCodeSdkInferencer)}
    assert "root_folder" not in cc
    assert "target_path" in cc
    print("OK: ClaudeCodeSdk migrated")

    # 4. DevmateCli no repo_path
    from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
        DevmateCliInferencer,
    )
    dc = {f.name for f in attr.fields(DevmateCliInferencer)}
    assert "repo_path" not in dc
    assert "target_path" in dc
    assert attr.fields(DevmateCliInferencer).has_local_access.default is True
    print("OK: DevmateCli migrated, has_local_access=True")

    # 5. DevmateSDK no root_folder, no source_path shadow
    from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (
        DevmateSDKInferencer,
    )
    ds = {f.name for f in attr.fields(DevmateSDKInferencer)}
    assert "root_folder" not in ds
    assert "target_path" in ds
    # source_path no longer shadowed: inherited from IB with same default (None).
    # Use `inherited` Attribute flag (attrs >=22). If unavailable, falls back to
    # default-match heuristic.
    sd_source = attr.fields(DevmateSDKInferencer).source_path
    ib_source_default = attr.fields(InferencerBase).source_path.default
    if hasattr(sd_source, "inherited"):
        assert sd_source.inherited, "DevmateSDK redeclares source_path (shadow)"
    else:
        assert sd_source.default == ib_source_default
    assert attr.fields(DevmateSDKInferencer).has_local_access.default is True
    print("OK: DevmateSDK migrated, source_path inherited from IB (no shadow)")

    # 6. RovoDevServe migrated
    from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_serve_inferencer import (
        RovoDevServeInferencer,
    )
    rs = {f.name for f in attr.fields(RovoDevServeInferencer)}
    assert "working_dir" not in rs
    assert "target_path" in rs
    assert attr.fields(RovoDevServeInferencer).has_local_access.default is True
    print("OK: RovoDevServe migrated, has_local_access=True")

    # 7. ToolAs migrated
    from agent_foundation.common.inferencers.agentic_inferencers.tool_inferencers.tool_as_inferencer import (
        ToolAsInferencer,
    )
    ta = {f.name for f in attr.fields(ToolAsInferencer)}
    assert "cwd" not in ta
    assert "target_path" in ta
    print("OK: ToolAs migrated")

    # 8. KiroCli leak fix
    from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.kiro_cli_inferencer import (
        KiroCliInferencer,
    )
    assert attr.fields(KiroCliInferencer).has_local_access.default is True
    print("OK: KiroCli has_local_access=True (leak fixed)")

    # 9. Legacy ClaudeCodeInferencer deleted
    try:
        import importlib
        importlib.import_module(
            "agent_foundation.common.inferencers.agentic_inferencers."
            "external.claude_code.claude_code_inferencer"
        )
        print("FAIL: legacy ClaudeCodeInferencer still importable")
        return 1
    except ImportError:
        print("OK: legacy ClaudeCodeInferencer deleted")

    # 10. get_source_repo_root removed
    from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
        common as devmate_common,
    )
    assert not hasattr(devmate_common, "get_source_repo_root")
    assert hasattr(devmate_common, "_detect_fbsource_root_for")
    print("OK: get_source_repo_root → _detect_fbsource_root_for")

    # 11. effective_cwd fallback chain (using ClaudeCodeSdk as concrete)
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )
    inf_a = ClaudeCodeSdkInferencer(target_path="/a")
    assert inf_a.effective_cwd == "/a"
    with tempfile.TemporaryDirectory() as tmpdir:
        inf_ws = ClaudeCodeSdkInferencer(workspace=InferencerWorkspace(root=tmpdir))
        assert inf_ws.effective_cwd == tmpdir
    inf_default = ClaudeCodeSdkInferencer()
    assert inf_default.effective_cwd == os.getcwd()
    print("OK: effective_cwd fallback chain works (target_path > workspace.root > os.getcwd())")

    # 12. DevmateCli ~/fbsource default + no cd_script
    from unittest.mock import patch
    with patch(
        "agent_foundation.common.inferencers.agentic_inferencers"
        ".external.devmate.devmate_cli_inferencer.sync_config_to_target"
    ):
        dc_inf = DevmateCliInferencer(target_path="/tmp/foo_devmate_test")
    assert dc_inf.target_path == "/tmp/foo_devmate_test"
    pre_scripts = dc_inf.pre_exec_scripts or []
    for script in pre_scripts:
        assert not script.startswith('cd "/tmp/foo_devmate_test"'), (
            f"cd_script still present: {script!r}"
        )
    print("OK: DevmateCli no cd_script in pre_exec_scripts")

    # 13. DevmateCli default ~/fbsource
    with patch(
        "agent_foundation.common.inferencers.agentic_inferencers"
        ".external.devmate.devmate_cli_inferencer.sync_config_to_target"
    ):
        dc_default = DevmateCliInferencer()
    assert dc_default.target_path == os.path.expanduser("~/fbsource")
    print("OK: DevmateCli defaults target_path to ~/fbsource")

    # 14. DevmateCli rejects repo_path
    try:
        DevmateCliInferencer(repo_path="/tmp/x")
        print("FAIL: DevmateCli still accepts repo_path kwarg")
        return 1
    except TypeError as e:
        if "repo_path" in str(e):
            print("OK: DevmateCli rejects repo_path kwarg (TypeError)")
        else:
            print(f"FAIL: unexpected TypeError: {e}")
            return 1

    # 15. Standalone DevmateSDKInferencer() construction works (Sweep F)
    # Previously got repo_root=None; now gets Path(os.getcwd()) via
    # effective_cwd. Should not throw.
    DevmateSDKInferencer()
    print("OK: standalone DevmateSDKInferencer() constructs cleanly")

    # 16. ClaudeCodeSdk also rejects root_folder
    try:
        ClaudeCodeSdkInferencer(root_folder="/tmp/x")
        print("FAIL: ClaudeCodeSdk still accepts root_folder kwarg")
        return 1
    except TypeError as e:
        if "root_folder" in str(e):
            print("OK: ClaudeCodeSdk rejects root_folder kwarg (TypeError)")
        else:
            print(f"FAIL: unexpected TypeError: {e}")
            return 1

    # 17. RovoDevServe rejects working_dir
    try:
        RovoDevServeInferencer(working_dir="/tmp/x")
        print("FAIL: RovoDevServe still accepts working_dir kwarg")
        return 1
    except TypeError as e:
        if "working_dir" in str(e):
            print("OK: RovoDevServe rejects working_dir kwarg (TypeError)")
        else:
            print(f"FAIL: unexpected TypeError: {e}")
            return 1

    # 18. ToolAs rejects cwd
    try:
        ToolAsInferencer(tool_name="test", command=["python3"], cwd="/tmp/x")
        print("FAIL: ToolAs still accepts cwd kwarg")
        return 1
    except TypeError as e:
        if "cwd" in str(e):
            print("OK: ToolAs rejects cwd kwarg (TypeError)")
        else:
            print(f"FAIL: unexpected TypeError: {e}")
            return 1

    print()
    print("ALL CONTRACT SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
