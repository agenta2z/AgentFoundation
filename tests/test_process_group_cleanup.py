"""Test that start_new_session + killpg kills orphaned grandchildren.

Verifies the core mechanism used by TerminalSessionInferencerBase to
prevent orphan subprocess leaks when CLI tools (e.g., acli rovodev)
spawn child processes (MCP servers) that outlive the main process.
"""

import asyncio
import os
import signal
import sys
import textwrap
import unittest

# Grandchild redirects stdout/stderr to /dev/null (releasing the pipe
# back to the parent) then sleeps — mimicking an MCP server that opens
# its own sockets instead of writing to inherited pipes.
_CHILD_SCRIPT = textwrap.dedent("""\
    import os, sys, time
    gc = os.fork()
    if gc == 0:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        time.sleep(30)
        os._exit(0)
    print(gc, flush=True)
    os._exit(0)
""")


@unittest.skipUnless(hasattr(os, "killpg"), "POSIX-only (os.killpg)")
class TestProcessGroupCleanup(unittest.TestCase):

    def test_killpg_reaches_orphaned_grandchild(self):
        """Grandchild survives parent exit but is killed via PGID."""
        async def run():
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _CHILD_SCRIPT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, _ = await process.communicate()
            gc_pid = int(stdout.strip())
            pgid = process.pid
            try:
                os.kill(gc_pid, 0)  # grandchild alive

                os.killpg(pgid, signal.SIGKILL)
                await asyncio.sleep(0.3)

                with self.assertRaises(ProcessLookupError):
                    os.kill(gc_pid, 0)  # grandchild dead
            finally:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

        asyncio.run(run())

    def test_pgid_persists_after_leader_reaped(self):
        """PGID remains reachable via killpg after the session leader exits."""
        async def run():
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _CHILD_SCRIPT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, _ = await process.communicate()
            gc_pid = int(stdout.strip())
            pgid = process.pid
            try:
                # Leader is reaped (communicate waited), but PGID lives on
                os.killpg(pgid, 0)  # group exists

                os.killpg(pgid, signal.SIGKILL)
                await asyncio.sleep(0.3)

                with self.assertRaises(ProcessLookupError):
                    os.killpg(pgid, 0)  # group gone

                with self.assertRaises(ProcessLookupError):
                    os.kill(gc_pid, 0)  # grandchild gone
            finally:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
