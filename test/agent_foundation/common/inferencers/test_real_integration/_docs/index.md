# `test_real_integration/_docs/` — investigation log

Long-lived debugging notes, post-mortems, and design walkthroughs for the
real-CLI integration tests in this directory. Pinned here (rather than in
git history or PR descriptions) so future maintainers hitting similar
opaque-failure modes can find the playbook fast.

## What lives here

Anything that would be valuable to someone debugging a real-CLI test six
months from now and doesn't belong in code comments:

- **Debugging post-mortems** — narrative of an investigation: starting
  symptom, hypotheses tried, dead ends, final fix, meta-lessons.
  Filename convention: `debugging_analogue_<YYYYMMDD>_<HHMM>.md`.
- **Design walkthroughs** — for non-obvious test wiring (e.g., how the
  aggregator routing tier system works, why the diagnostic DEBUG log
  exists). Filename convention: `design_<topic>.md`.
- **Run-of-the-mill READMEs** for sub-areas (e.g., how to run real-CLI
  tests under different profiles, how the `MFDUAL_REAL_NO_KIRO`
  bypass works). Filename convention: `readme_<topic>.md`.

## What does NOT live here

- Test code itself (lives one directory up).
- Production code (lives in `src/`).
- Throwaway notes (use scratch files, not this folder).
- Information that could be derived from `git log` or PR descriptions.

## Index of artifacts (most recent first)

| Date | File | Topic | Outcome |
| ---- | ---- | ----- | ------- |
| 2026-04-26 | [`debugging_analogue_20260426_1549.md`](debugging_analogue_20260426_1549.md) | `test_multi_flow_dual_real.py` 6-iteration debugging journey covering 8 distinct bugs (session resumption, dispatch state lifetime, swallowed retry exceptions, broad retry filter, silent dispatch no-op, Windows cp1252 encoding on `→`, BTA tuple-routing mis-ordering, hardcoded test path) | ✅ Test green; 145/145 unit tests pass; 19 new unit tests added |

## Conventions for future contributors

When adding a new debugging post-mortem:

1. **Lead with the observable symptom**, not the root cause. The next
   debugger will be searching by symptom (`grep -r "winner_idx: None"
   _docs/`), not by your understanding of the bug.
2. **Capture the actual log/code/filesystem evidence** that resolved
   each step. Future you will not remember why you ruled out
   hypothesis X — write it down.
3. **Distinguish "what we planned" from "what actually fixed it."**
   The post-mortem analysis is rarely 100% right; documenting where
   it was wrong is the most useful part for future investigations.
4. **End with a playbook section.** A concrete numbered list of
   "if you see X, check Y, run Z" — not just narrative.
5. **Cite by function name and file path, not just line number.**
   Line numbers drift; `grep -n "def foo" path/to/file.py` always works.
