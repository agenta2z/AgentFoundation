These recovery prompt templates were archived when the recovery verdict taxonomy
was simplified to "Option C" (PASS / RETRY / UPDATE).

  judge.jinja2                -> superseded by ../judge.jinja2   (now a pure sanity gate emitting PASS / RETRY / UPDATE)
  retry_with_reference.jinja2 -> superseded by ../retry.jinja2   (RETRY: re-run from scratch, always carrying the prior attempt as a negative-example reference)
  continue.jinja2             -> superseded by ../update.jinja2  (UPDATE: edit/complete the prior output in place; subsumes the old CONTINUE)

Kept for reference only. They are NOT loaded at runtime: the template manager
resolves recovery templates by path key (recovery/judge, recovery/retry,
recovery/update), so files under recovery/_archive/ are never requested.
