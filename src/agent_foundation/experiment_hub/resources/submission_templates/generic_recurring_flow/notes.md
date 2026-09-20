# Recurring flow — usage notes

This template assumes the user's team owns an FBLearner recurring flow
scheduled via `fbpkg build` + `fblearner_project_push`. The generated
script substitutes the hypothesis-flag mapping into the real flow's
config provider before calling `dispatch()`.

## Reference command shape

```
cd /data/users/<user>/fbsource/fbcode \
  && fbpkg build my_team.recurring \
  && fblearner_project_push my_team --tag <experiment-name>
```

## launch.json shape (PTI must emit this alongside submit_v1.py)

```json
{
  "cmd": ["buck", "run", "fbcode//my_team/recurring:run", "--",
          "--enable-flags", "${ENABLE_FLAGS}",
          "--experiment-name", "${EXP_NAME}"],
  "cwd": "/data/users/<user>/fbsource/fbcode"
}
```

## Common gotchas

- **Config-provider injection**: overrides MUST be applied via the
  team's standard config provider (e.g., `MaaSConfigProvider.with_overrides`),
  NOT by mutating the dataclass directly — production rollout depends on
  the override machinery's audit trail.
- **GRANDTETON 8GPU default**: confirm the flow's hardware profile is set
  before dispatch. The default in some teams is 4GPU which silently
  halves the throughput.
