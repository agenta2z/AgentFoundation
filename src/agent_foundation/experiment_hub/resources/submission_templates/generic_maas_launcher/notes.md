# MaaS one-shot launcher — usage notes

This template assumes the user's training is launched via
`app-layer main fire-app -d <maas-config>` (rather than a pre-published
recurring flow). Most main_feed_mtml MaaS coldstart / transfer paths.

## Reference command shape

```
cd /data/users/<user>/fbsource/fbcode \
  && app-layer main fire-app -d main_feed_mtml_hstu/maas_coldstart \
       --maas-config <experiment-name>
```

## launch.json shape

```json
{
  "cmd": ["app-layer", "main", "fire-app", "-d",
          "main_feed_mtml_hstu/maas_coldstart",
          "--enable-flags", "${ENABLE_FLAGS}",
          "--experiment-name", "${EXP_NAME}"],
  "cwd": "/data/users/<user>/fbsource/fbcode"
}
```

## Common gotchas

- **MaaS config provider**: overrides MUST be expressed as MaaS config
  YAML edits, NOT as raw kwargs to fire-app. Model code reads the
  resolved config; raw kwargs are silently ignored for non-flagged
  fields.
- **Long-running**: MaaS launches commonly fork a remote training job
  AND a local checkpointing daemon. The launcher should `print`
  `FLOW_URI:` AS SOON AS the remote handle is returned — don't wait
  for the local daemon to finish.
