# Color Migration Map

Mapping decisions for hardcoded colors → theme palette tokens.

## Inconsistent Hardcoded Colors

| Hardcoded Value | Location | Maps To | Rationale |
|---|---|---|---|
| `#2196f3` | AgentStreamDrawer (assistant role border) | `palette.info.main` | Closest MUI blue; distinct from `primary.main` (#4a90d9) |
| `#90caf9` | MarkdownRenderer (link color) | `palette.primary.light` | Light blue accent; aligns with `primary.light` (#7bb3eb) |
| `#81c784` | AgentStreamSection (complete chip) | `palette.success.light` | Light green; matches `success.light` (#81c784) exactly |

## Chat Role Border Colors

| Role | Hardcoded Value | Maps To |
|---|---|---|
| user | `#4caf50` | `palette.success.main` |
| assistant | `#2196f3` | `palette.info.main` |
| system | `#ff9800` | `palette.warning.main` |

## PRIORITY_COLORS — Intentional Visual Changes

| Priority | Before | After | Note |
|---|---|---|---|
| critical | red | `palette.error.main` | No visual change |
| high | orange | `palette.warning.dark` | No visual change |
| medium | blue | `palette.warning.main` | ⚠️ Intentional: blue → amber/orange |
| low | gray | `palette.success.main` | ⚠️ Intentional: gray → green |

These PRIORITY_COLORS changes in SprintBoardView are intentional — they normalize priorities to a severity-based palette scale (error → warning → success) rather than arbitrary colors.
