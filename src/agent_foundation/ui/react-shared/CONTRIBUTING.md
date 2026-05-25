# Contributing to @agent-foundation/shared-ui

## Rules

1. **New widget types** — add a constant to `widget_protocol.py` first, then regenerate JS via `npm run sync`.
2. **Regenerate** — every PR that touches `widget_protocol.py` or `input_modes.py` MUST run `npm run sync` and commit the generated files.
3. **Register** — add the widget to `src/protocol/registerBuiltins.js`.
4. **Test** — every new component MUST have a test.
5. **Props** — all input widgets MUST conform to Contract 2: `{ widget, onSubmit, onCancel?, onView?, onViewFolder?, disabled? }`.
6. **Domain widgets** — MUST use `namespace.type` naming and live in the consuming app, not here. Register via `registerWidget('openteam.sprint_progress', Component)`.

## Directory layout

- `src/common/` — pure UI primitives (no domain coupling)
- `src/theme/` — MUI-version-agnostic theme system
- `src/inputs/` — one file per canonical widget type
- `src/protocol/` — WidgetRegistry, codegen output, dispatchers
- `src/chat/` — chat/streaming primitives
- `src/layout/` — generic chrome (AppHeader, FileViewer, etc.)
- `src/progress/` — progress UI primitives
- `src/hooks/` — generic React hooks (no app state)

## Promotion rubric

A component is a promotion candidate if and only if:
1. Takes only structural props (string, number, array, callback), not domain objects
2. Has no imports from app-specific modules
3. Renders identically across codebases (modulo theming)
4. Maps to a stable concept in the agent-interaction domain
5. Can have a self-contained test
