# @agent-foundation/shared-ui

Shared React UI components, hooks, and theme system for AgentFoundation-based apps.

Despite the directory name `react-shared/`, this package contains more than widgets — it includes theme, hooks, layout, progress, and chat primitives. The package name `@agent-foundation/shared-ui` reflects this broader scope.

## Quick start

```json
// package.json (same repo)
"@agent-foundation/shared-ui": "file:../../react-shared"

// package.json (cross-repo dev)
"@agent-foundation/shared-ui": "file:../../../../AgentFoundation/src/agent_foundation/ui/react-shared"
```

## Usage

```js
import { MarkdownRenderer, LoadingIndicator, registerWidget, getWidget } from '@agent-foundation/shared-ui';
import { AppThemeProvider, ThemeSwitcher } from '@agent-foundation/shared-ui/theme';

// Register domain-specific widgets
registerWidget('myapp.custom_widget', MyCustomWidget);
```

## MUI 5/7 cross-version support

The `ThemeProvider` accepts a `createThemeFn` prop — pass your MUI version's `createTheme`:

```js
import { createTheme } from '@mui/material/styles'; // MUI 5 or 7
<AppThemeProvider createThemeFn={createTheme}>
  <App />
</AppThemeProvider>
```

## ChatMessage vs AgentMessageBubble

- **ChatMessage** — simple bubble for user/system text (58 lines, any role)
- **AgentMessageBubble** — rich agent renderer with thinking/response phases, session context, prompt viewer (227 lines)

They coexist. Use ChatMessage for simple messages, AgentMessageBubble for agent turns with phases.

## Build

```bash
npm run build    # produces dist/cjs + dist/esm
npm run sync     # regenerate widgetTypes.js from Python
npm run test     # vitest
```
