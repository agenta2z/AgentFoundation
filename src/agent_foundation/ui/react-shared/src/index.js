/**
 * @agent-foundation/shared-ui — public barrel.
 *
 * Auto-registers all built-in widgets on first import.
 */

// Auto-register built-in widgets
import './protocol/registerBuiltins';

// Protocol / registry
export { registerWidget, getWidget, unregisterWidget, listRegisteredWidgets } from './protocol/WidgetRegistry';

// Common primitives
export { default as EmptyState } from './common/EmptyState';
export { default as LoadingIndicator } from './common/LoadingIndicator';
export { default as PersonChip } from './common/PersonChip';
export { default as ProgressBar } from './common/ProgressBar';
export { default as QuickLinkButton } from './common/QuickLinkButton';
export { default as SectionCard } from './common/SectionCard';
export { default as StatusBadge } from './common/StatusBadge';
export { default as WelcomeScreen } from './common/WelcomeScreen';
export { default as PlanModeSelector } from './common/PlanModeSelector';
export { default as ClickToEditMarkdown } from './common/ClickToEditMarkdown';
export { default as MarkdownRenderer } from './common/MarkdownRenderer';
export { default as SplitActionButton } from './common/SplitActionButton';
export { default as PendingReasonPopover } from './common/PendingReasonPopover';

// Theme
export { AppThemeProvider, useAppTheme } from './theme/ThemeProvider';
export { default as ThemeSwitcher } from './theme/ThemeSwitcher';
export { getTheme, listThemes, registerTheme } from './theme/themeRegistry';
export { default as createAppTheme } from './theme/createAppTheme';

// Input widgets
export { default as TextInputWidget } from './inputs/TextInputWidget';
export { default as SingleChoiceWidget } from './inputs/SingleChoiceWidget';
export { default as MultipleChoiceWidget } from './inputs/MultipleChoiceWidget';
export { default as DropdownWidget } from './inputs/DropdownWidget';
export { default as ToggleWidget } from './inputs/ToggleWidget';
export { default as ConfirmationWidget } from './inputs/ConfirmationWidget';
export { default as ToolArgumentFormWidget } from './inputs/ToolArgumentFormWidget';
export { default as MultiInputWidget } from './inputs/MultiInputWidget';
export { default as GroupedWidget } from './inputs/GroupedWidget';
export { default as DefaultWidget } from './inputs/DefaultWidget';

// Chat
export { default as ChatInput } from './chat/ChatInput';
export { default as ChatMessage } from './chat/ChatMessage';
export { default as StreamingMessage } from './chat/StreamingMessage';
export { default as CommandAutocomplete } from './chat/CommandAutocomplete';

// Layout
export { default as AppHeader } from './layout/AppHeader';
export { default as FileViewer } from './layout/FileViewer';

// Progress
export { default as ProgressSection } from './progress/ProgressSection';
export { default as CompletedSection } from './progress/CompletedSection';
export { default as TaskProgressBar } from './progress/TaskProgressBar';
export { default as TaskProgressPanel } from './progress/TaskProgressPanel';
