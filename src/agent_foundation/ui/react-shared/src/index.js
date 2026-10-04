/**
 * @agent-foundation/shared-ui — public barrel.
 *
 * Auto-registers all built-in widgets on first import.
 */

// Auto-register built-in widgets
import './protocol/registerBuiltins';

// Protocol / registry
export { registerWidget, getWidget, unregisterWidget, listRegisteredWidgets } from './protocol/WidgetRegistry';
export { default as ConversationToolWidget } from './protocol/ConversationToolWidget';
export { default as ChatWidgetRenderer } from './protocol/ChatWidgetRenderer';

// Display widgets — registry-rendered inline in chat by ChatWidgetRenderer.
export { default as ApprovalWidget } from './protocol/ApprovalWidget';
export { default as ChoiceWidget } from './protocol/ChoiceWidget';
export { default as ProjectSummaryWidget } from './protocol/ProjectSummaryWidget';
export { default as SprintProgressWidget } from './protocol/SprintProgressWidget';
export { default as TaskAssignmentWidget } from './protocol/TaskAssignmentWidget';
export { default as TaskListWidget } from './protocol/TaskListWidget';
export { default as WorkloadChartWidget } from './protocol/WorkloadChartWidget';

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
export { ThemeSwitcher } from './theme/ThemeSwitcher';
export { getTheme, listThemes, registerTheme } from './theme/themeRegistry';
export { createAppTheme } from './theme/createAppTheme';

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
export { default as PathAutocompleteInput } from './inputs/PathAutocompleteInput';
export { default as MultiValueInput } from './inputs/MultiValueInput';
export { default as PathInputWidget } from './inputs/PathInputWidget';
export { default as DefaultWidget } from './inputs/DefaultWidget';
export { default as ProposalSelectionWidget } from './inputs/ProposalSelectionWidget';

// Chat
export { default as ChatInput } from './chat/ChatInput';
export { default as ChatMessage } from './chat/ChatMessage';
export { default as StreamingMessage } from './chat/StreamingMessage';
export { default as CommandAutocomplete } from './chat/CommandAutocomplete';
export { default as AgentMessageBubble } from './chat/AgentMessageBubble';
export { default as Breadcrumb } from './chat/Breadcrumb';
export { default as GraphFlowView } from './chat/GraphFlowView';
export { default as NodeDetailPanel } from './chat/NodeDetailPanel';
export { default as PromptViewerDrawer } from './chat/PromptViewerDrawer';
export { default as ThinkingFold, parseResponseTags, stripSessionContext, stripAnsi, stripAcliNoise, stripToolsToInvoke, stripResponseTags, parseSessionContext } from './chat/ThinkingFold';

// Layout
export { default as AppHeader } from './layout/AppHeader';
export { default as FileViewer } from './layout/FileViewer';

// Hooks
export { default as usePathComplete } from './hooks/usePathComplete';

// Progress
export { default as ProgressSection } from './progress/ProgressSection';
export { default as CompletedSection } from './progress/CompletedSection';
export { default as TaskProgressBar } from './progress/TaskProgressBar';
export { default as TaskProgressPanel } from './progress/TaskProgressPanel';

// Secondary tab bar (used by dashboards + multi-view panels)
export { default as ViewTabBar } from './nav/ViewTabBar';

// Generic Dashboard framework (DashboardPanel + view/dashboard registries + WidgetHostView)
export {
  registerView, getView, listRegisteredViews, unregisterView,
  registerDashboard, getDashboard, listRegisteredDashboards, createDashboardReducer,
  registerManifest, getManifest, normalizeManifest,
  evaluateViewRules, getPipelineStages, getDashboardStatus, deriveStatusMap,
  PipelineStatusBar, WidgetHostView, DashboardPanel,
} from './dashboard';

// Concrete dashboards — side-effect import registers the Experiment Hub's
// views + dashboard reducer on first load (must come after the framework
// export above so the registries exist).
import './dashboards/experiment_hub';
export { default as ExperimentHubDashboard } from './dashboards/experiment_hub';
