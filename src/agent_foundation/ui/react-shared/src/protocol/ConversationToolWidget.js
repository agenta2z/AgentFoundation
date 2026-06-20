/**
 * ConversationToolWidget — dispatcher for pending_input conversation tools.
 *
 * Receives a pendingInput object from useManagerChat when the ConversationalInferencer
 * needs user input (clarification, confirmation, single_choice, multiple_choices, compound).
 *
 * Data flow:
 *   Server sends: {"type": "pending_input", "content": "...", "input_mode": {...}}
 *   useManagerChat sets: pendingInput = { content, inputMode }
 *   This component dispatches to the right sub-widget based on inputMode.mode + metadata
 *   User interacts → onSubmit(response) → sendPendingInputResponse → WS → server
 *
 * AF input_mode protocol (from agent_foundation/ui/input_modes.py):
 *   mode "free_text" + metadata.widget_type "confirmation" → ConfirmationWidget
 *   mode "single_choice"   → SingleChoiceWidget
 *   mode "multiple_choices" → MultipleChoiceWidget
 *   mode "free_text" (default) → TextInputWidget (clarification)
 *   metadata.compound true (+ metadata.tools) → CompoundWidget (one tab per tool,
 *       all panels mounted; child submits collected locally; "Submit All" commits the
 *       direct-map JSON payload). The compound path always uses CompoundWidget and does
 *       not honor metadata.widget_type (e.g. "multi_input").
 */

import React from 'react';
import { Box, Button, Tab, Tabs } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { getWidget } from './WidgetRegistry';
import { parseResponseTags, stripSessionContext, stripAnsi, stripAcliNoise, stripToolsToInvoke } from '../chat/ThinkingFold';

/**
 * Compound widget: renders multiple sub-tools as parallel tabs in one widget.
 *
 * Each tool is a tab; ALL tab panels stay mounted (inactive ones CSS-hidden via
 * display:none) so each leaf widget keeps its local draft state across tab switches.
 *
 * Lossless collection contract (unchanged): a child tab's submit saves the raw child
 * response into responses[output_var] LOCALLY ONLY — no network send. A single
 * "Submit All" button (enabled once every required tab is complete) emits the
 * existing direct-map payload onSubmit(JSON.stringify(responses)).
 *
 * A single-tool compound renders without the tab bar (no regression).
 */
function CompoundWidget({ tools, onSubmit, onView, onViewFolder, pathAutocompleteProvider }) {
  const [activeTab, setActiveTab] = React.useState(0);
  const [responses, setResponses] = React.useState({});
  // Track per-tool completion by output_var so tab labels + Submit All can reflect it.
  const [completed, setCompleted] = React.useState({});

  const outputVarFor = (tool, index) => tool.output_var || `step_${index}`;

  // Child submit saves locally only — no onSubmit/network until "Submit All".
  const handleChildSubmit = (tool, index) => (rawChildResponse) => {
    const outputVar = outputVarFor(tool, index);
    setResponses((prev) => ({ ...prev, [outputVar]: rawChildResponse }));
    setCompleted((prev) => ({ ...prev, [outputVar]: true }));
  };

  const allComplete = tools.every((tool, index) => completed[outputVarFor(tool, index)]);

  const handleSubmitAll = () => {
    if (!allComplete) return;
    onSubmit(JSON.stringify(responses));
  };

  const multiTab = tools.length > 1;

  return (
    <Box>
      {multiTab && (
        <Tabs
          value={activeTab}
          onChange={(_e, next) => setActiveTab(next)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ mb: 1.5, minHeight: 0 }}
        >
          {tools.map((tool, index) => {
            const done = completed[outputVarFor(tool, index)];
            const label = tool.title || tool.prompt || outputVarFor(tool, index);
            return (
              <Tab
                key={outputVarFor(tool, index)}
                label={done ? `✓ ${label}` : label}
                sx={{ textTransform: 'none', minHeight: 0 }}
              />
            );
          })}
        </Tabs>
      )}

      {/* All panels mounted; inactive ones CSS-hidden so leaf draft state survives. */}
      {tools.map((tool, index) => (
        <Box
          key={outputVarFor(tool, index)}
          role="tabpanel"
          hidden={multiTab && activeTab !== index}
          sx={{ display: multiTab && activeTab !== index ? 'none' : 'block' }}
        >
          <StepWidget
            tool={tool}
            onSubmit={handleChildSubmit(tool, index)}
            onView={onView}
            onViewFolder={onViewFolder}
            pathAutocompleteProvider={pathAutocompleteProvider}
          />
        </Box>
      ))}

      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
        <Button variant="contained" disabled={!allComplete} onClick={handleSubmitAll}>
          Submit All
        </Button>
      </Box>
    </Box>
  );
}

function StepWidget({ tool, onSubmit, onView, onViewFolder, pathAutocompleteProvider }) {
  const config = { input_mode: tool.input_mode || {}, prompt: tool.prompt, pathAutocompleteProvider };
  const mode = tool.input_mode?.mode || 'free_text';
  const widgetType = tool.input_mode?.metadata?.widget_type
    || (tool.tool_type === 'confirmation' ? 'confirmation' : null)
    || mode;
  const Widget = getWidget(widgetType);
  return <Widget config={config} onSubmit={onSubmit} onView={onView} onViewFolder={onViewFolder} />;
}

/**
 * Main dispatcher — maps pendingInput.inputMode to the correct widget.
 */
export default function ConversationToolWidget({ pendingInput, onSubmit, onView, onViewFolder, pathAutocompleteProvider }) {
  const theme = useTheme();

  if (!pendingInput) return null;

  const inputMode = pendingInput.inputMode || {};
  const mode = inputMode.mode || 'free_text';
  const metadata = inputMode.metadata || {};

  // Clean the content: strip <Response> tags, thinking, ACLI noise, etc.
  let displayContent = pendingInput.content || '';
  const contentParsed = parseResponseTags(displayContent);
  if (contentParsed.phase === 'pre_response') {
    // Only thinking content — no user-visible text
    displayContent = '';
  } else if (contentParsed.phase !== 'no_tags') {
    displayContent = contentParsed.responseContent || displayContent;
  }
  displayContent = stripSessionContext(stripAnsi(stripAcliNoise(stripToolsToInvoke(displayContent)))).trim();

  console.debug('[ConversationToolWidget] mode:', mode, '| widget_type:', metadata.widget_type, '| displayContent len:', displayContent.length);

  // Build the config object that each sub-widget expects. The host-injected path
  // autocomplete provider is threaded through config so leaf widgets stay filesystem-free.
  const config = { input_mode: inputMode, prompt: inputMode.prompt || displayContent, pathAutocompleteProvider };

  // Determine which widget to render
  let preamble = null;

  if (metadata.compound && metadata.tools?.length > 0) {
    // Multiple conversation tools in one message — the compound path ALWAYS uses
    // CompoundWidget (tabbed). metadata.widget_type (e.g. "multi_input") is NOT honored
    // here: the round's preamble is its own committed bubble, so we do not pass/render
    // preamble text inside the widget.
    return (
      <WidgetContainer>
        <CompoundWidget
          tools={metadata.tools}
          onSubmit={onSubmit}
          onView={onView}
          onViewFolder={onViewFolder}
          pathAutocompleteProvider={pathAutocompleteProvider}
        />
      </WidgetContainer>
    );
  }

  // Registry lookup: prefer metadata.widget_type, fall back to mode
  const widgetType = metadata.widget_type || mode;
  const Widget = getWidget(widgetType);

  // For free_text / clarification: use AI preamble content as prompt if no explicit prompt set
  if (widgetType === 'free_text' || (!metadata.widget_type && mode === 'free_text')) {
    if (!inputMode.prompt && pendingInput.content) {
      config.prompt = pendingInput.content;
    }
  }

  return (
    <WidgetContainer>
      {/* Preamble: the AI's text before the tool invocation (already shown above,
          but kept here for cases where it's short and helpful inline) */}
      <Widget config={config} onSubmit={onSubmit} onView={onView} onViewFolder={onViewFolder} />
    </WidgetContainer>
  );
}

function WidgetContainer({ children }) {
  const theme = useTheme();
  return (
    <Box
      sx={{
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'primary.dark',
        backgroundColor: theme.custom?.surfaces?.highlight || 'rgba(74,144,217,0.06)',
        p: 2,
      }}
    >
      {children}
    </Box>
  );
}
