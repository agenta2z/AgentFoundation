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
 *   mode "free_text" + metadata.compound true → CompoundWidget (multiple tools)
 */

import React from 'react';
import { Box, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import { getWidget } from './WidgetRegistry';
import { parseResponseTags, stripSessionContext, stripAnsi, stripAcliNoise, stripToolsToInvoke } from '../chat/ThinkingFold';

/**
 * Compound widget: renders multiple sub-tools sequentially in one widget.
 * Collects all responses and submits as a JSON-encoded dict.
 */
function CompoundWidget({ tools, preamble, onSubmit, onView, onViewFolder }) {
  const [step, setStep] = React.useState(0);
  const [responses, setResponses] = React.useState({});

  const currentTool = tools[step];
  const isLast = step === tools.length - 1;

  const handleStepSubmit = (value) => {
    const outputVar = currentTool.output_var || `step_${step}`;
    const newResponses = { ...responses, [outputVar]: value };
    setResponses(newResponses);

    if (isLast) {
      // All steps done — submit combined response
      onSubmit(JSON.stringify(newResponses));
    } else {
      setStep(step + 1);
    }
  };

  return (
    <Box>
      {preamble && (
        <Box sx={{ mb: 1.5, '& p': { m: 0 } }}>
          <MarkdownRenderer content={preamble} />
        </Box>
      )}
      {tools.length > 1 && (
        <Typography variant="caption" sx={{ color: 'text.disabled', mb: 1, display: 'block' }}>
          Step {step + 1} of {tools.length}
        </Typography>
      )}
      <StepWidget tool={currentTool} onSubmit={handleStepSubmit} onView={onView} onViewFolder={onViewFolder} />
    </Box>
  );
}

function StepWidget({ tool, onSubmit, onView, onViewFolder }) {
  const config = { input_mode: tool.input_mode || {}, prompt: tool.prompt };
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
export default function ConversationToolWidget({ pendingInput, onSubmit, onView, onViewFolder }) {
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

  // Build the config object that each sub-widget expects
  const config = { input_mode: inputMode, prompt: inputMode.prompt || displayContent };

  // Determine which widget to render
  let preamble = null;

  if (metadata.compound && metadata.tools?.length > 0) {
    // Multiple conversation tools in one message — compound widget handles sequencing
    return (
      <WidgetContainer>
        <CompoundWidget
          tools={metadata.tools}
          preamble={displayContent}
          onSubmit={onSubmit}
          onView={onView}
          onViewFolder={onViewFolder}
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
