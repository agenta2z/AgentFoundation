/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * GroupedWidget — Plan v3: renders 2+ rich conversation widgets stacked
 * vertically as one resolution unit. Submitting any child resolves the
 * GROUP atomically; siblings are flattened/hidden/disabled per their
 * `on_group_resolve` policy on the server side.
 *
 * Backend envelope shape (from conversational_inferencer._handle_rich_group):
 *   config.input_mode.metadata = {
 *     widget_type: "grouped",
 *     group_id: "<gid>",
 *     tools: [
 *       {
 *         child_id: "tool0",
 *         tool_type: "<tt str>",
 *         prompt: "<child prompt>",
 *         input_mode: { mode, prompt, metadata: {...} },
 *         output_var: "...",
 *         metadata: { group_id, on_group_resolve, on_yes_action, hide_no_button, ... },
 *       },
 *       ...
 *     ],
 *   }
 *
 * Outbound submit shape (consumed by message_handlers._handle_pending_input_response):
 *   {
 *     widget_id: <group_widget_id>,
 *     values: { submitted_child: "tool0", payload: <child_response> },
 *   }
 */

import React, { useState } from 'react';
import { Box, Typography, Alert } from '@mui/material';
import { getWidget } from './WidgetRegistry';

// Lightweight "OR" divider between stacked children.
function OrDivider() {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        my: 2,
        gap: 1.5,
      }}
    >
      <Box
        sx={{
          flex: 1,
          height: '1px',
          bgcolor: 'rgba(255, 255, 255, 0.12)',
        }}
      />
      <Typography
        variant="caption"
        sx={{
          color: 'text.secondary',
          fontSize: '0.75rem',
          letterSpacing: '0.08em',
          fontWeight: 500,
        }}
      >
        OR
      </Typography>
      <Box
        sx={{
          flex: 1,
          height: '1px',
          bgcolor: 'rgba(255, 255, 255, 0.12)',
        }}
      />
    </Box>
  );
}

export default function GroupedWidget({ config, onSubmit, onView }) {
  // The grouped envelope lives at config.input_mode.metadata.tools[].
  // Defensive lookups: also accept top-level config.metadata for forward-compat.
  const metadata =
    config?.input_mode?.metadata
    || config?.metadata
    || {};
  const tools = metadata.tools || [];
  const groupId = metadata.group_id || '';

  // Track Hub-creation failure so the widget can show a retry toast
  // without flattening (per Plan v3 Piece 5: failure keeps interactive).
  const [error, setError] = useState(null);

  const handleChildSubmit = (childId, childResponse) => {
    setError(null);
    if (typeof onSubmit !== 'function') {
      console.error(
        '[GroupedWidget] onSubmit is not a function; cannot route child submit',
      );
      return;
    }
    // Wrap the child's response in the group envelope.
    onSubmit({
      submitted_child: childId,
      payload: childResponse,
    });
  };

  // Defensive empty-state: should never happen if backend dispatch is
  // correct, but render a useful diagnostic instead of a blank box.
  if (!Array.isArray(tools) || tools.length === 0) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="warning">
          GroupedWidget received no children (group_id={String(groupId)}).
          This is a backend dispatch error. Check
          conversational_inferencer._handle_rich_group.
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      {tools.map((child, idx) => {
        if (!child || typeof child !== 'object') return null;
        const childId = child.child_id || `tool${idx}`;
        // Determine the React component from the child's widget_type.
        // The child's input_mode.metadata.widget_type is set by the
        // child's handler.build_input_mode (e.g., "proposal_selection",
        // "confirmation"). Fall back to child.tool_type if absent.
        const childWidgetType =
          child?.input_mode?.metadata?.widget_type
          || child?.tool_type
          || child?.input_mode?.mode
          || 'default';
        const ChildComponent = getWidget(childWidgetType);

        // Build per-child config in the shape the existing widget expects.
        // Mirrors the pattern MultiInputWidget uses at lines 161-169 to
        // pass through subConfig — but for rich widgets (no scalar coercion).
        const childConfig = {
          ...child.input_mode,
          input_mode: child.input_mode,
          prompt: child.prompt || child.input_mode?.prompt || '',
          metadata: {
            ...(child.input_mode?.metadata || {}),
            // Forward useful per-child metadata from the conversation tool
            // (on_group_resolve, on_yes_action, hide_no_button) so child
            // widgets that care can adapt (e.g., ConfirmationWidget reads
            // hide_no_button to enter single-action CTA mode).
            ...(child.metadata || {}),
            expected_input_type: child.expected_input_type,
            prefix: child.prefix,
          },
        };

        return (
          <React.Fragment key={childId}>
            {idx > 0 && <OrDivider />}
            <Box
              sx={{
                // Subtle group-member framing — keep visually distinct
                // from the standard standalone widget chrome.
                position: 'relative',
              }}
            >
              <ChildComponent
                config={childConfig}
                onSubmit={(resp) => handleChildSubmit(childId, resp)}
                onView={onView}
              />
            </Box>
          </React.Fragment>
        );
      })}
      {error && (
        <Alert
          severity="error"
          onClose={() => setError(null)}
          sx={{ mt: 2 }}
        >
          {error}
        </Alert>
      )}
    </Box>
  );
}
