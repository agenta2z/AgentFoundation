/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * MultiInputWidget — Tabbed compound widget for collecting multiple inputs.
 * Renders each conversation tool as a tab using existing widget components.
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Tab,
  Tabs,
  Button,
  Typography,
  Badge,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
  Send as SendIcon,
  CheckCircle as CheckIcon,
  FolderOpen as FolderIcon,
  QuestionAnswer as QuestionIcon,
  ListAlt as ListIcon,
} from '@mui/icons-material';
import { getWidget } from './WidgetRegistry';

const TOOL_ICONS = {
  clarification: <QuestionIcon sx={{ fontSize: 16 }} />,
  single_choice: <ListIcon sx={{ fontSize: 16 }} />,
  multiple_choice: <ListIcon sx={{ fontSize: 16 }} />,
  confirmation: <QuestionIcon sx={{ fontSize: 16 }} />,
};

function getToolIcon(toolConfig) {
  if (toolConfig.expected_input_type === 'path') return <FolderIcon sx={{ fontSize: 16 }} />;
  return TOOL_ICONS[toolConfig.tool_type] || <QuestionIcon sx={{ fontSize: 16 }} />;
}

function getToolLabel(toolConfig, index) {
  // Short label from prompt or tool type
  const prompt = toolConfig.prompt || toolConfig.input_mode?.prompt || '';
  if (prompt.length <= 30) return prompt;
  // Extract first meaningful phrase
  const short = prompt.split(/[?.!,]/)[0].trim();
  return short.length <= 30 ? short : `Step ${index + 1}`;
}

export default function MultiInputWidget({ config, onSubmit }) {
  const theme = useTheme();
  const tools = config?.metadata?.tools || config?.tools || [];
  const [activeTab, setActiveTab] = useState(0);
  const [values, setValues] = useState({});

  const handleToolSubmit = useCallback((toolIndex, toolConfig, response) => {
    const varName = toolConfig.output_var || toolConfig.tool_type || `input_${toolIndex}`;

    // Extract value from the widget response
    let value;
    if (response?.content) {
      value = response.content;
    } else if (response?.choice_index !== undefined) {
      const options = toolConfig.input_mode?.options || [];
      value = options[response.choice_index]?.value || String(response.choice_index);
    } else if (response?.custom_text) {
      value = response.custom_text;
    } else if (typeof response === 'string') {
      value = response;
    } else {
      value = JSON.stringify(response);
    }

    setValues(prev => {
      const newValues = { ...prev, [varName]: value };
      if (response?.variable_override) {
        newValues.variable_override = {
          ...(prev.variable_override || {}),
          ...response.variable_override,
        };
      }
      return newValues;
    });

    // Auto-advance to next unfilled tab
    if (toolIndex < tools.length - 1) {
      setActiveTab(toolIndex + 1);
    }
  }, [tools]);

  const filledCount = Object.keys(values).length;
  const allFilled = filledCount >= tools.length;

  const handleSubmitAll = () => {
    onSubmit({ values });
  };

  if (tools.length === 0) return null;

  return (
    <Box>
      {/* Tabs */}
      <Tabs
        value={activeTab}
        onChange={(_, newVal) => setActiveTab(newVal)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          minHeight: 36,
          mb: 2,
          '& .MuiTab-root': {
            minHeight: 36,
            textTransform: 'none',
            fontSize: '0.85rem',
            py: 0.5,
            px: 2,
          },
          '& .MuiTabs-indicator': {
            backgroundColor: 'primary.main',
          },
        }}
      >
        {tools.map((tool, idx) => {
          const varName = tool.output_var || tool.tool_type || `input_${idx}`;
          const isFilled = values[varName] !== undefined;
          return (
            <Tab
              key={idx}
              icon={isFilled ? <CheckIcon sx={{ fontSize: 14, color: 'success.main' }} /> : getToolIcon(tool)}
              iconPosition="start"
              label={getToolLabel(tool, idx)}
              sx={{
                color: isFilled ? 'success.main' : 'text.secondary',
                '&.Mui-selected': { color: 'primary.main' },
              }}
            />
          );
        })}
      </Tabs>

      {/* Active tab content */}
      {tools.map((tool, idx) => {
        if (idx !== activeTab) return null;

        const widgetType = tool.input_mode?.mode || tool.tool_type || 'free_text';
        const WidgetComponent = getWidget(widgetType);

        // Build config for the sub-widget
        const subConfig = {
          ...tool.input_mode,
          input_mode: tool.input_mode,
          metadata: {
            ...tool.input_mode?.metadata,
            expected_input_type: tool.expected_input_type,
            prefix: tool.prefix,
          },
        };

        return (
          <Box key={idx} sx={{ minHeight: 120 }}>
            <WidgetComponent
              config={subConfig}
              onSubmit={(response) => handleToolSubmit(idx, tool, response)}
            />
          </Box>
        );
      })}

      {/* Summary + Submit All */}
      <Box sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        mt: 2,
        pt: 1.5,
        borderTop: `1px solid ${theme.custom.surfaces.overlayActive}`,
      }}>
        <Typography variant="caption" sx={{ color: 'text.disabled' }}>
          {filledCount} of {tools.length} inputs completed
        </Typography>
        <Button
          variant="contained"
          onClick={handleSubmitAll}
          disabled={!allFilled}
          endIcon={<SendIcon sx={{ fontSize: 16 }} />}
          sx={{
            textTransform: 'none',
            px: 3,
            py: 0.75,
            fontSize: '0.9rem',
          }}
        >
          Submit All
        </Button>
      </Box>
    </Box>
  );
}
