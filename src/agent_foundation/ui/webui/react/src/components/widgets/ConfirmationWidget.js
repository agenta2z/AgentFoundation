/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * ConfirmationWidget — styled confirmation with primary/secondary actions.
 * Matches the SuggestedActions visual pattern from the demo experiment UI.
 * Includes inline tool parameter configuration via ToolConfigPanel.
 *
 * Layout:
 *   [Prompt text]
 *   [Additional instructions text input — always visible]
 *   [✅ Proceed] [❌ No] [⚙️ Configure (if tool_params)]
 *   [ToolConfigPanel — expandable]
 */

import React, { useState } from 'react';
import {
  Box,
  Button,
  TextField,
  Collapse,
  keyframes,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import ToolConfigPanel from './ToolConfigPanel';

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
`;

export default function ConfirmationWidget({ config, onSubmit }) {
  const theme = useTheme();
  const prompt = config?.input_mode?.prompt || config?.prompt || config?.title || '';
  const yesLabel = config?.yes_label || config?.input_mode?.metadata?.yes_label || '✅ Proceed';
  const noLabel = config?.no_label || config?.input_mode?.metadata?.no_label || '❌ No';
  const toolParams = config?.input_mode?.metadata?.tool_params || config?.tool_params || [];
  const noteVariable = config?.input_mode?.metadata?.note_variable || 'additional_instructions';

  const [showConfig, setShowConfig] = useState(false);
  const [note, setNote] = useState('');
  const [paramValues, setParamValues] = useState({});

  const handleSubmit = (choice) => {
    const hasOverrides = Object.keys(paramValues).some(k => {
      const v = paramValues[k];
      return v !== '' && v !== false && v != null;
    });
    const trimmedNote = note.trim();

    if (hasOverrides || trimmedNote) {
      const response = { choice };
      if (hasOverrides) {
        const overrides = {};
        for (const [k, v] of Object.entries(paramValues)) {
          if (v !== '' && v !== false && v != null) {
            overrides[k] = v;
          }
        }
        response.param_overrides = overrides;
      }
      if (trimmedNote) {
        response.variables = { [noteVariable]: trimmedNote };
      }
      onSubmit(response);
    } else {
      onSubmit(choice);
    }
  };

  return (
    <Box
      sx={{
        backgroundColor: theme.custom.surfaces.highlight,
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'primary.dark',
        p: 2,
        animation: `${fadeIn} 0.3s ease-in-out`,
      }}
    >
      {prompt && (
        <Box sx={{ mb: 2, '& p': { m: 0 } }}>
          <MarkdownRenderer content={prompt} />
        </Box>
      )}

      {/* Additional instructions — always visible */}
      <TextField
        fullWidth
        multiline
        minRows={1}
        maxRows={4}
        placeholder="Additional instructions (optional)..."
        value={note}
        onChange={(e) => setNote(e.target.value)}
        variant="outlined"
        size="small"
        sx={{
          mb: 2,
          '& .MuiOutlinedInput-root': {
            backgroundColor: theme.custom.surfaces.inputBg,
            fontSize: '0.85rem',
            '& fieldset': { borderColor: theme.custom.surfaces.overlayActive },
            '&:hover fieldset': { borderColor: theme.custom.surfaces.inputBorderHover },
            '&.Mui-focused fieldset': { borderColor: 'primary.main' },
          },
          '& .MuiInputBase-input': { color: 'text.primary' },
        }}
      />

      {/* Action buttons */}
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
        <Button
          variant="contained"
          size="medium"
          onClick={() => handleSubmit('yes')}
          sx={{
            textTransform: 'none',
            px: 3,
            py: 1,
            fontSize: '0.9rem',
            fontWeight: 600,
          }}
        >
          {yesLabel}
        </Button>
        <Button
          variant="outlined"
          size="medium"
          onClick={() => handleSubmit('no')}
          sx={{
            textTransform: 'none',
            px: 3,
            py: 1,
            fontSize: '0.9rem',
            fontWeight: 400,
            borderColor: theme.custom.surfaces.inputBorder,
            color: 'text.secondary',
            '&:hover': {
              borderColor: theme.custom.surfaces.inputBorderHover,
              backgroundColor: theme.custom.surfaces.inputBg,
            },
          }}
        >
          {noLabel}
        </Button>
        {toolParams.length > 0 && (
          <Button
            variant="text"
            size="medium"
            onClick={() => setShowConfig(!showConfig)}
            sx={{
              textTransform: 'none',
              px: 2,
              py: 1,
              fontSize: '0.85rem',
              color: showConfig ? 'primary.main' : 'text.secondary',
              ml: 'auto',
            }}
          >
            {showConfig ? '⚙️ Hide Config' : '⚙️ Configure'}
          </Button>
        )}
      </Box>

      {/* Tool parameter configuration panel */}
      <Collapse in={showConfig}>
        <ToolConfigPanel
          params={toolParams}
          values={paramValues}
          onChange={setParamValues}
        />
      </Collapse>
    </Box>
  );
}
