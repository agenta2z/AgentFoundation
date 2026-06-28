/**
 * TextInputWidget — free-text input for clarification questions.
 * Adapted from AgentFoundation/ui/webui/react/src/components/widgets/TextInputWidget.js
 *
 * Props:
 *   config.input_mode.prompt - string (question text)
 *   config.placeholder       - string (optional)
 *   onSubmit({content})      - called with the user's typed text
 *
 * Compatibility fallback: if the input mode declares expected_input_type === "path"
 * without a widget_type routing it to PathInputWidget, this widget delegates to the
 * same path widget (see conditional_path_inputs_plan.md §D2). Free-text behavior is
 * otherwise byte-identical to before.
 */

import React, { useState } from 'react';
import { Box, Button, TextField, Typography } from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import PathInputWidget from './PathInputWidget';

export default function TextInputWidget({ config, onSubmit, submitLabel, readOnly, value }) {
  const theme = useTheme();

  // Compatibility fallback: typed path input that did not get a widget_type.
  if (config?.input_mode?.expected_input_type === 'path') {
    return <PathInputWidget config={config} onSubmit={onSubmit} submitLabel={submitLabel} readOnly={readOnly} value={value} />;
  }

  const prompt = config?.input_mode?.prompt || config?.prompt || '';
  const placeholder = config?.placeholder || 'Type your response...';
  // Prefill: a default the agent supplied (e.g. text the user already gave).
  const rawDefault = config?.input_mode?.default ?? config?.input_mode?.metadata?.default;
  const [text, setText] = useState(
    readOnly && value && typeof value.content === 'string'
      ? value.content
      : (typeof rawDefault === 'string' ? rawDefault : ''),
  );
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = () => {
    if (submitted) return; // double-submit guard
    if (!text.trim()) return;
    setSubmitted(true);
    onSubmit({ content: text.trim() });
  };

  // Read-only submitted view
  if (submitted) {
    return (
      <Box sx={{ color: 'text.secondary', fontStyle: 'italic', fontSize: '0.9rem', py: 0.5 }}>
        "{text.trim()}"
      </Box>
    );
  }

  return (
    <Box>
      {prompt && (
        <Box sx={{ mb: 1.5, '& p': { m: 0 } }}>
          <MarkdownRenderer content={prompt} />
        </Box>
      )}
      <TextField
        fullWidth
        multiline
        minRows={2}
        maxRows={6}
        placeholder={placeholder}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
          }
        }}
        autoFocus={!readOnly}
        disabled={readOnly}
        variant="outlined"
        sx={{
          '& .MuiOutlinedInput-root': {
            backgroundColor: theme.custom?.surfaces?.inputBg || 'rgba(0,0,0,0.2)',
            '& fieldset': { borderColor: theme.custom?.surfaces?.inputBorder || 'rgba(255,255,255,0.2)' },
            '&:hover fieldset': { borderColor: theme.custom?.surfaces?.inputBorderHover || 'rgba(255,255,255,0.35)' },
            '&.Mui-focused fieldset': { borderColor: 'primary.main' },
          },
          '& .MuiInputBase-input': { color: 'text.primary', fontSize: '0.95rem' },
        }}
      />
      {!readOnly && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
          <Typography variant="caption" sx={{ color: 'text.disabled' }}>
            Enter to submit · Shift+Enter for new line
          </Typography>
          <Button
            variant="contained"
            size="small"
            onClick={handleSubmit}
            disabled={!text.trim()}
            endIcon={<SendIcon sx={{ fontSize: 16 }} />}
            sx={{ textTransform: 'none', px: 2, fontSize: '0.85rem' }}
          >
            {submitLabel || 'Submit'}
          </Button>
        </Box>
      )}
    </Box>
  );
}
