/**
 * ConfirmationWidget — styled confirmation with Proceed / No buttons.
 * Adapted from AgentFoundation/ui/webui/react/src/components/widgets/ConfirmationWidget.js
 *
 * Props:
 *   config.input_mode.prompt       - string (question text, rendered as markdown)
 *   config.input_mode.metadata     - { yes_label, no_label, note_variable, tool_params }
 *   onSubmit(response)             - called with 'yes', 'no', or {choice, variables, param_overrides}
 */

import React, { useState } from 'react';
import { Box, Button, TextField, keyframes } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { MarkdownRenderer } from '../common/MarkdownRenderer';

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
`;

export default function ConfirmationWidget({ config, onSubmit, onView, onViewFolder }) {
  const theme = useTheme();
  const prompt = config?.input_mode?.prompt || config?.prompt || '';
  const yesLabel = config?.input_mode?.metadata?.yes_label || 'Proceed';
  const noLabel = config?.input_mode?.metadata?.no_label || 'No';
  const noteVariable = config?.input_mode?.metadata?.note_variable || 'additional_instructions';
  const viewPath = config?.input_mode?.metadata?.view || null;
  const viewLabel = config?.input_mode?.metadata?.view_label || 'View Document';
  const viewType = config?.input_mode?.metadata?.view_type || 'file';

  const [note, setNote] = useState('');
  const [submitted, setSubmitted] = useState(null); // 'yes' | 'no' | null

  const handleSubmit = (choice) => {
    if (submitted) return; // double-submit guard (matches SingleChoice/MultipleChoice pattern)
    setSubmitted(choice);
    const trimmedNote = note.trim();
    if (trimmedNote) {
      onSubmit({ choice, variables: { [noteVariable]: trimmedNote } });
    } else {
      onSubmit(choice);
    }
  };

  return (
    <Box
      sx={{
        backgroundColor: theme.custom?.surfaces?.highlight || 'rgba(74,144,217,0.08)',
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'primary.dark',
        p: 2,
        animation: `${fadeIn} 0.25s ease-in-out`,
      }}
    >
      {prompt && (
        <Box sx={{ mb: 1.5, '& p': { m: 0 } }}>
          <MarkdownRenderer content={prompt} />
        </Box>
      )}

      {submitted ? (
        <Box sx={{ color: submitted === 'yes' ? 'success.main' : 'text.secondary', fontWeight: 500, fontSize: '0.9rem' }}>
          {submitted === 'yes' ? '✅ Confirmed — proceeding' : '❌ Declined'}
        </Box>
      ) : (
        <>
          {/* Optional additional instructions */}
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
              mb: 1.5,
              '& .MuiOutlinedInput-root': {
                backgroundColor: theme.custom?.surfaces?.inputBg || 'rgba(0,0,0,0.2)',
                fontSize: '0.85rem',
                '& fieldset': { borderColor: theme.custom?.surfaces?.overlayActive || 'rgba(255,255,255,0.15)' },
                '&:hover fieldset': { borderColor: theme.custom?.surfaces?.inputBorderHover || 'rgba(255,255,255,0.3)' },
                '&.Mui-focused fieldset': { borderColor: 'primary.main' },
              },
              '& .MuiInputBase-input': { color: 'text.primary' },
            }}
          />

          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
            {viewPath && (onView || onViewFolder) && (
              <Button
                variant="outlined"
                size="medium"
                onClick={() => viewType === 'folder' && onViewFolder ? onViewFolder(viewPath) : onView(viewPath)}
                sx={{
                  textTransform: 'none',
                  px: 2,
                  borderColor: theme.custom?.surfaces?.overlayActive || 'rgba(74,144,217,0.4)',
                  color: 'primary.main',
                  '&:hover': {
                    borderColor: 'primary.main',
                    backgroundColor: 'rgba(74,144,217,0.08)',
                  },
                }}
              >
                📄 {viewLabel}
              </Button>
            )}
            <Button
              variant="contained"
              size="medium"
              onClick={() => handleSubmit('yes')}
              sx={{ textTransform: 'none', px: 3, fontWeight: 600 }}
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
                borderColor: theme.custom?.surfaces?.inputBorder || 'rgba(255,255,255,0.2)',
                color: 'text.secondary',
                '&:hover': {
                  borderColor: theme.custom?.surfaces?.inputBorderHover || 'rgba(255,255,255,0.4)',
                  backgroundColor: theme.custom?.surfaces?.inputBg || 'rgba(0,0,0,0.1)',
                },
              }}
            >
              {noLabel}
            </Button>
          </Box>
        </>
      )}
    </Box>
  );
}
