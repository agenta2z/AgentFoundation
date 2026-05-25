/**
 * TextInputWidget — free-text input for clarification questions.
 * Adapted from AgentFoundation/ui/webui/react/src/components/widgets/TextInputWidget.js
 *
 * Props:
 *   config.input_mode.prompt - string (question text)
 *   config.placeholder       - string (optional)
 *   onSubmit({content})      - called with the user's typed text
 */

import React, { useState } from 'react';
import { Box, Button, TextField, Typography } from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { MarkdownRenderer } from '../common/MarkdownRenderer';

export default function TextInputWidget({ config, onSubmit }) {
  const theme = useTheme();
  const prompt = config?.input_mode?.prompt || config?.prompt || '';
  const placeholder = config?.placeholder || 'Type your response...';
  const [text, setText] = useState('');
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
        autoFocus
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
          Submit
        </Button>
      </Box>
    </Box>
  );
}
