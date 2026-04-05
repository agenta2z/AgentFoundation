/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Suggested actions panel component.
 */

import React, { useState } from 'react';
import { Box, Button, Typography, TextField, IconButton } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { Close as CloseIcon, Send as SendIcon } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';

/**
 * Suggested actions panel with action buttons
 * @param {object} props
 * @param {object} props.suggestedActions - Object with message and actions array
 * @param {Function} props.onAction - Callback when an action is clicked
 * @param {Function} props.onBranchAction - Callback when a branch action is submitted with input
 * @param {boolean} props.disabled - Whether actions are disabled
 */
export function SuggestedActions({ suggestedActions, onAction, onBranchAction, disabled }) {
  // State for branch_with_input expanded action
  const [expandedBranchAction, setExpandedBranchAction] = useState(null);
  const [branchInputValue, setBranchInputValue] = useState('');
  const theme = useTheme();

  if (!suggestedActions?.actions?.length || disabled) {
    return null;
  }

  const handleActionClick = (index) => {
    const action = suggestedActions.actions[index];

    // Handle branch_with_input action type
    if (action?.action_type === 'branch_with_input') {
      setExpandedBranchAction({ index, action });
      setBranchInputValue(action.input_config?.default_value || '');
      return;
    }

    // For all other actions, call onAction directly
    onAction(index);
  };

  const handleBranchSubmit = () => {
    if (expandedBranchAction && branchInputValue.trim()) {
      // Call the branch action handler with the input value
      if (onBranchAction) {
        onBranchAction(expandedBranchAction.index, branchInputValue.trim());
      }
      setExpandedBranchAction(null);
      setBranchInputValue('');
    }
  };

  const handleBranchCancel = () => {
    setExpandedBranchAction(null);
    setBranchInputValue('');
  };

  // If a branch action is expanded, show the input form
  if (expandedBranchAction) {
    const inputConfig = expandedBranchAction.action.input_config || {};
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
          mb: 2,
          mt: 2,
          p: 2,
          backgroundColor: theme.custom.surfaces.activeHighlight,
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'primary.main',
          animation: 'fadeIn 0.3s ease-in-out',
          '@keyframes fadeIn': {
            '0%': { opacity: 0, transform: 'translateY(10px)' },
            '100%': { opacity: 1, transform: 'translateY(0)' },
          },
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, color: 'primary.light' }}>
            🔍 {expandedBranchAction.action.label}
          </Typography>
          <IconButton size="small" onClick={handleBranchCancel} sx={{ color: 'text.secondary' }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>

        <TextField
          fullWidth
          multiline
          minRows={2}
          maxRows={4}
          placeholder={inputConfig.placeholder || 'Enter your input...'}
          value={branchInputValue}
          onChange={(e) => setBranchInputValue(e.target.value)}
          autoFocus
          variant="outlined"
          sx={{
            '& .MuiOutlinedInput-root': {
              backgroundColor: theme.custom.surfaces.sidebarBg,
              borderRadius: 1,
              '& fieldset': { borderColor: theme.custom.surfaces.inputBorder },
              '&:hover fieldset': { borderColor: theme.custom.surfaces.inputBorderHover },
              '&.Mui-focused fieldset': { borderColor: 'primary.main' },
            },
            '& .MuiInputBase-input': { color: 'text.primary', fontSize: '0.95rem' },
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleBranchSubmit();
            }
            if (e.key === 'Escape') {
              handleBranchCancel();
            }
          }}
        />

        <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            onClick={handleBranchCancel}
            size="small"
            sx={{ fontWeight: 400 }}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleBranchSubmit}
            disabled={!branchInputValue.trim()}
            size="small"
            startIcon={<SendIcon />}
            sx={{ fontWeight: 600 }}
          >
            Start Deep Dive
          </Button>
        </Box>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
        mb: 2,
        mt: 2,
        p: 2,
        backgroundColor: theme.custom.surfaces.highlight,
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'primary.dark',
        animation: 'fadeIn 0.3s ease-in-out',
        '@keyframes fadeIn': {
          '0%': { opacity: 0, transform: 'translateY(10px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
      }}
    >
      {suggestedActions.message && (
        <Box sx={{ mb: 1 }}>
          <ReactMarkdown
            components={{
              p: ({ children }) => (
                <Typography
                  variant="body1"
                  sx={{ color: 'text.primary', mb: 0.5 }}
                >
                  {children}
                </Typography>
              ),
            }}
          >
            {suggestedActions.message}
          </ReactMarkdown>
        </Box>
      )}
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        {suggestedActions.actions.map((action, index) => (
          <Button
            key={index}
            variant={action.style === 'primary' ? 'contained' : 'outlined'}
            onClick={() => handleActionClick(index)}
            size="medium"
            sx={{
              fontWeight: action.style === 'primary' ? 600 : 400,
              px: 3,
              py: 1,
            }}
          >
            {action.label}
          </Button>
        ))}
      </Box>
    </Box>
  );
}

export default SuggestedActions;
