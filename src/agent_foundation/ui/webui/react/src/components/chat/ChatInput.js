/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Chat input form component.
 */

import React from 'react';
import { Paper, TextField, IconButton } from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';

/**
 * Chat input form with text field and send button
 * @param {object} props
 * @param {string} props.value - Current input value
 * @param {Function} props.onChange - Callback when input changes
 * @param {Function} props.onSubmit - Callback when form is submitted
 * @param {boolean} props.disabled - Whether input is disabled
 */
export function ChatInput({ value, onChange, onSubmit, disabled }) {
  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(e);
  };

  return (
    <Paper
      component="form"
      onSubmit={handleSubmit}
      elevation={0}
      sx={{
        p: 1,
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        backgroundColor: 'background.paper',
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <TextField
        fullWidth
        placeholder="Type your message..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        variant="standard"
        InputProps={{ disableUnderline: true }}
        sx={{ px: 1 }}
      />
      <IconButton
        type="submit"
        color="primary"
        disabled={!value.trim() || disabled}
      >
        <SendIcon />
      </IconButton>
    </Paper>
  );
}

export default ChatInput;
