/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * CommandAutocomplete — Shows matching commands when input starts with '/'.
 */

import React from 'react';
import { Box, Paper, Typography, List, ListItemButton, ListItemText } from '@mui/material';

const COMMANDS = [
  { command: '/task', description: 'Run dual-agent task (full workflow)' },
  { command: '/task-plan', description: 'Plan only (no implementation)' },
  { command: '/task-execute', description: 'Execute only (skip planning)' },
  { command: '/task-full', description: 'Full plan + implement workflow' },
  { command: '/task-confirm', description: 'Plan then confirm before implementing' },
  { command: '/model', description: 'Change the LLM model' },
  { command: '/target-path', description: 'Show or set codebase root' },
  { command: '/clear', description: 'Clear conversation history' },
  { command: '/help', description: 'Show available commands' },
  { command: '/kn', description: 'Knowledge management' },
  { command: '/exit', description: 'End session' },
];

export function CommandAutocomplete({ input, onSelect, onClose }) {
  const query = input.toLowerCase();
  const matches = COMMANDS.filter(cmd =>
    cmd.command.toLowerCase().startsWith(query)
  );

  if (matches.length === 0) return null;

  return (
    <Paper
      elevation={4}
      sx={{
        position: 'absolute',
        bottom: '100%',
        left: 0,
        right: 0,
        mb: 0.5,
        maxHeight: 300,
        overflow: 'auto',
        backgroundColor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        zIndex: 10,
      }}
    >
      <List dense disablePadding>
        {matches.map((cmd) => (
          <ListItemButton
            key={cmd.command}
            onClick={() => onSelect(cmd.command)}
            sx={{
              py: 0.75,
              '&:hover': { backgroundColor: 'rgba(74, 144, 217, 0.1)' },
            }}
          >
            <ListItemText
              primary={
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
                  {cmd.command}
                </Typography>
              }
              secondary={
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  {cmd.description}
                </Typography>
              }
            />
          </ListItemButton>
        ))}
      </List>
    </Paper>
  );
}

export default CommandAutocomplete;
