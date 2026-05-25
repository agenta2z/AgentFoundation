/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Single chat message component.
 */

import React from 'react';
import { Box, Paper, Button } from '@mui/material';
import { Description as FileIcon } from '@mui/icons-material';
import { MarkdownRenderer } from '../common/MarkdownRenderer';

/**
 * Renders a single chat message bubble
 * @param {object} props
 * @param {object} props.message - Message object with role, content, file_path
 * @param {Function} props.onOpenFile - Callback when file link is clicked
 */
export function ChatMessage({ message, onOpenFile }) {
  const isUser = message.role === 'user';

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
      }}
    >
      <Paper
        elevation={0}
        sx={{
          p: 2,
          maxWidth: '80%',
          backgroundColor: isUser ? 'primary.dark' : 'background.paper',
          borderRadius: 2,
          border: '1px solid',
          borderColor: isUser ? 'primary.main' : 'divider',
        }}
      >
        <Box sx={{ '& p': { m: 0 }, '& pre': { overflow: 'auto' } }}>
          <MarkdownRenderer content={message.content} />
        </Box>
        {message.file_path && (
          <Button
            size="small"
            startIcon={<FileIcon />}
            onClick={() => onOpenFile(message.file_path)}
            sx={{ mt: 1 }}
          >
            📄 View File
          </Button>
        )}
      </Paper>
    </Box>
  );
}

export default ChatMessage;
