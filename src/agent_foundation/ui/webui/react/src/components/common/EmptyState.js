/**
 * EmptyState — no-data placeholder with icon and message.
 *
 * Shows a centered message when a list or view has no content.
 *
 * Props:
 *   message - string
 *   icon    - React node (optional)
 */

import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import InboxIcon from '@mui/icons-material/Inbox';

export function EmptyState({ message = 'No data available', icon }) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        py: 6,
        gap: 1.5,
        color: 'text.secondary',
      }}
    >
      <Box sx={{ fontSize: 48, opacity: 0.4 }}>
        {icon || <InboxIcon sx={{ fontSize: 48 }} />}
      </Box>
      <Typography variant="body1" sx={{ opacity: 0.6 }}>
        {message}
      </Typography>
    </Box>
  );
}

export default EmptyState;
