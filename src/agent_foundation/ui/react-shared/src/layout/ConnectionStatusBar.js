/**
 * ConnectionStatusBar — thin status indicator at the top of ManagerChatView.
 *
 * Shows server connection state with color-coded backgrounds:
 *   🟢 Connected   — green tint, shows version + session mode
 *   🟡 Connecting  — yellow tint
 *   🔴 Disconnected — red tint, "Retrying..."
 *
 * Props:
 *   status     — 'connected' | 'connecting' | 'disconnected'
 *   serverInfo — { status, mode, real_sessions, version } | null
 */

import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import CircleIcon from '@mui/icons-material/Circle';

const STATUS_CONFIG = {
  connected: {
    color: '#4caf50',
    bg: 'rgba(76, 175, 80, 0.08)',
    border: 'rgba(76, 175, 80, 0.15)',
  },
  connecting: {
    color: '#ff9800',
    bg: 'rgba(255, 152, 0, 0.08)',
    border: 'rgba(255, 152, 0, 0.15)',
  },
  disconnected: {
    color: '#f44336',
    bg: 'rgba(244, 67, 54, 0.08)',
    border: 'rgba(244, 67, 54, 0.15)',
  },
};

function getStatusText(status, serverInfo) {
  if (status === 'connecting') return 'Connecting...';
  if (status === 'disconnected') return 'Disconnected — Retrying...';

  // Connected — prefer server_name if available
  if (serverInfo?.server_name) return `Connected — ${serverInfo.server_name}`;
  const parts = ['Connected'];
  if (serverInfo?.version) parts.push(`Server v${serverInfo.version}`);
  return parts.join(' — ');
}

export default function ConnectionStatusBar({ status = 'connecting', serverInfo = null }) {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.connecting;

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 0.75,
        px: 2,
        py: 0.5,
        backgroundColor: config.bg,
        borderBottom: `1px solid ${config.border}`,
        flexShrink: 0,
        minHeight: 28,
      }}
    >
      <CircleIcon sx={{ fontSize: 8, color: config.color }} />
      <Typography
        variant="caption"
        sx={{
          color: config.color,
          fontSize: '0.7rem',
          fontWeight: 500,
          letterSpacing: 0.3,
        }}
      >
        {getStatusText(status, serverInfo)}
      </Typography>
    </Box>
  );
}
