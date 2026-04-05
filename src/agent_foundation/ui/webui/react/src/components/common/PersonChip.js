/**
 * PersonChip — avatar + name + role chip for humans and AI agents.
 *
 * Shows a colored avatar with first letter (or robot icon for AI),
 * the person's name, and optionally their role.
 *
 * Props:
 *   name      - string
 *   role      - string (optional, shown as subtitle)
 *   type      - "human" | "ai"
 *   avatarUrl - optional image URL
 *   size      - "small" | "medium" (default: "medium")
 *   onClick   - optional click handler
 */

import React from 'react';
import Box from '@mui/material/Box';
import Avatar from '@mui/material/Avatar';
import Typography from '@mui/material/Typography';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import { useTheme } from '@mui/material/styles';

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  return Math.abs(hash);
}

export function PersonChip({ name, role, type = 'human', avatarUrl, size = 'medium', onClick }) {
  const theme = useTheme();
  const avatarSize = size === 'small' ? 28 : 36;
  const colors = theme.custom?.categorical || ['#4a90d9'];
  const bgColor = colors[hashString(name) % colors.length];

  return (
    <Box
      onClick={onClick}
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 0.75,
        px: 1,
        py: 0.5,
        borderRadius: 2,
        backgroundColor: 'rgba(255, 255, 255, 0.04)',
        cursor: onClick ? 'pointer' : 'default',
        '&:hover': onClick ? { backgroundColor: 'rgba(255, 255, 255, 0.08)' } : {},
        transition: 'background-color 0.15s',
      }}
    >
      <Avatar
        src={avatarUrl}
        sx={{ width: avatarSize, height: avatarSize, bgcolor: bgColor, fontSize: avatarSize * 0.45 }}
      >
        {type === 'ai' ? <SmartToyIcon sx={{ fontSize: avatarSize * 0.55 }} /> : name?.charAt(0)?.toUpperCase()}
      </Avatar>
      <Box sx={{ minWidth: 0 }}>
        <Typography variant="body2" sx={{ fontWeight: 500, lineHeight: 1.2, whiteSpace: 'nowrap' }}>
          {name}
        </Typography>
        {role && (
          <Typography variant="caption" sx={{ color: 'text.secondary', lineHeight: 1.2, whiteSpace: 'nowrap' }}>
            {role}
          </Typography>
        )}
      </Box>
    </Box>
  );
}

export default PersonChip;
