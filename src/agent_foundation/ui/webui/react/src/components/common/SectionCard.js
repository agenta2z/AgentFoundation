/**
 * SectionCard — dark card container with title, optional icon, and collapse.
 *
 * Used to group related content within larger cards (e.g. "Active AI Agents",
 * "AI-Generated Report", "Sprint Stats").
 *
 * Props:
 *   title           - string
 *   icon            - React node (optional, shown before title)
 *   children        - content
 *   collapsible     - boolean (default: false)
 *   defaultExpanded - boolean (default: true)
 *   action          - React node (optional, shown at right of header)
 *   subtitle        - string (optional, shown below title)
 */

import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Collapse from '@mui/material/Collapse';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

export function SectionCard({
  title,
  icon,
  children,
  collapsible = false,
  defaultExpanded = true,
  action,
  subtitle,
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <Paper
      elevation={0}
      sx={{
        backgroundColor: 'rgba(255, 255, 255, 0.03)',
        border: '1px solid rgba(255, 255, 255, 0.06)',
        borderRadius: 2,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        onClick={collapsible ? () => setExpanded(!expanded) : undefined}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          px: 2,
          py: 1.25,
          cursor: collapsible ? 'pointer' : 'default',
          '&:hover': collapsible ? { backgroundColor: 'rgba(255, 255, 255, 0.02)' } : {},
        }}
      >
        {icon && (
          <Box sx={{ display: 'flex', color: 'text.secondary', fontSize: 18 }}>
            {icon}
          </Box>
        )}
        <Box sx={{ flexGrow: 1, minWidth: 0 }}>
          <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 0.5, fontSize: '0.7rem' }}>
            {title}
          </Typography>
          {subtitle && (
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              {subtitle}
            </Typography>
          )}
        </Box>
        {action && <Box>{action}</Box>}
        {collapsible && (
          <IconButton size="small" sx={{ color: 'text.secondary' }}>
            <ExpandMoreIcon
              sx={{
                transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s',
                fontSize: 18,
              }}
            />
          </IconButton>
        )}
      </Box>

      {/* Content */}
      {collapsible ? (
        <Collapse in={expanded}>
          <Box sx={{ px: 2, pb: 2 }}>{children}</Box>
        </Collapse>
      ) : (
        <Box sx={{ px: 2, pb: 2 }}>{children}</Box>
      )}
    </Paper>
  );
}

export default SectionCard;
