/**
 * ProjectSummaryWidget — placeholder for project summary display.
 */
import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

export default function ProjectSummaryWidget({ data }) {
  if (!data) return null;
  return (
    <Box sx={{ p: 1 }}>
      <Typography variant="subtitle2">Project Summary</Typography>
      <Typography variant="body2" color="text.secondary">
        {JSON.stringify(data)}
      </Typography>
    </Box>
  );
}
