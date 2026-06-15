/**
 * TaskAssignmentWidget — placeholder for task assignment display.
 */
import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

export default function TaskAssignmentWidget({ data }) {
  if (!data) return null;
  return (
    <Box sx={{ p: 1 }}>
      <Typography variant="subtitle2">Task Assignment</Typography>
      <Typography variant="body2" color="text.secondary">
        {JSON.stringify(data)}
      </Typography>
    </Box>
  );
}
