/**
 * WorkloadChartWidget — placeholder for workload chart display.
 */
import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

export default function WorkloadChartWidget({ data }) {
  if (!data) return null;
  return (
    <Box sx={{ p: 1 }}>
      <Typography variant="subtitle2">Workload Chart</Typography>
      <Typography variant="body2" color="text.secondary">
        {JSON.stringify(data)}
      </Typography>
    </Box>
  );
}
