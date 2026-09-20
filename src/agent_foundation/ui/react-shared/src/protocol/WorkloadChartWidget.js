/**
 * WorkloadChartWidget — horizontal utilization bars per employee.
 *
 * Dark card showing "Workload Balance" header, per-employee utilization bars
 * with status indicators, and an optional suggestion line.
 *
 * Props:
 *   data.employees  - array of { name, type, utilization_percent, status, current_task }
 *   data.suggestion - optional string shown at the bottom
 */

import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useTheme } from '@mui/material/styles';

import { ProgressBar } from '../common/ProgressBar';
import { PersonChip } from '../common/PersonChip';

function getStatusIndicator(status, utilization) {
  if (status === 'pending') return '\u{23F3}';
  if (utilization > 85) return '\u26A0\uFE0F';
  if (utilization === 0) return '\u{1F4A4}';
  return null;
}

export default function WorkloadChartWidget({ data }) {
  const theme = useTheme();
  const { employees = [], suggestion } = data || {};

  const widgetBoxSx = {
    backgroundColor: theme.custom.surfaces.overlayLight,
    border: `1px solid ${theme.custom.surfaces.cardBorder}`,
    borderRadius: 2,
    p: 2,
    mt: 1.5,
  };

  return (
    <Box sx={widgetBoxSx}>
      {/* Header */}
      <Typography variant="subtitle2" sx={{ color: 'text.primary', fontWeight: 600, mb: 1.5 }}>
        Workload Balance
      </Typography>

      {/* Employee rows */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
        {employees.map((emp, i) => {
          const indicator = getStatusIndicator(emp.status, emp.utilization_percent);
          return (
            <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
              <Box sx={{ flex: '0 0 150px' }}>
                <PersonChip name={emp.name} type={emp.type || 'human'} size="small" />
              </Box>
              <Box sx={{ flexGrow: 1 }}>
                <ProgressBar percent={emp.utilization_percent || 0} height={6} />
              </Box>
              {indicator && (
                <Typography variant="body2" sx={{ flex: '0 0 24px', textAlign: 'center', fontSize: '1rem' }}>
                  {indicator}
                </Typography>
              )}
            </Box>
          );
        })}
      </Box>

      {/* Suggestion */}
      {suggestion && (
        <Typography
          variant="body2"
          sx={{
            color: 'text.secondary',
            fontStyle: 'italic',
            mt: 1.5,
            pt: 1.5,
            borderTop: `1px solid ${theme.custom.surfaces.cardBorder}`,
          }}
        >
          {suggestion}
        </Typography>
      )}
    </Box>
  );
}
