/**
 * TaskAssignmentWidget — task reassignment confirmation card.
 *
 * Shows the task being reassigned, from/to person chips with utilization,
 * a reason, and confirm/cancel buttons. Tracks confirmation state locally.
 *
 * Props:
 *   data.task_title - string
 *   data.from       - { name, type, utilization }
 *   data.to         - { name, type, utilization }
 *   data.reason     - string
 */

import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import { useTheme } from '@mui/material/styles';

import { PersonChip } from '../common/PersonChip';
import { ProgressBar } from '../common/ProgressBar';

function PersonRow({ label, person }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
      <Typography variant="caption" sx={{ color: 'text.secondary', flex: '0 0 40px', fontWeight: 500 }}>
        {label}
      </Typography>
      <PersonChip name={person.name} type={person.type || 'human'} size="small" />
      <Box sx={{ flex: '0 0 100px' }}>
        <ProgressBar percent={person.utilization || 0} height={4} showLabel />
      </Box>
    </Box>
  );
}

export default function TaskAssignmentWidget({ data }) {
  const theme = useTheme();
  const { task_title, from, to, reason } = data || {};
  const [confirmed, setConfirmed] = useState(null); // 'confirmed' | 'cancelled' | null

  const handleConfirm = () => {
    console.log('[TaskAssignmentWidget] Reassigned:', task_title, 'from', from?.name, 'to', to?.name);
    setConfirmed('confirmed');
  };

  const handleCancel = () => {
    console.log('[TaskAssignmentWidget] Cancelled reassignment:', task_title);
    setConfirmed('cancelled');
  };

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
        {'\u2696\uFE0F'} Reassign Task
      </Typography>

      {/* Task title */}
      <Typography variant="body2" sx={{ color: 'text.primary', mb: 1.5, fontStyle: 'italic' }}>
        &ldquo;{task_title}&rdquo;
      </Typography>

      {/* From / To */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mb: 1.5 }}>
        {from && <PersonRow label="From:" person={from} />}
        {to && <PersonRow label="To:" person={to} />}
      </Box>

      {/* Reason */}
      {reason && (
        <Typography
          variant="body2"
          sx={{ color: 'text.secondary', mb: 1.5, pl: 1.5, borderLeft: `2px solid ${theme.custom.surfaces.cardBorder}` }}
        >
          {reason}
        </Typography>
      )}

      {/* Actions or result */}
      {confirmed ? (
        <Typography
          variant="body2"
          sx={{
            color: confirmed === 'confirmed' ? 'success.main' : 'text.secondary',
            fontWeight: 500,
          }}
        >
          {confirmed === 'confirmed' ? 'Reassigned \u2713' : 'Cancelled'}
        </Typography>
      ) : (
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="contained"
            size="small"
            onClick={handleConfirm}
            sx={{
              backgroundColor: 'primary.main',
              color: 'common.white',
              textTransform: 'none',
              '&:hover': { backgroundColor: 'primary.light' },
            }}
          >
            Confirm
          </Button>
          <Button
            variant="outlined"
            size="small"
            onClick={handleCancel}
            sx={{
              borderColor: theme.custom.surfaces.cardBorder,
              color: 'text.secondary',
              textTransform: 'none',
              '&:hover': { borderColor: 'primary.main', backgroundColor: theme.custom.surfaces.hoverBg },
            }}
          >
            Cancel
          </Button>
        </Box>
      )}
    </Box>
  );
}
