/**
 * ApprovalWidget — decision prompt with approve/reject buttons.
 *
 * Shows a warning-styled header, question text, optional context,
 * and two action buttons. Tracks approval state locally.
 *
 * Props:
 *   data.question      - string
 *   data.context        - string (optional reasoning/context)
 *   data.approve_label  - string (default: "Approve")
 *   data.reject_label   - string (default: "Reject")
 */

import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import { useTheme } from '@mui/material/styles';

export default function ApprovalWidget({ data, onSubmit }) {
  const theme = useTheme();
  const { question, context, approve_label = 'Approve', reject_label = 'Reject' } = data || {};
  const [decision, setDecision] = useState(null); // 'approved' | 'rejected' | null

  const handleApprove = () => {
    if (decision) return; // double-submit guard
    setDecision('approved');
    onSubmit?.({ decision: 'approved', question });
  };

  const handleReject = () => {
    if (decision) return; // double-submit guard
    setDecision('rejected');
    onSubmit?.({ decision: 'rejected', question });
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
      <Typography
        variant="subtitle2"
        sx={{ color: 'warning.light', fontWeight: 600, mb: 1 }}
      >
        {'\u26A1'} Decision Required
      </Typography>

      {/* Question */}
      <Typography variant="body2" sx={{ color: 'text.primary', mb: 1 }}>
        {question}
      </Typography>

      {/* Context */}
      {context && (
        <Typography
          variant="body2"
          sx={{ color: 'text.secondary', mb: 1.5, pl: 1.5, borderLeft: `2px solid ${theme.custom.surfaces.cardBorder}` }}
        >
          {context}
        </Typography>
      )}

      {/* Actions or result */}
      {decision ? (
        <Typography
          variant="body2"
          sx={{
            color: decision === 'approved' ? 'success.main' : 'error.main',
            fontWeight: 500,
            mt: 0.5,
          }}
        >
          {decision === 'approved' ? `Approved \u2713` : `Rejected \u2717`}
        </Typography>
      ) : (
        <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
          <Button
            variant="contained"
            size="small"
            onClick={handleApprove}
            sx={{
              backgroundColor: 'success.dark',
              color: 'common.white',
              textTransform: 'none',
              '&:hover': { backgroundColor: 'success.main' },
            }}
          >
            {approve_label}
          </Button>
          <Button
            variant="outlined"
            size="small"
            onClick={handleReject}
            sx={{
              borderColor: theme.custom.surfaces.cardBorder,
              color: 'text.secondary',
              textTransform: 'none',
              '&:hover': { borderColor: 'primary.main', backgroundColor: theme.custom.surfaces.hoverBg },
            }}
          >
            {reject_label}
          </Button>
        </Box>
      )}
    </Box>
  );
}
