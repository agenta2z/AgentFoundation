/**
 * ChoiceWidget — multiple-option selector for manager decisions.
 *
 * Shows a prompt, selectable option cards with hover/selected states,
 * and a confirm button. Tracks selection state locally.
 *
 * Props:
 *   data.prompt  - string (header text)
 *   data.options - array of { id, label, description }
 */

import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import { useTheme } from '@mui/material/styles';

export default function ChoiceWidget({ data, onSubmit }) {
  const theme = useTheme();
  const { prompt, options = [] } = data || {};
  const [selectedId, setSelectedId] = useState(null);
  const [confirmed, setConfirmed] = useState(false);

  const handleConfirm = () => {
    if (confirmed) return; // double-submit guard
    const selected = options.find((o) => o.id === selectedId);
    setConfirmed(true);
    onSubmit?.({ selected_id: selectedId, selected_label: selected?.label });
  };

  const selectedLabel = options.find((o) => o.id === selectedId)?.label;

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
        {prompt || 'Choose an option'}
      </Typography>

      {confirmed ? (
        <Typography variant="body2" sx={{ color: 'primary.main', fontWeight: 500 }}>
          Selected: {selectedLabel}
        </Typography>
      ) : (
        <>
          {/* Option cards */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75, mb: 1.5 }}>
            {options.map((option) => {
              const isSelected = selectedId === option.id;
              return (
                <Box
                  key={option.id}
                  onClick={() => setSelectedId(option.id)}
                  sx={{
                    p: 1.5,
                    borderRadius: 1.5,
                    border: '1px solid',
                    borderColor: isSelected ? 'primary.main' : theme.custom.surfaces.cardBorder,
                    backgroundColor: isSelected ? theme.custom.surfaces.overlayMedium : theme.custom.surfaces.overlayLight,
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                    '&:hover': {
                      backgroundColor: isSelected ? theme.custom.surfaces.overlayActive : theme.custom.surfaces.hoverBg,
                      borderColor: isSelected ? 'primary.main' : theme.custom.surfaces.cardBorder,
                    },
                  }}
                >
                  <Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 500 }}>
                    {option.label}
                  </Typography>
                  {option.description && (
                    <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.25 }}>
                      {option.description}
                    </Typography>
                  )}
                </Box>
              );
            })}
          </Box>

          {/* Confirm button */}
          <Button
            variant="contained"
            size="small"
            disabled={selectedId == null}
            onClick={handleConfirm}
            sx={{
              backgroundColor: 'primary.main',
              color: 'common.white',
              textTransform: 'none',
              '&:hover': { backgroundColor: 'primary.light' },
              '&.Mui-disabled': { backgroundColor: theme.custom.surfaces.overlayMedium, color: 'text.secondary' },
            }}
          >
            Confirm Selection
          </Button>
        </>
      )}
    </Box>
  );
}
