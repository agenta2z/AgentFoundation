/**
 * MultiValueInput — controlled list of string values with add/remove chips.
 *
 * The "add" row is supplied by the caller through the `renderInput` render-prop, so the
 * same component hosts either a plain TextField or a PathAutocompleteInput. Submission of
 * the whole widget is owned by the parent; this component only manages the array.
 *
 * Props:
 *   values      - string[]
 *   onChange    - (string[]) => void
 *   renderInput - ({ value, onChange, onAdd, disabled }) => ReactNode  (the "add" row)
 *   addLabel    - string (button label, default "Add")
 *   disabled    - bool
 */

import React, { useState } from 'react';
import { Box, Button, Chip, Stack } from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

export default function MultiValueInput({
  values = [],
  onChange,
  renderInput,
  addLabel = 'Add',
  disabled = false,
}) {
  const theme = useTheme();
  const [draft, setDraft] = useState('');

  const handleAdd = () => {
    const trimmed = (draft || '').trim();
    if (!trimmed || disabled) return;
    if (values.includes(trimmed)) {
      setDraft('');
      return;
    }
    onChange?.([...values, trimmed]);
    setDraft('');
  };

  const handleRemove = (idx) => {
    if (disabled) return;
    onChange?.(values.filter((_, i) => i !== idx));
  };

  return (
    <Box>
      {values.length > 0 && (
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 1 }}>
          {values.map((v, idx) => (
            <Chip
              key={`${v}-${idx}`}
              label={v}
              size="small"
              onDelete={disabled ? undefined : () => handleRemove(idx)}
              sx={{
                backgroundColor: theme.custom?.surfaces?.cardBg || 'rgba(0,0,0,0.15)',
                color: 'text.primary',
                maxWidth: '100%',
                '& .MuiChip-label': { whiteSpace: 'normal' },
              }}
            />
          ))}
        </Stack>
      )}
      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
        <Box sx={{ flexGrow: 1 }}>
          {renderInput?.({ value: draft, onChange: setDraft, onAdd: handleAdd, disabled })}
        </Box>
        <Button
          variant="outlined"
          size="small"
          onClick={handleAdd}
          disabled={disabled || !draft.trim()}
          startIcon={<AddIcon sx={{ fontSize: 16 }} />}
          sx={{ textTransform: 'none', whiteSpace: 'nowrap', flexShrink: 0, mt: 0.25 }}
        >
          {addLabel}
        </Button>
      </Box>
    </Box>
  );
}
