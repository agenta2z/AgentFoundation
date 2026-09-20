/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AggregationSettingsPopover — small popover form for the LLM aggregator's
 * input filters (Refresh Learnings → Aggregation settings…).
 *
 * Ported verbatim from RankEvolve (components/views/AggregationSettingsPopover.js).
 */

import React, { useEffect, useState } from 'react';
import {
  Box, Button, Checkbox, FormControlLabel, Popover,
  TextField, Typography,
} from '@mui/material';

export default function AggregationSettingsPopover({
  open, anchorEl, onClose, settings, onSave,
}) {
  const [draft, setDraft] = useState(settings);
  useEffect(() => {
    if (open) setDraft(settings);
  }, [open, settings]);

  const set = (key, value) => setDraft((d) => ({ ...d, [key]: value }));

  const handleSave = () => {
    onSave(draft);
    onClose();
  };

  return (
    <Popover
      open={open}
      anchorEl={anchorEl}
      onClose={onClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      transformOrigin={{ vertical: 'top', horizontal: 'right' }}
    >
      <Box sx={{ p: 2, width: 340 }}>
        <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
          Aggregation settings
        </Typography>

        <TextField
          label="Min epochs to include"
          type="number"
          size="small"
          fullWidth
          value={draft.minEpochs}
          inputProps={{ min: 0, max: 1000 }}
          onChange={(e) => {
            const n = parseInt(e.target.value, 10);
            set('minEpochs', Number.isFinite(n) && n >= 0 ? n : 0);
          }}
          helperText="Skip rows with fewer than N completed epochs (0 = no filter)."
          sx={{ mb: 1 }}
        />

        <FormControlLabel
          control={
            <Checkbox
              size="small"
              checked={!!draft.includeIncomparable}
              onChange={(e) => set('includeIncomparable', e.target.checked)}
            />
          }
          label="Include incomparable verdicts"
          sx={{ display: 'block' }}
        />

        <FormControlLabel
          control={
            <Checkbox
              size="small"
              checked={!!draft.includeErrored}
              onChange={(e) => set('includeErrored', e.target.checked)}
            />
          }
          label="Include errored / failed runs"
          sx={{ display: 'block' }}
        />

        <FormControlLabel
          control={
            <Checkbox
              size="small"
              checked={!!draft.forceRefresh}
              onChange={(e) => set('forceRefresh', e.target.checked)}
            />
          }
          label="Force refresh (skip md5 no-op)"
          sx={{ display: 'block', mt: 0.5 }}
        />
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: 'block', ml: 4, mb: 1.5 }}
        >
          Not persisted — resets to off after each click.
        </Typography>

        <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
          <Button size="small" onClick={onClose}>Cancel</Button>
          <Button size="small" variant="contained" onClick={handleSave}>
            Save
          </Button>
        </Box>
      </Box>
    </Popover>
  );
}
